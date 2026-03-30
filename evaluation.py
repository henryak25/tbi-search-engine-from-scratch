import re
from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings
import math

import os
from lsi_faiss import LSIFAISSIndex

encoding_name = "elias"  # change to "vbe" to use VBEPostings
if encoding_name == "vbe":
  selected_postings = VBEPostings
else:
  selected_postings = EliasGammaPostings
######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ Calculates search effectiveness metric score with 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         biner vector like [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
    """
    Menghitung Discounted Cumulative Gain (DCG)
    """
    score = 0.0
    for i, rel in enumerate(ranking):
        # i dimulai dari 0, sehingga rank = i + 1. 
        # Sehingga penyebutnya log2(rank + 1) = log2(i + 2)
        score += rel / math.log2(i + 2)
    return score

def ndcg(ranking):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG)
    dengan membandingkan DCG terhadap Ideal DCG
    """
    dcg_val = dcg(ranking)
    
    ideal_ranking = sorted(ranking, reverse=True)
    idcg_val = dcg(ideal_ranking)
    
    # Agar tidak division by zero jika tidak ada dokumen relevan
    if idcg_val == 0.0:
        return 0.0
        
    return dcg_val / idcg_val

def ap(ranking, total_relevant):
    """
    Menghitung Average Precision
    """
    relevant_docs = 0
    cumulative_precision = 0.0
    
    for i, rel in enumerate(ranking):
        if rel == 1:
            relevant_docs += 1
            cumulative_precision += relevant_docs / (i + 1)
            
    if total_relevant == 0:
        return 0.0
        
    return cumulative_precision / total_relevant


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

def extract_doc_id(doc_path):
    """ Ekstrak ID dokumen """
    basename = os.path.basename(doc_path)
    match = re.search(r'(\d+)', basename)
    if match:
        return int(match.group(1))
    return -1

######## >>>>> EVALUASI !
def eval_comparison(qrels, query_file="queries.txt", k=1000):
    """ 
    Mengevaluasi dan membandingkan BM25 vs LSI (FAISS)
    """
    print("Inisialisasi BSBI Index")
    bsbi_instance = BSBIIndex(data_dir='collection', 
                              postings_encoding=selected_postings, 
                              output_dir='index')
    
    print("Inisialisasi LSI-FAISS Index")
    lsi_instance = LSIFAISSIndex(bsbi_instance, latent_dim=100)
    try:
        lsi_instance.load()
        print("LSI-FAISS model loaded from disk")
    except FileNotFoundError:
        print("LSI-FAISS model not found, building a new one...")
        lsi_instance.build()

    bm25_rbp, bm25_dcg, bm25_ndcg, bm25_ap  = [], [], [], []
    lsi_rbp, lsi_dcg, lsi_ndcg, lsi_ap = [], [], [], []

    print(f"\nMulai evaluasi komparasi {len(qrels)} queries")
    with open(query_file) as file:
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])
            
            if qid not in qrels:
                continue
                
            total_rel_in_qrels = sum(qrels[qid].values())
            
            # --- BM25 ---
            bm25_results = bsbi_instance.retrieve_bm25(query, k=k)
            
            ranking_bm25 = []
            for (score, doc) in bm25_results:
                did = extract_doc_id(doc)
                ranking_bm25.append(qrels[qid].get(did, 0))
                    
            bm25_rbp.append(rbp(ranking_bm25))
            bm25_dcg.append(dcg(ranking_bm25))
            bm25_ndcg.append(ndcg(ranking_bm25))
            bm25_ap.append(ap(ranking_bm25, total_rel_in_qrels))

            # --- LSI FAISS ---
            lsi_results = lsi_instance.retrieve(query, k=k)
            
            ranking_lsi = []
            for (score, doc) in lsi_results:
                did = extract_doc_id(doc)
                ranking_lsi.append(qrels[qid].get(did, 0))
                    
            lsi_rbp.append(rbp(ranking_lsi))
            lsi_dcg.append(dcg(ranking_lsi))
            lsi_ndcg.append(ndcg(ranking_lsi))
            lsi_ap.append(ap(ranking_lsi, total_rel_in_qrels))

    n = len(bm25_rbp)
    print("\n" + "=" * 50)
    print(f"Hasil Evaluasi ({n} Queries)")
    print("=" * 50)
    print(f"{'Metrik':<15} | {'BM25':<12} | {'LSI (FAISS)':<12}")
    print("-" * 50)
    print(f"{'Mean RBP':<15} | {sum(bm25_rbp)/n:<12.4f} | {sum(lsi_rbp)/n:<12.4f}")
    print(f"{'Mean DCG':<15} | {sum(bm25_dcg)/n:<12.4f} | {sum(lsi_dcg)/n:<12.4f}")
    print(f"{'Mean NDCG':<15} | {sum(bm25_ndcg)/n:<12.4f} | {sum(lsi_ndcg)/n:<12.4f}")
    print(f"{'MAP (Mean AP)':<15} | {sum(bm25_ap)/n:<12.4f} | {sum(lsi_ap)/n:<12.4f}")
    print("-" * 50)
    print("=" * 50)

if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    eval_comparison(qrels)