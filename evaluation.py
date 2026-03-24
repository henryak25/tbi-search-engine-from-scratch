import re
from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings
import math
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

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = EliasGammaPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ndcg_scores = []
        ap_scores = []
        
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])
            
            total_rel_in_qrels = sum(qrels[qid].values())

            ranking = []
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])
                
            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            ap_scores.append(ap(ranking, total_rel_in_qrels))

  print("Hasil evaluasi BM25 terhadap 30 queries")
  print(f"Mean RBP   = {sum(rbp_scores) / len(rbp_scores):.4f}")
  print(f"Mean DCG   = {sum(dcg_scores) / len(dcg_scores):.4f}")
  print(f"Mean NDCG  = {sum(ndcg_scores) / len(ndcg_scores):.4f}")
  print(f"Mean AP (MAP) = {sum(ap_scores) / len(ap_scores):.4f}")

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels)