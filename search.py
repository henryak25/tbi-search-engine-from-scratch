import time
from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings 

encoding_name = "elias"  # change to "vbe" to use VBEPostings
if encoding_name == "vbe":
    selected_postings = VBEPostings
else:
    selected_postings = EliasGammaPostings

BSBI_instance = BSBIIndex(data_dir='collection', 
                          postings_encoding=selected_postings, 
                          output_dir='index')

queries = ["alkylated with radioactive iodoacetate", 
           "psychodrama for disturbed children", 
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    print(f"=== Query: '{query}' ===")
    
    # 1. Test BM25 Biasa (Exhaustive)
    start_time_bm25 = time.time()
    results_bm25 = BSBI_instance.retrieve_bm25(query, k=10)
    time_bm25 = time.time() - start_time_bm25
    
    # 2. Test WAND
    start_time_wand = time.time()
    results_wand = BSBI_instance.retrieve_wand(query, k=10)
    time_wand = time.time() - start_time_wand
    
    # --- Print Results ---
    print(f"Waktu BM25 Biasa : {time_bm25:.5f} detik")
    print(f"Waktu WAND       : {time_wand:.5f} detik")
    
    # Calculate speedup if WAND is faster
    if time_wand > 0:
        speedup = time_bm25 / time_wand
        print(f"Speedup          : {speedup:.2f}x Lebih Cepat!")
    
    print("\nPerbandingan Hasil (Harus Sama Persis):")
    print(f"{'Rank':<5} | {'Skor BM25':<10} vs {'Skor WAND':<10} | {'Dokumen'}")
    print("-" * 60)
    
    # Print comparison results
    for i in range(len(results_bm25)):
        score_b, doc_b = results_bm25[i]
        
        if i < len(results_wand):
            score_w, doc_w = results_wand[i]
        else:
            score_w, doc_w = 0.0, "-"
            
        
        print(f"{i+1:<5} | {score_b:>9.3f} vs {score_w:>9.3f}")
        
    print("\n" + "="*60 + "\n")