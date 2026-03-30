import math
import time
import numpy as np
import faiss
import os
import pickle

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from index import InvertedIndexReader
from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings


class LSIFAISSIndex:
    def __init__(self, bsbi_instance, latent_dim=100):
        """
        Inisialisasi LSI Index menggunakan FAISS.

        Parameters
        ----------
        bsbi_instance : BSBIIndex
            Objek BSBIIndex yang sudah di-index.
        latent_dim : int
            Jumlah dimensi LSI (hasil SVD). Defaultnya di sini 100
            Nilai aktual yang dipakai bisa lebih kecil jika koleksi
            lebih kecil dari latent_dim (untuk ini hanya dikurangi 1).
        """
        self.bsbi = bsbi_instance
        self.latent_dim = latent_dim

        # svd_model dan faiss_index diinisialisasi di build() karena
        # actual latent_dim baru bisa ditentukan setelah ukuran koleksi diketahui
        self.svd_model = None
        self.faiss_index = None

    def save(self):
        """Menyimpan model SVD dan FAISS index ke disk agar tidak perlu build ulang."""
        os.makedirs(self.bsbi.output_dir, exist_ok=True)
        
        # Simpan SVD dengan pickle
        svd_path = os.path.join(self.bsbi.output_dir, 'lsi_svd.pkl')
        with open(svd_path, 'wb') as f:
            pickle.dump(self.svd_model, f)
            
        # Simpan FAISS Index (FAISS punya method bawaan yang sangat optimal)
        faiss_path = os.path.join(self.bsbi.output_dir, 'lsi.faiss')
        faiss.write_index(self.faiss_index, faiss_path)
        
        print("Model SVD dan FAISS Index berhasil disimpan ke disk.")

    def load(self):
        """Memuat model SVD dan FAISS index dari disk."""
        svd_path = os.path.join(self.bsbi.output_dir, 'lsi_svd.pkl')
        faiss_path = os.path.join(self.bsbi.output_dir, 'lsi.faiss')
        
        if not os.path.exists(svd_path) or not os.path.exists(faiss_path):
            raise FileNotFoundError("File model SVD atau FAISS tidak ditemukan. Jalankan build() dulu.")
            
        # Load SVD
        with open(svd_path, 'rb') as f:
            self.svd_model = pickle.load(f)
            
        # Load FAISS Index
        self.faiss_index = faiss.read_index(faiss_path)

    def build(self):
        """
        Membangun Term-Document Sparse Matrix, train SVD,
        lalu memasukkan embedding dokumen ke FAISS Index.

        Bisa dipanggil ulang untuk rebuild — index lama akan direset.
        """
        print("Load BSBI Mapping")
        if len(self.bsbi.term_id_map) == 0 or len(self.bsbi.doc_id_map) == 0:
            self.bsbi.load()

        num_docs  = len(self.bsbi.doc_id_map)
        num_terms = len(self.bsbi.term_id_map)

        # latent_dim tidak boleh >= min(num_docs, num_terms)
        # karena TruncatedSVD ada syarat n_components < min(n_samples, n_features)
        actual_dim = min(self.latent_dim, num_docs - 1, num_terms - 1)
        if actual_dim != self.latent_dim:
            print(f"   latent_dim dikurangi dari {self.latent_dim} menjadi {actual_dim} "
                  f"(batas koleksi: {num_docs} docs, {num_terms} terms)")

        self.svd_model  = TruncatedSVD(n_components=actual_dim, random_state=42)

        # IndexFlatIP melakukan Inner Product.
        # Setelah vektor di-L2 normalize, IP menjadi ekuivalen dengan Cosine Similarity.
        self.faiss_index = faiss.IndexFlatIP(actual_dim)

        rows = []
        cols = []
        data = []

        print("Building TF-IDF Sparse Matrix")
        with InvertedIndexReader(self.bsbi.index_name, self.bsbi.postings_encoding,
                                 directory=self.bsbi.output_dir) as merged_index:
            N = len(merged_index.doc_length)

            # iterasinya pakai range(num_terms) via id_to_str,
            # bukan str_to_id.values() yang tidak ada di implementasi Trie.
            # id_to_str adalah plain list yang selalu tersedia di IdMap.
            # dengan begitu bisa support baik Trie (use_trie=True) maupun dict (use_trie=False).
            for term_id in range(num_terms):
                if term_id not in merged_index.postings_dict:
                    continue

                df = merged_index.postings_dict[term_id][1]
                idf = math.log(N / df)

                postings, tf_list = merged_index.get_postings_list(term_id)
                for doc_id, tf in zip(postings, tf_list):
                    weight = (1 + math.log(tf)) * idf
                    rows.append(doc_id)
                    cols.append(term_id)
                    data.append(weight)

        # csr_matrix: sparse matrix (num_docs x num_terms) tanpa alokasi nilai 0
        tfidf_matrix = csr_matrix((data, (rows, cols)), shape=(num_docs, num_terms))
        print(f"   Ukuran Matrix: {num_docs} dokumen x {num_terms} term "
              f"({tfidf_matrix.nnz} non-zero entries)")

        print(f"Menjalankan Truncated SVD ke {actual_dim} dimensi")
        # fit_transform: (num_docs, num_terms) to (num_docs, actual_dim)
        doc_embeddings = self.svd_model.fit_transform(tfidf_matrix).astype(np.float32)

        print("Normalisasi L2 dan memasukkan vektor ke FAISS")
        # Normalisasi L2 agar IndexFlatIP menghasilkan Cosine Similarity
        faiss.normalize_L2(doc_embeddings)
        self.faiss_index.add(doc_embeddings)

        print(f"{self.faiss_index.ntotal} dokumen berhasil di-index "
              f"ke ruang laten {actual_dim} dimensi.\n")
        
        self.save()

    def retrieve(self, query, k=10):
        """
        Mencari dokumen terdekat menggunakan LSI dan FAISS.

        Parameters
        ----------
        query : str
            Query string, dipisahkan spasi.
        k : int
            Jumlah dokumen teratas yang dikembalikan.

        Returns
        -------
        List[Tuple[float, str]]
            List of (cosine_similarity_score, document_path), sorted descending.
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            try:
                self.load()
            except FileNotFoundError:
                raise RuntimeError("Index belum dibangun. Panggil build() terlebih dahulu.")

        if len(self.bsbi.term_id_map) == 0:
            self.bsbi.load()

        num_terms = len(self.bsbi.term_id_map)

        # Parsing query dan hitung TF per term
        term_counts = {}
        for word in query.split():
            if word in self.bsbi.term_id_map.str_to_id:
                tid = self.bsbi.term_id_map[word]
                term_counts[tid] = term_counts.get(tid, 0) + 1

        # Semua kata query di luar vocabulary
        if not term_counts:
            return []

        # Build vektor TF-IDF untuk query
        q_cols = []
        q_data = []

        with InvertedIndexReader(self.bsbi.index_name, self.bsbi.postings_encoding,
                                 directory=self.bsbi.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            for tid, count in term_counts.items():
                if tid not in merged_index.postings_dict:
                    continue
                df  = merged_index.postings_dict[tid][1]
                idf = math.log(N / df)
                weight = (1 + math.log(count)) * idf
                q_cols.append(tid)
                q_data.append(weight)

        if not q_cols:
            return []

        q_rows = [0] * len(q_cols)
        query_matrix = csr_matrix((q_data, (q_rows, q_cols)), shape=(1, num_terms))

        # Proyeksikan query ke latent space yang sama dengan dokumen
        query_embedding = self.svd_model.transform(query_matrix).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search FAISS, distances = cosine scores, indices = internal doc IDs
        distances, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for doc_id, score in zip(indices[0], distances[0]):
            if doc_id != -1:
                results.append((float(score), self.bsbi.doc_id_map[int(doc_id)]))

        return results


# Testing untuk memastikan LSI-FAISS bisa build dan retrieve tanpa error
if __name__ == "__main__":
    encoding_name     = "elias"
    postings_encoding = VBEPostings if encoding_name == "vbe" else EliasGammaPostings

    bsbi = BSBIIndex(data_dir='collection',
                     postings_encoding=postings_encoding,
                     output_dir='index')

    lsi_faiss = LSIFAISSIndex(bsbi, latent_dim=100)

    start_build = time.time()
    lsi_faiss.build()
    print(f"Waktu Build Index FAISS: {time.time() - start_build:.2f} detik\n")

    queries = [
        "psychodrama for disturbed children",
        "alkylated with radioactive iodoacetate",
        "lipid metabolism in toxemia and normal pregnancy",
    ]

    for query in queries:
        print(f"=== Query: '{query}' ===")
        start_search = time.time()
        results = lsi_faiss.retrieve(query, k=10)
        search_time = time.time() - start_search

        print(f"Search time FAISS: {search_time:.5f} detik")
        print(f"{'Rank':<5} | {'Cosine Score':<12} | Dokumen")
        print("-" * 60)
        for i, (score, doc) in enumerate(results):
            print(f"{i+1:<5} | {score:>11.4f} | {doc}")
        print()