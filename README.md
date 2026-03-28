# Information Retrieval - Search Engine from Scratch (TP2)
This project is an implementation for the Information Retrieval course assignment. It involves building a search engine from scratch using Python standard libraries, index compression optimization, search algorithms, and Information Retrieval evaluation.

## Highlights
- **Blocked Sort-Based Indexing (BSBI)**: Optimizes document collection processing in blocks to ensure system RAM efficiency during indexing.
- **Inverted Index Compression**: Saves disk space using two algorithm options: *Variable Byte Encoding (VBE)* and *Elias-Gamma Encoding*.
- **Search & Ranking Algorithms**:
    - **BM25**: A standard probabilistic model for top-k document retrieval. See implementation notes in [bm25.md](bm25.md).
    - **WAND (Weak AND) Algorithm**: A dynamic optimization for BM25 that reduces query latency when searching for the top-k results by safely skipping non-competitive documents. See implementation notes in [wand.md](wand.md).
- **IR Evaluation Module**: Search engine performance testing using Ground Truth (Qrels), supporting evaluation metrics:
    - RBP (*Rank Biased Precision*)
    - DCG (*Discounted Cumulative Gain*)
    - NDCG (*Normalized Discounted Cumulative Gain*)
    - MAP (*Mean Average Precision*)

## Project Structure
- `bsbi.py` : Implementation of the Blocked Sort-Based Indexing algorithm scheme.
- `compression.py` : Posting list compression algorithms (Standard, VBEPostings, EliasGammaPostings).
- `index.py` : Helper classes for the functionality and I/O operations of the Inverted Index data structure.
- `search.py` : Query execution module to test and operate the BM25 and WAND retrieval methods, including speedup comparisons.
- `bm25.md` : Detailed explanation of BM25 implementation in this project.
- `wand.md` : Detailed explanation of WAND implementation in this project.
- `evaluation.py` : Script for evaluating search effectiveness against the document collection.
- `collection/` : Raw dataset directory containing the `.txt` documents to be indexed.
- `index/` : Output directory for the generated index files produced by `bsbi.py`.
- `tmp/` : Temporary directory used during indexing for intermediate index files.
- `queries.txt` : 30 test queries used for evaluation.
- `qrels.txt`   : Relevance judgments for the test queries.

## How to Run

### 0. Compare Compression Methods
To compare index encoding size for:
- No compression (`StandardPostings`)
- VBE (`VBEPostings`)
- Elias-Gamma (`EliasGammaPostings`)

Run:
```bash
python compression.py
```

The script will print encoded size and compression ratio summary.

### 1. Build Index
The first step is to build the inverted index from the `collection/` directory.

Compression method is configured in the `__main__` block in `bsbi.py`:
- `EliasGammaPostings` (default)
- `VBEPostings`
- `StandardPostings`

Run:
```bash
python bsbi.py
```

### 2. Run Search (BM25 vs WAND)
After indexing is finished, run the search script to compare:
- BM25 exhaustive retrieval
- WAND optimized retrieval

Run:
```bash
python search.py
```

### 3. Run Evaluation (RBP, DCG, NDCG, MAP)
Evaluation uses:
- `queries.txt` for test queries
- `qrels.txt` for relevance judgments

Run:
```bash
python evaluation.py
```

Output will include mean scores across all queries:
- Mean RBP
- Mean DCG
- Mean NDCG
- Mean AP