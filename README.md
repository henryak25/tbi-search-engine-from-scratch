# Information Retrieval - Search Engine from Scratch (TP2)
This project is an implementation for the Information Retrieval course assignment. It involves building a search engine from scratch using Python standard libraries, index compression optimization, search algorithms, and Information Retrieval evaluation.

## Highlights
- **Blocked Sort-Based Indexing (BSBI)**: Optimizes document collection processing in blocks to ensure system RAM efficiency during indexing.
- **SPIMI Indexing**: Includes an alternative SPIMI implementation and a built-in runtime comparison against BSBI.
- **Inverted Index Compression**: Saves disk space using two algorithm options: *Variable Byte Encoding (VBE)* and *Elias-Gamma Encoding*.
- **Trie-Based Mapping**: `IdMap` supports either Python `dict` (default) or `Trie` backend for string-to-ID mapping experiments.
- **Latent Semantic Indexing (LSI) with FAISS**: Implements an advanced Vector Search engine to capture underlying semantic meanings and overcome exact-match limitations.
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
- `spimi.py` : SPIMI indexing implementation and SPIMI vs BSBI indexing time comparison script.
- `compression.py` : Posting list compression algorithms (Standard, VBEPostings, EliasGammaPostings).
- `index.py` : Helper classes for the functionality and I/O operations of the Inverted Index data structure.
- `util.py` : Utility data structures and helpers (`IdMap`, optional `Trie`, merge utilities).
- `search.py` : Query execution module to test and operate the BM25 and WAND retrieval methods, including speedup comparisons.
- `lsi_faiss.py` : LSI + FAISS semantic retrieval index.
- `bm25.md` : Detailed explanation of BM25 implementation in this project.
- `wand.md` : Detailed explanation of WAND implementation in this project.
- `evaluation.py` : Script for evaluating search effectiveness against the document collection.
- `memory_compare_bsbi_spimi.py` : Peak Python memory comparison between BSBI and SPIMI on the same block.
- `benchmark_trie.py` : Benchmark script comparing Python `dict` vs custom `Trie` for exact and prefix search.
- `results.md` : Collected sample benchmark and evaluation outputs.
- `collection/` : Raw dataset directory containing the `.txt` documents to be indexed.
- `index/` : Output directory for the generated index files produced by `bsbi.py`.
- `tmp/` : Temporary directory used during indexing for intermediate index files.
- `queries.txt` : 30 test queries used for evaluation.
- `qrels.txt`   : Relevance judgments for the test queries.

## Requirements
- `tqdm`
- `numpy`
- `scipy`
- `scikit-learn`
- `faiss-cpu`

Install all dependencies:

```bash
pip install -r requirements.txt
```

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

### 1a. Build Index with SPIMI (Alternative)
If you want to run SPIMI indexing (instead of BSBI), use:

```bash
python spimi.py
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

Output includes comparative metrics for BM25 vs LSI (FAISS):
- Mean RBP
- Mean DCG
- Mean NDCG
- MAP (Mean AP)

### 4. Compare Peak Memory Usage (BSBI vs SPIMI)
To compare peak Python memory usage between BSBI and SPIMI on the same block:

```bash
python memory_compare_bsbi_spimi.py
```

The script reports:
- Elapsed time for each run
- Peak Python memory for each run (`tracemalloc`)
- Peak memory ratio (`BSBI / SPIMI`)

Interpretation:
- Ratio `> 1`: SPIMI used less peak Python memory in that experiment.
- Ratio `~ 1`: memory usage was similar for that block/data distribution.

From the local run: Ratio value is 1.3x

### 5. Benchmark Trie vs Dictionary
To benchmark exact-match and prefix-search performance:

```bash
python benchmark_trie.py
```

Optional arguments:
```bash
python benchmark_trie.py --num-words 500000 --prefix-queries 200 --seed 42
```

What it measures:
- Insert time (dict vs trie)
- Exact lookup time (dict vs trie)
- Prefix search time

Interpretation:
- Python `dict` is faster for exact key lookup.
- Trie is better for prefix queries because it does not scan all keys for each prefix.

## Experimental Results
Sample outputs are documented in [results.md](results.md).

Quick highlights:
- Trie benchmark:
    - Dict is faster for exact insert/search.
    - Trie is much faster for prefix search (`379.66x` vs dict key scan in the sample run).
- BSBI vs SPIMI memory benchmark:
    - Peak Python memory ratio (`BSBI / SPIMI`) = `1.30x`.
    - Indicates lower peak Python memory for SPIMI in that run.
- Evaluation (`evaluation.py`):
    - BM25: Mean RBP=`0.6349`, Mean DCG=`5.7768`, Mean NDCG=`0.7985`, MAP=`0.4825`
    - LSI (FAISS): Mean RBP=`0.7413`, Mean DCG=`6.6411`, Mean NDCG=`0.8601`, MAP=`0.6342`
- Compression (`compression.py` sample):
    - StandardPostings: postings `20 bytes`, TF `20 bytes`
    - VBEPostings: postings `9 bytes`, TF `5 bytes`
    - EliasGammaPostings: postings `12 bytes`, TF `3 bytes`