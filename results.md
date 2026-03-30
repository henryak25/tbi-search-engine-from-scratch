# Experimental Results

This file stores sample benchmark/evaluation outputs captured from local runs.

## Environment
- OS: Windows (PowerShell)
- Date: 2026-03-29

## 1. Trie Benchmark (`benchmark_trie.py`)

### Output Summary
- Exact-match phase:
  - Python dict insert: `0.0984s`
  - Python dict search: `0.0454s`
  - Custom Trie insert: `2.0287s`
  - Custom Trie search: `0.3404s`
- Prefix-search phase:
  - Python dict prefix: `1.6561s` with `5247` matches
  - Custom Trie prefix: `0.0044s` with `5247` matches

### Reported Speedups
- Trie prefix speedup vs dict scan: `379.66x`
- Dict insert speedup vs trie: `20.61x`
- Dict exact-search speedup vs trie: `7.49x`

### Interpretation For the Speedups
- **Exact Lookup & Insert:** Python's built-in `dict` is a heavily C-optimized hash map yielding near $O(1)$ performance. The custom `Trie` incurs pure-Python overhead due to node object creation and character-by-character traversal.
- **Prefix Queries:** A `Trie` efficiently traverses only the relevant prefix branch ($O(L)$ time). Conversely, an unordered `dict` forces a highly inefficient full linear scan ($O(N)$) across all keys.
- **Trie Primary Use Cases:** Ideal for real-time autocomplete/typeahead, wildcard search queries (e.g., `comput*`).

## 2. BSBI vs SPIMI Memory Comparison (`memory_compare_bsbi_spimi.py`)

### Output Summary
- Measured block: `collection/1`
- BSBI peak Python memory: `3.83 MB`
- SPIMI peak Python memory: `2.95 MB`
- Peak memory ratio (BSBI/SPIMI): `1.30x`

### Timing (same run)
- BSBI elapsed: `0.883s`
- SPIMI elapsed: `0.199s`

### Interpretation
- In this run, SPIMI used less peak Python memory.
- This supports the claim that removing the `td_pairs` allocation reduces peak memory in this implementation.

## 3. Evaluation Results (`evaluation.py`)

### Output (BM25 vs LSI-FAISS)

| Metric | BM25 | LSI (FAISS) |
|---|---:|---:|
| Mean RBP | `0.6349` | `0.7413` |
| Mean DCG | `5.7768` | `6.6411` |
| Mean NDCG | `0.7985` | `0.8601` |
| MAP (Mean AP) | `0.4825` | `0.6342` |

### Interpretation
- In this run, LSI (FAISS) outperformed BM25 on all reported effectiveness metrics.

## 4. Compression Results (`compression.py`)

### StandardPostings
- Encoded postings size: `20 bytes`
- Encoded TF size: `20 bytes`

### VBEPostings
- Encoded postings size: `9 bytes`
- Encoded TF size: `5 bytes`

### EliasGammaPostings
- Encoded postings size: `12 bytes`
- Encoded TF size: `3 bytes`

### Interpretation
- `VBEPostings` gives the smallest postings-list size in this sample.
- `EliasGammaPostings` gives the smallest TF-list size in this sample.
- Both compressed methods significantly reduce size vs StandardPostings.

