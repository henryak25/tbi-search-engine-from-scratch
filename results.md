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

### Interpretation
- For exact lookup and insert, Python dict is substantially faster.
- For prefix queries, Trie is significantly faster because dictionary prefix search scans all keys.

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

### Output
- Mean RBP: `0.6349`
- Mean DCG: `5.7768`
- Mean NDCG: `0.7985`
- Mean AP (MAP): `0.4825`

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

