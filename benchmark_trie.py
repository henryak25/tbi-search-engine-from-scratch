import argparse
import random
import string
import time

from util import Trie


def generate_random_words(num_words, min_len=3, max_len=10):
    """Generate random lowercase words; duplicates may appear."""
    words = []
    for _ in range(num_words):
        length = random.randint(min_len, max_len)
        words.append("".join(random.choices(string.ascii_lowercase, k=length)))
    return words


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def insert_dict(std_dict, words):
    for i, w in enumerate(words):
        std_dict[w] = i


def search_dict(std_dict, words):
    for w in words:
        _ = std_dict[w]


def insert_trie(custom_trie, words):
    for i, w in enumerate(words):
        custom_trie[w] = i


def search_trie(custom_trie, words):
    for w in words:
        _ = custom_trie[w]


def benchmark(num_words=500000, prefix_queries=200, seed=42):
    random.seed(seed)

    print(f"Generating {num_words:,} random words...")
    test_words = generate_random_words(num_words)
    unique_words = list(set(test_words))
    print(f"Unique words after deduplication: {len(unique_words):,}")

    if not unique_words:
        raise RuntimeError("No words generated for benchmark")

    test_prefixes = [random.choice(unique_words)[:3] for _ in range(prefix_queries)]

    print("\n" + "=" * 60)
    print("Phase 1: Exact Match (insert + lookup)")
    print("=" * 60)

    std_dict = {}
    dict_insert_time, _ = timed(lambda: insert_dict(std_dict, test_words))
    dict_search_time, _ = timed(lambda: search_dict(std_dict, test_words))

    print("[1] Python dict")
    print(f"    - Insert time : {dict_insert_time:.4f}s")
    print(f"    - Search time : {dict_search_time:.4f}s")
    print("-" * 60)

    custom_trie = Trie()
    trie_insert_time, _ = timed(lambda: insert_trie(custom_trie, test_words))
    trie_search_time, _ = timed(lambda: search_trie(custom_trie, test_words))

    print("[2] Custom Trie")
    print(f"    - Insert time : {trie_insert_time:.4f}s")
    print(f"    - Search time : {trie_search_time:.4f}s")

    print("\n" + "=" * 60)
    print("Phase 2: Prefix Search")
    print("=" * 60)

    # Prefix search on dict scans all keys (O(N) per query).
    dict_prefix_time, dict_prefix_count = timed(
        lambda: sum(len([w for w in std_dict.keys() if w.startswith(p)]) for p in test_prefixes)
    )
    print("[1] Python dict")
    print(f"    - Prefix time : {dict_prefix_time:.4f}s")
    print(f"    - Matches     : {dict_prefix_count}")
    print("-" * 60)

    # Trie prefix search navigates prefix path then traverses subtree.
    trie_prefix_time, trie_prefix_count = timed(
        lambda: sum(len(custom_trie.search_prefix(p)) for p in test_prefixes)
    )
    print("[2] Custom Trie")
    print(f"    - Prefix time : {trie_prefix_time:.4f}s")
    print(f"    - Matches     : {trie_prefix_count}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if dict_prefix_time > 0:
        print(f"Trie prefix speedup vs dict scan: {dict_prefix_time / trie_prefix_time:.2f}x")
    if trie_insert_time > 0:
        print(f"Dict insert speedup vs trie: {trie_insert_time / dict_insert_time:.2f}x")
    if trie_search_time > 0:
        print(f"Dict exact-search speedup vs trie: {trie_search_time / dict_search_time:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark dict vs Trie (exact and prefix search).")
    parser.add_argument("--num-words", type=int, default=500000, help="Number of random words to generate")
    parser.add_argument("--prefix-queries", type=int, default=200, help="Number of prefix queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    benchmark(num_words=args.num_words, prefix_queries=args.prefix_queries, seed=args.seed)