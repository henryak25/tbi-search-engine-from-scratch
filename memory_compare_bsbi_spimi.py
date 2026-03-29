import gc
import os
import shutil
import time
import tracemalloc

from bsbi import BSBIIndex
from spimi import SPIMIIndex
from index import InvertedIndexWriter
from compression import EliasGammaPostings, VBEPostings


def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def measure_peak_python_memory(label, fn):
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label}: elapsed={elapsed:.3f}s, peak_python_memory={peak / (1024 * 1024):.2f} MB")
    return elapsed, peak


def run_bsbi_single_block(data_dir, output_dir, postings_encoding, block_dir_relative):
    idx = BSBIIndex(data_dir=data_dir, output_dir=output_dir, postings_encoding=postings_encoding)
    with InvertedIndexWriter("bsbi_block", postings_encoding, directory=output_dir) as writer:
        td_pairs = idx.parse_block(block_dir_relative)
        idx.invert_write(td_pairs, writer)


def run_spimi_single_block(data_dir, output_dir, postings_encoding, block_dir_relative):
    idx = SPIMIIndex(data_dir=data_dir, output_dir=output_dir, postings_encoding=postings_encoding)
    with InvertedIndexWriter("spimi_block", postings_encoding, directory=output_dir) as writer:
        idx.build_block_index(block_dir_relative, writer)


def main():
    data_dir = "collection"
    output_dir = "tmp_mem_compare"

    encoding_name = "elias"  # change to "vbe" if needed
    if encoding_name == "vbe":
        postings_encoding = VBEPostings
    else:
        postings_encoding = EliasGammaPostings

    blocks = sorted(next(os.walk(data_dir))[1])
    if not blocks:
        raise RuntimeError("No blocks found in collection directory")

    block = blocks[0]
    print(f"Measuring memory on block: {block}")
    print("Note: This compares Python-level peak memory (tracemalloc), not total OS RSS.")

    ensure_clean_dir(output_dir)
    _, bsbi_peak = measure_peak_python_memory(
        "BSBI (parse_block + invert_write)",
        lambda: run_bsbi_single_block(data_dir, output_dir, postings_encoding, block),
    )

    ensure_clean_dir(output_dir)
    _, spimi_peak = measure_peak_python_memory(
        "SPIMI (build_block_index)",
        lambda: run_spimi_single_block(data_dir, output_dir, postings_encoding, block),
    )

    print("-" * 60)
    if spimi_peak > 0:
        ratio = bsbi_peak / spimi_peak
        print(f"Peak memory ratio (BSBI/SPIMI): {ratio:.2f}x")
    else:
        print("SPIMI peak memory reported as 0 bytes (unexpected for non-empty block)")

    print("Interpretation:")
    print("- If ratio > 1, SPIMI uses less peak Python memory on this block.")
    print("- This supports the claim that skipping td_pairs reduces peak allocation.")


if __name__ == "__main__":
    main()
