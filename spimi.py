import os
import contextlib
import shutil
import time
from tqdm import tqdm
from bsbi import BSBIIndex
from index import InvertedIndexReader, InvertedIndexWriter
from compression import VBEPostings, EliasGammaPostings


class SPIMIIndex(BSBIIndex):
    """
    SPIMIIndex extends BSBIIndex with the SPIMI (Single-Pass In-Memory Indexing) algorithm.

    Key difference from BSBI:
    - BSBI (codebase from Pak Alfan): collects all (termID, docID) pairs first,
            then aggregates with hash maps and sorts term keys and postings lists.
    - SPIMI: streams tokens directly into an in-memory hashtable (no td_pairs list),
             then writes per-term postings from that dictionary.

    Complexity note:
    - Textbook BSBI is often described with an O(T log T) pair-sorting phase.
    - But the implementation of BSBI is closer to O(T) ingestion plus sorting overhead,
        i.e. O(U log U) for term keys and sum_t O(P_t log P_t) for postings per term.
        (T = tokens, U = unique terms, P_t = postings length for term t).
    - Therefore, the primary advantage of SPIMI here is Space Complexity: 
        skipping the O(T) RAM allocation for the td_pairs list drastically 
        reduces peak memory usage during indexing.

    This implementation uses one directory = one block, same as BSBIIndex,
    rather than a memory-limit-based spill mechanism.
    All retrieval methods (TF-IDF, BM25, WAND) are inherited unchanged from BSBIIndex.
    """

    def build_block_index(self, block_dir_relative, index):
        """
        Read all documents in one block directory and build an in-memory inverted
        index using hashtables, then write postings to disk.

        Terms are written in sorted term_id order so intermediate indexes stay
        compatible with the merge logic inherited from BSBIIndex.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to the block directory inside self.data_dir.
        index : InvertedIndexWriter
            Open writer for the intermediate index file for this block.
        """
        dir_path = os.path.normpath(os.path.join(".", self.data_dir, block_dir_relative))

        # In-memory SPIMI dictionaries (hashtable, not a sorted list)
        term_dict = {}  # term_id -> set of doc_ids
        term_tf   = {}  # term_id -> {doc_id: frequency}

        for filename in next(os.walk(dir_path))[2]:
            # Keep path format aligned with BSBI/evaluation expectations.
            docname = os.path.normpath(os.path.join(dir_path, filename)).replace("\\", "/")

            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in f.read().split():
                    term_id = self.term_id_map[token]
                    doc_id  = self.doc_id_map[docname]

                    if term_id not in term_dict:
                        term_dict[term_id] = set()
                        term_tf[term_id]   = {}

                    term_dict[term_id].add(doc_id)

                    if doc_id not in term_tf[term_id]:
                        term_tf[term_id][doc_id] = 0
                    term_tf[term_id][doc_id] += 1

        
        # While SPIMI builds the dictionary in arbitrary memory order (hashtable),
        # we have to sort the term_ids (keys) before writing the block to disk.
        # This is because the subsequent External Merge Sort phase (using heapq.merge) 
        # strictly requires each intermediate index block to be pre-sorted.
        # If left unsorted, the merge phase will fail to align and combine 
        # identical terms across different blocks
        for term_id in sorted(term_dict.keys()):
            sorted_doc_ids = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_ids]
            index.append(term_id, sorted_doc_ids, assoc_tf)

    def index(self):
        """
        Override BSBIIndex.index() with the SPIMI indexing loop.

        For each block (directory), opens an InvertedIndexWriter and passes it
        directly to build_block_index — no intermediate td_pairs list is created.
        After all blocks are processed, delegates save() and merge() to BSBIIndex.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        for block_dir_relative in tqdm(
            sorted(next(os.walk(self.data_dir))[1]),
            desc="SPIMI Indexing"
        ):
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)

            with InvertedIndexWriter(index_id, self.postings_encoding,
                                     directory=self.output_dir) as index_writer:
                self.build_block_index(block_dir_relative, index_writer)

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(index_id, self.postings_encoding,
                                            directory=self.output_dir)
                    )
                    for index_id in self.intermediate_indices
                ]
                self.merge(indices, merged_index)


def _run_and_time(label, index_class, data_dir, encoding, output_dir):
    """Helper: clears output_dir, runs index(), returns elapsed seconds."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    instance = index_class(data_dir=data_dir,
                           postings_encoding=encoding,
                           output_dir=output_dir)
    start = time.time()
    instance.index()
    elapsed = time.time() - start
    print(f"{label} finished in {elapsed:.2f}s")
    return elapsed


if __name__ == "__main__":
    DATA_DIR   = 'collection'
    OUTPUT_DIR = 'index'
    encoding_name = "elias"  # change to "vbe" to use VBEPostings
    if encoding_name == "vbe":
        ENCODING = VBEPostings
    else:
        ENCODING = EliasGammaPostings

    print("=" * 50)
    print("SPIMI vs BSBI Indexing Comparison")
    print("=" * 50)

    spimi_time = _run_and_time("SPIMI", SPIMIIndex, DATA_DIR, ENCODING, OUTPUT_DIR)
    bsbi_time  = _run_and_time("BSBI ", BSBIIndex,  DATA_DIR, ENCODING, OUTPUT_DIR)

    print("-" * 50)
    if spimi_time < bsbi_time:
        print(f"SPIMI is {bsbi_time / spimi_time:.2f}x faster than BSBI")
    else:
        print(f"BSBI is {spimi_time / bsbi_time:.2f}x faster than SPIMI")
    print("(Index left in place from the BSBI run for retrieval use)")