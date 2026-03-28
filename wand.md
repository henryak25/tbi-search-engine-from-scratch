## WAND Explanation for the Implementation in bsbi.py

This file explains the WAND retrieval logic implemented in [bsbi.py](bsbi.py#L271), specifically inside the retrieve_wand function.

## Where It Is Implemented

- Main function: [retrieve_wand](bsbi.py#L271)
- Default parameters: k = 10, k1 = 1.625, b = 0.75
- Scoring model used inside WAND: BM25 term scoring (same core scoring formula as BM25 retrieval)

## What is k, k1, and b?
### 1) k (Top-k size)
- Meaning: number of final documents returned.
- Practical effect:
	- Smaller k: usually faster queries and stricter output size.
	- Larger k: more results, but usually more work and slightly slower.


### 2) k1 (BM25 TF saturation)
- Meaning: controls how strongly term frequency tf increases score.
- Practical effect:
	- Lower k1: tf saturates faster (extra repetitions of a term add less gain).
	- Higher k1: tf has stronger influence before saturating.

Relevant code:
~~~python
numerator = tf * (k1 + 1)
denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
~~~

### 3) b (Length normalization strength)

- Meaning: controls document-length normalization in BM25.
- Range in BM25 practice: typically between 0 and 1.
- In this implementation:
	- b = 0 means almost no length normalization.
	- b = 1 means full normalization by doc_len / avgdl.
- Practical effect:
	- Higher b penalizes long documents more strongly.
	- Lower b reduces that penalty.

Relevant code:

~~~python
denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
~~~


## What WAND Does in This Code

WAND is used to speed up top-k retrieval by avoiding full scoring of all candidate documents.

At a high level, this implementation:

1. Builds posting iterators for query terms.
2. Computes a per-term upper bound ub.
3. Uses cumulative upper bounds to find a pivot.
4. Scores only valid pivot candidates exactly.
5. Skips ahead aggressively when exact scoring is not needed.

This reduces unnecessary scoring operations compared to exhaustive BM25.

## Code Excerpts from the Actual Implementation

1) Function signature

~~~python
def retrieve_wand(self, query, k=10, k1=1.625, b=0.75):
~~~

2) Query term filtering (unknown terms are ignored)

~~~python
terms = [self.term_id_map[word] for word in query.split()
		if word in self.term_id_map.str_to_id]
~~~

3) Iterator structure used by WAND

~~~python
class PostingIterator:
	def __init__(self, term, postings, tfs, ub, idf):
		self.term = term
		self.postings = postings
		self.tfs = tfs
		self.ub = ub
		self.idf = idf
		self.pos = 0
		self.n = len(postings)
		self.current_doc = postings[0] if self.n > 0 else float('inf')

	def next_geq(self, target_doc):
		while self.pos < self.n and self.postings[self.pos] < target_doc:
			self.pos += 1
		self.current_doc = self.postings[self.pos] if self.pos < self.n else float('inf')
		return self.current_doc
~~~

4) Upper-bound construction for each term

~~~python
df      = merged_index.postings_dict[term][1]
max_tf  = merged_index.postings_dict[term][4]
idf     = math.log((N - df + 0.5) / (df + 0.5) + 1)
ub      = idf * (max_tf * (k1 + 1)) / \
			(max_tf + k1 * (1 - b + b * min_doc_len / avgdl))
~~~

5) Pivot selection using cumulative upper bounds

~~~python
pivot_idx = -1
ub_sum = 0.0
for i, it in enumerate(iterators):
	ub_sum += it.ub
	if ub_sum > theta:
		pivot_idx = i
		break
~~~

6) Exact scoring only when candidate is valid

~~~python
if iterators[0].current_doc == pivot_doc:
	curr_doc = pivot_doc
	exact_score = 0.0
	doc_len = merged_index.doc_length[curr_doc]

	for it in iterators:
		if it.current_doc == curr_doc:
			tf = it.tfs[it.pos]
			numerator = tf * (k1 + 1)
			denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
			exact_score += it.idf * (numerator / denominator)
			it.next_geq(curr_doc + 1)
~~~

7) Skip path when candidate is not valid

~~~python
else:
	for i in range(pivot_idx + 1):
		if iterators[i].current_doc < pivot_doc:
			iterators[i].next_geq(pivot_doc)
~~~

8) Top-k maintenance with min-heap threshold theta

~~~python
if len(top_k_heap) < k:
	heapq.heappush(top_k_heap, (exact_score, curr_doc))
	if len(top_k_heap) == k:
		theta = top_k_heap[0][0]
elif exact_score > theta:
	heapq.heappushpop(top_k_heap, (exact_score, curr_doc))
	theta = top_k_heap[0][0]
~~~

9) Final sorted output

~~~python
docs = [(score, self.doc_id_map[doc_id]) for score, doc_id in top_k_heap]
return sorted(docs, key=lambda x: x[0], reverse=True)[:k]
~~~

## Step-by-Step Execution Flow

1. Ensure dictionaries are loaded.
If term_id_map or doc_id_map is empty, the function calls self.load().

2. Build query term list.
Only in-vocabulary terms are kept.

3. Open merged index and read global stats.
The function computes N, avgdl, and min_doc_len.

4. Build one PostingIterator per query term.
For each term, it loads postings and tfs, computes idf, and computes an upper bound ub.

5. Start WAND loop.
Iterators are pruned if exhausted and sorted by current_doc.

6. Find pivot using cumulative ub against theta.
If no pivot is found, retrieval stops.

7. If first iterator doc equals pivot doc, do exact scoring.
Then update heap and theta.

8. Otherwise, skip ahead.
All iterators before pivot that lag behind are advanced to pivot_doc.

9. Convert heap doc_id values to document paths and return ranked results.

## Why This Is Faster Than Exhaustive BM25

- Exhaustive BM25 scores many more documents directly.
- WAND uses ub-based pruning to avoid scoring candidates that cannot beat current threshold theta.
- Exact scoring is executed only for strong candidates.

Because of this, WAND usually gives the same top-k ranking goal with lower query latency.

## Implementation Notes

- The ub formula uses max_tf and min_doc_len to estimate the best possible term contribution.
- The heap stores tuples of score and doc_id, with theta as the minimum score in current top-k.
- Output uses document paths mapped from doc_id.
- If no iterators remain, the loop terminates early.
