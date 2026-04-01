# BM25 Search Engine Query Optimization — Reference

## Background
BM25 (Okapi BM25) is the de-facto lexical ranking function used in production search engines such as Elasticsearch, Solr, and Lucene. Efficient BM25 query execution over large corpora is a central concern in information retrieval systems, where query latency directly affects user experience and infrastructure cost.

## Baseline Approach
The unoptimized implementation re-tokenizes every document for each query, rebuilds per-document term frequencies from scratch, computes document frequency by scanning the entire corpus for each query term, and fully sorts all matching documents even though only top-10 results are needed. With a large synthetic corpus this produces ~2.1s query batch time — many orders of magnitude slower than a properly indexed engine.

## Possible Optimization Directions
1. **Inverted index / postings lists** — pre-build a map from term → [(docID, tf)] at index time; queries walk only relevant postings instead of scanning all documents (~10-100x speedup)
2. **Precomputed corpus statistics** — store `docFreq` and `avgDocLen` once at index build time; eliminate per-query corpus scans
3. **Pre-tokenized document storage** — tokenize each document once during indexing; avoid repeated `strings.Fields` + `strings.ToLower` on every query
4. **Top-k heap instead of full sort** — use a min-heap of size k to find top-10 in O(n log k) instead of O(n log n) full sort
5. **Reusable score accumulator buffers** — allocate `scoreBuf` and `seenEpoch` arrays once; use an epoch counter to avoid clearing the entire buffer between queries; track only `touched` document IDs
6. **DF-aware term ordering / WAND-style pruning** — process rare terms first to maximize early termination; WAND uses per-term upper-bound scores to skip documents that cannot enter the top-k

## Reference Solution
Builds a full inverted index at `NewEngine` time: tokenizes each document once, stores per-term posting lists of `(docID, tf)` pairs, and precomputes `docFreq` and `avgDocLen`. At query time, iterates only over postings for the query terms. Uses an epoch-based score buffer (`seenEpoch` + `scoreBuf` + `touched` list) to accumulate BM25 contributions across terms with zero allocation per query. Top-k selection uses a min-heap of size k, avoiding a full sort.

## Source
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Broder et al., *Efficient query evaluation using a two-level retrieval process* (WAND) (2003)
- Apache Lucene BM25Similarity — https://lucene.apache.org/core/8_11_0/core/org/apache/lucene/search/similarities/BM25Similarity.html
