# Agent Tool Routing — Reference

## Background
LLM agents that expose hundreds or thousands of callable tools must select the right one (or top-k) for each user instruction. Doing so by re-scanning every schema for every query is wasteful: at scale the per-query latency becomes the dominant cost in the agent loop, and the routing problem becomes a classical text-retrieval problem over short, structured documents (tool schemas).

## Baseline Approach
The starter retriever in `environment/retriever.py` is deliberately naive: `build_index()` only pre-tokenizes each tool's `name`, `parameters`, `domain`, and `description` fields into per-field token sets. For every query it then iterates over every tool, recomputes a weighted lexical-overlap score by checking each query token against every field set, and finally sorts the full candidate list to take the top-k. Cost is O(|tools| × |query tokens|) per query plus a full sort, which is fine for a handful of tools but scales linearly with the catalog and dominates runtime once the tool set grows past a few hundred entries.

## Reference Solution
The reference implementation in `solution/solve_optimized.py` keeps the same field-weighted lexical-overlap idea but moves all repeatable work into `build_index()` and answers each query by visiting only the tools that share at least one query token:

- Tokenize each schema field once with the same lowercase regex + light singular stripping + stopword filter.
- Per `(token, tool_id)`, accumulate a weighted score where `name` (×9), `parameters` (×5), `domain` (×3), and `description` (×2) contribute according to `FIELD_WEIGHTS`.
- Compute a smoothed inverse document frequency `idf = 1 + ln((1 + N) / (1 + df))` per token (Sparck Jones / Manning et al.) and fold it into the posting weight, scaled by 256 and stored as an integer for fast accumulation.
- Build an inverted index `token -> [(tool_id, weight), ...]` so queries touch only relevant postings.
- At query time, walk only the postings for query tokens, accumulating into a dense `scores[]` array using a per-query stamp in `seen[]` to avoid re-zeroing the array, and collect the touched tool IDs.
- Pick the top-k from the touched set with `heapq.nlargest`, breaking ties by smaller `tool_id`.

This is an IDF-weighted lexical inverted index, not BM25: there is no TF saturation (no `k1`) and no document-length normalization (no `b`).

## Possible Optimization Directions
1. **Full BM25 / BM25F scoring** — add `k1` TF saturation and `b` length normalization, or per-field BM25F variants, for better ranking quality on longer descriptions.
2. **Token compression / hashing** — replace string-keyed posting maps with hashed integer token IDs and arrays of `(tool_id, weight)` pairs to shrink memory and improve cache locality.
3. **Char n-gram or prefix index** — add a secondary index over 3-4 character n-grams or token prefixes to recover near-misses (typos, plural/singular gaps the stripper does not catch).
4. **Learned sparse retrieval** — train a SPLADE-style model to expand each tool schema into a weighted sparse vector, then reuse the same inverted-index machinery.
5. **Dense embedding retrieval** — embed each schema once with a sentence encoder and query with cosine similarity over a flat or HNSW index; usually combined with the lexical scorer in a hybrid.
6. **Approximate top-k via WAND / MaxScore** — skip postings whose remaining upper-bound score cannot enter the top-k, which matters once the catalog reaches tens of thousands of tools.

## Sources
- Manning, Raghavan, and Schütze, *Introduction to Information Retrieval* (Cambridge University Press, 2008) — inverted indexes, term weighting, and IDF.
- Sparck Jones, "A Statistical Interpretation of Term Specificity and Its Application in Retrieval" (*Journal of Documentation*, 1972) — the original IDF formulation.
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (*Foundations and Trends in Information Retrieval*, 2009) — BM25 family for the optimization-direction reference.
