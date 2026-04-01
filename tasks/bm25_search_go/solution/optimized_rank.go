package main

import (
	"container/heap"
	"sort"
)

type resultHeap []SearchResult

func (h resultHeap) Len() int { return len(h) }
func (h resultHeap) Less(i, j int) bool {
	if h[i].Score == h[j].Score {
		return h[i].DocID > h[j].DocID
	}
	return h[i].Score < h[j].Score
}
func (h resultHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *resultHeap) Push(x any)   { *h = append(*h, x.(SearchResult)) }
func (h *resultHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func RankTopK(results []SearchResult, k int) []SearchResult {
	if len(results) <= k {
		sort.Slice(results, func(i, j int) bool {
			if results[i].Score == results[j].Score {
				return results[i].DocID < results[j].DocID
			}
			return results[i].Score > results[j].Score
		})
		return results
	}

	h := make(resultHeap, 0, k)
	for _, r := range results {
		if len(h) < k {
			heap.Push(&h, r)
			continue
		}
		top := h[0]
		if r.Score > top.Score || (r.Score == top.Score && r.DocID < top.DocID) {
			h[0] = r
			heap.Fix(&h, 0)
		}
	}

	out := make([]SearchResult, len(h))
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(&h).(SearchResult)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].DocID < out[j].DocID
		}
		return out[i].Score > out[j].Score
	})
	return out
}
