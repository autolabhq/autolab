package main

import "sort"

func RankTopK(results []SearchResult, k int) []SearchResult {
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].DocID < results[j].DocID
		}
		return results[i].Score > results[j].Score
	})
	if len(results) > k {
		results = results[:k]
	}
	out := make([]SearchResult, len(results))
	copy(out, results)
	return out
}
