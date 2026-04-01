package main

import "math"

func RunQueries(e *Engine, queries []string, k int) BatchResult {
	result := BatchResult{QueryCount: len(queries)}
	for qi, query := range queries {
		hits := e.Search(query, k)
		result.HitCount += len(hits)
		for rank, hit := range hits {
			scoreBits := uint64(math.Round(hit.Score * 1e6))
			result.Checksum += mixHash(uint64(qi), uint64(rank), uint64(hit.DocID), scoreBits)
		}
	}
	return result
}

func mixHash(parts ...uint64) uint64 {
	h := uint64(1469598103934665603)
	for _, p := range parts {
		h ^= p + 0x9e3779b97f4a7c15
		h *= 1099511628211
	}
	return h
}
