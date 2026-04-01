package main

import (
	"flag"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

func tokenizeRef(text string) []string {
	return strings.Fields(strings.ToLower(text))
}

func rankTopKRef(results []SearchResult, k int) []SearchResult {
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

func referenceSearch(docs []Document, docLens []int, avgDocLen float64, query string, k int) []SearchResult {
	queryTokens := tokenizeRef(query)
	if len(queryTokens) == 0 {
		return nil
	}
	querySet := make(map[string]struct{}, len(queryTokens))
	for _, tok := range queryTokens {
		querySet[tok] = struct{}{}
	}

	docFreq := make(map[string]int, len(querySet))
	for term := range querySet {
		for _, candidate := range docs {
			candidateTokens := tokenizeRef(candidate.Text)
			for _, tok := range candidateTokens {
				if tok == term {
					docFreq[term]++
					break
				}
			}
		}
	}

	results := make([]SearchResult, 0, len(docs))
	for i, doc := range docs {
		docTokens := tokenizeRef(doc.Text)
		tf := make(map[string]int, len(docTokens))
		for _, tok := range docTokens {
			tf[tok]++
		}

		score := 0.0
		for term := range querySet {
			score += bm25TermScore(tf[term], docLens[i], docFreq[term], len(docs), avgDocLen)
		}

		if score > 0 {
			results = append(results, SearchResult{DocID: doc.ID, Score: score})
		}
	}
	return rankTopKRef(results, k)
}

func referenceRunQueries(docs []Document, queries []string, k int) BatchResult {
	docLens := make([]int, len(docs))
	totalLen := 0
	for i, doc := range docs {
		docLens[i] = len(tokenizeRef(doc.Text))
		totalLen += docLens[i]
	}
	avgDocLen := float64(totalLen) / float64(len(docs))

	result := BatchResult{QueryCount: len(queries)}
	for qi, query := range queries {
		hits := referenceSearch(docs, docLens, avgDocLen, query, k)
		result.HitCount += len(hits)
		for rank, hit := range hits {
			scoreBits := uint64(math.Round(hit.Score * 1e6))
			result.Checksum += mixHash(uint64(qi), uint64(rank), uint64(hit.DocID), scoreBits)
		}
	}
	return result
}

func median(xs []float64) float64 {
	cp := make([]float64, len(xs))
	copy(cp, xs)
	sort.Float64s(cp)
	return cp[len(cp)/2]
}

func main() {
	mode := flag.String("mode", "benchmark", "verify or benchmark")
	runs := flag.Int("runs", 5, "number of benchmark runs")
	flag.Parse()

	const topK = 10

	verifyDocs := GenerateCorpus(300, 11)
	verifyQueries := GenerateQueries(8, 19)
	verifyEngine := NewEngine(verifyDocs)
	gotVerify := RunQueries(verifyEngine, verifyQueries, topK)
	refVerify := referenceRunQueries(verifyDocs, verifyQueries, topK)
	if gotVerify != refVerify {
		fmt.Printf("result=wrong time=0.000000 checksum=%d hits=%d queries=%d\n",
			gotVerify.Checksum, gotVerify.HitCount, gotVerify.QueryCount)
		return
	}

	if *mode == "verify" {
		fmt.Printf("result=ok time=0.000000 checksum=%d hits=%d queries=%d\n",
			gotVerify.Checksum, gotVerify.HitCount, gotVerify.QueryCount)
		return
	}

	docs := GenerateCorpus(4000, 123)
	querySets := [][]string{
		GenerateQueries(40, 41),
		GenerateQueries(40, 43),
		GenerateQueries(40, 47),
		GenerateQueries(40, 53),
		GenerateQueries(40, 59),
	}

	times := make([]float64, *runs)
	var last BatchResult
	for i := 0; i < *runs; i++ {
		queries := querySets[i%len(querySets)]
		t0 := time.Now()
		engine := NewEngine(docs)
		last = RunQueries(engine, queries, topK)
		times[i] = time.Since(t0).Seconds()
	}
	med := median(times)

	fmt.Printf("result=ok time=%.6f hits=%d queries=%d\n",
		med, last.HitCount, last.QueryCount)
}
