package main

import "math"

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

func idf(df, totalDocs int) float64 {
	return math.Log(1.0 + (float64(totalDocs-df)+0.5)/(float64(df)+0.5))
}

func bm25TermScore(tf, docLen, docFreq, totalDocs int, avgDocLen float64) float64 {
	if tf == 0 || docFreq == 0 {
		return 0
	}
	tfF := float64(tf)
	norm := tfF + bm25K1*(1.0-bm25B+bm25B*float64(docLen)/avgDocLen)
	return idf(docFreq, totalDocs) * (tfF * (bm25K1 + 1.0) / norm)
}
