package main

type Document struct {
	ID   int
	Text string
}

type SearchResult struct {
	DocID int
	Score float64
}

type BatchResult struct {
	QueryCount int
	HitCount   int
	Checksum   uint64
}
