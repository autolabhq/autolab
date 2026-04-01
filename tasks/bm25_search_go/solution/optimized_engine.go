package main

type posting struct {
	docID int
	tf    int
}

type Engine struct {
	docs       []Document
	docLens    []int
	avgDocLen  float64
	postings   map[string][]posting
	docFreq    map[string]int
	scoreBuf   []float64
	seenEpoch  []uint32
	touched    []int
	epoch      uint32
	queryTerms map[string]struct{}
}

func NewEngine(docs []Document) *Engine {
	engine := &Engine{
		docs:       make([]Document, len(docs)),
		docLens:    make([]int, len(docs)),
		postings:   make(map[string][]posting, 4096),
		docFreq:    make(map[string]int, 4096),
		scoreBuf:   make([]float64, len(docs)),
		seenEpoch:  make([]uint32, len(docs)),
		touched:    make([]int, 0, 1024),
		queryTerms: make(map[string]struct{}, 16),
	}
	copy(engine.docs, docs)

	totalLen := 0
	for i, doc := range docs {
		tokens := Tokenize(doc.Text)
		engine.docLens[i] = len(tokens)
		totalLen += len(tokens)
		tf := make(map[string]int, len(tokens))
		for _, tok := range tokens {
			tf[tok]++
		}
		for term, count := range tf {
			engine.postings[term] = append(engine.postings[term], posting{docID: i, tf: count})
			engine.docFreq[term]++
		}
	}

	engine.avgDocLen = float64(totalLen) / float64(len(docs))
	return engine
}

func (e *Engine) Search(query string, k int) []SearchResult {
	queryTokens := Tokenize(query)
	if len(queryTokens) == 0 {
		return nil
	}

	e.epoch++
	e.touched = e.touched[:0]
	for term := range e.queryTerms {
		delete(e.queryTerms, term)
	}
	for _, tok := range queryTokens {
		e.queryTerms[tok] = struct{}{}
	}

	for term := range e.queryTerms {
		postings := e.postings[term]
		df := e.docFreq[term]
		for _, p := range postings {
			if e.seenEpoch[p.docID] != e.epoch {
				e.seenEpoch[p.docID] = e.epoch
				e.scoreBuf[p.docID] = 0
				e.touched = append(e.touched, p.docID)
			}
			e.scoreBuf[p.docID] += bm25TermScore(p.tf, e.docLens[p.docID], df, len(e.docs), e.avgDocLen)
		}
	}

	results := make([]SearchResult, 0, len(e.touched))
	for _, docID := range e.touched {
		score := e.scoreBuf[docID]
		if score > 0 {
			results = append(results, SearchResult{DocID: e.docs[docID].ID, Score: score})
		}
	}

	return RankTopK(results, k)
}
