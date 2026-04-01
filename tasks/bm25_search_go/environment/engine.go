package main

type Engine struct {
	docs      []Document
	docLens   []int
	avgDocLen float64
}

func NewEngine(docs []Document) *Engine {
	engine := &Engine{
		docs:    make([]Document, len(docs)),
		docLens: make([]int, len(docs)),
	}
	copy(engine.docs, docs)
	totalLen := 0
	for i, doc := range docs {
		l := len(Tokenize(doc.Text))
		engine.docLens[i] = l
		totalLen += l
	}
	engine.avgDocLen = float64(totalLen) / float64(len(docs))
	return engine
}

func (e *Engine) Search(query string, k int) []SearchResult {
	queryTokens := Tokenize(query)
	if len(queryTokens) == 0 {
		return nil
	}

	querySet := make(map[string]struct{}, len(queryTokens))
	for _, tok := range queryTokens {
		querySet[tok] = struct{}{}
	}

	docFreq := make(map[string]int, len(querySet))
	for term := range querySet {
		for _, candidate := range e.docs {
			candidateTokens := Tokenize(candidate.Text)
			for _, tok := range candidateTokens {
				if tok == term {
					docFreq[term]++
					break
				}
			}
		}
	}

	results := make([]SearchResult, 0, len(e.docs))
	for i, doc := range e.docs {
		docTokens := Tokenize(doc.Text)
		tf := make(map[string]int, len(docTokens))
		for _, tok := range docTokens {
			tf[tok]++
		}

		score := 0.0
		for term := range querySet {
			score += bm25TermScore(tf[term], e.docLens[i], docFreq[term], len(e.docs), e.avgDocLen)
		}

		if score > 0 {
			results = append(results, SearchResult{DocID: doc.ID, Score: score})
		}
	}

	return RankTopK(results, k)
}
