package main

import (
	"fmt"
	"strings"
)

type prng struct {
	state uint32
}

func (p *prng) next() uint32 {
	p.state = p.state*1664525 + 1013904223
	return p.state
}

func (p *prng) intn(n int) int {
	return int(p.next() % uint32(n))
}

func makeVocabulary() [][]string {
	topics := make([][]string, 12)
	for t := range topics {
		words := make([]string, 0, 96)
		for i := 0; i < 72; i++ {
			words = append(words, fmt.Sprintf("topic%02d_term%02d", t, i))
		}
		words = append(words,
			"latency", "throughput", "cache", "search", "index", "query",
			"ranking", "memory", "posting", "token", "document", "result",
			"service", "storage", "engine", "update", "batch", "stream",
			"merge", "parser", "filter", "vector", "score", "signal",
		)
		topics[t] = words
	}
	return topics
}

func GenerateCorpus(n int, seed uint32) []Document {
	rng := prng{state: seed}
	topics := makeVocabulary()
	docs := make([]Document, n)

	for i := 0; i < n; i++ {
		mainTopic := rng.intn(len(topics))
		auxTopic := (mainTopic + 1 + rng.intn(3)) % len(topics)
		length := 70 + rng.intn(40)
		parts := make([]string, 0, length+8)
		parts = append(parts, fmt.Sprintf("docgroup%d", i/64))
		parts = append(parts, fmt.Sprintf("region%d", i%23))
		for j := 0; j < length; j++ {
			v := rng.intn(100)
			switch {
			case v < 72:
				parts = append(parts, topics[mainTopic][rng.intn(len(topics[mainTopic]))])
			case v < 90:
				parts = append(parts, topics[auxTopic][rng.intn(len(topics[auxTopic]))])
			case v < 97:
				parts = append(parts, fmt.Sprintf("shared%02d", rng.intn(40)))
			default:
				parts = append(parts, fmt.Sprintf("rare%03d", rng.intn(200)))
			}
		}
		docs[i] = Document{ID: i, Text: strings.Join(parts, " ")}
	}

	return docs
}

func GenerateQueries(n int, seed uint32) []string {
	rng := prng{state: seed}
	topics := makeVocabulary()
	queries := make([]string, n)

	for i := 0; i < n; i++ {
		mainTopic := (i + rng.intn(len(topics))) % len(topics)
		auxTopic := (mainTopic + 1 + rng.intn(2)) % len(topics)
		parts := make([]string, 0, 5)
		parts = append(parts, topics[mainTopic][rng.intn(48)])
		parts = append(parts, topics[mainTopic][rng.intn(48)])
		parts = append(parts, topics[auxTopic][rng.intn(24)])
		if i%3 == 0 {
			parts = append(parts, fmt.Sprintf("shared%02d", rng.intn(20)))
		}
		if i%5 == 0 {
			parts = append(parts, fmt.Sprintf("region%d", rng.intn(23)))
		}
		queries[i] = strings.Join(parts, " ")
	}

	return queries
}
