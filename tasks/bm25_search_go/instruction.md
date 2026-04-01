# BM25 Search Engine Query Optimization

Optimize the Go search engine codebase to make query execution as fast as possible.

## Problem

The program generates a deterministic synthetic corpus and a sequence of search queries. It computes exact BM25 top-10 results for each query and reduces the output to a correctness checksum. The current baseline is slow.

## Files

You may edit any of these files:

- `tokenize.go`
- `score.go`
- `rank.go`
- `engine.go`
- `batch.go`
- `corpus.go`
- `types.go`

Do not modify: `main.go`, `go.mod`, `Makefile`

## Rules

- Standard library only. No external modules.
- The public names used by `main.go` must keep working.
- Output must remain exactly correct. Wrong checksum or hit count = score 0.
- Single-process only. Goroutines are allowed.
