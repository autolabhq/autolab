# Concurrent KV Store with WAL

Optimize the Go implementation of a WAL-backed in-memory key-value store to be as fast as possible.

## Problem

The store runs a deterministic multi-phase workload with **4 concurrent goroutines** (a load phase, a mixed read/write/query phase, and a read-heavy query phase). The public API in `main.go` is fixed:

- `Put(key, tag string, value []byte)` — insert or overwrite, append WAL record
- `Get(key string) ([]byte, bool)` — return latest live value
- `Delete(key string) bool` — remove key, append WAL record
- `CountPrefix(tag, prefix string) int` — count live keys under `tag` starting with `prefix`
- `Stats() StoreStats` — return exact live-key count, live bytes, total index keys, WAL record count, WAL bytes, WAL checksum

## Files

| Path | Status |
|------|--------|
| `/app/store.go` | **Edit this** |
| `/app/index.go` | **Edit this** |
| `/app/wal.go` | **Edit this** |
| `/app/main.go` | Read-only |
| `/app/types.go` | Read-only |
| `/app/go.mod` | Read-only |

## Rules

- Do not modify `go.mod`. Build configuration is fixed.
- Edit `/app/store.go`, `/app/index.go`, and `/app/wal.go` only.
- Standard library only. No external modules, no cgo.
- Goroutines, channels, `sync`, and `sync/atomic` are allowed.
- Correctness is strict: the verifier diffs the full output.
