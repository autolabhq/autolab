# Concurrent In-Memory KV Store with WAL — Reference

## Background
Small embedded key-value engines (e.g., LevelDB, Pebble, RocksDB) maintain a primary in-memory map, a secondary index for tag/prefix queries, and a write-ahead log for durability. Under concurrent workloads, coarse locking and per-write WAL flushes become the dominant bottleneck, mirroring the hot path in real storage engines used in databases, message queues, and time-series systems.

## Baseline Approach
A single global `sync.RWMutex` serializes all `Put`, `Get`, `Delete`, and `CountPrefix` operations across all keys. The secondary index stores `tag -> sorted []string` and performs a linear scan on every insert, delete, and prefix query. The WAL appends and flushes a formatted record on every mutation. `Get` copies the value slice before returning it. With 4 concurrent goroutines this bottleneck keeps all writers serialized at ~9.5s.

## Possible Optimization Directions
1. **Store sharding** — Partition the primary map into N shards by FNV hash of the key, each with its own `RWMutex`. Eliminates cross-key contention between unrelated workers (~2–3x speedup).
2. **Zero-copy Get** — Return the stored slice directly under `RLock` instead of copying it. Removes an allocation on every read.
3. **Hash-set secondary index** — Replace the `tag -> sorted []string` with `tag -> map[string]struct{}` to give O(1) insert and delete instead of O(n) linear search; prefix counting becomes a single map scan (~2x speedup on write-heavy phases).
4. **Index sharding** — Shard the tag index by FNV hash of the tag name to reduce contention when many goroutines share the same tags.
5. **WAL group commit** — Buffer WAL records and flush only when a pending-count threshold (e.g., 256 records) or size threshold (e.g., 1 MB) is reached, replacing per-write flushes with batched ones (~3–5x WAL throughput improvement).
6. **Atomic sequence counter** — Use `atomic.Uint64` for the WAL sequence number to avoid any lock on sequence generation.

## Reference Solution
The solution applies all major directions across all three files. `store.go` partitions the keyspace into 64 shards via FNV-32a with per-shard `RWMutex`; `Get` returns the slice without copying. `index.go` shards the tag index into 32 buckets and represents each tag's key set as `map[string]struct{}` for O(1) add/remove. `wal.go` buffers records in a `bytes.Buffer` and flushes only when 256 pending records accumulate or the buffer exceeds 1 MB, using `strconv.AppendUint` to avoid `fmt.Sprintf` allocations. The WAL sequence counter is an `atomic.Uint64`, keeping sequence generation outside any lock.

## Source
- Pebble (CockroachDB): memtable skiplist and commit pipeline design — https://github.com/cockroachdb/pebble
- RocksDB Write-Ahead Log docs and `FlushWAL` discussion — https://github.com/facebook/rocksdb/wiki/Write-Ahead-Log-%28WAL%29
- RocksDB blog: *How we improved WAL throughput* (2017) — https://rocksdb.org/blog/2017/08/25/flushwal.html
- Go `sync/atomic` package, read-mostly `atomic.Value` pattern — https://pkg.go.dev/sync/atomic
- VictoriaMetrics: *sync.Map tradeoffs for read-heavy workloads* — https://victoriametrics.com/blog/go-sync-map/
