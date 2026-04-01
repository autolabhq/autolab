# SSTable Compaction in Rust — Reference

## Background
LSM-based engines such as LevelDB, RocksDB, and Pebble spend substantial time
in compaction: scanning sorted runs, resolving shadowed versions, applying
tombstones, and writing new SSTables.

## Baseline Approach
The baseline fully decodes blocks into allocation-heavy record vectors, clones
fixed-size keys and values repeatedly through iteration and merge, and rebuilds
the output table with small-step writes.

## Possible Optimization Directions
1. **Inline fixed-size records** — avoid heap-allocating 8-byte keys and fixed-size values.
2. **Tighter iteration state** — reduce per-record work when moving through prefix-compressed blocks.
3. **Lower-overhead merge** — keep merge keys compact and avoid unnecessary cloning.
4. **Batched builder** — pre-size buffers and write encoded output with fewer resizes.

## Reference Solution
The reference solution keeps keys and values in inline fixed-size representations
for the hot path, uses a tighter table iterator and smaller merge payloads, and
sizes the output builder more aggressively.

## Source
- LevelDB implementation notes
- RocksDB compaction documentation
- Pebble design notes
