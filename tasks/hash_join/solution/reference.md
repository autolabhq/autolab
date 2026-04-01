# Hash Join — Reference

## Background
The inner equi-join is the most performance-critical operation in relational database engines. Every SQL query with a `JOIN ... ON` clause invokes it repeatedly; its throughput directly determines query latency in systems like PostgreSQL, DuckDB, and MonetDB. In-memory hash join has been an active research area since the 1980s, with modern CPU cache hierarchies driving the design of each successive generation.

## Baseline Approach
The unoptimized implementation performs a naive O(|R| × |S|) nested-loop join: for each of the 5,000,000 S rows it scans all 20,000 R rows, yielding ~100 billion key comparisons with no algorithmic improvement. On reference hardware this takes ~20 seconds.

## Possible Optimization Directions
1. **Hash join (build + probe)** — build a hash table over R (20K rows), then probe it once per S row; reduces complexity from O(R × S) to O(R + S), ~500× fewer operations (reward ≈ 0.59 with any correct implementation)
2. **Open-addressing with Knuth multiplicative hash** — flat power-of-2 array with linear probing and `h(k) = (k × 2654435761u) & mask`; avoids `malloc`-per-node cache thrashing of chained tables; 512 KB table fits in L3 (~0.10s)
3. **Struct-of-Arrays (SoA) layout** — store keys and payloads in separate arrays so the probe inner loop reads only the keys array (256 KB), halving effective cache traffic and fitting closer to L2
4. **Radix partitioning** — partition both R and S into 256 buckets by `key & 0xFF`; each per-bucket R hash table is ~2 KB and fits in L1 (~4-cycle probe latency vs. ~40-cycle L3), targeting ~20 ms total
5. **AVX2 SIMD probe** — compare 8 consecutive hash-table slots per `_mm256_cmpeq_epi32` instruction, reducing key-comparison cost ~8× for linear-probe sequences
6. **Software prefetching** — issue `__builtin_prefetch(ht_keys + next_h, 0, 1)` one iteration ahead to overlap L3 latency (~40 cycles) with useful work on the current probe

## Reference Solution
Flat open-addressing hash join with Struct-of-Arrays layout. Build phase inserts all 20K R rows into a 65,536-slot table (load factor ~0.31) using Knuth multiplicative hash and linear probing. Keys and payloads are stored in separate arrays so the probe loop's linear scan touches only the keys array. Probe phase makes one sequential pass over S, performing ~1.15 probes per key on average with nearly all hits landing in L3.

## Source
- Shapiro, *Join Processing in Database Systems with Large Main Memories* (1986)
- Boncz et al., *Breaking the Memory Wall in MonetDB* (2013)
- Kim et al., *Sort vs Hash Revisited: Fast Join Implementation on Modern Multi-Core CPUs* (2009)
- CMU 15-721 Advanced Database Systems, Join Algorithms lecture (https://15721.courses.cs.cmu.edu/spring2023/slides/09-hashjoin.pdf)
