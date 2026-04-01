/*
 * solve_optimized.c — reference solution for hash_join
 *
 * Algorithm: flat open-addressing hash join (linear probing)
 * ──────────────────────────────────────────────────────────
 *
 * Phase 1 — Build
 *   Insert all R_ROWS rows into a flat power-of-2 hash table using
 *   Knuth's multiplicative hash and linear probing.  With a load factor
 *   of ~0.3 the average probe length is ≈1.15 slots; collisions are rare.
 *   Table size: next_pow2(R_ROWS * 3) = 65536 slots × 8 bytes = 512 KB.
 *   At the benchmark scale this fits comfortably in L3 cache.
 *
 * Phase 2 — Probe
 *   For each S row, compute h = (key × MULTIPLIER) & mask and walk forward
 *   until an empty slot is reached, accumulating every match found.
 *   The entire probe phase is a single sequential pass over S (streaming
 *   reads) plus random L3 hits into the 512 KB hash table.
 *
 * Why this beats the baseline by ~500×
 * ──────────────────────────────────────
 *   Baseline: O(R × S) = 20K × 5M = 10¹¹ scalar comparisons.
 *   This: O(R + S) = 5M probe operations, nearly all hit L3 in ≤2 probes.
 *
 * Remaining opportunities (left for the participant to discover)
 * ──────────────────────────────────────────────────────────────
 *   • SIMD: use _mm256_cmpeq_epi32 to compare 8 hash-table entries per
 *     instruction, reducing probe cost ≈8× for long linear-probe sequences.
 *   • Software prefetch: __builtin_prefetch(ht_keys + next_h, 0, 1) ahead
 *     of the current probe to hide L3 latency.
 *   • Radix-partitioned join: partition both tables into NPARTS=256 buckets
 *     so each per-bucket hash table fits in L1 (≈ 2 KB).  Total probe time
 *     drops to L1-latency × S_ROWS ≈ 4 ns × 5M = 20 ms.  However, the
 *     scatter (partitioning) phase must be tuned to avoid cache thrashing
 *     on the 40 MB S buffer.
 *   • Struct-of-arrays layout: store all keys in one array and all payloads
 *     in another; the probe inner loop only touches the keys array, halving
 *     effective cache traffic during the comparison scan.
 */

#include "solve.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ─── Helper: next power of 2 ≥ x ──────────────────────────────────────── */
static inline uint32_t next_pow2(uint32_t x)
{
    if (x == 0) return 1;
    x--;
    x |= x >> 1;  x |= x >> 2;  x |= x >> 4;
    x |= x >> 8;  x |= x >> 16;
    return x + 1;
}

/* Sentinel for empty slots: a key value that can never appear in input data.
 * All input keys are in [0, KEY_RANGE), so INT32_MIN = -2147483648 is safe. */
#define EMPTY_KEY   INT32_MIN

/* Knuth multiplicative hash constant (32-bit golden ratio). */
#define MULTIPLIER  2654435761u

/* ─── hash_join ─────────────────────────────────────────────────────────── */

void hash_join(const Row *r, int r_count,
               const Row *s, int s_count,
               JoinResult *out)
{
    /* ── Allocate hash table ─────────────────────────────────────────────── */
    /* Load factor ≈ 0.31 at cap = next_pow2(3 * r_count).
     * For R_ROWS = 20,000: cap = 65536, table = 512 KB → fits in L3. */
    uint32_t cap  = next_pow2((uint32_t)(r_count * 3));
    uint32_t mask = cap - 1;

    /* Separate key and payload arrays (SoA) so the probe loop reads only
     * keys into cache during the linear scan. */
    int32_t *ht_keys     = (int32_t *)malloc(cap * sizeof(int32_t));
    int32_t *ht_payloads = (int32_t *)malloc(cap * sizeof(int32_t));
    if (!ht_keys || !ht_payloads) {
        free(ht_keys); free(ht_payloads); return;
    }

    /* Initialise: mark every slot empty. */
    for (uint32_t i = 0; i < cap; i++)
        ht_keys[i] = EMPTY_KEY;

    /* ── Phase 1: Build ─────────────────────────────────────────────────── */
    for (int i = 0; i < r_count; i++) {
        uint32_t h = ((uint32_t)r[i].key * MULTIPLIER) & mask;
        while (ht_keys[h] != EMPTY_KEY)
            h = (h + 1) & mask;
        ht_keys[h]     = r[i].key;
        ht_payloads[h] = r[i].payload;
    }

    /* ── Phase 2: Probe ─────────────────────────────────────────────────── */
    int64_t  count    = 0;
    uint64_t checksum = 0;

    for (int j = 0; j < s_count; j++) {
        const int32_t sk = s[j].key;
        const int32_t sp = s[j].payload;

        uint32_t h = ((uint32_t)sk * MULTIPLIER) & mask;

        /* Walk the linear probe sequence until an empty slot.
         * Any slot with a matching key is a join output pair. */
        while (ht_keys[h] != EMPTY_KEY) {
            if (ht_keys[h] == sk) {
                count++;
                checksum += (uint64_t)(uint32_t)ht_payloads[h]
                          + (uint64_t)(uint32_t)sp;
            }
            h = (h + 1) & mask;
        }
    }

    free(ht_keys);
    free(ht_payloads);

    out->match_count = count;
    out->checksum    = checksum;
}
