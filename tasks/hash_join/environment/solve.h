/*
 * solve.h — interface for the hash_join task.  DO NOT MODIFY.
 *
 * Your entire submission lives in solve.c.  You may NOT modify this file,
 * main.c, or the Makefile.
 */

#ifndef SOLVE_H
#define SOLVE_H

#include <stdint.h>

/* ─── Benchmark parameters ────────────────────────────────────────────────── */

#define R_ROWS      20000    /* build-side table row count */
#define S_ROWS    5000000    /* probe-side table row count */
#define KEY_RANGE   50000    /* keys are drawn uniformly from [0, KEY_RANGE)  */

/* For the fast correctness-check run (--verify flag) */
#define VERIFY_R_ROWS     200
#define VERIFY_S_ROWS    2000

/* ─── Data types ──────────────────────────────────────────────────────────── */

/*
 * Row — one record in either table.
 *   key     : join attribute; values in [0, KEY_RANGE).
 *   payload : data value carried along for the output checksum.
 */
typedef struct {
    int32_t key;
    int32_t payload;
} Row;

/*
 * JoinResult — output of one hash_join() call.
 *   match_count : total number of (r, s) pairs where r.key == s.key.
 *   checksum    : commutative aggregate over all matching pairs:
 *                   SUM of ((uint64_t)(uint32_t)r.payload
 *                          + (uint64_t)(uint32_t)s.payload)
 *                 accumulated into a uint64_t.  Each per-pair contribution
 *                 is the 64-bit zero-extended sum of the two unsigned 32-bit
 *                 payloads (no 32-bit wrap).  Order of summation does not
 *                 matter; results must be identical regardless of traversal
 *                 order.
 */
typedef struct {
    int64_t  match_count;
    uint64_t checksum;
} JoinResult;

/* ─── Public interface ────────────────────────────────────────────────────── */

/*
 * hash_join() — compute the inner equi-join of R and S on the key attribute.
 *
 *   r, r_count : build-side table and its row count (== R_ROWS in benchmark)
 *   s, s_count : probe-side table and its row count (== S_ROWS in benchmark)
 *   out        : caller-allocated result struct; must be zeroed by the caller
 *
 * Semantics (reference):
 *   for every i in [0, r_count):
 *     for every j in [0, s_count):
 *       if r[i].key == s[j].key:
 *         out->match_count++
 *         out->checksum += (uint64_t)(uint32_t)r[i].payload
 *                       + (uint64_t)(uint32_t)s[j].payload
 *
 * The function must produce exactly the same match_count and checksum as the
 * reference above, regardless of the order in which matching pairs are visited.
 */
void hash_join(const Row *r, int r_count,
               const Row *s, int s_count,
               JoinResult *out);

#endif /* SOLVE_H */
