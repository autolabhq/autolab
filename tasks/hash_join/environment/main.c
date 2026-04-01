/*
 * main.c — benchmark driver for the hash_join task.  DO NOT MODIFY.
 *
 * Usage:
 *   ./join                # full benchmark run  (R_ROWS × S_ROWS)
 *   ./join --verify       # quick correctness run (VERIFY_R_ROWS × VERIFY_S_ROWS)
 *
 * Full benchmark output (one line to stdout):
 *   time=<seconds>  matches=<int64>  checksum=<uint64>
 *
 * Correctness run output (one line to stdout):
 *   verify matches=<int64> checksum=<uint64>
 *
 * Input tables are generated deterministically from a fixed PRNG seed so no
 * external files are needed.  The same seed is reproduced in Python inside
 * tests/verify_correctness.py.
 *
 * The full benchmark warms up with one un-timed call, then measures one
 * timed call.  The warm-up result is discarded; only the timed result is
 * printed and checked.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "solve.h"

/* ── Timing ────────────────────────────────────────────────────────────────── */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── xorshift64 PRNG (matches the Python version in verify_correctness.py) ── */

static uint64_t rng_state;

static void rng_seed(uint64_t seed) { rng_state = seed; }

static uint32_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

/* ── Table generation ──────────────────────────────────────────────────────── */

static void gen_table(Row *tbl, int n, int key_range, uint64_t seed)
{
    rng_seed(seed);
    for (int i = 0; i < n; i++) {
        tbl[i].key     = (int32_t)(rng_next() % (uint32_t)key_range);
        tbl[i].payload = (int32_t)rng_next();
    }
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    int verify_mode = (argc >= 2 && strcmp(argv[1], "--verify") == 0);

    int r_count = verify_mode ? VERIFY_R_ROWS : R_ROWS;
    int s_count = verify_mode ? VERIFY_S_ROWS : S_ROWS;

    Row *r = (Row *)malloc((size_t)r_count * sizeof(Row));
    Row *s = (Row *)malloc((size_t)s_count * sizeof(Row));
    if (!r || !s) { fputs("OOM\n", stderr); return 1; }

    /* Fixed seeds — must match verify_correctness.py */
    gen_table(r, r_count, KEY_RANGE, 0xDEADBEEF12345678ULL);
    gen_table(s, s_count, KEY_RANGE, 0xCAFEBABE87654321ULL);

    JoinResult result = {0, 0};

    if (verify_mode) {
        /* Single run, no timing */
        hash_join(r, r_count, s, s_count, &result);
        printf("verify matches=%lld checksum=%llu\n",
               (long long)result.match_count,
               (unsigned long long)result.checksum);
    } else {
        /* Warm-up (un-timed) */
        hash_join(r, r_count, s, s_count, &result);

        /* Timed run */
        result.match_count = 0;
        result.checksum    = 0;
        double t0 = now_sec();
        hash_join(r, r_count, s, s_count, &result);
        double elapsed = now_sec() - t0;

        printf("time=%.6f matches=%lld checksum=%llu\n",
               elapsed,
               (long long)result.match_count,
               (unsigned long long)result.checksum);
    }

    free(r);
    free(s);
    return 0;
}
