/*
 * main.c — benchmark driver for radix_sort.
 *
 * DO NOT MODIFY THIS FILE.
 *
 * Generates N_ELEMS pseudo-random 32-bit integers using a deterministic
 * xorshift64 PRNG, calls radix_sort(), verifies correctness (sorted order
 * + checksum preservation), and reports median wall-clock time over
 * N_RUNS timed trials.
 *
 * Output (single line to stdout):
 *   runs=<N> time=<seconds> checksum=<uint64> sorted=ok
 *   OR
 *   runs=<N> time=<seconds> checksum=0 sorted=FAIL
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "solve.h"

/* ── Benchmark parameters ────────────────────────────────────────────────── */
#define N_ELEMS  50000000UL   /* 50 million 32-bit integers (~200 MB)        */
#define N_RUNS   5

/* ── Deterministic PRNG (xorshift64) ────────────────────────────────────── */
static uint64_t prng_state = 0x123456789ABCDEF0ULL;

static inline uint64_t xorshift64(void)
{
    uint64_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    prng_state = x;
    return x;
}

/* ── Timing ──────────────────────────────────────────────────────────────── */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void)
{
    /* Allocate two working buffers: orig (master copy) and work (sorted each run) */
    uint32_t *orig = (uint32_t *)malloc(N_ELEMS * sizeof(uint32_t));
    uint32_t *work = (uint32_t *)malloc(N_ELEMS * sizeof(uint32_t));
    if (!orig || !work) {
        fprintf(stderr, "OOM: cannot allocate %lu MB\n",
                (unsigned long)(2 * N_ELEMS * sizeof(uint32_t) / (1024 * 1024)));
        return 1;
    }

    /* Generate reproducible random data */
    prng_state = 0x123456789ABCDEF0ULL;
    uint64_t pre_checksum = 0;
    for (size_t i = 0; i < N_ELEMS; i++) {
        uint32_t v = (uint32_t)(xorshift64() >> 32);
        orig[i]     = v;
        pre_checksum += v;
    }

    /* Warm-up run (un-timed) */
    memcpy(work, orig, N_ELEMS * sizeof(uint32_t));
    radix_sort(work, N_ELEMS);

    /* Timed runs */
    double times[N_RUNS];
    for (int r = 0; r < N_RUNS; r++) {
        memcpy(work, orig, N_ELEMS * sizeof(uint32_t));
        double t0 = now_sec();
        radix_sort(work, N_ELEMS);
        times[r] = now_sec() - t0;
    }

    /* ── Correctness check ───────────────────────────────────────────────── */
    int sorted_ok = 1;
    uint64_t post_checksum = 0;
    for (size_t i = 0; i < N_ELEMS; i++) {
        post_checksum += work[i];
        if (i > 0 && work[i] < work[i - 1]) {
            sorted_ok = 0;
        }
    }
    if (pre_checksum != post_checksum) sorted_ok = 0;

    /* ── Median ──────────────────────────────────────────────────────────── */
    qsort(times, N_RUNS, sizeof(double), cmp_double);
    double median = times[N_RUNS / 2];

    printf("runs=%d time=%.6f checksum=%llu sorted=%s\n",
           N_RUNS, median,
           (unsigned long long)pre_checksum,
           sorted_ok ? "ok" : "FAIL");

    free(orig);
    free(work);
    return 0;
}
