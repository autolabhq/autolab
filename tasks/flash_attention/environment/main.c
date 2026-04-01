/*
 * main.c — benchmark driver for flash_attention.
 *
 * DO NOT MODIFY THIS FILE.
 *
 * Normal mode (no arguments):
 *   Initialises Q, K, V of shape [N × D] from a deterministic LCG,
 *   calls attention() once (untimed warm-up), then once more (timed).
 *   Prints one line:
 *       n=<N> d=<D> time=<seconds> checksum=<double>
 *
 * Verify mode (--verify):
 *   Uses smaller N=VFY_N, D=VFY_D and a different LCG seed.
 *   Prints one line:
 *       verify n=<N> d=<D> sum=<double>
 *   Used by tests/verify_correctness.py to check numerical correctness.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "solve.h"

/* ── Benchmark parameters ──────────────────────────────────────────────── */
#define BEN_N  4096
#define BEN_D  64

/* ── Verify parameters (small — correctness check only) ───────────────── */
#define VFY_N  256
#define VFY_D  32

/* ── Reproducible LCG (matches verify_correctness.py exactly) ─────────── */
static unsigned long long lcg_state;

static double lcg_next(void)
{
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((lcg_state >> 11) & ((1ULL << 53) - 1))
           * (1.0 / (double)(1ULL << 53));
}

/* Returns a float in [-0.1, 0.1] — typical scale for transformer Q, K, V. */
static float lcg_float(void)
{
    return (float)(lcg_next() - 0.5) * 0.2f;
}

/* ── 32-byte-aligned allocation ────────────────────────────────────────── */
static float *alloc32(size_t n_elems)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 32, n_elems * sizeof(float)) != 0) return NULL;
    return (float *)ptr;
}

/* ── Monotonic clock ───────────────────────────────────────────────────── */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Main ──────────────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    int verify_mode = (argc >= 2 && strcmp(argv[1], "--verify") == 0);

    int n    = verify_mode ? VFY_N : BEN_N;
    int d    = verify_mode ? VFY_D : BEN_D;
    unsigned long long seed = verify_mode
                              ? 0xBEEFDEAD42ULL
                              : 0xDEADBEEF42ULL;

    float *Q      = alloc32((size_t)n * d);
    float *K      = alloc32((size_t)n * d);
    float *V      = alloc32((size_t)n * d);
    float *output = alloc32((size_t)n * d);

    if (!Q || !K || !V || !output) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }

    /* Reproducible initialisation — order must match verify_correctness.py */
    lcg_state = seed;
    for (int i = 0; i < n * d; i++) Q[i] = lcg_float();
    for (int i = 0; i < n * d; i++) K[i] = lcg_float();
    for (int i = 0; i < n * d; i++) V[i] = lcg_float();

    if (verify_mode) {
        /* Run once and emit sum of all output elements. */
        attention(Q, K, V, output, n, d);
        double sum = 0.0;
        for (int i = 0; i < n * d; i++) sum += (double)output[i];
        printf("verify n=%d d=%d sum=%.10e\n", n, d, sum);

    } else {
        /* One untimed warm-up then one timed run. */
        attention(Q, K, V, output, n, d);   /* warm-up */

        double t0 = now_sec();
        attention(Q, K, V, output, n, d);   /* timed  */
        double elapsed = now_sec() - t0;

        double checksum = 0.0;
        for (int i = 0; i < n * d; i++) checksum += (double)output[i];

        printf("n=%d d=%d time=%.6f checksum=%.10e\n",
               n, d, elapsed, checksum);
    }

    free(Q); free(K); free(V); free(output);
    return 0;
}
