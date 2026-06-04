#define _POSIX_C_SOURCE 200809L
#include "solve.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* 1 million pairs, strings 16–64 lowercase ASCII characters. */
#define N_PAIRS   1000000
#define MIN_LEN   16
#define MAX_LEN   64
#define N_RUNS    3

#define FNV_OFFSET           0xcbf29ce484222325ULL
#define FNV_PRIME            0x100000001b3ULL

/* Stride per string in the flat storage buffer. */
#define STRIDE    (MAX_LEN + 1)

/* Default seed used only for local development if LEV_SEED is unset.
 * The verifier (tests/test.sh) ALWAYS sets LEV_SEED to a fresh 64-bit
 * random value per run so that input data and call order vary every run,
 * defeating precomputed lookup-table and static-counter attacks. */
#define LEV_DEFAULT_SEED 0xdeadbeefcafe1234ULL

static uint64_t lcg_state;
static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(lcg_state >> 33);
}

/* Parse an unsigned 64-bit value from a string (decimal or 0x-prefixed hex). */
static uint64_t parse_u64(const char *s, int *ok)
{
    if (!s || !*s) { *ok = 0; return 0; }
    char *end = NULL;
    int base = 10;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16;
    unsigned long long v = strtoull(s, &end, base);
    *ok = (end != s && (*end == '\0' || *end == '\n'));
    return (uint64_t)v;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void update_checksums(const int *results, uint64_t *checksum, uint64_t *fingerprint)
{
    uint64_t sum = 0;
    uint64_t fp = FNV_OFFSET;

    for (int i = 0; i < N_PAIRS; i++) {
        uint64_t d = (uint64_t)(unsigned int)results[i];
        sum += d;
        fp ^= ((uint64_t)(uint32_t)i << 32) ^ d;
        fp *= FNV_PRIME;
    }

    *checksum = sum;
    *fingerprint = fp;
}

int main(void)
{
    /* ── Seed selection ──────────────────────────────────────────────── */
    uint64_t seed = LEV_DEFAULT_SEED;
    const char *env_seed = getenv("LEV_SEED");
    if (env_seed && *env_seed) {
        int ok = 0;
        uint64_t parsed = parse_u64(env_seed, &ok);
        if (!ok) {
            fprintf(stderr, "LEV_SEED is set but unparseable: '%s'\n", env_seed);
            return 1;
        }
        seed = parsed;
    }

    char *sa      = (char *)malloc((size_t)N_PAIRS * STRIDE);
    char *sb      = (char *)malloc((size_t)N_PAIRS * STRIDE);
    int  *la      = (int  *)malloc((size_t)N_PAIRS * sizeof(int));
    int  *lb      = (int  *)malloc((size_t)N_PAIRS * sizeof(int));
    int  *results = (int  *)malloc((size_t)N_PAIRS * sizeof(int));
    int  *perm    = (int  *)malloc((size_t)N_PAIRS * sizeof(int));

    if (!sa || !sb || !la || !lb || !results || !perm) {
        fprintf(stderr, "OOM\n");
        return 1;
    }

    /* ── Input generation (LCG seeded from LEV_SEED) ─────────────────── */
    lcg_state = seed;
    for (int i = 0; i < N_PAIRS; i++) {
        la[i] = (int)(MIN_LEN + lcg_next() % (MAX_LEN - MIN_LEN + 1));
        lb[i] = (int)(MIN_LEN + lcg_next() % (MAX_LEN - MIN_LEN + 1));
        char *pa = sa + i * STRIDE;
        char *pb = sb + i * STRIDE;
        for (int k = 0; k < la[i]; k++)
            pa[k] = (char)('a' + lcg_next() % 26);
        pa[la[i]] = '\0';
        for (int k = 0; k < lb[i]; k++)
            pb[k] = (char)('a' + lcg_next() % 26);
        pb[lb[i]] = '\0';
    }

    /* ── Per-run permutation (Fisher–Yates, same LCG) ────────────────── */
    for (int i = 0; i < N_PAIRS; i++) perm[i] = i;
    for (int i = N_PAIRS - 1; i > 0; i--) {
        /* Combine two 32-bit lcg outputs to get a wider random value before
         * reducing to [0, i]. */
        uint64_t r = ((uint64_t)lcg_next() << 32) | (uint64_t)lcg_next();
        int j = (int)(r % (uint64_t)(i + 1));
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    /* Warm-up run — not timed. Walk in shuffled order. */
    for (int k = 0; k < N_PAIRS; k++) {
        int i = perm[k];
        results[i] = levenshtein(sa + i * STRIDE, la[i],
                                 sb + i * STRIDE, lb[i]);
    }

    /* Initial checksum (over unshuffled result array — deterministic per seed). */
    uint64_t checksum = 0;
    uint64_t fingerprint = 0;
    update_checksums(results, &checksum, &fingerprint);

    /* ── Timed runs ──────────────────────────────────────────────────── */
    double times[N_RUNS];
    for (int r = 0; r < N_RUNS; r++) {
        double t0 = now_sec();
        for (int k = 0; k < N_PAIRS; k++) {
            int i = perm[k];
            results[i] = levenshtein(sa + i * STRIDE, la[i],
                                     sb + i * STRIDE, lb[i]);
        }
        times[r] = now_sec() - t0;

        uint64_t c2 = 0, f2 = 0;
        update_checksums(results, &c2, &f2);
        if (c2 != checksum || f2 != fingerprint) {
            /* Internal inconsistency between runs — almost certainly a bug
             * in solve.c (e.g. depending on call order or shared state). */
            printf("runs=%d time=0.000000 checksum=%llu fingerprint=0x%016llx "
                   "result=inconsistent\n",
                   N_RUNS,
                   (unsigned long long)c2,
                   (unsigned long long)f2);
            free(sa); free(sb); free(la); free(lb); free(results); free(perm);
            return 0;
        }
    }

    /* Sort for median. */
    for (int i = 0; i < N_RUNS - 1; i++)
        for (int j = 0; j < N_RUNS - 1 - i; j++)
            if (times[j] > times[j+1]) {
                double tmp = times[j]; times[j] = times[j+1]; times[j+1] = tmp;
            }
    double median = times[N_RUNS / 2];

    /* The verifier compares (checksum, fingerprint) against values produced
     * by a trusted Python implementation using the SAME LEV_SEED. */
    printf("runs=%d time=%.6f checksum=%llu fingerprint=0x%016llx result=ok\n",
           N_RUNS, median,
           (unsigned long long)checksum,
           (unsigned long long)fingerprint);

    free(sa); free(sb); free(la); free(lb); free(results); free(perm);
    return 0;
}
