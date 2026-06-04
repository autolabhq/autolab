/*
 * Trusted reference implementation used by the verifier to compute the
 * expected (checksum, fingerprint) for a given LEV_SEED.
 *
 * MUST mirror the LCG, input generation and checksum logic in
 * environment/main.c bit-for-bit. The Levenshtein implementation here is
 * a textbook two-row DP — known correct, independent from the agent's
 * solve.c.
 *
 * Usage:
 *   ./compute_expected <seed-as-decimal-or-0xhex>
 *
 * Output:
 *   checksum=<u64-decimal> fingerprint=0x<16-hex>
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define N_PAIRS   1000000
#define MIN_LEN   16
#define MAX_LEN   64

#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

#define STRIDE (MAX_LEN + 1)

static uint64_t lcg_state;
static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(lcg_state >> 33);
}

static int min2(int a, int b) { return a < b ? a : b; }
static int min3(int a, int b, int c) { return min2(a, min2(b, c)); }

/* Trusted scalar two-row DP Levenshtein. */
static int lev(const char *a, int la, const char *b, int lb)
{
    int prev[MAX_LEN + 1], curr[MAX_LEN + 1];
    for (int j = 0; j <= lb; j++) prev[j] = j;
    for (int i = 1; i <= la; i++) {
        curr[0] = i;
        for (int j = 1; j <= lb; j++) {
            int sub = prev[j-1] + (a[i-1] != b[j-1]);
            curr[j] = min3(prev[j] + 1, curr[j-1] + 1, sub);
        }
        memcpy(prev, curr, (size_t)(lb + 1) * sizeof(int));
    }
    return prev[lb];
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <seed>\n", argv[0]);
        return 1;
    }

    char *end = NULL;
    int base = 10;
    if (argv[1][0] == '0' && (argv[1][1] == 'x' || argv[1][1] == 'X')) base = 16;
    unsigned long long seed_ull = strtoull(argv[1], &end, base);
    if (end == argv[1] || (*end != '\0' && *end != '\n')) {
        fprintf(stderr, "Bad seed: %s\n", argv[1]);
        return 1;
    }
    uint64_t seed = (uint64_t)seed_ull;

    char *sa = (char *)malloc((size_t)N_PAIRS * STRIDE);
    char *sb = (char *)malloc((size_t)N_PAIRS * STRIDE);
    int  *la = (int *)malloc((size_t)N_PAIRS * sizeof(int));
    int  *lb = (int *)malloc((size_t)N_PAIRS * sizeof(int));
    if (!sa || !sb || !la || !lb) { fprintf(stderr, "OOM\n"); return 1; }

    /* Mirror environment/main.c input generation exactly. */
    lcg_state = seed;
    for (int i = 0; i < N_PAIRS; i++) {
        la[i] = (int)(MIN_LEN + lcg_next() % (MAX_LEN - MIN_LEN + 1));
        lb[i] = (int)(MIN_LEN + lcg_next() % (MAX_LEN - MIN_LEN + 1));
        char *pa = sa + i * STRIDE;
        char *pb = sb + i * STRIDE;
        for (int k = 0; k < la[i]; k++) pa[k] = (char)('a' + lcg_next() % 26);
        pa[la[i]] = '\0';
        for (int k = 0; k < lb[i]; k++) pb[k] = (char)('a' + lcg_next() % 26);
        pb[lb[i]] = '\0';
    }

    /* The permutation does NOT affect the checksum (it's computed over
     * results[i] in natural order), but we must still consume the same
     * RNG outputs as main.c so any future change that uses the post-
     * permutation state would stay in sync. We don't actually need the
     * permutation values here, so we skip generating them. */

    /* Compute distances and the same checksum/fingerprint as main.c. */
    uint64_t sum = 0;
    uint64_t fp = FNV_OFFSET;
    for (int i = 0; i < N_PAIRS; i++) {
        int d = lev(sa + i * STRIDE, la[i], sb + i * STRIDE, lb[i]);
        uint64_t du = (uint64_t)(unsigned int)d;
        sum += du;
        fp ^= ((uint64_t)(uint32_t)i << 32) ^ du;
        fp *= FNV_PRIME;
    }

    printf("checksum=%llu fingerprint=0x%016llx\n",
           (unsigned long long)sum,
           (unsigned long long)fp);

    free(sa); free(sb); free(la); free(lb);
    return 0;
}
