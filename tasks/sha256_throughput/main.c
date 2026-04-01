#define _POSIX_C_SOURCE 200809L
#include "solve.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* 512 MiB input buffer, 3 timed hashes per invocation. */
#define DATA_MB   512
#define DATA_LEN  ((size_t)DATA_MB * 1024 * 1024)
#define N_RUNS    3

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    uint8_t *data = (uint8_t *)malloc(DATA_LEN);
    if (!data) { fprintf(stderr, "OOM\n"); return 1; }

    /* Fill with a deterministic, non-trivial pattern. */
    for (size_t i = 0; i < DATA_LEN; i++)
        data[i] = (uint8_t)(i * 6364136223846793005ULL >> 56);

    uint8_t digest[32];

    /* Warm-up run — not timed. */
    sha256(data, DATA_LEN, digest);

    /* Checksum: first 8 bytes of the digest, fixed for correct implementations. */
    char checksum[17];
    for (int i = 0; i < 8; i++)
        snprintf(checksum + i*2, 3, "%02x", digest[i]);

    /* Timed runs. */
    double times[N_RUNS];
    for (int r = 0; r < N_RUNS; r++) {
        double t0 = now_sec();
        sha256(data, DATA_LEN, digest);
        times[r] = now_sec() - t0;

        /* Verify output consistency across runs. */
        for (int i = 0; i < 8; i++) {
            char tmp[3];
            snprintf(tmp, 3, "%02x", digest[i]);
            if (tmp[0] != checksum[i*2] || tmp[1] != checksum[i*2+1]) {
                printf("runs=%d time=0.000000 checksum=%s result=MISMATCH\n",
                       N_RUNS, checksum);
                free(data);
                return 0;
            }
        }
    }

    /* Median of N_RUNS. */
    for (int i = 0; i < N_RUNS - 1; i++)
        for (int j = 0; j < N_RUNS - 1 - i; j++)
            if (times[j] > times[j + 1]) {
                double tmp = times[j]; times[j] = times[j+1]; times[j+1] = tmp;
            }
    double median = times[N_RUNS / 2];

    printf("runs=%d time=%.6f checksum=%s result=ok\n",
           N_RUNS, median, checksum);

    free(data);
    return 0;
}
