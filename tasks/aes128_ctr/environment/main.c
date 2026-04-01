#define _POSIX_C_SOURCE 200809L
#include "solve.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* 256 MiB input buffer, 5 timed runs per invocation. */
#define DATA_MB   256
#define DATA_LEN  ((size_t)DATA_MB * 1024 * 1024)
#define N_RUNS    5

/* ─── NIST SP 800-38A Appendix F.5.1 test vector (AES-128-CTR) ─────────── */
static const uint8_t NIST_KEY[16] = {
    0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,
    0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c
};
static const uint8_t NIST_IV[16] = {
    0xf0,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,
    0xf8,0xf9,0xfa,0xfb,0xfc,0xfd,0xfe,0xff
};
/* Plaintext blocks 1–4 (64 bytes) */
static const uint8_t NIST_PT[64] = {
    0x6b,0xc1,0xbe,0xe2,0x2e,0x40,0x9f,0x96,
    0xe9,0x3d,0x7e,0x11,0x73,0x93,0x17,0x2a,
    0xae,0x2d,0x8a,0x57,0x1e,0x03,0xac,0x9c,
    0x9e,0xb7,0x6f,0xac,0x45,0xaf,0x8e,0x51,
    0x30,0xc8,0x1c,0x46,0xa3,0x5c,0xe4,0x11,
    0xe5,0xfb,0xc1,0x19,0x1a,0x0a,0x52,0xef,
    0xf6,0x9f,0x24,0x45,0xdf,0x4f,0x9b,0x17,
    0xad,0x2b,0x41,0x7b,0xe6,0x6c,0x37,0x10
};
/* Expected ciphertext blocks 1–4 */
static const uint8_t NIST_CT[64] = {
    0x87,0x4d,0x61,0x91,0xb6,0x20,0xe3,0x26,
    0x1b,0xef,0x68,0x64,0x99,0x0d,0xb6,0xce,
    0x98,0x06,0xf6,0x6b,0x79,0x70,0xfd,0xff,
    0x86,0x17,0x18,0x7b,0xb9,0xff,0xfd,0xff,
    0x5a,0xe4,0xdf,0x3e,0xdb,0xd5,0xd3,0x5e,
    0x5b,0x4f,0x09,0x02,0x0d,0xb0,0x3e,0xab,
    0x1e,0x03,0x1d,0xda,0x2f,0xbe,0x03,0xd1,
    0x79,0x21,0x70,0xa0,0xf3,0x00,0x9c,0xee
};

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    /* ── Correctness gate: NIST SP 800-38A F.5.1 ─────────────────────── */
    uint8_t ct_check[64];
    aes128_ctr_encrypt(NIST_KEY, NIST_IV, NIST_PT, ct_check, 64);
    if (memcmp(ct_check, NIST_CT, 64) != 0) {
        printf("runs=%d time=0.000000 result=WRONG_ANSWER\n", N_RUNS);
        return 0;
    }

    /* ── Allocate + fill benchmark buffer ────────────────────────────── */
    uint8_t *plaintext  = (uint8_t *)malloc(DATA_LEN);
    uint8_t *ciphertext = (uint8_t *)malloc(DATA_LEN);
    if (!plaintext || !ciphertext) {
        fprintf(stderr, "OOM\n");
        free(plaintext);
        free(ciphertext);
        return 1;
    }

    /* Deterministic non-trivial input (LCG pattern). */
    for (size_t i = 0; i < DATA_LEN; i++)
        plaintext[i] = (uint8_t)(i * 6364136223846793005ULL >> 56);

    /* ── Warm-up run (not timed) ──────────────────────────────────────── */
    aes128_ctr_encrypt(NIST_KEY, NIST_IV, plaintext, ciphertext, DATA_LEN);

    /* ── Timed runs ───────────────────────────────────────────────────── */
    double times[N_RUNS];
    for (int r = 0; r < N_RUNS; r++) {
        double t0 = now_sec();
        aes128_ctr_encrypt(NIST_KEY, NIST_IV, plaintext, ciphertext, DATA_LEN);
        times[r] = now_sec() - t0;
    }

    /* Median of N_RUNS. */
    for (int i = 0; i < N_RUNS - 1; i++)
        for (int j = 0; j < N_RUNS - 1 - i; j++)
            if (times[j] > times[j + 1]) {
                double tmp = times[j]; times[j] = times[j+1]; times[j+1] = tmp;
            }
    double median = times[N_RUNS / 2];

    printf("runs=%d time=%.6f result=ok\n", N_RUNS, median);

    free(plaintext);
    free(ciphertext);
    return 0;
}
