#include "solve.h"
#include <string.h>
#include <stdint.h>

/* ─── AES-128 S-box (FIPS 197 Figure 7) ────────────────────────────────── */
static const uint8_t SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

/* Round constants (RCON[r] used in key expansion round r, 1-indexed) */
static const uint8_t RCON[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

/* ─── GF(2⁸) multiply-by-2 ─────────────────────────────────────────────── */
static uint8_t xtime(uint8_t a)
{
    return (uint8_t)((a << 1) ^ ((a & 0x80u) ? 0x1bu : 0x00u));
}

/* ─── AES-128 key schedule ──────────────────────────────────────────────── */
/*
 * State layout: flat 16-byte array, column-major.
 * state[r + 4*c] = AES state row r, column c.
 * This matches the FIPS 197 byte-order convention.
 */
typedef struct { uint8_t rk[11][16]; } aes_ctx;

static void key_expand(const uint8_t key[16], aes_ctx *ctx)
{
    memcpy(ctx->rk[0], key, 16);
    for (int r = 1; r <= 10; r++) {
        const uint8_t *prev = ctx->rk[r - 1];
        uint8_t       *curr = ctx->rk[r];

        /* RotWord + SubWord + Rcon applied to the last word of prev round */
        curr[0] = SBOX[prev[13]] ^ RCON[r] ^ prev[0];
        curr[1] = SBOX[prev[14]]            ^ prev[1];
        curr[2] = SBOX[prev[15]]            ^ prev[2];
        curr[3] = SBOX[prev[12]]            ^ prev[3];

        /* Words 1–3: simple XOR chain */
        for (int j = 4; j < 16; j++)
            curr[j] = curr[j - 4] ^ prev[j];
    }
}

/* ─── AES-128 encrypt one 16-byte block ────────────────────────────────── */
static void aes_encrypt_block(const aes_ctx *ctx,
                               const uint8_t in[16], uint8_t out[16])
{
    uint8_t s[16];
    memcpy(s, in, 16);

    /* AddRoundKey (round 0) */
    for (int i = 0; i < 16; i++) s[i] ^= ctx->rk[0][i];

    /* Rounds 1–9: SubBytes + ShiftRows + MixColumns + AddRoundKey */
    for (int round = 1; round <= 9; round++) {
        /* SubBytes */
        for (int i = 0; i < 16; i++) s[i] = SBOX[s[i]];

        /* ShiftRows (column-major state: row r is at indices r, r+4, r+8, r+12) */
        uint8_t t;
        /* Row 1: shift left by 1 */
        t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
        /* Row 2: shift left by 2 */
        t = s[2]; s[2] = s[10]; s[10] = t;
        t = s[6]; s[6] = s[14]; s[14] = t;
        /* Row 3: shift left by 3 (= right by 1) */
        t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;

        /* MixColumns: each 4-byte column through the MDS matrix */
        for (int c = 0; c < 4; c++) {
            uint8_t a = s[4*c], b = s[4*c+1], cc = s[4*c+2], d = s[4*c+3];
            uint8_t xa = xtime(a), xb = xtime(b), xcc = xtime(cc), xd = xtime(d);
            s[4*c+0] = xa ^ xb ^ b ^ cc ^ d;
            s[4*c+1] = a ^ xb ^ xcc ^ cc ^ d;
            s[4*c+2] = a ^ b ^ xcc ^ xd ^ d;
            s[4*c+3] = xa ^ a ^ b ^ cc ^ xd;
        }

        /* AddRoundKey */
        for (int i = 0; i < 16; i++) s[i] ^= ctx->rk[round][i];
    }

    /* Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns) */
    for (int i = 0; i < 16; i++) s[i] = SBOX[s[i]];
    {
        uint8_t t;
        t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
        t = s[2]; s[2] = s[10]; s[10] = t;
        t = s[6]; s[6] = s[14]; s[14] = t;
        t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
    }
    for (int i = 0; i < 16; i++) s[i] ^= ctx->rk[10][i];

    memcpy(out, s, 16);
}

/* ─── CTR mode counter increment (big-endian 128-bit) ───────────────────── */
static void ctr_inc(uint8_t ctr[16])
{
    for (int i = 15; i >= 0; i--) {
        if (++ctr[i]) break;
    }
}

/* ─── Public API ─────────────────────────────────────────────────────────── */
void aes128_ctr_encrypt(const uint8_t key[16], const uint8_t iv[16],
                         const uint8_t *plaintext, uint8_t *ciphertext,
                         size_t len)
{
    aes_ctx ctx;
    key_expand(key, &ctx);

    uint8_t counter[16];
    memcpy(counter, iv, 16);

    uint8_t keystream[16];
    size_t off = 0;

    /* Full 16-byte blocks */
    for (; off + 16 <= len; off += 16) {
        aes_encrypt_block(&ctx, counter, keystream);
        ctr_inc(counter);
        for (int i = 0; i < 16; i++)
            ciphertext[off + i] = plaintext[off + i] ^ keystream[i];
    }

    /* Partial final block (if any) */
    if (off < len) {
        aes_encrypt_block(&ctx, counter, keystream);
        for (size_t i = 0; off + i < len; i++)
            ciphertext[off + i] = plaintext[off + i] ^ keystream[i];
    }
}
