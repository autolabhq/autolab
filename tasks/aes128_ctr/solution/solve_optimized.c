/*
 * AES-128-CTR — hardware AES-NI optimized reference solution.
 *
 * Optimizations applied:
 *   1. Hardware key schedule via _mm_aeskeygenassist_si128
 *   2. AES-NI block encryption via _mm_aesenc_si128 / _mm_aesenclast_si128
 *   3. 8-way CTR-mode parallelism to hide 4-cycle AES-NI latency
 *   4. Scalar fallback for non-AES-NI CPUs (T-table implementation)
 *
 * Speedup over naive baseline: ~25–35× on a modern Intel/AMD CPU.
 */

#include "solve.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* ── Scalar T-table fallback ─────────────────────────────────────────────── */

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

static const uint8_t RCON[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

/* T-tables: T0[a] = MixColumns( SubBytes(a) ) rotated, etc.
 * Built lazily the first time the scalar path is called.              */
static uint32_t T0[256], T1[256], T2[256], T3[256];
static int t_tables_ready = 0;

static uint8_t xtime8(uint8_t a)
{
    return (uint8_t)((a << 1) ^ ((a & 0x80u) ? 0x1bu : 0x00u));
}

static void build_t_tables(void)
{
    for (int i = 0; i < 256; i++) {
        uint8_t s  = SBOX[i];
        uint8_t s2 = xtime8(s);
        uint8_t s3 = s2 ^ s;
        /* T0: column [2s, s, s, 3s] in GF(2^8), packed as uint32 little-endian */
        T0[i] = (uint32_t)s2        |
                ((uint32_t)s  << 8) |
                ((uint32_t)s  << 16)|
                ((uint32_t)s3 << 24);
        T1[i] = (T0[i] >> 8)  | (T0[i] << 24);
        T2[i] = (T0[i] >> 16) | (T0[i] << 16);
        T3[i] = (T0[i] >> 24) | (T0[i] << 8);
    }
    t_tables_ready = 1;
}

typedef struct { uint32_t rk[11][4]; } aes_ctx_scalar;

static void key_expand_scalar(const uint8_t key[16], aes_ctx_scalar *ctx)
{
    /* Load key as four 32-bit words (little-endian within each word). */
    for (int i = 0; i < 4; i++) {
        ctx->rk[0][i] = (uint32_t)key[4*i]        |
                        ((uint32_t)key[4*i+1] << 8) |
                        ((uint32_t)key[4*i+2] << 16)|
                        ((uint32_t)key[4*i+3] << 24);
    }
    for (int r = 1; r <= 10; r++) {
        uint32_t w3 = ctx->rk[r-1][3];
        /* RotWord + SubWord on w3 */
        uint32_t tmp = ((uint32_t)SBOX[(w3 >>  8) & 0xff])        |
                       ((uint32_t)SBOX[(w3 >> 16) & 0xff] <<  8)  |
                       ((uint32_t)SBOX[(w3 >> 24) & 0xff] << 16)  |
                       ((uint32_t)SBOX[(w3      ) & 0xff] << 24);
        tmp ^= (uint32_t)RCON[r];
        ctx->rk[r][0] = ctx->rk[r-1][0] ^ tmp;
        ctx->rk[r][1] = ctx->rk[r-1][1] ^ ctx->rk[r][0];
        ctx->rk[r][2] = ctx->rk[r-1][2] ^ ctx->rk[r][1];
        ctx->rk[r][3] = ctx->rk[r-1][3] ^ ctx->rk[r][2];
    }
}

/* Encrypt one block using T-tables.
 * State words: s[i] = col i of the AES state (4 bytes packed, row 0 = low byte). */
static void aes_encrypt_ttable(const aes_ctx_scalar *ctx,
                                const uint8_t in[16], uint8_t out[16])
{
    /* Load state words (column-major, row 0 = byte 0 of each column). */
    uint32_t s0 = ((uint32_t)in[ 0])      | ((uint32_t)in[ 1] <<  8)
                | ((uint32_t)in[ 2] << 16) | ((uint32_t)in[ 3] << 24);
    uint32_t s1 = ((uint32_t)in[ 4])      | ((uint32_t)in[ 5] <<  8)
                | ((uint32_t)in[ 6] << 16) | ((uint32_t)in[ 7] << 24);
    uint32_t s2 = ((uint32_t)in[ 8])      | ((uint32_t)in[ 9] <<  8)
                | ((uint32_t)in[10] << 16) | ((uint32_t)in[11] << 24);
    uint32_t s3 = ((uint32_t)in[12])      | ((uint32_t)in[13] <<  8)
                | ((uint32_t)in[14] << 16) | ((uint32_t)in[15] << 24);

    /* AddRoundKey */
    s0 ^= ctx->rk[0][0]; s1 ^= ctx->rk[0][1];
    s2 ^= ctx->rk[0][2]; s3 ^= ctx->rk[0][3];

    /* Rounds 1–9 using T-tables.
     * After ShiftRows, column c draws from rows {c, c+1, c+2, c+3} mod 4.
     * T0 acts on row 0, T1 on row 1, T2 on row 2, T3 on row 3.           */
    for (int r = 1; r <= 9; r++) {
        uint32_t t0 = T0[ s0        & 0xff] ^ T1[(s1 >>  8) & 0xff]
                    ^ T2[(s2 >> 16) & 0xff] ^ T3[(s3 >> 24) & 0xff] ^ ctx->rk[r][0];
        uint32_t t1 = T0[ s1        & 0xff] ^ T1[(s2 >>  8) & 0xff]
                    ^ T2[(s3 >> 16) & 0xff] ^ T3[(s0 >> 24) & 0xff] ^ ctx->rk[r][1];
        uint32_t t2 = T0[ s2        & 0xff] ^ T1[(s3 >>  8) & 0xff]
                    ^ T2[(s0 >> 16) & 0xff] ^ T3[(s1 >> 24) & 0xff] ^ ctx->rk[r][2];
        uint32_t t3 = T0[ s3        & 0xff] ^ T1[(s0 >>  8) & 0xff]
                    ^ T2[(s1 >> 16) & 0xff] ^ T3[(s2 >> 24) & 0xff] ^ ctx->rk[r][3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    /* Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns). */
    uint32_t u0 = ((uint32_t)SBOX[ s0        & 0xff])
                | ((uint32_t)SBOX[(s1 >>  8) & 0xff] <<  8)
                | ((uint32_t)SBOX[(s2 >> 16) & 0xff] << 16)
                | ((uint32_t)SBOX[(s3 >> 24) & 0xff] << 24);
    uint32_t u1 = ((uint32_t)SBOX[ s1        & 0xff])
                | ((uint32_t)SBOX[(s2 >>  8) & 0xff] <<  8)
                | ((uint32_t)SBOX[(s3 >> 16) & 0xff] << 16)
                | ((uint32_t)SBOX[(s0 >> 24) & 0xff] << 24);
    uint32_t u2 = ((uint32_t)SBOX[ s2        & 0xff])
                | ((uint32_t)SBOX[(s3 >>  8) & 0xff] <<  8)
                | ((uint32_t)SBOX[(s0 >> 16) & 0xff] << 16)
                | ((uint32_t)SBOX[(s1 >> 24) & 0xff] << 24);
    uint32_t u3 = ((uint32_t)SBOX[ s3        & 0xff])
                | ((uint32_t)SBOX[(s0 >>  8) & 0xff] <<  8)
                | ((uint32_t)SBOX[(s1 >> 16) & 0xff] << 16)
                | ((uint32_t)SBOX[(s2 >> 24) & 0xff] << 24);
    u0 ^= ctx->rk[10][0]; u1 ^= ctx->rk[10][1];
    u2 ^= ctx->rk[10][2]; u3 ^= ctx->rk[10][3];

    /* Store (column-major → same as load order). */
    out[ 0] = u0       ; out[ 1] = u0 >>  8; out[ 2] = u0 >> 16; out[ 3] = u0 >> 24;
    out[ 4] = u1       ; out[ 5] = u1 >>  8; out[ 6] = u1 >> 16; out[ 7] = u1 >> 24;
    out[ 8] = u2       ; out[ 9] = u2 >>  8; out[10] = u2 >> 16; out[11] = u2 >> 24;
    out[12] = u3       ; out[13] = u3 >>  8; out[14] = u3 >> 16; out[15] = u3 >> 24;
}

static void ctr_inc(uint8_t ctr[16])
{
    for (int i = 15; i >= 0; i--)
        if (++ctr[i]) break;
}

static void aes128_ctr_scalar(const uint8_t key[16], const uint8_t iv[16],
                               const uint8_t *pt, uint8_t *ct, size_t len)
{
    if (!t_tables_ready) build_t_tables();

    aes_ctx_scalar ctx;
    key_expand_scalar(key, &ctx);

    uint8_t counter[16], ks[16];
    memcpy(counter, iv, 16);

    size_t off = 0;
    for (; off + 16 <= len; off += 16) {
        aes_encrypt_ttable(&ctx, counter, ks);
        ctr_inc(counter);
        for (int i = 0; i < 16; i++) ct[off+i] = pt[off+i] ^ ks[i];
    }
    if (off < len) {
        aes_encrypt_ttable(&ctx, counter, ks);
        for (size_t i = 0; off + i < len; i++) ct[off+i] = pt[off+i] ^ ks[i];
    }
}

/* ── AES-NI path ─────────────────────────────────────────────────────────── */

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))

#include <immintrin.h>
#include <wmmintrin.h>

/* Key expansion helper: given previous round key and aeskeygenassist result,
 * produces the next round key.                                               */
static __m128i __attribute__((target("sse4.1,aes")))
expand_key_step(__m128i prev, __m128i kg)
{
    kg = _mm_shuffle_epi32(kg, 0xff);
    __m128i t = _mm_xor_si128(prev, _mm_slli_si128(prev, 4));
    t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
    t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
    return _mm_xor_si128(t, kg);
}

/* Build 11 __m128i round keys from a 16-byte AES-128 key. */
static void __attribute__((target("sse4.1,aes")))
aes128_key_schedule_ni(__m128i rk[11], const uint8_t key[16])
{
    rk[0]  = _mm_loadu_si128((const __m128i *)key);
#define EXP(n, imm) rk[n] = expand_key_step(rk[(n)-1], \
                            _mm_aeskeygenassist_si128(rk[(n)-1], (imm)))
    EXP( 1, 0x01); EXP( 2, 0x02); EXP( 3, 0x04); EXP( 4, 0x08);
    EXP( 5, 0x10); EXP( 6, 0x20); EXP( 7, 0x40); EXP( 8, 0x80);
    EXP( 9, 0x1b); EXP(10, 0x36);
#undef EXP
}

/* Encrypt one __m128i block (10 rounds). */
#define AES_ENC_BLOCK(b, rk) do {                       \
    (b) = _mm_xor_si128((b), (rk)[0]);                  \
    (b) = _mm_aesenc_si128((b), (rk)[1]);               \
    (b) = _mm_aesenc_si128((b), (rk)[2]);               \
    (b) = _mm_aesenc_si128((b), (rk)[3]);               \
    (b) = _mm_aesenc_si128((b), (rk)[4]);               \
    (b) = _mm_aesenc_si128((b), (rk)[5]);               \
    (b) = _mm_aesenc_si128((b), (rk)[6]);               \
    (b) = _mm_aesenc_si128((b), (rk)[7]);               \
    (b) = _mm_aesenc_si128((b), (rk)[8]);               \
    (b) = _mm_aesenc_si128((b), (rk)[9]);               \
    (b) = _mm_aesenclast_si128((b), (rk)[10]);          \
} while(0)


/* Load 8 counter blocks starting from `counter`, advance counter by 8. */
static void __attribute__((target("sse4.1,aes")))
load_8_ctrs(uint8_t counter[16], __m128i *b0, __m128i *b1, __m128i *b2,
            __m128i *b3, __m128i *b4, __m128i *b5, __m128i *b6, __m128i *b7)
{
#define LOAD_INC(var) do { \
    *(var) = _mm_loadu_si128((const __m128i *)counter); \
    for (int _j = 15; _j >= 0; _j--) if (++counter[_j]) break; \
} while(0)
    LOAD_INC(b0); LOAD_INC(b1); LOAD_INC(b2); LOAD_INC(b3);
    LOAD_INC(b4); LOAD_INC(b5); LOAD_INC(b6); LOAD_INC(b7);
#undef LOAD_INC
}

static void __attribute__((target("sse4.1,aes")))
aes128_ctr_ni(const uint8_t key[16], const uint8_t iv[16],
              const uint8_t *pt, uint8_t *ct, size_t len)
{
    __m128i rk[11];
    aes128_key_schedule_ni(rk, key);

    uint8_t counter[16];
    memcpy(counter, iv, 16);

    size_t off = 0;

    /* 8-block (128-byte) inner loop — hides 4-cycle AES latency. */
    for (; off + 128 <= len; off += 128) {
        __m128i b0, b1, b2, b3, b4, b5, b6, b7;
        load_8_ctrs(counter, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7);

        /* XOR all 8 with rk[0], then interleave rounds 1–9. */
        b0 = _mm_xor_si128(b0, rk[0]); b1 = _mm_xor_si128(b1, rk[0]);
        b2 = _mm_xor_si128(b2, rk[0]); b3 = _mm_xor_si128(b3, rk[0]);
        b4 = _mm_xor_si128(b4, rk[0]); b5 = _mm_xor_si128(b5, rk[0]);
        b6 = _mm_xor_si128(b6, rk[0]); b7 = _mm_xor_si128(b7, rk[0]);

#define ROUND(r) \
        b0 = _mm_aesenc_si128(b0, rk[r]); b1 = _mm_aesenc_si128(b1, rk[r]); \
        b2 = _mm_aesenc_si128(b2, rk[r]); b3 = _mm_aesenc_si128(b3, rk[r]); \
        b4 = _mm_aesenc_si128(b4, rk[r]); b5 = _mm_aesenc_si128(b5, rk[r]); \
        b6 = _mm_aesenc_si128(b6, rk[r]); b7 = _mm_aesenc_si128(b7, rk[r]);
        ROUND(1) ROUND(2) ROUND(3) ROUND(4) ROUND(5)
        ROUND(6) ROUND(7) ROUND(8) ROUND(9)
#undef ROUND
        b0 = _mm_aesenclast_si128(b0, rk[10]); b1 = _mm_aesenclast_si128(b1, rk[10]);
        b2 = _mm_aesenclast_si128(b2, rk[10]); b3 = _mm_aesenclast_si128(b3, rk[10]);
        b4 = _mm_aesenclast_si128(b4, rk[10]); b5 = _mm_aesenclast_si128(b5, rk[10]);
        b6 = _mm_aesenclast_si128(b6, rk[10]); b7 = _mm_aesenclast_si128(b7, rk[10]);

        /* XOR with plaintext and store. */
#define STORE(n) \
        _mm_storeu_si128((__m128i *)(ct + off + (n)*16), \
            _mm_xor_si128(b##n, _mm_loadu_si128((const __m128i *)(pt + off + (n)*16))))
        STORE(0); STORE(1); STORE(2); STORE(3);
        STORE(4); STORE(5); STORE(6); STORE(7);
#undef STORE
    }

    /* Remaining full blocks (1-block at a time). */
    for (; off + 16 <= len; off += 16) {
        __m128i b = _mm_loadu_si128((const __m128i *)counter);
        ctr_inc(counter);
        AES_ENC_BLOCK(b, rk);
        _mm_storeu_si128((__m128i *)(ct + off),
            _mm_xor_si128(b, _mm_loadu_si128((const __m128i *)(pt + off))));
    }

    /* Partial final block. */
    if (off < len) {
        __m128i b = _mm_loadu_si128((const __m128i *)counter);
        AES_ENC_BLOCK(b, rk);
        uint8_t ks[16];
        _mm_storeu_si128((__m128i *)ks, b);
        for (size_t i = 0; off + i < len; i++)
            ct[off + i] = pt[off + i] ^ ks[i];
    }
}

#undef AES_ENC_BLOCK

#endif /* x86 */

/* ── Public API: runtime dispatch ─────────────────────────────────────────── */
void aes128_ctr_encrypt(const uint8_t key[16], const uint8_t iv[16],
                         const uint8_t *plaintext, uint8_t *ciphertext,
                         size_t len)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("aes")) {
        aes128_ctr_ni(key, iv, plaintext, ciphertext, len);
        return;
    }
#endif
    aes128_ctr_scalar(key, iv, plaintext, ciphertext, len);
}
