/*
 * solve_optimized.c — High-throughput SHA-256 for sha256_throughput benchmark.
 *
 * Strategy:
 *   1. If the CPU supports SHA-NI + SSE4.1 (runtime CPUID check), use Intel
 *      SHA Extensions: sha256rnds2 / sha256msg1 / sha256msg2.
 *      One 64-byte block in ~130 cycles on modern x86.
 *   2. Otherwise fall back to a scalar optimised version: manually unrolled
 *      64 rounds, precomputed message schedule, traditional ROR macros that
 *      GCC -O2 compiles to native `ror` instructions.
 *
 * Compiles with: gcc -O2 -march=native -std=c99 -Wall -Wextra
 *
 * SHA-NI implementation derived from the Go standard library sha256block_amd64.s
 * (src/crypto/sha256/sha256block_amd64.s), which is based on the Intel SHA
 * Extensions white paper (2013) and Gulley et al.
 */

#include "solve.h"
#include <stdint.h>
#include <string.h>
#include <immintrin.h>  /* SHA-NI + SSE4.1 intrinsics */

/* =========================================================================
 * SHA-256 constants
 * ========================================================================= */

static const uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static const uint32_t H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

/* =========================================================================
 * CPUID helper — detect SHA-NI + SSE4.1
 * ========================================================================= */

static int have_sha_ni(void)
{
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "0"(0u), "2"(0u));
    if (eax < 7u) return 0;
    __asm__ __volatile__("cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "0"(1u), "2"(0u));
    if (!(ecx & (1u << 19))) return 0;  /* SSE4.1 */
    __asm__ __volatile__("cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "0"(7u), "2"(0u));
    return (int)((ebx >> 29) & 1u);    /* SHA */
#else
    return 0;
#endif
}

/* =========================================================================
 * SHA-NI block compression — one 64-byte block
 *
 * Register conventions (matching Go stdlib sha256block_amd64.s):
 *
 *   After state load:
 *     STATE0 = ABEF = { F, E, B, A }   (lane 0..3 = lo..hi in memory)
 *     STATE1 = CDGH = { H, G, D, C }
 *
 *   sha256rnds2(a=CDGH, b=ABEF, k): performs 2 rounds, returns new CDGH.
 *   shuffle_epi32(k, 0x0E) gives the upper 64 bits (rounds i+2, i+3).
 *
 *   Message registers (rotated by sha256msg2 output):
 *     MSG0 initially = w[0..3]  (lane 0..3)
 *     MSG1 initially = w[4..7]
 *     MSG2 initially = w[8..11]
 *     MSG3 initially = w[12..15]
 *
 *   sha256msg1(a, b): returns a + partial_s0, using b[0] for a[3].
 *     C form: new_a = _mm_sha256msg1_epu32(a, b)
 *     Called to prepare a register 2 groups before it's needed.
 *
 *   sha256msg2(a, b): completes message schedule using b[2],b[3] as W14,W15.
 *     C form: new_a = _mm_sha256msg2_epu32(a, b)
 *
 *   alignr(a, b, 4): returns { b[1], b[2], b[3], a[0] }
 *     C: _mm_alignr_epi8(a, b, 4)
 *
 *   For each group (rounds 12-59), the message schedule step is:
 *     TMP       = alignr(m_current, m_prev, 4)   = {m_prev[1..3], m_current[0]}
 *     t         += TMP                           t was prepared by sha256msg1 2 groups ago
 *     t         = sha256msg2(t, m_current)        uses m_current[2,3] as W14,W15
 *     m_prev_of_prev = sha256msg1(m_prev_of_prev, m_current)  prepare for 2 groups later
 * ========================================================================= */

/* Macro: 4 rounds using SHA-NI (rounds i..i+3).
 * MSG_KI = MSG_I + K[i..i+3] already computed.
 * After the macro, STATE0 and STATE1 hold the new compression state.
 */
#define DO4RNDS(s0, s1, mk)                                  \
    do {                                                      \
        __m128i _mk = (mk);                                   \
        (s1) = _mm_sha256rnds2_epu32((s1), (s0), _mk);       \
        _mk  = _mm_shuffle_epi32(_mk, 0x0E);                  \
        (s0) = _mm_sha256rnds2_epu32((s0), (s1), _mk);       \
    } while (0)

__attribute__((target("sse4.1,sha")))
static void sha256_block_sha_ni(uint32_t state[8], const uint8_t *block)
{
    __m128i STATE0, STATE1, ABEF_SAVE, CDGH_SAVE;
    __m128i MSG, TMP, MSG0, MSG1, MSG2, MSG3;
    const __m128i MASK = _mm_set_epi64x(
        (long long)0x0c0d0e0f08090a0bULL,
        (long long)0x0405060700010203ULL);

    /* ---- Load and reorder state ----------------------------------------
       state[] = [A,B,C,D,E,F,G,H].
       Rearrange to: STATE0={F,E,B,A}, STATE1={H,G,D,C}
    -------------------------------------------------------------------- */
    TMP    = _mm_loadu_si128((const __m128i *)(state + 0)); /* {A,B,C,D} */
    STATE1 = _mm_loadu_si128((const __m128i *)(state + 4)); /* {E,F,G,H} */
    TMP    = _mm_shuffle_epi32(TMP,    0xB1); /* {B,A,D,C} */
    STATE1 = _mm_shuffle_epi32(STATE1, 0x1B); /* {H,G,F,E} */
    STATE0 = _mm_alignr_epi8(TMP, STATE1, 8); /* {F,E,B,A} */
    STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0); /* {H,G,D,C} */
    ABEF_SAVE = STATE0;
    CDGH_SAVE = STATE1;

    /* ---- Load message block (big-endian byte-swap) --------------------- */
    MSG0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(block +  0)), MASK);
    MSG1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(block + 16)), MASK);
    MSG2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(block + 32)), MASK);
    MSG3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(block + 48)), MASK);

    /* ---- Rounds 0-3  (no sha256msg1) ----------------------------------- */
    MSG = _mm_add_epi32(MSG0, _mm_set_epi32(
              (int)K[ 3], (int)K[ 2], (int)K[ 1], (int)K[ 0]));
    DO4RNDS(STATE0, STATE1, MSG);

    /* ---- Rounds 4-7 ---------------------------------------------------- */
    MSG = _mm_add_epi32(MSG1, _mm_set_epi32(
              (int)K[ 7], (int)K[ 6], (int)K[ 5], (int)K[ 4]));
    DO4RNDS(STATE0, STATE1, MSG);
    /* Prepare MSG0 (= w[0..3]) for use in rounds 12-15:
       MSG0 = sha256msg1(MSG0, MSG1)  →  MSG0[i] += s0(MSG0[i+1]), MSG0[3] += s0(MSG1[0]) */
    MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

    /* ---- Rounds 8-11 --------------------------------------------------- */
    MSG = _mm_add_epi32(MSG2, _mm_set_epi32(
              (int)K[11], (int)K[10], (int)K[ 9], (int)K[ 8]));
    DO4RNDS(STATE0, STATE1, MSG);
    /* Prepare MSG1 (= w[4..7]) for rounds 16-19: */
    MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

    /* ---- Rounds 12-15 — also produce w[16..19] in MSG0 ---------------- */
    MSG = _mm_add_epi32(MSG3, _mm_set_epi32(
              (int)K[15], (int)K[14], (int)K[13], (int)K[12]));
    DO4RNDS(STATE0, STATE1, MSG);
    /* alignr(MSG3, MSG2_orig, 4) = {MSG2[1],MSG2[2],MSG2[3],MSG3[0]} = {w[9],w[10],w[11],w[12]}
       NOTE: MSG2 not yet sha256msg1'd at this point — it's still original w[8..11] */
    TMP  = _mm_alignr_epi8(MSG3, MSG2, 4);
    MSG0 = _mm_add_epi32(MSG0, TMP);
    /* sha256msg2(MSG0_partial, MSG3) uses MSG3[2]=w[14], MSG3[3]=w[15] → produces w[16..19] */
    MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
    /* Prepare MSG2 (= w[8..11]) for rounds 20-23: */
    MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

    /* ---- Rounds 16-19 — also produce w[20..23] in MSG1 ---------------- */
    MSG = _mm_add_epi32(MSG0, _mm_set_epi32(
              (int)K[19], (int)K[18], (int)K[17], (int)K[16]));
    DO4RNDS(STATE0, STATE1, MSG);
    /* alignr(MSG0=w16-19, MSG3_orig=w12-15, 4) = {w[13],w[14],w[15],w[16]} */
    TMP  = _mm_alignr_epi8(MSG0, MSG3, 4);
    MSG1 = _mm_add_epi32(MSG1, TMP);
    /* sha256msg2(MSG1_partial, MSG0=w16-19) uses MSG0[2]=w[18], MSG0[3]=w[19] → w[20..23] */
    MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
    /* Prepare MSG3 (= w[12..15]) for rounds 24-27: */
    MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

    /* ---- Rounds 20-23 — also produce w[24..27] in MSG2 ---------------- */
    MSG = _mm_add_epi32(MSG1, _mm_set_epi32(
              (int)K[23], (int)K[22], (int)K[21], (int)K[20]));
    DO4RNDS(STATE0, STATE1, MSG);
    /* alignr(MSG1=w20-23, MSG0=w16-19, 4) = {w[17],w[18],w[19],w[20]} */
    TMP  = _mm_alignr_epi8(MSG1, MSG0, 4);
    MSG2 = _mm_add_epi32(MSG2, TMP);
    MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
    /* Prepare MSG0 (= w[16..19]) for rounds 28-31: */
    MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

    /* ---- Rounds 24-27 — also produce w[28..31] in MSG3 ---------------- */
    MSG = _mm_add_epi32(MSG2, _mm_set_epi32(
              (int)K[27], (int)K[26], (int)K[25], (int)K[24]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG2, MSG1, 4);
    MSG3 = _mm_add_epi32(MSG3, TMP);
    MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
    MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

    /* ---- Rounds 28-31 — also produce w[32..35] in MSG0 ---------------- */
    MSG = _mm_add_epi32(MSG3, _mm_set_epi32(
              (int)K[31], (int)K[30], (int)K[29], (int)K[28]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG3, MSG2, 4);
    MSG0 = _mm_add_epi32(MSG0, TMP);
    MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
    MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

    /* ---- Rounds 32-35 — also produce w[36..39] in MSG1 ---------------- */
    MSG = _mm_add_epi32(MSG0, _mm_set_epi32(
              (int)K[35], (int)K[34], (int)K[33], (int)K[32]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG0, MSG3, 4);
    MSG1 = _mm_add_epi32(MSG1, TMP);
    MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
    MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

    /* ---- Rounds 36-39 — also produce w[40..43] in MSG2 ---------------- */
    MSG = _mm_add_epi32(MSG1, _mm_set_epi32(
              (int)K[39], (int)K[38], (int)K[37], (int)K[36]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG1, MSG0, 4);
    MSG2 = _mm_add_epi32(MSG2, TMP);
    MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
    MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

    /* ---- Rounds 40-43 — also produce w[44..47] in MSG3 ---------------- */
    MSG = _mm_add_epi32(MSG2, _mm_set_epi32(
              (int)K[43], (int)K[42], (int)K[41], (int)K[40]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG2, MSG1, 4);
    MSG3 = _mm_add_epi32(MSG3, TMP);
    MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
    MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

    /* ---- Rounds 44-47 — also produce w[48..51] in MSG0 ---------------- */
    MSG = _mm_add_epi32(MSG3, _mm_set_epi32(
              (int)K[47], (int)K[46], (int)K[45], (int)K[44]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG3, MSG2, 4);
    MSG0 = _mm_add_epi32(MSG0, TMP);
    MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
    MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

    /* ---- Rounds 48-51 — also produce w[52..55] in MSG1 ---------------- */
    MSG = _mm_add_epi32(MSG0, _mm_set_epi32(
              (int)K[51], (int)K[50], (int)K[49], (int)K[48]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG0, MSG3, 4);
    MSG1 = _mm_add_epi32(MSG1, TMP);
    MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
    MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

    /* ---- Rounds 52-55 — also produce w[56..59] in MSG2 ---------------- */
    MSG = _mm_add_epi32(MSG1, _mm_set_epi32(
              (int)K[55], (int)K[54], (int)K[53], (int)K[52]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG1, MSG0, 4);
    MSG2 = _mm_add_epi32(MSG2, TMP);
    MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
    /* no more sha256msg1 needed (last two groups don't need future schedule) */

    /* ---- Rounds 56-59 — also produce w[60..63] in MSG3 ---------------- */
    MSG = _mm_add_epi32(MSG2, _mm_set_epi32(
              (int)K[59], (int)K[58], (int)K[57], (int)K[56]));
    DO4RNDS(STATE0, STATE1, MSG);
    TMP  = _mm_alignr_epi8(MSG2, MSG1, 4);
    MSG3 = _mm_add_epi32(MSG3, TMP);
    MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);

    /* ---- Rounds 60-63 -------------------------------------------------- */
    MSG = _mm_add_epi32(MSG3, _mm_set_epi32(
              (int)K[63], (int)K[62], (int)K[61], (int)K[60]));
    DO4RNDS(STATE0, STATE1, MSG);

    /* ---- Add back saved state ------------------------------------------ */
    STATE0 = _mm_add_epi32(STATE0, ABEF_SAVE);
    STATE1 = _mm_add_epi32(STATE1, CDGH_SAVE);

    /* ---- Reorder state back to [A,B,C,D,E,F,G,H] ----------------------- */
    TMP    = _mm_shuffle_epi32(STATE0, 0x1B); /* {A,B,E,F} */
    STATE1 = _mm_shuffle_epi32(STATE1, 0xB1); /* {D,C,H,G} */
    STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0); /* {D,C,B,A} */
    STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);    /* {H,G,F,E} */
    _mm_storeu_si128((__m128i *)(state + 0), STATE0);
    _mm_storeu_si128((__m128i *)(state + 4), STATE1);
}

/* =========================================================================
 * SHA-NI streaming SHA-256
 * ========================================================================= */

__attribute__((target("sse4.1,sha")))
static void sha256_sha_ni(const uint8_t *data, size_t len, uint8_t *digest)
{
    uint32_t state[8];
    uint8_t  buf[128];
    size_t   i;

    for (i = 0; i < 8; i++) state[i] = H0[i];

    for (i = 0; i + 64 <= len; i += 64)
        sha256_block_sha_ni(state, data + i);

    size_t rem = len - i;
    memset(buf, 0, sizeof(buf));
    memcpy(buf, data + i, rem);
    buf[rem] = 0x80u;

    size_t pad_len = (rem < 56) ? 64 : 128;
    uint64_t bits = (uint64_t)len * 8;
    for (int k = 0; k < 8; k++)
        buf[pad_len - 8 + k] = (uint8_t)(bits >> ((7 - k) * 8));

    sha256_block_sha_ni(state, buf);
    if (pad_len == 128)
        sha256_block_sha_ni(state, buf + 64);

    for (i = 0; i < 8; i++) {
        digest[i*4    ] = (uint8_t)(state[i] >> 24);
        digest[i*4 + 1] = (uint8_t)(state[i] >> 16);
        digest[i*4 + 2] = (uint8_t)(state[i] >>  8);
        digest[i*4 + 3] = (uint8_t)(state[i]);
    }
}

/* =========================================================================
 * Scalar optimised fallback
 * ========================================================================= */

#define ROR32(x, n) (((uint32_t)(x) >> (n)) | ((uint32_t)(x) << (32u - (n))))
#define CH(x,y,z)   (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z)  (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define S0(x)  (ROR32(x, 2)  ^ ROR32(x,13) ^ ROR32(x,22))
#define S1(x)  (ROR32(x, 6)  ^ ROR32(x,11) ^ ROR32(x,25))
#define s0(x)  (ROR32(x, 7)  ^ ROR32(x,18) ^ ((uint32_t)(x) >> 3))
#define s1(x)  (ROR32(x,17)  ^ ROR32(x,19) ^ ((uint32_t)(x) >> 10))

#define ROUND(a,b,c,d,e,f,g,h,ki,wi)                           \
    do {                                                        \
        uint32_t _t1 = (h) + S1(e) + CH(e,f,g) + (ki) + (wi); \
        uint32_t _t2 = S0(a) + MAJ(a,b,c);                     \
        (h) = (g); (g) = (f); (f) = (e); (e) = (d) + _t1;     \
        (d) = (c); (c) = (b); (b) = (a); (a) = _t1 + _t2;     \
    } while (0)

static void sha256_compress_scalar(uint32_t st[8], const uint8_t blk[64])
{
    uint32_t w[64];
    int i;

    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)blk[i*4    ] << 24)
             | ((uint32_t)blk[i*4 + 1] << 16)
             | ((uint32_t)blk[i*4 + 2] <<  8)
             |  (uint32_t)blk[i*4 + 3];
    }
    for (i = 16; i < 64; i++)
        w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];

    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32_t e = st[4], f = st[5], g = st[6], h = st[7];

    ROUND(a,b,c,d,e,f,g,h, K[ 0], w[ 0]);
    ROUND(a,b,c,d,e,f,g,h, K[ 1], w[ 1]);
    ROUND(a,b,c,d,e,f,g,h, K[ 2], w[ 2]);
    ROUND(a,b,c,d,e,f,g,h, K[ 3], w[ 3]);
    ROUND(a,b,c,d,e,f,g,h, K[ 4], w[ 4]);
    ROUND(a,b,c,d,e,f,g,h, K[ 5], w[ 5]);
    ROUND(a,b,c,d,e,f,g,h, K[ 6], w[ 6]);
    ROUND(a,b,c,d,e,f,g,h, K[ 7], w[ 7]);
    ROUND(a,b,c,d,e,f,g,h, K[ 8], w[ 8]);
    ROUND(a,b,c,d,e,f,g,h, K[ 9], w[ 9]);
    ROUND(a,b,c,d,e,f,g,h, K[10], w[10]);
    ROUND(a,b,c,d,e,f,g,h, K[11], w[11]);
    ROUND(a,b,c,d,e,f,g,h, K[12], w[12]);
    ROUND(a,b,c,d,e,f,g,h, K[13], w[13]);
    ROUND(a,b,c,d,e,f,g,h, K[14], w[14]);
    ROUND(a,b,c,d,e,f,g,h, K[15], w[15]);
    ROUND(a,b,c,d,e,f,g,h, K[16], w[16]);
    ROUND(a,b,c,d,e,f,g,h, K[17], w[17]);
    ROUND(a,b,c,d,e,f,g,h, K[18], w[18]);
    ROUND(a,b,c,d,e,f,g,h, K[19], w[19]);
    ROUND(a,b,c,d,e,f,g,h, K[20], w[20]);
    ROUND(a,b,c,d,e,f,g,h, K[21], w[21]);
    ROUND(a,b,c,d,e,f,g,h, K[22], w[22]);
    ROUND(a,b,c,d,e,f,g,h, K[23], w[23]);
    ROUND(a,b,c,d,e,f,g,h, K[24], w[24]);
    ROUND(a,b,c,d,e,f,g,h, K[25], w[25]);
    ROUND(a,b,c,d,e,f,g,h, K[26], w[26]);
    ROUND(a,b,c,d,e,f,g,h, K[27], w[27]);
    ROUND(a,b,c,d,e,f,g,h, K[28], w[28]);
    ROUND(a,b,c,d,e,f,g,h, K[29], w[29]);
    ROUND(a,b,c,d,e,f,g,h, K[30], w[30]);
    ROUND(a,b,c,d,e,f,g,h, K[31], w[31]);
    ROUND(a,b,c,d,e,f,g,h, K[32], w[32]);
    ROUND(a,b,c,d,e,f,g,h, K[33], w[33]);
    ROUND(a,b,c,d,e,f,g,h, K[34], w[34]);
    ROUND(a,b,c,d,e,f,g,h, K[35], w[35]);
    ROUND(a,b,c,d,e,f,g,h, K[36], w[36]);
    ROUND(a,b,c,d,e,f,g,h, K[37], w[37]);
    ROUND(a,b,c,d,e,f,g,h, K[38], w[38]);
    ROUND(a,b,c,d,e,f,g,h, K[39], w[39]);
    ROUND(a,b,c,d,e,f,g,h, K[40], w[40]);
    ROUND(a,b,c,d,e,f,g,h, K[41], w[41]);
    ROUND(a,b,c,d,e,f,g,h, K[42], w[42]);
    ROUND(a,b,c,d,e,f,g,h, K[43], w[43]);
    ROUND(a,b,c,d,e,f,g,h, K[44], w[44]);
    ROUND(a,b,c,d,e,f,g,h, K[45], w[45]);
    ROUND(a,b,c,d,e,f,g,h, K[46], w[46]);
    ROUND(a,b,c,d,e,f,g,h, K[47], w[47]);
    ROUND(a,b,c,d,e,f,g,h, K[48], w[48]);
    ROUND(a,b,c,d,e,f,g,h, K[49], w[49]);
    ROUND(a,b,c,d,e,f,g,h, K[50], w[50]);
    ROUND(a,b,c,d,e,f,g,h, K[51], w[51]);
    ROUND(a,b,c,d,e,f,g,h, K[52], w[52]);
    ROUND(a,b,c,d,e,f,g,h, K[53], w[53]);
    ROUND(a,b,c,d,e,f,g,h, K[54], w[54]);
    ROUND(a,b,c,d,e,f,g,h, K[55], w[55]);
    ROUND(a,b,c,d,e,f,g,h, K[56], w[56]);
    ROUND(a,b,c,d,e,f,g,h, K[57], w[57]);
    ROUND(a,b,c,d,e,f,g,h, K[58], w[58]);
    ROUND(a,b,c,d,e,f,g,h, K[59], w[59]);
    ROUND(a,b,c,d,e,f,g,h, K[60], w[60]);
    ROUND(a,b,c,d,e,f,g,h, K[61], w[61]);
    ROUND(a,b,c,d,e,f,g,h, K[62], w[62]);
    ROUND(a,b,c,d,e,f,g,h, K[63], w[63]);

    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

static void sha256_scalar(const uint8_t *data, size_t len, uint8_t *digest)
{
    uint32_t state[8];
    uint8_t  buf[128];
    size_t   i;

    for (i = 0; i < 8; i++) state[i] = H0[i];

    for (i = 0; i + 64 <= len; i += 64)
        sha256_compress_scalar(state, data + i);

    size_t rem = len - i;
    memset(buf, 0, sizeof(buf));
    memcpy(buf, data + i, rem);
    buf[rem] = 0x80u;

    size_t pad_len = (rem < 56) ? 64 : 128;
    uint64_t bits = (uint64_t)len * 8;
    for (int k = 0; k < 8; k++)
        buf[pad_len - 8 + k] = (uint8_t)(bits >> ((7 - k) * 8));

    sha256_compress_scalar(state, buf);
    if (pad_len == 128)
        sha256_compress_scalar(state, buf + 64);

    for (i = 0; i < 8; i++) {
        digest[i*4    ] = (uint8_t)(state[i] >> 24);
        digest[i*4 + 1] = (uint8_t)(state[i] >> 16);
        digest[i*4 + 2] = (uint8_t)(state[i] >>  8);
        digest[i*4 + 3] = (uint8_t)(state[i]);
    }
}

/* =========================================================================
 * Public entry point — runtime dispatch
 * ========================================================================= */

void sha256(const uint8_t *data, size_t len, uint8_t *digest)
{
    static int checked = 0;
    static int use_ni  = 0;

    if (!checked) {
        use_ni  = have_sha_ni();
        checked = 1;
    }

    if (use_ni)
        sha256_sha_ni(data, len, digest);
    else
        sha256_scalar(data, len, digest);
}
