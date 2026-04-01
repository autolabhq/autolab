#include "solve.h"
#include <string.h>

/*
 * Baseline: portable reference SHA-256 (FIPS 180-4).
 *
 * Bottlenecks (intentional):
 *  1. Message schedule w[16..63] computed inline, one word at a time.
 *  2. 64-round loop with per-round macro expansion for S0/S1/s0/s1.
 *  3. No loop unrolling; compiler sees all round-to-round data dependencies.
 *  4. ROR implemented as two shifts + OR (no SIMD, no SHA-NI).
 */

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

#define ROR32(x, n) (((uint32_t)(x) >> (n)) | ((uint32_t)(x) << (32 - (n))))
#define CH(x,y,z)   (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z)  (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define S0(x)  (ROR32(x, 2)  ^ ROR32(x,13) ^ ROR32(x,22))
#define S1(x)  (ROR32(x, 6)  ^ ROR32(x,11) ^ ROR32(x,25))
#define s0(x)  (ROR32(x, 7)  ^ ROR32(x,18) ^ ((uint32_t)(x) >> 3))
#define s1(x)  (ROR32(x,17)  ^ ROR32(x,19) ^ ((uint32_t)(x) >> 10))

static void sha256_compress(uint32_t st[8], const uint8_t blk[64])
{
    uint32_t w[64];
    int i;

    /* Load message block in big-endian order. */
    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)blk[i*4    ] << 24)
             | ((uint32_t)blk[i*4 + 1] << 16)
             | ((uint32_t)blk[i*4 + 2] <<  8)
             | ((uint32_t)blk[i*4 + 3]);
    }

    /* Extend message schedule to 64 words — computed one at a time. */
    for (i = 16; i < 64; i++)
        w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];

    /* Initialize working variables. */
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32_t e = st[4], f = st[5], g = st[6], h = st[7];

    /* 64 compression rounds. */
    for (i = 0; i < 64; i++) {
        uint32_t t1 = h + S1(e) + CH(e,f,g) + K[i] + w[i];
        uint32_t t2 = S0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

void sha256(const uint8_t *data, size_t len, uint8_t *digest)
{
    uint32_t st[8];
    uint8_t  buf[128];
    size_t   i;

    /* Initialize state. */
    for (i = 0; i < 8; i++) st[i] = H0[i];

    /* Process complete 64-byte blocks. */
    for (i = 0; i + 64 <= len; i += 64)
        sha256_compress(st, data + i);

    /* Padding and final block(s). */
    size_t rem = len - i;
    memset(buf, 0, sizeof(buf));
    memcpy(buf, data + i, rem);
    buf[rem] = 0x80u;

    /* Choose 1 or 2 padding blocks. */
    size_t pad_len = (rem < 56) ? 64 : 128;

    /* Append bit-length as 64-bit big-endian integer. */
    uint64_t bits = (uint64_t)len * 8;
    for (int k = 0; k < 8; k++)
        buf[pad_len - 8 + k] = (uint8_t)(bits >> ((7 - k) * 8));

    sha256_compress(st, buf);
    if (pad_len == 128)
        sha256_compress(st, buf + 64);

    /* Write digest in big-endian byte order. */
    for (i = 0; i < 8; i++) {
        digest[i*4    ] = (uint8_t)(st[i] >> 24);
        digest[i*4 + 1] = (uint8_t)(st[i] >> 16);
        digest[i*4 + 2] = (uint8_t)(st[i] >>  8);
        digest[i*4 + 3] = (uint8_t)(st[i]);
    }
}
