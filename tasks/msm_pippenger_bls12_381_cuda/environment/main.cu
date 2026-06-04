/*
 * main.cu -- harness for the BLS12-381 G1 MSM CUDA task.  DO NOT MODIFY.
 *
 * Modes:
 *   ./msm_bls12381 --verify
 *       Public small N (VERIFY_N), runtime-random seed.  Compares the agent's
 *       Q against an in-binary CPU reference MSM (naive double-and-add) and
 *       prints  __VERIFIER_CORRECTNESS__=PASS  on success.
 *
 *   ./msm_bls12381 --benchmark
 *       Public, fixed-seed timing run on the bench shape.  Useful for the
 *       agent's iteration loop; result is unscored.
 *
 *   ./msm_bls12381 --benchmark-verify
 *       Hidden-seed scored run.  Reads a 64-bit seed from $MSM_BENCH_SEED
 *       (set by tests/test.sh) and runs:
 *         (a) CORRECTNESS pass on the small verify shape against the in-
 *             binary CPU reference, and
 *         (b) TIMING pass on the bench shape (BENCH_N), warmup +
 *             N_TIMED iterations; takes the median.  Also re-checks the
 *             scored output bit-exactly against the CPU reference (run on
 *             a small hidden subset of the bench inputs to keep the cost
 *             under control: we verify a separate small-N MSM derived
 *             from the same hidden seed has Q matching the agent's run on
 *             that same small-N input).
 *       Both phases must succeed to print sentinels:
 *           __VERIFIER_CORRECTNESS__=PASS
 *           __VERIFIER_BENCHMARK__=PASS
 *           __VERIFIER_SCORE__=<median_milliseconds>
 *       The seed is mixed into both the points and the scalars so the
 *       agent cannot precompute the answer.
 *
 * All Montgomery constants and curve helpers live in an anonymous
 * namespace so they cannot be reached from solve.cu by linking or by
 * textual inclusion of main.cu.
 */

#include "solve.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK_BOOL(expr)                                                   \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                    \
                         cudaGetErrorString(err__), __FILE__, __LINE__);        \
            return false;                                                       \
        }                                                                       \
    } while (0)

namespace {

// ─── BLS12-381 base-field constants (Montgomery, 384-bit) ────────────────
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
// (little-endian limbs).
static constexpr uint64_t FP_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
// p_inv = -p^{-1} mod 2^64 = 0x89f3fffcfffcfffd
static constexpr uint64_t FP_P_INV = 0x89f3fffcfffcfffdULL;
// R^2 = (2^384)^2 mod p, used to convert raw-> Montgomery via mul_mont(a, R^2).
static constexpr uint64_t FP_R2[6] = {
    0xf4df1f341c341746ULL, 0x0a76e6a609d104f1ULL, 0x8de5476c4c95b6d5ULL,
    0x67eb88a9939d83c0ULL, 0x9a793e85b519952dULL, 0x11988fe592cae3aaULL
};
// R = 2^384 mod p (= "one" in Montgomery form).
static constexpr uint64_t FP_ONE_M[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL, 0x5f48985753c758baULL,
    0x77ce585370525745ULL, 0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};
// Curve B = 4 in Montgomery form (b * R mod p). Used for y^2 = x^3 + b.
static constexpr uint64_t FP_B_M[6] = {
    0xaa270000000cfff3ULL, 0x53cc0032fc34000aULL, 0x478fe97a6b0a807fULL,
    0xb1d37ebee6ba24d7ULL, 0x8ec9733bbf78ab2fULL, 0x09d645513d83de7eULL
};
// Generator G1 (canonical), Montgomery form.
// G_x = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
// G_y = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1
static constexpr uint64_t FP_GX_M[6] = {
    0x5cb38790fd530c16ULL, 0x7817fc679976fff5ULL, 0x154f95c7143ba1c1ULL,
    0xf0ae6acdf3d0e747ULL, 0xedce6ecc21dbf440ULL, 0x120177419e0bfb75ULL
};
static constexpr uint64_t FP_GY_M[6] = {
    0xbaac93d50ce72271ULL, 0x8c22631a7918fd8eULL, 0xdd595f13570725ceULL,
    0x51ac582950405194ULL, 0x0e1c8c3fad0059c0ULL, 0x0bbc3efc5008a26aULL
};

// ─── PRNG ─────────────────────────────────────────────────────────────
struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

// ─── 384-bit field arithmetic (host, Montgomery) ──────────────────────────

using fp_t = uint64_t[6];

static inline void fp_set_zero(uint64_t* r) { for (int i = 0; i < 6; ++i) r[i] = 0; }
static inline void fp_copy(uint64_t* r, const uint64_t* a) { for (int i = 0; i < 6; ++i) r[i] = a[i]; }
static inline bool fp_is_zero(const uint64_t* a) {
    uint64_t x = 0; for (int i = 0; i < 6; ++i) x |= a[i]; return x == 0;
}
static inline bool fp_eq(const uint64_t* a, const uint64_t* b) {
    for (int i = 0; i < 6; ++i) if (a[i] != b[i]) return false;
    return true;
}
static inline bool fp_ge_p(const uint64_t* a) {
    for (int i = 5; i >= 0; --i) {
        if (a[i] > FP_P[i]) return true;
        if (a[i] < FP_P[i]) return false;
    }
    return true;  // equal
}

// r = a + b mod p
static inline void fp_add(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t carry = 0;
    uint64_t t[6];
    for (int i = 0; i < 6; ++i) {
        __uint128_t s = (__uint128_t)a[i] + b[i] + carry;
        t[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
    // Conditional subtract p
    if (carry || fp_ge_p(t)) {
        uint64_t borrow = 0;
        for (int i = 0; i < 6; ++i) {
            __uint128_t d = (__uint128_t)t[i] - FP_P[i] - borrow;
            r[i] = (uint64_t)d;
            borrow = (uint64_t)((d >> 64) & 1);
        }
    } else {
        fp_copy(r, t);
    }
}

// r = a - b mod p
static inline void fp_sub(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t borrow = 0;
    uint64_t t[6];
    for (int i = 0; i < 6; ++i) {
        __uint128_t d = (__uint128_t)a[i] - b[i] - borrow;
        t[i] = (uint64_t)d;
        borrow = (uint64_t)((d >> 64) & 1);
    }
    if (borrow) {
        uint64_t c = 0;
        for (int i = 0; i < 6; ++i) {
            __uint128_t s = (__uint128_t)t[i] + FP_P[i] + c;
            r[i] = (uint64_t)s;
            c = (uint64_t)(s >> 64);
        }
    } else {
        fp_copy(r, t);
    }
}

// r = -a mod p
static inline void fp_neg(uint64_t* r, const uint64_t* a) {
    static const uint64_t zero[6] = {0,0,0,0,0,0};
    if (fp_is_zero(a)) { fp_set_zero(r); return; }
    fp_sub(r, FP_P, a);
    (void)zero;
}

// CIOS Montgomery multiplication: r = a * b * R^{-1} mod p.
static inline void fp_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 6; ++i) {
        // t += a * b[i]
        uint64_t carry = 0;
        for (int j = 0; j < 6; ++j) {
            __uint128_t s = (__uint128_t)a[j] * b[i] + t[j] + carry;
            t[j] = (uint64_t)s;
            carry = (uint64_t)(s >> 64);
        }
        __uint128_t s = (__uint128_t)t[6] + carry;
        t[6] = (uint64_t)s;
        t[7] += (uint64_t)(s >> 64);

        // m = t[0] * p_inv mod 2^64
        uint64_t m = t[0] * FP_P_INV;

        // t += m * p
        carry = 0;
        for (int j = 0; j < 6; ++j) {
            __uint128_t s2 = (__uint128_t)m * FP_P[j] + t[j] + carry;
            t[j] = (uint64_t)s2;
            carry = (uint64_t)(s2 >> 64);
        }
        __uint128_t s3 = (__uint128_t)t[6] + carry;
        t[6] = (uint64_t)s3;
        t[7] += (uint64_t)(s3 >> 64);

        // shift t right by one limb
        for (int j = 0; j < 7; ++j) t[j] = t[j+1];
        t[7] = 0;
    }
    // Now result is in t[0..6], may need final reduction.
    uint64_t out[6];
    for (int i = 0; i < 6; ++i) out[i] = t[i];
    if (t[6] || fp_ge_p(out)) {
        uint64_t borrow = 0;
        for (int i = 0; i < 6; ++i) {
            __uint128_t d = (__uint128_t)out[i] - FP_P[i] - borrow;
            r[i] = (uint64_t)d;
            borrow = (uint64_t)((d >> 64) & 1);
        }
    } else {
        fp_copy(r, out);
    }
}

static inline void fp_sqr(uint64_t* r, const uint64_t* a) {
    fp_mul(r, a, a);
}

// Convert raw (non-Montgomery) -> Montgomery: a_m = a * R mod p = mul(a, R^2).
static inline void fp_to_mont(uint64_t* r, const uint64_t* a) {
    fp_mul(r, a, FP_R2);
}

// ─── Jacobian point arithmetic on G1 (host, Montgomery form internally) ───
// Identity is (X=1, Y=1, Z=0).

struct G1Jac {
    uint64_t X[6];
    uint64_t Y[6];
    uint64_t Z[6];
};

static inline bool jac_is_zero(const G1Jac& p) {
    return fp_is_zero(p.Z);
}

static inline void jac_set_zero(G1Jac& p) {
    fp_copy(p.X, FP_ONE_M);
    fp_copy(p.Y, FP_ONE_M);
    fp_set_zero(p.Z);
}

// Convert affine to Jacobian. Affine zero (x=0, y=0) -> Jac zero.
static inline void affine_to_jac(G1Jac& out, const G1Affine& a) {
    if (fp_is_zero(a.x) && fp_is_zero(a.y)) {
        jac_set_zero(out);
        return;
    }
    fp_copy(out.X, a.x);
    fp_copy(out.Y, a.y);
    fp_copy(out.Z, FP_ONE_M);
}

// Doubling (Jacobian). 2P. y^2 = x^3 + 4 (a=0), use a=0 doubling formula.
// X3 = (3 X^2)^2 - 8 X Y^2
// Y3 = (3 X^2)(4 X Y^2 - X3) - 8 Y^4
// Z3 = 2 Y Z
// IMPORTANT: this routine is alias-safe (r and p may be the same object).
// We compute everything into local temporaries and only assign r.X/Y/Z at
// the very end.
static inline void jac_double(G1Jac& r, const G1Jac& p) {
    if (jac_is_zero(p) || fp_is_zero(p.Y)) {
        jac_set_zero(r);
        return;
    }
    uint64_t A[6], B[6], C[6], D[6], E[6], F[6], tmp[6];
    uint64_t newX[6], newY[6], newZ[6];
    fp_sqr(A, p.X);                  // A = X^2
    fp_sqr(B, p.Y);                  // B = Y^2
    fp_sqr(C, B);                    // C = B^2 = Y^4
    fp_add(D, p.X, B);
    fp_sqr(D, D);
    fp_sub(D, D, A);
    fp_sub(D, D, C);
    fp_add(D, D, D);                 // D = 4 X B
    fp_add(E, A, A);
    fp_add(E, E, A);                 // E = 3 X^2
    fp_sqr(F, E);
    fp_sub(tmp, F, D);
    fp_sub(newX, tmp, D);            // X3 = F - 2D
    fp_sub(tmp, D, newX);
    fp_mul(tmp, E, tmp);
    uint64_t eight_c[6];
    fp_add(eight_c, C, C);
    fp_add(eight_c, eight_c, eight_c);
    fp_add(eight_c, eight_c, eight_c);
    fp_sub(newY, tmp, eight_c);      // Y3 = E*(D - X3) - 8C
    fp_mul(tmp, p.Y, p.Z);
    fp_add(newZ, tmp, tmp);          // Z3 = 2 Y Z
    fp_copy(r.X, newX);
    fp_copy(r.Y, newY);
    fp_copy(r.Z, newZ);
}

// Mixed Jacobian + Affine addition: r = p + q, q affine, q != 0.
// If q is the identity (affine zero), just copy p. Caller may pass q=zero.
static inline void jac_add_affine(G1Jac& r, const G1Jac& p, const G1Affine& q) {
    if (fp_is_zero(q.x) && fp_is_zero(q.y)) { r = p; return; }
    if (jac_is_zero(p)) {
        fp_copy(r.X, q.x); fp_copy(r.Y, q.y); fp_copy(r.Z, FP_ONE_M);
        return;
    }
    // From "Hankerson/Menezes/Vanstone" / "Cohen/Miyaji/Ono" mixed addition:
    // Z1Z1 = Z1^2
    // U2   = X2 * Z1Z1
    // S2   = Y2 * Z1 * Z1Z1
    // H    = U2 - X1
    // HH   = H^2
    // I    = 4 HH
    // J    = H * I
    // rr   = 2*(S2 - Y1)
    // V    = X1 * I
    // X3   = rr^2 - J - 2 V
    // Y3   = rr*(V - X3) - 2 Y1 J
    // Z3   = (Z1 + H)^2 - Z1Z1 - HH
    uint64_t Z1Z1[6], U2[6], S2[6], H[6], HH[6], I[6], J[6], rr[6], V[6], tmp[6];
    fp_sqr(Z1Z1, p.Z);
    fp_mul(U2, q.x, Z1Z1);
    fp_mul(tmp, q.y, p.Z);
    fp_mul(S2, tmp, Z1Z1);
    if (fp_eq(U2, p.X)) {
        if (fp_eq(S2, p.Y)) {
            // p == q -> double
            G1Jac qj;
            fp_copy(qj.X, q.x); fp_copy(qj.Y, q.y); fp_copy(qj.Z, FP_ONE_M);
            jac_double(r, qj);
            return;
        }
        // p == -q -> identity
        jac_set_zero(r);
        return;
    }
    fp_sub(H, U2, p.X);
    fp_sqr(HH, H);
    fp_add(I, HH, HH); fp_add(I, I, I);                 // I = 4 HH
    fp_mul(J, H, I);
    fp_sub(rr, S2, p.Y); fp_add(rr, rr, rr);            // rr = 2 (S2 - Y1)
    fp_mul(V, p.X, I);
    uint64_t newX[6], newY[6], newZ[6];
    fp_sqr(tmp, rr);
    fp_sub(tmp, tmp, J);
    fp_sub(tmp, tmp, V);
    fp_sub(newX, tmp, V);                                // X3 = rr^2 - J - 2V
    fp_sub(tmp, V, newX);
    fp_mul(tmp, rr, tmp);
    uint64_t y1j[6];
    fp_mul(y1j, p.Y, J); fp_add(y1j, y1j, y1j);
    fp_sub(newY, tmp, y1j);                              // Y3 = rr*(V-X3) - 2 Y1 J
    fp_add(tmp, p.Z, H);
    fp_sqr(tmp, tmp);
    fp_sub(tmp, tmp, Z1Z1);
    fp_sub(newZ, tmp, HH);                               // Z3 = (Z1+H)^2 - Z1Z1 - HH
    fp_copy(r.X, newX); fp_copy(r.Y, newY); fp_copy(r.Z, newZ);
}

// Scalar multiply: r = k * P (k as 4 little-endian uint64s, 256 bits).
// Naive double-and-add, MSB-first.
static inline void jac_scalar_mul(G1Jac& r, const G1Affine& P, const uint64_t k[4]) {
    G1Jac acc; jac_set_zero(acc);
    bool started = false;
    for (int word = 3; word >= 0; --word) {
        for (int bit = 63; bit >= 0; --bit) {
            if (started) jac_double(acc, acc);
            uint64_t b = (k[word] >> bit) & 1ULL;
            if (b) {
                if (!started) {
                    fp_copy(acc.X, P.x); fp_copy(acc.Y, P.y); fp_copy(acc.Z, FP_ONE_M);
                    if (fp_is_zero(P.x) && fp_is_zero(P.y)) jac_set_zero(acc);
                    started = true;
                } else {
                    jac_add_affine(acc, acc, P);
                }
            }
        }
    }
    r = acc;
}

// Convert Jacobian -> affine (for output). Uses inversion via Fermat.
// p inverse: a^{p-2} mod p. p-2 in little-endian limbs.
static inline void fp_inv(uint64_t* r, const uint64_t* a) {
    // p - 2 limbs (since p ends in 0xaaab, p-2 ends in 0xaaa9):
    static constexpr uint64_t P_MINUS_2[6] = {
        0xb9feffffffffaaa9ULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
        0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
    };
    uint64_t result[6];
    fp_copy(result, FP_ONE_M);
    uint64_t base[6]; fp_copy(base, a);
    for (int word = 0; word < 6; ++word) {
        for (int bit = 0; bit < 64; ++bit) {
            if ((P_MINUS_2[word] >> bit) & 1ULL) {
                fp_mul(result, result, base);
            }
            fp_sqr(base, base);
        }
    }
    fp_copy(r, result);
}

static inline void jac_to_affine(G1Affine& out, const G1Jac& p) {
    if (jac_is_zero(p)) {
        fp_set_zero(out.x); fp_set_zero(out.y);
        return;
    }
    uint64_t z_inv[6], z_inv2[6], z_inv3[6];
    fp_inv(z_inv, p.Z);
    fp_sqr(z_inv2, z_inv);
    fp_mul(z_inv3, z_inv2, z_inv);
    fp_mul(out.x, p.X, z_inv2);
    fp_mul(out.y, p.Y, z_inv3);
}

static inline void jac_add_jac(G1Jac& r, const G1Jac& p, const G1Jac& q) {
    if (jac_is_zero(p)) { r = q; return; }
    if (jac_is_zero(q)) { r = p; return; }
    // Cohen-Miyaji-Ono full Jacobian addition.
    uint64_t Z1Z1[6], Z2Z2[6], U1[6], U2[6], S1[6], S2[6];
    uint64_t H[6], I[6], J[6], rr[6], V[6], tmp[6];
    fp_sqr(Z1Z1, p.Z);
    fp_sqr(Z2Z2, q.Z);
    fp_mul(U1, p.X, Z2Z2);
    fp_mul(U2, q.X, Z1Z1);
    fp_mul(tmp, p.Y, q.Z);
    fp_mul(S1, tmp, Z2Z2);
    fp_mul(tmp, q.Y, p.Z);
    fp_mul(S2, tmp, Z1Z1);
    if (fp_eq(U1, U2)) {
        if (fp_eq(S1, S2)) { jac_double(r, p); return; }
        jac_set_zero(r); return;
    }
    fp_sub(H, U2, U1);
    fp_add(tmp, H, H); fp_sqr(I, tmp);                  // I = (2H)^2
    fp_mul(J, H, I);
    fp_sub(rr, S2, S1); fp_add(rr, rr, rr);
    fp_mul(V, U1, I);
    uint64_t newX[6], newY[6], newZ[6];
    fp_sqr(tmp, rr); fp_sub(tmp, tmp, J);
    fp_sub(tmp, tmp, V); fp_sub(newX, tmp, V);
    fp_sub(tmp, V, newX); fp_mul(tmp, rr, tmp);
    uint64_t s1j[6]; fp_mul(s1j, S1, J); fp_add(s1j, s1j, s1j);
    fp_sub(newY, tmp, s1j);
    fp_add(tmp, p.Z, q.Z); fp_sqr(tmp, tmp);
    fp_sub(tmp, tmp, Z1Z1); fp_sub(tmp, tmp, Z2Z2);
    fp_mul(newZ, tmp, H);
    fp_copy(r.X, newX); fp_copy(r.Y, newY); fp_copy(r.Z, newZ);
}

// ─── Test-input synthesis ────────────────────────────────────────────────
//
// To make the CPU reference cheap, we do NOT generate random-looking points
// from scratch (which would require N random scalar mults of the generator,
// quadratically expensive on the bench shape). Instead:
//
//   - Pick a per-i scalar t_i and a per-i scalar s_i (both from the seed).
//   - Set P_i = t_i * G  (the harness does this once on the host).
//   - Then  Q = sum_i s_i * P_i  =  (sum_i s_i * t_i) * G.
//
// The sum-of-products lives in the 256-bit ring; its overflow is absorbed
// because (s_i * t_i) is a 512-bit value and we sum over N. We carry a
// 768-bit accumulator on the host, then do ONE scalar mult to get Q.
// This keeps the CPU reference O(log r) instead of O(N log r).
//
// IMPORTANT: this synthesis is identifiable by the agent (every P_i is on
// the cyclic group generated by G, which is true for ALL of G1 anyway since
// G1 has prime order, so this is not a structural shortcut). The agent's
// MSM implementation must still compute Q correctly element-by-element.

struct Inputs {
    int N;
    std::vector<G1Affine> points;
    std::vector<Scalar256> scalars;
    G1Affine expected_q;
};

// Multi-precision shift-and-add: acc[12] += s * t  (each 4-limb 256-bit).
static inline void mac_512_into_768(uint64_t acc[12], const uint64_t s[4], const uint64_t t[4]) {
    // First compute prod[8] = s * t (8 x uint64 = 512 bits).
    uint64_t prod[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            __uint128_t v = (__uint128_t)s[i] * t[j] + prod[i+j] + carry;
            prod[i+j] = (uint64_t)v;
            carry = (uint64_t)(v >> 64);
        }
        prod[i+4] += carry;
    }
    // Then acc += prod (with full carry propagation through the upper words).
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
        __uint128_t v = (__uint128_t)acc[i] + prod[i] + carry;
        acc[i] = (uint64_t)v;
        carry = (uint64_t)(v >> 64);
    }
    for (int i = 8; i < 12; ++i) {
        __uint128_t v = (__uint128_t)acc[i] + carry;
        acc[i] = (uint64_t)v;
        carry = (uint64_t)(v >> 64);
    }
}

// G1 scalar mul where the scalar is up to 768 bits (12 limbs little-endian).
static inline void jac_scalar_mul_768(G1Jac& r, const G1Affine& P, const uint64_t k[12]) {
    G1Jac acc; jac_set_zero(acc);
    bool started = false;
    for (int word = 11; word >= 0; --word) {
        for (int bit = 63; bit >= 0; --bit) {
            if (started) jac_double(acc, acc);
            uint64_t b = (k[word] >> bit) & 1ULL;
            if (b) {
                if (!started) {
                    fp_copy(acc.X, P.x); fp_copy(acc.Y, P.y); fp_copy(acc.Z, FP_ONE_M);
                    if (fp_is_zero(P.x) && fp_is_zero(P.y)) jac_set_zero(acc);
                    started = true;
                } else {
                    jac_add_affine(acc, acc, P);
                }
            }
        }
    }
    r = acc;
}

static void scalar_from_rng(uint64_t out[4], SplitMix64& rng) {
    out[0] = rng.next();
    out[1] = rng.next();
    out[2] = rng.next();
    out[3] = rng.next();
    // Mask the top bit so we stay below 2^255 (well within scalar field but also
    // safe for arithmetic: 256-bit scalars are explicitly accepted by the kernel).
    out[3] &= 0x7fffffffffffffffULL;
}

static bool generate_inputs(int N, uint64_t seed, Inputs& out, bool verbose=false) {
    out.N = N;
    out.points.assign(N, G1Affine{});
    out.scalars.assign(N, Scalar256{});

    G1Affine G;
    fp_copy(G.x, FP_GX_M);
    fp_copy(G.y, FP_GY_M);

    // Master accumulator for sum_i s_i * t_i (768 bits).
    uint64_t acc[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

    SplitMix64 rng(seed ^ 0xa1b2c3d4e5f6a789ULL);
    for (int i = 0; i < N; ++i) {
        uint64_t t[4]; scalar_from_rng(t, rng);
        uint64_t s[4]; scalar_from_rng(s, rng);

        // P_i = t_i * G
        G1Jac Pj; jac_scalar_mul(Pj, G, t);
        jac_to_affine(out.points[i], Pj);

        // s_i
        for (int k = 0; k < 4; ++k) out.scalars[i].limbs[k] = s[k];

        // acc += s_i * t_i
        mac_512_into_768(acc, s, t);
    }

    // Q = acc * G  (acc is up to 768 bits)
    G1Jac Qj; jac_scalar_mul_768(Qj, G, acc);
    jac_to_affine(out.expected_q, Qj);
    if (verbose) {
        std::fprintf(stderr, "ref Q.x[0]=0x%016llx Q.y[0]=0x%016llx\n",
                     (unsigned long long)out.expected_q.x[0],
                     (unsigned long long)out.expected_q.y[0]);
    }
    return true;
}

// ─── Buffer helpers ────────────────────────────────────────────────────

struct DeviceBuffers {
    G1Affine* d_points = nullptr;
    Scalar256* d_scalars = nullptr;
    G1Affine* d_q = nullptr;
    void* d_workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaStream_t stream = nullptr;
    int N = 0;
};

static size_t workspace_bytes_for(int N) {
    // Generous: 256 MiB + 2 KiB per point. Pippenger with chunked phase-1
    // needs ~ NUM_WINDOWS * num_chunks * NUM_BUCKETS * sizeof(JacD) bytes,
    // which for c=8, CHUNK=1024, N=2^18 is ~300 MiB. Provide plenty.
    size_t base = (size_t)512 * 1024 * 1024;
    size_t per = (size_t)2048;
    return base + per * (size_t)N;
}

static void release_device(DeviceBuffers& dev) {
    if (dev.d_points)    cudaFree(dev.d_points);
    if (dev.d_scalars)   cudaFree(dev.d_scalars);
    if (dev.d_q)         cudaFree(dev.d_q);
    if (dev.d_workspace) cudaFree(dev.d_workspace);
    if (dev.stream)      cudaStreamDestroy(dev.stream);
    dev = DeviceBuffers{};
}

static bool alloc_device(int N, DeviceBuffers& dev) {
    dev.N = N;
    CUDA_CHECK_BOOL(cudaStreamCreate(&dev.stream));
    CUDA_CHECK_BOOL(cudaMalloc(&dev.d_points, (size_t)N * sizeof(G1Affine)));
    CUDA_CHECK_BOOL(cudaMalloc(&dev.d_scalars, (size_t)N * sizeof(Scalar256)));
    CUDA_CHECK_BOOL(cudaMalloc(&dev.d_q, sizeof(G1Affine)));
    dev.workspace_bytes = workspace_bytes_for(N);
    CUDA_CHECK_BOOL(cudaMalloc(&dev.d_workspace, dev.workspace_bytes));
    return true;
}

static bool reset_input(const Inputs& in, DeviceBuffers& dev) {
    CUDA_CHECK_BOOL(cudaMemcpyAsync(dev.d_points, in.points.data(),
                                    (size_t)in.N * sizeof(G1Affine),
                                    cudaMemcpyHostToDevice, dev.stream));
    CUDA_CHECK_BOOL(cudaMemcpyAsync(dev.d_scalars, in.scalars.data(),
                                    (size_t)in.N * sizeof(Scalar256),
                                    cudaMemcpyHostToDevice, dev.stream));
    // Clear out_q so a forgetful agent is detected.
    CUDA_CHECK_BOOL(cudaMemsetAsync(dev.d_q, 0, sizeof(G1Affine), dev.stream));
    CUDA_CHECK_BOOL(cudaStreamSynchronize(dev.stream));
    return true;
}

static bool fetch_q(DeviceBuffers& dev, G1Affine& got) {
    CUDA_CHECK_BOOL(cudaMemcpy(&got, dev.d_q, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    return true;
}

static bool q_eq(const G1Affine& a, const G1Affine& b) {
    return fp_eq(a.x, b.x) && fp_eq(a.y, b.y);
}

// ─── Run modes ──────────────────────────────────────────────────────────

constexpr int VERIFY_N = 1 << 14;   // 16384
constexpr int BENCH_N  = 1 << 18;   // 262144

constexpr int N_WARMUP = 10;
constexpr int N_TIMED  = 30;

static bool run_correctness(int N, uint64_t seed) {
    Inputs in;
    if (!generate_inputs(N, seed, in)) return false;

    DeviceBuffers dev;
    if (!alloc_device(N, dev)) { release_device(dev); return false; }
    if (!reset_input(in, dev)) { release_device(dev); return false; }

    msm_bls12_381_g1(dev.d_points, dev.d_scalars, dev.d_q,
                     dev.d_workspace, dev.workspace_bytes,
                     N, dev.stream);
    cudaError_t err = cudaStreamSynchronize(dev.stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "stream sync error: %s\n", cudaGetErrorString(err));
        release_device(dev);
        return false;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err));
        release_device(dev);
        return false;
    }

    G1Affine got;
    if (!fetch_q(dev, got)) { release_device(dev); return false; }
    release_device(dev);

    if (!q_eq(got, in.expected_q)) {
        std::fprintf(stderr, "MSM mismatch:\n");
        for (int i = 0; i < 6; ++i)
            std::fprintf(stderr, "  got.x[%d]=0x%016llx ref.x[%d]=0x%016llx\n",
                         i, (unsigned long long)got.x[i],
                         i, (unsigned long long)in.expected_q.x[i]);
        for (int i = 0; i < 6; ++i)
            std::fprintf(stderr, "  got.y[%d]=0x%016llx ref.y[%d]=0x%016llx\n",
                         i, (unsigned long long)got.y[i],
                         i, (unsigned long long)in.expected_q.y[i]);
        return false;
    }
    return true;
}

static bool run_timed(int N, uint64_t seed, double* median_ms) {
    Inputs in;
    if (!generate_inputs(N, seed, in)) return false;

    DeviceBuffers dev;
    if (!alloc_device(N, dev)) { release_device(dev); return false; }

    for (int i = 0; i < N_WARMUP; ++i) {
        if (!reset_input(in, dev)) { release_device(dev); return false; }
        msm_bls12_381_g1(dev.d_points, dev.d_scalars, dev.d_q,
                         dev.d_workspace, dev.workspace_bytes,
                         N, dev.stream);
        if (cudaStreamSynchronize(dev.stream) != cudaSuccess) {
            release_device(dev); return false;
        }
        if (cudaGetLastError() != cudaSuccess) {
            release_device(dev); return false;
        }
    }

    cudaEvent_t e0, e1;
    if (cudaEventCreate(&e0) != cudaSuccess) { release_device(dev); return false; }
    if (cudaEventCreate(&e1) != cudaSuccess) {
        cudaEventDestroy(e0); release_device(dev); return false;
    }

    std::vector<double> times; times.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        if (!reset_input(in, dev)) {
            cudaEventDestroy(e0); cudaEventDestroy(e1); release_device(dev); return false;
        }
        if (cudaStreamSynchronize(dev.stream) != cudaSuccess) {
            cudaEventDestroy(e0); cudaEventDestroy(e1); release_device(dev); return false;
        }
        cudaEventRecord(e0, dev.stream);
        msm_bls12_381_g1(dev.d_points, dev.d_scalars, dev.d_q,
                         dev.d_workspace, dev.workspace_bytes,
                         N, dev.stream);
        cudaEventRecord(e1, dev.stream);
        if (cudaEventSynchronize(e1) != cudaSuccess) {
            cudaEventDestroy(e0); cudaEventDestroy(e1); release_device(dev); return false;
        }
        if (cudaGetLastError() != cudaSuccess) {
            cudaEventDestroy(e0); cudaEventDestroy(e1); release_device(dev); return false;
        }
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        times.push_back((double)ms);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    G1Affine got;
    if (!fetch_q(dev, got)) { release_device(dev); return false; }
    release_device(dev);

    // The FINAL timed output must equal the CPU reference for the bench shape.
    if (!q_eq(got, in.expected_q)) {
        std::fprintf(stderr, "timed MSM mismatch (bench shape):\n");
        for (int i = 0; i < 6; ++i)
            std::fprintf(stderr, "  got.x[%d]=0x%016llx ref.x[%d]=0x%016llx\n",
                         i, (unsigned long long)got.x[i],
                         i, (unsigned long long)in.expected_q.x[i]);
        return false;
    }

    std::sort(times.begin(), times.end());
    *median_ms = times[times.size() / 2];
    return true;
}

}  // anonymous namespace

int main(int argc, char** argv) {
    const char* mode = (argc >= 2) ? argv[1] : "--benchmark";

    if (std::strcmp(mode, "--verify") == 0) {
        uint64_t seed = (uint64_t)std::time(nullptr)
                      ^ ((uint64_t)getpid() << 32)
                      ^ 0xCAFEBABEDEADBEEFULL;
        if (!run_correctness(VERIFY_N, seed)) {
            std::printf("FAIL verify\n");
            return 1;
        }
        std::printf("PASS verify N=%d\n", VERIFY_N);
        std::fprintf(stderr, "__VERIFIER_CORRECTNESS__=PASS\n");
        return 0;
    }

    if (std::strcmp(mode, "--benchmark") == 0) {
        double median_ms = 0.0;
        if (!run_timed(BENCH_N, 0xC0FFEE0123456789ULL, &median_ms)) {
            std::printf("FAIL benchmark\n");
            return 1;
        }
        std::printf("result=ok N=%d median_ms=%.6f\n", BENCH_N, median_ms);
        return 0;
    }

    if (std::strcmp(mode, "--benchmark-verify") == 0) {
        const char* sv = std::getenv("MSM_BENCH_SEED");
        if (!sv || !*sv) {
            std::fprintf(stderr, "ERROR: --benchmark-verify requires MSM_BENCH_SEED env\n");
            return 2;
        }
        uint64_t seed = std::strtoull(sv, nullptr, 0);

        // Phase 1: small-N correctness with the same hidden seed.
        if (!run_correctness(VERIFY_N, seed)) {
            std::printf("FAIL benchmark-verify (correctness)\n");
            return 1;
        }
        std::fprintf(stderr, "__VERIFIER_CORRECTNESS__=PASS\n");

        // Phase 2: timed bench-N with same hidden seed; the timing harness
        // also checks bit-exact correctness on the timed output.
        double median_ms = 0.0;
        if (!run_timed(BENCH_N, seed, &median_ms)) {
            std::printf("FAIL benchmark-verify (timed)\n");
            return 1;
        }
        std::printf("result=ok N=%d median_ms=%.6f\n", BENCH_N, median_ms);
        std::fprintf(stderr, "__VERIFIER_BENCHMARK__=PASS\n");
        std::fprintf(stderr, "__VERIFIER_SCORE__=%.6f\n", median_ms);
        return 0;
    }

    std::fprintf(stderr, "usage: %s [--verify|--benchmark|--benchmark-verify]\n", argv[0]);
    return 2;
}
