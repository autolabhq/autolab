#include "solve.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

// ─── BLS12-381 base-field constants (device, Montgomery, 384-bit) ────────
__device__ __constant__ uint64_t D_FP_P[6] = {
    0xb9feffffffffaaabULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
};
__device__ __constant__ uint64_t D_FP_P_INV = 0x89f3fffcfffcfffdULL;
__device__ __constant__ uint64_t D_FP_ONE_M[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL, 0x5f48985753c758baULL,
    0x77ce585370525745ULL, 0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL
};

// ─── 384-bit Montgomery field arithmetic (device) ────────────────────────
// All routines operate on 6-limb little-endian uint64 arrays in Montgomery
// form. They use schoolbook 6x6 multiplication followed by CIOS Montgomery
// reduction.

__device__ __forceinline__ bool fp_is_zero_d(const uint64_t* a) {
    uint64_t x = 0;
    for (int i = 0; i < 6; ++i) x |= a[i];
    return x == 0;
}

__device__ __forceinline__ bool fp_eq_d(const uint64_t* a, const uint64_t* b) {
    for (int i = 0; i < 6; ++i) if (a[i] != b[i]) return false;
    return true;
}

__device__ __forceinline__ void fp_copy_d(uint64_t* r, const uint64_t* a) {
    for (int i = 0; i < 6; ++i) r[i] = a[i];
}

__device__ __forceinline__ void fp_set_zero_d(uint64_t* r) {
    for (int i = 0; i < 6; ++i) r[i] = 0;
}

// Returns true if a >= p (treating a as 384-bit unsigned, big-endian compare).
__device__ __forceinline__ bool fp_ge_p_d(const uint64_t* a) {
    for (int i = 5; i >= 0; --i) {
        if (a[i] > D_FP_P[i]) return true;
        if (a[i] < D_FP_P[i]) return false;
    }
    return true;
}

// r = a + b mod p
__device__ __forceinline__ void fp_add_d(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t t[6];
    unsigned long long carry = 0;
    for (int i = 0; i < 6; ++i) {
        unsigned long long s = a[i] + carry;
        unsigned long long c1 = (s < a[i]) ? 1ULL : 0ULL;
        unsigned long long s2 = s + b[i];
        unsigned long long c2 = (s2 < s) ? 1ULL : 0ULL;
        t[i] = s2;
        carry = c1 + c2;
    }
    bool need_sub = (carry != 0) || fp_ge_p_d(t);
    if (need_sub) {
        unsigned long long borrow = 0;
        for (int i = 0; i < 6; ++i) {
            unsigned long long s = t[i] - D_FP_P[i];
            unsigned long long b1 = (t[i] < D_FP_P[i]) ? 1ULL : 0ULL;
            unsigned long long s2 = s - borrow;
            unsigned long long b2 = (s < borrow) ? 1ULL : 0ULL;
            r[i] = s2;
            borrow = b1 + b2;
        }
    } else {
        fp_copy_d(r, t);
    }
}

// r = a - b mod p
__device__ __forceinline__ void fp_sub_d(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t t[6];
    unsigned long long borrow = 0;
    for (int i = 0; i < 6; ++i) {
        unsigned long long s = a[i] - b[i];
        unsigned long long b1 = (a[i] < b[i]) ? 1ULL : 0ULL;
        unsigned long long s2 = s - borrow;
        unsigned long long b2 = (s < borrow) ? 1ULL : 0ULL;
        t[i] = s2;
        borrow = b1 + b2;
    }
    if (borrow) {
        unsigned long long carry = 0;
        for (int i = 0; i < 6; ++i) {
            unsigned long long s = t[i] + D_FP_P[i];
            unsigned long long c1 = (s < t[i]) ? 1ULL : 0ULL;
            unsigned long long s2 = s + carry;
            unsigned long long c2 = (s2 < s) ? 1ULL : 0ULL;
            r[i] = s2;
            carry = c1 + c2;
        }
    } else {
        fp_copy_d(r, t);
    }
}

// CIOS Montgomery multiplication: r = a * b * R^{-1} mod p.
// Schoolbook 6x6 with interleaved CIOS reduction; inputs and output in [0, p).
__device__ void fp_mul_d(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    unsigned long long t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 6; ++i) {
        // t += a * b[i]
        unsigned long long carry = 0;
        for (int j = 0; j < 6; ++j) {
            unsigned __int128 s = (unsigned __int128)a[j] * b[i] + t[j] + carry;
            t[j] = (unsigned long long)s;
            carry = (unsigned long long)(s >> 64);
        }
        unsigned __int128 s6 = (unsigned __int128)t[6] + carry;
        t[6] = (unsigned long long)s6;
        t[7] += (unsigned long long)(s6 >> 64);

        // m = t[0] * p_inv mod 2^64
        unsigned long long m = t[0] * D_FP_P_INV;

        // t += m * p
        carry = 0;
        for (int j = 0; j < 6; ++j) {
            unsigned __int128 s2 = (unsigned __int128)m * D_FP_P[j] + t[j] + carry;
            t[j] = (unsigned long long)s2;
            carry = (unsigned long long)(s2 >> 64);
        }
        unsigned __int128 s3 = (unsigned __int128)t[6] + carry;
        t[6] = (unsigned long long)s3;
        t[7] += (unsigned long long)(s3 >> 64);

        // shift right by one limb
        for (int j = 0; j < 7; ++j) t[j] = t[j+1];
        t[7] = 0;
    }
    uint64_t out[6];
    for (int i = 0; i < 6; ++i) out[i] = t[i];
    if (t[6] || fp_ge_p_d(out)) {
        unsigned long long borrow = 0;
        for (int i = 0; i < 6; ++i) {
            unsigned long long s = out[i] - D_FP_P[i];
            unsigned long long b1 = (out[i] < D_FP_P[i]) ? 1ULL : 0ULL;
            unsigned long long s2 = s - borrow;
            unsigned long long b2 = (s < borrow) ? 1ULL : 0ULL;
            r[i] = s2;
            borrow = b1 + b2;
        }
    } else {
        fp_copy_d(r, out);
    }
}

__device__ __forceinline__ void fp_sqr_d(uint64_t* r, const uint64_t* a) {
    fp_mul_d(r, a, a);
}

// fp_inv via Fermat: a^{p-2}.
__device__ void fp_inv_d(uint64_t* r, const uint64_t* a) {
    // p - 2 limbs.
    const uint64_t pm2[6] = {
        0xb9feffffffffaaa9ULL, 0x1eabfffeb153ffffULL, 0x6730d2a0f6b0f624ULL,
        0x64774b84f38512bfULL, 0x4b1ba7b6434bacd7ULL, 0x1a0111ea397fe69aULL
    };
    uint64_t result[6];
    fp_copy_d(result, D_FP_ONE_M);
    uint64_t base[6]; fp_copy_d(base, a);
    for (int word = 0; word < 6; ++word) {
        for (int bit = 0; bit < 64; ++bit) {
            if ((pm2[word] >> bit) & 1ULL) fp_mul_d(result, result, base);
            fp_sqr_d(base, base);
        }
    }
    fp_copy_d(r, result);
}

// ─── G1 Jacobian point arithmetic (device) ───────────────────────────────
// Identity is Z=0.

struct JacD {
    uint64_t X[6], Y[6], Z[6];
};

__device__ __forceinline__ bool jac_is_zero_d(const JacD& p) {
    return fp_is_zero_d(p.Z);
}

__device__ __forceinline__ void jac_set_zero_d(JacD& p) {
    fp_copy_d(p.X, D_FP_ONE_M);
    fp_copy_d(p.Y, D_FP_ONE_M);
    fp_set_zero_d(p.Z);
}

__device__ void jac_double_d(JacD& r, const JacD& p) {
    if (jac_is_zero_d(p) || fp_is_zero_d(p.Y)) { jac_set_zero_d(r); return; }
    uint64_t A[6], B[6], C[6], D[6], E[6], F[6], tmp[6];
    uint64_t newX[6], newY[6], newZ[6];
    fp_sqr_d(A, p.X);
    fp_sqr_d(B, p.Y);
    fp_sqr_d(C, B);
    fp_add_d(D, p.X, B);
    fp_sqr_d(D, D);
    fp_sub_d(D, D, A);
    fp_sub_d(D, D, C);
    fp_add_d(D, D, D);
    fp_add_d(E, A, A);
    fp_add_d(E, E, A);
    fp_sqr_d(F, E);
    fp_sub_d(tmp, F, D);
    fp_sub_d(newX, tmp, D);
    fp_sub_d(tmp, D, newX);
    fp_mul_d(tmp, E, tmp);
    uint64_t eight_c[6];
    fp_add_d(eight_c, C, C);
    fp_add_d(eight_c, eight_c, eight_c);
    fp_add_d(eight_c, eight_c, eight_c);
    fp_sub_d(newY, tmp, eight_c);
    fp_mul_d(tmp, p.Y, p.Z);
    fp_add_d(newZ, tmp, tmp);
    fp_copy_d(r.X, newX); fp_copy_d(r.Y, newY); fp_copy_d(r.Z, newZ);
}

// Mixed Jacobian + Affine. q is affine (Z=1). q affine zero (x=0,y=0) -> identity.
__device__ void jac_add_affine_d(JacD& r, const JacD& p, const uint64_t* qx, const uint64_t* qy) {
    bool q_is_zero = fp_is_zero_d(qx) && fp_is_zero_d(qy);
    if (q_is_zero) { r = p; return; }
    if (jac_is_zero_d(p)) {
        fp_copy_d(r.X, qx); fp_copy_d(r.Y, qy); fp_copy_d(r.Z, D_FP_ONE_M);
        return;
    }
    uint64_t Z1Z1[6], U2[6], S2[6], H[6], HH[6], I[6], J[6], rr[6], V[6], tmp[6];
    fp_sqr_d(Z1Z1, p.Z);
    fp_mul_d(U2, qx, Z1Z1);
    fp_mul_d(tmp, qy, p.Z);
    fp_mul_d(S2, tmp, Z1Z1);
    if (fp_eq_d(U2, p.X)) {
        if (fp_eq_d(S2, p.Y)) {
            JacD qj;
            fp_copy_d(qj.X, qx); fp_copy_d(qj.Y, qy); fp_copy_d(qj.Z, D_FP_ONE_M);
            jac_double_d(r, qj);
            return;
        }
        jac_set_zero_d(r);
        return;
    }
    fp_sub_d(H, U2, p.X);
    fp_sqr_d(HH, H);
    fp_add_d(I, HH, HH); fp_add_d(I, I, I);
    fp_mul_d(J, H, I);
    fp_sub_d(rr, S2, p.Y); fp_add_d(rr, rr, rr);
    fp_mul_d(V, p.X, I);
    uint64_t newX[6], newY[6], newZ[6];
    fp_sqr_d(tmp, rr);
    fp_sub_d(tmp, tmp, J);
    fp_sub_d(tmp, tmp, V);
    fp_sub_d(newX, tmp, V);
    fp_sub_d(tmp, V, newX);
    fp_mul_d(tmp, rr, tmp);
    uint64_t y1j[6]; fp_mul_d(y1j, p.Y, J); fp_add_d(y1j, y1j, y1j);
    fp_sub_d(newY, tmp, y1j);
    fp_add_d(tmp, p.Z, H);
    fp_sqr_d(tmp, tmp);
    fp_sub_d(tmp, tmp, Z1Z1);
    fp_sub_d(newZ, tmp, HH);
    fp_copy_d(r.X, newX); fp_copy_d(r.Y, newY); fp_copy_d(r.Z, newZ);
}

// Full Jacobian + Jacobian addition.
__device__ void jac_add_jac_d(JacD& r, const JacD& p, const JacD& q) {
    if (jac_is_zero_d(p)) { r = q; return; }
    if (jac_is_zero_d(q)) { r = p; return; }
    uint64_t Z1Z1[6], Z2Z2[6], U1[6], U2[6], S1[6], S2[6];
    uint64_t H[6], I[6], J[6], rr[6], V[6], tmp[6];
    fp_sqr_d(Z1Z1, p.Z);
    fp_sqr_d(Z2Z2, q.Z);
    fp_mul_d(U1, p.X, Z2Z2);
    fp_mul_d(U2, q.X, Z1Z1);
    fp_mul_d(tmp, p.Y, q.Z); fp_mul_d(S1, tmp, Z2Z2);
    fp_mul_d(tmp, q.Y, p.Z); fp_mul_d(S2, tmp, Z1Z1);
    if (fp_eq_d(U1, U2)) {
        if (fp_eq_d(S1, S2)) { jac_double_d(r, p); return; }
        jac_set_zero_d(r); return;
    }
    fp_sub_d(H, U2, U1);
    fp_add_d(tmp, H, H); fp_sqr_d(I, tmp);
    fp_mul_d(J, H, I);
    fp_sub_d(rr, S2, S1); fp_add_d(rr, rr, rr);
    fp_mul_d(V, U1, I);
    uint64_t newX[6], newY[6], newZ[6];
    fp_sqr_d(tmp, rr); fp_sub_d(tmp, tmp, J);
    fp_sub_d(tmp, tmp, V); fp_sub_d(newX, tmp, V);
    fp_sub_d(tmp, V, newX); fp_mul_d(tmp, rr, tmp);
    uint64_t s1j[6]; fp_mul_d(s1j, S1, J); fp_add_d(s1j, s1j, s1j);
    fp_sub_d(newY, tmp, s1j);
    fp_add_d(tmp, p.Z, q.Z); fp_sqr_d(tmp, tmp);
    fp_sub_d(tmp, tmp, Z1Z1); fp_sub_d(tmp, tmp, Z2Z2);
    fp_mul_d(newZ, tmp, H);
    fp_copy_d(r.X, newX); fp_copy_d(r.Y, newY); fp_copy_d(r.Z, newZ);
}

__device__ void jac_to_affine_d(uint64_t* out_x, uint64_t* out_y, const JacD& p) {
    if (jac_is_zero_d(p)) {
        fp_set_zero_d(out_x); fp_set_zero_d(out_y); return;
    }
    uint64_t z_inv[6], z_inv2[6], z_inv3[6];
    fp_inv_d(z_inv, p.Z);
    fp_sqr_d(z_inv2, z_inv);
    fp_mul_d(z_inv3, z_inv2, z_inv);
    fp_mul_d(out_x, p.X, z_inv2);
    fp_mul_d(out_y, p.Y, z_inv3);
}

// ─── One group per (point, scalar), double-and-add; tree reduction. ──────
//
// Each group walks the 256-bit scalar MSB-first, doing one Jacobian double
// per bit and a mixed-add when the bit is set. Only lane 0 of the group
// performs the scalar multiplication. The N partial products are then
// accumulated by a pairwise tree reduction.

constexpr int BLOCK_THREADS = 128;
constexpr int LANES_PER_POINT = 4;    // group size; only lane 0 computes

// Phase 1: per-thread partial product (in Jacobian) -> stored as JacD per i.
__global__ void naive_scalar_mul_kernel(const G1Affine* __restrict__ points,
                                        const Scalar256* __restrict__ scalars,
                                        JacD* __restrict__ partials,
                                        int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid / LANES_PER_POINT;
    int lane = gid - i * LANES_PER_POINT;
    if (i >= N) return;
    if (lane != 0) return;   // only lane 0 does the scalar mul

    const G1Affine& P = points[i];
    const Scalar256& s = scalars[i];

    JacD acc; jac_set_zero_d(acc);
    bool started = false;

    // MSB-first walk over 256 bits.
    for (int word = 3; word >= 0; --word) {
        uint64_t k = s.limbs[word];
        for (int bit = 63; bit >= 0; --bit) {
            if (started) jac_double_d(acc, acc);
            uint64_t b = (k >> bit) & 1ULL;
            if (b) {
                if (!started) {
                    // First set bit: acc = P (handle affine zero).
                    bool pz = fp_is_zero_d(P.x) && fp_is_zero_d(P.y);
                    if (pz) {
                        jac_set_zero_d(acc);
                    } else {
                        fp_copy_d(acc.X, P.x);
                        fp_copy_d(acc.Y, P.y);
                        fp_copy_d(acc.Z, D_FP_ONE_M);
                    }
                    started = true;
                } else {
                    jac_add_affine_d(acc, acc, P.x, P.y);
                }
            }
        }
    }

    partials[i] = acc;
}

// Phase 2: tree-reduce the N partials in-place by halving. Each launch
// reduces by a factor of 2: thread i computes partials[i] += partials[i + half].
// Repeated launches until one partial remains.
__global__ void reduce_pair_kernel(JacD* __restrict__ partials, int half) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    JacD a = partials[i];
    JacD b = partials[i + half];
    JacD c;
    jac_add_jac_d(c, a, b);
    partials[i] = c;
}

// Phase 3: convert the single Jacobian sum to affine and write to out_q.
__global__ void finalize_kernel(JacD* __restrict__ partials, G1Affine* __restrict__ out_q) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        JacD q = partials[0];
        jac_to_affine_d(out_q->x, out_q->y, q);
    }
}

// Tiny on-device copy kernel: copies one JacD from src to dst (used to
// thread the odd-leftover element back into the reduction array without
// touching the host or the runtime API).
__global__ void copy_one_jac_kernel(JacD* __restrict__ partials, int dst, int src) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        partials[dst] = partials[src];
    }
}

// Tiny on-device kernel that writes the affine zero (identity) to out_q.
__global__ void zero_out_q_kernel(G1Affine* __restrict__ out_q) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 6; ++i) { out_q->x[i] = 0; out_q->y[i] = 0; }
    }
}

}  // anonymous namespace

void msm_bls12_381_g1(const G1Affine* points,
                      const Scalar256* scalars,
                      G1Affine* out_q,
                      void* workspace,
                      size_t workspace_bytes,
                      int N,
                      cudaStream_t stream)
{
    if (!points || !scalars || !out_q || !workspace || N <= 0) {
        zero_out_q_kernel<<<1, 1, 0, stream>>>(out_q);
        return;
    }

    // Use the workspace for the partial-products buffer.
    size_t need = (size_t)N * sizeof(JacD);
    if (need > workspace_bytes) {
        zero_out_q_kernel<<<1, 1, 0, stream>>>(out_q);
        return;
    }
    JacD* partials = static_cast<JacD*>(workspace);

    // Launch one group per point; only lane 0 of each group computes.
    int total_threads = N * LANES_PER_POINT;
    int grid = (total_threads + BLOCK_THREADS - 1) / BLOCK_THREADS;
    naive_scalar_mul_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(points, scalars, partials, N);

    // Reduce N partials by repeated pair-halving. Odd N: shuffle the leftover
    // up via a 1-thread copy kernel (no runtime memcpy, by design).
    int n = N;
    while (n > 1) {
        int half = n >> 1;
        if (n & 1) {
            int g = (half + BLOCK_THREADS - 1) / BLOCK_THREADS;
            reduce_pair_kernel<<<g, BLOCK_THREADS, 0, stream>>>(partials, half);
            copy_one_jac_kernel<<<1, 1, 0, stream>>>(partials, half, 2*half);
            n = half + 1;
        } else {
            int g = (half + BLOCK_THREADS - 1) / BLOCK_THREADS;
            reduce_pair_kernel<<<g, BLOCK_THREADS, 0, stream>>>(partials, half);
            n = half;
        }
    }

    finalize_kernel<<<1, 1, 0, stream>>>(partials, out_q);
}
