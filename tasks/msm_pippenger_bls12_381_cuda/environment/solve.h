#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// ─── BLS12-381 G1 MSM ────────────────────────────────────────────────────
//
// Compute the multi-scalar multiplication
//
//      Q = sum_{i=0..N-1} s_i * P_i
//
// over the BLS12-381 G1 elliptic curve, where P_i are affine G1 points and
// s_i are 256-bit scalars.
//
// Field representation: BLS12-381 base prime p is 381 bits. We carry one
// extra word so each base-field element occupies 6 x uint64 = 384 bits.
// All field elements (input, internal, output) are in **Montgomery form**:
//
//      a_montgomery = a * R   mod p,   with R = 2^384 mod p.
//
// The Montgomery constants (p, p_inv = -p^{-1} mod 2^64, R, R^2, etc.) are
// provided as device constants in solve.cu via the BLS12_381_* macros below.
//
// Curve equation: y^2 = x^3 + 4 over F_p, with the canonical generator G.
//
// Scalars: 256-bit little-endian, stored as 4 x uint64. Only the low 255
// bits are guaranteed to be in [0, r), the BLS12-381 scalar-field order;
// the implementation must accept the full 256-bit range and just do
// repeated additions / doublings over the curve. Reduction to F_r is NOT
// required.
//
// Point at infinity: encoded as the affine pair where BOTH x and y are
// the field-zero element (i.e. 6 x uint64 of zero, in Montgomery form).
// This is not a valid affine curve point and is reserved as the zero
// flag. The output Q is also written in this form when the MSM result
// is the identity.
//
// ─────────────────────────────────────────────────────────────────────

// 6 x uint64 = 384 bits per coordinate.
static constexpr int FP_LIMBS = 6;

// Affine G1 point: (x, y) in Montgomery form. 12 limbs total.
struct G1Affine {
    uint64_t x[FP_LIMBS];
    uint64_t y[FP_LIMBS];
};

// 256-bit scalar, 4 x uint64, little-endian.
struct Scalar256 {
    uint64_t limbs[4];
};

// MAX_N is the largest input size the harness will request, used by
// the implementation for any precomputed tables sized in N.
static constexpr int MSM_MAX_N = 1 << 20;

// Compute Q = sum_i s_i * P_i and write it to *out_q in Montgomery affine
// form. All pointers (points, scalars, out_q, workspace) are device
// pointers. workspace is a pre-allocated scratch region of workspace_bytes
// bytes that the implementation may use for any precomputed tables,
// buckets, or temporaries. The harness will synchronize the stream before
// reading out_q.
//
// On entry:
//   points  : [N] G1Affine, Montgomery form, on device
//   scalars : [N] Scalar256, on device
//   out_q   : pointer to one G1Affine on device (output)
//   workspace, workspace_bytes : scratch (device)
//   N       : number of (point, scalar) pairs, 1 <= N <= MSM_MAX_N
//   stream  : CUDA stream
void msm_bls12_381_g1(const G1Affine* points,
                      const Scalar256* scalars,
                      G1Affine* out_q,
                      void* workspace,
                      size_t workspace_bytes,
                      int N,
                      cudaStream_t stream);
