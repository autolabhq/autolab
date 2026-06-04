# BLS12-381 G1 Multi-Scalar Multiplication CUDA

`msm_bls12_381_g1()` computes a multi-scalar multiplication on the BLS12-381 G1 elliptic curve, the dominant kernel of zkSNARK provers (Groth16, PLONK, Halo2). Optimize the implementation in `/app/solve.cu` to run as fast as possible on a single NVIDIA GPU.

Compute `Q = sum_{i=0..N-1} s_i * P_i` over BLS12-381 G1. Each `P_i` is an affine G1 point with coordinates in Montgomery form over the BLS12-381 base field `F_p`; the prime `p` is 381 bits and each coordinate is stored as 6 × uint64 (384 bits). Each `s_i` is a plain (non-Montgomery) 256-bit little-endian scalar (4 × uint64).

```cpp
void msm_bls12_381_g1(const G1Affine* points,
                      const Scalar256* scalars,
                      G1Affine* out_q,
                      void* workspace,
                      size_t workspace_bytes,
                      int N,
                      cudaStream_t stream);
```

`points`, `scalars`, `out_q`, and `workspace` are device pointers; the stream is synchronized before `out_q` is read. All G1 coordinates (input, internal, output) are in **Montgomery form** with `R = 2^384 mod p`. The identity (point at infinity) is encoded as `(x, y) = (0, 0)`; write `Q` in this same form when the result is the identity. The shipped baseline in `/app/solve.cu` is a textbook GPU-naive double-and-add MSM (one thread per `(P_i, s_i)` pair).

## Setup

| Item | Path / Value |
|------|--------------|
| Editable | `/app/solve.cu` |
| Function signature, data structures, constants | `/app/solve.h` (read-only) |
| Harness | `/app/main.cu` (read-only) |
| Build | `/app/Makefile` (read-only) |
| GPU | Single NVIDIA GPU |
| Local build / run | `make -C /app && /app/<binary>` |

## Your Goal

Minimize the median wall-clock runtime of `msm_bls12_381_g1()` on the timed workload. Output coordinates must equal the reference Q bit-exactly (in Montgomery form, including the affine-zero encoding of the identity). Wrong answers score 0.

## Evaluation

The evaluation times `msm_bls12_381_g1()` over several trials via CUDA events after a warmup and reports the median. Lower is better.

## Rules

- Edit `/app/solve.cu` only. You may create scratch files while working, but when verification runs, `/app` must contain only the original files plus your edits to `solve.cu`. Any extra file under `/app` causes the run to score 0.
- CUDA C++ and the CUDA runtime / math headers are available. Inline PTX is allowed.
- Do not use Thrust, CUB, CUTLASS, cuBLAS, cuDNN, cuFFT, cuRAND, NCCL, Torch, or any third-party finite-field / elliptic-curve / zkSNARK library (e.g. blst, arkworks, libff, mcl, icicle, GMP, MPFR, NTL).
- Inside `msm_bls12_381_g1()`, no `cudaMemcpy*` (including device-to-device), no `cudaMalloc*`/`cudaFree*`/`cudaHostAlloc`/`cudaHostRegister`, no `malloc`/`calloc`/`free`, no `new`/`delete`. Use the supplied workspace for scratch and a custom kernel for any in-device copy.
- Do not read environment variables, command-line arguments, `/proc`, `/tmp`, or any file from `solve.cu`. All inputs are passed through the function arguments.
- Do not emit `__VERIFIER_*` sentinels from `solve.cu`.
- Single GPU only. No internet access.
- Time budget: 2 hours.
