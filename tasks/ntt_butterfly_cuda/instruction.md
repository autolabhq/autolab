# NTT Butterfly CUDA

`ntt_forward_cuda()` applies an in-place forward Number Theoretic Transform over the Goldilocks prime field. Optimize the implementation in `/app/solve.cu` to run as fast as possible on a single NVIDIA GPU.

```cpp
void ntt_forward_cuda(uint64_t* data,
                      void* workspace, size_t workspace_bytes,
                      int batch, int n,
                      cudaStream_t stream);
```

Apply the forward NTT in place to each row of a batched `uint64_t` array stored in row-major device memory. The prime field is `F_p` with `p = 2^64 - 2^32 + 1` (Goldilocks); input and output values are in `[0, p)`. The reference output is in **natural index order** (no bit-reversal). `data` and `workspace` are device pointers; `n` is a power of two; `workspace` is a generous pre-allocated scratch region you may use freely. The stream is synchronized before the output is read.

## Setup

| Item | Path / Value |
|------|--------------|
| Editable | `/app/solve.cu` |
| Function signature and constants | `/app/solve.h` (read-only) |
| Harness | `/app/main.cu` (read-only) |
| Build | `/app/Makefile` (read-only) |
| GPU | Single NVIDIA GPU |
| Local build / run | `make -C /app && /app/<binary>` |

## Your Goal

Minimize the median wall-clock runtime of `ntt_forward_cuda()` on the timed workload. Output values must equal the reference forward NTT, bit-exact, for every element. Wrong answers score 0.

## Evaluation

The evaluation times `ntt_forward_cuda()` over several trials via CUDA events after a warmup and reports the median. Lower is better.

## Rules

- Edit `/app/solve.cu` only. You may create scratch files while working, but when verification runs, `/app` must contain only the original files plus your edits to `solve.cu`. Any extra file under `/app` causes the run to score 0.
- CUDA C++ and the CUDA runtime / math headers are available. Inline PTX is allowed.
- Do not use Thrust, CUB, CUTLASS, cuBLAS, cuDNN, cuFFT, cuRAND, NCCL, Torch, or any third-party NTT / FFT / large-integer library.
- Inside `ntt_forward_cuda()`, no `cudaMemcpy*` (including device-to-device), no `cudaMalloc*`/`cudaFree*`/`cudaHostAlloc`/`cudaHostRegister`, no `malloc`/`calloc`/`free`, no `new`/`delete`. Use the supplied workspace for scratch and a custom kernel for any in-device copy.
- Do not read environment variables, command-line arguments, `/proc`, `/tmp`, the verifier path, or any file from `solve.cu`. All inputs are passed through the function arguments.
- Do not emit `__VERIFIER_*` sentinels from `solve.cu`.
- Single GPU only. No internet access.
- Time budget: 2 hours.
