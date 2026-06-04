#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Goldilocks prime: p = 2^64 - 2^32 + 1.
// Used by Plonky2 / Plonky3. Has 2-adicity 32, so any power-of-two length up to 2^32 is supported.
static constexpr uint64_t NTT_PRIME = 0xFFFFFFFF00000001ULL;

// Maximum supported transform length, in elements (per row). The harness drives the kernel with
// power-of-two lengths; the agent's implementation must support every length in [4, NTT_MAX_LEN].
static constexpr int NTT_MAX_LEN = 1 << 16;

// Apply the forward Number Theoretic Transform, in place, to each of the `batch` rows of length
// `n`. Rows are stored consecutively in row-major device memory: `data[row * n + i]`. Each input
// element is in [0, NTT_PRIME); each output element must also be in [0, NTT_PRIME). The transform
// is the standard forward NTT (no bit-reversal output permutation -- the harness compares against a
// matching natural-order forward-NTT reference).
//
// `workspace` is a pre-allocated device scratch region of `workspace_bytes` bytes that the
// implementation may use for any precomputed tables, twiddles, or temporaries. `stream` may be
// used; the harness will synchronize before reading the output.
void ntt_forward_cuda(uint64_t* data,
                      void* workspace,
                      size_t workspace_bytes,
                      int batch,
                      int n,
                      cudaStream_t stream);
