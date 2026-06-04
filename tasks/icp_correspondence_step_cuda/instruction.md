# ICP Correspondence Step CUDA

Optimize `icp_correspondence()` in `/app/solve.cu` to run one Iterative-Closest-Point correspondence step as fast as possible on a single NVIDIA GPU.

```cpp
void icp_correspondence(const float* src,
                        const float* tgt,
                        int N, int M,
                        float d_max,
                        const void* kd_nodes,
                        int n_nodes,
                        float* H_out,         // 9 floats, row-major 3x3
                        float* err_out,       // 1 float
                        int*   count_out,     // 1 int
                        cudaStream_t stream);
```

`src` is a source point cloud `P` of shape `(N, 3)` fp32; `tgt` is a target cloud `T` of shape `(M, 3)` fp32; `kd_nodes` is an opaque spatial index over `T` of `n_nodes` entries (layout in `kdtree.h`), built once on the host before timing and **not** to be rebuilt inside this call. `d_max` is the maximum correspondence distance. All pointers are device memory.

For every source point `p_i`, find its nearest neighbor `q_i` in `T`. Pairs whose squared Euclidean distance exceeds `d_max^2` are invalid and excluded from all accumulations. Over the valid pairs, compute and write: the 3×3 cross-covariance matrix `H = sum_i (p_i - p_centroid) (q_i - q_centroid)^T` (fp64 accumulators cast to fp32 on output), the uncentered sum of squared distances `err = sum_i ||p_i - q_i||^2`, and the valid-pair count.

Bench shape (fixed): `N = 200000`, `M = 500000`, `d_max = 0.05`. Clouds are drawn from a synthetic noisy distribution; only the seed varies. The verifier uses a hidden seed different from the public verify/bench paths, but the shape is the same.

## Setup

| Item | Path / Value |
|------|--------------|
| Editable | `/app/solve.cu` |
| Function signature | `/app/solve.h` (read-only) |
| KD-tree node layout | `/app/kdtree.h` (read-only) |
| Harness | `/app/main.cu` (read-only) |
| Build | `/app/Makefile` (read-only) |
| GPU | Single NVIDIA GPU |
| Local build / run | `make -C /app && /app/<binary>` |

## Your Goal

Minimize the median wall-clock runtime of `icp_correspondence()` on the timed workload. Outputs must match the CPU brute-force reference within `1e-4` relative tolerance, with `count` matching exactly. Wrong answers score 0.

## Evaluation

The evaluation times `icp_correspondence()` over several trials via CUDA events after a warmup and reports the median. Lower is better.

## Rules

- Edit `/app/solve.cu` only. You may create scratch files while working, but when verification runs, `/app` must contain only the original files plus your edits to `solve.cu`. Any extra file under `/app` causes the run to score 0.
- CUDA C++ and CUDA runtime/math headers are allowed.
- Do not use Thrust, CUB, CUTLASS, cuBLAS, cuDNN, cuSPARSE, cuSOLVER, cuRAND, cuTensor, nvJitLink, NCCL, Torch, PCL, FLANN, nanoflann, Open3D, libnabo, or any framework / vendor library that provides k-NN, spatial-index, sort, scan, SVD, or GEMM primitives.
- Do not copy inputs or outputs to the host inside `icp_correspondence()`. The centroid divide must run on the GPU.
- Do not allocate device, managed, pinned, or host memory inside `icp_correspondence()`. State that must persist across calls may be cached in file-scope `static` variables initialized lazily on first use.
- Do not read files, environment variables, `/proc`, command lines, or verifier paths from `solve.cu`.
- Single GPU only. No internet access.
- Time budget: 2 hours.
