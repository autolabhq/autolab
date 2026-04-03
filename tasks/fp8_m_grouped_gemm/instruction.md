# M-Grouped FP8 GEMM (NT, Contiguous Layout) — Triton Optimization

## Background

In modern LLM inference, Mixture-of-Experts (MoE) models route each token to one of several expert groups. The core compute primitive is a **grouped GEMM**: tokens belonging to the same expert group are batched and multiplied against that expert's weight matrix.

This task focuses on the **contiguous layout** variant: tokens are laid out sequentially in memory, and a `m_indices` tensor records which expert group each row belongs to. Weights are FP8-quantized for memory efficiency, with mixed-precision scaling factors.

## Your Task

The file `/app/solve.py` contains a **pure-PyTorch reference implementation** of this operation. It is correct but slow — it dequantizes to FP32 on the CPU path, loops over expert groups, and calls `torch.matmul`.

**Rewrite the `Model.forward()` method in `/app/solve.py` using Triton** to make it as fast as possible.

You may only edit `/app/solve.py`. All other files are read-only.

## Operation Specification

```
IN:
  a : (Tensor[m, k, fp8_e4m3fn],  Tensor[m, ceil(k/128), f32])   # data, per-token scale
  b : (Tensor[G, n, k, fp8_e4m3fn], Tensor[G, ceil(n/128), ceil(k/128), f32])  # data, per-block scale
  m_indices : Tensor[m, i32]   # group id per row; -1 = padding row

OUT:
  d : Tensor[m, n, bf16]       # d[i,:] = 0 if m_indices[i]==-1
```

**Dequantization semantics:**
- **A** uses *per-token* scaling: `a_fp32[i, j] = a_fp8[i, j] * a_sf[i, j//128]`
- **B** uses *per-block* (128×128) scaling: `b_fp32[g, i, j] = b_fp8[g, i, j] * b_sf[g, i//128, j//128]`
- For each group `g`, compute: `d[rows_g, :] = (a_fp32[rows_g, :] @ b_fp32[g, :, :].T).to(bf16)`
- Padding rows (`m_indices == -1`) must be zeroed in output

## Benchmark Configurations

Five test cases are defined in `test_cases.py`:

| num_groups | expected_m_per_group | n    | k    |
|:----------:|:--------------------:|:----:|:----:|
| 8          | 4096                 | 4096 | 4096 |
| 8          | 6144                 | 4096 | 4096 |
| 8          | 5120                 | 4096 | 4096 |
| 4          | 8192                 | 4096 | 4096 |
| 8          | 7168                 | 4096 | 4096 |

## Scoring

The reward is based on **average TFLOPS across all 5 test cases**:

```
reward = clamp(agent_tflops / (deepgemm_tflops * 0.80), 0.0, 1.0)
```

- `reward = 0.0` → your implementation is as slow as (or slower than) the PyTorch reference
- `reward = 1.0` → your Triton kernel reaches **≥ 80% of DeepGEMM** throughput

Partial credit is awarded continuously — every TFLOP gained counts.

## Correctness Gate

Before benchmarking, your implementation is checked with:
```
torch.allclose(agent_output, torch_ref_output, rtol=3e-2, atol=3e-2)
```
If this fails on any test case, `reward = 0.0`.

## Constraints

- **Only `/app/solve.py` may be modified.**
- You must keep the `Model` class interface: `__init__(m, num_groups, n, k, expected_m_per_group)` and `forward(a, b, m_indices)`.
- No external libraries beyond `torch`, `triton`, and the standard library.
- Do not import `deep_gemm` in your solution.

## Hints

- Fuse dequantization into the GEMM kernel — materializing FP32 tensors wastes bandwidth.
- The `m_indices` tensor can have arbitrary row orderings within each group; you may want to precompute tile metadata.
- Block sizes of 128×128 or 128×256 in (M, N) often work well; sweep `BLOCK_K` and pipeline stages.
- On H800, Triton's `tl.dot` can use FP8 inputs directly — check `tl.float8e4nv`.
- `@triton.autotune` with several `num_stages` and `num_warps` configurations helps.
- Watch out for groups with `m_indices == -1` (padding rows): they should not contribute to any group's GEMM.
