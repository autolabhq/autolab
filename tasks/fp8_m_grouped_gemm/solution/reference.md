# Reference Solution Notes

## Approach

The reference Triton implementation (`triton/020_m_grouped_fp8_gemm_nt_contiguous.py` in kernel-anno-2)
achieves near-deepgemm performance through the following techniques:

### Key Optimizations

1. **Fused dequant + GEMM in a single kernel**
   Rather than materializing full FP32 A and B tensors, the kernel loads FP8 tiles directly
   and applies scaling factors inline in the inner loop.

2. **Tile-based group dispatch**
   Precompute per-tile metadata: `(group_id, row_start, valid_rows)` so each threadblock
   knows exactly which rows to process without conditional branches in the hot path.

3. **FP8 tl.dot with native hardware support**
   On H800/H100 (SM90), use `tl.dot` with `fp8e4nv` dtype directly — avoids the upcast
   path and achieves near-peak tensor core utilization.

4. **Double-buffered pipeline**
   `num_stages=3` or `num_stages=4` with software pipelining hides memory latency.

5. **Autotune over {MICRO_K, num_warps, num_stages}**
   Sweep at least: `MICRO_K ∈ {32, 64}`, `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3, 4}`.

### Block Sizes

- `BLOCK_M = 128`, `BLOCK_N = 128`, `BLOCK_K = 128` as the starting point.
- The scaling factor granularity is 128 in both K (per-token A) and N/K (per-block B),
  so tile sizes must be multiples of 128 for clean scale indexing.

### Padding Handling

Rows where `m_indices == -1` are padding. The safest approach is to compute tile metadata
excluding these rows and zero the output slice at the end (one masked store).

## Performance Expectations (H800)

| Implementation | Avg TFLOPs (5 cases) |
|---|---|
| torch_ref      | ~2–8 TFLOPs  |
| Triton (this)  | ~150–300 TFLOPs |
| DeepGEMM       | ~300–500 TFLOPs |

Target: agent_tflops / deepgemm_tflops >= 0.80 → reward = 1.0
