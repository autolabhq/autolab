# BLS12-381 G1 MSM CUDA — Reference

## Background

Multi-scalar multiplication on BLS12-381 G1,

    Q = sum_{i=0..N-1} s_i * P_i,

is the dominant kernel of every modern zkSNARK prover and zkRollup. The
base field `F_p` is 381 bits (carried as 6 x uint64) and every coordinate
operation is a 384-bit Montgomery multiply.

## Baseline

`environment/solve.cu` is the textbook GPU-naive MSM: one warp per
`(P_i, s_i)` pair with only lane 0 doing work, walking 256 bits MSB-first
with one Jacobian double per bit and a mixed Jacobian+affine add when the
bit is set, then `log2 N` halving kernels to reduce the partials. Three
of every four lanes idle and the per-thread serial chain dominates.

## Reference Oracle

The shipped reference is adapted from an agent run that beat the prior
oracle by ~6x. It is a Pippenger bucket method with `c = 8`, organized
into four kernels:

1. `chunk_accumulate_kernel` — split the N points into chunks of
   `CHUNK_SIZE = 512`. Launch `num_chunks * 32` threads; thread
   `(chunk_id, window_idx)` sweeps its 512 points, extracts the c=8
   window value, and accumulates the point into its private bucket
   array of 255 Jacobian slots in global memory via mixed
   Jacobian+affine add. No atomics required — each thread owns its
   `(chunk_id, window_idx)` slice.

2. `tree_reduce_kernel` — pairwise merge of the per-chunk bucket
   arrays over `log2(num_chunks)` launches. Each launch processes
   `num_chunks/(2*stride) * 32 * 255` independent (window, bucket)
   merges in parallel.

3. `running_sum_kernel` — one thread per window walks buckets from
   254 down to 0 with the standard Pippenger running-sum recurrence
   `(running, total)`, computing `W_w = sum_v v * bucket[v]` using
   only point adds.

4. `combine_windows_kernel` — single thread aggregates the 32 window
   sums MSB-first: 8 Jacobian doubles + 1 Jacobian-Jacobian add per
   window step. One Fermat field inversion converts the Jacobian
   result to affine.

Field arithmetic is plain CIOS Montgomery in `__int128` codegen. There
is no inline PTX, no batch inversion, no radix-sort step.

## Why It Wins

- The N points are spread across thousands of independent
  `(chunk, window)` accumulators instead of being serialized inside a
  single warp lane.
- Per-scalar work drops from ~384 Jacobian-equivalent ops to ~32 mixed
  adds; the `O(2^c) = 255` bucket reductions are amortized across all
  N scalars in that window.
- The tree-merge phase is fully parallel over `(window, bucket)` and
  costs `log2(num_chunks)` lightweight launches.

## Possible Further Wins (Not Taken)

- Inline-PTX `mad.wide.u64` Montgomery multiply.
- Affine-batched bucket fills with Montgomery's batch inversion.
- Sort-by-bucket radix to remove the per-chunk sequential bucket scan.
- Larger window size with signed-digit recoding.

The current implementation is already a single-digit-multiplier short
of theoretical peak on H100 and is intentionally the calibration target,
not the ceiling.
