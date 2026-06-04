# NTT Butterfly CUDA — Reference

## Background

The Number Theoretic Transform (NTT) is the modular-arithmetic analogue of the FFT and is the dominant inner kernel in zero-knowledge proving systems (Plonky2 / Plonky3, RISC Zero, Halo2) and in lattice-based / fully-homomorphic cryptography (BFV, BGV, CKKS). The task uses the **Goldilocks prime** `p = 2^64 - 2^32 + 1` adopted by Plonky2: a 64-bit Solinas-style prime with 2-adicity 32, which means any power-of-two transform length up to `2^32` admits a primitive root of unity in `F_p`.

## Baseline Approach

The supplied implementation is a textbook iterative Cooley-Tukey radix-2 NTT:

- One bit-reverse permutation kernel.
- One launch per butterfly stage; each stage uses one thread per `(k+j)` butterfly pair.
- Each thread independently recomputes the stage twiddle `omega_m` (full modular exponentiation chain) and then `omega_m^j`.
- Modular multiplication uses `__umul64hi` to get the 128-bit product followed by an integer `%` of the high half (each `%` on `uint64_t` is dozens of cycles on modern GPUs).
- No shared memory, no twiddle precompute, no special-form reduction.

The combination is genuinely slow: the modular multiply dominates, and the per-thread re-derivation of the twiddle per stage adds redundant work proportional to `log n`.

## Possible Optimization Directions

1. **Fast Goldilocks reduction.** `p = 2^64 - 2^32 + 1` admits a divisionless reduction of the 128-bit product. Since `2^64 = 2^32 - 1 (mod p)`, the high 64 bits can be folded into the low 64 bits with two adds and a couple of corrections, replacing the `%` with a handful of integer ops. (Plonky2's `reduce128` is the canonical version.)
2. **Precompute the twiddle table once per launch.** All stages share the same `omega_n^k` table (stage `s` uses every `n / 2^s`-th entry). One small kernel writes `omega_n^k` for `k in [0, n/2)` into the workspace; subsequent butterfly kernels just index in.
3. **Shared-memory tiled butterfly.** With `n = 2^16` the full row does not fit in shared memory, but a 1024-element chunk does. Each block loads its chunk, runs all 10 intra-chunk butterfly stages from shared memory, then writes back. The remaining `log n - 10` stages span across chunks and stay in global memory.
4. **Decimation-in-frequency / Stockham variants.** Avoid the explicit bit-reverse permutation by interleaving it with the first stages, or use the auto-sort Stockham layout.
5. **Twiddle compression.** Store only one quadrant of roots and reconstruct the rest via the symmetry `omega^(j + n/4) = omega^(n/4) * omega^j`.

## Reference Solution

The shipped reference combines (1), (2), and (3):

- A small kernel precomputes `omega_n^k` for `k = 0 .. n/2 - 1` into the supplied workspace, replacing the per-thread twiddle exponentiation.
- A bit-reverse permutation kernel (one thread per element, swap-with-reverse) preserves the natural-order output convention used by the harness.
- An "intra-chunk" kernel (one block per chunk, 1024 elements / 512 threads, dynamic shared memory) runs the first 10 butterfly stages in shared memory using the precomputed twiddle table indexed by `j * (n / 2^s)`.
- An "inter-stage" kernel (one launch per remaining stage) handles the cross-chunk butterflies in global memory.
- All modular multiplications use the `reduce128` Goldilocks-specific path: compute `__umul64hi`, split the high half into top 32 and bottom 32 bits, fold via `2^64 = 2^32 - 1 (mod p)`, and finish with at most two corrective subtractions of `p`.

The combined effect is a single-digit-multiplier speedup over the baseline on the configured `(batch, n)` workload.

## Source

- Cooley, J. W. and Tukey, J. W., "An Algorithm for the Machine Calculation of Complex Fourier Series." Mathematics of Computation, vol. 19, no. 90, 1965, pp. 297–301. https://doi.org/10.1090/S0025-5718-1965-0178586-1
- Longa, P. and Naehrig, M., "Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based Cryptography." CANS 2016, LNCS vol. 10052, Springer. ePrint: https://eprint.iacr.org/2016/504
- Polygon Zero, Plonky2 — `goldilocks` crate `reduce128` implementation. https://github.com/0xPolygonZero/plonky2
