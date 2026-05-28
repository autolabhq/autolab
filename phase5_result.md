# Phase 5: Quantum-inspired classical algorithms (de-quantization)

**Date:** 2026-05-27  
**Status:** completed, negative result (literature survey)

## Background

Shor's algorithm solves ECDLP in **polynomial time** on a fault-
tolerant quantum computer. The crucial subroutine is **Quantum Fourier
Transform-based period finding**: for the function `f(x, y) = x P + y Q`
mod `n`, QFT extracts the period `(1, -k)` from a superposition over
all `(x, y)`.

**De-quantization** (Tang 2018+) has produced classical analogs of
*some* quantum subroutines, primarily for **low-rank linear algebra**
(matrix sampling, recommendation systems, principal component
analysis). The classical algorithms require specific data-structure
preconditions (`ℓ_2`-norm samplable, low rank).

## Survey of de-quantization literature relevant to ECDLP

The bundled CRYPTO 2022/2023/2025 proceedings contain de-quantization
references but **none** address ECDLP. The bundled proceedings
explicitly mention:

- Lattice de-quantization (Tang, Bruna, Regev, Song 2021): for LWE-
  related problems, not DLP.
- Black-hole information dequantization (Kim-Tang-Preskill 2020): not
  cryptographic.

**Beyond the bundled corpus** (from cryptanalytic knowledge):

- Period-finding de-quantization for ECDLP: **not published**. The
  QFT's exact-coherent superposition over all `(x, y)` has no known
  classical sampling analog that yields the period in polynomial
  time.
- The closest classical period-finder is Pollard rho, which is `O(√n)`
  — the same as the deterministic generic lower bound (Shoup 1997).

## Conclusion

Phase 5 produces no breakthrough. The de-quantization technique that
worked for low-rank linear algebra fundamentally requires the matrix-
sampling preconditions that don't apply to the DLP function `f(x, y)
= x P + y Q`. The DLP function's image is uniformly distributed
across the group, not low-rank.

A future Phase 5' could investigate:

1. **Approximate dequantization via Markov-chain Monte Carlo
   sampling.** Recent work (Bharti-Anand 2023+) explores classical
   sampling that mimics QFT post-selection. No ECDLP application
   known.
2. **Hybrid classical-quantum protocols.** If small quantum devices
   become available, hybrid algorithms might give partial speedup.
   This is engineering, not algorithmic.

Both are open-ended research directions. Neither offers a near-term
breakthrough.
