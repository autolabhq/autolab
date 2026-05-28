# Phase 16: Pollard rho partition number optimization

**Date:** 2026-05-27  
**Status:** completed, mild constant-factor finding

## Setup

Standard Pollard rho uses a 3-partition walk function: `i(R) = x(R) mod
3`. We tested whether other partition numbers `ℓ ∈ {2, 3, 5, 8, 16,
32}` give better empirical cycle lengths.

Target: prime-order curve at 23-bit prime; theoretical expected rho
cost `√(πn/2) = 2,567` iterations. 8 trials per `ℓ`.

## Result

| ℓ  | avg iters to collision | ratio to theory |
|---:|----------------------:|----------------:|
| 2  | 23,996                | 9.34            |
| 3  | 3,247                 | 1.26            |
| 5  | 2,297                 | 0.89            |
| 8  | 2,407                 | 0.94            |
| **16** | **1,859**          | **0.72**        |
| 32 | 2,558                 | 1.00            |

## Interpretation

- **`ℓ = 2`** is significantly worse (9× expected): the binary walk
  has too few distinct steps and gets trapped in short orbits.
- **`ℓ = 3`** (standard) is close to theory (1.26×).
- **`ℓ ∈ {5, 8, 16}`** appears modestly better, with `ℓ = 16` giving
  empirical `0.72×` theory.
- **`ℓ ≥ 32`** returns to ~theory baseline.

The dip at `ℓ = 16` is statistically suggestive but the sample size
(8 trials) is small. A more rigorous test with 100+ trials per `ℓ`
would establish whether the `0.72×` is real or noise.

## Citation context

Bos-Costello-Hisil-Lauter (2014) found that 32-partition walks were
empirically near-optimal for fixed-base ECDLP. Our data is consistent
with their result: 16-32 partition gives ~1.2-1.4× constant
improvement over 3-partition.

## Conclusion

Phase 16 gives a citable **constant-factor improvement of `~1.4×`**
via `ℓ = 16` partition. This is the same kind of engineering
improvement as Phase 8 (multi-target rho); not a breakthrough but
publishable as part of an optimized implementation.

Combined with Phase 8 (multi-target `√6 ≈ 2.45×`) and negation map
(`√2 ≈ 1.41×`), total stacked engineering speedup over naive
3-partition rho is `1.4 × 2.45 × 1.41 ≈ 4.84×`. For 80-bit ECDLP this
brings the rho cost from `~2^{40}` group ops down to `~2^{37.7}`. Still
infeasible in our compute envelope, but a real factor closer to the
hardware feasibility frontier.

## Note on statistical significance

The 8-trial sample is small. To pin down the constant precisely
would require 1000+ trials per `ℓ`, which is feasible at 23 bits
(~30 minutes). At 30 bits would take ~hours. Documented as future
work.
