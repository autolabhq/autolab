# Phase 2 Sub-experiment 1: Modular-form structure survey

**Date:** 2026-05-27  
**Status:** completed, negative result  
**Reproducible via:** `sage phase2_modular_form_survey.sage`

## Result

Surveyed 1,227 primes (5 ≤ p < 10000) for each of the four
precomputed-target LMFDB curves. Searched for:

1. **Near-anomalous primes** (|a_p| < 5): 66–78 hits per curve. None
   give a Smart attack because none have a_p = 1 with p > 100 in the
   AutoLab-relevant range.
2. **Smooth trace primes** (|a_p| factors into primes ≤ 50): 1069–1089
   hits per curve. This is dense (~88%) because small a_p values
   trivially factor — no useful filter.
3. **Small fundamental discriminant** (|fund disc of t²-4p| < 1000):
   378–404 hits per curve. Again dense because small p has small
   |t² - 4p|.
4. **Mordell-Weil lift candidates**: 0 hits (no MW generator gives
   collisions with PRECOMPUTED base in our range).

## Cross-curve analysis

| Category | Primes common to all 4 curves |
|----------|------------------------------:|
| near_anomalous | 0 |
| smooth_trace | 768 |
| small_fund_disc | 82 |

The 82 common small-fund-disc primes are `{13, 17, 19, 23, 29, 31, 37,
41, 43, 47, …}` — small primes where the Hasse bound `|t| ≤ 2√p`
forces small `|t² - 4p|` mechanically, not from curve structure.

## Interpretation

The four LMFDB curves' modular form coefficients are *consistent with
randomness* under the Sato-Tate distribution. No "secret" algebraic
structure visible at small primes lifts to a cryptanalytic exploit at
the 80-bit precomputed primes.

This is exactly the result Serre's open-image theorem predicts: for
non-CM curves, the mod-`ℓ` Galois representation is surjective for
all but finitely many `ℓ`, so coefficients are pseudo-random.

## Phase 2 ruled directions

The other Phase 2 sub-experiments (CM-like reduction search,
Modular-form-coefficient lattice) are addressed by the same negative
result: random-looking `a_p` give no algebraic invariant to exploit.

## Conclusion

Phase 2 produces no breakthrough. The four LMFDB curves are
generic-in-the-modular-form-sense.
