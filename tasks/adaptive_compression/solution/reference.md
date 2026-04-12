# Adaptive Compression — Reference Solution

## Background

This task asks an agent to build a byte-level predictor that minimizes cross-entropy
(bits-per-byte) across 9 diverse data families. The metric is the average negative
log2 probability assigned to each actual byte — mathematically equivalent to the
compression ratio achievable with optimal arithmetic coding. This tests the agent's
ability to build adaptive statistical models and combine them effectively.

## Baseline Approach (~5.0 bpb)

Order-1 Markov model with Laplace smoothing: a 256x256 count matrix of byte
transitions. Only captures single-byte context. Performs reasonably on order-1
Markov data and run-length data, but fails on all other families due to its
structural inability to model longer dependencies, periodicity, or nesting.

## Possible Optimization Directions

1. **Higher-order context models** — Extend from order-1 to order-2, 4, 8 using
   hash tables. Each higher order captures longer dependencies but requires more
   data to converge.

2. **Match model** — Search history for repeated byte patterns. When a long match
   is found, the next byte is highly predictable. Particularly effective for
   periodic and recurrence data.

3. **Period detector** — Autocorrelation-based detection of repeating patterns.
   Once the period is identified, predict the byte from one period ago.

4. **Context mixing** — Run multiple heterogeneous models simultaneously and
   combine their predictions. The key insight: no single model wins all families,
   so a mixture that adapts online will outperform any individual model.

5. **Specialized models** — CFG-aware stack model for nested structures,
   change-point detector for multi-regime data, run-length model for RLE data.

## Reference Solution (~3.8 bpb)

The reference combines optimizations 1-4:

**PPM Model (orders 0-6):**
- Single unified context model with hash-table frequency tracking at each order
- PPM-C escape-based blending: higher orders override lower when confident
- Small alpha smoothing (0.005, configurable) for probability estimation
- Minimum count threshold (2) before a context contributes

**Auxiliary Models:**
- Match model (min match length 3, 16-bit hash) — exploits repeated patterns
- Period detector (autocorrelation, max period 300, checks every 400 bytes)

**Mixer:**
- Exponentially weighted mixture (Hedge/multiplicative weights algorithm)
- Three components: PPM, match model, period detector
- Learning rate 0.5 for fast adaptation
- Automatically learns which component is useful for each data family

**Measured per-family performance (may vary slightly by seed):**
- markov_order1: ~4.3 bpb (PPM order-1 helps but limited by inherent entropy)
- markov_order2: ~5.4 bpb (PPM order-2 captures some 2-byte structure)
- periodic_clean: ~0.05 bpb (period detector achieves near-perfect prediction)
- periodic_noisy: ~2.6 bpb (period detector + noise handling)
- nested_structure: ~5.0 bpb (match model helps slightly with repeated tags)
- math_recurrence: ~0.05 bpb (match/period detector catches the cycle)
- multi_regime: ~5.5-6.0 bpb (PPM adapts but slowly across regime switches)
- run_length: ~1.1-2.4 bpb (PPM captures same-byte transitions)
- incompressible: ~8.0 bpb (no model helps; mixer learns near-uniform)

**Room for improvement beyond reference:**
- Bit-level modeling instead of byte-level (finer probability estimation)
- Specialized run-length model (explicit run counting)
- Regime change detection for multi_regime
- CFG-aware stack model for nested_structure
- More aggressive context mixing with per-context learned weights

## Sources

- Mahoney, M. "Adaptive Weighing of Context Models for Lossless Data Compression"
  (2005) — Foundation of PAQ-family compressors
- Krichevsky, R. & Trofimov, V. "The Performance of Universal Encoding" (1981) —
  KT estimator for sequential probability assignment
- Willems, F., Shtarkov, Y. & Tjalkens, T. "The Context-Tree Weighting Method"
  (1995) — Optimal Bayesian mixing of Markov models
- PAQ family of compressors: http://mattmahoney.net/dc/
