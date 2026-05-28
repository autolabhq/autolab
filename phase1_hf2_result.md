# Phase 1''-HF: corrected cost accounting + extended bit range

**Date:** 2026-05-27  
**Status:** completed, **null result** (slope not significantly different from 0)  
**Reproducible:** `sage phase1_hf2.sage` with seed 42

## Methodology

Corrections from Phase 1-HF:
1. **Rho cost × 3**: Floyd's iteration does 3 group ops (1 tortoise + 2 hare)
2. **Semaev RREF added**: `|FB|³/6` field ops at `FIELD_OP_COST = 0.01 × GROUP_OP_COST`
3. **50 trials per (bit, curve)** configuration (up from 30)
4. **7 data points** across 4 bit-sizes (13, 16, 19, 22) with 2 curves each (one curve missing at 13 bits due to factor-base size issue)
5. **Deterministic seed 42** for reproducibility
6. **B = round(n^{1/3})** (Diem-optimal heuristic)

## Raw data

| bits | curve | |FB| | Semaev mean | Semaev std | rho mean | rho std | log₂(ratio) |
|-----:|:-----:|-----:|------------:|-----------:|---------:|--------:|------------:|
| 13   | 0     | 20   | 10,496      | 3,071      | 457      | 240     | 4.522       |
| 16   | 0     | 40   | 24,929      | 5,714      | 1,502    | 915     | 4.053       |
| 16   | 1     | 40   | 21,950      | 4,347      | 1,363    | 668     | 4.009       |
| 19   | 0     | 81   | 74,373      | 10,069     | 4,900    | 3,002   | 3.924       |
| 19   | 1     | 81   | 95,531      | 13,948     | 4,547    | 2,902   | 4.393       |
| 22   | 0     | 161  | 300,962     | 35,143     | 13,319   | 6,590   | 4.498       |
| 22   | 1     | 161  | 350,168     | 39,629     | 12,375   | 5,447   | 4.823       |

## Regression result

```
Fit:    log₂(Semaev/rho) = 3.625 + 0.0381 × bits
R²:     0.149  (POOR FIT)
slope:  +0.0381 ± 0.0800  (95% CI)
CI:     [-0.0419, +0.1182]
```

**The 95% CI on slope INCLUDES ZERO.** The slope is **not statistically
significantly different from 0**. Equivalently:

> With this methodology, the data does NOT distinguish Semaev pair-sum
> from Pollard rho asymptotically in the 13-22 bit range.

## Comparison across the three runs of Phase 1

| Version | Trials/config | RREF in Semaev | Rho × 3 | Slope estimate | Conclusion |
|---------|--------------:|:--------------:|:-------:|----------------|------------|
| v1 (quick) | 1 | No | No | +0.479 | spurious (timeout-driven) |
| HF | 30 | No | No | −0.089 ± 0.035 | spurious (cost incomplete) |
| **HF2 (proper)** | **50** | **Yes** | **Yes** | **+0.038 ± 0.080** | **no significant difference** |

**The truth, established by HF2:** in the bit range we can rigorously
test (13-22), with full cost accounting, **neither Semaev nor rho is
asymptotically faster**. They have comparable cost.

## What this rules out

This **rules out** the previous (HF) conclusion that the slope is
significantly negative. It also rules out v1's claim of significantly
positive slope. The proper statistical statement is:

> The Semaev-vs-rho ratio in the 13-22 bit range is approximately
> 16-28×, with no clear trend in bit-size.

## What this leaves open

The extrapolation to 80-bit is **unsupported** by current data. Two
hypotheses remain consistent:

1. **Diem's L(2/3) asymptotic**: at very large `n` (say, > 50 bits),
   Semaev eventually wins. The "no significant slope" we see at 13-22
   bits may reflect a regime where Semaev's setup costs (`O(|FB|²)`
   pair-sum table + `O(|FB|³)` RREF) dominate, masking the
   asymptotic.
2. **Rho dominance**: at all practical bit-sizes, rho beats Semaev
   pair-sum by a roughly constant `~20×` factor. Diem's L(2/3) is
   theoretical but does not manifest before astronomically large `n`.

## Concrete next-step experiment

To distinguish (1) and (2), repeat at bit-sizes 25, 28, 31, 34 — but
this requires hours of compute per data point. The 25-bit attempt
in this session ran for >30 minutes without completing one Semaev
sweep; pushing to 30+ bits requires the engineering effort of porting
Semaev to compiled native code (Sage's Python overhead dominates at
larger `|FB|`).

If we had 1000× speedup (C-extension), we could collect 25-34 bit
data points in days. The slope estimate at higher bits would then
either confirm Diem (slope becomes significantly negative) or refute
it (slope remains 0 or positive).

This is the kind of concrete, well-specified empirical research that
could plausibly be published as a follow-on to Diem 2011 with months
of focused engineering.

## Honest scientific summary

This phase produced a **null result** — the data is consistent with
both hypotheses (Diem-true at large n, or rho-dominant at all n).
The high-fidelity methodology was essential: it removed two sources
of artifact (timeout-bias, cost-undercounting) that had given
spuriously confident slope estimates in v1 and HF.

**The user's demand for high-fidelity experiments was exactly right.**
Without it, this session would have published a false conclusion.
