# P1553 R112 gauge-normalized endpoint Query2P1 gate

## Scope

R112 quotients the R110/R111 determinant gauge at the predicate level by
using the complete projective endpoint key `[X:Y:Z]`, with
`O = [0:1:0]`. This is an exact typed key, not a scalar discrete-log
coordinate. The audit covers direct endpoint hashing, canonical `2C+3C`
splitting, fresh-target updates, no-relation controls, and scan-backed dyadic
source recovery.

## Exact replay

- All 16 actual and matched instances reproduce the 18 R111 sources.
- The `5A`, `2C`, and `3C` endpoint tables have exponents `B^2`,
  `B^(6/5)`, and `B^(9/5)` and jointly fit the `B^(9/4)` setup cap.
- The boundary `max(C2) <= min(C3)` gives a one-to-one split of canonical
  `5C` sources.
- Direct `5C` lookup and canonical `2C x 3C` lookup inspect the same
  `B^(3+o(1))` C-side source family.
- Scan-backed dyadic recovery inspects at most twice the `5C` list. Every
  deterministic shifted no-relation control also requires the full scan.
- The two R105 double fibers, R105 markers, and R108 weight 14400 replay.

## Cost boundary

The exact typed normalization and all three thin target-independent tables
are admitted. They do not supply an inside-cap `Query2P1` decision oracle.
The minimum standard pair materialization is `C2+C3` at
`B^(3+o(1))`; `A5+C2` and `A5+C3` are larger. Target updates and subset
source recovery remain `B^(3+o(1))`, above fresh work/workspace
`B^(5/4+o(1))`.

This closes typed hashing and direct pair-table grammars only. Integer
3SUM/kSUM transplants and standard resultant/Fitting routes are already
deduplicated against ECDLP-IDEA-012 R3/R4. No lower bound for arbitrary
query data structures is claimed.

## Disposition

Preserve one representation-sensitive exception: a gauge-invariant nonlinear
elliptic orbit-product recurrence for the canonical `C2 x C3` join, with
an exact target update and subset-stable source witness.

No exact inside-cap locator, factor-log solve, identical target descent,
generic-prime Shoup improvement, or ECDLP breakthrough is supplied.

## Exactly one next action

Construct or refute one gauge-invariant nonlinear elliptic orbit-product
recurrence for the canonical `C2 x C3` endpoint join, charging recurrence
order, target updates, source unranking, markers, logs, and descent.
