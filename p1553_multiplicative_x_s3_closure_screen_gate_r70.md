# P1553 multiplicative-x S3 closure screen R70

## Classification

- Owner: existing IDEA-057 auxiliary-ECFFT screens and IDEA-195
  non-Cartesian `S3` source-router residual; no new idea ID.
- Evidence: four deterministic prime-field families, public
  multiplicative-subgroup x-coordinate bases, 32 matched scalar-blind hash
  controls per family, exact Semaev `S3` replay, exact modular collision rank,
  and source-cost floors.
- Status: `DRAFT_REVIEW_REQUIRED_MULTIPLICATIVE_X_S3_NEGATIVE_SCREEN`.
- Cryptanalytic result: the frozen multiplicative-x prefixes show no
  transferable independent collision-rank excess over controls. Their complete
  pair probes exceed rho on all four toys and no sub-pair locator or fresh
  target descent is supplied. This is not a Shoup-bound improvement or ECDLP
  breakthrough.

## Bound predecessor

| Input | SHA-256 |
|---|---|
| R69 constructive closure report | `401b8ae56607f4962c9ae10ed99889d5c27ac865d0e27b3359f7b163a04dd451` |

R69 permits rank-reduction credit only for independently colliding residuals.
Fresh residual rows receive no information credit.

## Frozen candidate

For each curve, take a public multiplicative subgroup `H` of `F_p^*`, enumerate
its x-coordinates, choose the canonical square root on the target curve,
cofactor-clear into the prime-order subgroup, and freeze the first eight
distinct points. No scalar labels are consumed.

Every unordered seed pair emits the exact relation

```text
P_i + P_j + R_ij = O.                              (1)
```

For every nonidentity triple, R70 independently verifies the standard third
Semaev polynomial

```text
S3(x(P_i),x(P_j),x(R_ij)) = 0.                    (2)
```

The potential algorithmic attraction is that the x-domain has public
multiplicative indices. That is not enough: a useful route still needs an
exact sparse zero-locator for (2). Fast univariate arithmetic, ECFFT setup, or
a dense `p-1` character transform does not by itself locate the required
independent rows below pair cost.

## Four-family result

| Family | Candidate collision rank | Control mean | Control max | Pair/rho |
|---|---:|---:|---:|---:|
| `p193_a2_b3_q103_h2` | 3 | 3.03125 | 5 | 2.15385 |
| `p257_a1_b7_q281_h1` | 1 | 1.34375 | 3 | 1.27273 |
| `p337_a1_b3_q163_h2` | 4 | 1.56250 | 4 | 1.64706 |
| `p449_a1_b3_q463_h1` | 0 | 1.03125 | 3 | 1.03704 |

Each family uses 32 matched SHA-256 point bases of the same size. No candidate
rank exceeds the maximum control rank. The p337 candidate reaches the control
maximum and exceeds twice the control mean, but this does not transfer: the
p449 candidate has zero independent collision rank.

All 112 candidate `S3` checks pass. The complete pair probe uses 28 group
additions per family, above the respective rho baselines `13,22,17,27`.

## Admission

Three of seven obligations pass:

1. the coordinate source is public and scalar-blind;
2. four distinct prime-field families replay; and
3. every candidate relation passes exact `S3` verification.

Four obligations fail:

1. collision-rank excess does not beat every matched control on any family;
2. the complete pair probe is not below rho on any family;
3. no subquadratic `S3` locator is supplied; and
4. no fresh-target descent is supplied.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_MULTIPLICATIVE_X_S3_NEGATIVE_SCREEN
FOUR_PUBLIC_SCALAR_BLIND_PRIME_FIELD_FAMILIES
ALL_112_CANDIDATE_S3_CHECKS_EXACT
CANDIDATE_COLLISION_RANKS_3_1_4_0
NO_FAMILY_EXCEEDS_32_CONTROL_MAXIMUM
ALL_COMPLETE_PAIR_PROBES_ABOVE_RHO
NO_SUBPAIR_S3_LOCATOR
NO_TARGET_DESCENT
NO_NEW_IDEA
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: close the raw multiplicative-x prefix candidate unless
independent review identifies a non-prefix coset family with a prospective
density theorem. Route new work to a source with an explicit sub-pair collision
locator, not merely FFT-compatible coordinates.
