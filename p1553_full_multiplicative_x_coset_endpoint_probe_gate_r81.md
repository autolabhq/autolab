# P1553 full multiplicative-x coset endpoint probe R81

## Classification

- Owner: existing P1553 R3/R70/R80 structured-factor-base frontier; no P1554
  and no new idea ID.
- Evidence: 16 prospectively frozen complete coordinate cosets on four
  prime-order toy families, 32 matched hash-to-curve controls, exact group
  endpoint replay, and verifier-only density/rank controls.
- Status:
  `REJECT_FULL_MULTIPLICATIVE_X_COSET_LIFT_MASK_GEOMETRY_ONLY`.
- Labels: `exact-finite`, `prospective`, `scalar-blind-source`,
  `verifier-dlp-separated`, `novelty-unverified`.
- Cryptanalytic result: no admitted relation source, factor-log solve,
  identical fresh-target descent, Shoup improvement, or ECDLP breakthrough.

R70 tested short prefixes taken from multiplicative-x sources. R81 tests the
distinct full-coset residual: the complete multiplicative coordinate coset is
frozen before outcomes and retains its exact square-root lift mask and
cofactor map.

The input domain has the sparse polynomial `X^d-c`, but its actual S4 triple
endpoint support behaves like a random factor base. Every frozen unordered
triple has a distinct endpoint-set key. The root-support log-log slope is
`3.0175`, versus `2.9825` for matched controls, and no coset meets the
`B^(9/4)` root-support cap.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R80 batched compiler report | `936537fb78908dd2916bf6fa5b2091f336b9a47217a1ff787b068ae0491992c5` |
| R70 multiplicative-x prefix report | `9e58c6178eb18b7535c59e117ad942465d6c7853890291dec2c7c0abdb4ffd89` |
| R31 registry containing R3 Query2P1 gate | `0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f` |
| Bound R3 Query2P1 gate | `b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e` |

## Prior-art boundary

Semaev introduced summation polynomials as a prime-field ECDLP relation
interface in [ePrint 2004/031](https://eprint.iacr.org/2004/031).
Amadori, Pintore, and Sala study prime-field factor-base variants and
known-RHS relation collection in
[ePrint 2017/609](https://eprint.iacr.org/2017/609).

R81 does not claim the use of multiplicative coordinate cosets itself as
novel. Its scoped contribution is the complete-coset, lift-mask-preserving,
scale-matched endpoint-support and known-RHS rank audit relative to R70 and
R80.

## Frozen geometry

Each family uses

```text
E/F_p: y^2 = x^3 + 1,
p = 2 mod 3,
#E(F_p) = p+1 = 6q,
q prime.                                                        (1)
```

Let `H` be the complete order-`d` multiplicative subgroup of `F_p^*`.
For fixed offsets `e=0,1,2,3`, freeze the coset

```text
C_e = g^e H,
product_(x in C_e) (X-x) = X^d-(g^e)^d.                        (2)
```

For every `x in C_e`, take the canonical square root when `x^3+1` is a
quadratic residue, apply cofactor multiplication by six, canonicalize the
Kummer sign, and deduplicate the output x-coordinate. This produces the
factor base.

The complete coset, lift decision, cofactor map, and canonical sign use only
public field and curve operations. No scalar label is consumed.

## Prospective families

| Family | `q` | `d` | Four resulting `B` values |
|---|---:|---:|---|
| `p98561_j0_q16427_h6_d16` | 16,427 | 16 | 5, 10, 7, 9 |
| `p3148097_j0_q524683_h6_d32` | 524,683 | 32 | 11, 18, 24, 16 |
| `p9603641_j0_q1600607_h6_d40` | 1,600,607 | 40 | 19, 21, 18, 20 |
| `p100683137_j0_q16780523_h6_d64` | 16,780,523 | 64 | 31, 30, 36, 35 |

All field and subgroup orders pass exact finite primality checks, each
coordinate subgroup divides `p-1`, every generator has order `q`, and every
factor-base point replays in that subgroup. Across the 16 cosets,

```text
0.0659 <= q/B^5 <= 5.2567,
q/B^3 >= 16.427.                                               (3)
```

Thus none of the measured triple supports is truncated by the ambient group.
Every coset has two independently frozen matched hash-to-curve controls of
the same factor-base size.

## Endpoint support

For each unordered triple `i<=j<=k`, compute the four Kummer roots

```text
x(P_i+P_j+P_k),
x(P_i+P_j-P_k),
x(P_i-P_j+P_k),
x(-P_i+P_j+P_k).                                               (4)
```

Repeated roots within one triple are deduplicated exactly. The triple receives
weight 1, 3, or 6 according to its number of ordered permutations, and the
weighted endpoint-set histogram sums to `B^3`.

Across all 16 complete cosets:

```text
every unordered-triple -> endpoint-set map is injective,
distinct endpoint-set count = C(B+2,3),
root-support exponent range = [2.7304, 2.8870],
root-support log-log slope = 3.0175,
endpoint-set log-log slope = 2.7976.                            (5)
```

The endpoint-set slope is depressed at these finite sizes by the exact
`1/6` coefficient in `C(B+2,3)`; injectivity gives the exact cubic
cardinality on every frozen instance. Only the smallest `B=5` endpoint-set
count falls below the numerical `B^(9/4)` cap, while no root support does.

The 32 controls have root-support slope `2.9825`. The candidate-minus-control
mean root exponent is `-0.0064`, with maximum absolute difference `0.0655`.
The sparse input equation (2) therefore supplies no transferred compression
after the lift mask and elliptic addition.

## Density and rank controls

For audit only, BSGS recovers the exact scalar label of every candidate and
control point. Those labels verify sampled group endpoints, exact pair/triple
convolutions, five-term target counts, source coefficient rows, and
known-RHS row rank.

The candidate construction and proposed algorithm may not consume those
labels:

```text
verifier BSGS work                  B^(5/2+o(1)),
setup/state cap                     B^(9/4+o(1)),
verifier labels receive algorithmic credit     false.           (6)
```

The controls show that the toys are not starved:

```text
candidate cosets with full sampled known-RHS rank   13/16,
matched controls with full sampled known-RHS rank   27/32,
candidate mean density minus control mean          -0.0079,
candidate cosets with verified generator source     12/16.       (7)
```

These are verifier diagnostics, not a factor-log algorithm. They show that
the endpoint-support failure is present even when ordinary density and rank
behavior survives.

## Cost ledger

The complete coordinate coset has a two-coefficient input description, and
the factor base plus lift mask costs `B^(1+o(1))` state. That compact input
does not survive the map to triple endpoints:

```text
exact endpoint-set state                 Theta(B^3) on all rows,
measured root-support slope              3.0175,
standard exact triple construction       B^(3+o(1)),
standard fresh pair construction         B^(2+o(1)),
standard scalar-blind source return       B^(3+o(1)).            (8)
```

The required limits remain:

```text
setup/state                    B^(9/4+o(1)),
fresh-target work/workspace    B^(5/4+o(1)).                     (9)
```

No sparse resultant, multiplicative FFT, or verifier DLP is credited as a
source query. A compact input polynomial is not the required compact S4
endpoint representation.

## Scope

R81 rejects only complete one-dimensional multiplicative-x cosets with their
canonical lift mask and cofactor map. It does not prove a lower bound against
higher-dimensional compact divisors, non-Cartesian factor bases, sparse
straight-line divisor circuits, or unrestricted structured endpoint
representations.

No generic-prime family theorem, scalar-blind subcap source query, candidate
factor-log solve, identical target descent, or Shoup-bound improvement is
supplied.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_full_multiplicative_x_coset_endpoint_probe_r81.py` | `c755542a96dba8423a4ea4f4e2b861263c45eb8cfe73db10e5dff2789632c3d0` |
| `p1553_full_multiplicative_x_coset_endpoint_probe_report_r81.json` | `e556efa7c1e639f76915f152ebdcc3d00db2a932d8ef207ae8653c94117f026f` |
| `frozen_full_multiplicative_x_coset_geometry.json` | `1766734ec232564b79e06f0ab8a31f1ad4fc4fa019a8d7f8ad30c8362accde71` |
| `triple_endpoint_support_and_lift_mask_replay.json` | `27982f64300420c02246a5e0de97140d194c4ea3ba03a10896136402b7f762c9` |
| `matched_random_density_rank_controls.json` | `5ef6c3d870db0246f7235cde5196fac001a4fd9fb38e4b0621213e38d9f7caa0` |
| `scalar_blind_source_and_cost_ledger.json` | `5f3e284794595f6301e7f0926d21ca0fcb6d9b047f14fd6e37b77ea803309eaa` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_full_multiplicative_x_coset_endpoint_probe_r81.py` | `bd040c951d760cf2ea78d6205628459bdeffd79121accffc5f09a8f3e29ba11c` |

Targeted replay:

```text
Ran 4 tests in 0.006s
OK
families=4 cosets=16 root_cap=0 endpoint_cap=1 lane_admitted=False
```

## Disposition

```text
REJECT_FULL_MULTIPLICATIVE_X_COSET_LIFT_MASK_GEOMETRY_ONLY__FOUR_Q_THETA_B5_PRIME_ORDER_FAMILIES__SIXTEEN_COMPLETE_COSETS__THIRTY_TWO_MATCHED_HASH_CONTROLS__SPARSE_XD_MINUS_C_DOMAIN__EXACT_LIFT_MASK_AND_COFACTOR_MAP__EXACT_GROUP_ENDPOINT_REPLAY__TRIPLE_ROOT_AND_ENDPOINT_SET_CAP_FAILURE__VERIFIER_BSGS_B2P5_EXCLUDED__NO_SCALAR_BLIND_SUBCAP_SOURCE__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Leave one-dimensional field cosets and construct one scalar-blind compact
divisor factor base whose triple S4 endpoint representation is proved at
most `B^(9/4+o(1))` before enumeration. Require a field-level source query at
most `B^(5/4+o(1))`, prospective density, matched rank, verified logs, and
identical target descent; do not use verifier DLP labels.
