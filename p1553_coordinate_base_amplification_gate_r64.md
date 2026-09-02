# P1553 coordinate-base amplification gate R64

## Classification

- Owner: existing P1553/IDEA-195 primitive product-pencil frontier; no new
  idea ID.
- Evidence: preregistered twelve-family public-base search with exact smooth
  catalogs, 6,144 disjoint pencils, fixed-degree locators, multiplicity-aware
  membership, and multiple-testing correction; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_COORDINATE_BASE_DIAGNOSTIC_GATE`.
- Labels: `toy`, `preregistered-multiple-hypothesis-search`,
  `heuristic-cost-model`, `non-run`, `model-bound`, `novelty-unverified`.
- Cryptanalytic result: none of the twelve sampled families yields a smooth
  new root. The preregistered root-level binomial endpoint is invalid because
  roots are clustered within pencils, so the amplification inference is
  inconclusive rather than negative. No Shoup-bound improvement or ECDLP
  breakthrough is claimed.

## Inputs

| Input | SHA-256 |
|---|---|
| R63 public factor-base closure gate | `c1fb6244d057f2ee5d37788c3cb98081ee5e852ea068af109c73de31027d33cd` |
| R63 deterministic closure report | `bba00af8c72646a2405f9932ea81778b6b359db040d1ea9591686baa5010bfcc` |
| R63 imported implementation | `7f5d061a539d62d2b3d362fed8252fead5b8e14dbc9250f5543e686f3a8d89cb` |
| R61 directly consumed locator report | `f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64` |

## Preregistered search

R64 fixes `B=40`, 512 accepted disjoint pencils per family, seed 1564, and
twelve public orderings before running the search:

```text
two salted hashes;
x and y ascending/descending;
x+y, x-y, 2x+y, x+2y modulo p;
x^2 and y^2 modulo p.                             (1)
```

Both the family list and `B=40` were selected after R63; in particular, R63
identified 40 as its first positive base size. The within-R64 declaration is
therefore reproducible but not prospectively independent.

The declared replay statistic is a one-sided root-level binomial tail using
the unconditional factor-pair reference intensity

```text
q = multiplicity-one smooth sections / |P^2(F_193)|^2. (2)
```

Roots are deduplicated within each pencil and must pass a rational rank-one
lift. Their population is therefore conditioned while (2) is unconditional;
in addition, each pencil contributes zero to four dependent roots. Bonferroni
corrects twelve-family multiplicity but cannot repair either mismatch. R64
retains those tails only as uncalibrated replay statistics.

## Result

All twelve families produce 512 disjoint, nondegenerate source pencils. Their
class counts are near the random-base scale: 87--107 triples per class. After
root deduplication and rank-one lifting, each family yields 448--519 rational
product sections.

No new lifted section is smooth in any family. Therefore every amplification
point estimate is zero and every uncalibrated upper-tail value is one. R64
supplies no positive coordinate-base candidate, but the invalid endpoint does
not support a statistically significant negative conclusion.

The preregistered test asked only for amplification. Post hoc, the pooled data
contain

```text
5830 deduplicated rational rank-one product sections,
0 smooth roots,
0.531138 as reference intensity times conditioned count. (3)
```

The value in (3) is not an expected count for the filtered population. If the
conditioning mismatch were ignored and all roots and families were treated as
independent, the zero-count replay statistic would be `0.5879`. Neither
assumption is valid, so there is no pooled amplification or suppression signal.

## Cost boundary

The toy run charges exact counts of `2*C(40,3)` public triple tests and all
lifted factor-pair candidates per family, plus 512 sextic restrictions and
factorizations. It also performs 5,867 constant-size Groebner rank-one lift
attempts, accepting 5,830. It does not extrapolate those counts to a generic
family.

The literal setup performs `N` separate double-and-add scalar multiplications
and sorts the records for every family, for `O(N log N)` toy work. A
successive-addition implementation could reduce point generation, but is not
used. Setup remains uncharged. Relation rank, projected scalar logs,
factor-base linear algebra, and fresh-target descent remain absent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_COORDINATE_BASE_DIAGNOSTIC_GATE
TWELVE_PREREGISTERED_PUBLIC_COORDINATE_FAMILIES
6144_DISJOINT_NONDEGENERATE_SMOOTH_SOURCE_PENCILS
5830_DEDUPLICATED_RATIONAL_RANK_ONE_PRODUCT_SECTIONS_ZERO_SMOOTH
ROOT_LEVEL_BINOMIAL_ENDPOINT_CLUSTER_INVALID
UNCONDITIONAL_REFERENCE_CONDITIONED_TRIAL_MISMATCH
AMPLIFICATION_INFERENCE_INCONCLUSIVE
BASE_SETUP_RANK_LOGS_LINEAR_ALGEBRA_DESCENT_UNCHARGED
NO_ASYMPTOTIC_ALGORITHM
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: replace the invalid root-level endpoint with an
exact conditional incidence count by exhaustively classifying primitive
smooth trisecants for the largest tractable public base, preserving divisor
multiplicity and endpoint-disjointness.
