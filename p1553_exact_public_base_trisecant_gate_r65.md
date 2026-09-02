# P1553 exact public-base trisecant gate R65

## Classification

- Owner: existing P1553/IDEA-195 primitive product-pencil frontier; no new
  idea ID.
- Evidence: exhaustive all-pairs Plucker incidence scan with two 64-bit
  candidate fingerprints (jointly 128-bit), full exact-key replay,
  divisor-overlap checks, and
  one pinned primitive witness; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_EXACT_PUBLIC_BASE_INCIDENCE_GATE`.
- Labels: `toy`, `exact-exhaustive-incidence-scan`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: R65 exactly counts smooth trisecants for the R63
  `B=24` public base. One primitive basepoint-free smooth trisecant exists, but
  finding it costs quadratic work in the smooth-product catalog and supplies
  no generic incidence law, rank, logs, linear algebra, or descent. It is not
  a Shoup-bound improvement or ECDLP breakthrough.

## Inputs

| Input | SHA-256 |
|---|---|
| R64 coordinate-base diagnostic gate | `fda9fce027da62857dd5ac7327a3ebf03b24474a456c63a7ed4cded017a81ac9` |
| R64 structured-base report | `c63fb54f8aa8df42bbc1f4bd5118519af28933fee50bcf4e9dab01aa5e1e08a9` |
| R63 imported implementation | `7f5d061a539d62d2b3d362fed8252fead5b8e14dbc9250f5543e686f3a8d89cb` |
| R62 transitively used implementation | `15ca1a87202d069b79325b7a1743762d221f0726e793020910fb74c3633d900a` |
| R61 directly consumed locator report | `f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64` |

## Exact catalog and scan

The R63 coordinate-hash prefix at `B=24` has 14 and 21 projected triples in
classes 46 and 54. Four even two-torsion lifts per triple give 56 and 84 line
sections. Their 3,816 admissible factor pairs aggregate to 3,748 distinct
degree-six product sections:

```text
factorization multiplicity 1: 3680,
factorization multiplicity 2:   68.               (1)
```

Catalog multiplicity is not the same event as a unique rational rank-one
fiber. Exact R62 lifting accepts 3,656 multiplicity-one sections and rejects
24 more; all 68 multiplicity-two sections are also nonunique.

R65 fingerprints all

```text
C(3748,2) = 7,021,878                           (2)
```

section pairs using normalized 15-coordinate Plucker vectors. Fingerprints
are only candidate filters. All 15,170 repeated-fingerprint ranges are
recomputed with their complete exact Plucker keys; no fingerprint collision
group occurs.

There are 15,170 exact catalog trisecant lines. Their common base-divisor
degree histogram is

```text
degree 0:     1,
degree 1:    39,
degree 2:   244,
degree 3: 10296,
degree 4:  4590.                                 (3)
```

Every pair overlap on each line equals the line's fixed base degree. Thus only
the degree-zero line is a primitive pencil; it contains exactly three
pairwise-disjoint smooth products. The report pins all three normalized
sections, six representative factor lines per relation fiber, projected atom
indices, torsion lift patterns, and factorization multiplicities.

## Conditional rate

Of (2), exactly 2,700,566 endpoint pairs have disjoint six-point divisors. The
unique primitive three-point line contributes three endpoint-pair incidences,
so the exact conditional smooth-closure rate is

```text
3 / 2,700,566 = 1.110878238e-6.                  (4)
```

The unconditional independent-factor-pair reference for a smooth section with
a unique rational rank-one lift is

```text
3656 / |P^2(F_193)|^2 = 2.607743738e-6.          (5)
```

This unconditional reference is about 2.347 times (4). It is not a matched
conditional null for disjoint source pencils, so the ratio is descriptive.
Uniform sampling from the exact disjoint-endpoint population would have mean
only `0.00056877` hits in 512 draws. Equations (4)--(5) are exact finite-toy
facts, not an asymptotic suppression theorem.

## Cost boundary

The scan materializes `C(M,2)` pair records and lexicographically sorts their
fingerprints, so the conservative runtime is `O(M^2 log M)` and memory is
`Theta(M^2)` for `M=3748`. That is already 7.0 million pair records at `B=24`
and is not a relation-generation improvement. The one relation does not reveal
projected factor-base logs, and no independent relation matrix or fresh-target
descent is present.

Base materialization and the legacy labelled subgroup setup remain uncharged.
The inherited R61 multiplication-map and sextic construction also contains an
uncharged `Theta(N^3)` triple-enumeration preprocessing pass. No generic-prime
family or incidence bound is established.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_EXACT_PUBLIC_BASE_INCIDENCE_GATE
B24_3748_DISTINCT_SMOOTH_PRODUCTS_7021878_ALL_PAIRS
15170_EXACT_SMOOTH_TRISECANT_LINES
15169_LINES_HAVE_POSITIVE_BASE_DIVISOR
ONE_PRIMITIVE_LINE_THREE_PAIRWISE_DISJOINT_SMOOTH_PRODUCTS
EXACT_CONDITIONAL_RATE_3_OVER_2700566
3656_UNIQUE_RATIONAL_LIFT_SMOOTH_SECTIONS
UNCONDITIONAL_UNIQUE_LIFT_REFERENCE_OVER_EXACT_RATE_2_347
M2_LOG_M_DISCOVERY_TIME_AND_M2_MEMORY
NO_GENERIC_INCIDENCE_THEOREM_RANK_LOGS_LINEAR_ALGEBRA_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: express the unique primitive witness as a rank-one
tensor relation modulo the three-dimensional multiplication kernel and test
whether that relation admits a public generation rule cheaper than quadratic
smooth-product pair search.
