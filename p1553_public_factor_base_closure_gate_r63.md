# P1553 public factor-base closure gate R63

## Classification

- Owner: existing P1553/IDEA-195 primitive product-pencil frontier; no new
  idea ID.
- Evidence: deterministic nested public-base census, exact fixed-class
  catalogs with torsion lifts, sampled sextic pencil restrictions, and pinned
  smooth-closure witnesses; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_FACTOR_BASE_CLOSURE_MODEL_GATE`.
- Labels: `toy`, `deterministic-public-factor-base-probe`,
  `heuristic-cost-model`, `non-run`, `model-bound`, `novelty-unverified`.
- Cryptanalytic result: R63 finds two rare smooth-output pencils. After
  correcting the factor-section denominator and root-lift accounting, this
  exploratory sample supports neither amplification nor its absence. The
  asymptotic cost models are assumption-bound, not a generic-prime
  impossibility theorem. No Shoup-bound improvement or ECDLP breakthrough is
  claimed.

## Inputs

| Input | SHA-256 |
|---|---|
| R62 scan-free factor-lift gate | `38c785bd7d1e6fb1b52d04f07e363de6d63c840187f5b38000e203bddb7187f4` |
| R62 exact factor-lift report | `31d06e333d24d79382d065fc88ae0b88cd0d58b4307110588275dc1d3995ec37` |
| R62 imported implementation | `15ca1a87202d069b79325b7a1743762d221f0726e793020910fb74c3633d900a` |
| R61 directly consumed locator report | `f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64` |

## Public projected-atom bases

R63 orders the 103 prime-subgroup points by

```text
SHA256("P1553-R63|" || canonical point encoding)   (1)
```

and takes nested prefixes of sizes

```text
B = 12,16,20,24,28,32,40,48.                     (2)
```

This is a public coordinate definition. The toy implementation materializes
the full labelled subgroup by `N` independent double-and-add scalar
multiplications and then sorts it; that literal `Theta(N log N)` group work
plus `O(N log N)` sorting is uncharged. Scalar labels accelerate enumeration
only: every accepted triple is independently checked by public curve addition
against the class points `46G` and `54G`.

The relation variables are projected subgroup atoms, not only points in the
subgroup coset itself. Each projected class triple therefore expands to the
four lifts with even two-torsion parity. This correction is essential: using
only the all-`H` lift gives the wrong smoothness model.

The projected triple counts for the two classes are

```text
B:    12     16     20      24      28      32      40      48
C46:   3      4      7      14      26      36      92     169
C54:   3      7      9      21      39      52      91     165. (3)
```

After four torsion lifts per triple, disjoint pairing, and exact aggregation
by normalized degree-six section, the largest base has 393,764 distinct
smooth products. Partition multiplicities are retained rather than counted as
new sections; at `B=48` the histogram is

```text
multiplicity 1: 382660,  2: 10980,  3: 124.       (4)
```

## Direct closure probe

For each base, R63 selects 512 deterministic pencils through two disjoint
smooth products. Every sextic restriction is nonzero. Polynomial roots are
deduplicated within each pencil, and every resulting section must pass the
rank-one lift before entering the denominator. Failed rational hypersurface
points and repeated roots are recorded separately.

No lifted product section is smooth for `B <= 32`. At `B=40`, the raw count of
500 roots becomes 497 distinct roots and 496 rational rank-one lifts; one is
smooth at parameter `177`. At `B=48`, 500 becomes 496 distinct roots and 494
rational lifts; one is smooth at parameter `181`. Both reports pin the two
source products, their factor lines and torsion lift patterns, the target
section, and its smooth factorization.

The original denominator `1717^2` is invalid because 1717 counts only factor
sections already known to split over the subgroup. A rational factor section
ranges over

```text
|P^2(F_193)| = 193^2 + 193 + 1 = 37443.           (5)
```

Dividing multiplicity-one smooth catalog sections by the unconditional
factor-pair universe `37443^2` gives reference intensities `8.1585e-5` at
`B=40` and `2.7294e-4` at `B=48`. The observed root population is conditioned
on having a unique rational rank-one lift, so multiplying these unconditional
intensities by 496 or 494 is not an expected count for the filtered trials.
The resulting 0.0405 and 0.1348 values are retained only as descriptive
reference-intensity products. Roots are also clustered within pencils, bases
are nested, and the family was exploratory. R63's amplification inference is
therefore `inconclusive`.

## Cost boundary

If six projected atoms behave independently, a base `B=N^beta` gives

```text
Pr[smooth] ~= (B/N)^6,
relation collection for B rows ~= N^6/B^5.        (6)
```

The fixed-sum constraint permits at most `B(B-1)/6` distinct triples in one
class. Under optimistic independence between the two factor classes, this
changes (6) to `N^4/B^3`; under an even more optimistic perfect-correlation
model it becomes `N^2/B`. Both remain at least linear in `N` for `B<=N`.

Those are distribution models, not universal lower bounds: pencils chosen
from the same structured base could bias their third products. The direct
coordinate-hash sample is too small and too dependent to establish the sign
or size of that bias.

The harness also does not charge base materialization, construction of smooth
source products, relation rank, projected scalar logs, sparse linear algebra,
or fresh-target descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_FACTOR_BASE_CLOSURE_MODEL_GATE
PUBLIC_PROJECTED_ATOM_BASES_WITH_FOUR_EVEN_TORSION_LIFTS
PARTITION_MULTIPLICITIES_PRESERVED
4096_ACCEPTED_DISJOINT_SMOOTH_SOURCE_PENCILS
TWO_REPLAYABLE_SMOOTH_OUTPUT_WITNESSES_AT_B40_AND_B48
ROOTS_DEDUPLICATED_AND_RATIONAL_RANK_ONE_LIFTS_REQUIRED
UNCONDITIONAL_MULTIPLICITY_ONE_REFERENCE_OVER_P2_SQUARED
CONDITIONED_ROOT_EXPECTATION_NOT_CLAIMED
AMPLIFICATION_INFERENCE_INCONCLUSIVE
ASYMPTOTIC_COST_OBSTRUCTION_IS_MODEL_BOUND_NOT_A_THEOREM
BASE_SOURCE_RANK_LOGS_LINEAR_ALGEBRA_DESCENT_UNCHARGED
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: search a preregistered family of public
coordinate-defined bases against the corrected uniform-factor-pair null, using
pencils as sampling clusters while charging base construction, source-product
generation, and the relation/linear-algebra exponent required to beat rho.
