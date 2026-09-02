# P1553 external-pair pencil gate R50

## Classification

- Owner: existing P1553/IDEA-195 primitive degree-nine pencil frontier; no new
  idea ID.
- Evidence: exact exhaustive pair scan inside the deterministic R49 external
  block sample; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_EXTERNAL_PAIR_PENCIL_NEGATIVE_GATE`.
- Labels: `toy`, `exact`, `exhaustive-finite-sample-pairs`, `non-run`,
  `model-bound`, `novelty-unverified`.
- Cryptanalytic result: all 499,500 pairs among the 1,000 R49 external block
  sections are accounted for. Exactly 162,438 pairs have disjoint zero
  divisors and define 162,438 distinct basepoint-free sample pencils. Every
  pencil has exactly its two generating complete nine-point fibers; none has a
  third. The best has 104 selected collision pairs, versus at least 324 and
  nine complete fibers required. Unsampled blocks remain open. No Shoup-bound
  improvement or ECDLP breakthrough follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R46 catalog pencil-fiber gate | `6302cf24ff82dfc1d96c0cc21c0980c862a3218c0100ffb29905a35a8b63e2e4` |
| R49 external-denominator slice gate | `717ed5f1e01f8b96fd95598811369ec7debbebe04efde3e784172e875dddb7e2` |
| R49 bundle hash list | `58992545d6ecaac770d8e2c809f5ade30080b7871b154d33c22b126bdaea0d8f` |
| R49 staging receipt | `58fdf9f0381e1bd7db24b0df01c6f70818ac1410caa76af337d1d23d49de7e28` |

## Pair accounting

R50 reconstructs the 1,000 accepted external blocks and their exact normalized
sections from the R49 report. The complete overlap histogram is

| Shared selected points | Pairs |
|---:|---:|
| 0 | 162,438 |
| 1 | 205,988 |
| 2 | 102,122 |
| 3 | 25,202 |
| 4 | 3,465 |
| 5 | 282 |
| 6 | 3 |

These entries sum to

```text
C(1000,2)=499,500.                              (1)
```

Every zero-overlap pair has a distinct normalized Plucker key. R50 evaluates
all 162,438 resulting pencils on the full order-103 subgroup and checks the
selected 81-point fiber histogram.

## Result

Every pencil has

```text
complete nine-point fibers=2,
complete coverage=18.                           (2)
```

Selected collision-pair counts range from 72 to 104. The strongest witness
has multiplicity histogram

```text
{1:32, 2:5, 3:3, 4:3, 9:2},
collision pairs=104.                             (3)
```

No linear combination produces a third complete fiber. Together with R46's
142,286 catalog-catalog pencils and R49's exhaustive scoring of all 264,806
disjoint external-catalog pair instances during seed selection, the enlarged
1,807-section sample contributes

```text
142,286+264,806+162,438=569,530                  (4)
```

direct disjoint pair instances without a third complete selected fiber.

## Scope

The pair scan is exhaustive only inside the deterministic 1,000-block R49
sample. The combined 1,807 sections are tiny relative to all eligible blocks,
and absence of three collinear sampled sections is not a theorem for the full
degree-nine linear system.

R50 supplies no asymptotic pencil family, fresh-target locator, R10 queried
coefficients, relation-rank campaign, factor-base logarithms, or scalar-blind
descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_EXTERNAL_PAIR_PENCIL_NEGATIVE_GATE
ALL499500_EXTERNAL_SECTION_PAIRS_ACCOUNTED
162438_DISJOINT_PAIRS_DEFINE_162438_DISTINCT_SAMPLE_PENCILS
EVERY_EXTERNAL_PAIR_PENCIL_HAS_EXACTLY_TWO_COMPLETE_FIBERS
BEST_COLLISION_COUNT104_REQUIRED_AT_LEAST324
ENLARGED1807_SECTION_SAMPLE_569530_DIRECT_DISJOINT_PAIR_INSTANCES_NO_THIRD_FIBER
UNSAMPLED_BLOCKS_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: test a construction rather than another arbitrary
sample. Tile the 81-point R38 path by nine translated copies of its three
complete degree-three fibers. Enumerate the 253 degree-nine blocks formed by
unions of three local triple fibers whose window indices sum to 12, construct
all exact product-block sections, hash every two-section pencil by Plucker
coordinates, and test whether any pencil contains nine disjoint blocks that
cover all 27 labeled local triples.
