# P1553 external-denominator collision-slice gate R49

## Classification

- Owner: existing P1553/IDEA-195 primitive degree-nine pencil frontier; no new
  idea ID.
- Evidence: exact deterministic finite sample outside the retained block
  catalog plus exhaustive declared slices; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_EXTERNAL_DENOMINATOR_SAMPLE_NEGATIVE_GATE`.
- Labels: `toy`, `exact`, `deterministic-finite-sample`, `non-run`,
  `model-bound`, `novelty-unverified`.
- Cryptanalytic result: R49 deterministically generates 1,000 distinct
  equal-Abel-sum nine-point blocks outside the R45 catalog, verifies every
  exact section zero set, chooses the highest-collision disjoint catalog seed
  for each, and exhausts seven 194-pencil collision slices per denominator.
  The sweep generates 1,358,000 instances and scores 1,289,607
  subgroup-basepoint-free candidates. Ninety-one denominators improve their
  secondary collision score, by at most eight, but none gains a third complete
  nine-point fiber. Nine are required. Unsampled blocks and pencils outside the
  declared slices remain open. No Shoup-bound improvement or ECDLP
  breakthrough follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R42 single-interval pencil DLP gate | `8a8de8726e4fc9542f97cf7e1ba4fc54596307cc544da0d0b18b588553022dae` |
| R46 catalog pencil-fiber gate | `6302cf24ff82dfc1d96c0cc21c0980c862a3218c0100ffb29905a35a8b63e2e4` |
| R48 all-catalog-denominator slice gate | `76952071e56ade008852176cd17d71e23a8f98f0b7b674464fd1dd85f8c1f81f` |
| R48 bundle hash list | `ce1a6f65097d907417494fba01c5c4ed529fa5509cbad3ad971c7befc2a6e611` |
| R48 staging receipt | `815ca9b7afd5af60e1737d7d2109d28b7739d5a46bf585aa38d1c5fc5acabc57` |

## External block generation

For the translated interval point associated with offset `k`, the scalar is
`42*(k-40) mod 103`. A nine-offset block has Abel sum zero exactly when

```text
sum(k) = 9*40 = 360 = 51 mod 103.                (1)
```

Using deterministic seed 1553, R49 draws eight distinct offsets and defines
the ninth by (1). It accepts the block only when the ninth lies in `0,...,80`,
is distinct, is outside the R45 catalog, and has not already been accepted.
The frozen sampling receipt is

```text
draw attempts=1,398,
valid congruent candidates=1,000,
catalog rejections=0,
duplicate rejections=0,
accepted external blocks=1,000.                 (2)
```

For every accepted block, the exact kernel section in `H^0(E,O(9O))` has
exactly those nine selected points as its zero set and exactly nine zeros on
the full order-103 subgroup. All 1,000 section vectors are distinct.

## Seeds and slices

Each external denominator is scored against all 807 catalog sections with
disjoint interval zero sets. Its highest-collision catalog partner is retained.
Every one of the 1,000 seeds has two complete nine-point fibers and a
rank-seven nontrivial collision basis.

R49 then drops each of the seven basis rows and exhausts the corresponding 194
projective pencils. Exact totals are

```text
generated slice instances=1,000*7*194=1,358,000,
unique pencil keys including seeds=1,352,000,
duplicate slice instances=6,000,
subgroup-basepoint-free candidates=1,289,607,
subgroup-basepoint rejections=61,393.            (3)
```

Every unique outcome is accounted for.

## Result

Seed and best-per-denominator scores all retain exactly two complete fibers.
The seed collision-pair counts range from 88 to 103; the post-slice best counts
range from 89 to 103. In detail,

```text
denominators with strict secondary improvement=91,
maximum collision-pair gain=8,
global best score=(2,103,9),
global complete coverage=18.                    (4)
```

No candidate has three complete selected fibers. The required score remains

```text
(9, at least 324, 9), complete coverage 81.     (5)
```

The external denominators are not descendants of the R45 trade search and can
have integer offset sums different from 360 while satisfying (1). Their flat
complete-fiber count therefore extends the negative evidence beyond that
catalog's combinatorial construction.

## Scope

The 1,000 blocks are a deterministic finite sample from roughly 2.5 billion
eligible nine-subsets, not an exhaustive block catalog or a probabilistic
theorem. Each denominator is searched only through seven one-row-dropped
slices around one catalog seed. Subgroup basepoint checks do not certify
geometric basepoint freeness over the algebraic closure.

R49 supplies no asymptotic pencil family, fresh-target locator, R10 queried
coefficients, relation-rank campaign, factor-base logarithms, or scalar-blind
descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_EXTERNAL_DENOMINATOR_SAMPLE_NEGATIVE_GATE
SEED1553_ACCEPTS_1000_DISTINCT_OUTSIDE_CATALOG_EQUAL_SUM_BLOCKS
ALL1000_EXACT_SECTION_ZERO_SETS_AND_RANK_SEVEN_SEEDS_CHECKED
SEVEN_194_PENCIL_SLICES_EXHAUSTED_PER_EXTERNAL_DENOMINATOR
1358000_INSTANCES_1289607_SCORED_CANDIDATES
91_DENOMINATORS_IMPROVE_SECONDARY_SCORE_MAX_GAIN8
NO_CANDIDATE_HAS_THREE_COMPLETE_NINE_POINT_FIBERS
GLOBAL_BEST_TWO_FIBERS_103_COLLISION_PAIRS
REQUIRED_NINE_FIBERS_324_COLLISION_PAIRS_ABSENT
UNSAMPLED_BLOCKS_AND_OUTSIDE_SLICE_PENCILS_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: complete the direct pairwise pencil audit for the
enlarged 1,807-section sample. R46 already covers catalog-catalog pairs and R49
seed selection scores every external-catalog pair. Exhaust every disjoint pair
among the 1,000 external sections, deduplicate by Plucker coordinates, evaluate
all 103 subgroup points, and test whether any such pencil has a third complete
selected fiber.
