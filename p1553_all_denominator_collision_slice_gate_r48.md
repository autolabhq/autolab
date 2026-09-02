# P1553 all-denominator collision-slice gate R48

## Classification

- Owner: existing P1553/IDEA-195 primitive degree-nine pencil frontier; no new
  idea ID.
- Evidence: exact exhaustive declared-slice sweep for every retained R46
  denominator; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_ALL_CATALOG_DENOMINATOR_SLICE_NEGATIVE_GATE`.
- Labels: `toy`, `exact`, `exhaustive-declared-slices`, `non-run`,
  `model-bound`, `novelty-unverified`.
- Cryptanalytic result: for each of all 807 retained catalog sections, R48
  chooses its highest-collision disjoint catalog partner and exhausts the seven
  194-pencil slices obtained by dropping one row from the seed's rank-seven
  collision basis. The sweep generates 1,095,906 instances and scores
  1,040,723 subgroup-basepoint-free candidates. Seventy-two denominators
  improve their secondary collision score, by at most 11, but none gains a
  third complete nine-point fiber. Nine are required. Pencils outside the
  declared slices and denominators outside the catalog remain open. No
  Shoup-bound improvement or ECDLP breakthrough follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R42 single-interval pencil DLP gate | `8a8de8726e4fc9542f97cf7e1ba4fc54596307cc544da0d0b18b588553022dae` |
| R46 catalog pencil-fiber gate | `6302cf24ff82dfc1d96c0cc21c0980c862a3218c0100ffb29905a35a8b63e2e4` |
| R47 fixed-denominator collision gate | `ea56391b5ab951bea92390f3ae490c89d68885f29b02a8433adc5efba80415e3` |
| R47 bundle hash list | `7407fccf1684d4cbe11d7f9b16111516233523cc7e4864dd549a3bd96e2f69b3` |
| R47 staging receipt | `1488ef90bf65eecc857ed63f73d0409247cc19db62ed88abb7b75e9d41c64162` |

## Seed selection

R48 reconstructs all 807 retained block sections and all 142,286 disjoint
catalog pairs. For each section `s_0`, it chooses the disjoint partner `s_1`
maximizing the exact lexicographic score

```text
(complete nine-point fibers,
 selected collision pairs,
 maximum selected fiber multiplicity).          (1)
```

Every selected seed has two complete fibers and a rank-seven basis among its
nontrivial linear collision equations. The seed collision-pair counts range
from 90 to 106.

## Exact slice sweep

For each denominator, R48 applies the R47 identity: dropping one of the seven
independent collision rows leaves a projective line of 194 pencils. It
exhausts all

```text
807*7*194=1,095,906                             (2)
```

slice instances. Deduplication within each denominator gives 1,091,064 unique
pencil keys including the 807 seeds. Of the nonseed outcomes,

```text
subgroup-basepoint-free candidates=1,040,723,
subgroup-basepoint rejections=49,534,
duplicate slice instances=4,842.                (3)
```

All candidate outcomes and duplicates are exactly accounted for.

## Result

Every one of the 807 best-per-denominator scores still has exactly two
complete nine-point fibers and maximum multiplicity nine. The collision-pair
distribution ranges from 90 to 106. Relative to its own seed,

```text
denominators with a strict score improvement=72,
denominators unchanged=735,
maximum collision-pair gain=11.                 (4)
```

The global best remains the R46 seed score

```text
(2,106,9), complete coverage 18.                (5)
```

No generated candidate has three complete fibers. The required interval
pencil score is

```text
(9, at least 324, 9), complete coverage 81.     (6)
```

The positive secondary gains in (4) show that the one-row-dropped slices do
leave the catalog pencils and can improve collision structure. Their complete
fiber count nevertheless remains flat across every catalog denominator.

## Scope

The sweep is exhaustive only for seven declared projective slices around one
catalog seed per denominator. It does not exhaust each denominator's
projective seven-space. More importantly, all denominators are still drawn
from the budgeted 807-section R45 catalog. Subgroup basepoint checks do not
certify geometric basepoint freeness over the algebraic closure.

R48 supplies no asymptotic pencil family, fresh-target locator, R10 queried
coefficients, relation-rank campaign, factor-base logarithms, or scalar-blind
descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_ALL_CATALOG_DENOMINATOR_SLICE_NEGATIVE_GATE
ALL807_CATALOG_DENOMINATORS_HAVE_RANK_SEVEN_COLLISION_SEEDS
SEVEN_194_PENCIL_SLICES_EXHAUSTED_PER_DENOMINATOR
1095906_GENERATED_INSTANCES_1040723_SCORED_CANDIDATES
72_DENOMINATORS_IMPROVE_SECONDARY_SCORE_MAX_GAIN11
NO_CANDIDATE_HAS_THREE_COMPLETE_NINE_POINT_FIBERS
GLOBAL_BEST_REMAINS_TWO_FIBERS_106_COLLISION_PAIRS
REQUIRED_NINE_FIBERS_324_COLLISION_PAIRS_ABSENT
OUTSIDE_SLICE_AND_OUTSIDE_CATALOG_PENCILS_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: expand the denominator set itself. With deterministic
seed 1553, generate 1,000 distinct nine-subsets of offsets `0,...,80` outside
the R45 catalog whose offset sum is `360 mod 103`; construct their exact
degree-nine sections, choose the highest-collision disjoint catalog partner for
each, and exhaust the same seven 194-pencil slices. Preserve the sampling
attempt count and reject any section whose subgroup zero set is not exactly its
named block.
