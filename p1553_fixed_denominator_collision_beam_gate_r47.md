# P1553 fixed-denominator collision-beam gate R47

## Classification

- Owner: existing P1553/IDEA-195 primitive degree-nine pencil frontier; no new
  idea ID.
- Evidence: exact deterministic collision-hyperplane slice search on the R38
  toy; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_FINITE_FIXED_DENOMINATOR_COLLISION_NEGATIVE_GATE`.
- Labels: `toy`, `exact`, `deterministic-finite-search`, `non-run`,
  `model-bound`, `novelty-unverified`.
- Cryptanalytic result: fixing either section in the strongest R46 pair as
  denominator, an eight-layer width-32 beam searches exact projective slices
  of numerator-section space. It generates 611,100 slice instances and scores
  576,607 layer-unique subgroup-basepoint-free candidates. Neither orientation
  improves the seed's two complete nine-point fibers or 106 selected collision
  pairs; nine fibers and at least 324 collision pairs are required. Other
  denominators and pencils outside the beam remain open. No Shoup-bound
  improvement or ECDLP breakthrough follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R42 single-interval pencil DLP gate | `8a8de8726e4fc9542f97cf7e1ba4fc54596307cc544da0d0b18b588553022dae` |
| R45 rank-minor-guided partition gate | `61f894cbc7a135c811202e91ced1ddb63c665b2ec3a08c78475e96fd01dfe3d6` |
| R46 catalog pencil-fiber gate | `6302cf24ff82dfc1d96c0cc21c0980c862a3218c0100ffb29905a35a8b63e2e4` |
| R46 bundle hash list | `ce3d817daa4c16a3025ab0b7236453bdc1313fe70b0df18caa82ffb5046042a5` |
| R46 staging receipt | `739e33ade7027c9d07e6f738c46bcf76c3dc364bfc75a1ee35050d17b1f96f56` |

## Linear collision arrangement

Fix a nonzero denominator section `s_0`. Adding a multiple of `s_0` to a
numerator `s_1` only translates the target coordinate, so numerator pencils
form projective seven-space

```text
P(H^0(E,O(9O))/<s_0>) = P^7(F_193).             (1)
```

For selected subgroup points `P,Q`, the condition that they share one pencil
value is

```text
s_0(P)s_1(Q)-s_0(Q)s_1(P)=0.                    (2)
```

Equation (2) is linear in the nine coefficients of `s_1` and automatically
annihilates `s_0`. Seven independent collision equations therefore have a
two-dimensional kernel in the full section space: exactly one isolated pencil
containing `s_0`.

Dropping one equation leaves six independent rows. Their kernel has dimension
three and contains `s_0`; modulo the denominator it is two-dimensional. Hence
all possible replacement pencils in that slice form

```text
P^1(F_193), with 193+1=194 exact candidates.     (3)
```

R47 exhausts (3) for each of the seven dropped rows at every beam state. This
is stronger and more reproducible than sampling replacement equations.

## Frozen search

The two orientations of the strongest R46 pair are searched independently.
Each seed has selected score

```text
(complete nine-point fibers, collision pairs, maximum multiplicity)
=(2,106,9).                                      (4)
```

For each orientation, R47 uses eight layers and retains the best 32 isolated
pencils after every layer. The exact totals over both orientations are

```text
generated projective-line instances=611,100,
layer-unique candidates=576,607,
subgroup-basepoint rejections=27,376,
retained isolated states including seeds=514.   (5)
```

Every one of the `2*8*32=512` selected nonseed states has collision-basis rank
seven, so no beam slot is lost to an underdetermined collision family.

The best selected layer scores fluctuate between 87 and 103 collision pairs
in the first orientation and between 87 and 100 in the second. No selected
state exceeds the seed's 106. More importantly, no scored slice candidate has
three complete nine-point fibers: a three-fiber candidate would outrank every
two-fiber state and enter the beam under the primary score.

The required interval pencil has

```text
complete fibers=9,
complete coverage=81,
within-fiber collision pairs at least 9*C(9,2)=324. (6)
```

R47 remains at `(2,106,9)` and complete coverage 18.

## Scope

Only the two denominator sections in the strongest R46 pair are searched. The
beam retains 32 isolated pencils per layer, so it is not exhaustive in either
projective seven-space. Subgroup basepoint checks also do not certify geometric
basepoint freeness over the algebraic closure; any survivor would need that
additional divisor check.

R47 supplies no asymptotic pencil family, fresh-target locator, R10 queried
coefficients, relation-rank campaign, factor-base logarithms, or scalar-blind
descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_FINITE_FIXED_DENOMINATOR_COLLISION_NEGATIVE_GATE
COLLISION_EQUATIONS_LINEAR_IN_NUMERATOR_MODULO_FIXED_DENOMINATOR
SIX_ROW_SLICE_EXHAUSTS_194_PROJECTIVE_PENCILS
TWO_DENOMINATOR_ORIENTATIONS_EIGHT_LAYERS_WIDTH32
611100_SLICE_INSTANCES_576607_LAYER_UNIQUE_CANDIDATES
ALL512_SELECTED_NONSEED_COLLISION_BASES_HAVE_RANK_SEVEN
GLOBAL_BEST_REMAINS_TWO_COMPLETE_FIBERS_106_COLLISION_PAIRS
REQUIRED_NINE_FIBERS_324_COLLISION_PAIRS_ABSENT
OTHER_DENOMINATORS_AND_OUTSIDE_BEAM_PENCILS_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: broaden before deepening. For each of all 807 R46
catalog sections as denominator, choose its highest-collision disjoint catalog
partner, extract a rank-seven collision basis, drop each basis row in turn,
and exhaust the resulting seven 194-pencil projective slices. Retain the exact
best fiber score per denominator and require any three-fiber improvement to
pass a full divisor/basepoint check before further search.
