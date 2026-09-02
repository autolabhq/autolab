# P1553 full-curve primitive product-pencil gate R58

## Classification

- Owner: existing P1553/IDEA-195 primitive split-pencil frontier; no new idea
  ID.
- Evidence: exact deterministic full-rational-curve search and divisor,
  section-rank, factor-class, coset, and public-projection replay; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_POSITIVE_PRIMITIVE_TOY_GATE`.
- Labels: `toy`, `exact-positive-construction-seed`, `deterministic-search`,
  `non-run`, `model-bound`, `novelty-unverified`.
- Cryptanalytic result: R58 finds the first primitive coprime degree-nine
  product pencil in this chain. Two marked nine-point subgroup products and a
  third nine-point rational fiber are pairwise disjoint, lie in one rank-two
  section pencil, and each splits into the same three degree-three Picard
  classes. The third fiber has three prime-subgroup points and six points in
  the two-torsion coset; public multiplication by 104 projects every root to a
  subgroup atom without a DLP. This is one finite toy, not an asymptotic
  relation algorithm, a Shoup-bound improvement, or an ECDLP breakthrough.

## Inputs

| Input | SHA-256 |
|---|---|
| R57 maximal-class orbit gate | `6902f76953535858c85b8a5b2bf6da5418f60749ce7799934768056bc1d0c609` |
| R57 orbit report | `05f6bb2ded46cd3ba45664c0bda7f96000f285c4788f6badc61993d9c3fdec33` |

## Search domain

The curve is

```text
E/F_193: y^2=x^3+2x+3,
#E(F_193)=206=2*103,
H=<G>, |H|=103,
T_2=(192,0).                                     (1)
```

R58 leaves the 81-point path and uses all 103 points of `H` to construct
degree-three factors with class sums

```text
(46,54,96), total 93 mod 103.                    (2)
```

Each class has exactly 1,717 distinct reduced subgroup triples. A product
chooses one triple from each class and rejects repeated points. Two products
with disjoint nine-point divisors generate a basepoint-free degree-nine
pencil. R58 evaluates every rational curve point and looks for a third finite
fiber with nine distinct roots that partitions back into the three classes.

## Deterministic witness

With seed 1559, the first valid witness occurs after

```text
attempts=25,996,
accepted coprime pencils=11,104,
complete rational nine-point third fibers seen=6. (3)
```

The generator zero blocks in subgroup scalar labels are

```text
A={9,20,23,56,69,73,79,87,89},
B={14,25,44,48,49,62,70,91,102}.                 (4)
```

Their marked degree-three factors are

```text
A: (20,56,73), (9,69,79), (23,87,89),
B: (14,44,91), (25,62,70), (48,49,102).          (5)
```

At line parameter `t=33`, `A+33B` has the nine rational zeros whose canonical
curve-point indices are

```text
{9,33,48,82,121,142,147,152,156}.                (6)
```

They partition into class factors

```text
(33,82,152), (48,142,156), (9,121,147).          (7)
```

The corresponding translated plane-line equations are

```text
(1,62,151), (1,51,34), (1,184,86).               (8)
```

The divisor-kernel section from (6) equals the normalized vector `A+33B`.
The three degree-nine sections have exact rank two, all three zero divisors
are pairwise disjoint, the common base divisor has degree zero, and the
effective map degree is nine. This is a primitive pencil, unlike every
R51-R57 positive line.

## Cofactor structure

The third fiber contains

```text
3 points in H,
6 points in H+T_2.                                (9)
```

Each degree-three factor in (7) contains exactly two outside-coset points, so
its two-torsion contribution cancels. Since

```text
104=1 mod 103,
104=0 mod 2,                                     (10)
```

the public map `[104]` is the projector `E(F_193)->H`. R58 applies it to all
nine roots and verifies their subgroup points against exhaustive toy labels.
The point projection uses only public scalar multiplication; the scalar logs
of the projected atoms remain unknown factor-base variables.

## Scope

R58 proves finite existence of a primitive product pencil after the full
rational curve replaces the prime-subgroup-only fiber requirement. It does
not supply an asymptotic family, low-boundary schedule, restricted factor-base
density, target-local pencil locator, relation-rank campaign, factor-base
logs, or scalar-blind descent. The deterministic search scans all 193 field
parameters of each accepted pencil, and that cost must be charged.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_POSITIVE_PRIMITIVE_TOY_GATE
FIRST_PRIMITIVE_COPRIME_PRODUCT_PENCIL_IN_R44_R58_CHAIN
DEGREE_NINE_COMMON_BASE_DIVISOR_ZERO
THREE_PAIRWISE_DISJOINT_NINE_POINT_RATIONAL_FIBERS
ALL_THREE_FIBERS_SPLIT_IN_CLASSES_46_54_96
SEED1559_FIRST_WITNESS_ACCEPTED_PENCIL_11104
THIRD_FIBER_THREE_H_SIX_H_PLUS_T2
PUBLIC_104_PROJECTOR_RECOVERS_SUBGROUP_POINTS
SCALAR_LOGS_REMAIN_UNKNOWN_FACTOR_BASE_VARIABLES
NO_ASYMPTOTIC_FAMILY_TARGET_LOCATOR_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: prove and exhaust the public cofactor projection on
all 206 curve points, retain projection collisions with multiplicity, and
charge the split-fiber plus ordered class-partition success probability before
claiming any relation-rate gain.
