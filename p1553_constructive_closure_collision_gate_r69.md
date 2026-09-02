# P1553 constructive closure-collision gate R69

## Classification

- Owner: existing P1553/IDEA-195 constructive product-section frontier; no new
  idea ID.
- Evidence: a general rank/nullity identity, a scalar-blind prime-order toy
  replay, exact recovered-log verification, fresh-target descent, and a
  uniform collision-cost gate.
- Status: `DRAFT_REVIEW_REQUIRED_CONSTRUCTIVE_CLOSURE_COLLISION_GATE`.
- Labels: `exact-rank-nullity-control`, `scalar-blind-toy-replay`,
  `model-bound-collision-cost`, `novelty-unverified`.
- Cryptanalytic result: closure collisions recover all toy factor logs and a
  fresh target log without consuming scalar labels. Fresh constructive
  residuals themselves reduce no unresolved-log nullity. The complete source
  takes 66 pair proposals and target descent takes 60 lookups, versus a toy rho
  baseline of 13 before omitted costs. This is not a Shoup-bound improvement or
  ECDLP breakthrough.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R67 scalar-blind source | `60cf8dc258f2ea097a8581bd95b8002829be492a512cf336c270abda6876129e` |
| R67 exact report | `fc47d3d509597fe5b7a0928c5fcffc4cd6c60679ea7efc7d531f4b24b28f143b` |
| R68 information-conservation source | `b8aa38732e577d10c7e7326f01ed5d3a688627963f7d0b98dee3f6a2078deddc` |
| R68 exact report | `5c064a36630c052093ebb7bd3b4429d37f836982ba097eacbad2f2ebe1bb6b51` |

## Rank identity

Suppose one constructive step adds `f` fresh atom variables and at most `t`
independent equations. If the old matrix has `n` columns and rank `r`, the new
matrix has `n+f` columns and rank at most `r+t`. Therefore

```text
new nullity - old nullity >= f-t.                 (1)
```

A line intersection, summation-polynomial residual, or factored fiber that
introduces one fresh residual point together with one relation has `f=t=1`.
It can preserve nullity, but cannot reduce it. A rank-reducing event requires
the residual to be an atom already supplied independently or reached by a
different constructive path.

R68 showed that product quotients add no information beyond their factor rows.
Equation (1) locates the remaining information event more precisely: a new
factor row helps only when it closes on already represented atoms.

## Exact toy replay

The source freezes the first 12 points of the R67 public SHA-256 base,
including the identity and generator anchors, before observing outcomes. It
then evaluates all 66 unordered seed pairs and emits

```text
P_i + P_j + R_ij = O.                              (2)
```

The pair stream introduces 48 distinct residual atoms. Those 48 rows have rank
48 on 60 final atoms, leaving the original nullity 12 exactly. The remaining
18 proposals are closure collisions:

```text
independent collision rows: 11,
dependent collision rows:    7,
final relation rank:         59,
final nullity:                1.                   (3)
```

The identity and generator orient the one-dimensional nullspace. All 60
recovered logs verify by public scalar multiplication.

The caller-supplied R67 target `(137,171)` is absent from precomputation. A
linear complement pass over the 60 recovered atoms finds 18 decompositions,
all yielding the unique verified target log `53 mod 103`.

## Charged boundary

For the order-103 toy,

```text
ceil(sqrt(pi*N/2)) = 13,
source pair proposals = 66 = 5.077 rho,
online target lookups = 60 = 4.615 rho.            (4)
```

The source is exactly the complete initial pair stream, so it saves zero work
against direct pair enumeration. Hash-to-curve arithmetic, row reduction,
nullspace orientation, and public verification are omitted from (4); charging
them only worsens the comparison.

For `B` uniform public seeds and `M` residual proposals, expected seed hits are
`Theta(MB/N)` and repeated-residual collisions are `Theta(M^2/N)`. Obtaining
`Theta(B)` collision rows requires

```text
M = Omega(sqrt(B*N)).                              (5)
```

For every growing `B=N^beta`, (5) has exponent `(1+beta)/2 > 1/2`. The
complete pair stream has `M=Theta(B^2)` and reaches `B` expected collisions
only at `B=Omega(N^(1/3))`, where work is `Omega(N^(2/3))`.

This is a uniform-model obstruction, not an unrestricted lower bound. A valid
escape must exhibit a public coordinate structure that concentrates
independent closure collisions and locates them without materializing the pair
graph.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_CONSTRUCTIVE_CLOSURE_COLLISION_GATE
FRESH_RESIDUAL_ROW_PRESERVES_UNRESOLVED_LOG_NULLITY
ONLY_INDEPENDENT_CLOSURE_COLLISIONS_REDUCE_NULLITY
TOY_48_FRESH_ROWS_PRESERVE_NULLITY12
TOY_11_INDEPENDENT_COLLISIONS_REDUCE_TO_NULLITY1
ALL_60_FACTOR_LOGS_AND_FRESH_TARGET_LOG53_VERIFY
SOURCE_66_OVER_RHO13_ONLINE_60_OVER_RHO13
UNIFORM_B_COLLISIONS_REQUIRE_SQRT_BN_PROPOSALS
NO_NEW_IDEA
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: preregister one public coordinate-defined
closure-collision locator acting before pair materialization, then require
superuniform independent collision rank, total source and descent below rho,
and unchanged transfer across four generic-prime target families.
