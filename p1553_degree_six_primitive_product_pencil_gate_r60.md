# P1553 degree-six primitive product-pencil gate R60

## Classification

- Owner: existing P1553/IDEA-195 primitive product-pencil frontier; no new
  idea ID.
- Evidence: exact deterministic degree-six full-curve search with divisor,
  section-rank, class-factor, coset, and public-projection replay; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_POSITIVE_DEGREE_SIX_TOY_GATE`.
- Labels: `toy`, `exact-positive-construction-seed`, `deterministic-search`,
  `non-run`, `model-bound`, `novelty-unverified`.
- Cryptanalytic result: R60 confirms R59's constant-pencil-count prediction.
  Seed 1560 finds a primitive coprime degree-six product pencil after 18
  accepted pencils, close to the heuristic 19.31. Its third six-point rational
  fiber splits into the same two degree-three classes and projects publicly to
  six prime-subgroup atoms. This does not charge the `Theta(p)` parameter scan
  per pencil or restricted factor-base smoothness, so it is not a sub-rho
  algorithm, Shoup-bound improvement, or ECDLP breakthrough.

## Inputs

| Input | SHA-256 |
|---|---|
| R59 projection/rate gate | `d6839fb19226c106db20e441998ff56ea1b306ba6a4155b940f998ebafe1c3d1` |
| R59 exact/heuristic report | `2b3931459c521d8e68b20bec1dd58ec9dfa5688efe34859f59386ffe6cb3eea7` |

## Search and witness

R60 fixes class sums `(46,54)`. Each class has 1,717 reduced triples in the
order-103 subgroup. A product therefore has degree six and total class 100.
Two marked products with disjoint zeros generate a basepoint-free pencil.

With seed 1560, the first class-split third fiber occurs at

```text
attempts=24,
accepted coprime pencils=18,
complete rational six-point third fibers seen=2. (1)
```

The generator scalar blocks are

```text
A={32,38,76,85,87,91},
B={19,20,57,62,68,80}.                            (2)
```

with marked factors

```text
A: (76,85,91), (32,38,87),
B: (19,62,68), (20,57,80).                        (3)
```

At line parameter `t=29`, the third fiber has rational curve-point indices

```text
{47,103,107,119,134,150}                         (4)
```

and class factors

```text
(47,103,107), (119,134,150).                      (5)
```

Their translated line equations are `(1,81,24)` and `(1,136,11)`. The first
factor has two points in `H+T_2`; the second lies wholly in `H`, so both have
even torsion parity. Public `[104]` projection recovers six subgroup atoms.

The divisor-kernel section from (4) equals normalized `A+29B`; all three
degree-six divisors are pairwise disjoint, the three sections have rank two,
the common base divisor has degree zero, and the effective map degree is six.

## Cost boundary

The observed accepted-pencil count matches R59's fixed-degree heuristic:

```text
q_2 ~= (p-1)/(6^2*N),
1/q_2=19.31 at p=193,N=103.                       (6)
```

But the current implementation obtains (4) by evaluating all 206 rational
curve points and implicitly scans all 193 field parameters. Therefore its
charged work per relation is still `Theta(p)=Theta(N)`, before:

- restricting the output roots to a factor base of size `B=o(N)`;
- preserving multiplicities after cofactor projection;
- accumulating enough independent relations;
- solving factor-base logs; or
- descending a fresh target.

There is a concrete algebraic locator question. For fixed degree-three line
bundles `L_1,L_2`, the product image of

```text
P(H^0(L_1)) x P(H^0(L_2)) -> P(H^0(L_1 L_2))    (7)
```

has dimension four in `P^5`, hence is expected to be a hypersurface. If its
defining equation can be constructed, restricting it to the pencil line gives
a constant-degree univariate polynomial whose roots include all product
sections, replacing the full parameter scan. R60 does not yet construct or
degree-bound that equation in the required model.

## Scope

R60 is one positive finite toy. It establishes neither a generic-prime
hypersurface locator nor useful density in a restricted point factor base.
The source factors in (3) were generated from known scalar labels, while the
output logs remain unknown variables.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_POSITIVE_DEGREE_SIX_TOY_GATE
SEED1560_FIRST_WITNESS_ACCEPTED_PENCIL18_PREDICTED19_31
DEGREE_SIX_COMMON_BASE_DIVISOR_ZERO
THREE_PAIRWISE_DISJOINT_SIX_POINT_RATIONAL_FIBERS
ALL_FIBERS_SPLIT_IN_CLASSES_46_54
PUBLIC_104_PROJECTION_RECOVERS_SIX_SUBGROUP_ATOMS
CURRENT_PER_PENCIL_SCAN_THETA_P
RESTRICTED_FACTOR_BASE_DENSITY_UNCHARGED
PRODUCT_IMAGE_HYPERSURFACE_LOCATOR_ROUTED_NOT_CONSTRUCTED
NO_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: construct the fixed-class degree-six product-image
hypersurface (or an equivalent elimination polynomial), restrict it to the
R60 pencil, and verify that its constant-degree roots recover every product
fiber without scanning `F_p`; then charge restricted factor-base density.
