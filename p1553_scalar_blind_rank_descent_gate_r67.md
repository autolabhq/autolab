# P1553 scalar-blind rank and descent gate R67

## Classification

- Owner: existing P1553/IDEA-195 degree-six product-section frontier; no new
  idea ID.
- Evidence: exact public fixed-sum catalogs, exact relation matrices over the
  prime subgroup order, a caller-supplied target action, public scalar
  multiplication checks, direct quotient controls, and exhaustive FFE
  incidence controls.
- Status: `DRAFT_REVIEW_REQUIRED_FIXED_SUM_REDUNDANCY_GATE`.
- Labels: `toy`, `exact-relation-rank`, `scalar-blind-target-action`,
  `model-bound`, `novelty-unverified`.
- Cryptanalytic result: the toy target log is recovered without consuming any
  scalar label, but the fixed-sum group equations solve the system before the
  product, sextic, factor-lift, or trisecant machinery is used. Under the
  uniform generic-group model, the resulting precomputation costs
  `Theta(N)` and online target lookup costs `Theta(sqrt(N))`. This is not a
  Shoup-bound improvement or ECDLP breakthrough.

## Inputs

| Input | SHA-256 |
|---|---|
| R66 kernel generator gate | `7032c45a1f49249951bcad021ff2ac6b7510be84b102e3ad7348fce586a3bf58` |
| R66 kernel generator report | `753c16d1f15e444517ac7bc503fd9f888d8219e724e1dced6c384ed6b80a4081` |
| R65 exact incidence report | `30d686228a271cbdf4a61e2bf4b7d553bc297d7e6203955863b15b7ff98fdb8a` |
| R61 multiplication/locator report | `f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64` |

The executable also pins all eleven loaded P1553 source dependencies.

## Scalar-blind factor base

The target-independent core base contains the identity, the public generator,
and 22 points produced by a public SHA-256 counter-to-x map, Tonelli-Shanks
square roots, and cofactor-two clearing. It accepts 22 points after 48 counter
attempts. No subgroup table or scalar labels are materialized.

For class sums `[46]G` and `[54]G`, pair-complement lookup finds 19 and 17
unordered triples. The 36 public equations

```text
log(P_i) + log(P_j) + log(P_k) = 46 or 54 mod 103
```

have coefficient rank 24. They uniquely recover all 24 core logs, which are
then checked by public scalar multiplication.

## Fresh target action

The default caller-supplied target is `Q=(137,171)`; its scalar label is not an
input. For each core point and class sum, one complement lookup tests

```text
P_i + P_j = [c]G - Q.                              (1)
```

Exactly 48 hash lookups find five pairs for each class. All ten equations give

```text
log_G(Q) = 53 mod 103,                              (2)
```

and `[53]G=Q` verifies publicly. The 36 precomputed core rows plus these target
rows have rank 25. The target step is therefore amortized and scalar-blind on
this toy.

## FFE redundancy controls

The same core base produces 4,162 smooth degree-six product sections. Taking
ratios against one reference section already gives 4,161 principal-divisor
rows of rank 22 and nullity two; the nullspace is the affine span of the
constant vector and the true log vector. The anchors orient it to exactly the
logs already recovered by the fixed-sum system.

The exhaustive trisecant control fingerprints 8,659,041 product pairs and
finds 16,171 lines. It has no primitive line on the core base, yet its
nonprimitive rows again have rank 22 and recover the same logs.

After adjoining the target, 6,710 product sections give direct quotient rank
23. The exhaustive 22,508,695-pair incidence scan gives the same rank and the
same target log. Thus:

```text
fixed-sum equations -> unique logs,
direct product quotients -> no new log information,
FFE trisecants -> no new log information.           (3)
```

Collinearity is useful only for locating another product section from a
pencil. It is unnecessary once two smooth product sections were constructed,
and even those products are unnecessary once their fixed-sum factor triples
are known.

## Charged generic cost

For a scalar-blind base of size `B`, a constant number `K` of public class sums
has about

```text
K*B^3/N                                             (4)
```

fixed-sum triples under the uniform model. Obtaining `B` independent rows
requires `B^2` on the order of `N`, so `B` is at least on the order of
`sqrt(N)`. Pair-complement precomputation costs `Theta(K*B^2)=Theta(N)`.
Sparse linear algebra at this scale is also about `B^2=Theta(N)` absent new
matrix structure.

A fresh target has about `K*B^2/N` pairs satisfying (1). Constant success again
requires `B` on the order of `sqrt(N)`, and the complement lookup costs
`Theta(K*B)=Theta(sqrt(N))`. The online phase matches rho; the precomputation is
worse. The product and incidence scans are redundant and more expensive.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_FIXED_SUM_REDUNDANCY_GATE
SCALAR_BLIND_B24_CORE_36_FIXED_SUM_ROWS_RANK24
FRESH_TARGET_48_LOOKUPS_TEN_ROWS_LOG53_PUBLICLY_VERIFIED
DIRECT_PRODUCT_QUOTIENT_AND_FFE_INCIDENCE_LAYERS_ADD_NO_LOG_INFORMATION
UNIFORM_RANK_THRESHOLD_B_ON_ORDER_SQRT_N
THETA_N_PRECOMPUTATION_THETA_SQRT_N_ONLINE_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: prove or refute a public scalar-blind base/class
family with superuniform fixed-sum rank and subquadratic enumeration after
semantic deduplication against the complete ledger.
