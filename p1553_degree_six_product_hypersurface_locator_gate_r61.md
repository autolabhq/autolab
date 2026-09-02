# P1553 degree-six product-hypersurface locator gate R61

## Classification

- Owner: existing P1553/IDEA-195 primitive product-pencil frontier; no new
  idea ID.
- Evidence: exact finite-field multiplication tensor, sampled implicitization,
  exact pullback identity certificate, and deterministic R60 pencil replay; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_POSITIVE_ALGEBRAIC_LOCATOR_TOY_GATE`.
- Labels: `toy`, `exact-algebraic-locator`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: R61 removes the linear field scan from parameter
  location in the R60 toy. It does not yet lift a hypersurface root to its two
  factor sections without a scan, charge factor-base density or source logs,
  prove relation rank, or descend a target. It is not a Shoup-bound
  improvement or ECDLP breakthrough.

## Inputs

| Input | SHA-256 |
|---|---|
| R60 degree-six gate | `89aa1c7cbd31d898c6bd06a8950b46f646898912e1ba6179023d137ddc251137` |
| R60 deterministic report | `b15ac1b6cb5df18b4fb5fdcad4d0982fb8c122777dfc6009cbd5318fc28dd364` |

## Exact multiplication image

For the fixed class sums `(46,54)` on `E/F_193`, both translated degree-three
section spaces have dimension three. R61 calibrates their nonconstant
line-bundle trivialization and constructs the exact multiplication map

```text
mu: F_193^3 tensor F_193^3 -> F_193^6.             (1)
```

The resulting `6 x 9` matrix has rank six and kernel dimension three. All 256
deterministic replay products agree exactly with the tensor evaluation. The
matrix digest is

```text
95187a49addbd3cb9cfa24e480b4b30b88fc1a928bd0a4ceab6acc6dd92ef871. (2)
```

No homogeneous equation appears in sampled degrees one through five. In
degree six, the 494-by-462 evaluation matrix has rank 461 and a unique kernel
generator. Its normalized equation digest is

```text
8cbdf33b001fa92bd86d04535d290c964a9296979db64c7967268d842b423ea5. (3)
```

This is not accepted from sampling alone. Pulling the sextic back through
`mu(x tensor y)` gives a bidegree `(6,6)` ternary form. R61 evaluates it on
two independently constructed 28-point unisolvent sets. All `28^2=784`
values vanish, while both ternary evaluation matrices have full rank 28.
Therefore every one of the 784 pullback coefficients is exactly zero over
`F_193`; (3) is a certified equation of the product image.

## R60 pencil restriction

Restricting (3) to the normalized R60 pencil `A+tB` gives

```text
t*(121 + 150*t + 11*t^2 + 104*t^3 + 179*t^4).     (4)
```

The missing sextic leading term records the product point `B` at infinity.
Fixed-degree finite-field factorization, without a field scan, gives

```text
t * (t + 160) * (t + 164) * (t^2 + 27*t + 151),  (5)
```

up to unit 179. Thus the finite rational roots are `0,29,33`; the R60 witness
parameter `29` is recovered exactly. A full 206-point curve pass is retained
only as a verifier: roots 0 and 29 have six rational zeros in the bound
classes, root 33 has none, and the quadratic contributes no `F_193` root.

## Cost boundary

Conditioned on a cached multiplication map, sextic, and pencil sections, the
online parameter locator is fixed-degree polynomial evaluation and
factorization. It has no `Theta(p)` parameter scan; a standard fixed-degree
finite-field factorization uses expected `O(log p)` field operations, with
field-operation bit costs charged separately.

That qualifier is essential. The current R61 toy run constructs the cache by
enumerating two `C(N,3)` fixed-class catalogs, calibrating over the subgroup,
and replaying 256 products with subgroup-wide interpolation. The catalog pass
alone examines `Theta(N^3)` triples. R61 neither charges nor amortizes this
preprocessing and therefore does not establish an end-to-end scan-free
relation algorithm.

This is only a parameter-membership locator. The artifact does not yet invert
the projected Segre map at a root, recover the two factor line sections, or
factor their cubic curve intersections without enumerating field elements.
It also does not charge:

- source-factor selection or factor-base density;
- source scalar logarithms after public cofactor projection;
- relation collection and independent rank;
- factor-base linear algebra; or
- fresh-target descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_POSITIVE_ALGEBRAIC_LOCATOR_TOY_GATE
FIXED_CLASS_MULTIPLICATION_RANK6_KERNEL3
PRODUCT_IMAGE_UNIQUE_SEXTIC_AFTER_DEGREES1_TO5_ZERO
EXACT_784_COEFFICIENT_PULLBACK_IDENTITY_CERTIFICATE
R60_RESTRICTION_ROOTS_0_29_33_PLUS_INFINITY_AND_QUADRATIC
PARAMETER_SCAN_THETA_P_REMOVED
ONLINE_CLAIM_CONDITIONAL_ON_UNCHARGED_THETA_N3_TOY_PREPROCESSING
FACTOR_LIFT_STILL_ABSENT
FACTOR_BASE_DENSITY_LOGS_RANK_DESCENT_UNCHARGED
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: invert the fixed-class multiplication map at the
`t=29` root using a constant-size rank-one lifting system, then factor the two
recovered cubic line intersections without scanning `F_193`.
