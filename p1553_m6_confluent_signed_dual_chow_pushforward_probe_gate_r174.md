# P1553 M6 confluent signed dual-Chow pushforward gate R174

Date: 2026-08-01

## Scope

R174 turns the R173 target-sign split into a commutative signed incidence
operator. It uses a factored target dual-Chow form and a tangent-aware
confluent chart to recover exactly the R166 verified roots.

The identity is exact and removes the opposite-sign Kummer branch. Standard
represented Chow and query-grid routes are at or above rho, so a fused factored
outer norm remains unproved. This is not an ECDLP algorithm, a general
resultant or circuit lower bound, or a Pollard-rho or Shoup improvement.

## Signed Line

For `P=(X,V(X))`, `Q=(Z,V(Z))`, and `T=(u,v)`, collinearity of `P,Q,-T` is

```text
L(P,Q,-T) = (X-u)V(Z) + (u-Z)V(X) + (X-Z)v.
```

For `Q != P`, this vanishes exactly when `P+Q=T`. Dividing by `Z-X` gives

```text
L/(Z-X) = (X-u) Delta_V(X,Z) - (V(X)+v),
Delta_V(X,Z) = (V(Z)-V(X))/(Z-X).
```

## Tangent Chart

The raw determinant vanishes identically at `Q=P`, so an ordinary product of
line values is useless. More subtly, the derivative of the interpolated side
table `V` is not the tangent derivative of the elliptic curve.

The diagonal must use the scaled geometric tangent

```text
(X-u)(3X^2+a) - 2V(X)(V(X)+v).
```

It vanishes exactly when `2P=T`. The six controls verify 68,326 signed factor
biconditionals: 5 tangent zeros and 236 secant zeros.

## Target Dual Chow

Define the factored target dual-Chow form

```text
C_T(alpha,beta,gamma)
  = product_T (alpha*u_T + beta*v_T + gamma).
```

For fixed `P,Q`, the target product is evaluated in two charts:

```text
Q != P:
  C_T(-Delta_V, -1, X Delta_V - V(X))

Q = P:
  C_T(-(3X^2+a), -2V(X), X(3X^2+a)-2V(X)^2).
```

The full signed locator is the product of these values over all `Q` in the
selected divisor. The controls verify 8,922 target-Chow evaluations and exact
equality between target-first and target-last multiplication.

The aggregate roots are exactly all 140 R166 independently verified roots.
No opposite-sign Kummer roots remain, and no root, count, marginal, rank,
source, or DLP oracle is consumed.

## Selected Chow Derivative

The dual formulation

```text
C_S(alpha,beta,gamma)
  = product_Q (alpha*y_Q + beta*x_Q + gamma)
```

has one identically zero factor at `Q=P`. Its chart-corrected norm is

```text
(-1)^(n-1) geometric_tangent * partial_gamma(C_S) / U'(X).
```

All 1,486 selected-Chow derivative identities are exact. This derivative is a
confluent correction, not permission to differentiate the interpolated `V` as
though it were the curve.

## Cost Boundary

```text
compact selected/target divisor state:        B^(9/4)
factored target dual-Chow state:              B^(5/4)
represented target dual-Chow body:            B^(5/2)
selected-target query grid:                   B^(7/2)
represented selected dual-Chow body:          B^(9/2)
target-first selected-pair query grid:         B^(9/2)
factored target Chow on selected-pair grid:    B^(23/4)
represented aggregate output:                 B^(9/4)
rho proxy:                                     B^(5/2)
```

The six target Chow forms use all 204 represented slots. The selected Chow
forms use all 4,770 represented slots. These finite full-density observations
receive no asymptotic or lower-bound credit.

Dense multivariate multipoint evaluation does not repair the boundary: its
represented degree-`N` bivariate input already has `Theta(N^2)` coefficients,
and it does not accept the factored linear forms while returning only the
fused outer norm modulo arbitrary squarefree `U`.

## Admission

Admit the signed collinearity identity, off-diagonal divided difference,
geometric tangent chart, selected-Chow derivative, target dual-Chow
pushforward, target-order interchange, and exact R166 root replay.

Do not admit a fused factored outer norm, deterministic hash-to-curve transfer,
a generic-prime coordinate-family algorithm, a complete attack, or a
Pollard-rho or Shoup improvement.

Disposition:

```text
ADMIT_SIGNED_COLLINEARITY_AND_TANGENT_CONFLUENT_DEFLATION__EXACT_TARGET_DUAL_CHOW_PUSHFORWARD__TARGET_FIRST_EQUALS_TARGET_LAST__R166_VERIFIED_ROOTS_REPLAYED_WITHOUT_OPPOSITE_SIGN_BRANCH__REPRESENTED_TARGET_CHOW_N2_B5O2_AT_RHO__SELECTED_CHOW_N2_B9O2__NN_AND_N2_GRIDS_ABOVE_RHO__FACTORED_CONFLUENT_OUTER_NORM_OPEN__NO_GENERAL_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute a fused operator that keeps the `N` target linear factors
factored, handles `Delta_V` and the geometric tangent chart symbolically, and
emits the outer norm modulo `U` in softly `O(n+N)` work below `B^(5/2)`.
Reject `N^2` or `n^2` Chow coefficients, `nN` or `n^2` query grids,
target-dependent uncharged transforms, candidate inversions, and unit-cost
multipoint, norm, derivative, resultant, root, count, marginal, rank, or source
oracles.
