# P1436 autoresearch harness V123 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V123 binds R174 as the 110th closed frontier lane and routes the first
experiment to `s66_fused_factored_dual_chow_outer_norm_mod_u` at priority 310.

## R174 Result

For `P=(X,V(X))`, `Q=(Z,V(Z))`, and `T=(u,v)`, the signed collinearity
determinant is

```text
L(P,Q,-T) = (X-u)V(Z) + (u-Z)V(X) + (X-Z)v.
```

Off the diagonal `Q != P`, dividing by `Z-X` gives

```text
L/(Z-X) = (X-u) Delta_V(X,Z) - (V(X)+v),
Delta_V(X,Z) = (V(Z)-V(X))/(Z-X).
```

This factor is zero exactly when `P+Q=T`.

## Tangent Correction

The raw line determinant vanishes identically at `Q=P`. The derivative of the
interpolated side table `V` is not the elliptic-curve tangent, so substituting
`V'(X)` would produce a false locator. R174 uses a separate diagonal chart:

```text
(X-u)(3X^2+a) - 2V(X)(V(X)+v).
```

It vanishes exactly when `2P=T`. Six controls verify 68,326 signed factor
biconditionals, including 5 tangent zeros and 236 secant zeros.

## Dual Chow

Define the factored target dual-Chow form

```text
C_T(alpha,beta,gamma)
  = product_T (alpha*u_T + beta*v_T + gamma).
```

For each fixed `P,Q`, the target product is

```text
Q != P:
  C_T(-Delta_V, -1, X Delta_V - V(X))

Q = P:
  C_T(-(3X^2+a), -2V(X), X(3X^2+a)-2V(X)^2).
```

The full signed locator is the product of those values over all selected `Q`.
The controls verify 8,922 target-Chow evaluations and exact equality between
target-first and target-last products.

The selected dual-Chow form

```text
C_S(alpha,beta,gamma)
  = product_Q (alpha*y_Q + beta*x_Q + gamma)
```

has one identically zero factor at `Q=P`. Its chart-corrected norm is

```text
(-1)^(n-1) geometric_tangent * partial_gamma(C_S) / U'(X).
```

All 1,486 confluent selected-Chow derivative identities are exact. The final
roots exactly replay all 140 R166 verified roots and exclude the opposite-sign
Kummer branch by construction.

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

The six represented target Chow forms use all 204 slots, reaching the rho
exponent before the outer norm. The selected Chow forms use all 4,770 slots.
These finite full-density observations receive no asymptotic or general
lower-bound credit.

The audited dense multivariate multipoint contract starts from a represented
degree-`N` bivariate input with `Theta(N^2)` coefficients. It does not accept
the factored target linear forms while returning only their fused outer norm
modulo arbitrary squarefree `U`.

R174 passes 16 of 23 obligations. It admits the signed two-chart identity and
closes the standard represented and query-grid routes only. A fused factored
outer norm remains open. Lane admission, rho improvement, Shoup improvement,
and breakthrough flags remain false.

## V123 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v110`.
- Bound and closed frontier lanes: 110.
- First focus: `s66_fused_factored_dual_chow_outer_norm_mod_u`.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R174 focused tests: 15 passed in 40.784 seconds.
- Harness tests: 140 passed in 0.625 seconds.
- Full ECDLP suite: 1,147 passed in 478.115 seconds.
- R174 clean replay: all six JSON outputs byte-identical.
- V123 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R174: 99 receipts, 1,925 recursive path/hash bindings, zero
  mismatches, missing paths or rounds, and duplicate rounds.

## R174 Hashes

- Producer: `9b30baf9bea77492816bd81bcbd6cfae01ae05b467b8793562101da8afe9765c`
- Report: `a8b1d1d5fd4ffeaef17726325ebd85c343285ef61d16ca4dd1ce91dfcce28496`
- Frozen interface: `4673a2fd8dea6f212d4761b8ac60f99aadd99ddb43656db4e64992fb7cbd5d3b`
- Cost ledger: `dee4664a442779b2c531714bcfe8af33deff1af1b903c99801c363438a3693b1`
- Replay: `fbac601d12b26ce98237c3481af4a7c8b956df658585bb085acf16c3afb8e3aa`
- Controls: `6d69f7146857cfd017281550e7a44b99aa4c53623b767c852da40b81e8bc9d40`
- Chow ledger: `a87d2bb7b4b68d874c1f74867cd3d34ba5d69b2c60c7f58903647579c1cfd3f8`
- Tests: `747064ba9bc417572ab40ca6bac9169c35fc130bb112739d893ff76c06d274a7`
- Gate: `46597f7afd91ada661b5822031489b7d08445b041dfd4399b4343199457afebb`
- Parent: `cac550192f9da4090c1e5459a5baed1f0ea3437c89ec012436995fcfe5bb4911`

## V123 Hashes

- Harness: `24f8ecff4793f6b412173b0ff3067ed75058b42d44f2395b90f1465c9d666b11`
- Harness tests: `e2ec70b9987ef858c9ef30021dddcc6bb598c0e72b14557054cf73f66243e486`
- Focus report: `e78677f5bec31608632b8e026a787c7b5919169a7297b9e951af2d9d26d85b8c`
- Note: `070a7e7cba0488af9aa2c0462c3e1811e49c6f6e0039437bc3fc1c8a280e7d9e`
- Evidence inventory: `2993139c349af800ed5762fdf8e3c9f2a7618638a5de04591beaf9da9e54a618`
- Replay plan: `ca57f7d2159dcf563828f74c526c2b728ee3a7f417e88eafa477af790dea4fdc`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact identities, finite controls,
signed root replay, full-density observations, and verifier passes receive no
asymptotic attack or lower-bound credit.

## Next Action

Construct or refute an arithmetic DAG that keeps the `N` target linear factors
factored, handles `Delta_V` and the geometric tangent chart symbolically, and
emits the signed outer norm or its gcd modulo `U` in softly `O(n+N)` work below
`B^(5/2)`. Reject `N^2` or `n^2` Chow coefficients, `nN` or `n^2` query grids,
`N` independent quotient transforms, interpolant-derivative tangents,
candidate inversions, and unit-cost multipoint, norm, derivative, resultant,
root, count, marginal, rank, source, or DLP oracles.
