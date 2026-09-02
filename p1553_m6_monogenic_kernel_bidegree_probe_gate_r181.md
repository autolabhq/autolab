# P1553 M6 monogenic-kernel bidegree gate R181

Date: 2026-08-01

## Scope

R181 tests the last R180 exact-monogenic opening. It separates a representation
identity from a constructor:

```text
C_h = H(a) mod U
```

always exists by taking `a=X` and `H=C_h`, but modular composition starts only
after the coefficients of `H` are available. That tautology does not construct
the signed aggregate or its gcd with `U`.

The exact bounded-bidegree scalar-kernel route does not survive the finite
controls. A compact high-degree SLP or gcd-equivalent output-sensitive elliptic
composed resultant remains open. This is not an asymptotic rank or circuit lower
bound, an ECDLP algorithm, or a Pollard-rho or Shoup improvement.

## Canonical Kernel

Fix a selected point `P`. Each tangent-corrected R174 line factor is linear in
the target coordinates `(u,v)`. Multiplying over the `n` selected points `Q`
and reducing by

```text
v^2 = u^3 + a*u + b
```

gives the unique normal form

```text
K_P(u,v) = A_P(u) + v*B_P(u).
```

For every control, this canonical body has exactly `3n` coefficient slots per
source point, every slot is nonzero, and its pole order at infinity is exactly
`3n`. Translation by `P` maps the selected divisor to `n` distinct desired
targets `P+Q`, all of which are exact zeros of `K_P`.

## Exact Controls

R181 replays the six R180 controls and adds seed `18104` on all three curves as
a held-out family:

```text
control count:                              9
development / held-out controls:          6 / 3
selected divisor degree sum:                303
retained target count sum:                   60
canonical coefficient slots:            40,149
canonical nonzero coefficients:          40,149
desired signed incidence zeros:          13,383
charged scan pair evaluations:          658,473
```

All nine canonical coefficient matrices have full source rank `n`. Evaluation
on the first `n` admissible public subgroup multiples also has rank `n` and
equals the independently reconstructed R180 target factors entry for entry.
All six development candidate factors match R180, and all three held-out
candidate factors match fresh R174 executions.

If exact factors have a common scalar form

```text
K(a(P),T) = sum_(i=0)^d a(P)^i*c_i(T),
```

their evaluation matrix has rank at most `d+1`. The finite full-rank witnesses
therefore force `d >= n-1` for these fixed controls. This statement covers
exact values and separable row or column scalar normalizations. It does not
cover arbitrary target-and-source-dependent units when only the gcd zero set
is required.

## Cost Boundary

```text
n selected degree:                          B^(9/4)
N target degree:                            B^(5/4)
candidate output:                           B^(3/4)
postcompiled degree-n modular composition:  B^(9/4)
canonical explicit bidegree body:            B^(9/2)
N explicit degree-n factors:                 B^(7/2)
flattened selected-target algebra:            B^(7/2)
full composed-resultant output degree:        B^(7/2)
rho proxy:                                    B^(5/2)
```

Kedlaya-Umans can make finite-field modular composition and power projection
near-linear in represented input dimension. Poteaux-Schost similarly measure
norm and multivariate modular-composition costs in the represented triangular
algebra dimension. Flattening the selected and target algebras makes that
dimension `nN`, not `n+N`.

Bostan, Flajolet, Salvy, and Schost accelerate special composed resultants, but
the complete composed polynomial has degree `nN`. R181 gives no credit for
constructing that full output before reduction modulo `U`.

## Admission

Admit the canonical signed target-kernel normal form, all nine exact replays,
the three held-out candidate controls, the finite full-rank witnesses, and the
standard represented-input cost specializations.

Close post-construction `H(a)`, explicit exact bounded-source-degree scalar
kernels on the controls, the canonical dense bidegree body, flattened tensor
norms, and full degree-`nN` composed-resultant output.

Do not infer an asymptotic rank theorem or a lower bound against compact
high-degree SLPs, target-dependent gcd units, transposed subresultant traces, or
an output-sensitive elliptic composed resultant modulo `U`. No factor logs,
target descent, complete attack, or generic-prime Shoup improvement is supplied.

Disposition:

```text
ADMIT_CANONICAL_SIGNED_TARGET_KERNEL_NORMAL_FORM__NINE_CONTROLS_INCLUDING_THREE_HELD_OUT_REPLAYED__TARGET_POLE_ORDER_3N__EXACT_COEFFICIENT_AND_PUBLIC_SCAN_RANK_N__CANONICAL_BODY_3N2_FULLY_DENSE__TAUTOLOGICAL_H_OF_A_GETS_NO_CONSTRUCTOR_CREDIT__EXPLICIT_BOUNDED_SOURCE_DEGREE_FOLD_CLOSED_ON_CONTROLS__FLATTENED_NN_NORM_AND_FULL_COMPOSED_RESULTANT_B7O2__GCD_EQUIVALENT_OUTPUT_SENSITIVE_RESULTANT_OPEN__NO_ASYMPTOTIC_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute one gcd-equivalent output-sensitive elliptic composed
resultant modulo `U`. Accept the `O(n)` signed line-product SLP and `O(N)` target
divisor, discard target-dependent units, and emit `G_1` in softly `O(n+N)` work
without constructing the canonical `3n^2` coefficient body, an `nN` tensor
element, or the full degree-`nN` composed resultant. Test direct transposed
power projections or subresultant traces modulo `U` on seed `18104`. Reject
candidate oracles, unit-cost norms, and post-construction modular composition.
