# P1553 M6 gcd-equivalent target-subresultant gate R182

Date: 2026-08-01

## Scope

R182 instantiates the R181 gcd-only opening as an exact target-divisor
subresultant. It tests standard represented remainder, norm, power-projection,
inverse-certificate, and half-GCD interfaces. It does not prove a general
resultant or circuit lower bound, construct an ECDLP algorithm, or improve the
Pollard-rho or Shoup bounds.

## Target Divisor

Publicly remove repeated target points, then partition the unique targets into
at most two x-injective charts. Each chart has

```text
W(u)   = product_j (u-u_j)
V_T(u) = the interpolant with V_T(u_j)=v_j.
```

The second chart is necessary only when both elliptic points `(u,v)` and
`(u,-v)` occur. R182 includes a deterministic adversarial control that repeats
one held-out target and adds its opposite point, forcing exactly two charts.

For the R181 normal form `K_P=A_P+v*B_P`, restriction to one chart is

```text
R_P(u) = A_P(u) + V_T(u)*B_P(u) mod W(u).
```

Since deduplication makes `W` squarefree,

```text
gcd(W,R_P) nonconstant
  iff R_P is a nonunit in F[u]/W
  iff Norm(R_P) = Res(W,R_P) = 0.
```

## Exact Controls

R182 replays nine R181 controls and adds held-out seed `18206` on all three
curve families:

```text
control count:                                  12
R181 replay / held-out controls:              9 / 3
selected divisor degree sum:                    404
raw / unique retained target sum:           80 / 79
ordinary target chart count:                     12
remainder coefficient slots:                   2,962
remainder nonzero coefficients:                2,962
candidate / noncandidate source components: 253 / 151
source-target gcd-degree sum:                    409
unit inverse certificates:                      151
inverse certificate coefficient slots:        1,130
Newton trace projections:                     2,962
```

Every ordinary remainder body is fully dense and every remainder matrix has
full target rank. Direct evaluation, polynomial remainder evaluation, Newton
reconstruction of the norm, norm-zero tests, subresultant gcd tests, and all
unit inverse certificates agree exactly. All nine inherited factors equal R181;
all three held-out factors equal fresh R174 executions.

The opposite-point adversarial control has three raw targets, two unique
targets sharing one x-coordinate, and exactly two charts. Both charts preserve
the direct kernel values and the norm-zero/nonunit/gcd equivalence, and their
combined candidate roots equal the gcd roots.

## Cost Boundary

```text
n selected degree:                         B^(9/4)
N target degree:                           B^(5/4)
candidate output:                          B^(3/4)
desired total work:                        B^(9/4)
represented remainder body nN:             B^(7/2)
N degree-n Newton trace outputs:            B^(7/2)
noncandidate inverse-certificate body:      B^(7/2)
represented triangular norm dimension:      B^(7/2)
standard target-variable half-GCD:           B^(7/2)
rho proxy:                                   B^(5/2)
```

Poteaux-Schost and Kedlaya-Umans make modular composition, power projection,
and norms fast relative to represented algebra dimension. Here that represented
dimension or requested trace body is `nN`, not `n+N`. Fast special and generic
bivariate resultants likewise do not by themselves supply a compact SLP-direct,
candidate-only resultant modulo arbitrary `U`.

The finite density and rank controls diagnose the represented routes. They do
not establish an asymptotic lower bound against every straight-line program.

## Admission

Admit public target deduplication, the two-chart elliptic target divisor, the
exact remainder/norm/nonunit/gcd identity, all twelve candidate replays, the
adversarial opposite-point control, Newton norm reconstruction, and unit inverse
certificates.

Close the standard represented remainder matrix, direct `N` by degree-`n`
Newton trace body, per-component represented inverses, triangular norm, and
target-variable half-GCD at scale `nN=B^(7/2)`.

Preserve one opening: an SLP-direct determinant-zero or resultant-mod-`U`
algorithm that consumes the compact R181 line-product kernel and target charts,
emits only `G_1`, and uses softly `O(n+N)` work. No factor logs, identical fresh
target descent, complete attack, or generic-prime Shoup improvement is supplied.

Disposition:

```text
ADMIT_GCD_EQUIVALENT_TARGET_DIVISOR_SUBRESULTANT_IDENTITY__TWELVE_CONTROLS_INCLUDING_THREE_HELD_OUT_REPLAYED__PUBLIC_DUPLICATES_REMOVED_AND_TWO_X_CHART_ADVERSARIAL_CONTROL_COMPLETE__NORM_ZERO_IFF_NONUNIT_IFF_GCD_NONTRIVIAL__NEWTON_TRACES_AND_UNIT_INVERSES_EXACT__REMAINDER_BODY_FULLY_DENSE_AND_FULL_TARGET_RANK__REPRESENTED_REMAINDER_TRACE_INVERSE_NORM_AND_HALF_GCD_B7O2__SLP_DIRECT_OUTPUT_SENSITIVE_RESULTANT_MOD_U_OPEN__NO_GENERAL_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute one SLP-direct determinant-zero or resultant-mod-`U`
algorithm. Consume the compact R181 signed kernel and deduplicated target
charts, and emit only `G_1` in softly `O(n+N)` work without the `nN` remainder
matrix, `N` degree-`n` Newton traces, noncandidate inverse certificates, or a
represented triangular norm. Test a transposed single-functional minimal-
polynomial or subresultant certificate on seed `18206` and one new divisor
family. Reject hidden `nN` projection outputs, candidate oracles, and unit-cost
determinant or resultant calls.
