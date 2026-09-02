# P1553 M6 weighted C3 Mobius-gcd trace gate R145

## Claim boundary

R145 gives a root-free exact realization of the R144 weighted six-factor
count as a common-divisor quotient trace on the pairing torus. It also
charges the standard dense support-polynomial realization. That realization
fits the frozen setup cap, but one target pullback exceeds the fresh-work cap
and the full `A6` batch exceeds Pollard rho.

This is a scoped negative for the explicit coefficient route. It is not a
lower bound for implicit modular resultants, target-batched remainder trees,
arithmetic circuits, RAM, or cell probes. It does not construct weighted
marginals, factor logs, identical descent, a generic-prime algorithm, or an
ECDLP attack.

It claims no Pollard-rho improvement, Shoup improvement, or breakthrough.

Classification:

```text
WEIGHTED_C6_COUNT_IS_EXACT_C3_DIVISOR_MOBIUS_GCD_QUOTIENT_TRACE__ROOT_FREE_POSITIVE_AND_EMPTY_REPLAY_ON_EIGHT_CONTROLS__EXPLICIT_C3_POLYNOMIAL_DEGREE_B9O4_FITS_SETUP__ONE_PULLBACK_FAILS_FRESH_CAP__A6_BATCH_B11O4_EXCEEDS_RHO__IMPLICIT_BATCHED_MODULAR_RESULTANT_OPEN__NO_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Exact divisor identity

Use a Cayley parameter on the norm-one pairing torus. For nonsquare `d`,

```text
x oplus y = (x+y)/(1+d*x*y),
y = (tau-x)/(1-d*tau*x).
```

Let `mu_C3(x)` be the ordered three-`C` occurrence multiplicity and freeze

```text
P(X) = product_(x in supp(C3)) (X-x),
W(x) = mu_C3(x).
```

For target parameter `tau`, homogenize the Mobius pullback:

```text
P_tau(X)
  = (1-d*tau*X)^n
    P((tau-X)/(1-d*tau*X)),
G_tau(X) = gcd(P(X), P_tau(X)).
```

The roots of `G_tau` are exactly the support atoms whose target partners are
also in the `C3` support, on the admitted Cayley chart. The weighted ordered
six-`C` count is the quotient-algebra trace over
`F_p[X]/G_tau` of

```text
W(X) W((tau-X)/(1-d*tau*X)).
```

The algorithm uses polynomial GCD, inversion, reduction, and trace. It does
not extract roots or consume discrete logarithms.

## Exact controls

The verifier uses all four R82 curve families and both offsets. Their `C3`
support degrees are

```text
10, 10, 35, 35, 56, 56, 84, 84.
```

On every control:

- the support polynomial has exactly the finite `C3` support as roots;
- the interpolant reproduces every ordered occurrence weight;
- twelve positive targets have exact common-divisor degree and weighted
  quotient trace;
- six empty targets have zero common divisor and zero trace;
- the trace agrees with the independent direct canonical `C6` counter;
- every sampled integer count is below the field prime;
- no sampled denominator pole occurs on the support.

Finite enumeration and direct support intersections are verifier-only and
receive no candidate or asymptotic credit.

## Standard-route cost

At the R115 Cartesian vertex,

```text
N                    = B^(5+o(1)),
deg P                = B^(9/4+o(1)),
frozen setup cap     = B^(9/4+o(1)),
fresh-work cap       = B^(5/4+o(1)),
number of A6 targets = B^(1/2+o(1)).
```

The standard coefficient representation materializes `B^(9/4)` coefficients
for one Mobius pullback and then performs a degree-`B^(9/4)` GCD and quotient
trace. Therefore:

```text
one explicit target query = B^(9/4+o(1)) > B^(5/4+o(1)),
full A6 target batch       = B^(11/4+o(1))
                           = N^(11/20+o(1))
                           > N^(1/2).
```

Fast polynomial arithmetic changes polylogarithmic factors, not these output
and batch exponents. This accounting applies only to the explicit dense
pullback route.

## Admission

Seventeen of twenty-eight obligations pass. The Cayley partner law,
support-divisor identity, root-free quotient-trace count, all finite semantic
controls, and the standard explicit-route cost rejection are admitted.
An implicit batched operator, reusable transposed marginals, generic integer
lifting, structured rank and density, factor logs, identical descent, and
generic-prime transfer are not admitted.

Disposition:

```text
ADMIT_EXACT_ROOT_FREE_WEIGHTED_C3_GCD_TRACE_IDENTITY__REJECT_STANDARD_DENSE_PULLBACK_AT_FROZEN_QUERY_AND_RHO_CAPS__PRESERVE_IMPLICIT_BATCHED_MODULAR_RESULTANT_ROUTE__NO_FACTOR_LOGS__NO_DESCENT__NO_GENERIC_TRANSFER__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct one implicit target-batched modular resultant or remainder operator
for the weighted `C3` divisor. It must consume the compact `C` deck and an
`A6`-shifted target batch without emitting a degree-`B^(9/4)` transformed
polynomial per target, compute exact integer `C6` counts and reusable weight
transposes inside `B^(9/4+o(1))` setup and `B^(5/4+o(1))` fresh work, and
preserve the R144 known-RHS rank, factor-log, and shifted-descent identities.
It may use no DLP, root, GCD, count, marginal, rank, or source oracle.
