# P1553 M6 logarithmic-derivative elliptic Cauchy trace gate R168

Date: 2026-08-01

## Scope

R168 differentiates the exact R167 product identity. The goal is to replace a
multiplicative elliptic resultant by an additive trace while retaining the
candidate roots as denominator poles.

The finite producer enumerates endpoint-target and endpoint-support pairs to
verify local multiplicities and derivatives. Those tables receive no
asymptotic attack credit.

## Invariant derivative

On

```text
E: y^2 = x^3 + ax + b,
```

use the invariant derivation

```text
D = 2y d/dx + (3x^2+a) d/dy,
```

dual to `omega=dx/(2y)`. For every nonzero elliptic function `g`,

```text
(Dg/g) omega = dg/g.
```

At a zero of order `m`, this logarithmic differential has a simple pole with
residue `m`. At a pole of order `m`, its residue is `-m`.

Let the regularized R166 candidate product be

```text
G(P) = product_j U(x(T_j-P)).
```

Every Kummer candidate has multiplicity between one and `N`. Since
`N=B^(5/4)` and the field characteristic is `B^5`, every candidate
multiplicity is nonzero in the field. Therefore no candidate disappears by
characteristic-`p` residue cancellation.

The denominator support of `DG/G` on the selected signed divisor is exactly
the true-or-opposite-sign Kummer candidate support, after the public `P=T`
pole correction below.

## Additive trace

R167 gives

```text
G(P)
  = correction(P)
    product_(Q in S union -S) h(Q+P) / h(P)^(2n).
```

Logarithmic differentiation gives the additive identity

```text
Dlog G(P)
  = Dlog correction(P)
    + sum_(Q in S union -S) Dlog h(Q+P)
    - 2n Dlog h(P).
```

The compact target witness `h` has degree `N`, and `Dh/h` has the same
`B^(5/4+o(1))` representation scale. In the R167 controls, every denominator
of `h(Q+P)`, every `h(P)`, and every auxiliary correction is a unit on the
selected divisor. Candidate poles arise only from translated numerator zeros.

This replaces the multiplicative resultant interface with a
denominator-aware elliptic Cauchy trace. The required output is

```text
gcd(U, denominator(Dlog G)).
```

A value-only trace formed after inverting `h` in a tensor quotient is not a
candidate locator: the candidate points are exactly where those inverses do
not exist. A valid constructor must preserve the relevant zero-divisor,
Fitting-ideal, or subresultant denominator information.

## Public equality correction

When a public target equals the selected endpoint `P`, the rational factor
meets the order-`2n` pole of `U(x(Q))` at `O`. Its rational logarithmic residue
is `-2n`, not a candidate-zero residue. R166 assigns this semantic factor one.

The equality is public and sign-sensitive:

```text
U(x(T)) = 0,  V(x(T)) = y(T).
```

Fast multipoint evaluation of `U,V` on all targets costs
`B^(9/4+o(1))`, within the setup cap. The known equality factors are removed
before the trace locator, giving semantic logarithmic derivative zero.

## Finite controls

Across six curve/seed controls:

```text
candidate poles:                       140
translated zero occurrences:           241
maximum candidate multiplicity:          4
true-orientation occurrences:           241
opposite-orientation occurrences:         0
public P=T pole corrections:              6
```

Every direct zero derivative `-2y U'(x)` is nonzero. Every corresponding
translated numerator derivative in the R167 witness is nonzero. Direct and
swapped multiplicities agree at every selected endpoint, all multiplicity
residues are nonzero, and the 140 pole roots equal the R167 candidate roots.

Each batch contains one `denominator_exception` equality. All six satisfy the
signed `U,V` test, have nonzero rational residue `-2n`, and are reset to the
semantic factor one and logarithmic derivative zero.

No DLP, trace, inverse, resultant, root, count, marginal, rank, or source
oracle is consumed.

## Full cost boundary

At the campaign caps:

```text
compact h and Dh/h state:             B^(5/4)
public target-equality prefilter:      B^(9/4)
direct n-by-N logarithmic table:       B^(7/2)
raw swapped 2n-by-n trace table:       B^(9/2)
standard tensor quotient A tensor A:  B^(9/2)
expected candidate denominator degree: B^(3/4)
signed candidate verification:        B^2
rho proxy:                             B^(5/2)
```

The compact logarithmic witness and public equality prefilter are below rho.
All explicit direct, swapped, and tensor-quotient routes are above rho.

The surviving primitive is a denominator-aware transposed elliptic Cauchy
trace modulo `U`, preferably in `B^(9/4+o(1))` work and in all cases strictly
below `B^(5/2)`, without `nN`, `n^2`, or tensor-quotient materialization.

## Deduplication

R167 supplies the compact principal divisor and exact multiplicative
reciprocity identity. R168 differentiates that admitted identity; it does not
claim a fast trace implementation.

Eagen's primary paper uses logarithmic derivatives to linearize divisor
witness products in a proof system. R168 adapts the algebraic identity as a
candidate-pole interface and attributes no ECDLP complexity result to the
paper.

R147 verifies occurrence-pair local valuations of an implicit resultant.
R168 instead tracks logarithmic residues of the R167 target divisor and still
requires a new shared denominator-aware trace.

R166 supplies Kummer true-or-opposite candidate semantics, public equality
regularization, and exact signed verification. Those outputs remain unchanged.

## Admission

Twenty-four of thirty-two obligations pass. Admit the invariant logarithmic
derivative, nonzero-residue candidate-pole biconditional, exact additive trace
identity, compact `Dh/h` state, public equality correction, and six finite
controls.

Do not admit a denominator-aware trace constructor, candidate-safe
zero-divisor handling below rho, deterministic hash-to-curve transfer, an
unconditional generic-prime algorithm, Pollard-rho or Shoup improvement, or an
ECDLP breakthrough.

Disposition:

```text
ADMIT_INVARIANT_LOG_DERIVATIVE_CANDIDATE_POLE_INTERFACE__241_DIRECT_AND_SWAPPED_ZERO_OCCURRENCES__140_NONZERO_RESIDUE_ROOTS__SIX_PUBLIC_EQUALITY_POLES_REGULARIZED__ADDITIVE_ELLIPTIC_TRACE__DIRECT_B7O2_RAW_B9O2_TENSOR_B9O2__DENOMINATOR_AWARE_TRACE_MOD_U_OPEN__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next action

Construct or refute a denominator-aware transposed elliptic Cauchy trace
modulo `U` for the compact `Dh/h` witness. Test structured subresultants,
displacement rank, and transposed multipoint methods while retaining Fitting
information at candidate nonunits and forbidding `nN`, `n^2`, and tensor-state
materialization.
