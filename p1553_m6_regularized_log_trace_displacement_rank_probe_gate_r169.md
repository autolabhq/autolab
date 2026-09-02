# P1553 M6 regularized log-trace displacement-rank gate R169

Date: 2026-08-01

## Scope

R169 sharpens the open R168 denominator-aware trace interface. It introduces
a scalar regularization parameter and tests whether ordinary diagonal
displacement structure compresses the resulting elliptic Cauchy kernel.

The finite matrices and ranks are controls only. They receive no asymptotic
attack credit and establish no arithmetic-circuit lower bound.

## Scalar resolvent

Write the compact R167 witness as `h=F_num/F_den`, after the public R168
target-equality regularization. For a selected endpoint `P`, define

```text
chi_P(lambda)
  = product_(Q in S union -S) (h(Q+P) + lambda).
```

Every inherited `F_den(Q+P)` is a unit on the selected divisor. Consequently,
the `lambda`-adic valuation of `chi_P` at zero is exactly the number of
translated numerator zeros, hence exactly the R168 candidate multiplicity.
This gives a candidate-safe Fitting interface without evaluating `Dh/h` at a
candidate nonunit.

For nonzero regularizing `lambda`, the differentiated kernel is

```text
K_lambda(P,Q)
  = Dlog((F_num + lambda F_den) / F_den)(Q+P).
```

The zero specialization must be retained symbolically. Evaluating only at
generic nonzero `lambda` values and discarding the zero-locus does not recover
candidate multiplicities.

## Finite controls

Six controls span three curves, signed-divisor degrees 10, 35, and 56, target
witness degrees 5, 7, and 8, and two seeds per curve. In every control:

```text
chi_P is monic of degree 2n
ord_(lambda=0) chi_P equals the R168 candidate multiplicity
pencil candidate roots equal the R168 roots
the regularized kernel has full row rank
```

The following ordinary diagonal displacements were tested:

```text
x Sylvester minus and plus
y Sylvester minus and plus
x Stein
y Stein
```

Every tested displacement has full row rank. The `x`-Sylvester displacement
also remains full row rank for powers one through four.

On the largest curve, a two-seed degree sweep over witness degrees two through
eight gives fourteen further rows. Both the raw kernel and the ordinary
`x`-Sylvester displacement have rank 56 in every row.

These observations close only the tested ordinary diagonal `x/y` generators
on these finite controls. They do not refute a custom elliptic companion
operator, a fraction-free subresultant/Fitting algorithm, or another compact
arithmetic circuit.

No DLP, root, count, marginal, rank, source, inverse, resultant, or trace
oracle is consumed.

## Full cost boundary

At the campaign caps:

```text
compact h and logarithmic witness state:      B^(5/4)
regularized n-by-2n kernel matrix:             B^(9/2)
full-rank displacement generator state:       B^(9/2)
generic degree-2n pencil coefficients/samples: B^(9/2)
preferred fraction-free output work:          B^(9/4)
rho proxy:                                    B^(5/2)
```

Generic interpolation needs `2n+1` scalar values for each selected endpoint.
Its `n(2n+1)` coefficient or sample state is `Theta(n^2)=B^(9/2)`, above rho.
A full-rank displacement generator has the same state exponent. Neither route
is an admitted compression.

## Literature and deduplication

Bostan, Jeannerod, Mouilleron, and Schost define Sylvester and Stein
displacement operators and give fast algorithms when an explicit low-rank
generator exists. R169 uses those definitions and does not attribute an
elliptic low-rank theorem to that paper.

R150 tested a different rational-convolution closure route. R167 supplies the
compact principal-divisor witness, and R168 supplies the exact logarithmic
candidate-pole interface. R169 preserves both and tests only the next proposed
compression mechanism.

## Admission

Twenty-one of twenty-nine obligations pass. Admit the exact scalar resolvent,
the zero-valuation/candidate-multiplicity biconditional, six exact pencil
controls, and the scoped full-rank results for the tested ordinary diagonal
operators and degree sweep.

Do not admit a fraction-free Fitting/subresultant constructor, a custom
elliptic companion displacement, deterministic hash-to-curve transfer, an
unconditional generic-prime algorithm, Pollard-rho or Shoup improvement, or an
ECDLP breakthrough.

Disposition:

```text
ADMIT_EXACT_SCALAR_RESOLVENT_FITTING_INTERFACE__LAMBDA_VALUATION_EQUALS_R168_MULTIPLICITY__SIX_NATURAL_XY_SYLVESTER_STEIN_DISPLACEMENTS_FULL_ROW_RANK__DEGREE_TWO_THROUGH_EIGHT_SWEEP_FULL_RANK__GENERIC_PENCIL_AND_FULL_GENERATOR_B9O2__CUSTOM_ELLIPTIC_COMPANION_OR_FRACTION_FREE_SUBRESULTANT_OPEN__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next action

Construct or refute a fraction-free elliptic Fitting/subresultant modulo `U`
directly from `F_num`, `F_den`, and their invariant derivatives, below
`B^(5/2)` and preferably `B^(9/4+o(1))`. Do not interpolate the degree-`2n`
lambda pencil or reuse the full-rank diagonal displacements. A custom elliptic
companion operator is admissible only with an explicit low-rank generator and
fully charged application cost.
