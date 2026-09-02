# P1553 5A/5C Nonlinear Tensor-Tower Trace Gate R98

Date: 2026-07-29

Status: `SCOPED_NEGATIVE`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can arbitrary nonlinear encoders on the compact A and C divisor circuits meet
through one small tensor bond and exactly evaluate the resultant zero
projector?

R98 freezes a one-bond separation:

```text
K(u,v) = sum_(r=1)^w L_r(u) R_r(v),
```

where `L` and `R` may be arbitrary nonlinear local functions. It does not
freeze a multi-edge digitized encoding, a nonalgebraic lookup table, or a
restriction to an unproved smaller image of the actual EC divisor circuits.

## Exact theorem

On the monic-quadratic coefficient subfamily

```text
A_u(z) = z^2+u,
C_v(z) = z^2+v,
Res(A_u,C_v) = (v-u)^2.
```

Therefore, over every odd prime field,

```text
delta_0(Res(A_u,C_v)) = delta_0(u-v).
```

Across all `u,v in F_p`, this kernel is the `p` by `p` identity matrix and has
rank `p`. Any one-bond factorization has matrix rank at most `w`, independent
of how nonlinear the local encoders are. Exact evaluation therefore requires

```text
w >= p.
```

The bound is tight: one-hot local encoders give width `p`.

R98 verifies the identity, rank, and one-hot factorization exactly for
`p=3,5,7,11,13`. Restricted kernels on `4,8,16,32` distinct messages over
`F_101` have the corresponding full ranks.

## Occurrence controls

The occurrence list `[2,2,5,7,9,12,20,31]` with target `2` gives exact
integer count two and an exact dyadic source. Target `3` gives count zero and
bottom. Collapsing equal values reduces the positive count from two to one,
so value-only state loses occurrence semantics.

## Cost gate

The campaign scaling is `p=Theta(B^5)`. The full-field one-bond state and
contraction therefore cost `B^5`.

Even a restriction to `D=Theta(B^(12/5))` distinct messages has minimum width
`D`, outside both the `B^(9/4)` setup/state and `B^(5/4)` fresh-work/workspace
caps. R98 does not claim that the actual EC divisor image contains `D`
distinct messages; that reachability theorem remains unsupplied.

## Admission

Passed obligations: `14/30`

Lane admitted: `false`

Missing gates include an actual-image reachability theorem, a compact
multi-edge digit extractor, actual 5A/5C integer and source replay, projective
infinity, proper-subsum, tangent and multiplicity branches, known-RHS rank,
factor logs, identical descent, a generic-prime family algorithm, and a
complete Shoup comparison.

## Scope boundary

This closes one-bond A/C tensor contractions over the full monic-quadratic
coefficient subfamily and restricted distinct-message sets. It is not a
lower bound on multi-edge digitized encodings or nonalgebraic lookup.

## Exactly one next action

Construct or refute one multi-edge digitized algebraic equality projector over
`F_p`. Freeze the digit or channel extractor from compact A/C divisor circuits
before outcomes; require total cut capacity, extractor state, and setup below
`B^(9/4)`, fresh evaluation and source return below `B^(5/4)`, no `p`-size
lookup/interpolation table, exact occurrence multiplicity and all exceptional
branches, known-RHS rank, factor logs, and identical target descent.
