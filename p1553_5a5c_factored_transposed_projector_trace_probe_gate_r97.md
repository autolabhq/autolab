# P1553 5A/5C Factored Transposed Projector Trace Gate R97

Date: 2026-07-29

Status: `SCOPED_NEGATIVE`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can transposition evaluate the Fermat zero projector and a source-reporting
range child from the compact 5A/5C divisor circuits without retaining the
`D = Theta(B^(12/5))` source dimension?

R97 freezes and tests the standard pointwise grammar:

1. `delta_0(h_i) = 1 - h_i^(p-1)`;
2. reverse-mode adjoints of the pointwise projector;
3. every node of the complete dyadic interval-mask tree; and
4. the scalar product `P(h) = product_i h_i` and its reverse gradient.

It does not freeze or refute an arbitrary nonlinear tensor-tower trace whose
node states are constructed directly from compact A/C divisor circuits.

## Exact controls

Over `F_101`, the projector derivative is

`delta_0'(h) = h^(p-2)`.

For blind nonzero vectors of dimensions `4, 8, 16, 32`, the pointwise
Jacobian has rank `D`. The pullbacks of all `2D-1` dyadic interval masks also
have rank `D`; the leaf masks alone expose the full source space.

The finite source controls are exact:

- one zero gives projector count one and the dyadic replay returns its index;
- no zero gives count zero and bottom;
- two zeros give projector count two.

The scalar projector gradient has a zero entry at every zero source and
requires a materialized `D`-word adjoint to recover the complement. The
product gradient has exactly one nonzero entry when there is exactly one zero,
but becomes the zero vector when there are two zeros. It therefore cannot
provide multiplicity-complete source semantics.

## Cost gate

For the smaller R84 source side, `D = Theta(B^(12/5))`.

- Pointwise Fermat work is `Theta(D log p)`.
- A full transposed adjoint contains `Theta(D)` field words.
- A balanced product tree has `D` leaves, `D-1` multiplications, `2D-1`
  forward nodes, and `Theta(D)` reverse adjoints.
- Complete dyadic linearized range state spans `D` coordinates.

These standard implementations miss both the `B^(9/4)` setup/state cap and
the `B^(5/4)` fresh-work/workspace cap.

## Admission

Passed obligations: `13/28`

Lane admitted: `false`

Missing gates include a compact nonlinear tensor-tower trace, actual 5A/5C
integer and source replay, projective infinity, proper-subsum, tangent and
multiplicity branches, known-RHS rank, factor logs, identical descent, a
generic-prime family algorithm, and a complete Shoup comparison.

## Scope boundary

The full-rank finite sweep is not an asymptotic lower bound. R97 closes
pointwise Fermat powering, source-valued product trees, their reverse
adjoints, and linearized dyadic masks only.

## Exactly one next action

Construct or refute one nonlinear tensor-tower projector trace whose node
states are derived directly from compact A/C divisor circuits rather than
source leaves, quotient bases, moment vectors, or linearized dyadic masks.
Freeze its tensor algebra and target transition before outcomes; require an
exact integer count, one complete coupled source, all exceptional branches,
`B^(9/4)` setup/state, `B^(5/4)` fresh work/workspace, known-RHS rank, factor
logs, and identical target descent.
