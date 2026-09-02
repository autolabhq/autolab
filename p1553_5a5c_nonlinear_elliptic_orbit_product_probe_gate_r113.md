# P1553 R113 nonlinear elliptic orbit-product gate

## Scope

R113 tests the surviving R112 exception in two standard forms: a fixed
translation orbit over the `C3` endpoint set and a nonlinear prefix/product
tree over the exact `C5` membership leaves.

## Exact replay

- All target and no-relation product roots are exact on 16 actual and
  matched instances.
- Product-tree zero leaves return all 18 R112 sources, including both double
  fibers, R105 markers, and R108 weight 14400.
- Every `C3` endpoint set is a nonempty proper subset of a prime-order group.
  Its translation stabilizer is therefore trivial, and its canonical
  ordering has nonconstant successor differences.
- The endpoint zero-divisor support occupies at least 99.4 percent of each
  toy source body.

## Cost boundary

A nonzero fixed translation in the prime-order group has full orbit length
`q = B^(5+o(1))`, outside setup. The nonlinear prefix recurrence has order
one and one scalar of state, but its transition still evaluates all
`B^(3+o(1))` distinct `C5` membership leaves. A standard product tree stores
the same number of leaves/nodes. Compiling all target-independent endpoint
zeros instead has `B^(5+o(1))` occurrence degree and state.

This closes fixed-translation orbits, standard product trees, and the claim
that bounded prefix state implies bounded work. It does not prove a lower
bound for arbitrary variable-coefficient or transposed evaluators.

## Disposition

Preserve one exact exception: a transposed nonuniform `C5` leaf generator
that evaluates the target existence functional and an adjoint source without
forming `B^3` leaves or `B^5` endpoint support.

No inside-cap locator, factor-log solve, identical target descent,
generic-prime Shoup improvement, or ECDLP breakthrough is supplied.

## Exactly one next action

Construct or refute one transposed nonuniform `C5` leaf generator, charging
target updates, source adjoint, complete charts, markers, logs, and descent.
