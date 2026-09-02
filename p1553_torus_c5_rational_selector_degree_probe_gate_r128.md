# P1553 torus C5 rational-selector degree gate R128

## Claim boundary

R128 closes dense single rational C2 selectors and explicit target-to-C2
selector tables for the R127 cap-tight singleton-C3 interface.

It does not close high-degree low-SLP rational functions, compact piecewise
branch selectors, adaptive cell probes, shared transposed selector
evaluation, or general arithmetic circuits and data structures. Algebraic
degree is not treated as a circuit-size lower bound. R128 supplies no
complete source index, rank, factor logs, identical descent, Shoup
improvement, or ECDLP breakthrough.

Classification:

```text
RATIONAL_C2_SELECTOR_FIBER_COUNT_FORCES_DEGREE_AT_LEAST_C5_OVER_C2_B9O4__ALL_EIGHT_ACTUAL_CANONICAL_SELECTOR_INTERPOLANTS_HAVE_FULL_C5_MINUS1_DEGREE_AND_EXACT_SOURCES__DENSE_B9O4_SELECTOR_FITS_SETUP_BUT_MISSES_POLYLOG_QUERY__HIGH_DEGREE_LOW_SLP_OR_COMPACT_PIECEWISE_SELECTOR_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Rational selector theorem

Let

```text
R(Y) = A(Y)/D(Y)
```

be defined on every positive C5 target and return a C2 factor `x` for which
`Y/x` belongs to C3. Put

```text
d = max(deg A, deg D).
```

For each `x` in C2, the equation

```text
A(Y) - x D(Y) = 0
```

has at most `d` roots unless the selector is identically `x`. A constant
selector covers at most `|C3|` distinct products `x*C3` and cannot cover C5
when `|C5|>|C3|`. Therefore a nonconstant selector covering all positive
targets satisfies

```text
|C5| <= |C2| d,
d >= ceil(|C5|/|C2|).
```

Under the inherited iid-distinct-support exponents,

```text
d >= B^(15/4-3/2+o(1))
  = B^(9/4+o(1)).
```

## Representation cost

A densely represented numerator and denominator at the lower-bound degree
use `B^(9/4+o(1))` coefficients and fit the setup cap exactly. Standard
dense arbitrary-target evaluation still costs `B^(9/4+o(1))` field
operations and misses the polylogarithmic query cap.

An explicit target-to-C2 selector table has one entry for each positive C5
target, or `B^(15/4+o(1))` state, above setup.

The degree theorem does not reject a high-degree rational function with a
polylogarithmic straight-line program. Nor does the total-degree count
reject a compact piecewise family if a cheap exact branch selector exists.
Those are the preserved routes.

## Actual controls

For each of the eight R82 pairing decks, R128 chooses the first canonical
C2+C3 split for every distinct C5 target and interpolates the unique
polynomial selector on that domain.

Every control verifies:

- the selector domain is the complete distinct C5 support;
- every selected C2+C3 source replays its target;
- each selected-C2 fiber has size at most `|C3|`;
- Newton interpolation returns every selected C2 value exactly;
- the unique interpolation polynomial has degree `|C5|-1`.

The observed full degree is finite evidence only and receives no asymptotic
or circuit-complexity credit. No pairing-image discrete logarithm is used.

## Scope limits

The asymptotic theorem is a fiber-count degree bound for one globally
defined rational selector. Repeated squaring demonstrates in general why
large degree need not imply a large arithmetic circuit. Compact branch
routing could also evaluate only one low-cost selector per target. R128
preserves those possibilities and proves no generic SLP, cell-probe, RAM,
or data-structure lower bound.

## Admission

Twelve of twenty obligations pass. Rational selector semantics and the
scoped degree negative are admitted. The low-SLP or piecewise selector,
complete five-source index, known-RHS rank, logs, identical descent,
Pollard-rho improvement, Shoup improvement, and breakthrough obligations
remain false.

Disposition:

```text
ADMIT_ACTUAL_RATIONAL_SELECTOR_INTERPOLATION_AND_SOURCE_SEMANTICS_ONLY__REJECT_DENSE_SINGLE_SELECTOR_AND_EXPLICIT_TARGET_TABLE_AT_FROZEN_CAPS__PRESERVE_HIGH_DEGREE_LOW_SLP_AND_COMPACT_PIECEWISE_SELECTOR__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one high-degree low-SLP or compact piecewise rational
C2 selector for the cap-tight singleton-C3 index. It must evaluate in
polylogarithmic arbitrary-target work despite degree `B^(9/4+o(1))`, return
a matching C2+C3 source or exact empty certificate, use at most
`B^(9/4+o(1))` state, avoid field DLP, and expose branch routing, rank,
logs, identical descent, memory, field-operation, and bit costs.
