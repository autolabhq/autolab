# P1553 torus C5 two-atom geometric-progression gate R134

## Claim boundary

R134 closes a single represented extension-field zero or pole predicate
with at most six distinct modes on the asymptotic structured color
supports. It does not close seven-or-more-mode predicates,
multiple-predicate or adaptive DAGs, nonzero-value tests,
high-expansion low-SLP predicates, or general arithmetic circuits and data
structures.

It supplies no inside-cap source index, rank, factor logs, identical
descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
DEGREE_FIVE_TWO_ATOM_SOURCES_FORCE_SIX_TERM_GEOMETRIC_PROGRESSIONS_IN_EVERY_ASYMPTOTIC_COLOR__PRIME_ORDER_RATIO_MAKES_ALL_AT_MOST_SIX_MODE_EVALUATION_MATRICES_VANDERMONDE__SINGLE_REPRESENTED_ZERO_OR_POLE_PREDICATES_THROUGH_SIX_MODES_REJECTED_DETERMINISTICALLY_ON_STRUCTURED_FACTOR_BASE__TWELVE_ACTUAL_PROGRESSION_WITNESSES_EXACT__SEVEN_MODE_MULTI_PREDICATE_NONZERO_VALUE_LOW_SLP_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Two-atom progression theorem

Let `x` and `y` be two distinct atoms in one color part. For
`j=0,...,5`, the degree-five source

```text
x^(5-j) y^j
```

contains five atoms from that color and is accepted by the color rule.
Its target is

```text
z_j = x^5 (y/x)^j.
```

The ratio `r=y/x` is a nonidentity element of the prime-order subgroup, so
it has exact order `q`. The six targets are therefore distinct and form a
geometric progression.

Now let

```text
f(z) = sum_(i=1)^t c_i z^(e_i)
```

have `1 <= t <= 6` nonzero coefficients and distinct exponents modulo
`q`. If `f` vanishes on `z_0,...,z_(t-1)`, then its coefficient vector is
in the kernel of the Vandermonde matrix with nodes `r^(e_i)`. Those nodes
are distinct, and the determinant

```text
product_(i<k) (r^(e_k) - r^(e_i))
```

is nonzero. All coefficients would have to be zero, a contradiction.

Thus no nonzero represented predicate with at most six modes can vanish
on all six targets. The same argument applies separately to a represented
rational numerator-zero or denominator-zero set.

This is a field-independent Vandermonde argument. It uses no random-deck
model and no candidate discrete logarithm.

## Structured factor base

At deck size `n=B^(3/4+o(1))`, every balanced color part has
`Theta(n)` atoms and therefore contains two distinct atoms. Every
asymptotic color acceptance support deterministically contains a six-term
progression of the required form.

Consequently, a single at-most-six-mode zero or pole predicate cannot
equal, or contain, a full color acceptance support. Seven modes are the
first represented term count not excluded by this witness. The theorem
does not rule out combining several small predicates in a decision DAG.

## Exact controls

All eight pairing decks are replayed directly in `Fp2`. Across the 30
active finite colors, 12 have two atoms and supply an exact progression
witness. Every witness has:

- six distinct targets;
- a ratio of exact prime order `q`;
- the exact geometric-progression recurrence;
- six matching degree-five source tuples;
- color multiplicity five for every source; and
- a nonzero sample six-mode Vandermonde determinant.

The remaining 18 active finite colors contain one atom. They have no
two-atom witness and receive no asymptotic credit. This finite gap does not
affect the asymptotic balanced-color theorem.

## Admission

Thirteen of twenty-one obligations pass. The deterministic structured
one-through-six-mode negative and 12 exact finite witnesses are admitted.
An inside-cap surviving selector and source index, known-RHS rank, logs,
identical descent, Pollard-rho improvement, Shoup improvement, and
breakthrough remain false.

Disposition:

```text
ADMIT_DETERMINISTIC_TWO_ATOM_SIX_POINT_PROGRESSION_THEOREM__REJECT_SINGLE_REPRESENTED_ZERO_OR_POLE_PREDICATES_THROUGH_SIX_MODES_ON_ASYMPTOTIC_STRUCTURED_COLORS__ADMIT_TWELVE_EXACT_FINITE_WITNESSES_WITHOUT_FIXTURE_GAP_CREDIT__PRESERVE_SEVEN_MODE_MULTI_PREDICATE_NONZERO_VALUE_AND_LOW_SLP_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Test the first surviving asymmetric classes after the deterministic
six-mode obstruction: a seven-or-more-mode `F_(p^2)` predicate, a
multiple-small-predicate decision DAG, a nonzero-value
Frobenius-coordinate branch, or a high-expansion low-SLP predicate. Freeze
all coefficients and nodes, replay exact positive/empty paths and C2+C3
sources, fit `B^(9/4+o(1))` state and polylogarithmic arbitrary-target
work, avoid field DLP, and charge rank, logs, identical descent, memory,
field operations, and bits.
