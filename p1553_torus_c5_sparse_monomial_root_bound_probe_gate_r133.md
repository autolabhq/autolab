# P1553 torus C5 sparse-monomial root-bound gate R133

## Claim boundary

R133 adapts the proof of Kelley Theorem 2.3 to a prime-order cyclic
subgroup and closes only small represented extension-field zero predicates.
It does not close five-mode predicates at the threshold, larger
polylogarithmic support, high-expansion low-SLP predicates, multiple
predicates, adaptive coordinate DAGs, nonzero-value tests, or general
arithmetic circuits and data structures.

It supplies no inside-cap source index, rank, factor logs, identical
descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
PRIME_ORDER_SUBGROUP_ADAPTATION_OF_KELLEY_SPARSE_ROOT_BOUND__ONE_TO_FOUR_EXTENSION_MODES_CANNOT_COVER_Q_TO_THREE_QUARTERS__FIVE_MODES_FIRST_NOT_EXCLUDED__UNIFORM_RANDOM_DECK_PURE_FIFTH_UNION_BOUND_EXCLUDES_UP_TO_QUARTER_LOG2_Q_MODES_WITH_OVERWHELMING_PROBABILITY_ONLY__NO_STRUCTURED_FACTOR_BASE_TRANSFER__LARGER_POLYLOG_LOW_SLP_MULTI_PREDICATE_DAG_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Pinned theorem and subgroup adaptation

The pinned source is Zander Kelley, *Roots of Sparse Polynomials over a
Finite Field*, arXiv:1602.00208, with local SHA-256:

```text
250daacc1b157043cd9c7345036eb332fc69fee8e48a12ba81eddcc2b91995b7
```

Kelley Theorem 2.3 bounds the roots of a represented `t`-nomial using the
size of its largest root coset. Its proof uses exponent reduction modulo a
cyclic domain order, power maps and coset decomposition, a
geometry-of-numbers degree reduction in `t-1` exponents, and the ordinary
polynomial degree bound.

The same proof applies to a cyclic subgroup `H` of prime order `q` in a
coefficient field `K` of characteristic different from `q`. Distinct
exponent modes modulo `q` are distinct characters and are linearly
independent over `K`, so a nonzero represented sparse sum cannot vanish
identically on `H`. Because `q` is prime, the largest root coset therefore
has size one. For `t >= 2`,

```text
number of roots in H <= 2 q^(1 - 1/(t-1)).
```

This is a proof adaptation of a pinned theorem, not a new sparse-root
theorem.

## Deterministic threshold

A balanced color contains `q^(3/4+o(1))` accepted C5 targets. The root
exponents for one through five represented modes are:

```text
t=1: 0
t=2: 0
t=3: 1/2
t=4: 2/3
t=5: 3/4
```

Thus one through four modes cannot vanish on the required color support
asymptotically. Five modes are the first term count not excluded by this
bound. This does not prove that a suitable five-mode predicate exists.

## Random-deck model

Under the frozen model, the deck is an ordered uniform sample without
replacement from `H`, the balanced color partition is fixed independently
of deck values, `q` is prime and not five, and the coefficient field has
size `p^2` with `p < 6q`.

Every atom in a color contributes its pure fifth power to that color's C5
acceptance set. Fifth powering permutes `H`, so these pure fifth powers are
a uniform sample without replacement. If

```text
t - 1 <= log2(q)/4,
```

the root density is at most `1/8`. There are fewer than
`36^t q^(3t)` projectively distinct represented predicates of exact term
count `t`. For

```text
T = 1 + floor(log2(q)/4)
```

and smallest color size `m`, the probability that any such predicate for
any color contains every pure fifth power is at most

```text
4 T 36^T q^(3T) 8^(-m).
```

At `m=q^(3/20+o(1))/4`, this tends to zero. This is only a
uniform-random-deck result. It does not transfer to the deterministic
structured factor base and receives no candidate credit.

## Exact controls

All eight actual pairing controls have prime subgroup order,
`p=6q-1`, and a bijective fifth-power map. Every color's pure fifth
sources replay exactly in `Fp2`, their targets are distinct, and all lie
in the corresponding color acceptance set.

The fixtures have only one or two pure fifth targets per active color.
Their numerical union bounds are above one and receive no probability or
asymptotic credit.

## Admission

Thirteen of twenty-one obligations pass. The deterministic one-through-four
mode negative is admitted. The small-log negative is admitted only under
the explicit uniform-random-deck model. Structured-factor-base transfer,
an inside-cap selector and source index, known-RHS rank, logs, identical
descent, Pollard-rho improvement, Shoup improvement, and breakthrough
remain false.

Disposition:

```text
ADMIT_PINNED_SUBGROUP_SPARSE_ROOT_BOUND_AND_ONE_TO_FOUR_MODE_NEGATIVE__ADMIT_SMALL_LOG_NEGATIVE_UNDER_RANDOM_DECK_MODEL_ONLY__WITHHOLD_STRUCTURED_TRANSFER__PRESERVE_FIVE_MODE_LARGER_POLYLOG_LOW_SLP_AND_MULTI_PREDICATE_DAGS__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Test the first surviving asymmetric selector classes: a five-mode
`F_(p^2)` zero predicate at the `q^(3/4)` threshold, a represented
predicate above the small-log regime, a high-expansion low-SLP predicate,
or a multi-predicate Frobenius-coordinate DAG. Freeze every coefficient
and node, replay exact positive/empty paths and C2+C3 sources, fit
`B^(9/4+o(1))` state and polylogarithmic arbitrary-target work, avoid field
DLP, and charge rank, logs, identical descent, memory, field operations,
and bits.
