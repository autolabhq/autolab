# P1553 torus C5 consecutive-mode predicate gate R131

## Claim boundary

R131 closes dense consecutive-mode color predicates in the actual
order-two pairing fields. The proof is field-independent and therefore
does not rely on transferring a complex uncertainty theorem.

It does not close lacunary finite-field Fourier supports, high-degree
low-SLP predicates, shared multi-predicate decision DAGs, adaptive cell
probes, non-Fourier predicates, or general arithmetic circuits and data
structures. It supplies no inside-cap source index, rank, factor logs,
identical descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
FIELD_INDEPENDENT_ROOT_BOUND_CLOSES_DENSE_CONSECUTIVE_MODE_COLOR_PREDICATES__ALL_EIGHT_ACTUAL_ORDER2_CONTROLS_HAVE_NONZERO_VANDERMONDE_DETERMINANTS_AND_EXACT_DENSE_ANNIHILATOR_ZERO_SETS__B15O4_STATE_AND_SEQUENTIAL_QUERY_REJECTED__LACUNARY_LOW_SLP_OR_NONFOURIER_SHARED_DAG_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Exact theorem

Over every field, a nonzero degree-`d` polynomial has at most `d`
distinct roots. On a nonzero cyclic subgroup, a Laurent polynomial with
consecutive modes

```text
a, a+1, ..., a+s-1
```

has at most `s-1` distinct zeros unless it is identically zero: multiplication
by `X^(-a)` reduces it to an ordinary polynomial of degree at most `s-1`.
Equivalently, the Vandermonde matrix on `s` distinct points and modes
`0,...,s-1` has determinant

```text
product_{i<j} (x_j-x_i) != 0.
```

This statement holds in finite characteristic, including every actual
`ord_q(characteristic)=2` field used by the pairing controls.

## Selector consequence

Partition the `n=B^(3/4+o(1))` deck atoms into four balanced colors as in
R129. A color accepts precisely the degree-five source multisets containing
at least two atoms from that color. For color size `m=Theta(n)`, the exact
count is

```text
binom(n+4,5) - binom(n-m+4,5) - m*binom(n-m+3,4)
  = Theta(n^5)
  = B^(15/4+o(1)).
```

A single consecutive-mode zero predicate for that accepted set therefore
needs `B^(15/4+o(1))` serialized modes. A dense coefficient block misses
the `B^(9/4+o(1))` setup cap, while Horner evaluation misses the
polylogarithmic arbitrary-target query cap. Storing the accepted roots in
an explicit product tree has the same state exponent and still touches
every leaf for one arbitrary target.

This is not a circuit lower bound. A lacunary polynomial can have large
degree with few represented terms, and a high-degree polynomial can have a
short arithmetic circuit.

## Exact controls

All eight inherited pairing decks are replayed in their actual quadratic
coefficient fields. Every active color acceptance count matches the
multiset formula. The accepted target values are distinct, every exact
Vandermonde determinant is nonzero, and the dense root annihilator has
exactly the intended zero set among all C5 targets while rejecting zero.

For each positive source, the first repeated color chooses an R129-optimal
within-color C2 branch. Its remaining C3 source multiplies back to the
target. These verifier source labels establish semantics only; they are not
an online source locator and receive no asymptotic credit. No candidate
discrete logarithm is used.

## Admission

Twelve of twenty obligations pass. The actual-field consecutive-mode
negative and dense cost rejection are admitted. The lacunary or
non-Fourier selector, inside-cap source index, known-RHS rank, logs,
identical descent, Pollard-rho improvement, Shoup improvement, and
breakthrough obligations remain false.

Disposition:

```text
ADMIT_FIELD_INDEPENDENT_CONSECUTIVE_MODE_ROOT_BOUND_AND_EXACT_ACTUAL_ORDER2_CONTROLS_ONLY__REJECT_DENSE_B15O4_PREDICATE_STATE_AND_QUERY__PRESERVE_LACUNARY_LOW_SLP_AND_NONFOURIER_SHARED_DAG__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute a lacunary predicate in the actual
`ord_q(characteristic)=2` fields, or a non-Fourier shared decision DAG. It
must exploit more than a dense consecutive mode block, choose a valid C2
branch in polylogarithmic arbitrary-target work, return exact C2+C3 sources
or an empty certificate, fit `B^(9/4+o(1))` state, avoid field DLP, and
include rank, logs, identical descent, memory, field-operation, and bit
costs.
