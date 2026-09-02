# P1553 5A/5C Compact Pre-Endpoint S3/FFE Pushdown Gate R104

Date: 2026-07-29

Status: `MERGEABLE_PUSHDOWN_EXACT_BUT_OVER_CAP__ACTUAL_DECK_CIRCUIT_OPEN`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can a sign-resolved, target-specialized S3/FFE recurrence be pushed through
the compact `D_A,D_C` addition circuit before endpoint emission while
preserving exact integer multiplicity and one coupled source inside the
`B^(9/4)` setup and `B^(5/4)` fresh-work caps?

R104 freezes a universal mergeable child-state grammar, full projective sign
state, target residual transition, optional FFE encoding, and a source
adjoint before outcomes.

## Exact recurrence

After sign resolution, the correct S3 branch is exactly

```text
R = T-L.
```

Thus the predicate is the elliptic group-algebra coefficient test
`[L+R=T]`. The complete homogeneous multiset recurrence is

```text
H_(i,k) = H_(i-1,k) + [P_i] H_(i,k-1)
```

in `Z[E(F_p)]`. On all eight frozen actual families, this recurrence equals
direct `5A+5C` enumeration, and every returned source replays. A reverse
target-specialized recurrence materializes the residuals `T-(A+4C)`.
All 32 positive, blind, maximum-multiplicity, and projective-identity
queries reproduce exact counts and sources.

## Mergeable-state separation

Fix a target `T`. If two child histograms differ at endpoint `P`, merge each
with the singleton context `[T-P]`. Their exact outputs at `T` differ.
Therefore a universal mergeable summary must distinguish every endpoint
coefficient.

For `D` distinguishable suffix endpoints:

```text
nonlinear summaries: at least 2^D states and D bits
linear/semilinear summaries: a D-by-D identity readout minor
FFE scalar extension: the same identity-minor rank D.
```

The actual `1A+4C` endpoint maps are injective in every frozen family. Their
finite singleton-context matrices have ranks `30, 30, 210, 210, 378, 378,
840, 840`, equal to their suffix occurrence counts.

Asymptotically the distinguishable suffix body is `B^(14/5)`, above both
caps. The `4A+1C` prefix body remains `B^(11/5)`, inside setup.

## Scope

This closes explicit residual sets and universal mergeable nonlinear,
linear, semilinear, and FFE child summaries. It is not a word lower bound
for one fixed actual deck represented by a nonlinear, non-mergeable,
target-specific arithmetic circuit or short program.

## Admission

Passed obligations: `22/33`

Mergeable pre-endpoint pushdown class closed: `true`

Full lane admitted: `false`

Missing gates include an actual-deck-specific non-mergeable circuit,
known-RHS rank, factor logs, identical descent, the complete charged
workspace bound, and a Shoup improvement.

## Exactly one next action

Construct or refute one actual-deck-specific nonlinear, non-mergeable target
pullback circuit for the frozen `D_A,D_C` program. Inject `T` before any
child summary, expand every arithmetic/FFE gate, return exact multiplicity
and one source inside both caps, and replay matched random-deck, sign,
infinity, tangent, blind, rank, factor-log, and identical-descent controls.
