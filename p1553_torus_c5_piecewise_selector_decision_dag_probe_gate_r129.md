# P1553 torus C5 piecewise-selector decision-DAG gate R129

## Claim boundary

R129 closes sequential scans of piecewise-constant C2 translate branches
and explicit target-to-branch routing tables for the R128 cap-tight
singleton-C3 interface.

It does not close compact shared-predicate decision DAGs, high-degree
low-SLP rational selectors, adaptive cell probes, filtered decks with
exploitable collisions, or general arithmetic circuits and data
structures. Branch count is not treated as a query lower bound. R129
supplies no asymptotic complete source index, rank, factor logs, identical
descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
INJECTIVE_SOURCE_MONOMIAL_PIECEWISE_C2_SELECTOR_NEEDS_TURAN_OPTIMAL_B3O2_BRANCHES__ALL_EIGHT_ACTUAL_OPTIMAL_BRANCH_COVERS_RETURN_EXACT_SOURCES_AND_EMPTY_CERTIFICATES__SEQUENTIAL_SCAN_B3O2_QUERY_AND_EXPLICIT_ROUTER_B15O4_STATE_REJECTED__DEGREE_ONLY_FORCES_LOGARITHMIC_SLP_MULTIPLICATIONS__COMPACT_SHARED_PREDICATE_DAG_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Branch theorem

In the inherited iid-distinct-support model, Ck is the injective support of
degree-k commutative monomials in

```text
n = B^(3/4+o(1))
```

deck atoms. A piecewise-constant branch labeled by a degree-two monomial
covers the degree-five targets divisible by that label.

Every pure fifth power `Xi^5` forces the square branch `Xi^2`. Once all
squares are selected, only square-free degree-five monomials remain. A
cross-term branch is an edge, and the selected edges must hit every
five-vertex subset. Equivalently, the graph of unselected edges is
`K5`-free.

Turan's theorem gives

```text
Lmin = n + binom(n,2) - ex(n,K5).
```

This is attained by partitioning the atoms into four balanced parts,
selecting all squares and every within-part pair. Every square-free
five-subset contains two atoms in one part. Therefore

```text
Lmin = Theta(n^2) = B^(3/2+o(1)).
```

## Representation cost

Store the complete C3 dictionary once in `B^(9/4+o(1))` state. Scanning
the optimal branch family and testing `Y/x` in that dictionary uses
`B^(3/2+o(1))` arbitrary-target probes, so it misses the polylogarithmic
query cap even though the setup fits.

An explicit target-to-branch router uses one entry per C5 target, or
`B^(15/4+o(1))` state, above setup.

A balanced branch tree has only logarithmic depth if its internal union
predicates are compact and exact. R129 neither constructs nor lower-bounds
such shared predicates. Likewise, degree `B^(9/4)` forces only
`Omega(log B)` multiplication depth in an unrestricted arithmetic SLP,
which is compatible with the query cap.

## Actual controls

For each of the eight R82 pairing decks, R129 constructs the four-part
optimal branch family and a shared exact C3 dictionary.

Every control verifies:

- C2, C3, and C5 source products are injective;
- the Turan branch count agrees with exhaustive finite minimization;
- every degree-five source monomial is covered;
- every positive target returns a replayable C2+C3 source;
- zero is rejected after scanning every selected branch.

For deck sizes `3, 5, 6, 7`, the exact selected branch counts are
`3, 6, 8, 10`. No pairing-image discrete logarithm is used. These finite
controls receive no asymptotic credit.

## Scope limits

The branch theorem is scoped to injective commutative source monomials and
piecewise-constant C2 translates. Algebraic collisions in a specially
filtered deck could change the support problem. More importantly, a
compact shared-predicate DAG could route to one of the
`B^(3/2+o(1))` leaves without scanning them. R129 proves no generic SLP,
cell-probe, RAM, decision-DAG, or data-structure lower bound.

## Admission

Twelve of twenty obligations pass. Piecewise selector semantics and the
scoped Turan branch negative are admitted. The compact shared-predicate
DAG, asymptotic five-source index, known-RHS rank, logs, identical descent,
Pollard-rho improvement, Shoup improvement, and breakthrough obligations
remain false.

Disposition:

```text
ADMIT_ACTUAL_OPTIMAL_PIECEWISE_BRANCH_COVERS_AND_SOURCE_SEMANTICS_ONLY__REJECT_SEQUENTIAL_BRANCH_SCAN_AND_EXPLICIT_TARGET_ROUTER_AT_FROZEN_CAPS__PRESERVE_SHARED_PREDICATE_DAG_AND_HIGH_DEGREE_LOW_SLP__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute a compact shared-predicate decision DAG for the
`B^(3/2)` optimal C2 translate branches, or a high-degree low-SLP selector.
It must choose a valid branch in polylogarithmic arbitrary-target work
without an explicit `B^(15/4)` target table, return exact C2+C3 sources or
an empty certificate, fit `B^(9/4+o(1))` total state, avoid field DLP, and
include rank, logs, identical descent, memory, field-operation, and bit
costs.
