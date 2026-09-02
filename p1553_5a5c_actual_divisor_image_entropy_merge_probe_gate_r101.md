# P1553 5A/5C Actual Divisor Image Entropy/Merge Gate R101

Date: 2026-07-29

Status: `LOCAL_STRUCTURED_ORACLES_PASS__FULL_TWO_SIDED_JOIN_OPEN`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can the actual R84 side images, rather than an arbitrary endpoint set, be
queried for exact integer multiplicity and one source inside the direct setup
and online caps?

R101 freezes the public compact atom decks `D_A,D_C`, both offsets, and all
four R82 parameter families before evaluating either side.

## Exact local oracles

For the left image `2A+3C`, R101 stores a weighted dictionary of unordered
`3C` endpoints and streams the unordered `2A` multisets at query time:

```text
stored state = Theta(v^3) = B^(9/5)
fresh work  = Theta(u^2) = B^(4/5).
```

For the right image `3A+2C`, it stores a weighted dictionary of unordered
`2C` endpoints and streams the unordered `3A` multisets:

```text
stored state = Theta(v^2) = B^(6/5)
fresh work  = Theta(u^3) = B^(6/5).
```

Both fit the direct `B^(9/4)` setup/state and `B^(5/4)` online caps. The
dictionaries retain exact multiset weights and one canonical source for each
endpoint. They do not persist leaves of the full side image.

## Finite replay

All 16 side queries from eight actual family/offset instances agree with the
R84 direct multiset histograms for every attained endpoint. Blind targets,
the identity endpoint, and repeated-atom paths are exact. A separate cyclic
collision control has duplicate atom values, multiplicity above one, and
exact weighted source replay.

The construction consumes only public curve points and atom indices; it does
not consume scalar labels.

## Remaining join

A fresh `5A+5C` target requires

```text
find ell in (2A+3C) such that T-ell is in (3A+2C).
```

R101 supplies exact local membership oracles for both predicates, but no
subcap method to select a common `ell`. Enumerating the left and right images
costs `B^(13/5)` and `B^(12/5)` respectively, above the online cap. No exact
joint integer count or jointly coupled source is claimed.

## Admission

Passed obligations: `19/32`

Local side oracles admitted: `true`

Full lane admitted: `false`

Missing gates include the two-sided implicit join, complete joint
multiplicity and exceptional replay, known-RHS rank, factor logs, identical
fresh-target descent, a generic-prime family algorithm, and a full Shoup
comparison.

## Exactly one next action

Construct or refute one two-sided implicit-intersection algorithm combining
the passing `2A+3C` and `3A+2C` local oracles. Freeze shared state and the
target transition before outcomes; require `B^(9/4)` setup/state,
`B^(5/4)` fresh work/workspace, exact joint count and one coupled source,
all exceptional branches, known-RHS rank, factor logs, and identical descent,
without enumerating either full side.
