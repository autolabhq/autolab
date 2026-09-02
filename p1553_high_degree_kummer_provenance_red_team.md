# Independent P1553 high-degree Kummer provenance R12 red-team transcript

Reviewer: `019f7d16-e67d-7bf2-a60a-1c2ba91519d3`
Record: coordinator transcription of the independent theorem-only response
Evidence: read-only review; no run

## Terminal verdict

```text
ACCEPT__INDEXED_PSI_RESPECTING_HEADERS_AT_LEAST_S_B_OVER_D__DENSE_STATE_CAP_FORCES_D_AT_LEAST_B_THREE_QUARTERS__FIFTH_LABEL_ERASED_FIBERS_AT_MOST_8D_SQUARED__DIRECT_PREIMAGES_COST_S_B_STATE__COUNT_PLUS_ONE_WITNESS_COSTS_S_B_CONSTRUCTION__PRIME_ORDER_ISOGENY_AND_ECFFT_MAPS_INJECTIVE__COMPOSITE_TORSION_AND_ARITHMETIC_PROGRESSION_CONTROLS_ACCEPTED__IMPLICIT_NONISOGENY_IMAGE_AND_INVERSION_OPEN__NO_NEW_IDEA__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

The first reviewed draft received `REVISE`. The producer then separated a
count-plus-one-witness construction pass from persistent all-preimage state,
changed auxiliary quotient counts to Kummer `Theta` statements, charged the
complete auxiliary source index, and specified all five arithmetic-progression
decks. The reviewer matched the revised hashes and returned `ACCEPT`.

## Accepted indexed header bound

Let `S` be the number of distinct pair-endpoint Kummer classes, not the number
of pair occurrences. For fixed fifth occurrence `q`, every state with key

```text
(q,psi(u),[f(A+Q)]+[f(A-Q)])
```

lies in one degree-`d` fiber of `psi`. Projective fibers contain at most `d`
distinct points, including ramified, inseparable, and infinity fibers. Thus

```text
K_indexed >= B*ceil(S/d) >= S*B/d.
```

At certified dense support `S=Theta(B^2)`, the `B^(9/4+o(1))` state cap forces
`d=Omega(B^(3/4-o(1)))`. Pair occurrences imply this only after separately
bounding endpoint multiplicity `kappa`, with `S>=B^2/kappa`.

## Accepted erased-key bound

After erasing `q`, retain `(psi(u),D_(u,q))`. For a fixed divisor
`D=[alpha]+[beta]`, put `X=A+Q` and `Y=A-Q`. The odd-order transformation is
bijective. Each of the two assignments of `alpha,beta` has at most `(2d)^2`
signed preimages, so a complete erased-key fiber has at most `8d^2` signed
pairs and no more Kummer-class pairs. Hence

```text
K_erased >= S*B/(8d^2).
```

This does not construct the image or invert a source. It is weaker than R11's
independent `d=Omega(B^(1/2))` useful-merge floor.

## Accepted state and construction split

Four routes are distinct:

1. Fifth-indexed headers obey `S*B/d`.
2. Storing every erased-key preimage uses `S*B` provenance entries.
3. A direct `S*B` pass may retain an exact count and one witness per header;
   its persistent state can be smaller, but its setup work remains `S*B`.
4. An implicit image constructor and algebraic source inverter may avoid that
   pass; this is outside the theorem and remains open.

At dense support, routes 2 and 3 expose a `B^3` charge. Exact replay alone does
not prove `B^3` persistent state.

## Accepted controls

For a separable isogeny `phi:E->E'` of degree `d<N`, the intersection of its
kernel with the prime-order subgroup `G` is trivial. Therefore `phi` and its
induced Kummer map are injective on `G`. Isogeny, Lattes, and ECFFT quotient
maps through degree `O(B)` give exact cheap branch transport but no compression
on the campaign subgroup.

On an auxiliary `C_B x C_B` torsion grid, a quotient kernel of size
`d=B^(3/4)` gives `Theta(B^2)` pair Kummer classes,
`Theta(B^(5/4))` quotient labels, and `Theta(B^(9/4))` label/fifth states. The
source index charges `Theta(B^2)` endpoint entries, `Theta(B^(5/4))` quotient
representatives, and `Theta(B^(3/4))` kernel data. Enumerating and verifying
the `d` kernel candidates fits the online cap. The control fails the
prime-order gate by construction.

Five known-multiple scalar-interval decks and a degree-`O(B)` interpolation
polynomial collapse the frozen endpoint and branch coordinates. Their
fivefold endpoint support is only `O(B)`, so uniform relation density is
`O(B/N)` and even favorable collection needs `Omega(N)` target attempts.

## Scope retained

The theorem does not implement R10's exact rank-two sparse multiplicative
convolution, a fresh-target query, adaptive dyadic restriction, factor-log
recovery, or scalar-blind descent. It is not a lower bound for arbitrary
implicit arithmetic circuits, list-specific non-isogeny maps, or data
structures. No new idea, experiment, Shoup claim, or breakthrough status is
authorized.

## Exactly one next action

Derive or refute the frozen fifth-label-erased implicit algebraic
image-and-inversion route under the stated setup, fresh-target query, adaptive
restriction, R10 rank-two control, source replay, density, rank, factor-log,
and identical blind-descent requirements. A negative closes only that grammar;
a positive image or count remains model-bound until the complete ECDLP path is
proved.
