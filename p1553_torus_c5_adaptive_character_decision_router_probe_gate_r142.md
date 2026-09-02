# P1553 torus C5 adaptive character decision-router gate R142

## Claim boundary

R142 closes direct adaptive decision trees over the frozen nonidentity
deck parameters on the eight actual controls. It does not close arbitrary
Mobius parameters, arithmetic combinations of character values, compiled
algebraic circuits, RAM, cell probes, or structured asymptotic factor-base
families.

It supplies no inside-cap source index, relation rank, factor logs,
identical target descent, Pollard-rho improvement, Shoup improvement, or
ECDLP breakthrough.

Classification:

```text
ADAPTIVE_DECK_PARAMETER_MU6_TREE_EXHAUSTIVE__FIVE_OF_EIGHT_CONTROLS_HAVE_FULL_SIGNATURE_OBSTRUCTIONS__THREE_EXACT_TREES_REQUIRE_33_360_645_LEAVES__ALL_CONTROLS_FAIL_FINITE_B9_OVER_4_LEAF_COMPARATOR__ALL_ACCEPTED_C2_C3_AND_INVERSE_EMPTY_PATHS_REPLAY__RANDOM_MODEL_PREDICTS_B5_LEAVES_WITHOUT_STRUCTURED_CREDIT__ARBITRARY_PARAMETERS_AND_ALGEBRAIC_COMPOSITION_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Frozen grammar

Each internal node chooses one unused parameter from the nonidentity
pairing deck, evaluates

```text
chi_z(x) = T_z(x)^q in mu_6,
```

and takes the branch named by that exact value. Repeating a parameter on
the same path cannot refine the path, so omitting repeats loses no tree in
this grammar.

An accept leaf must contain no inverse-empty target and must name one C2
pair occurring in every positive five-source multiset at the leaf. A
reject leaf may contain inverse-empty targets only. These conditions are
both necessary and sufficient for the frozen source-reporting contract.

The dynamic program enumerates every parameter choice at every
inadmissible cell. It minimizes leaf count, then maximum depth, then node
count and parameter index. Its result is exact for this finite grammar.

## Exact controls

All eight C5 supports are injective, and every inverse of a positive
target is absent from the support.

Five controls retain at least one inadmissible cell after all deck
parameters are queried. No adaptive tree in the grammar can refine such a
full-signature cell.

The other three controls have exact trees with minimum leaf counts and
maximum depths

```text
(33, 3), (360, 5), (645, 5).
```

Every one exceeds the corresponding finite
`floor(B^(9/4))` leaf comparator. Each accepted positive replays to a
valid C2 pointer, a concrete C3 certificate, the exact target product,
and the original sorted five-source multiset. Every inverse-empty target
replays to a reject leaf.

These finite comparisons receive no asymptotic credit.

## Random comparator

For independent uniform five-subsets of a `B`-element deck, a fixed pair
is contained with probability

```text
20 / (B*(B-1)).
```

The union bound that `m` independent sources share some pair is

```text
binom(B,2) * (20/(B*(B-1)))^m.
```

At `m=2` this is `200/(B*(B-1))=o(1)`. Under an additional independent
random-signature model, routable cells therefore approach singleton
targets and direct-router state approaches `B^5`. This is a comparator,
not a theorem about the structured character labels.

Nearby shifted-character work gives useful context but no transfer:
[van Dam and Hallgren](https://arxiv.org/abs/quant-ph/0011067) give
efficient quantum algorithms for shifted quadratic-character problems,
while [Bourgain, Garaev, Konyagin, and
Shparlinski](https://arxiv.org/abs/1110.0812) study classical hidden
shifted powers. R142 claims neither a reduction to those models nor a
classical lower bound.

## Admission

Sixteen of twenty-seven obligations pass. The exact frozen-grammar
negative, full-signature obstruction witnesses, optimal finite trees, and
complete positive/negative replays are admitted. The nonlinear source
router and the campaign lane are not admitted.

Disposition:

```text
ADMIT_EXACT_FROZEN_DECK_LANDMARK_TREE_NEGATIVE_ONLY__REJECT_DIRECT_DECK_PARAMETER_ROUTER__PRESERVE_ARBITRARY_PARAMETER_AND_ALGEBRAIC_CHARACTER_COMPOSITION_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Leave direct deck-landmark trees. Derive an algebraic composition of
`chi_z(x)` across the C2-by-C3 split, or admit a broader parameter family
with a proved sub-`B^(9/4)` compiled-state bound. Freeze the composition
law, arbitrary-target branch trace, C2 pointer, C3 certificate,
inverse-empty rejection, reverse five-source replay, and full memory,
field-operation, extension-degree, bit, rank, factor-log, and
identical-descent costs.
