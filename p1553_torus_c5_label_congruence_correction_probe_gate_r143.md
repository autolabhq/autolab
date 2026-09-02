# P1553 torus C5 label-congruence correction gate R143

## Claim boundary

R143 closes total composition laws computed only from fixed Mobius-character
signatures and one ideal projected four-list merge that must emit explicit
relation rows. It does not close arithmetic circuits over character values,
RAM or cell-probe algorithms, nonuniform structured factor bases, compressed
linear algebra, or implicit summation-polynomial/FFE relation-span operators.

It supplies no compact source locator, known-RHS rank, factor logs, identical
target descent, generic-prime family algorithm, Pollard-rho improvement,
Shoup improvement, or ECDLP breakthrough.

Classification:

```text
EXACT_MOBIUS_CROSS_RATIO_PRODUCT_DEFECT__UNKNOWN_C2_OPERAND_REMAINS__PRIME_ORDER_TOTAL_LABEL_ONLY_PRODUCT_LAW_CONSTANT_OR_INJECTIVE__TWO_OF_EIGHT_FROZEN_CONTROLS_NONDETERMINISTIC__OTHER_SIX_REQUIRE_FULL_C2_BY_C3_TABLE__IDEAL_EXPLICIT_FOUR_LIST_ROW_ENVELOPE_MAX_BETA_1_MINUS_BETA_BOTTOMS_AT_RHO__IMPLICIT_TRANSPOSED_RELATION_SPAN_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Product defect

For `t = x*y`, the exact defect

```text
delta_z(x,y) = chi_z(t)/(chi_z(x)*chi_z(y))
```

is the `q`th power of

```text
-(t+x*z+x)*(x+z+1)*(t*z+t+z)
/
((t+z+1)*(t*z+t+x*z)*(x*z+x+z)).
```

As a rational function of `x`, this has two numerator roots and two
denominator roots. The degree is fixed, but evaluating the formula still
requires the unknown C2 operand `x`; it is a correction identity, not a
source locator.

Every defined defect in all eight frozen controls replays exactly. No field
discrete-log oracle is consumed.

## Label congruence

Let `G` have prime order. If a signature `sigma:G->A` and total law
`F:A*A->A` satisfy

```text
sigma(x*y) = F(sigma(x), sigma(y))
```

for every `x,y`, equality of signatures is a multiplication congruence.
Its identity class is a subgroup and every class is its coset. The subgroup
is therefore `G` or `{1}`, so `sigma` is constant or injective.

Consequently, a nonconstant tuple of `k` sextic labels with `6^k < |G|`
cannot support a total label-only product law. This theorem does not rule
out algorithms that retain field elements, use arithmetic circuits, or act
implicitly on a relation span.

The exact frozen controls agree with the boundary. Two controls have no
deterministic C2-by-C3 composition for any nonempty deck-parameter subset.
In the other six, the smallest deterministic table has one entry for every
C2-by-C3 operand pair. All eight fail the finite `floor(B^(9/4))` table
comparator. These finite results receive no asymptotic credit.

## Explicit-row envelope

For factor-base size `n=N^beta` and an ideal compatible projection of size
`N^mu`, a projected pair list has exponent `2*beta-mu`, while the expected
four-source output count has exponent `4*beta-mu-1`.

Supplying `N^beta` explicit rows for `N^beta` factor-log unknowns requires

```text
mu <= 3*beta - 1.
```

The merge work is therefore at least `N^(1-beta)`, and row output costs
`N^beta`. The total envelope

```text
max(beta, 1-beta)
```

is minimized at `beta=1/2` with exponent `1/2`. At the campaign value
`beta=9/20`, the ideal merge exponent is `11/20`.

This is an exponent boundary for an explicit two-pair merge, not a lower
bound for implicit relation-span operators or generic ECDLP algorithms.

## Admission

Fifteen of twenty-three obligations pass. The exact defect identity,
label-only congruence negative, frozen correction-table controls, and
explicit-row birthday envelope are admitted. The implicit relation-span
operator and campaign lane are not admitted.

Disposition:

```text
ADMIT_EXACT_PRODUCT_DEFECT_AND_LABEL_ONLY_NEGATIVE__ADMIT_EXPLICIT_ROW_BIRTHDAY_BOUNDARY__REJECT_COMPACT_LABEL_COMPOSITION__PRESERVE_IMPLICIT_TRANSPOSED_SUMMATION_POLYNOMIAL_FFE_OPERATOR__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Build one target-batched transposed summation-polynomial/FFE relation-span
operator at `beta=9/20`. It must expose enough independent known-RHS row
action to solve factor logs and support identical target descent in
`N^(1/2-epsilon)` total field and bit work with `N^(9/20+o(1))` state. It
may not enumerate the `N^(11/20)` pair merge, emit `N^(9/20)` rows without
charging them, consume a DLP or root oracle, or infer asymptotic rank from
finite fixtures.
