# Independent P1553 small-root R6 red-team transcript

Reviewers: `019f7c7a-1e58-7b52-a2ea-50d1d8be6af8`,
`019f7c7a-2cb9-75a3-b012-8c93e3377e23`
Record: coordinator transcription of independent theorem-only responses
Evidence: read-only review, no run

## Terminal verdict

```text
REVISE_SCOPED_PARTIAL_INPUT_COMPILER__PAIR_LABEL_SELECTOR_B2_REJECTED__X_CLASS_DOMAIN_LAGRANGE_MAPS_B1_POLYLOG__S6_EXACT_FOR_SIGNED_X_CLASS_RELATIONS__ONE_ROOT_TO_SOURCE_COST_CONSTANT__DENSE_B5_EXPANSION_ONLY_A_REPRESENTATION_GATE__TENSORIZED_SMALL_ROOT_DECISION_WITNESS_UNSUPPLIED__RANK_LOGS_DESCENT_UNSUPPLIED__NO_P1554__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Corrections accepted

The first audit confirmed that the five original labels improve IDEA-049's
input boundary: local degree-`<s_i` coordinate maps are target independent,
cost `B^(1+o(1))` over canonical trees, and avoid the degree-`B^2` pair-index
selector. It also located the first full dense coefficient representation at
`Theta(B^5)` positions, before LLL.

The adversarial audit rejected the original oriented-deck wording. `S_6`
recognizes a union of sign classes, and constant checks per candidate do not
bound the number of wrong-orientation roots. It also corrected three scope
errors:

- replay compatibility of the maps is not a bound on root or exact-miss cost;
- `B^5` is a dense expansion envelope, not a lower bound on every lattice; and
- the required output is exact no-root or one verified root, not every root.

The coordinator then re-froze each factor-base label as a nonidentity x-class
`{P_C,-P_C}` with a canonical representative and sign-closed dyadic
restrictions. The follow-up audit accepted the resulting biconditional:

```text
S_6(x(C_1),...,x(C_5),x(R))=0
iff R + sum_i sigma_i P_(C_i)=O for some sigma_i in {+1,-1}.
```

The target sign is normalized by a global sign flip. Given one returned label
tuple, enumerating `2^5` signs and checking complete projective addition is an
exact source inverse. No wrong-sign root family remains.

## Domain and row semantics

For a node of size `s_i`, legal labels are bound by

```text
D_i(z)=product_(a=0)^(s_i-1)(z-a)=0 mod p.
```

Every x-class stores its global factor-base column and occurrence
backpointers. If a class appears in several colours, the row coefficient is

```text
c_C=sum_(i:C_i=C) sigma_i.
```

The row remains correct but receives rank credit only after this aggregation.
Known-log and blind equations are respectively

```text
r + sum_C c_C log_P(P_C)=0 mod N,
x=-t-sum_C c_C log_P(P_C) mod N.
```

Factor decks exclude `O`. Known-log collection resamples a zero target; a
blind zero target directly gives `x=-t`. Repeated classes and intermediate
exceptional additions are handled by the Semaev existential predicate and
final complete verification, not by an incomplete affine chart.

## Cost and scope

```text
pair-label coordinate selector per target       B^2, rejected,
five x-class domain/coordinate trees             B^(1+o(1)),
one supplied predicate evaluation                B^(1+o(1)),
full dense monomial coefficient envelope         B^5,
required online decision-and-witness cap          B^(5/4+o(1)).
```

The construction is coefficient complete as a separated
`{D_1,...,D_5,F_R}` circuit and exact source inverse. It is not a bounded-root
algorithm, determinant theorem, relation-rank proof, factor-log completion, or
blind-descent complexity proof. General tensorized, sparse-shift, circuit, and
nonlattice locators remain open.

## Exactly one next action

Derive or refute one exact `no root / one verified root` algorithm for the
separated system `{D_1,...,D_5,F_R}`, including a determinant/root-region
theorem and complete construction, negative-answer, workspace, bit, and
dyadic replay costs inside `B^(5/4+o(1))`; do not require all-root enumeration.
