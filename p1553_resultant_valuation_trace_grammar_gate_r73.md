# P1553 resultant-valuation trace grammar gate R73

## Classification

- Owner: existing P1553 R9-R10 non-CP trace-contraction frontier; no P1554.
- Evidence: exact finite-field polynomial replay and direct cost theorem.
- Status: `REJECT_RESULTANT_VALUATION_GRAMMAR_ONLY`.
- Labels: `exact-toy`, `representation-bound`, `source-faithful`,
  `novelty-unverified`.
- Cryptanalytic result: no S6 trace contraction, relation campaign,
  factor-log solve, scalar-blind descent, Shoup-bound improvement, or ECDLP
  breakthrough.

R72 rejected two natural centered-carry CP lifts but preserved R9-R10's exact
non-CP projector-trace route. R73 freezes one concrete non-CP candidate:
represent the mandatory rank-two sparse multiplicative convolution as the
root valuation of a parametric product resultant.

The construction is exact. It preserves integer occurrence multiplicity,
zero signatures, queried-rectangle identity, and one joint source. It does
not meet the direct caps. Expanding the parametric resultant has degree
`B^4`; keeping it implicit restores degree-`B^2` specialized polynomial
arithmetic per requested coefficient, or `B^3` for the required batch of `B`
coefficients. Extending the same grammar to the actual pair/triple S6 image
emits the forbidden `B^3` triple occurrence object.

This rejects one frozen grammar, not quotient-algebra transducers, structured
arithmetic circuits, or every exact data structure.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R72 S6 carry report | `7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43` |

## Exact valuation identity

On R10's all-nonzero rank-two stratum, let the two pair occurrence multisets
be

```text
A = {lambda_1(a_1) lambda_2(a_2)},
C = {lambda_3(a_3) lambda_4(a_4)}.
```

Occurrences are retained with multiplicity. Define

```text
R_(A,C)(Z) = product_(u in A, v in C) (Z-u*v).       (1)
```

For every nonzero query `z`,

```text
ord_(Z=z) R_(A,C)
  = #{(u occurrence,v occurrence):u*v=z}
  = sum_u D_12(u) D_34(z/u).                        (2)
```

Equation (2) follows factor by factor: every ordered occurrence pair whose
product is `z` contributes one copy of `(Z-z)`, and no other factor does.
It remains valid when either pair histogram has repeated values.

Equivalently, with

```text
P_A(X) = product_(u in A)(X-u),
Q_(C,z)(X) = product_(v in C)(X-z/v),               (3)
```

a positive coefficient implies

```text
gcd(P_A,Q_(C,z)) != 1.                              (4)
```

The gcd supplies one pair-product value and, through stored pair occurrence
backpointers, one four-label source. The fifth label completes the R10 source.
The degree of (4) is not generally the coefficient in (2): it uses minimum
multiplicity at a shared root, while (2) uses the product of occurrence
multiplicities. Root valuation, not gcd degree, is the exact count.

## Zero strata

Before division, each coordinate is partitioned into the four disjoint states

```text
(x_i,y_i) in {(0,0),(0,nonzero),(nonzero,0),
              (nonzero,nonzero)}.                  (5)
```

For nonzero rank-two coefficients, an off-all-nonzero pattern is a root
exactly when at least one `x_i` and at least one `y_i` vanish. Its exact
multiplicity is the product of five unary status counts. There are only
`4^5` patterns.

R73 checks this formula against direct enumeration. Its zero-signature
mutation has exact count

```text
zero-stratum count = 304,
all-nonzero count = 0,
direct tensor count = 304.                         (6)
```

No division by zero, omitted branch, or modular count decoding is hidden.

## Exact replay

The frozen field is `F_1009`, with five public size-four decks. It includes:

1. a blind all-nonzero target with count zero;
2. a forced positive all-nonzero target with count one; and
3. the zero-signature mutation in (6).

For all twelve fifth-label coefficient queries:

```text
resultant root valuation = direct convolution count. (7)
```

A duplicate-heavy unit control uses two copies on each side. The desired
coefficient is four, the root valuation is four, and a source is recovered.

Splitting the first coordinate into its two canonical children gives counts
one and zero. Their sum equals the parent count one, and the positive child
returns the original occurrence-labelled source. Thus the finite replay
preserves the adaptive rectangle identity needed for dyadic source recovery.

## Direct cost fork

Each pair multiset has at most `B^2` occurrences.

### Expanded route

The polynomial in (1) has

```text
degree |A|*|C| <= B^4.                            (8)
```

Representing its coefficients costs `B^(4+o(1))` work and state, above the
`B^(9/4+o(1))` setup/state cap.

### Specialized route

Keeping (1) implicit avoids the `B^4` coefficient vector, but one query (3)
uses degree-`B^2` polynomials. Standard product-tree, resultant,
subresultant, or gcd evaluation costs

```text
B^(2+o(1))                                       (9)
```

per coefficient. R10 requires `B` target-shifted coefficients, so the direct
batch costs

```text
B^(3+o(1)).                                      (10)
```

Even the single coefficient exponent two exceeds the
`B^(5/4+o(1))` online/workspace cap. Fast univariate gcd after both
polynomials are represented does not remove their input degree.

### Actual S6 extension

The R10 survivor is

```text
C_(R,I) = sum_(a_3,a_4,a_5) h_I(V_R(a_3,a_4,a_5)).
```

Replacing the second pair multiset in (1) by the actual triple signature
multiset creates `B^3` occurrences before any resultant is evaluated. Its
faithful occurrence polynomial and source map therefore restore
`B^(3+o(1))` state/work. R73 supplies no compressed trace transducer that
avoids this object.

## Scope

The result proves exact semantics and direct costs for
`resultant_valuation_v1`. It does not prove an arithmetic-circuit,
cell-probe, quotient-algebra, or data-structure lower bound. In particular,
it leaves open a structured transposed trace evaluator that never represents
the degree-`B^4` output polynomial or a degree-`B^2` target specialization.

No relation density, independent signed rank, factor-base logarithms,
linear-algebra cost, or identical fresh-target descent is supplied.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_resultant_valuation_trace_grammar_r73.py` | `f3066472d96f1ccbb4e5c30dc74ba48101c713536eadd89ff8b5c4b254e0780a` |
| `p1553_resultant_valuation_trace_grammar_report_r73.json` | `00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915` |
| `frozen_noncp_trace_circuit_grammar.json` | `6a0ee3c3941944fe78770ee027cd1a466fb287596056691284f9a8e3a5eb1b45` |
| `restricted_projector_trace_replay.json` | `4b17e06c6c3bd838da1859062abfae8dbecb563ef18048ed2a67628ccfde5d31` |
| `rank_two_sparse_convolution_control.json` | `09b9aa3d324f66e2c8c729b4ebd8225eb708e75ed5e0b487524c18de1d02eb8e` |
| `dyadic_joint_source_and_direct_cost_ledger.json` | `6d2c07a623da4dc982728eabd563fcde6ae03ef4dfb68eb7a04fb8f0166b0cae` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_resultant_valuation_trace_grammar_r73.py` | `40577d69b2518354e82015d416ea000cae87dcefd60e00b65ca1da51a5336d0b` |

Targeted replay:

```text
Ran 3 tests in 0.054s
OK
classification=RESULTANT_VALUATION_GRAMMAR_EXACT_BUT_OVER_CAP
rank_two=True
lane_admitted=False
```

## Disposition

```text
REJECT_RESULTANT_VALUATION_GRAMMAR_ONLY__EXACT_R10_RANK_TWO_MULTIPLICITIES__ZERO_STRATA_AND_DYADIC_SOURCE_REPLAY_PASS__EXPANDED_RESULTANT_DEGREE_B4__SPECIALIZED_QUERY_B2__B_QUERY_BATCH_B3__ACTUAL_TRIPLE_EXTENSION_B3__NO_UNRESTRICTED_CIRCUIT_LOWER_BOUND__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Freeze a quotient-algebra trace-transducer grammar whose persistent state is
built only from dyadic unary subproduct trees, not expanded product
resultants. It must evaluate both the exact R10 rank-two coefficient batch and
the actual S6 pair/triple pullback within `B^(9/4+o(1))` setup/state and
`B^(5/4+o(1))` fresh-target work/workspace, preserve zero strata and rectangle
identity, and return one occurrence-labelled source. Reject it if any
transposed operation first constructs the degree-`B^2` target polynomial,
the degree-`B^4` parametric resultant, or the `B^3` triple signature deck.
