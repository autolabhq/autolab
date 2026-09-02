# P1553 S6 residual decision-diagram probe R74

## Classification

- Owner: existing P1553 R9-R10 non-CP trace-contraction frontier; no P1554.
- Evidence: exact group-law residual-support enumeration on four standardized
  prime-order curves.
- Status: `REJECT_S6_SQUAREFREE_RESIDUAL_RADICAL_MEMOIZATION_GRAMMAR_ONLY`.
- Labels: `exact-finite`, `representation-bound`, `scalar-blind-source`,
  `novelty-unverified`.
- Cryptanalytic result: no S6 trace contraction, relation campaign,
  factor-log solve, scalar-blind descent, Shoup-bound improvement, or ECDLP
  breakthrough.

V23 selected a quotient-algebra trace transducer after R73 rejected the exact
product-resultant valuation grammar. R14 already owns the quotient
minimal-polynomial, trace-Hankel, Krylov, power-projection, and represented
tensor-algebra routes. R74 therefore does not rerun that duplicate. It freezes
a mechanism-distinct non-CP representation: memoize exact squarefree S4
residual supports in a reduced algebraic decision diagram.

On secp256k1, P-256, P-384, and P-521, every triple residual key is distinct
for every frozen prefix `B=6,10,14,18`. The target-independent algebraic
diagram therefore has exactly `B^3` middle states. The outcome-aware Boolean
diagram has only one state for blind targets and two for forced-positive
targets, but constructing that diagram already requires the relation
incidences. It is root-presupposing compression and receives no cost credit.

This rejects one residual-key grammar, not support-adaptive transposed
incidence algorithms, arbitrary decision diagrams, or arithmetic circuits.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R72 S6 carry report | `7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43` |
| R73 resultant-valuation report | `00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915` |
| R14 tensor-trace minimal-polynomial gate | `da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac` |

## Exact residual key

For rational points `P_1,P_2,P_3`, define the Kummer endpoint support

```text
K(P_1,P_2,P_3)
  = {x(P_1 + e_2 P_2 + e_3 P_3):e_2,e_3 in {+1,-1}},
                                                        (1)
```

with a separate tag when a signed sum is the identity. Fixing the sign of
`P_1` loses nothing because simultaneous sign reversal preserves x.

The roots of

```text
S4(x(P_1),x(P_2),x(P_3),z)                      (2)
```

are exactly the x-coordinates in (1), with multiplicity. Hence the projective
squarefree radical of (2), including the infinity tag, is exactly `K`.

For the S6 split

```text
S6 = Res_z(
       S4(x_1,x_2,x_3,z),
       S4(x_4,x_5,x_R,z)
     ),                                          (3)
```

the residual predicate for one triple prefix accepts one suffix exactly when

```text
K(P_1,P_2,P_3) intersects K(R,P_4,P_5).          (4)
```

Thus `K` is a coefficient-complete, multiplicity-free algebraic residual
state for existence. It is target-independent on the prefix side and does
not use scalar labels.

R74 independently constructs sampled S4 polynomials, takes their exact
finite-field squarefree radicals, and matches them to (1) on every family and
prefix.

## Frozen replay

The public R72 SHA-256 decks are reused unchanged. No scalar label or
discrete logarithm is consumed. The exact counts are:

```text
curve families                         4
prefix sizes                   6,10,14,18
prefix instances                      16
target instances                      32
```

For every curve and prefix:

```text
triple occurrences                     B^3
distinct algebraic residual keys       B^3
residual-key collisions                   0.       (5)
```

At `B=18`, each family therefore has

```text
5,832 triple occurrences,
5,832 distinct algebraic states.                  (6)
```

Every blind target has zero incidences and one oracle Boolean state: the
empty residual. Every forced-positive target has exactly one incidence and
two oracle states: empty and the single positive suffix signature. All 16
forced witness instances are recognized.

The finite result is unusually uniform, but it is not an asymptotic
collision theorem for arbitrary deck families.

## Constructive versus oracle diagrams

The algebraic residual diagram can be constructed from (1) or (2) without
knowing target outcomes. Equation (5) forces:

```text
target-independent state       B^3,
prefix construction work       B^3.               (7)
```

Both exceed the `B^(9/4+o(1))` setup/state cap. Constructing the same states
on demand for a fresh target exceeds the `B^(5/4+o(1))` online cap.

The reduced Boolean truth-table diagram is tiny only because almost every
residual function is identically false on the frozen suffix deck. To know
which prefix has the one nonempty signature, the literal constructor compares
`B^3` prefix keys against `B^2` suffix keys:

```text
literal incidence work = B^5.                    (8)
```

Granting the two-state result without (8) presupposes the positive source and
cannot serve as a blind zero certificate. R9's observation that a sparse
projector can have tiny tensor rank has the same boundary: low
outcome-conditioned representation size is not a constructor.

A pair-side endpoint table has `B^2` states and fits setup, but extending it
through the fifth deck produces `B^3` transitions. R73's exact multiplicity
and source control remains mandatory for any compressed replacement.

## R14 deduplication

R14 already proves that a supplied compact trace sequence and certified image
size can be decoded output-sensitively, while standard quotient, Krylov,
minimal-polynomial, norm, and power-projection constructors restore the
represented source width. Calling the same object a quotient-algebra
transducer would duplicate that gate.

R74 instead measures exact residual-state reuse in the actual S4-by-S4 S6
split. It does not alter or strengthen R14 into an unrestricted lower bound.

## Scope

R74 closes only
`s6_squarefree_residual_radical_memoization_v1`. Different algebraic residual
keys could merge states not merged by the squarefree support. A
support-adaptive incidence algorithm could conceivably discover only the
nonempty intersections without constructing every key. Neither possibility
is excluded.

No relation density, independent signed rank, factor-base logarithms, sparse
linear algebra, or identical fresh-target descent is supplied.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_s6_residual_decision_diagram_probe_r74.py` | `04dc04e2cc700793ed973405da6d6a41c7a3635c7410e7858c96f5242ddc84e2` |
| `p1553_s6_residual_decision_diagram_probe_report_r74.json` | `1558482f504bd5e05b112464ee7c6734dd30ee518789735ac7abc8c305a5f740` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_s6_residual_decision_diagram_probe_r74.py` | `9ed44ad10b448914a1e6f5e451bc9c087b761560519f83fc41d19ca38dcc3873` |

Targeted replay:

```text
Ran 3 tests in 0.085s
OK
families=4
algebraic_distinct=True
max_oracle_states=2
lane_admitted=False
```

## Disposition

```text
REJECT_S6_SQUAREFREE_RESIDUAL_RADICAL_MEMOIZATION_GRAMMAR_ONLY__FOUR_STANDARD_CURVES__B6_10_14_18__ALL_TRIPLE_ALGEBRAIC_KEYS_DISTINCT__STATE_AND_CONSTRUCTION_B3__ORACLE_BOOLEAN_DIAGRAM_TINY_ONLY_AFTER_ROOT_INCIDENCES__R14_KRYLOV_DUPLICATE_NOT_RERUN__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct a support-adaptive transposed incidence algorithm that emits only
nonempty S4-prefix/S4-suffix intersections without enumerating the `B^3`
prefix radicals or `B^5` incidence grid. It must provide exact blind zero
certificates, recover the forced source, preserve R73 occurrence
multiplicities and every dyadic child, and fit `B^(9/4+o(1))` setup/state and
`B^(5/4+o(1))` fresh-target work/workspace. Reject any route that obtains its
sparse output support from the measured oracle signatures.
