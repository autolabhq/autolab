# P1553 torus C5 Chebotarev fiber-cover gate R138

## Claim boundary

R138 proves a conditional tuple-fiber lower bound for structured sparse
zero-test trees and audits whether a finite-field Chebotarev theorem can
supply the required atom-matrix hypothesis.

If every `t`-column character-evaluation submatrix on an `m`-atom color
is full spark, then a `t`-mode node vanishes on at most
`(t-1)*m^4` ordered five-source tuples. A rejecting all-nonzero path
therefore has depth at least `ceil(m/(t-1))`. For trinomials this is
`B^(3/4+o(1))`.

The pinned corrected source does not certify the actual norm-one
families. Its explicit finite-field theorem requires the characteristic
to be primitive modulo the prime Fourier size. Every actual family has
`p=6q-1`, hence `ord_q(p)=2`. Forcing primitive order moves the relevant
roots to extension degree `q-1=B^(5+o(1))`, beyond the setup cap.

R138 does not prove atom-restricted full spark for the asymptotic
structured color, close four-plus-mode or nonzero-value circuits, or
supply a source index, known-RHS rank, factor logs, identical descent,
Shoup improvement, or ECDLP breakthrough.

Classification:

```text
FULL_SPARK_T_MODE_ATOM_CODE_IMPLIES_NODE_FIBER_WEIGHT_AT_MOST_T_MINUS_ONE_TIMES_M_FOUR_AND_REJECTING_PATH_DEPTH_AT_LEAST_M_OVER_T_MINUS_ONE__TRINOMIAL_TREE_CONDITIONALLY_B_THREE_QUARTERS__CORRECTED_FINITE_FIELD_CHEBOTAREV_REQUIRES_PRIMITIVE_CHARACTERISTIC_ORDER_BUT_ACTUAL_NORM_ONE_FAMILIES_HAVE_ORDER_TWO__FORCING_PRIMITIVE_ORDER_COSTS_EXTENSION_DEGREE_B_FIVE__TWO_EXACT_ALL_MODE_AND_73920_WINDOW_MINOR_CONTROLS_WITHOUT_ASYMPTOTIC_CREDIT__DIRECT_CHARACTERISTIC_SPECIFIC_SPARK_FOUR_PLUS_NONZERO_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Tuple-fiber lemma

Let `A` be an `m`-element atom set. Suppose every square submatrix with
`t` atom rows and `t` distinct character columns is nonsingular. Every
nonzero `t`-mode polynomial then has at most `t-1` zeros on `A`.

For a node polynomial `f` and fixed `a1,a2,a3,a4 in A`, the fiber

```text
f(a1*a2*a3*a4*x), x in A
```

has the same distinct modes and only rescales nonzero coefficients.
Consequently one node vanishes on at most

```text
(t-1)*m^4
```

ordered tuples in `A^5`. If `d` node root sets cover the complete
positive support, their pullbacks cover all `m^5` tuples, so

```text
d >= ceil(m/(t-1)).
```

This argument is deliberately performed on source tuples. Collisions
between tuple products cannot reduce the lower bound.

For `t=3` and `m=B^(3/4+o(1))`, a rejecting all-nonzero path needs
`B^(3/4+o(1))` depth. If the all-nonzero leaf accepts, the inherited
global trinomial root bound already forces polynomial depth to cover the
`q-o(q)` complement. Thus actual atom-restricted three-column full spark
would close structured trinomial trees.

## Corrected source boundary

The pinned source is Tarek Emmrich and Stefan Kunis, *Real and finite
field versions of Chebotarev's theorem*, arXiv:2506.02947.

The corrected PDF states Theorem 16 and Corollary 17 under the condition
that the field characteristic is primitive modulo the prime Fourier
size. Its Section 4.1 explicitly says that it cannot provide an analogous
explicit bound in the nonprimitive case. The acknowledgement records that
an earlier nonprimitive version was flawed.

For every actual R82 family,

```text
p = 6q-1
ord_q(p) = 2
```

while the cited corollary requires `ord_q(p)=q-1`. No transfer is
admitted.

Changing the representation to primitive order would place primitive
`q`-th roots first in `F_(p^(q-1))`. Since `q=B^(5+o(1))`, one generic
extension element already has `B^(5+o(1))` base-field coordinates. That
exceeds the `B^(9/4+o(1))` setup cap and abandons constant-degree FFE.

## Exact controls

All eight frozen pairing decks are replayed directly in `Fp2`.

- Every characteristic has multiplicative order exactly two modulo its
  prime subgroup order.
- No actual family satisfies the cited primitive-order condition.
- The two three-atom decks exhaust all normalized mode pairs and prove
  every three-by-three character minor nonsingular for those decks.
- Across all eight uncolored decks, 73,920 minors with modes in
  `[0,15]` are checked and none is singular.
- Every balanced finite color has at most two atoms, so no finite colored
  three-atom control exists.

The exact small and bounded-window controls verify implementation
semantics only. They receive no probability or asymptotic credit.

## Admission

Fifteen of twenty-three obligations pass. The tuple-fiber lemma,
conditional trinomial depth bound, corrected-source hypothesis audit,
order-two obstruction, extension-degree charge, and finite diagnostics
are admitted. Actual atom-restricted full spark, four-plus-mode and
nonzero-value selectors, source indexing, rank, logs, identical descent,
Pollard-rho improvement, Shoup improvement, and breakthrough remain
false.

Disposition:

```text
ADMIT_TUPLE_FIBER_COVER_LEMMA_AND_CONDITIONAL_TRINOMIAL_DEPTH__REJECT_GENERIC_PRIMITIVE_ORDER_CHEBOTAREV_TRANSFER_TO_DEGREE_TWO_NORM_ONE_FFE__ADMIT_TWO_EXACT_ALL_MODE_AND_73920_WINDOW_MINOR_CONTROLS_WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_DIRECT_CHARACTERISTIC_SPECIFIC_SPARK_FOUR_PLUS_LOW_SLP_AND_NONZERO_VALUE_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Probe a characteristic-specific three-column restricted-minor theorem
for the actual norm-one atom set, or construct a surviving four-plus-mode
or nonzero-value selector. Freeze every mode, coefficient, circuit node,
branch, and reverse C2+C3 source pointer; replay positives and inverse
empties; fit `B^(9/4+o(1))` state and polylogarithmic arbitrary-target
work; avoid field DLP; and charge rank, logs, identical descent, memory,
field operations, extension degree, and bits.
