# P1553 actual-S6 Fermat tensor-train probe R78

## Classification

- Owner: existing P1553 R2/R9-R10/R75 nonlinear projector frontier; no P1554
  and no new idea ID.
- Evidence: exact target-specialized S6 values, finite-field tensor ranks, and
  Fermat masks on four standardized prime-order curves.
- Status: `REJECT_ACTUAL_S6_VALUE_FIRST_TT_FERMAT_GRAMMAR_ONLY`.
- Labels: `exact-finite`, `nonlinear-representation-bound`,
  `scalar-blind-source`, `novelty-unverified`.
- Cryptanalytic result: no admitted relation source, factor-log solve,
  fresh-target descent, Shoup improvement, or ECDLP breakthrough.

R78 freezes one nonlinear target-specialized grammar left open by R75 and
R77: evaluate the actual S4-by-S4 resultant on the five-deck tensor, represent
every intermediate exactly as a tensor train, and apply the Fermat zero
projector by binary Hadamard powering. The final masks are tiny, but every raw
value tensor already has full middle-cut rank and forces a `B^5` TT center
core.

This closes only the value-first TT grammar. A scalar straight-line
resultant/norm circuit that never represents tuple values remains open.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R72 actual S6 report | `7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43` |
| R75 iterated-norm support report | `c5f41fbb7325fe3f4e85fe6084fbaed6ab2df9d9101bd031cd1c589a95a3ba8c` |
| R76 exact subset-incidence report | `de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93` |
| R31 registry containing R2 finite-deck gate | `0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f` |
| Bound R2 finite-deck gate | `55acc1457e7fd5a740da57c2c1db957374c7c18561c67b1748176dc8c61fcda5` |

## Frozen grammar

For each five-tuple occurrence, compute the actual target-specialized value

```text
T(i1,...,i5)
  = Res_z(
      S4(x1,x2,x3,z),
      S4(x_R,x4,x5,z)
    )
  = S6(x1,...,x5,x_R).                            (1)
```

The zero indicator over `F_p` is

```text
Z(i1,...,i5) = 1 - T(i1,...,i5)^(p-1).            (2)
```

R78 freezes binary repeated squaring and exact five-core tensor trains after
every value-first operation. It does not use scalar labels, group characters,
or materialized suffix roots.

Equation (2) counts tuple occurrences, not internal common roots. Duplicate
tuple records remain separate entries. A synthetic `[0,0,7]` resultant
control over `F_101` produces indicators `[1,1,0]`, so duplicate occurrences
count twice while one tuple with a higher-degree polynomial gcd still counts
once.

## Exact TT obstruction

For a five-mode tensor of side length `B`, let `r_k` be the exact matrix rank
across the cut after mode `k`. Every exact TT has those intrinsic ranks. Its
center core contains

```text
r_2 * B * r_3                                (3)
```

field entries.

On secp256k1, P-256, P-384, and P-521, for blind and forced-positive targets
and every `B=3,4,5,6`, R78 obtains

```text
(r_1,r_2,r_3,r_4) = (B,B^2,B^2,B).                (4)
```

Thus (3) is exactly `B^5` in all 32 target instances. The first Hadamard
square has the same full ranks.

At `B=6`, the mandatory center core contains `7,776` entries. This is not an
empirical fit: each reported rank is exact row reduction over the curve's
prime field. The cross-prefix uniformity is finite evidence, not an
asymptotic rank theorem for every deck family.

## Final-mask trap

Every blind target has zero relation occurrences. Its Fermat nonzero mask is
all ones with TT ranks `[1,1,1,1]`, and its final zero mask is identically
zero.

Every forced target has exactly one occurrence. Its nonzero mask has ranks
`[2,2,2,2]`, and its one-hot zero mask has ranks `[1,1,1,1]`. All forced
sources verify by exact signed group law, and every dyadic parent count equals
the sum of its children.

The maximum final zero-mask TT parameter count is only 30. That compression
is outcome-conditioned: the value-first constructor has already crossed the
full-rank state in (4). It receives no setup or query credit.

## Cost ledger

The frozen grammar incurs

```text
raw target-specific value state          B^5,
minimal TT center-core state             B^5,
binary Fermat pointwise work             B^5 * O(log p).    (5)
```

This misses both required caps:

```text
setup/persistent state                    B^(9/4+o(1)),
fresh-target work/workspace               B^(5/4+o(1)).      (6)
```

Streaming the final sum cannot remove the value computation or provide the
TT-guided source and child predicates without replaying the same tuple
traffic.

## Deduplication

R75 closes expanded coefficient norms, while R78 closes a value-first exact TT
and Hadamard-Fermat representation. R2 measured determinant-channel TT ranks
and endpoint traffic; R78 measures the actual target-specialized S6 value
tensor and its projector constructor path. The mechanism is a scoped
refinement, not a new idea ID.

## Scope

R78 is not a circuit lower bound. It does not reject a black-box scalar
resultant, norm, transposed program, randomized Las Vegas evaluator, or other
nonlinear circuit that never represents a tuple-value tensor, TT core,
coefficient cube, quotient algebra, or suffix polynomial.

No relation-density theorem, independent factor-base rank, verified factor
logs, or identical fresh-target descent is supplied.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_actual_s6_fermat_tensor_train_probe_r78.py` | `fb31a8b420df349d2220d0fd4ffb9451b73ec6313ec0fd1baf058a000f30ae04` |
| `p1553_actual_s6_fermat_tensor_train_probe_report_r78.json` | `e590002c433c6504725d5ab7ff1dba97ad8c15400bf0117846742da7359c5e60` |
| `frozen_nonlinear_nested_resultant_functional_grammar.json` | `320c0ae21cbb56451c1dcd4e9f96e53bfd1cb87d07d7a7fc6d6402e835d107f3` |
| `actual_s4_target_specialization_replay.json` | `6e32afd5ad8bd6527c780e42a7e4208417c59bfd8c87e8497119a9e991c7cf62` |
| `r76_mobius_zero_source_and_child_replay.json` | `713918450c94b64d608fca70c16ee1b0b4731e83e661c261c0ba2287a3a82a13` |
| `nonlinear_functional_state_and_direct_cost_ledger.json` | `f5a51829590e004b9c303252adb376c023e980bf1a264121accb61e65cbcad4b` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_actual_s6_fermat_tensor_train_probe_r78.py` | `8952a4bc7a0242c67e04ff96e12b299707403fa72e0a0a62195c67454392667f` |

Targeted replay:

```text
Ran 4 tests in 1.379s
OK
families=4
middle_full=True
sources_verified=True
lane_admitted=False
```

## Disposition

```text
REJECT_ACTUAL_S6_VALUE_FIRST_TT_FERMAT_GRAMMAR_ONLY__FOUR_STANDARD_CURVES__B3_4_5_6__BLIND_ZERO__FORCED_COUNT_ONE__SOURCE_AND_DYADIC_REPLAY__ALL_RAW_MIDDLE_CUTS_FULL__CENTER_CORE_B5__FINAL_MASK_TINY_ONLY_AFTER_VALUE_CONSTRUCTION__R2_R75_BOUNDARIES_SHARPENED__SCALAR_ONLY_CIRCUITS_OPEN__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct a straight-line black-box nested S4 norm functional whose every
node is scalar or at most `B^(9/4)` persistent state and `B^(5/4)`
fresh-target state/work. It may never materialize a tuple-value tensor, TT
core, coefficient cube, quotient algebra, or suffix polynomial, and must
retain R76 zero, count, source, multiplicity, and dyadic replay.
