# P1553 scalar-only nested-norm SLP probe R79

## Classification

- Owner: existing P1553 R2/R76/R78 scalar-functional frontier; no P1554 and
  no new idea ID.
- Evidence: exact actual-S4 scalar products on four standardized curves plus
  a hash-bound replay of all 32 R78 target instances.
- Status: `REJECT_SCALAR_ONLY_LEAF_PRODUCT_NORM_SLP_ONLY`.
- Labels: `exact-finite`, `per-node-costed`, `scalar-blind-source`,
  `novelty-unverified`.
- Cryptanalytic result: no admitted relation source, factor-log solve,
  fresh-target descent, Shoup improvement, or ECDLP breakthrough.

R78 rejects value-first tensor trains because their raw actual-S6 center core
has `B^5` entries. R79 removes that state entirely: every node of a five-level
deck norm is a scalar, and a depth-first evaluator keeps only six live
scalars. The resulting program still evaluates all `B^5` actual S4-by-S4
resultant leaves on every blind zero certificate and exact count.

This closes only the explicit leaf product/norm SLP. A batched norm-node
compiler with a proved sub-`B^5` field-operation implementation remains open.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R78 actual-S6 TT report | `e590002c433c6504725d5ab7ff1dba97ad8c15400bf0117846742da7359c5e60` |
| R78 frozen grammar | `320c0ae21cbb56451c1dcd4e9f96e53bfd1cb87d07d7a7fc6d6402e835d107f3` |
| R76 exact subset-incidence report | `de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93` |
| R31 registry containing R2 finite-deck gate | `0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f` |
| Bound R2 finite-deck gate | `55acc1457e7fd5a740da57c2c1db957374c7c18561c67b1748176dc8c61fcda5` |

## Frozen scalar grammar

Each leaf is the scalar

```text
t(i1,...,i5)
  = Res_z(
      S4(x1,x2,x3,z),
      S4(x_R,x4,x5,z)
    ).                                             (1)
```

For a fixed four-index prefix, multiply (1) over the fifth deck. Continue
recursively over decks four through one. Every internal node is a scalar
product, and the root is

```text
Omega_R = product_(i1,...,i5) t(i1,...,i5).        (2)
```

Since the coefficient field is integral,

```text
Omega_R = 0 iff at least one tuple is a relation.  (3)
```

The exact tuple count on the frozen `B^5<p` range is

```text
C_R = sum_(i1,...,i5) (1-t(i1,...,i5)^(p-1)).     (4)
```

Duplicate tuple occurrences occupy separate leaves. A tuple whose two S4
factors share multiple roots contributes one to (4), not its gcd degree. An
exact synthetic control has tuple count two and gcd-degree sum three.

## Direct replay

R79 evaluates (1)-(4) directly at `B=3` on secp256k1, P-256, P-384, and
P-521. On every curve:

```text
blind root product is nonzero,
blind exact count is zero,
forced root product is zero,
forced exact count is one,
forced source indices are [0,0,0,0,0],
forced source verifies by exact signed group law. (5)
```

The 32 R78 instances at `B=3,4,5,6` are hash-bound for full source and child
replay. Every blind count is zero, every forced count is one, every source
verifies, and every first-mode dyadic parent equals the sum of its children.

## Per-node ledger

A full B-ary scalar product tree with `B^5` leaves has:

```text
actual S6 leaf evaluations       B^5,
scalar multiplications           B^5-1,
cached scalar nodes              1+B+B^2+...+B^5. (6)
```

At `B=6`, this is 7,776 leaves and 9,331 cached scalar nodes.

Two implementations expose the tradeoff:

```text
depth-first streaming:
  live scalar workspace          <= 6,
  blind failed-zero work         B^5,
  forced exact-count work        B^5;

cached product tree:
  source/child navigation        available,
  target-specific state          B^5.              (7)
```

Short-circuiting does not improve the blind path. On a positive target it can
find one zero early, but proving the exact count and all children still
evaluates the complement. Calling a deck product `NORM` does not change (6)
unless a separate compiler supplies and charges a faster implementation.

Thus streaming meets the workspace cap but misses the work cap, while caching
misses the state cap:

```text
required setup/state             B^(9/4+o(1)),
required fresh-target work/state B^(5/4+o(1)).      (8)
```

## Scope

R79 does not reject batched subproduct, resultant, remainder-tree,
transposed, modular-composition, or other norm-node compilers. It requires
such a compiler to expose field operations and intermediate state rather than
receiving unit-cost black-box credit.

No relation-density theorem, independent factor-base rank, verified factor
logs, or identical fresh-target descent is supplied.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_scalar_only_nested_norm_slp_probe_r79.py` | `6641d70e033b44537c3b5f76db790ab000b9427a75df4f8b4aaf07c48ff47441` |
| `p1553_scalar_only_nested_norm_slp_probe_report_r79.json` | `b5bd22b7256d80f5fa475517699c42f2201a1c8c6209718ea38ab832146a8929` |
| `frozen_scalar_only_nested_norm_slp.json` | `4be19fcf588878e574d1ba7c38d811010bef5fe9e5f7b088a4c371e1bcf43818` |
| `actual_s4_scalar_zero_count_replay.json` | `8886798d1534c5a9f380157bcd71c7a5511755e2cca42d5d3fe237cb9959eedd` |
| `rectangle_source_and_dyadic_transcript.json` | `b925c92e578c46c82ccf3aaf45b51c5a2fb8ece8e5d14457e867b88ff522865c` |
| `per_node_state_work_and_failed_zero_ledger.json` | `623b1aaf244bc0d037d7482202dfc4545b58b2ce8bb00e5f77a770fb4d48449c` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_scalar_only_nested_norm_slp_probe_r79.py` | `91177f7f4fb123a81c1e3dc60bdc2995b9158a68c02d589e02c91ab605c48657` |

Targeted replay:

```text
Ran 4 tests in 1.268s
OK
bound_instances=32
direct_exact=True
sources_verified=True
lane_admitted=False
```

## Disposition

```text
REJECT_SCALAR_ONLY_LEAF_PRODUCT_NORM_SLP_ONLY__ACTUAL_S4_BY_S4_LEAVES__FOUR_DIRECT_STANDARD_CURVE_CONTROLS__R78_32_INSTANCE_BINDING__BLIND_ZERO__FORCED_COUNT_SOURCE__DUPLICATE_AND_MULTIPLE_ROOT_SEMANTICS__DYADIC_CHILDREN__STREAMING_WORKSPACE_CONSTANT_BUT_FAILED_ZERO_WORK_B5__CACHED_STATE_B5__BATCHED_NORM_COMPILER_OPEN__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Freeze one batched nested-norm node compiler from unary deck polynomials to
scalar output. Expand every resultant, subproduct, remainder, transposed, and
source operation into field-operation and state receipts. Require a strict
improvement over `B^5` while preserving R76 zero, count, source, and children
before testing the direct caps.
