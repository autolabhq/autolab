# P1553 batched nested-norm node compiler probe R80

## Classification

- Owner: existing P1553 R3/R76/R79 scalar-functional frontier; no P1554 and
  no new idea ID.
- Evidence: exact product-polynomial, gcd, source-descent, and Mobius-count
  replays on four standardized curves.
- Status:
  `REJECT_STANDARD_BATCHED_S4_SUBPRODUCT_GCD_COMPILER_ONLY`.
- Labels: `exact-finite`, `per-node-costed`, `scalar-blind-source`,
  `novelty-unverified`.
- Cryptanalytic result: no admitted relation source, factor-log solve,
  fresh-target descent, Shoup improvement, or ECDLP breakthrough.

R79 evaluates all `B^5` actual-S4 resultant leaves. R80 applies the standard
batched compiler that separates a triple prefix from a target-plus-pair
suffix. This is a real asymptotic improvement over the explicit leaf
program: exact existence and a relation source can be recovered with
`B^(3+o(1))` field work and `B^(3+o(1))` persistent prefix state.

That improvement is insufficient. The required setup-state cap is
`B^(9/4+o(1))`; the required fresh-target work and workspace caps are
`B^(5/4+o(1))`. The exact compiler costs `B^3` setup state, `B^3`
fresh-target work, and `B^2` fresh-target workspace.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R79 scalar-only nested-norm report | `b5bd22b7256d80f5fa475517699c42f2201a1c8c6209718ea38ab832146a8929` |
| R76 exact subset-incidence report | `de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93` |
| R31 registry containing R3 Query2P1 gate | `0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f` |
| Bound R3 Query2P1 gate | `b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e` |

## Frozen compiler

For every ordered triple `(i1,i2,i3)`, form the squarefree endpoint factor

```text
A_i(z) = product_(a in roots(S4(P_i1,P_i2,P_i3,z))) (z-a).       (1)
```

Identity endpoints are carried as a separate tagged multiplicity. Build a
balanced product tree with root

```text
A(z) = product_i A_i(z),                deg A = O(B^3).           (2)
```

For a fresh target and every ordered pair `(i4,i5)`, form the corresponding
target-specialized factor `D_j(z)` and product tree

```text
D_R(z) = product_j D_j(z),              deg D_R = O(B^2).         (3)
```

There is a relation exactly when the finite-root gcd is nonconstant or both
sides carry an identity endpoint:

```text
exists relation iff gcd(A,D_R) != 1 or identity tags intersect.  (4)
```

Repeated gcd tests down the two balanced trees return one prefix leaf and
one suffix leaf. Their five deck indices verify by the exact signed group
law. Scalar labels are never consumed.

## Exact count

The degree of `gcd(A,D_R)` is not the number of relation tuples because
product multiplicity records repeated endpoint roots, while a tuple with
several common roots must still count once.

R80 therefore retains the R76 endpoint-subset histograms. If `L(S)` and
`R(S)` count prefix and suffix occurrences containing nonempty root subset
`S`, then

```text
C_R = sum_(nonempty S) (-1)^(|S|+1) L(S) R(S).                   (5)
```

An exact synthetic control has two copies of left root set `{1,2}` and
right root sets `{1,2}` and `{2,3}`:

```text
exact tuple count       4,
product gcd degree      3,
naive singleton count   6.                                      (6)
```

Thus the product gcd is an exact existence and source primitive, while (5)
is required for exact occurrence count.

## Replay

The frozen run uses secp256k1, P-256, P-384, and P-521 at
`B=3,4,5`. Across 12 prefix instances and 24 blind/forced target instances:

```text
all product-gcd existence decisions exact        true,
all blind exact counts                           zero,
all forced exact counts                          one,
all positive product-tree sources verified       true,
maximum prefix product degree                    500,
maximum suffix product degree                    100.             (7)
```

These finite controls validate the compiler semantics. They do not provide
a relation-density theorem, factor-base rank, factor logs, or descent.

## Cost ledger

With fast polynomial multiplication `M(n)=n^(1+o(1))`:

```text
prefix factor enumeration and product tree       B^(3+o(1)),
prefix product/correction persistent state        B^(3+o(1)),
suffix factor enumeration and product tree        B^(2+o(1)),
fresh A mod D_R, gcd, and prefix source descent   B^(3+o(1)),
fresh suffix/correction workspace                 B^(2+o(1)).      (8)
```

The first remainder must read and reduce a degree-`B^3` prefix polynomial;
it is not a unit-cost black box and cannot be charged as only `B^2`.
Gcd-guided prefix source descent has the same `B^(3+o(1))` dominant work.

The strict `B^5` to `B^3` improvement therefore misses every direct cap:

```text
setup state:       3 > 9/4,
fresh work:        3 > 5/4,
fresh workspace:   2 > 5/4.                                      (9)
```

## Scope

R80 rejects only the standard materialized triple-factor product,
fresh suffix product, polynomial remainder/gcd, and R76 correction path. It
does not prove a lower bound against a structured factor base whose triple
endpoint multiset has a new compressed representation.

No independent relation rank, verified factor logs, identical fresh-target
descent, full ECDLP solve, or generic-prime Shoup-bound improvement is
supplied.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_batched_nested_norm_node_compiler_probe_r80.py` | `e69a431cccc584c2ce11e7af7a497c4c4f7bfe751a0617cd1a5f0add64fb29ce` |
| `p1553_batched_nested_norm_node_compiler_probe_report_r80.json` | `936537fb78908dd2916bf6fa5b2091f336b9a47217a1ff787b068ae0491992c5` |
| `frozen_batched_norm_node_compiler.json` | `eee3c93b0fcd9ef5eb9a70f2c218ac7f3dc736fc02f153f4f9e2ae1924cb3592` |
| `resultant_subproduct_remainder_receipts.json` | `f7c57aaedd3bae886e7b4904c63baead05198fdddec8da24e18679c9c4f8589e` |
| `r76_batched_zero_count_source_replay.json` | `49e8cf96cae42bb6b825f96668680524682ace5942a29291d7121cb674215d34` |
| `compiled_node_field_state_cost_ledger.json` | `fa63c9572e7477cb75e2c7f825c8097ab464a3561c5da77962a8e70fad1f652e` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_batched_nested_norm_node_compiler_probe_r80.py` | `5e277796ce9d8427b6ee87dd77433746b9c55c5cd60dbf37ed5918237fcd43c3` |

Targeted replay:

```text
Ran 4 tests in 0.619s
OK
families=4 gcd_exact=True sources_verified=True lane_admitted=False
```

## Disposition

```text
REJECT_STANDARD_BATCHED_S4_SUBPRODUCT_GCD_COMPILER_ONLY__FOUR_STANDARD_CURVES__B3_4_5__STRICT_B5_TO_B3_IMPROVEMENT__EXACT_EXISTENCE_AND_PRODUCT_TREE_SOURCE__GCD_DEGREE_NOT_TUPLE_COUNT__R76_MOBIUS_CORRECTION__SETUP_STATE_B3__FRESH_TARGET_WORK_B3_STATE_B2__STRUCTURED_FACTOR_BASE_GEOMETRY_OPEN__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct one scalar-blind structured factor-base geometry whose `B^3`
triple endpoint multiset has a proved representation of at most
`B^(9/4+o(1))`, while fresh target-plus-pair queries and source return cost
at most `B^(5/4+o(1))`. Require a prospective relation-density theorem,
matched random controls, exact rank, factor logs, and identical target
descent.
