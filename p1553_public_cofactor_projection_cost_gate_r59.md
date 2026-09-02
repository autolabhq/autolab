# P1553 public cofactor-projection and rate gate R59

## Classification

- Owner: existing P1553/IDEA-195 primitive product-pencil frontier; no new
  idea ID.
- Evidence: exhaustive finite-group projection proof, exact R58 witness replay,
  and heuristic random-fiber/class-sum cost model; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_EXACT_PROJECTION_HEURISTIC_COST_GATE`.
- Labels: `toy`, `exact-projection`, `heuristic-fixed-degree-cost-model`,
  `non-run`, `model-bound`, `novelty-unverified`.
- Cryptanalytic result: `[104]` is exactly the public projector from the
  206-point curve group to its 103-point prime subgroup. It recovers every R58
  third-fiber subgroup atom without a DLP, while scalar logs remain unknown.
  The fixed-degree random-fiber model predicts one valid degree-nine pencil per
  11,935 accepted pencils, nearly matching the observed 11,104, but this is
  `Theta(N)` pencil work before the full field scan is charged. Degree six is
  the only fixed degree-three-factor case with constant predicted pencil
  count. No Shoup-bound improvement or ECDLP breakthrough follows.

## Inputs

| Input | SHA-256 |
|---|---|
| R58 primitive degree-nine pencil gate | `592438ed51e3fa39cc869927fad40726949ee6c645b487db4e4795d5b8118f25` |
| R58 exact report | `728037585618fbe680747e5b68a40005fc8a6c8b0979213662c0b638cd616c3b` |

## Exact projection

For `E(F_193)=H direct_sum <T_2>` with orders 103 and 2,

```text
104=1 mod 103,
104=0 mod 2.                                     (1)
```

Therefore

```text
pi_H(Q)=[104]Q                                   (2)
```

is identity on `H` and kills `T_2`. R59 exhausts all 206 points and all
`206^2=42,436` group pairs. It verifies that (2) is an idempotent
homomorphism with image exactly `H` and kernel `{O,T_2}`.

The nine projected R58 atom labels, shown only as exhaustive toy verification,
are

```text
(21,82,55,11,8,53,67,56,49).                    (3)
```

Their three factor blocks are

```text
(82,11,56), sum 46,
(55,53,49), sum 54,
(21,8,67),  sum 96.                              (4)
```

All nine projected points are distinct. Projection does merge cofactor
cosets across fibers: atom 56 also occurs in the first generator and atom 49
in the second. Those are valid repeated factor-base atoms and must retain
multiplicity; they are not evidence of a base point on the original curve
pencil.

## Rate model

This subsection is explicitly heuristic. Assume a degree-`d` fiber has random
permutation splitting statistics and prescribed class sums behave as uniform
independent subgroup equations.

For R58, `d=9` and there are three ordered degree-three classes. A rational
fiber splits completely with probability approximately `1/9!`. The number of
ordered partitions into the three classes is

```text
9!/(3!^3)=1,680.                                 (5)
```

Two independent class-sum equations cost `N^2`. Scanning `p` parameters in
one pencil therefore gives heuristic success probability

```text
q_3 ~= (p-1)*1680/(9!*N^2)=((p-1)/(6^3*N^2)).  (6)
```

At `p=193,N=103`, (6) predicts

```text
q_3=8.378630e-5,
1/q_3=11,935.125 pencils.                        (7)
```

The observed first hit at accepted pencil 11,104 is 0.930 times the
prediction. This agreement is consistent with the frozen toy model only; it is
not an asymptotic theorem.

For `r` prescribed degree-three factors, the same cancellation gives

```text
q_r ~= (p-1)/(6^r*N^(r-1))
     ~= 1/(6^r*N^(r-2)) when p~N.                (8)
```

Thus `r=3` is linear in `N`; larger `r` is worse. The only nontrivial constant
rate case is `r=2`, map degree six, with toy prediction one hit per 19.31
accepted pencils. This still omits the cost of scanning all `p` line
parameters and the probability that output atoms lie in a restricted factor
base.

## Scope

The projection theorem is exact. The rate model is heuristic and fixed-degree.
It does not establish independence, generic monodromy, a restricted
factor-base smoothness law, an implicit line-intersection locator, relation
rank, factor-base logs, or descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_EXACT_PROJECTION_HEURISTIC_COST_GATE
PUBLIC_104_MAP_IMAGE_H_KERNEL_O_T2
ALL206_POINTS_AND_42436_GROUP_PAIRS_CHECKED
R58_THIRD_FIBER_PROJECTS_TO_NINE_DISTINCT_H_ATOMS
PROJECTION_COLLISIONS_56_AND_49_RETAINED
ORDERED_CLASS_PARTITIONS_1680
DEGREE9_MODEL_PREDICTS_11935_PENCILS_OBSERVED11104
DEGREE9_FIXED_FACTOR_MODEL_LINEAR_IN_N
DEGREE6_TWO_FACTOR_MODEL_CONSTANT_PENCIL_COUNT_CANDIDATE
FULL_PARAMETER_SCAN_AND_FACTOR_BASE_DENSITY_UNCHARGED
NO_TARGET_R10_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: test the degree-six prediction on the full rational
curve. Use two prescribed degree-three classes, require coprime marked
generator products, and preserve the first third six-point rational fiber that
partitions into the same two classes.
