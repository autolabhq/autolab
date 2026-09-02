# P1553 primitive degree-nine section-rank gate R44

## Classification

- Owner: existing P1553/IDEA-195 asymptotic interval-pencil frontier; no new
  idea ID.
- Evidence: exact finite-field section construction, exhaustive affine scans
  of two structured partitions, and a deterministic finite trade walk; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_STRUCTURED_TOY_PRIMITIVE_PENCIL_NEGATIVE_GATE`.
- Labels: `toy`, `exact`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: the tested primitive degree-nine candidates on the R38
  toy curve do not lie in a pencil. All 10,506 affine instances of the normal
  9-by-9 magic-square row partition have section-vector rank 9. The column
  partition has rank 8 in 206 instances and rank 9 in 10,300. A deterministic
  20,000-step equal-pair-sum trade walk, evaluated in eight affine contexts,
  adds 160,008 exact rank instances and also has minimum rank 8. Rank 2 is
  required. Arbitrary equal-sum partitions and primitive degree-nine pencils
  remain open. No Shoup-bound improvement or ECDLP breakthrough follows.

R43 closed every degree-three by degree-three composition on the exact toy and
left a primitive degree-nine pencil as the next coordinate-level branch. R44
implements the first exact search of that branch. It distinguishes the Abel-sum
condition from the much stronger common-pencil condition and preserves the
rank witnesses needed to steer a nonrandom search.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R38 target-walk and pencil gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R42 single-interval pencil DLP gate | `8a8de8726e4fc9542f97cf7e1ba4fc54596307cc544da0d0b18b588553022dae` |
| R43 degree-three composition gate | `b02901b1a822ce98f886146d5a04d2ea35b9515ec549358d2c4b7c02b6700ec8` |
| R43 bundle hash list | `5deeefbbf9fef16f424c494c260b781532eef37bb1f6ff859829b8ea300e45ab` |
| R43 staging receipt | `b647080192cf34b92687f1e30bf0cff6a313b096a4c7847d08780d80f67d3e35` |

## Primitive-pencil requirement

On the toy curve

```text
E/F_193: y^2=x^3+2x+3,
|G|=103,                                          (1)
```

the complete linear system `|9O|` has dimension eight and its section space is

```text
H^0(E,O(9O))=<1,x,x^2,x^3,x^4,y,xy,x^2y,x^3y>,
dim H^0(E,O(9O))=9.                              (2)
```

Let an 81-point subgroup interval be partitioned into nine disjoint blocks
`D_1,...,D_9`, each of degree nine. For all blocks to be fibers of one
degree-nine map, two conditions are necessary:

```text
sum(D_1)=...=sum(D_9) in E(F_193),                (3)
rank(s_1,...,s_9)=2,                             (4)
```

where `s_i` is the unique projective section of `H^0(E,O(9O))` vanishing on a
common translate of `D_i`. Equation (3) puts the divisors in one linear
equivalence class. Equation (4) says all nine sections lie in one
two-dimensional vector subspace, namely a pencil. Because the blocks are
disjoint, their sections are not all proportional, so the required rank is
exactly two rather than at most two.

The script `p1553_primitive_degree_nine_rank_search_r44.py` constructs each
section as the one-dimensional kernel of its exact 9-by-9 evaluation matrix
over `F_193`, normalizes the projective vector, and computes the rank of all
nine vectors by finite-field row reduction.

## Structured equal-sum partitions

The normal order-nine Siamese magic square partitions the offsets `0,...,80`
in two ways: its nine rows and its nine columns. Every block has integer sum

```text
0+...+80 over nine blocks = 9*360,
block sum=360.                                   (5)
```

For every affine interval

```text
start+step*k mod 103, 0<=k<81,
start in Z/103Z, step in (Z/103Z)^*,             (6)
```

all blocks therefore have common Abel sum

```text
9*start+360*step mod 103.                        (7)
```

The common source translation

```text
t=-(9*start+360*step)/9 mod 103                  (8)
```

makes every block sum to `O`, so all nine sections are represented in the
fixed basis (2). There are exactly

```text
103*102=10,506                                   (9)
```

affine contexts for each partition, and the scan exhausts all of them.

## Exact affine-scan result

The row partition has the rank histogram

| Section rank | Instances |
|---:|---:|
| 9 | 10,506 |

The column partition has

| Section rank | Instances |
|---:|---:|
| 8 | 206 |
| 9 | 10,300 |

Thus neither structured partition has a rank-two instance. The rank-eight
column cases are retained because they prove that the implementation can see
nontrivial section dependence and provide a closer starting family than the
full-rank rows.

## Equal-pair-sum trade walk

Starting from the row partition, one trade chooses two blocks and one pair
from each block having the same integer pair sum, then swaps the pairs. This
preserves block sizes, the partition of `0,...,80`, and every block sum 360.
The deterministic walk uses seed 1553 and accepts 20,000 such trades after
20,022 attempts.

Every accepted partition is tested in these eight affine contexts:

```text
(start,step)=(0,1),(1,1),(0,42),(1,42),
             (0,43),(12,43),(0,102),(50,17).     (10)
```

Including each initial state gives

```text
8*(20,000+1)=160,008                             (11)
```

exact rank instances. Their combined histogram is

| Section rank | Instances |
|---:|---:|
| 8 | 822 |
| 9 | 159,186 |

The first minimum witness occurs at trade 47 in context `(0,43)` and has rank
8. No tested trade state has rank 7 or below, much less the required rank 2.
The complete minimum block partition and per-context histograms are preserved
in `p1553_primitive_degree_nine_rank_search_report_r44.json`.

## Scope

The row and column scans are exhaustive only within those two structured
normal-magic-square partitions and all affine embeddings into the order-103
subgroup. The trade walk is a deterministic finite correlated family, not a
random-sampling theorem and not an exhaustive search of equal-sum partitions.
Rank greater than two is a conclusive negative witness for one tested
partition and affine context only.

R44 therefore does not exclude arbitrary primitive degree-nine pencils, other
curves, or an asymptotic interval-pencil family. It supplies no target locator,
R10 queried coefficients, relation-rank campaign, factor-base logarithms, or
scalar-blind descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_STRUCTURED_TOY_PRIMITIVE_PENCIL_NEGATIVE_GATE
NINE_EQUAL_ABEL_SUM_BLOCKS_REQUIRE_SECTION_VECTOR_RANK_TWO
MAGIC_ROW_AFFINE_SCAN_10506_INSTANCES_ALL_RANK_NINE
MAGIC_COLUMN_AFFINE_SCAN_206_RANK_EIGHT_10300_RANK_NINE
EQUAL_PAIR_SUM_TRADE_WALK_160008_RANK_INSTANCES_MINIMUM_EIGHT
NO_RANK_TWO_SURVIVOR
ARBITRARY_PRIMITIVE_DEGREE_NINE_PENCIL_OPEN
NO_TARGET_R10_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: replace the undirected trade walk with a
rank-minor-guided search. Starting from every rank-eight column witness,
enumerate all equal-sum pair trades, score each child by exact section rank and
the number of vanishing 8-by-8 minors, retain a deterministic beam, and require
any rank-two survivor to pass the R42 all-secret marked-target test before any
asymptotic claim.
