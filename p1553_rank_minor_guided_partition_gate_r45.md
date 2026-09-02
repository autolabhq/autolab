# P1553 rank-minor-guided partition gate R45

## Classification

- Owner: existing P1553/IDEA-195 primitive degree-nine pencil frontier; no new
  idea ID.
- Evidence: exact finite-field rank and cofactor calculations, one exhaustive
  local component, and a deterministic budgeted beam search; no cryptanalytic
  run.
- Status: `DRAFT_REVIEW_REQUIRED_FINITE_RANK_GUIDED_PRIMITIVE_PENCIL_NEGATIVE_GATE`.
- Labels: `toy`, `exact`, `finite-search`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: the exact equal-pair-sum trade component containing the
  R44 rank-eight column partition has 25 rank-eight states and no lower-rank
  state. Its 6,647-state rank-nine boundary seeds an eight-layer, width-64
  cofactor-guided search. Across 512 expanded rank-nine bridges, 198,401
  layer-unique children, and 281 retained rank-eight states, the minimum rank
  remains eight. The required rank is two. The later closures and bridge search
  are explicitly budgeted, so arbitrary partitions and primitive degree-nine
  pencils remain open. No Shoup-bound improvement or ECDLP breakthrough
  follows.

R44 found rank eight in 206 affine column contexts but used an undirected
finite trade walk. R45 first removes the redundant start coordinate, then
exhausts the rank-eight component containing the structured witness, and
finally crosses selected rank-nine bridges using exact determinantal sparsity.
It preserves all 281 retained rank-eight partitions for a pencil-first search.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R42 single-interval pencil DLP gate | `8a8de8726e4fc9542f97cf7e1ba4fc54596307cc544da0d0b18b588553022dae` |
| R43 degree-three composition gate | `b02901b1a822ce98f886146d5a04d2ea35b9515ec549358d2c4b7c02b6700ec8` |
| R44 primitive section-rank gate | `bc5e2068f25814087f228c5bc422109b3f7d5122cfad6bb0600b6208d372885a` |
| R44 bundle hash list | `e24ce23353862aed15fa5f67fdffcad9cb1a5d3364160aad2da7ef0e394e7cf9` |
| R44 staging receipt | `e914625038aedffecab5164f4d5aa5a009ce8fa77a6b5dbeaf44f46a905a04d6` |

## Start cancellation and step involution

Every searched block contains nine offsets with common integer sum 360. For
an affine interval context `(start,step)`, R44 translates the source by

```text
t=-(9*start+360*step)/9 mod 103.                 (1)
```

The scalar used for offset `k` is therefore

```text
start+step*k+t=step*(k-40) mod 103.              (2)
```

It is independent of `start`. Thus the 206 rank-eight affine column contexts
are 103 copies of each of two step contexts. The two steps are

```text
42 and 61=-42 mod 103.                           (3)
```

Negating the source acts invertibly on `H^0(E,O(9O))`: it fixes the five
`x`-basis terms and negates the four `y`-basis terms. It preserves section
rank and the zero pattern of every cofactor. The script nevertheless checks
all 206 root start contexts exactly and checks both step scores on every
retained rank-eight and expanded rank-nine state.

## Exact cofactor score

Let `A` be the 9-by-9 matrix whose rows are the normalized section vectors of
one equal-sum partition. Rank two is the pencil condition from R44. R45 orders
candidate states first by rank and then by the number of vanishing 8-by-8
minors of `A`.

For rank nine,

```text
adj(A)=det(A)*A^(-1),                            (4)
```

so the vanishing-minor count is exactly the number of zero entries in the
finite-field inverse. For rank eight, the adjugate is an outer product of a
right-null vector and a left-null vector. If their support sizes are `r` and
`l`, respectively, then

```text
vanishing 8-by-8 minors=81-l*r.                 (5)
```

This is an exact score, but cofactor sparsity is only a search heuristic. It
does not imply that another trade will lower rank.

## Exhaustive initial component

Starting from the R44 column partition at step 42, R45 enumerates every
equal-pair-sum trade. It recursively follows every child of rank at most eight
without a state cap. The resulting complete rank-eight component has

```text
rank-eight states=25,
rank-eight edge occurrences=52,
rank-nine edge occurrences=6,852,
distinct rank-nine boundary states=6,647.        (6)
```

No edge reaches rank seven or below. The root remains the strongest
rank-eight cofactor witness:

```text
left-null support=8,
right-null support=4,
vanishing 8-by-8 minors=49.                      (7)
```

The other 24 component states do not improve (7).

## Budgeted bridge search

The rank-nine boundary is sorted by decreasing vanishing-minor count, with the
canonical partition as a deterministic tie breaker. R45 expands eight layers
of 64 rank-nine states. At each layer it retains at most 32 new rank-eight
closure states; deferred queue sizes and truncation flags are recorded.

The frozen totals are

```text
expanded rank-nine states=512,
sum of layer-unique child counts=198,401,
rank-eight child counts within layers=1,108,
rank-nine child counts within layers=197,293,
retained rank-eight states=25+8*32=281,
final distinct rank-nine pool=303,694.            (8)
```

The 281 retained rank-eight states have score histogram

| Vanishing 8-by-8 minors | States |
|---:|---:|
| 0 | 248 |
| 9 | 31 |
| 18 | 1 |
| 49 | 1 |

The 512 expanded rank-nine bridges have score histogram

| Vanishing 8-by-8 minors | States |
|---:|---:|
| 2 | 17 |
| 3 | 89 |
| 4 | 234 |
| 5 | 137 |
| 6 | 27 |
| 7 | 7 |
| 8 | 1 |

Every evaluated child has rank eight or nine. The beam does not produce rank
seven, and neither the rank-eight score nor the rank-nine score trends toward
the rank-two condition after the fixed budget.

## Scope

Only the initial 25-state component is exhaustively closed. Each later
rank-eight closure stops after 32 retained states, and only 64 rank-nine
bridges are expanded per layer. The total child count in (8) is unique within
each layer; a state may occur in multiple layers. The search is therefore a
finite negative for one deterministic trade graph traversal, not an
exhaustion of equal-sum partitions or primitive degree-nine pencils.

All retained exact rank-eight partitions are included in
`p1553_rank_minor_guided_partition_search_report_r45.json`. R45 supplies no
asymptotic pencil family, fresh-target locator, R10 queried coefficients,
relation-rank campaign, factor-base logarithms, or scalar-blind descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_FINITE_RANK_GUIDED_PRIMITIVE_PENCIL_NEGATIVE_GATE
ALL206_R44_RANK_EIGHT_START_CONTEXTS_COLLAPSE_TO_STEPS_42_AND_61
INITIAL_RANK_EIGHT_TRADE_COMPONENT_EXHAUSTIVE_25_STATES
INITIAL_DISTINCT_RANK_NINE_BOUNDARY_6647
EIGHT_LAYER_WIDTH64_RANK_NINE_BEAM_EXPANDS_512_STATES
RETAINED_RANK_EIGHT_CATALOG_281_STATES
MINIMUM_SECTION_RANK_EIGHT_REQUIRED_TWO
LATER_CLOSURES_AND_RANK_NINE_BRIDGES_BUDGETED
ARBITRARY_PRIMITIVE_DEGREE_NINE_PENCIL_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: perform a pencil-first exhaustive scan over the R45
catalog. Deduplicate every degree-nine block section from the 281 retained
partitions, enumerate every unordered pair of nonproportional section vectors,
evaluate the resulting rational pencil on all 81 interval points, and measure
complete nine-point fibers and covered points. A survivor must have nine
disjoint complete fibers covering all 81 points before it is passed to R42's
all-secret marked-target test.
