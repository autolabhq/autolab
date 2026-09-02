# P1553 translated-triple product-pencil gate R51

## Classification

- Owner: existing P1553/IDEA-195 primitive degree-nine pencil frontier; no new
  idea ID.
- Evidence: exact exhaustive structured-family section and Plucker-line scan
  on the R38 toy; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_TRANSLATED_TRIPLE_PRODUCT_STRUCTURAL_GATE`.
- Labels: `toy`, `exact`, `exhaustive-structured-family`, `non-run`,
  `model-bound`, `novelty-unverified`.
- Cryptanalytic result: tiling the 81-point path by nine translated copies of
  the R38 three-fiber pattern yields 253 admissible degree-nine blocks formed
  from unions of three local triple fibers. Their 31,878 section pairs define
  31,422 distinct pencils: 31,194 contain two candidate blocks and 228 contain
  three. This is the first tested primitive branch with genuine third
  collinear block sections, but no pencil contains the nine disjoint blocks
  needed to cover all 81 points. The maximum is three, and a canonical exact
  cover has section rank 9. No Shoup-bound improvement or ECDLP breakthrough
  follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R38 low-boundary pencil gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R43 degree-three composition gate | `b02901b1a822ce98f886146d5a04d2ea35b9515ec549358d2c4b7c02b6700ec8` |
| R44 primitive section-rank gate | `bc5e2068f25814087f228c5bc422109b3f7d5122cfad6bb0600b6208d372885a` |
| R50 external-pair pencil gate | `b7b8024af071ddf11bef600236c41a97cf3e9ed3bb56ff13e7e6fd48189d9791` |
| R50 bundle hash list | `5d128e21d1548fa0a367af33b58f63489124298fc3772803a9718d6482dade58` |
| R50 staging receipt | `62d8bf4d6ff265ee1d8588378d278790f7632c8786a3487391a183e8e8d89b7f` |

## Translated local fibers

In path coordinates `k=0,...,8`, the three complete R38 fibers are

```text
A={0,5,7}, B={1,3,8}, C={2,4,6}.                (1)
```

Each triple sums to 12. R51 tiles the length-81 path by windows `w=0,...,8`.
The local triple `(w,color)` contains offsets

```text
9w+A, 9w+B, or 9w+C.                            (2)
```

There are 27 labeled local triples. A degree-nine product block is a union of
three such triples. Its offset sum is

```text
27*(w_1+w_2+w_3)+36.                            (3)
```

For the common value 360 required by the fixed complete linear system,

```text
w_1+w_2+w_3=12.                                 (4)
```

Exhausting all three-item subsets satisfying (4) gives exactly 253 distinct
nine-point blocks and 253 distinct exact sections in `H^0(E,O(9O))`.

## Pencil-line scan

R51 computes the normalized Plucker key of every

```text
C(253,2)=31,878                                 (5)
```

section pair. If more than two section points share a key, they lie in one
two-dimensional pencil. Exact accounting gives

| Candidate blocks on pencil line | Pencil lines |
|---:|---:|
| 2 | 31,194 |
| 3 | 228 |

Hence

```text
distinct pencil lines=31,422,
maximum candidate blocks on one line=3.         (6)
```

The 228 three-block lines are a positive structural distinction from the R46
and R50 arbitrary-block catalogs, where every scanned pencil contained only
its two generating blocks. They are nevertheless far below the nine blocks
required for an interval cover.

## Exact-cover control

An eligible degree-nine interval pencil needs nine pairwise disjoint candidate
blocks whose labeled local triples cover all 27 items exactly once. R51 runs an
exact-cover test on every pencil line with at least nine candidate blocks.
There are none by (6), so no rank-two cover line survives.

As a positive combinatorial control, the 27 items do admit exact covers by nine
admissible blocks. One canonical cover groups each color separately using
window triples

```text
{0,4,8}, {1,5,6}, {2,3,7}.                     (7)
```

Its nine exact section vectors have rank 9, not 2. Thus the failure is the
common-pencil condition, not absence of a degree-nine block cover.

## Scope

R51 exhausts only blocks that are unions of three translated copies of the
specific R38 local triple fibers. Other products, sums of products, and
arbitrary primitive degree-nine sections remain open. The 228 three-block
lines require further divisor analysis: they may reflect shared base factors
rather than basepoint-free degree-nine maps.

R51 supplies no asymptotic pencil family, fresh-target locator, R10 queried
coefficients, relation-rank campaign, factor-base logarithms, or scalar-blind
descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_TRANSLATED_TRIPLE_PRODUCT_STRUCTURAL_GATE
R38_LOCAL_FIBER_PATTERNS_TILED_ACROSS_NINE_WINDOWS
253_ADMISSIBLE_DEGREE_NINE_TRIPLE_UNION_BLOCKS
31878_SECTION_PAIRS_31422_DISTINCT_PENCIL_LINES
31194_LINES_HAVE_TWO_BLOCKS_228_LINES_HAVE_THREE_BLOCKS
MAXIMUM_THREE_BLOCKS_REQUIRED_NINE
CANONICAL_27_ITEM_EXACT_COVER_EXISTS_WITH_SECTION_RANK_NINE
NO_RANK_TWO_EXACT_COVER_LINE
OTHER_PRODUCT_AND_PRIMITIVE_SECTIONS_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: classify all 228 three-block lines at divisor level.
For each line, intersect its three labeled local-triple sets and its three
nine-point zero divisors. Test whether exactly two local triples, hence six
points, are common base factors and whether the three residual local triples
are the `A,B,C` fibers in one translated R38 window. If so, divide out the
common factor and record the line as an inherited degree-three pencil rather
than a primitive degree-nine survivor.
