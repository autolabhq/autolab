# P1553 degree-three inner-composition gate R43

## Classification

- Owner: existing P1553/IDEA-195 asymptotic interval-pencil frontier; no new
  idea ID.
- Evidence: exhaustive finite-field pencil enumeration and exact interval
  scan; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_EXHAUSTIVE_TOY_COMPOSITION_NEGATIVE_GATE`.
- Labels: `toy`, `exact`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no rational degree-three map on the R38 toy curve can
  serve as the inner map of a degree-nine composition whose nine selected
  fibers contain an 81-point subgroup interval. Exhausting both rational
  translation classes and all 37,237 valid line-pencil centers gives maximum
  complete triple-fiber coverage 51, below the required 81. The original R38
  pencil also maps every 81-point orbit interval to at least 60 inner values,
  above the compositional maximum 27. Primitive degree-nine pencils remain
  open. No Shoup-bound improvement or ECDLP breakthrough follows.

R42 requires the next scale of a pencil-only coordinate mechanism. The first
construction is composition: use a degree-three inner pencil, a degree-three
outer rational map on `P^1`, and select nine outer values. R39 rejected the
literal self-composition on one interval. R43 exhausts every rational
degree-three inner pencil on the exact toy curve and closes the whole
composition class there.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R31 near-period compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R38 target-walk and pencil gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R39 implicit norm/composition gate | `2901305c2ac83fe51d2f23957389cb2dfb6d58b3c4bc0d2e439ad4ef3f6085f6` |
| R42 single-interval pencil DLP gate | `8a8de8726e4fc9542f97cf7e1ba4fc54596307cc544da0d0b18b588553022dae` |
| R42 bundle hash list | `fa218b20a5b79da63d5883497606c4d1fa3f7fd1a9ad9417c410f7c289ba20b8` |
| R42 staging receipt | `8e128b8cd0452a24f64407a5989569230cd1b6dd3ad66061364b1f149b77ab56` |

## Composition requirement

Let

```text
g:E->P^1,       degree(g)=3,
phi:P^1->P^1,   degree(phi)=3.                  (1)
```

Then `phi o g` has degree nine. Selecting nine outer values gives at most

```text
9*degree(phi)=27                                (2)
```

inner `g`-values, counting multiplicity. If their selected subgroup union has
81 distinct points, every inner value must have exactly three subgroup
preimages and all 27 complete inner fibers must be used. Thus a necessary
condition is

```text
complete three-point subgroup-fiber coverage of g >=81. (3)
```

This condition is independent of how the outer map groups the 27 values. If
(3) fails, no degree-three outer map can repair it.

## Theorem 1: rational degree-three maps reduce to two embedding cosets

The exact toy curve has

```text
E/F_193: y^2=x^3+2x+3,
|E(F_193)|=206,
|G|=103.                                         (4)
```

Every rational degree-three line bundle is a rational translate of
`O_E(3O)`. Indeed `Pic^3(E)(F_193)` is an `E(F_193)` torsor, and multiplication
by three is bijective on `E(F_193)` because `gcd(3,206)=1`. After translating
the source, a basepoint-free degree-three map is a two-dimensional subspace of
`H^0(E,O(3O))`, equivalently the pencil of lines through one rational point of
the standard plane cubic not lying on the curve.

Translations fall into two classes relative to `G`:

```text
T in G:       the embedded source set is G,
T notin G:    the embedded source set is the other coset T+G. (5)
```

Translations within one class only permute that 103-point set. It therefore
suffices to enumerate every rational line-pencil center against these two
cosets.

The projective plane has

```text
193^2+193+1=37,443                              (6)
```

rational points. Removing the 206 curve points leaves exactly

```text
37,237                                           (7)
```

valid basepoint-free degree-three pencil centers in each embedding class.

## Theorem 2: exhaustive maximum coverage is 51

For each valid center `B` and each of the 103 embedded source points `P`, the
script `p1553_degree_three_composition_exhaustive_r43.py` normalizes the line
through `B` and `P`. Equal normalized lines are one pencil fiber. A complete
subgroup triple is a line label occurring exactly three times.

On the subgroup embedding, the complete-triple coverage histogram over all
37,237 centers is:

| Covered points | Centers |
|---:|---:|
| 6 | 8 |
| 9 | 63 |
| 12 | 274 |
| 15 | 1,011 |
| 18 | 2,662 |
| 21 | 5,137 |
| 24 | 7,277 |
| 27 | 7,473 |
| 30 | 6,538 |
| 33 | 4,012 |
| 36 | 1,840 |
| 39 | 672 |
| 42 | 206 |
| 45 | 51 |
| 48 | 11 |
| 51 | 2 |

The maximum is 17 complete fibers, covering

```text
17*3=51 points.                                  (8)
```

It occurs only at centers

```text
(83,63,1), (83,130,1).                           (9)
```

Since `51<81`, condition (3) fails for every rational degree-three pencil in
the subgroup embedding.

For the other coset, every one of the 37,237 centers has zero complete
three-point fibers. There is also a direct quotient explanation: the sum of
three points in the nontrivial coset of `E(F_193)/G` remains in that coset
because `3=1 mod 2`, while three collinear points on the cubic must sum to
`O in G`. Hence no line can contain three points of the other coset.

Together, the two exhaustive classes cover every rational degree-three map,
so no such map reaches the 81-point requirement.

## Theorem 3: the original pencil has no eligible 81-point interval

As a separate exact control, the script evaluates the original R38 pencil on
all

```text
102*103=10,506                                  (10)
```

length-81 subgroup intervals, over every nonzero step and every start. The
number of distinct inner values ranges from 60 to 73. Exact minimum data are

```text
minimum distinct values=60,
step=43,
start=12,
multiplicity histogram={1:49,2:1,3:10}.         (11)
```

No interval has at most 27 inner values. The original R38 interval from R39
had 66; it was not an unlucky choice hiding another compositional interval.

## Scope and remaining primitive pencil

R43 excludes only degree-nine maps that factor as degree three followed by
degree three on the exact R38 curve. It does not exclude a primitive
degree-nine map

```text
f:E->P^1, degree(f)=9                            (12)
```

whose nine selected fibers directly partition an 81-point interval. Such a
map corresponds to a pencil inside a degree-nine complete linear system. Its
nine fiber sections must all lie in one two-dimensional subspace of the
nine-dimensional `H^0(E,L)`. Equal Abel sums of the nine blocks are necessary
but not sufficient for this pencil-rank condition.

The result is toy-only. It gives no asymptotic theorem against composed maps
on other curves, no primitive degree-nine search result, and no target
locator.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_EXHAUSTIVE_TOY_COMPOSITION_NEGATIVE_GATE
ALL_RATIONAL_DEGREE3_INNER_PENCILS_ON_R38_CURVE_ENUMERATED
37237_CENTERS_IN_EACH_OF_TWO_TRANSLATION_CLASSES
SUBGROUP_COMPLETE_TRIPLE_COVERAGE_MAX51_REQUIRED81
OTHER_COSET_COMPLETE_TRIPLE_COVERAGE_ZERO
ALL10506_ORIGINAL_PENCIL_LENGTH81_INTERVALS_HAVE_AT_LEAST60_VALUES
DEGREE3_BY_DEGREE3_COMPOSITION_CLOSED_ON_TOY
PRIMITIVE_DEGREE9_PENCIL_OPEN
NO_TARGET_R10_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: search primitive degree-nine pencils rather than
compositions. Partition a length-81 orbit interval into nine degree-nine
blocks with equal Abel sum, construct their section vectors in `H^0(E,L)`,
and test whether all nine vectors have projective rank two. Start with
structured equal-sum partitions such as normal 9-by-9 magic-square rows, retain
all negative rank witnesses, and require any rank-two survivor to pass the R42
all-secret marked target test before considering asymptotic lifting.
