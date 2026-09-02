# P1553 fixed-class multiplication-kernel gate R56

## Classification

- Owner: existing P1553/IDEA-195 primitive split-pencil frontier; no new idea
  ID.
- Evidence: exact fixed-Picard-class multiplication matrix, cocycle-calibrated
  product replay, and exhaustive product-pair Plucker scan on the R38 toy; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_EXACT_FIXED_CLASS_RULING_GATE`.
- Labels: `toy`, `exact-multiplication-matrix`,
  `exhaustive-fixed-class-product-lines`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: the class-residue triple `(96,42,16)` gives factor
  catalogs of sizes `(14,14,16)` and 3,136 distinct split degree-nine product
  sections. Its exact multiplication map has dimensions `27->9`, rank nine,
  and kernel dimension 18. All 4,915,680 product pairs were classified:
  5,348 lines contain at least three catalog products, but every line varies
  exactly one factor and has a common divisor of degree six or seven. There
  are zero primitive coprime trisecants. No Shoup-bound improvement or ECDLP
  breakthrough follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R53 genus-one abc scope gate | `0af36c7dd73916d2bdc4d149e2c6176864148d552d22cf7daf55c73e930635d5` |
| R54 broadened coprime probe gate | `635c7daba279059e07860cc63698a2ae2d23e621b369dfbe69c2cc846d391b0b` |
| R55 maximum-factorization star gate | `01cfaa0c114c813f53b40edc47e95d86ea1d330eabf3fb7f1cae5572721bb96e` |
| R55 exact star report | `359c71d0e6d501d42625d2de0eec8150a90a608ccc2432b701fa8545ef682c14` |

## Fixed factor classes

For one within-window degree-three factor, let its class residue be the sum of
its path offsets modulo the subgroup order 103. R56 fixes

```text
(rho_1,rho_2,rho_3)=(96,42,16),
rho_1+rho_2+rho_3=51=360 mod 103.                 (1)
```

The corresponding factor catalogs have exact sizes

```text
|F_1|=14, |F_2|=14, |F_3|=16.                   (2)
```

Every Cartesian product is disjoint at the factor level in this class triple,
so there are

```text
14*14*16=3,136                                   (3)
```

distinct degree-nine zero divisors and section points. Each factor catalog
spans its complete three-dimensional section space. Deterministic basis
indices are `(0,1,3)` in all three catalogs.

The associated scalar Abel sums and plane-embedding translations are

| Class residue | Abel sum scalar | Translation |
|---:|---:|---:|
| 96 | 46 | 19 |
| 42 | 54 | 85 |
| 16 | 96 | 71 |

## Multiplication map and cocycle control

Put

```text
V_i=H^0(E,L_i), dim(V_i)=3,
W=H^0(E,L_1 tensor L_2 tensor L_3), dim(W)=9,
mu:V_1 tensor V_2 tensor V_3 -> W.               (4)
```

The first raw implementation multiplied translated plane-line evaluations and
compared them directly with the translated degree-nine basis. It failed the
constant-scaling assertion. That failure was preserved during development:
translation identifies the line bundles only after a nonconstant
trivialization cocycle is charged.

R56 obtains that cocycle from one reference basis product. The ratio between
its factor-product evaluation and degree-nine section evaluation is defined
on 94 of the 103 subgroup points, exactly the complement of its nine zeros.
Dividing every product by the same ratio makes its scale constant. With that
calibration, the 27 basis-tensor images form an exact `9 x 27` matrix with

```text
rank(mu)=9,
dim ker(mu)=18.                                  (5)
```

Every one of the 3,136 factor-coordinate tensors is multiplied through this
matrix and reproduces its independently constructed normalized degree-nine
section. Thus (5) is bound to all finite catalog products, not just the 27
basis columns.

## Exhaustive Plucker scan

There are

```text
C(3136,2)=4,915,680                               (6)
```

product pairs. R56 computes each normalized 36-coordinate Plucker key. Two
deterministic 64-bit linear fingerprints only group possible duplicate keys;
every group is rechecked with the full exact key, so fingerprint collisions
cannot create or erase a reported line.

The complete line classification is

| Catalog products on line | Lines |
|---:|---:|
| 3 | 4,900 |
| 4 | 448 |

Hence

```text
exact catalog trisecant lines=5,348.              (7)
```

Their common base-divisor degrees are

| Common degree | Lines |
|---:|---:|
| 6 | 196 |
| 7 | 5,152 |

Most importantly, direct comparison of all factor-index triples gives

```text
varying factor count per line: {1:5,348}.         (8)
```

Two factors stay fixed on every line. They supply a degree-six common divisor;
degree-seven cases occur when the varying degree-three factors share one
additional zero. Thus all 5,348 lines are product rulings inherited from a
degree-three factor pencil. None is a non-ruling projected-Segre trisecant:

```text
primitive coprime trisecant lines=0.              (9)
```

The canonical full-classification digest is

```text
d4008153e91b37eac62b2c357db8b105bea9f4f0654be9be906c401c28cb6257.
```

The JSON report preserves six representative exact lines, three at each base
degree, and the deterministic replay reconstructs the digest.

## Scope

R56 exhausts the finite split product points only for class residues
`(96,42,16)`. It does not classify all rational points of the projected Segre
variety, other Picard-class triples, factors crossing window boundaries, or
nonproduct degree-nine sections. The large kernel in (5) alone does not
produce a useful split trisecant.

R56 supplies no asymptotic pencil family, fresh-target locator, R10 queried
coefficients, relation-rank campaign, factor-base logarithms, or scalar-blind
descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_EXACT_FIXED_CLASS_RULING_GATE
CLASS_RESIDUES_96_42_16_FACTOR_COUNTS_14_14_16
3136_DISTINCT_SPLIT_PRODUCT_SECTIONS
COCYCLE_CALIBRATED_MULTIPLICATION_MATRIX_27_TO_9
MULTIPLICATION_RANK_NINE_KERNEL_DIMENSION_18
ALL3136_PRODUCTS_RECONSTRUCTED
4915680_PRODUCT_PAIRS_EXHAUSTED
5348_CATALOG_TRISECANT_LINES
4900_THREE_POINT_LINES_448_FOUR_POINT_LINES
196_BASE_DEGREE_SIX_5152_BASE_DEGREE_SEVEN
EVERY_LINE_VARIES_EXACTLY_ONE_FACTOR
ZERO_PRIMITIVE_COPRIME_TRISECANTS
OTHER_CLASS_TRIPLES_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: enumerate the maximal 3,136-product class triples up
to factor permutation and path reflection, then repeat the exact
multiplication and Plucker classification for one representative of every
remaining orbit. Preserve any non-ruling line; otherwise close only the
maximal-product orbit family.
