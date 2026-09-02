# P1553 coprime product-trisecant probe gate R54

## Classification

- Owner: existing P1553/IDEA-195 primitive split-pencil frontier; no new idea
  ID.
- Evidence: exact broadened product catalog, deterministic sampled coprime
  pencil probe, and inherited positive control on the frozen R38 toy; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_SAMPLED_COPRIME_TRISECANT_NEGATIVE_GATE`.
- Labels: `toy`, `exact-catalog`, `deterministic-sampled-pencils`, `non-run`,
  `model-bound`, `novelty-unverified`.
- Cryptanalytic result: all 756 within-window degree-three factors generate
  296,600 admissible factorizations and 278,321 distinct degree-nine product
  sections. A seed-1554 sample of 100,000 pairs with disjoint zero divisors
  scanned all 192 interior projective points per generated line, 19.2 million
  points total, and found zero third catalog sections. The same scanner finds
  the inherited R52 third section on its six-base-point control line. This is
  negative evidence only for the sampled coprime pencils; no exhaustive or
  asymptotic theorem, Shoup-bound improvement, or ECDLP breakthrough follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R44 primitive degree-nine section-rank gate | `bc5e2068f25814087f228c5bc422109b3f7d5122cfad6bb0600b6208d372885a` |
| R52 shared-factor gate | `060f27a6830a685f47ec241920bb5d3cb793336bba032e95f926a899bab018a2` |
| R52 audit report | `6f4d3645197eaa67f959bdefe886bfd586555b88f62426eccb761a047e317000` |
| R53 genus-one abc scope gate | `0af36c7dd73916d2bdc4d149e2c6176864148d552d22cf7daf55c73e930635d5` |

## Broadened factor catalog

The frozen path has 81 points split into nine consecutive windows of nine.
R51 used only the three translated R38 fibers in each window. R54 instead
takes every three-point subset of every window:

```text
9*C(9,3)=756 local degree-three factors.          (1)
```

Three pairwise disjoint factors are admissible when their nine point offsets
sum to 360, the fixed Abel class used by R44-R52. Exact enumeration gives

```text
admissible factorizations=296,600,
unique nine-point divisors=278,321,
unique section vectors=278,321.                   (2)
```

The factorization multiplicity histogram is

| Factorizations of one divisor | Divisors |
|---:|---:|
| 1 | 276,320 |
| 10 | 2,000 |
| 280 | 1 |

The unique 280-factorization divisor is the complete middle window

```text
{36,37,38,39,40,41,42,43,44}.                    (3)
```

The catalog is exact for products of three factors, each contained in one
frozen nine-point window. It is not the full split-divisor locus on the
elliptic curve and supplies no asymptotic parameterization.

## Coprime line probe

The deterministic seed-1554 sampler draws distinct section pairs and accepts
only pairs whose zero-divisor masks are disjoint. For each accepted pair
`(A,B)`, it evaluates and projectively normalizes

```text
A+tB,  t in F_193^*.                              (4)
```

Together with endpoints `A` and `B`, these are all 194 points of the
projective line. The 192 values in (4) are looked up in the complete
278,321-section catalog. Because the generating divisors are disjoint, any
third catalog hit would be a basepoint-free primitive trisecant in this
finite product family.

The frozen probe records

```text
sampling attempts=279,530,
accepted coprime pairs=100,000,
interior points per line=192,
interior catalog lookups=19,200,000,
positive coprime trisecant lines=0.                (5)
```

This is a deterministic sample, not all coprime pairs. Equation (5) cannot be
promoted to a nonexistence statement for the 278,321-section catalog.

## Positive control

R54 imports one pinned R52 line. Its three blocks share exactly the six points

```text
{18,23,25,55,57,62}.                              (6)
```

The first two section vectors generate a line whose unique interior catalog
hit at `t=1` is the expected third block. Thus the same normalization and hash
lookup that returns zero on the coprime sample detects the known inherited
trisecant. The control does not make the sampled negative result exhaustive.

## Scope

R54 broadens the factor catalog by more than three orders of magnitude over
R51 and directly tests the factor-sensitive object left open by R53. It does
not exhaust all pairs, factors crossing window boundaries, arbitrary
degree-three divisor classes, or nonproduct degree-nine sections. It supplies
no asymptotic pencil family, fresh-target locator, R10 queried coefficients,
relation-rank campaign, factor-base logarithms, or scalar-blind descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_SAMPLED_COPRIME_TRISECANT_NEGATIVE_GATE
756_WITHIN_WINDOW_DEGREE_THREE_FACTORS
296600_ADMISSIBLE_FACTORIZATIONS
278321_UNIQUE_DEGREE_NINE_PRODUCT_SECTIONS
FACTORIZATION_MULTIPLICITIES_276320x1_2000x10_1x280
100000_SEED1554_COPRIME_PENCILS
19200000_INTERIOR_PROJECTIVE_POINTS_CHECKED
R52_INHERITED_THIRD_SECTION_POSITIVE_CONTROL_DETECTED
ZERO_PRIMITIVE_COPRIME_TRISECANTS_IN_SAMPLE
PAIR_SPACE_NOT_EXHAUSTED
NO_ASYMPTOTIC_PRODUCT_LINE_THEOREM
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: exhaust the structured star through the unique
280-factorization middle-window section. Scan every catalog section with a
disjoint divisor, and every interior point on each resulting line, against
the complete catalog. Any hit is a primitive coprime trisecant; a zero count
closes exactly that high-multiplicity star and nothing broader.
