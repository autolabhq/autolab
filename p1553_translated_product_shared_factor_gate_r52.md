# P1553 translated-product shared-factor gate R52

## Classification

- Owner: existing P1553/IDEA-195 primitive degree-nine pencil frontier; no new
  idea ID.
- Evidence: exact exhaustive divisor audit of every positive R51 three-block
  line; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_EXHAUSTIVE_SHARED_FACTOR_CLASSIFICATION_GATE`.
- Labels: `toy`, `exact`, `exhaustive-divisor-audit`, `non-run`,
  `model-bound`, `novelty-unverified`.
- Cryptanalytic result: every one of the 228 R51 three-block pencil lines has
  exactly two common local triple factors and six common zero points. Removing
  that degree-six base divisor leaves the three `A,B,C` residual fibers in one
  translated R38 window, hence the original degree-three pencil. There are
  zero primitive degree-nine lines in the positive R51 family. No Shoup-bound
  improvement or ECDLP breakthrough follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R38 low-boundary pencil gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R51 translated-triple product gate | `4b60aab9732a448c718f570262b6a44dd28fbdde6797090bba78fb1f59761ca2` |
| R51 bundle hash list | `0883d51fb4c15eb0865b2d6b34662863d4f560bc6976c8a14df2c97215762f35` |
| R51 staging receipt | `b05d5dd13b531d1fe01a75e57ef40898054f413144b07c4a292a3a19dbb57da6` |

## Divisor classification

Each R51 candidate section has a reduced nine-point zero divisor that is the
union of three labeled local triples `(window,color)`. R52 recovers all 228
Plucker lines containing three such sections and intersects both their labeled
triple sets and their point divisors.

Every line has the exact form

```text
s_A = h * a_w,
s_B = h * b_w,
s_C = h * c_w,                                  (1)
```

where

```text
div_0(h) = two common local triples, degree 6,
div_0(a_w), div_0(b_w), div_0(c_w)
          = A, B, C in one translated window w. (2)
```

The exhaustive histograms are

```text
common labeled local triples per line: {2:228},
common divisor points per line:        {6:228},
residual color set per line:           {{0,1,2}:228}. (3)
```

Residual-window counts are

| Window | Lines |
|---:|---:|
| 0 | 21 |
| 1 | 27 |
| 2 | 21 |
| 3 | 27 |
| 4 | 36 |
| 5 | 27 |
| 6 | 21 |
| 7 | 27 |
| 8 | 21 |

They sum to all 228 lines. Every common two-factor pair names exactly one line.

## Cancellation

The three sections in (1) are linearly dependent because `a_w,b_w,c_w` are
the three fibers of the translated R38 degree-three pencil. Their apparent
degree-nine pencil has the fixed base divisor `div_0(h)`. Cancelling `h` gives

```text
effective map degree = 9-6 = 3.                 (4)
```

The six common points are not moving fiber points and cannot contribute to a
basepoint-free nine-point fiber. Each residual fiber contains only three
points. Thus the 228 positive collinear triples are inherited degree-three
pencils, not primitive degree-nine maps.

Exact result:

```text
inherited translated R38 pencil lines=228,
primitive degree-nine lines=0.                  (5)
```

## Scope

R52 classifies only the positive lines in the R51 translated-product family.
It does not exclude non-product primitive degree-nine pencils or prove an
asymptotic theorem about lines among split divisor sections. The exact common-
factor pattern suggests such a theorem, but possible three-term S-unit
relations without a large gcd must be handled rather than assumed absent.

R52 supplies no asymptotic pencil family, fresh-target locator, R10 queried
coefficients, relation-rank campaign, factor-base logarithms, or scalar-blind
descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_EXHAUSTIVE_SHARED_FACTOR_CLASSIFICATION_GATE
ALL228_R51_THREE_BLOCK_LINES_AUDITED
EVERY_LINE_HAS_TWO_COMMON_LOCAL_TRIPLES_SIX_COMMON_POINTS
EVERY_RESIDUAL_TRIPLE_IS_ABC_IN_ONE_TRANSLATED_R38_WINDOW
FIXED_BASE_DIVISOR_DEGREE_SIX_RESIDUAL_MAP_DEGREE_THREE
ALL228_LINES_INHERITED_FROM_R38
ZERO_PRIMITIVE_DEGREE_NINE_SURVIVORS
NONPRODUCT_PRIMITIVE_PENCILS_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: turn the finite pattern into a theorem candidate.
For three linearly dependent split sections of one elliptic line bundle,
factor out their divisor gcd and apply the genus-one function-field `abc` or
S-unit bound to the residual relation. Determine whether a residual relation
among products of translated low-degree pencil fibers must vary only one
factor, or else pays support proportional to the number of factors. State the
exceptional hypotheses explicitly and test the resulting bound against the R51
family before any asymptotic exclusion claim.
