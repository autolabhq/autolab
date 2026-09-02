# P1553 genus-one abc product-line gate R53

## Classification

- Owner: existing P1553/IDEA-195 primitive split-pencil frontier; no new idea
  ID.
- Evidence: self-contained separable Riemann-Hurwitz theorem and exact divisor
  accounting against R52; no cryptanalytic run.
- Status: `STAGED_INDEPENDENT_REVIEW_PASS_COORDINATOR_ARCHIVAL_REQUIRED`.
- Labels: `theorem-only`, `exact`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: after cancelling the common divisor from three
  dependent sections on an elliptic curve, the single three-fiber
  function-field `abc`
  inequality is `m<=n`, where `m` is the moving map degree and `n` is the
  total radical size of its zero, one, and pole fibers. It correctly accepts
  R52 as `3<=9`, but it also accepts every hypothetical primitive reduced
  degree-nine split pencil as `9<=27`. The bound cannot see how residual
  points are grouped into low-degree factors and therefore cannot by itself
  prove a one-varying-factor theorem or an asymptotic exclusion. No Shoup-bound
  improvement or ECDLP breakthrough follows.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R38 low-boundary pencil gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R51 translated-triple product gate | `4b60aab9732a448c718f570262b6a44dd28fbdde6797090bba78fb1f59761ca2` |
| R52 shared-factor gate | `060f27a6830a685f47ec241920bb5d3cb793336bba032e95f926a899bab018a2` |
| R52 bundle hash list | `ea4392c4dcc8e5f34a455baa40622ac6aa320422875c392489fa95ae9fd8ff37` |
| R52 staging receipt | `8edcb0e589ef9a6b9745c54c4791d9239b29ccc8f5c379fbe76e27f391339962` |

## Theorem 1: divisor gcd and three fibers

Let `X/k` be a smooth projective geometrically integral curve of genus `g`,
let `L` be a line bundle, and let nonzero sections satisfy

```text
A+B=C in H^0(X,L).                                (1)
```

Let

```text
G=gcd(div_0(A),div_0(B),div_0(C)),
m=deg(L)-deg(G).                                  (2)
```

Assume `m>0`.

Dividing the sections by the canonical section of `O(G)` gives `a+b=c` in
`H^0(X,L(-G))` with no common zero. If a point were a zero of any two of
`a,b,c`, equation (1) would make it a zero of the third. Hence the residual
zero divisors

```text
D_0=div_0(a), D_1=div_0(b), D_infinity=div_0(c)   (3)
```

are pairwise disjoint and each has degree `m`. For

```text
f=a/c,
1-f=b/c,                                         (4)
```

they are exactly the fibers of `f` over `0,1,infinity`. The pair `[a:c]`
therefore defines a nonconstant finite morphism `X->P^1` of degree `m`.
This cancellation is mandatory: common zeros are base points, not moving
fiber points.

## Theorem 2: separable genus-one abc inequality

After base change to an algebraic closure, write

```text
n=#supp(D_0)+#supp(D_1)+#supp(D_infinity).        (5)
```

Assume `f` is separable. Riemann-Hurwitz gives

```text
deg Diff(f)=2g-2+2m.                              (6)
```

For each point over `0`, `1`, or `infinity`, its different exponent is at
least its ramification index minus one. Each of the three fibers has total
multiplicity `m`, so their contribution is at least

```text
(m-#supp(D_0))+(m-#supp(D_1))
 +(m-#supp(D_infinity))=3m-n.                    (7)
```

Combining (6) and (7) proves

```text
m<=n+2g-2.                                       (8)
```

On an elliptic curve this is exactly

```text
m<=n.                                            (9)
```

The proof works in arbitrary characteristic for a separable map, including
wild ramification because the different exponent can only increase. In
characteristic `p`, `m<p` implies separability: a nontrivial inseparable
degree is divisible by `p`. The P1553 asymptotic regime has moving degree
`m=O(B)` and, from `N=B^(5+o(1))` together with the elliptic Hasse interval,
prime field characteristic `p=B^(5+o(1))`. Thus, for sufficiently large `B`,
the candidate degrees under discussion are below `p`. Outside that regime,
inseparable maps require a separate reduction before (8) is used.

This is the three-term function-field `abc` bound. Brownawell-Masser
generalizes the height method to longer vanishing sums in characteristic
zero, but that stronger term-count framework is not needed for (8) and does
not add factor labels to the three valuations used here.

## Corollary 3: radical deficit is the only gate from this inequality

Put `n_i=#supp(D_i)`. Since each `D_i` has degree `m`, a reduced residual
fiber has `n_i=m`. Thus three reduced split fibers give

```text
n=3m,
m<=3m,                                           (10)
abc slack=n-m=2m.                                (11)
```

More generally, suppose each residual section is a product of `r`
degree-`d` factors and all residual zeros are reduced. Then

```text
m=rd,
n=3rd,
abc slack=2rd.                                   (12)
```

The valuation radical records only the union of zero points. It contains no
field for a factor label, translated window, multiplication-tree node, or
factor ownership. Consequently (8) alone cannot distinguish one varying
factor from many varying factors. It excludes a candidate only when the total
residual radical satisfies

```text
n<m-2g+2,                                        (13)
```

which on genus one is the strict radical deficit `n<m`. Any factor-sensitive
claim needs additional geometry of the multiplication image or auxiliary
factor-sensitive relations, not another application of the same support
inequality. More elaborate `abc`/S-unit arguments combined with such extra
input remain open.

## R52 audit

Every R52 degree-nine triple has a common reduced degree-six divisor. The
correct moving degree is therefore

```text
m=9-6=3.                                         (14)
```

The three residual `A,B,C` fibers are reduced and disjoint with three points
each, so

```text
n=3+3+3=9,
R52 abc check: 3<=9,
slack=6.                                         (15)
```

The six base points cancel from `A/C`; counting them as moving support would
misstate the map degree. R52 is closed by its exact common-factor
classification, not by a tight `abc` inequality.

For the requested asymptotic control, a hypothetical primitive reduced
degree-nine product pencil has

```text
G=0, m=9, n=27,
primitive control: 9<=27,
slack=18.                                        (16)
```

The inequality becomes less restrictive as independent reduced factors are
added. It supplies no contradiction to the primitive object that R51-R52
were searching for.

The deterministic self-check
`p1553_genus_one_abc_product_line_selfcheck_r53.py` loads the pinned R52
report, verifies its pass flag and divisor fields, then checks the R52
arithmetic, the primitive control, the reduced `r`-factor formula, the
`m<p` separability condition at the frozen toy, and a deliberately impossible
`n=m-1` negative control. The arithmetic checks the theorem's application; it
is not a computer proof of Riemann-Hurwitz.

## Scope and literature

Primary context:

- W. W. Stothers, *Polynomial identities and Hauptmoduln*, Quarterly Journal
  of Mathematics 32 (1981), 349-370,
  `doi:10.1093/qmath/32.3.349`.
- W. D. Brownawell and D. W. Masser, *Vanishing sums in function fields*,
  Mathematical Proceedings of the Cambridge Philosophical Society 100
  (1986), 427-434, `doi:10.1017/S0305004100066184`.

Those sources motivate the `abc`/S-unit language. The positive-characteristic
statement used here is the self-contained separable Riemann-Hurwitz argument
in Theorem 2, not an imported characteristic-zero theorem.

R53 does not classify trisecant lines of the elliptic multiplication image,
exclude coprime multi-factor identities, construct an asymptotic split-pencil
family, or supply a fresh-target locator. It supplies no R10 queried
coefficients, relation-rank campaign, factor-base logarithms, or scalar-blind
descent.

## Disposition

```text
STAGED_INDEPENDENT_REVIEW_PASS_COORDINATOR_ARCHIVAL_REQUIRED
DIVISOR_GCD_CANCELLED_BEFORE_MAP_DEGREE_IS_COUNTED
SEPARABLE_CURVE_ABC_M_LE_N_PLUS_2G_MINUS_2
GENUS_ONE_SPECIALIZATION_M_LE_N
R52_RESIDUAL_CHECK_THREE_LE_NINE_SLACK_SIX
PRIMITIVE_REDUCED_DEGREE_NINE_CHECK_NINE_LE_27_SLACK_18
REDUCED_R_FACTOR_PRODUCTS_HAVE_SLACK_2RD
SINGLE_THREE_FIBER_INEQUALITY_SEES_RADICAL_NOT_FACTOR_OWNERSHIP
NO_ONE_VARYING_FACTOR_THEOREM_FROM_THIS_INEQUALITY_ALONE
FACTOR_SENSITIVE_AUXILIARY_S_UNIT_ARGUMENTS_OPEN
MULTIPLICATION_IMAGE_TRISECANTS_REMAIN_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: replace support counting with a factor-sensitive
trisecant probe. On the frozen R38 elliptic toy, enumerate the full catalog of
within-window split degree-three factors and their admissible degree-nine
products, then scan a frozen deterministic sample of coprime generated lines
against the complete product catalog. Preserve any primitive coprime
trisecant as a construction seed; a zero count closes only the sampled lines.
