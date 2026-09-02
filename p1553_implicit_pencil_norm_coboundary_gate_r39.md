# P1553 implicit pencil norm and coboundary gate R39

## Classification

- Owner: existing P1553/IDEA-195 pencil and marked-target frontier; no new
  idea ID.
- Evidence: exact divisor accounting, scoped standard-route cost screen, and
  deterministic toy self-check; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE_STANDARD_IMPLICIT_GATE`.
- Labels: `theorem-only`, `toy`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: the implicit translated-product predicate is exact,
  but its standard norm, resultant, divisor, radical, and constant-recurrence
  realizations do not meet the R36 density-adjusted target gate. The R38 toy
  locator is not a multiplicative translation coboundary, its naive degree-nine
  composition fails a necessary saturation condition, and every tested
  arithmetic pair deck gives essentially full cyclic linear complexity. A
  special high-degree Miller/net circuit or another source-returning nonlinear
  locator is not ruled out. No Shoup-bound improvement or ECDLP breakthrough
  follows.

R38 left open an operation that tests a translated endpoint intersection
without scanning its changed support. R39 writes down the strongest natural
candidate. If `F_C=H_C(g)` vanishes on the selected pencil union `Z`, then

```text
A_V(R)=product_(v in V) F_C(R-v)                 (1)
```

vanishes exactly when the other pair deck `V` intersects `R-Z`. Equation (1)
is an exact implicit target predicate, not merely a heuristic. The issue is
representation: it is a degree-`B^4` rational function, while a successful
masked batch has to represent at least `B^3` distinct shifted endpoints.
Standard algebra exposes one of those payloads before it returns a marked
intersection.

The only meaningful compression left inside this formula is structural. A
translated product can telescope if `F_C` is a multiplicative coboundary, and
special divisors can sometimes have short Miller or elliptic-net programs even
when their degree is large. R39 tests the first condition exactly and preserves
the second as the open exception.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R36 source/target-support gate | `103670908f33baee0915372434899a3a710861f501e19c91599f4221fe3f2aa5` |
| R37 generic path reduction gate | `b8a9423edef4041fcc78ec9f4808ab63ab2f97aa4fc1b157afaedabadb87c772` |
| R38 target-walk support/work gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R38 bundle hash list | `4aa30dd51575016598aebdc1debda7d2ccbf0864d396db86dabe398c0d995b7f` |
| R38 staging receipt | `885ffe5e76cf8fb2232d2bf98aefeda27f1a7d6b79440d635bb9d65f12b0a64a` |
| P1513 translated-product/common-norm V3 | `ce24397ea1686d081dac51b790fcfdf09f17e0a714dc0e8fef399fbb97c2d551` |
| P1513 KU common-norm gate | `bc7f9e44852ff6a7a7e59d52676154e82f3b43988b7824ce8a2fbb8c0cd260c6` |
| P1513 KU circuit reduction V2 | `6fcca1d12e911f6eb2142ac96b6d0a83b6ac20db11efd06bc24c0abb7c99dc48` |
| P1540 translated-pole annihilator gate | `d9a4040230022c24f7011932ef7cd9b5bcea51236a80c042bb498d2012428437` |
| P1540 independent audit | `8032be2d3a645ac64c046783191cc9c634715518eb18e4702acf66e077223d45` |

## Frozen implicit target interface

Let `E/k` contain the prime-order subgroup `G=<P_0>` with

```text
N=B^(5+o(1)).                                    (2)
```

Retain the R36 selected pencil data

```text
g:E->P^1,       degree(g)=B,
C subset P^1(k), |C|=B,
Z={P in G:g(P) in C}, |Z|=Theta(B^2),
V subset G,     |V|<=B^2.                        (3)
```

Choose a homogeneous polynomial `H_C(U,V)` of degree `B` whose roots are the
selected values. After dividing by a disjoint reference fiber of the same
degree, obtain a rational function `F_C` with

```text
zero_degree(F_C)=pole_degree(F_C)=B^2,
{P in G:F_C(P)=0}=Z.                             (4)
```

Reference-fiber multiplicity and exceptional charts are retained. A claimed
algorithm must return an actual `(v,z)` and the R36 source marks, not only one
field bit. False positives are replayed by complete group addition, while false
negatives are forbidden.

## Theorem 1: the exact norm predicate has degree B^4

Define (1) as a rational function of `R`. Its zero divisor is the translated
sum, with occurrence multiplicity,

```text
div_0(A_V)=sum_(v in V) tau_v(div_0(F_C)).        (5)
```

Consequently

```text
A_V(R)=0  iff  exists (v,z) in V x Z with R=v+z, (6)
deg(A_V)<=|V|*deg(F_C)<=B^4.                     (7)
```

Equation (6) is valid even when several sources have the same endpoint. It
does not reveal which factor vanished. Binary subdivision of `V` can recover a
marked factor from an exact subset-stable predicate in `O(log B)` calls, but it
does not reduce the cost of one call.

The standard representations expose the following payloads:

```text
dense numerator/denominator or full divisor      B^4,
pair-deck quotient plus translated F evaluations B^2 per target before F cost,
resultant/norm in the pair-deck algebra          dimension at least B^2,
expanded composed-sum endpoint polynomial       B^4.                 (8)
```

The exact exponents differ from P1513's selector norm because R39 starts after
the selected pencil has already compressed one endpoint family. The mechanism
is not new: output-degree norm/resultant, quotient, dense modular composition,
and post-hoc factor extraction are the same standard grammars screened by
P1513. None is an output-sensitive marked zero-factor locator.

This is a representation and route screen, not an arithmetic-circuit lower
bound. A circuit may have size much smaller than the degree of the rational
function it computes.

## Theorem 2: radicalizing the batch exposes the R38 support charge

For a known-log target-offset batch `T`, put

```text
D=T-Z.                                           (9)
```

For a random known-log base mask `R_0`, the batch can be positive only on

```text
R_0 in V-D.                                      (10)
```

R38 proved

```text
Pr[positive batch]<=min(1,|V|*|D|/N).           (11)
```

Therefore a batch with constant scalar-blind success needs

```text
|D|>=N/|V|=B^(3-o(1)).                           (12)
```

Any method that first materializes the radical support `D`, its roots, or a
dense squarefree polynomial with those roots writes at least `B^3` field
elements. This is already above rho's `B^(5/2)` work boundary. A smaller batch
may use less support, but independent masks restore the exact support/work
cancellation of R38.

Finite-field exponentiation, Fermat masks, gcd, or squarefree extraction after
forming the norm does not alter (7) or (12). This does not exclude an implicit
predicate whose circuit represents neither the degree-`B^4` norm nor the
degree-`B^3` successful radical.

## Theorem 3: the R38 locator is not a translation coboundary

Let `q` be the known shift and let `tau_q` translate rational functions. If

```text
F=tau_q(h)/h,                                    (13)
```

then on every `q`-translation orbit `Omega` in `E(kbar)` the valuation mass
telescopes:

```text
sum_(P in Omega) ord_P(F)=0.                     (14)
```

For the exact R38 pencil, write its numerator and denominator line sections as
`N` and `D`. The self-check takes

```text
F=N*D*(N-192D)/(N-D)^3.                          (15)
```

The value `1` has no subgroup preimage. Thus (15) has nine zeros on the
subgroup orbit `G`, no pole on that orbit, and

```text
sum_(P in G) ord_P(F)=9.                         (16)
```

Equations (14) and (16) contradict (13). Translating (15) over any nonempty
endpoint deck multiplies the positive mass on `G`, so its norm is not a
coboundary either. The obvious one-step multiplicative telescope is absent in
the positive toy.

There is a broader degree warning. Suppose another locator is a coboundary
(13), has a zero at every point of a length-`L` `q`-interval, and has no pole
on that interval. Write

```text
d_i=ord_(R+[i]q)(h).
```

Across the interval, (13) forces `d_(i+1)-d_i>=1`. The `L+1` valuations span a
range of at least `L`. After any integral shift, either their positive mass or
their negative mass is `Omega(L^2)`. Since a principal divisor has equal total
positive and negative degree,

```text
degree(h)=Omega(L^2).                             (17)
```

At `L=B^2`, a coboundary primitive has divisor degree `Omega(B^4)`. Equation
(17) does not prove that its straight-line program has that size: Miller
functions and division polynomials are the explicit warning that high-degree
elliptic functions can have short addition-chain descriptions. A special
Miller/net circuit for the required interval divisor remains outside the gate.

## Theorem 4: the naive degree-nine scale-up fails inner saturation

The most direct attempt to scale the R38 degree-three witness is to retain its
shift and take an 81-point interval for a degree-nine selected union. If a
degree-three outer map were composed with the inner pencil and nine outer
values were selected, the interval could meet at most

```text
3*9=27                                           (18)
```

inner pencil values.

The exact self-check evaluates the R38 pencil on the first 81 path points. It
finds

```text
66 distinct inner values,
57 values with multiplicity 1,
 3 values with multiplicity 2,
 6 values with multiplicity 3.                   (19)
```

Since `66>27`, the interval cannot be the selected preimage of that naive
degree-three composition. This rejects only the literal self-composition of
the toy. It is not a theorem against a new degree-nine pencil or another
asymptotic low-boundary family.

## Theorem 5: the toy norm has no short constant recurrence

The script `p1553_implicit_pencil_norm_selfcheck_r39.py` evaluates (15) on all
103 subgroup points. For each nondegenerate arithmetic pair deck

```text
V_(a,b)={i*a+j*b:0<=i,j<3}, |V_(a,b)|=9,         (20)
```

with `1<=a<=b<103`, it evaluates the complete periodic sequence

```text
s_R=product_(v in V_(a,b)) F(R-v).               (21)
```

Berlekamp-Massey is run on two periods and the recovered recurrence is checked
on a third. All 4,896 nondegenerate grids have cyclic linear complexity

```text
LC(s)=102                                        (22)
```

over `F_193`. Three exact controls are:

| Deck | `|V+Z|` | `LC(s)` | zero-indicator LC |
|---|---:|---:|---:|
| aligned nine-point path | 17 | 102 | 103 |
| grid `(1,7)` | 75 | 102 | 103 |
| grid `(11,29)` | 45 | 102 | 103 |

Thus even the aligned deck with only 17 zero positions does not induce a short
constant-coefficient recurrence in either the intrinsic norm values or its
zero indicator. P1540 already supplies the symbolic translated-pole warning
for x-coordinate sequences; R39 adds a route-matched positive-toy control.

This is not an asymptotic recurrence lower bound. It does not exclude a
variable-coefficient recurrence, a nonlinear state machine, a target-dependent
annihilator, or a recurrence available only in a specially constructed
asymptotic pencil family. In particular, the explicit ratio
`A_V(R+q)/A_V(R)` for a path deck is a boundary update and is already charged
by R38; high linear complexity does not contradict that variable-coefficient
identity.

## Cost and source gate

The R36 density-adjusted requirement is an average fresh-target operation

```text
kappa<1/2-o(1)                                   (23)
```

under setup `s<=9/4+o(1)`, with all exponents measured in `B`. The screened
objects have costs or represented sizes

```text
single explicit pair-deck intersection          B^2,
constant-success materialized batch radical     B^3,
dense translated norm                           B^4,
naive toy norm constant-recurrence state         full subgroup order. (24)
```

None meets (23). A surviving implicit operation must simultaneously provide:

1. an asymptotic degree-`B` pencil and `B^2` selected endpoint set;
2. target-independent setup at most `B^(9/4+o(1))`;
3. all-negative-mask and initialization cost below `B^(5/2-o(1))` total;
4. average scalar-blind query below `B^(1/2-o(1))` after density;
5. an exact marked `(v,z)` output, not only a product value;
6. R36 complete source replay and R10 queried coefficients;
7. independent relation rank, factor-base logs, and blind descent.

No current object supplies item 1 together with items 2-5. Consequently there
is nothing to bind to R10, rank, logs, or descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE_STANDARD_IMPLICIT_GATE
EXACT_TRANSLATED_NORM_PREDICATE_POSITIVE
STANDARD_NORM_RESULTANT_DIVISOR_AND_RADICAL_PAYLOADS_MISS_GATE
R38_TOY_LOCATOR_NOT_A_MULTIPLICATIVE_COBOUNDARY
COBOUNDARY_INTERVAL_PRIMITIVE_HAS_DEGREE_OMEGA_B4
NAIVE_DEGREE9_COMPOSITION_FAILS_66_GT_27
ALL_4896_TOY_ARITHMETIC_PAIR_DECK_NORMS_HAVE_LC_102_OF_103
MILLER_NET_OR_NONLINEAR_SOURCE_LOCATOR_OPEN
NO_TARGET_R10_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: construct or refute the remaining orbit-segment
primitive. Put the poles of a selected-path locator on the subgroup orbit,
derive the minimum principal transfer divisor `h`, and test whether Miller,
elliptic-net, or addition-chain identities evaluate `tau_q(h)/h` and return one
marked vanishing factor below the density-adjusted `B^(1/2)` query gate without
materializing its `Omega(B^4)` divisor. The same construction must exhibit a
degree-nine lift of the R38 pencil before any asymptotic or R10 claim.
