# P1553 random fifth-deck complete-correlation gate R23

## Classification

- Owner: existing P1553/IDEA-057/IDEA-121/IDEA-195 correlation and
  common-factor-free image-router lane; no new idea ID.
- Evidence: exact collision identity and probabilistic theorem; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: a fifth Kummer deck sampled independently after the
  rational map and endpoint deck are frozen has no asymptotically useful
  complete-key compression. This is not a theorem for adversarially co-designed
  decks, an unrestricted lower bound, a Shoup-bound improvement, or an ECDLP
  breakthrough.

R21 closed every growing global common-right-factor block. R22 then showed
that fitting a denominator to one finite fifth list creates small-field
collisions which disappear on a disjoint holdout. R23 isolates the
list-independence issue without another search.

For a fixed separable rational `psi` and any fixed endpoint deck, the exact
second moment of the complete Kummer keys is a sum of incidences on pairwise
branch correspondences. Every such correspondence has only `O(d)` choices in
one fifth coordinate after the other is fixed. Sampling one `L`-element
fifth deck from the `Theta(N)` prime-subgroup Kummer classes therefore leaves
only `O(B)` expected nontrivial ordered collisions among `Theta(B^3)`
sources when `d<=B`. The attained complete image has expected size
`B^3-O(B)`, far above the `B^(9/4)` state cap.

The theorem is prospective: `psi`, every endpoint, and every chart must be
frozen before the random fifth deck is drawn. If coefficients or supports are
adapted to the sampled deck, the theorem does not apply. That is exactly the
R22 training loophole.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R13 erased-image geometric-resolution gate | `78ff09226852f140a8c33ce5ba98106ed62007b7126673d3f3af2c49d08a2cc5` |
| R15 translation-correlation gate | `ea007be0b237b91127e6a42501873f914ad31d6d4fd4393c3ae6b85005893b9b` |
| R21 quadratic-intermediate-field gate | `ad8d9e83b919ea92c98eefcb657831f01378b6760d49b2aaea0a892d6ed52be1` |
| R21 bundle hash list | `362b820cc46124e01e3c0d5f9e07cf6fac2287fd09b78160bb040e3e1ac2d64b` |
| R22 list-only toy gate | `ba1661d183f4e8b7291a1a34b48bdab2d01485da45c9753bbfb16e82b52208ab` |
| R22 exact search report | `40ef33016a6e12b802ee7c40183c853e4d180d89b7993abf85f1d6e6c48a5af8` |
| R22 exact holdout report | `b32b017c2d126e537891c5891cb31d5720caeee660eb1d715d0fbe1e04190302` |
| R22 bundle hash list | `77aac4a770d4717a43078fd2573f92c22048f5ff55fe931139a990e0cb02e5d1` |

## Frozen model

Let `G=<P>` have odd prime order `N` different from the field
characteristic, and let

```text
X=(G minus {O})/{plus or minus 1},       M=|X|=(N-1)/2.
```

Work over a field of characteristic greater than three. Fix, before sampling
the fifth deck,

```text
psi:P^1->P^1,                 separable degree d,
f=psi composed with x:E->P^1, degree 2d<N,
U subset X,                   |U|=S.
```

For endpoint class `u` with either signed lift `A`, define the complete
projective branch map

```text
F_u(q)=[f(A+Q)]+[f(A-Q)] in Sym^2(P^1),
Phi(u,q)=(psi(u),F_u(q)).
```

The unordered divisor makes this independent of both sign choices. Tangent,
identity, pole, and infinity values are retained in the saturated projective
map rather than deleted by an affine denominator test.

Let `Delta` be an upper bound for the degree of every nonconstant morphism
`F_u:P^1_q->F_u(P^1_q)`. The constant-branch resultant used in R14 gives

```text
Delta <= c_0*d
```

for an absolute chart-independent constant `c_0`: the two branch values are
the roots of a quadratic addition polynomial and applying a degree-`d`
rational map gives trace and norm of degree `O(d)` after homogeneous
saturation. The proof below is stated in `Delta`, so it does not depend on an
optimized value of `c_0`.

Finally choose `Q` uniformly from the `L`-element subsets of `X`, independently
of `psi` and `U`. Put `n=S*L`.

## Lemma 1: branch maps do not coincide across endpoint classes

Every `F_u` is nonconstant. Otherwise both the trace and norm of

```text
f(A+Q), f(A-Q)
```

would be constant. The connected curve `E` could then map through only two
values under `Q->f(A+Q)`, forcing `f` to be constant.

If `u` and `v` are distinct Kummer classes, then `F_u` and `F_v` are not the
same rational map. Indeed, equality of their trace and norm makes
`f(A+Q)` a root of the quadratic with roots `f(B+Q),f(B-Q)`. Since the
function field is a domain, identically either

```text
f(A+Q)=f(B+Q)
```

or

```text
f(A+Q)=f(B-Q).
```

The first identity makes `f` invariant under translation by `A-B`; using the
evenness of `f`, the second makes it invariant under translation by `A+B`.
For distinct Kummer classes both translations are nonzero in the prime-order
group and have order `N`. A nonconstant separable map of degree `2d<N`
cannot factor through that order-`N` quotient. Both alternatives are
impossible.

The endpoint label adds one more elementary restriction: distinct endpoints
can collide under `Phi` only when `psi(u)=psi(v)`. Since `psi` has degree `d`,
the number of ordered distinct endpoint pairs with equal label is at most

```text
S*(d-1).
```

## Lemma 2: each pairwise branch relation is sparse in the full deck

For endpoints `u,v`, define

```text
R_(u,v)={(q,r) in X^2:F_u(q)=F_v(r)}.
```

Fixing `q`, the point `F_u(q)` has at most `Delta` preimages under the
nonconstant map `F_v`. Therefore

```text
|R_(u,v)| <= M*Delta.                              (1)
```

For distinct endpoint classes, Lemma 1 also bounds the diagonal section.
After homogeneous cross-multiplication, at least one coordinate difference
is a nonzero univariate form of degree at most `2*Delta`, so

```text
#{q in X:F_u(q)=F_v(q)} <= 2*Delta.                (2)
```

These bounds allow common image curves, ramification, and non-Galois fibers.
No genericity or common-factor classification is used. They require only the
complete projective map and its degree.

## Theorem 1: exact complete collision-correlation identity

For a realized fifth deck `Q`, let `m_y` be the number of sources in
`U x Q` attaining complete key `y`, let `K_Q` be the number of attained keys,
and define the ordered nontrivial collision count

```text
C_Q=sum_y m_y*(m_y-1).
```

Then exactly

```text
sum_y m_y^2 = n+C_Q,                               (3)

C_Q=sum_((u,q)!=(v,r))
      1[psi(u)=psi(v)]*1[F_u(q)=F_v(r)],           (4)

n-K_Q <= C_Q,                                     (5)

K_Q >= n^2/(n+C_Q).                               (6)
```

Equation (4) is the list-independent complete three-point collision
correlation. It retains both translated branches and every projective chart;
it is not a trace-only, one-branch, or random-key heuristic. Equation (6) is
Cauchy-Schwarz, while (5) follows fiber by fiber from
`m-1<=m*(m-1)`.

This identity also states the necessary incidence budget for any proposed
compression. Achieving `K_Q=O(n/d)` requires

```text
C_Q=Omega(n*d).                                    (7)
```

A finite-list constructor must therefore exhibit and verify that much exact
pairwise branch-correlation mass; a small number of visually large fibers is
not enough.

## Theorem 2: an independent random fifth deck is almost injective

For sampling without replacement put

```text
rho_1=L/M,
rho_2=L*(L-1)/(M*(M-1)).
```

A fixed diagonal fifth pair is selected with probability `rho_1`; a fixed
ordered pair of distinct fifth classes is selected with probability `rho_2`.
Using (1)-(2), the same-endpoint contribution and the distinct-endpoint
contribution give

```text
E[C_Q]
 <= S*M*Delta*rho_2
    +S*(d-1)*(M*Delta*rho_2+2*Delta*rho_1)

 <= S*d*M*Delta*rho_2+2*S*d*Delta*rho_1.          (8)
```

Since `Delta=O(d)`, this is

```text
E[C_Q]
 =O(S*d^2*L^2/M + S*d^2*L/M).                    (9)
```

In the P1553 rectangle

```text
M=Theta(B^5), S=Theta(B^2), L=Theta(B), d<=B,
```

equations (3), (5), and (9) yield

```text
E[C_Q]=O(B),
E[sum_y m_y^2]=n+O(B),
E[n-K_Q]=O(B),
E[K_Q]>=n-O(B).                                  (10)
```

For every fixed `epsilon>0`, Markov's inequality gives

```text
Pr[K_Q<=(1-epsilon)*n]=O(1/(epsilon*B^2)).         (11)
```

Thus even constant-factor complete-key compression vanishes with probability
`1-O(B^-2)`. In particular, the `Theta(d)` compression needed to bring the
R13 image down to `O(SL/d)` is absent with the same asymptotic probability.

## Consequence for moment and generating-function routes

R14 and R15 already give the exact trace-Hankel and two-frequency correlation
interfaces. IDEA-121 separately closes generic short Newton and logarithmic-
derivative locators. R23 adds the missing output-size fact for an independent
fifth deck: with high probability the radical complete-image polynomial has

```text
degree K_Q=SL-O(B)=Theta(B^3),
```

so even a hypothetical quasi-linear output-sensitive decoder exceeds the
`B^(9/4)` setup and state cap. A nested norm, rational generating function,
or collision sketch cannot claim a smaller output merely because its input
has `O(S+L+d^2)` words.

This does not say that the exact generating function is difficult to write.
It says that, on an independent deck, its squarefree denominator is itself
too large. Any useful successor must co-design a deterministic fifth support
with `psi` and prove the incidence budget (7) prospectively.

## Controls and limits

1. `psi`, its coefficients, the endpoint deck, and all chart choices are
   frozen before `Q` is sampled. Adaptive denominator fitting is outside the
   theorem.
2. The result concerns distinct geometric Kummer classes. Repeated endpoint
   occurrences can raise multiplicity but do not create independent columns
   or geometric image compression.
3. The theorem is exact over the sampled finite set; only the final deck
   choice is probabilistic.
4. A deterministic field-defined factor-base interval is not asserted to be
   random. Such a deck needs its own incidence proof and held-out controls.
5. Adversarial support co-design, target-dependent advice, inseparable maps,
   degree at least `N`, composite source torsion, and arbitrary nonlinear
   circuits remain outside scope.
6. No R23 result supplies a fresh multivalued target action, R10 coefficients,
   relation density, independent rank, factor-base logs, or blind descent.
7. This is not an unrestricted generic-group lower bound and makes no Shoup
   improvement or breakthrough claim.

## Interpretation of R22

R22 selected `psi` after observing its training fifth list, so Theorem 2 does
not explain away the training collisions. Its frozen disjoint holdout did
restore near-injectivity: 145 regular sources produced 136 keys, maximum
fiber two, and no zero-branch collision excess. That observation is
consistent with R23 but is not used to prove it.

The combined boundary is now precise:

```text
global common-factor compression: closed by R21 pending review
training-list denominator fitting: rejected by R22
independent random fifth deck: almost injective by R23 pending review
prospective adversarial support co-design: open
```

## Exactly one next action

Derive or refute one prospective common-factor-free support co-design. Freeze
an algebraic rule for constructing the fifth deck from `psi` before endpoint,
target, and holdout choices; prove the exact `C_Q=Omega(SLd)` complete-chart
incidence budget on at least two independently generated endpoint decks and a
fresh target interface, with source backpointers and R10 queried coefficients
charged. Reject it at the first adaptive coefficient, pole deletion,
training-only incidence gain, hidden common factor, `SL` table, or missing
rank/log/descent path.
