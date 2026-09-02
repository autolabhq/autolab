# P1553 maximal-isogeny translate-diversity gate R26

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 Kummer image and algebraic
  addition-law lane; no new idea ID.
- Evidence: exact function-field theorem and algorithm-interface consequence;
  no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: a globally low-degree algebraic translate law for a
  high-degree Kummer function comes only from its isogeny translation
  stabilizer, and that same isogeny is injective on the campaign's prime
  subgroup. It cannot simultaneously create the `B^2` pullback support and a
  constant-degree translate alphabet. Finite-list exceptional saturation
  remains open. No Shoup-bound improvement or ECDLP breakthrough is supplied.

R25 leaves one prospective object: a degree-`Theta(B)` Kummer map whose
`Theta(B)` selected values have `Theta(B^2)` prime-subgroup preimages and
whose `Theta(B)` popular shifts yield only `O(B^2)` complete triples. A
natural positive route is an algebraic addition theorem of degree independent
of `B`.

Every elliptic function admits some algebraic addition theorem. The issue is
its degree. R26 proves a quantitative translate statement from the function
field and the pole divisor: after removing the maximal isogeny through which
`f` factors, any nonzero prime-subgroup translate paired with `f` generates
the full quotient function field. The irreducible translate correspondence
has degree equal to the remaining outer degree.

If that outer degree is small, the selected `B` values have only `O(B)`
prime-subgroup preimages because the isogeny kernel misses the prime subgroup.
If they have the required `B^2` preimages, the translate correspondence degree
is `Omega(B)`. Thus a constant-degree or `o(B)` global addition law cannot be
the missing complete-triple compressor.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R21 quadratic-intermediate-field gate | `ad8d9e83b919ea92c98eefcb657831f01378b6760d49b2aaea0a892d6ed52be1` |
| R23 random fifth-deck correlation gate | `de9a0a496fc6696ed6e976351476c56262484124c8c7d7cc2cdbddd16a6cc184` |
| R24 distinguished branch-value mass gate | `dd9b27fa5114e2121463612f5afaf09db12de30a0a0c4397f1dc37aafac1e99f` |
| R25 popular-difference complete-triple gate | `28b24d76a8fbecf50b755c7f808283664bfd52ba50395cc097d43a191ef7f70a` |
| R25 parent report | `6996b1b5c86fcee41d4f18364a02cba4f99a75658ec29f3f42ff46f9188d1266` |
| R25 bundle hash list | `f0935e23417296866b1f31bc8c9b8321cd31d3dc472f295464293ece74c65416` |
| R25 staging receipt | `2c5819dd6f1226637e46116c5ae0df2629e1c31f1a62c34852b6706d55a67a1a` |

## Frozen function-field model

Work over an algebraically closed field `k` of characteristic greater than
`m` and greater than three. Let `E/k` contain an odd prime-order subgroup
`G=<P>` of order `N`, with `N` different from the characteristic. Fix a
nonconstant separable even function

```text
f:E->P^1,        degree m<N.
```

For the P1553 Kummer map `f=psi composed with x`, one has `m=2d`.

Define the translation stabilizer

```text
H_f={K in E(k):f(R+K)=f(R) identically in R}.
```

It is a finite subgroup. Put

```text
e=|H_f|,
alpha:E->E'=E/H_f,
f=g composed with alpha,
c=degree(g)=m/e.                                  (1)
```

The quotient is separable in the frozen characteristic range. It is the
maximal separable isogeny quotient through which `f` factors: the kernel of
any isogeny quotient carrying `f` is contained in `H_f`.

Since `e<=m<N` and `G` has prime order,

```text
H_f intersect G={O}.                              (2)
```

Thus `alpha` is injective on `G`.

## Theorem 1: exact degree of every nonzero translate pair

For nonzero `T in G`, let

```text
K_T=k(f(R),f(R+T)) subset k(E)
```

and let `Y_T` be the smooth curve with function field `K_T`. Write

```text
e_T=[k(E):K_T].                                   (3)
```

Both displayed functions factor through the finite map `E->Y_T` of degree
`e_T`. Riemann-Hurwitz gives `genus(Y_T)<=1`.

Assume first that `Y_T` has genus zero. Let `D=(f)_infinity`, of degree `m`.
The pole divisors of the two functions are

```text
D and t_(-T)(D).
```

They are pullbacks of divisors of the same degree `m/e_T` on `P^1`. All
divisors of a fixed degree on `P^1` are linearly equivalent, so

```text
D linearly_equivalent t_(-T)(D).
```

Taking Abel sums on `E` gives

```text
m*T=O.
```

This is impossible because `0<m<N` and `T` is a nonzero point of the
prime-order group. Therefore every nontrivial intermediate `Y_T` has genus
one.

In that case `E->Y_T` is unramified and, after origins are chosen, is a
separable isogeny followed by a translation. Because `f` factors through it,
its kernel is contained in `H_f`; hence

```text
e_T<=e.                                            (4)
```

Conversely, both `f(R)` and `f(R+T)` are invariant under every translation in
`H_f`, so

```text
K_T subset k(E)^(H_f),
e_T>=e.                                            (5)
```

Combining (4)-(5) proves the exact identity

```text
[k(E):k(f(R),f(R+T))]=e=|H_f|                     (6)
```

for every nonzero `T in G`.

Equivalently, after passing through `alpha`, the map

```text
R' -> (g(R'),g(R'+alpha(T)))
```

is birational onto its image.

## Theorem 2: exact translate diversity

Let `Gamma_T` be the irreducible image curve of

```text
R -> (f(R),f(R+T)) in P^1 x P^1.
```

The first projection has generic fiber size

```text
degree(f)/degree(E->Gamma_T)=m/e=c.                (7)
```

The same holds for the second projection. Hence the irreducible polynomial
of the translate correspondence has bidegree

```text
(c,c).                                             (8)
```

For a generic value `a`, as `R` runs through the full geometric fiber
`f(R)=a`, the translated values `f(R+T)` assume exactly `c` distinct values,
each with the isogeny-kernel multiplicity `e`.

This is the quantitative addition-law dichotomy:

```text
small global translate degree c
  iff
most of degree m is an isogeny translation stabilizer.
```

No monodromy assumption is used.

## Theorem 3: prime-subgroup pullback versus addition degree

Let `C subset P^1(k)` contain `r` projective values and put

```text
Z_C={R in G:f(R) in C}.
```

By (2), `alpha` is injective on `G`. Every value of the degree-`c` outer map
`g` has at most `c` preimages on `alpha(G)`. Therefore

```text
|Z_C|<=c*r.                                        (9)
```

This improves the raw degree-`m*r` bound whenever a large isogeny factor is
present.

If a prospective P1553 support requires

```text
|Z_C|>=kappa*S
```

for a fixed `kappa>0`, then necessarily

```text
c>=kappa*S/r.                                      (10)
```

At the R25 point `S=Theta(B^2)` and `r=Theta(B)`,

```text
c=Omega(B).                                        (11)
```

By (8), every nonzero prime-subgroup translate correspondence then has degree
`Omega(B)` in each value coordinate.

Conversely, a constant-degree or `o(B)` translate relation has
`c=o(B)` and can expose only `o(B^2)` prime-subgroup points above `B` selected
values. It fails the support-mass gate before image construction.

## Consequence for the complete addition-surface route

The complete triple

```text
(f(A),f(A+Q),f(A-Q))
```

lies on an algebraic addition surface. R26 does not deny the classical
existence of that surface. It proves that neither one translated branch nor
any global addition law that projects through a degree-`o(B)` translate
correspondence can carry the R25 support.

The standard low-degree positive control is Lattes/isogeny structure: take a
large isogeny `alpha` and a constant-degree outer Kummer coordinate `g`. Then
`c=O(1)` and the addition law is small. Equation (9) shows why it fails here:
the isogeny is injective on `G`, so `B` output values contain only `O(B)`
prime-subgroup points, not `B^2`.

The surviving loophole is finite-list exceptional saturation. A degree-`c`
correspondence can have branch or exceptional values on which fewer than `c`
translated outputs occur. R26 does not yet bound a prospectively chosen
`C` of size `B` that is exceptional simultaneously for `B` shifts, nor prove
that its complete triples have large image. That finite intersection is the
next theorem target.

## Primary-source boundary

- Villarino,
  [On meromorphic mappings admitting an Algebraic Addition Theorem](https://arxiv.org/abs/math/9806078),
  supplies a modern structural account of algebraic addition theorems and
  associated algebraic group laws. R26's quantitative bidegree identity is an
  elementary function-field deduction in the frozen elliptic setting.
- Chalcraft and Fryers,
  [Kummer structures](https://arxiv.org/abs/0806.0409),
  supplies the two-valued Kummer setting used by the complete branch key.
- Shoup,
  [Lower Bounds for Discrete Logarithms and Related Problems](https://www.shoup.net/papers/dlbounds1.pdf),
  remains the generic-group baseline. R26 does not extend or beat it.

## Deduplication

- R21 owns the quadratic-intermediate-field closure for a shared complete
  branch factor across growing endpoint blocks.
- R24 owns distinguished-value source mass.
- R25 owns popular differences, additive energy, and the separate complete
  collision budget.
- R26 adds the exact maximal-isogeny formula (6), translate bidegree (8), and
  prime-subgroup pullback bound (9).

## Controls and limits

1. The theorem requires separability and `m<N` in characteristic greater than
   `m`.
2. The isogeny quotient is not declared absent; it is a passing algebraic
   control rejected by prime-subgroup injectivity.
3. The result is global in the rational functions. It does not classify a
   selected finite exceptional value set.
4. An `Omega(B)` translate correspondence may still have a succinct circuit;
   represented degree alone is not an arithmetic-circuit lower bound.
5. No complete-triple image constructor, source section, fresh target, R10
   index, relation density, independent rank, factor logs, or blind descent
   is supplied.
6. No unrestricted lower bound, Shoup improvement, or breakthrough is
   claimed.

## Scoped disposition

```text
maximal isogeny translate-pair degree: exact
constant-degree Lattes addition law: fails B^2 prime-subgroup pullback
mass-admissible B^2 pullback: forces translate degree Omega(B)
finite B-value/B-shift exceptional saturation: open
complete B^2 triple image: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Classify the finite exceptional intersection left by (8)-(11). For a
degree-`Omega(B)` outer map `g`, prove or refute that `B` prospectively chosen
values can be exceptional for `B` distinct prime-subgroup translates while
their `B^2` pullback points have `B` popular shifts and only `O(B^2)` complete
triples. Require all charts, exact sources, fresh-target/R10, and full costs;
reject an isogeny kernel, adaptive value selection, scalar orientation, or a
represented `B^3` incidence table.
