# P1553 translate-curve Bezout-saturation gate R27

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 finite exceptional-value and
  complete-collision lane; no new idea ID.
- Evidence: exact singularity, intersection, and incidence-energy bounds; no
  run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: an `O(B^2)` complete image in the R26 exceptional
  branch must asymptotically saturate the Bezout intersection budget for a
  positive fraction of all selected shift pairs. The saturation must occur
  either through selected singularities consuming `Omega(B^3)` total delta
  budget or through a value-pair grid carrying `Omega(B)` distinct
  shifts on positive incidence mass. No such algebraic design, fresh-target
  interface, or ECDLP breakthrough is supplied.

R26 removes the maximal isogeny stabilizer and leaves a degree-`c` map
`g:E'->P^1` with no nonzero translation stabilizer. For every nonzero shift
`T`, the map

```text
phi_T(P)=(g(P),g(P+T))
```

is birational onto an integral curve `Gamma_T` of bidegree `(c,c)` and
genus-one normalization.

R27 quantifies the only remaining way a finite `C` and shift set `Q` can beat
the generic translate diversity. Same-shift identifications are singularities
of `Gamma_T` and consume its arithmetic-genus defect. Different shifts are
intersections of two `(c,c)` curves and consume at most `2c^2` each, unless
the curves are identical. Identical curves are controlled by the at-most-six
automorphisms of `E'` preserving `g`.

At `c=|C|=|Q|=Theta(B)`, the resulting image floor is `Omega(B^2)`. Reaching
that floor requires `Theta(c^2)` selected intersections for `Theta(|Q|^2)`
shift pairs. A second incidence double count then forces one of two exact
objects: selected singularities saturating the genus defect, or a
fiber-difference grid with `Theta(c^2)` value pairs carrying
`Theta(|Q|)` shifts. This dichotomy is the next exact construction target.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R23 random fifth-deck correlation gate | `de9a0a496fc6696ed6e976351476c56262484124c8c7d7cc2cdbddd16a6cc184` |
| R25 popular-difference complete-triple gate | `28b24d76a8fbecf50b755c7f808283664bfd52ba50395cc097d43a191ef7f70a` |
| R26 maximal-isogeny translate-diversity gate | `0f9d0d8f7d2cbc2b73ba848f252f0f0d9a1d2b870411c5610725ff33c67b364c` |
| R26 parent report | `a77f0d76c6c7b52e17d89efc8ef5907bf0eceb29e62272efabc774d2b3737fa0` |
| R26 bundle hash list | `a410f9433d0b2291fb70151c648f5ec6ab3cb01e0ced7bd8ed70948c9a8d3457` |
| R26 staging receipt | `64cb06101dcff48488f7d13ab9806b8bb564c2bc4ff9df176d4ab091c418d4a4` |

## Frozen quotient model

Work over an algebraically closed field of characteristic greater than
`c` and greater than three. Let

```text
g:E'->P^1
```

be a nonconstant separable map of degree `c` with trivial translation
stabilizer. Let `G' subset E'(k)` be the injective image of the campaign's
prime-order subgroup under the maximal isogeny quotient from R26.

Fix prospectively

```text
C subset P^1(k), |C|=r,
Z={P in G':g(P) in C}, |Z|=z,
Q subset G' minus {O}, |Q|=L.
```

The mass-admissible regime has

```text
z=Theta(c*r), c=Theta(B), r=Theta(B), L=Theta(B).
```

For `T in Q`, define the complete translated key

```text
K_T(P)=(g(P), unordered pair {g(P+T),g(P-T)}).
```

The source rectangle has `n=z*L` points. Let `K` be the number of attained
complete keys and

```text
Energy=sum_y multiplicity(y)^2.
```

Then exactly `K>=n^2/Energy`.

## Theorem 1: singularity budget for one translate curve

For `T!=O`, let `Gamma_T` be the image of `phi_T`. R26 gives bidegree `(c,c)`
and normalization `E'`, of genus one. The arithmetic genus of an integral
`(c,c)` curve in `P^1 x P^1` is

```text
p_a(Gamma_T)=(c-1)^2.
```

Therefore the total delta invariant is

```text
sum_(y singular) delta_y=(c-1)^2-1.                (1)
```

If a singular point has `s_y` distinct normalization preimages, its local
delta invariant is at least `binomial(s_y,2)`. Hence the total number of
ordered distinct pairs `P,R in E'` with

```text
phi_T(P)=phi_T(R)
```

is at most

```text
2*((c-1)^2-1)=O(c^2).                              (2)
```

This counts every finite exceptional value, including nonordinary
singularities, with its normalization multiplicity. Restricting to `Z` can
only reduce the count.

## Theorem 2: distinct translate curves have a `2c^2` intersection budget

For shifts `T,U`, the intersection number of two distinct integral bidegree
`(c,c)` curves is

```text
Gamma_T dot Gamma_U=2*c^2.                         (3)
```

Every ordered normalization-preimage pair above a geometric intersection
uses at least one unit of local intersection multiplicity. Thus, when the
curves are distinct,

```text
#{(P,R):phi_T(P)=phi_U(R)}<=2*c^2.                 (4)
```

Complete unordered-branch equality has two possible branch matchings. It is
therefore bounded by comparisons of

```text
Gamma_T with Gamma_U,
Gamma_T with Gamma_(-U),
```

and costs at most `4c^2` normalization pairs when both compared curves are
distinct.

It remains to control equal curves. Define

```text
A_g={sigma in Aut(E'):g composed with sigma=g}.
```

Because `g` has no nonzero translation stabilizer, the map taking an
automorphism to its origin-fixing linear part is injective on `A_g`. In
characteristic greater than three,

```text
|A_g|<=6.                                           (5)
```

If `Gamma_T=Gamma_U`, normalization gives one `sigma in A_g` satisfying

```text
g(sigma(P)+U)=g(P+T).
```

For fixed `T` and `sigma`, at most one `U` works: two choices would make their
difference a nonzero translation stabilizer of `g`. Consequently, for each
`T`, at most six shifts `U` have `Gamma_T=Gamma_U`, and the same constant
applies to `Gamma_T=Gamma_(-U)`.

For an equal-curve comparison, the normalization identifies at most one
generic source in the other curve for each `P in Z`; exceptional extra pairs
are charged by (1). A safe bound is

```text
z+2*((c-1)^2-1)=O(z+c^2).                          (6)
```

## Theorem 3: complete energy and image floor

Sum (2), (4), and (6) over the `L^2` ordered shift pairs and the two branch
matchings. Absolute constants are retained only to prevent an asymptotic
claim from hiding an exceptional family. One obtains

```text
Energy
 <=4*L^2*c^2+12*L*(z+2*c^2).                       (7)
```

The diagonal `n=zL` is already covered by the equal-curve term; adding it
separately would only weaken the bound.

Therefore

```text
K
 >= z^2*L^2 /
    (4*L^2*c^2+12*L*(z+2*c^2)).                    (8)
```

When

```text
z=Theta(c*r), c=Theta(r)=Theta(L)=Theta(B),
```

equation (8) gives

```text
K=Omega(r^2)=Omega(B^2).                           (9)
```

This independently recovers the optimal R23/R13 image scale in the finite
exceptional branch. It does not exclude an `O(B^2)` constructor; it proves
that such a constructor must saturate the geometric bounds rather than obtain
additional asymptotic compression from exceptional values.

## Theorem 4: an `O(r^2)` image forces Bezout saturation

Assume fixed constants `kappa,A>0` with

```text
z>=kappa*c*r,
K<=A*r^2,
c=Theta(r),
L->infinity.                                       (10)
```

Cauchy-Schwarz forces

```text
Energy>=z^2*L^2/(A*r^2)
      >=kappa^2*c^2*L^2/A.                         (11)
```

The `O(L*(z+c^2))` equal-curve and singularity terms are lower order in the
balanced regime. Since every distinct curve comparison contributes at most
`2c^2`, equations (7) and (11) imply:

```text
a positive fraction of the ordered shift/matching pairs must contribute
Omega(c^2) selected intersection multiplicity.     (12)
```

If R25's both-branch containment holds, every selected one-branch
intersection used by a complete collision has coordinates in `C x C`.
Thus (12) is a finite-grid Bezout-saturation requirement, not merely a large
geometric intersection over the algebraic closure.

## Theorem 5: singularity saturation or a distinct-shift fiber design

For `(a,b) in C x C`, define

```text
s_T(a,b)=#{P in Z:phi_T(P)=(a,b)},
k_(a,b)=sum_(T in Q) s_T(a,b).
```

Thus `k_(a,b)` counts selected source incidences represented as

```text
T=R-P,
g(P)=a,
g(R)=b.                                            (13)
```

Define

```text
I=sum_((a,b) in C^2) k_(a,b),
D=sum_(T,(a,b)) s_T(a,b)*(s_T(a,b)-1),
J=sum_((a,b) in C^2) k_(a,b)^2.                    (14)
```

Here `I` is the number of selected normalization/source incidences on the
value-pair grid, `D` is the same-curve singular collision defect, and `J` is
the one-branch source-pair energy inside that grid. Theorem 1 gives

```text
D<=2*L*((c-1)^2-1)=O(L*c^2).                       (15)
```

For each `(a,b)`, Cauchy-Schwarz over the at most `L` shifts gives

```text
k_(a,b)^2
 <=L*sum_T s_T(a,b)^2
 =L*(sum_T s_T(a,b)*(s_T(a,b)-1)+k_(a,b)).
```

Summing yields the exact global bound

```text
J<=L*(D+I).                                        (16)
```

The complete-image saturation from Theorem 4 forces one of the two branch
matchings to have

```text
J=Omega(L^2*c^2).                                   (17)
```

To separate singular multiplicity from distinct shifts, put

```text
ell_(a,b)=#{T in Q:s_T(a,b)>0},
e_(a,b)=k_(a,b)-ell_(a,b).
```

For every grid point,

```text
e_(a,b)^2
 <=L*sum_T (s_T(a,b)-1)_+^2
 <=L*sum_T s_T(a,b)*(s_T(a,b)-1).
```

Also `sum e_(a,b)<=D`. Since `ell_(a,b)<=L`,

```text
0<=sum k_(a,b)^2-sum ell_(a,b)^2
  <=2*L*sum e_(a,b)+sum e_(a,b)^2
  <=3*L*D.                                          (18)
```

Combining (15)-(18) gives the necessary dichotomy:

```text
(A) D=Omega(L*c^2): selected singularities consume a constant fraction of
    the total arithmetic-genus defect across the shift family; or

(B) D=o(L*c^2), and the distinct-shift counts retain
    sum ell_(a,b)^2=Omega(L^2*c^2).                 (19)
```

In branch (B), `ell_(a,b)<=L`, so (19) forces

```text
sum ell_(a,b)=Omega(L*c^2)
```

and positive distinct-shift incidence mass on value pairs carrying
`Omega(L)` shifts. With `r=Theta(c)`, this uses `Omega(c^2)` grid points up to
constants. The conclusion counts distinct shifts even if a grid point is
geometrically singular; singular normalization multiplicity has been removed
by (18).

Branch (A) is not relabelled as a grid design. It is a separate, equally
rigid singularity-saturation object: a positive fraction of the selected
translate curves must place a constant fraction of their entire
`(c-1)^2-1` delta budget on `C x C`. A candidate may use either branch, but it
must report which one and provide the corresponding normalization sources.

Random exceptional values, isolated singularities, or a handful of popular
differences satisfy neither branch at the required scale.

## Controls and boundaries

1. The singularity and Bezout bounds count algebraic multiplicity, so
   tangencies and nonordinary singularities are not discarded.
2. Equal translate curves are admitted and bounded through `A_g`; they are
   not assumed absent.
3. The constants in (7) are safe rather than optimized. The exponent and
   saturation conclusions do not use a hidden genericity assumption.
4. The singularity/grid dichotomy (19) is necessary, not a construction or
   impossibility theorem for either branch.
5. A scalar progression may have many popular point differences but still
   owes the value-fiber grid, complete triple collisions, and DLP-free chart.
6. No output-sensitive constructor, fresh target, R10 index, relation
   density, independent rank, factor logs, or blind descent is supplied.
7. No unrestricted lower bound, Shoup improvement, or breakthrough is
   claimed.

## Deduplication

- R23 owns the complete collision identity and `Omega(SLd)` budget.
- R25 owns point-set popular differences and additive energy.
- R26 owns maximal-isogeny translate diversity and the global low-degree
  addition-law exclusion.
- R27 adds singularity defect, pairwise Bezout saturation, and the value-fiber
  difference design (13)-(19).

## Scoped disposition

```text
finite exceptional complete image: K=Omega(B^2)
candidate O(B^2) image: requires Bezout saturation across Omega(B^2) shift pairs
selected singularity branch: requires Omega(B^3) total delta saturation
distinct-shift value-pair branch: requires Omega(B) shifts on positive incidence mass
algebraic singularity or fiber-difference design: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Construct or refute both branches of (19). Freeze a degree-`Theta(B)` outer
map `g`, `Theta(B)` values `C`, and `Theta(B)` public prime-subgroup shifts
`Q`; require either selected delta saturation `Omega(B^3)` or a value grid
with `Omega(B)` represented shifts on positive incidence mass, followed by
only `O(B^2)` complete triples and exact source replay. Then charge
construction, fresh-target/R10, rank, logs, and descent. Reject a scalar
chart, adaptive grid selection, isogeny stabilizer, pair table, or failure of
the complete branch matching.
