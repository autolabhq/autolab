# P1553 distinct-shift partial-orthogonal-grid gate R29

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 finite exceptional-value and
  complete-image lane; no new idea ID.
- Evidence: exact incidence extraction, partition theorem, and finite-list
  Bezout interpolation bound; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: R27 branch (B) forces `Theta(B)` public shifts for
  which two balanced fiber partitions cross in `Theta(B^2)` almost-singleton
  cells. This is a partial orthogonal-array design, not merely a popular
  difference set. Exact full-grid coverage by even one nonzero shift is
  impossible on a proper subset of the prime-order group. Any algebraic
  curve that captures a constant fraction of one surviving transition grid
  has total bidegree `Omega(B)`, unless it contains the full translate curve.
  Affine planes show that the orthogonal-partition scale is combinatorially
  attainable when a rank-two additive chart is supplied; that chart is not
  available inside the prime cyclic source group. The approximate partial
  design, compact exact incidence compiler, complete two-branch matching,
  and ECDLP path remain open.

R28 typed the selected-singularity branch. R29 handles the complementary
R27 regime, where normalization multiplicity is lower order but many
distinct shifts land on the same selected value pairs. The right object is
the transition matrix between the fiber partition of `g` and its translate.
Low singular defect makes this matrix nearly `0/1`; high distinct-shift mass
makes it dense for many shifts.

This reduction exposes two controls. A full `c by c` transition grid would
make the selected support invariant under a nonzero prime-group translation,
which forces the entire group and violates `c^2<N`. On the other hand, the
`c+1` parallel classes of an affine plane are exactly pairwise orthogonal
partitions of `c^2` points. They realize the abstract design but require a
public rank-two finite-vector-space chart or a represented incidence table.
Neither control settles a constant-density partial design on an algebraic
degree-`c` pencil, so no impossibility or breakthrough is claimed.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R15 translation-correlation gate | `ea007be0b237b91127e6a42501873f914ad31d6d4fd4393c3ae6b85005893b9b` |
| R19 Galois-orbit target-action gate | `b5aef048b9748763f6449e845fa7554755e12715dce695cd99bdc8f0b77df61f` |
| R25 popular-difference complete-triple gate | `28b24d76a8fbecf50b755c7f808283664bfd52ba50395cc097d43a191ef7f70a` |
| R26 maximal-isogeny translate-diversity gate | `0f9d0d8f7d2cbc2b73ba848f252f0f0d9a1d2b870411c5610725ff33c67b364c` |
| R27 translate-curve Bezout-saturation gate | `dee8a0913854e0ce3e717ecdbc3e8aea73b987b2b2fb71790a470f501dd0c2d9` |
| R28 selected-singularity common-subdivisor gate | `c24183bfe4823a9318ae06769763c94cfed99cb4291807f06b85f397fd53f662` |
| R28 parent report | `38daaae8f16ae1bfccd2695d26e34130a379f549d064742e9d1951d3afc96cbe` |
| R28 bundle hash list | `38970d51cf0592cfe34eb020f5d2798c44f47aa95e2abad1208d13addf42e48d` |
| R28 staging receipt | `cd701dd5034752b4ad0dab9563345c09077b69f87e228eea576f60c81b4baeeb` |

## Frozen quotient model

Retain the R27-R28 model. Over an algebraically closed field of
characteristic greater than `c` and greater than three, let

```text
g:E'->P^1
```

be separable of degree `c<N`, with trivial translation stabilizer, and let
`G'` be the prime-order subgroup of order `N`. Fix

```text
C subset P^1, |C|=r,
Z_a={P in G':g(P)=a},
Z=union_(a in C) Z_a, |Z|=z,
Q subset G' minus {O}, |Q|=L.
```

The sets `Z_a` are disjoint and `|Z_a|<=c`. Work in the balanced regime

```text
c=Theta(r)=Theta(L)=Theta(B),
z=Theta(c*r)=Theta(B^2),
N=Theta(B^5).
```

For `T in Q`, define the integer transition matrix

```text
S_T(a,b)=s_T(a,b)=|Z_a intersect (Z_b-T)|.
```

Put

```text
Omega_T=#{(a,b):s_T(a,b)>0},
D_T=sum_(a,b) s_T(a,b)*(s_T(a,b)-1),
I_T=sum_(a,b) s_T(a,b)=|Z intersect (Z-T)|.
```

Globally,

```text
Omega=sum_T Omega_T,
D=sum_T D_T.
```

R27 branch (B) supplies constants on a subsequence for which

```text
Omega=Omega(L*c^2),
D=o(L*c^2).                                        (1)
```

## Theorem 1: many dense, almost-simple transition matrices

There is a subset

```text
Q_good subset Q, |Q_good|=Theta(L),
```

and a sequence `epsilon_B->0` such that every `T in Q_good` satisfies

```text
Omega_T=Omega(c^2),
D_T<=epsilon_B*c^2.                                (2)
```

To see this, the first part of (1) and the upper bound
`Omega_T<=r^2=O(c^2)` force a positive fraction of shifts to have
`Omega_T=Omega(c^2)`. Set

```text
epsilon_B=sqrt(D/(L*c^2)).
```

The second part of (1) makes `epsilon_B->0`, and Markov's inequality shows
that only `o(L)` shifts have `D_T>epsilon_B*c^2`. Removing them preserves a
positive fraction of the dense shifts.

For every good shift, at most `D_T/2=o(c^2)` cells have multiplicity at least
two. Therefore

```text
Theta(c^2) cells of S_T are exactly one,            (3)
I_T>=Omega_T=Theta(c^2).                            (4)
```

Equation (4) recovers R25's popular-difference scale. Equation (3) is the new
fiber information: on a constant-density part of `C x C`, one point of the
fiber `Z_a` moves by `T` to one point of `Z_b`, and almost no second point
does so.

Equivalently, the original fiber partition and the translated fiber
partition have `Theta(c^2)` singleton block intersections. R29 calls this a
partial orthogonal fiber grid.

## Theorem 2: exact full-grid coverage is impossible

Consider the strongest boundary case

```text
r=c,
z=c^2,
s_T(a,b)=1 for every (a,b) in C^2                 (5)
```

for one nonzero `T`. Summing (5) gives

```text
I_T=c^2=z,
```

so every point of `Z` remains in `Z` after translation by `T`:

```text
Z+T=Z.                                             (6)
```

Because `G'` has prime order and `T` is nonzero, translation by `T` is one
cycle on all of `G'`. A nonempty invariant subset must therefore be `G'`
itself. But

```text
|Z|=c^2<N.
```

This contradiction proves that an exact full Latin grid cannot occur even
for one admitted shift.

There is no constant quantitative gap in this argument. A scalar interval
of length `c^2` loses only `j` points under translation by `jP`. Thus a
near-full overlap can exist in a known scalar chart. The open branch is a
constant-density partial grid generated by one degree-`c` algebraic pencil,
not the exact boundary (5).

## Theorem 3: finite-list algebraic interpolation costs degree `Omega(c)`

For nonzero `T`, let

```text
Gamma_T={(g(P),g(P+T)):P in E'} subset P^1 x P^1.
```

R26 proves that `Gamma_T` is integral of bidegree `(c,c)` and has
normalization `E'`. Let `R_T` be any projective curve of bidegree `(u,v)`
that does not contain `Gamma_T`. Bezout on `P^1 x P^1` gives

```text
Gamma_T dot R_T=c*(u+v).                           (7)
```

Every distinct selected normalization point captured by `R_T` consumes at
least one unit of this intersection multiplicity. Hence

```text
#{P in Z:g(P+T) in C and (g(P),g(P+T)) in R_T}
 <=c*(u+v).                                        (8)
```

Capturing the `eta*c^2` singleton transitions of one good shift for fixed
`eta>0` therefore requires

```text
u+v>=eta*c.                                        (9)
```

If `R_T` contains `Gamma_T`, its bidegree is at least `(c,c)`, so the same
linear-degree conclusion holds more strongly.

For a single-valued rational router

```text
g(P+T)=rho_T(g(P)), degree(rho_T)=e,
```

the graph has bidegree `(1,e)` up to convention, and (8) gives at most
`c*(e+1)` finite agreements unless the identity is global. Thus

`Theta(c^2)` agreements force `e=Omega(c)`. A global identity is the
single-valued semiconjugacy excluded in R19 for a nonzero prime-group
translation and a separable degree below `N`.

Equation (9) closes bounded-degree finite-list interpolation. It does not
close a degree-`Theta(c)` relation with a sparse circuit, nor a trivial
containment polynomial such as `H_C(Y)`, which accepts every pair with second
coordinate in `C` but does not report exact occupancy, counts, or a source.

## Theorem 4: the affine-plane design is the exact abstract control

Let `c` be a prime power and take the point set

```text
X=F_c^2.
```

For each slope `m in F_c`, partition `X` into the `c` affine lines

```text
y=m*x+b,
```

and add the vertical-line partition. Any block from one partition meets any
block from another in exactly one point. Thus `X` has `c+1` pairwise
orthogonal partitions into `c` blocks of size `c`, attaining precisely the
singleton `c by c` transition-grid scale.

The standard dimension argument shows this is extremal. For each partition,
the real vector space of zero-mean functions constant on its blocks has
dimension `c-1`. Orthogonality of partitions makes these subspaces mutually
orthogonal inside the `(c^2-1)`-dimensional zero-mean function space. Hence
the number `M` of pairwise orthogonal partitions obeys

```text
M*(c-1)<=c^2-1,
M<=c+1.                                             (10)
```

This is a positive control for the combinatorics, not for ECDLP. It uses an
explicit rank-two additive chart and exact block labels. The prime cyclic
group `G'` has no subgroup of order `c^2`, and Theorem 2 rejects exact
translation invariance of a proper `c^2`-point subset. Encoding the affine
plane as an external pair table or known scalar chart restores the missing
representation cost rather than producing a generic-prime operation.

## Consequence for a compact grid compiler

The branch-B survivor is now typed as follows:

```text
Theta(B) public nonzero shifts;
for each, Theta(B^2) almost-singleton intersections between B fibers;
the same prospective degree-B map and B values for every shift;
exact occupancy/count/source access without materializing B^3 incidences;
complete matching of both +T and -T branches into only O(B^2) keys.
```

R15 shows why ordinary one-dimensional convolution does not provide this
interface: the complete branch is a two-translation correlation and exact
source recovery restores the paired incidence problem. R29 adds that a
bounded-degree finite-list relation cannot encode each dense transition
matrix. The remaining possible operation is a compact family of
degree-`Theta(B)` correspondences or an implicit orthogonal-array compiler
whose shared structure supports exact occupancy, multiplicity, one source,
dyadic restrictions, and fresh targets.

Listing `C x C` is only `Theta(B^2)` and is allowed as a superset. It is not
an exact image certificate: false cells have zero multiplicity, and the
complete unordered branch couples two transition matrices at the same
source. The candidate must compute those coupled counts rather than infer
them from one-branch containment.

## Controls and boundaries

1. Transition counts use exact subgroup points and complete projective fiber
   values; ramification is charged by R28 and R27.
2. The good-shift extraction is deterministic and uses no randomness or
   genericity assumption.
3. Theorem 2 closes only exact full coverage. Scalar intervals remain valid
   near-invariance controls.
4. Theorem 3 is a degree/intersection bound, not an arithmetic-circuit or
   sparse-representation lower bound.
5. The affine plane proves that orthogonal partitions at this scale are
   combinatorially possible with an external rank-two chart.
6. Pairwise orthogonality among all translated partitions is not inferred
   from one-branch branch (B); it requires the corresponding difference-shift
   singularity audit.
7. No compact partial-grid design, complete source package, fresh target,
   R10 index, density, rank, factor logs, blind descent, unrestricted lower
   bound, Shoup improvement, or breakthrough is supplied.

## Deduplication

- R15 owns the two-frequency translation-correlation and source interface.
- R25 owns popular point differences and additive energy.
- R26 owns global bidegree `(c,c)` translate diversity.
- R27 owns Bezout saturation and the singularity/distinct-shift dichotomy.
- R28 owns common sub-divisors and residual Abel sums.
- R29 adds good-shift transition matrices, partial orthogonal partitions,
  exact-full-grid impossibility, finite-list interpolation degree, and the
  affine-plane control.

## Scoped disposition

```text
branch-B good shifts: Theta(B)
almost-singleton transition cells per good shift: Theta(B^2)
exact full B by B grid on proper prime-group subset: impossible
finite-list relation capturing constant grid fraction: degree Omega(B)
affine-plane orthogonal design: passes combinatorics with external rank-two chart
compact approximate partial-grid compiler: absent
complete plus/minus coupled image and sources: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Construct or refute the remaining approximate partial-orthogonal design.
Freeze one degree-`B` pencil, `B` values, and `B` prime-group shifts; require
`Theta(B^2)` almost-singleton transition cells for `Theta(B)` shifts and a
compact exact compiler for the coupled `+T/-T` image, counts, and sources.
Reject exact affine-plane or scalar charts, false-positive `C x C` supersets,
degree-`B` relations with dense per-shift coefficient tables, isogeny
stabilizers, or any route missing fresh target, R10, rank, logs, and descent.
