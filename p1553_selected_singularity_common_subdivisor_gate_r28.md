# P1553 selected-singularity common-subdivisor gate R28

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 finite exceptional-value lane;
  no new idea ID.
- Evidence: exact divisor and incidence reductions with a scalar-interval
  positive control; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: R27's selected-delta branch is not itself an
  impossibility certificate. Unless it already contains the dense
  distinct-shift grid, its mass must be carried by growing common
  sub-divisors of a fiber of `g` and a translated fiber. Every such overlap
  has an exact complementary-divisor certificate whose Abel sum is `c*T`.
  A scalar interval realizes the delta scale on one fiber, so delta
  saturation alone is a false positive; the missing object is one
  degree-`c` pencil that simultaneously supplies `Theta(c)` selected fibers,
  `Theta(c^2)` prime-subgroup points, and the complete collision image. No
  such pencil, target interface, or ECDLP breakthrough is supplied.

R27 leaves two mechanisms for an `O(B^2)` complete image. R28 isolates the
first, in which normalization multiplicity consumes `Omega(B^3)` selected
delta defect. The key observation is elementary but useful for the harness:
a multiplicity-`s` cell is not an abstract singularity. It is an actual
degree-`s` common effective sub-divisor of two translated fibers. Removing
that common divisor leaves two degree-`c-s` residual divisors whose group-law
sums differ by exactly `c*T`.

This turns the singularity branch into a prospective algebraic object rather
than a count. It also exposes the correct positive control. A scalar interval
has large overlap with its short translates and can saturate the full delta
scale on one declared fiber. It fails the `Theta(c^2)` support mass and
degree-`c` pencil/decomposition requirements, so the harness must preserve it
as a control rather than report the overlap count as evidence of compression.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R16 common-right-factor compiler gate | `9f7ec4ecb6821e30affce39b25b459891f6be0b945b6d64332971dd86df2237c` |
| R21 arbitrary-monodromy factor gate | `ad8d9e83b919ea92c98eefcb657831f01378b6760d49b2aaea0a892d6ed52be1` |
| R25 popular-difference complete-triple gate | `28b24d76a8fbecf50b755c7f808283664bfd52ba50395cc097d43a191ef7f70a` |
| R27 translate-curve Bezout-saturation gate | `dee8a0913854e0ce3e717ecdbc3e8aea73b987b2b2fb71790a470f501dd0c2d9` |
| R27 parent report | `0106c779c665550a24ec562404d7b8ff7c4b832048b31c1ec4ec96d31105e6e3` |
| R27 bundle hash list | `38503161972ea8ec40575a4def14fd6619b883772f8d6c9e249d7573ab5d5ba6` |
| R27 staging receipt | `5c56925dde1d0a39409605e8a54058392e127f6a2380feda10a99bc62714f3ef` |

## Frozen quotient model

Use R27's algebraically closed field of characteristic greater than `c` and
greater than three. Let

```text
g:E'->P^1
```

be separable of degree `c<N`, with trivial translation stabilizer, and let
`G'` be the prime-order subgroup of order `N`. Fix prospectively

```text
C subset P^1, |C|=r,
Z={P in G':g(P) in C}, |Z|=z,
Q subset G' minus {O}, |Q|=L.
```

The balanced mass-admissible regime is

```text
c=Theta(r)=Theta(L)=Theta(B),
z=Theta(c*r)=Theta(B^2),
N=Theta(B^5).
```

For a projective value `a`, write `F_a=g^*(a)` for its degree-`c`
effective fiber divisor, including ramification multiplicity. For
`T in Q` and `(a,b) in C^2`, define

```text
H_(T,a,b)=gcd(F_a,t_(-T)(F_b)),
u_T(a,b)=degree(H_(T,a,b)),
s_T(a,b)=#{P in G':g(P)=a and g(P+T)=b}.
```

The selected count is geometric and unweighted, so

```text
s_T(a,b)<=u_T(a,b).
```

Repeated or ramified points only increase `u`; they are never deleted from
the complete projective interface.

As in R27, put

```text
D=sum_(T,a,b) s_T(a,b)*(s_T(a,b)-1),
Omega=#{(T,a,b):s_T(a,b)>0}.
```

R27 branch (A) is `D=Omega(L*c^2)`.

## Theorem 1: no translated pair of full fibers

For every nonzero `T in G'` and all `a,b`,

```text
u_T(a,b)<=c-1.                                      (1)
```

Indeed, equality `u_T(a,b)=c` would give the divisor identity

```text
F_a=t_(-T)(F_b).
```

All fibers of `g` are linearly equivalent and have the same Abel sum in
`Pic^c(E')`. Translation by `-T` subtracts `c*T` from that sum. The displayed
identity would therefore imply

```text
c*T=O.
```

This is impossible because `T` has prime order `N` and `0<c<N`. Thus even a
singularity-saturating cell has a nonempty residual divisor. This is the
finite-fiber form of the pole-divisor obstruction used in R26; it does not
bound a near-full overlap away from `c-1`.

## Theorem 2: sparse selected cells force growing common sub-divisors

Let

```text
M=max_(T,a,b) s_T(a,b).
```

Every occupied cell contributes at most `M*(M-1)` to `D`, hence

```text
D<=M*(M-1)*Omega.                                  (2)
```

Consequently, if

```text
D>=delta*L*c^2
```

for fixed `delta>0`, then either

```text
Omega=Omega(L*c^2),                                (3)
```

which is already the dense distinct-shift incidence scale, or any family
with `Omega=o(L*c^2)` must have

```text
M>=sqrt(delta*L*c^2/Omega)->infinity.              (4)
```

The mass can be localized without choosing a single maximum. Partition the
occupied cells into dyadic layers

```text
2^j<=s_T(a,b)<2^(j+1),  1<=j<=ceil(log_2 c).
```

For some layer with `h=2^j`, its contribution `D_h` obeys

```text
D_h>=D/ceil(log_2 c),                              (5)
```

and the number `n_h` of cells in that layer obeys

```text
n_h>=D_h/(4*h^2)
   >=D/(4*h^2*ceil(log_2 c)).                      (6)
```

Every one of those cells supplies an explicit common sub-divisor
`H_(T,a,b)` of degree at least `h`. In the balanced branch-A regime,

```text
n_h=Omega(B^3/(h^2*log B)).                        (7)
```

Equations (2)-(7) are a classification, not a circuit lower bound. A compact
formula may represent many cells at once. An explicit cell table, however,
is already over the `B^(9/4)` setup cap unless its typical multiplicity is at
least `B^(3/8-o(1))`; such a claim must expose the formula rather than hide
the cells in advice.

## Theorem 3: exact residual-divisor certificate

For one occupied cell, remove its full geometric common divisor:

```text
F_a=H_(T,a,b)+A_(T,a,b),
t_(-T)(F_b)=H_(T,a,b)+B_(T,a,b),
degree(A_(T,a,b))=degree(B_(T,a,b))=c-u_T(a,b).
```

Taking Abel sums and cancelling the common divisor gives the exact identity

```text
sum(A_(T,a,b))-sum(B_(T,a,b))=c*T in E'.          (8)
```

Thus a multiplicity-`u` overlap is equivalently a pair of complementary
degree-`c-u` divisors representing the known class `c*T`. The identity is
projective, includes ramification, and is directly checkable from the
factorization of the two fiber divisors.

For a near-full overlap `u=c-k`, equation (8) is a `k`-point minus `k`-point
representation of `c*T`. For intermediate `u`, it remains a large common
sub-divisor constraint rather than a short relation. Since `c` is invertible
modulo `N`, the right side does not vanish and cannot be absorbed by a
prime-subgroup translation stabilizer.

Equation (8) is not a DLP relation: `T` and `c` are already public, residual
points may lie outside `G'`, and no factor-base logs follow from it. Its role
is to make every claimed singularity family auditable and to prevent a
singularity count with no normalization sources.

## Theorem 4: support mass also requires one decomposed pencil

Choose homogeneous linear forms for the values in `C` and let

```text
H_C(Y)=product_(a in C) ell_a(Y).
```

Then the selected fibers are not independent divisors. They are exactly the
zero divisor of the decomposed function

```text
w=H_C composed with g,
degree(w)=c*r,
(w)_0=sum_(a in C) F_a.                            (9)
```

In the full unramified mass endpoint `z=c*r`, all points of every selected
geometric fiber lie in `G'`, and `Z` is the complete zero divisor in (9). For
`z>=kappa*c*r`, the selected prime-subgroup points still form a
`kappa`-dense sub-divisor of this one decomposed pencil.

Therefore it is insufficient to prescribe `r` unrelated degree-`c` blocks
with convenient translate overlaps. Two fibers determine the pencil; all
remaining selected fibers must arise from the same map `g`, and their union
must retain the degree-`r` left factor in (9). An arbitrary degree-`c*r`
function with a progression-like zero divisor is only a positive divisor
control. It does not supply the required decomposition `H_C composed with g`.

This decomposition condition has the same information-flow fingerprint as
the earlier common-factor screens, but R16-R21 do not by themselves close
it: those gates concern a common right factor of complete fifth-branch maps.
R28 records the new divisor-level obligation without relabelling the old
theorem as an unrestricted impossibility result.

## Positive scalar-interval control

Work only in the hidden scalar coordinate of `G'=<P>`. Let

```text
F={R+i*P:0<=i<c}
```

and take `T=j*P` for `1<=j<=L<=c/2`. If `F` is declared to be one value
fiber, then

```text
|F intersect (F-T)|=c-j
```

and the single cell `(T,a,a)` contributes

```text
(c-j)*(c-j-1)
```

to the selected collision defect. Hence for `L=Theta(c)`,

```text
sum_(j=1)^L (c-j)*(c-j-1)=Theta(L*c^2)=Theta(B^3). (10)
```

This exactly passes the branch-A delta scale and realizes the residual
identity (8) with boundary divisors of degree `j`. It fails as an ECDLP
constructor for four separate reasons:

1. one fiber supplies only `c=Theta(B)` points, not the required
   `c*r=Theta(B^2)` support;
2. the interval is defined in the known scalar chart;
3. it does not produce `r=Theta(c)` fibers of one degree-`c` pencil or the
   decomposition (9); and
4. it supplies neither the complete `O(B^2)` triple image nor a compact
   fresh-target/source interface.

The control proves that selected delta saturation cannot be promoted on its
own. The missing theorem must use the simultaneous pencil, mass, and complete
branch conditions.

## Harness obligations for a branch-A candidate

A passing candidate must provide all of the following as one prospective
package:

1. a separable degree-`Theta(B)` map `g` with trivial translation stabilizer;
2. fixed `C,Q`, each of size `Theta(B)`, and `Theta(B^2)` exact subgroup
   points above `C`;
3. the actual common sub-divisors `H_(T,a,b)` or a compact identity generating
   them, with residual-divisor checks (8);
4. a proof that the selected defects total `Omega(B^3)` without an explicit
   `B^3` advice table;
5. only `O(B^2)` complete unordered triples, exact integer multiplicities,
   and one jointly coupled source per key; and
6. fresh-target action, R10 queried coefficients, independent rank, factor
   logs, and scalar-blind descent under the full cost cap.

Failure of any item leaves a divisor design or control, not an ECDLP
algorithm.

## Controls and boundaries

1. Fiber and gcd divisors retain ramification and projective infinity strata.
2. Equation (1) uses `c<N` and a prime-order nonzero shift; it is not asserted
   for composite source torsion or degree divisible by the source order.
3. The dyadic reduction permits compact algebraic generation; it does not
   infer a universal table-size lower bound.
4. The scalar interval is a positive control for delta saturation and a
   negative control for the public DLP-free interface.
5. R28 does not prove that a decomposed pencil with the required overlaps is
   impossible. It types the exact object that remains.
6. No run, fresh target, R10 index, density, rank, factor logs, blind descent,
   unrestricted lower bound, Shoup improvement, or breakthrough is claimed.

## Deduplication

- R25 owns point-set popular differences and additive energy.
- R26 owns maximal-isogeny translate diversity.
- R27 owns the total delta/Bezout budget and the two-branch saturation
  dichotomy.
- R16-R21 own common right factors of complete branch maps.
- R28 adds the common-subdivisor histogram, the exact residual Abel-sum
  certificate, the one-pencil decomposition obligation, and the
  scalar-interval delta control.

## Scoped disposition

```text
selected delta saturation alone: insufficient
full translated fibers: impossible for 0<c<N
sparse occupied-cell saturation: forces growing common sub-divisors
every common sub-divisor: exact residual class c*T
scalar interval: passes delta scale but fails mass, pencil, and interface
degree-c pencil with c selected fibers and c^2 subgroup points: absent
complete O(B^2) image and source compiler: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Classify the residual-divisor design in (8)-(9): either construct one
degree-`B` pencil with `B` prospective fibers, `B^2` prime-subgroup points,
and `B` public shifts whose common-subdivisor histogram supplies the selected
`B^3` defect, or prove that any such simultaneous pencil reduces to a scalar
chart, an isogeny stabilizer, an explicit super-cap table, or the dense
distinct-shift grid. Only then test complete triples, fresh target, R10,
rank, logs, and descent.
