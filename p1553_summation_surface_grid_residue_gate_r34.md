# P1553 summation-surface grid-residue gate R34

## Classification

- Owner: existing P1553/IDEA-195 summation-polynomial, erased-image,
  finite-field-embedding, and low-moduli joint-spectrum lane; no new idea ID.
- Evidence: exact addition-surface degree theorem, represented-grid audit, and
  deterministic affine-resultant toy control; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `toy-control`, `model-bound`,
  `novelty-unverified`, `independent-review-required`.
- Cryptanalytic result: the actual elliptic labels do admit a compact
  summation-polynomial geometry. The ordered complete triples lie on an
  irreducible addition surface whose three coordinate degrees are at most
  `4c^2,c^2,c^2`; the `L=c` fixed-shift images form a reduced union of
  genus-one curves of total Segre degree at most `3c^2`. This avoids treating
  the labels as arbitrary matrices. It does not yet avoid the `c^3` source
  divisor: reducing the surface or curve equations on the explicit
  `c by c by c` value grid materializes a split algebra of dimension `c^3`,
  and a graph-first norm has degree `c^3`. A radical-first selected-shift
  curve compiler remains open. No target interface, Shoup-bound improvement,
  or ECDLP breakthrough is supplied.

R33 left the exact joint spectrum of multiplication by `g` and its partial
translation conjugates. R34 tests the most direct summation-polynomial and
FFE reformulation. Instead of `c` unrelated spectra, extend the shift to a
second elliptic variable. The resulting two-dimensional addition surface is
genuinely low degree. Restricting that surface to the selected shifts gives a
one-dimensional curve union whose represented degree already matches the
desired `O(c^2)` image scale.

That is a useful positive geometry, but the order of operations is decisive.
Evaluating its equations on the complete selected value grid, or taking a
norm over all selected sources before radicalization, restores `c^3` words.
The surviving operation must build and intersect the reduced selected-shift
curve union before that source multiplicity is represented.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R13 erased-image geometric-resolution gate | `78ff09226852f140a8c33ce5ba98106ed62007b7126673d3f3af2c49d08a2cc5` |
| R14 tensor trace/minimal-polynomial gate | `da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac` |
| R31 near-period run compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R33 low-moduli joint-spectrum gate | `ab15335a61790398bace9b8420165480017ae58cc85c60bf048b3eccdd6ed601` |
| R33 bundle hash list | `57bdcc05ae9594478b43d173908670cea07527cd0f14687aee4c058894849198` |
| R33 staging receipt | `309decd21ca90b440ecf58f5fbd7fddbc4dba9d22e94ec7fa73d7bedb7f3d629` |

## Frozen model

Work over `k=F_p`, with characteristic different from two and greater than
the degrees below. After the R26 maximal translation-stabilizer quotient,
freeze a separable map

```text
g:E->P^1, degree(g)=c<N,
C subset P^1(k), |C|=c,
Q subset G minus {O}, |Q|=L=c,
Z={P in G:g(P) in C}, |Z|=c^2.
```

For an ordered source `(P,q)`, put

```text
a=g(P), b=g(P+q), d=g(P-q).                       (1)
```

The actual complete key replaces `(b,d)` by the unordered projective divisor
`{b,d}`. Passing from the ordered cover to that quotient changes only a
constant degree and retains separate diagonal, pole, tangent, and infinity
strata.

## Theorem 1: the repeated-addition surface has quadratic degree

Use independent points `(P,B)` on `E x E`, set

```text
D=2P-B, q=B-P,
Theta(P,B)=(g(P),g(B),g(D)).                       (2)
```

The first two coordinates are `g x g`, so the image closure `S` has dimension
two. Since `(P^1)^3` has dimension three, `S` is an irreducible hypersurface,
defined by one irreducible multihomogeneous addition-surface polynomial

```text
F_g(a,b,d)=0.                                     (3)
```

The coordinate degrees have the exact generic-fiber upper bounds

```text
degree_d(F_g)<=c^2,
degree_b(F_g)<=c^2,
degree_a(F_g)<=4c^2.                              (4)
```

For fixed generic `(a,b)`, there are `c^2` pairs `(P,B)` and each determines
`D`, proving the first bound. The map

```text
(P,B)->(P,D=2P-B)
```

is an automorphism of `E x E`, proving the second. The map

```text
(P,B)->(B,D=2P-B)
```

is the isogeny represented by the integer matrix

```text
[[0,1],[2,-1]],
```

whose determinant is `-2` and whose degree is four. Composing with `g x g`
gives at most `4c^2` preimages for fixed `(b,d)`, proving the third bound.
Generic degree of `Theta` onto `S` can only divide these counts, so (4)
remains valid when fibers collide.

Equation (3) is the relevant generalized summation-polynomial object for
`B+D=2P`. The usual degree-`c` four-point summation polynomial has a coarser
`c^3` per-variable upper bound. The repeated point and the group-law surface
sharpen it to (4). A factor or degree drop caused by evenness, isogenies, or
special automorphisms is retained and charged; it is not assumed generic.

## Theorem 2: selected shifts form a degree-`O(c^2)` curve union

For fixed nonzero `q`, define

```text
Theta_q(P)=(g(P),g(P+q),g(P-q)),
Y_q=closure(Theta_q(E)) subset (P^1)^3.            (5)
```

R26 proves that `(g(P),g(P+q))` is birational onto its image after the
maximal stabilizer quotient. Therefore `Theta_q` is birational too. The
normalization of `Y_q` is `E`, every coordinate projection has degree `c`,
and the Segre degree is

```text
degree_Segre(Y_q)=3c.                              (6)
```

Let

```text
Y_Q=union_(q in Q) Y_q                             (7)
```

with repeated irreducible components merged. Then

```text
degree_Segre(Y_Q)<=3Lc=3c^2.                      (8)
```

Thus the selected-shift image has a reduced algebraic carrier at exactly the
desired quadratic degree scale. This is the concrete low-moduli escape from
R32: the actual input is a degree-`c^2` union of translated elliptic curves,
not an arbitrary aligned-run label tensor.

The source multiplicity has not disappeared. In the favorable R30-R31
boundary, each `q` has `Theta(c^2)` complete points and

```text
sum_(q in Q) |Z intersect (Z-q) intersect (Z+q)|=Theta(c^3).  (9)
```

The reduced union (7) can have degree `O(c^2)` while the selected source
divisor mapping to it has degree `Theta(c^3)`. Exact multiplicities and one
source are additional data, not consequences of (8).

## Theorem 3: the explicit selected grid restores cubic traffic

On one affine chart put

```text
H_C(T)=product_(u in C)(T-u).
```

For distinct selected values, the complete grid algebra is

```text
R_C=k[A,B,D]/(H_C(A),H_C(B),H_C(D)),
dim_k(R_C)=c^3.                                    (10)
```

Reducing `F_g`, an ideal presentation of `Y_Q`, or a selected-shift Chow form
on the full grid and then storing its coefficient or evaluation residue
materializes an element or module over this `c^3`-dimensional split algebra.
Likewise, a norm or characteristic polynomial formed first over the complete
source divisor in (9) has degree `Theta(c^3)`; taking its radical afterwards
does not refund that represented work.

The degree bounds (4) do not give a grid root bound because every coordinate
degree is already at least `c`. Modulo the grid ideal, every function on
`C^3` has a unique representative of individual degree below `c`, and that
residue has `c^3` possible coefficients. A literal multivariate multipoint
evaluation route also starts from the represented coefficient vector and
the represented evaluation points. It therefore does not supply the missing
radical-first compiler merely because fast evaluation is nearly linear in
its input and output sizes.

This is a representation gate, not an arithmetic-circuit lower bound. The
low-moduli equations may admit a compact quotient, gcd, subresultant, or
common-base-locus circuit that never writes the residue (10). R34 preserves
that route explicitly.

## Theorem 4: the addition surface alone omits the shift selector

Equation (3) records that there exists some `q=B-P` giving the triple. It does
not enforce `q in Q`. On the two-dimensional surface, the selected-shift
condition is a divisor whose reduced image is the curve union (7). A single
surface equation evaluated on `C^3` therefore admits triples coming from
unselected shifts and cannot certify one selected fifth occurrence.

A valid summation-polynomial compiler must include one of:

1. a saturated ideal or Chow presentation of `Y_Q` derived from the selected
   shift divisor;
2. a rational inverse on `S` that recovers `q` and tests the selected-shift
   divisor without source expansion; or
3. an equivalent marked resultant that returns a jointly coupled `(P,q)`.

An existential value-only summation polynomial is insufficient for the R13
source section, R31 restrictions, or target replay.

## Theorem 5: affine recursive resultants need projective strata

For the short Weierstrass control

```text
E:y^2=x^3+A*x+B,
```

let `S_3` be the ordinary affine third summation polynomial and form

```text
R(a,b,d)=Res_X(S_3(b,d,X),S_3(a,a,X)).             (11)
```

The desired regular branch has `B+D=2P`; the two pair sums in (11) then have
the same finite `x`-coordinate, so (11) vanishes. But a valid four-point sum
can also split into proper zero subsums. For example

```text
R_1+(-R_1)+R_2+(-R_2)=O.                          (12)
```

Both intermediate pair sums in (12) are the point at infinity, absent from
the affine elimination variable `X`. Consequently the raw affine recursive
resultant need not vanish on this valid projective stratum.

The deterministic R34 control over

```text
F_101, E:y^2=x^3+2x+3
```

checks 8,742 regular desired `(P,q)` branches with no failure. The resultant
has degrees `(8,2,2)` in `(a,b,d)`, total degree ten, and 82 symbolic terms.
It then gives the nonzero value `96 mod 101` on the valid cancellation source

```text
(1,39)+(1,-39)+(3,6)+(3,-6)=O,
arguments (1,1,3,3).                              (13)
```

This is a toy interface control, not a refutation of complete projective
summation polynomials. It proves that an implementation using recursive
affine resultants must homogenize and retain the infinity/proper-subsum
components, then saturate only with a proved source-preserving rule. Deleting
them as ``degenerate'' would lose valid relations; retaining them without
tags can create false multiplicities.

## Radical-first compiler target

The surviving operation is now narrower than R33's abstract joint spectrum:

```text
derive a saturated reduced presentation of Y_Q from the O(c)-parameter map g
and selected shift divisor Q; intersect Y_Q with the selected projective
value grid before representing the c^3 source divisor; return the O(c^2)
exact complete keys, integer multiplicities, and one coupled source in
c^(9/4+o(1)) setup work and state.
```

A promising subroute is a common-base-locus or pairwise-curve-intersection
compiler. If the `c^3` complete occurrences occupy only `O(c^2)` keys, the
fixed-shift curves have large aggregate intersection multiplicity. Any use of
that fact must prove how a cap-sized prospective family of shift pairs covers
the accepted keys and must charge missing low-multiplicity keys rather than
assuming every key lies on every component.

After construction, the package still needs compressed containment, exact
adaptive child counts, fresh-target action, R10's rank-two coefficients,
relation density, independent signed rank, factor-base logs, and identical
scalar-blind descent.

## Controls and boundaries

1. All degree statements are over the algebraic closure and retain
   ramification. Rational-point and prime-subgroup restrictions are separate.
2. The unordered branch divisor is a constant-degree quotient of the ordered
   triple carrier; diagonal and fixed-point strata remain explicit.
3. `Y_Q` degree `O(c^2)` is a representation opportunity, not a constructor.
4. The grid-residue charge applies only to algorithms that materialize the
   quotient (10) or the degree-`c^3` source norm.
5. The affine toy checks one ordinary `x`-coordinate control and is not
   asymptotic evidence.
6. No selected-shift ideal, radical-first intersection, multiplicity
   compiler, target, R10 index, density, rank, logs, descent, unrestricted
   lower bound, Shoup improvement, or breakthrough is supplied.

## Deduplication

- Classical and generalized summation polynomials are prior art; R34 claims
  only the campaign-specific repeated-addition degree and interface audit.
- R13 owns the erased-image representation and graph-first source-degree
  gate.
- R14 owns trace/Hankel decoding once exact moments and containment exist.
- R26-R30 own translate-curve geometry and saturation.
- R31 owns DLP-free path/run construction.
- R33 owns the low-moduli joint-spectrum formulation.
- R34 adds the addition surface, selected-shift curve union, grid residue,
  and projective affine-resultant control.

## Scoped disposition

```text
addition surface: irreducible hypersurface in (P^1)^3
coordinate degrees: at most (4B^2,B^2,B^2)
B selected shift curves: reduced Segre degree at most 3B^2
explicit selected grid algebra: dimension B^3
graph-first selected source norm: degree B^3
raw affine recursive resultant: incomplete on proper-subsum infinity strata
radical-first selected-shift curve compiler: absent
complete target and R10 path: absent
```

## Primary-source boundary

Faugere, Huot, Joux, Renault, and Vitse define summation polynomials for a
general nonconstant morphism `E->P^1`, prove the irreducible value relation,
and describe resultant construction and degree growth:
[Symmetrized Summation Polynomials](https://www.iacr.org/archive/eurocrypt2014/84410158/84410158.pdf).
Semaev's original construction is
[Summation Polynomials and the Discrete Logarithm Problem on Elliptic Curves](https://eprint.iacr.org/2004/031.pdf).
The represented coefficient-and-point boundary for fast evaluation follows
[Fast Multivariate Multipoint Evaluation Over All Finite Fields](https://arxiv.org/abs/2205.00342).
These sources do not supply the R34 radical-first selected-shift compiler.

## Exactly one next action

Build or refute one saturated radical-first presentation of `Y_Q`: derive the
selected-shift divisor on the addition surface, preserve all projective and
proper-subsum strata, and test whether a prospective common-base-locus or
pairwise-curve-intersection schedule emits exact selected-grid keys, counts,
and coupled sources below `B^(9/4+o(1))` without materializing the `B^3` grid
residue or source norm. Then bind any survivor to restrictions, fresh target,
R10, rank, logs, and descent.
