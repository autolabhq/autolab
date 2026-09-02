# P1553 low-moduli joint-spectrum gate R33

## Classification

- Owner: existing P1553/IDEA-195 elliptic translation-consistent histogram,
  finite-field algebra, and source-router lane; no new idea ID.
- Evidence: exact parameter/entropy bound and split-algebra reformulation; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: R32's arbitrary matrix encoding cannot be imported
  canonically into the actual degree-`B` elliptic family. On a fixed elliptic
  curve, a degree-`c` map, `c` selected values, and `c` shifts have only
  `O(c)` field parameters and at most `p^(O(c))` instances, whereas two
  arbitrary `c by c` Boolean matrices have `2^(Theta(c^2))` instances. For
  `p=c^(5+o(1))`, no injective encoding exists without `Omega(c^2)` extra
  advice bits. Arbitrary ordering of the `c^2` selected points is large
  enough to hide that advice and must be canonicalized. After canonicalization,
  the complete histogram is the joint spectral measure of one multiplication
  operator and its `c` run-compressed partial translation conjugates in a
  split finite algebra of dimension `c^2`. No sub-`c^(9/4)` joint-spectrum
  algorithm, target interface, or ECDLP breakthrough is supplied.

R32 shows that a generic aligned-run color histogram at the campaign cap
would improve dense matrix multiplication. The actual P1553 labels are not
arbitrary strings. They are fibers of one separable degree-`c` map `g`, and
the run mappings are restrictions of commuting elliptic translations. R33
quantifies this escape and rewrites it in the finite-field-embedding language
needed for a constructive attack.

The low parameter count is not an algorithm. A low-dimensional algebraic
family can still have high elimination degree and difficult arithmetic. Its
purpose is to reject the generic matrix instance as an automatic lower bound
and to forbid hidden `c^2`-bit advice in list order, occurrence labels, or
backpointers.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R15 translation-correlation gate | `ea007be0b237b91127e6a42501873f914ad31d6d4fd4393c3ae6b85005893b9b` |
| R28 common-subdivisor gate | `c24183bfe4823a9318ae06769763c94cfed99cb4291807f06b85f397fd53f662` |
| R31 near-period run compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R32 aligned-run matrix-multiplication gate | `a9acc39df3f504685671ab7d0c16280c9826b6fb1250ffc2fd715ebecc93a02a` |
| R32 parent report | `88f23451bfb21c0fc0f0d114fdfa26811a29f1aab7d2477e9a08530d00859012` |
| R32 bundle hash list | `04ea33c41dec39f64514b8cba91368c361145d78b8c5d0452fdec22ed1b1e327` |
| R32 staging receipt | `1914f64aeadab9b1585bf4e03e99615e557fcf3fc01b2ef44be4b45322d42b6b` |

## Frozen actual-family model

Fix an elliptic curve `E/F_p` and a prime-order subgroup `G` of order

```text
N=p^(1+o(1)),
c=N^(1/5)=p^(1/5+o(1)).
```

After the R26 maximal-isogeny quotient, freeze:

```text
g:E->P^1, separable degree c,
C subset P^1(F_p), |C|=c,
Q subset G/{plus or minus 1}, |Q|=c,
Z={P in G:g(P) in C}, |Z|=c^2.
```

The full-size equalities are the favorable boundary. Partial fibers and
constant factors do not increase the parameter exponents.

The input representation of `g`, `C`, and `Q` is prospective and canonical.
The explicit selected points, fibers, pair backpointers, R31 paths, and run
records must be derived deterministically from those inputs and the frozen
factor decks. Their order is not a free advice channel.

## Theorem 1: degree-`c` maps have only `O(c)` moduli

For one degree-`c` line bundle `L` on `E`, Riemann-Roch gives

```text
dim H^0(E,L)=c.
```

A morphism `g:E->P^1` with `g^*O(1)=L` is represented by a basepoint-free
ordered pair of sections `(s_0,s_1)`, modulo one common nonzero scalar. Thus
the fixed-`L` parameter space has dimension at most

```text
2c-1.
```

The line bundle varies in `Pic^c(E)`, of dimension one. Therefore

```text
dim Mor_c(E,P^1)<=2c.                              (1)
```

The open conditions of exact degree, separability, and no common section
zero can only reduce or preserve this bound. Adding an unordered `c`-tuple
of selected values and a `c`-tuple of shifts gives total geometric parameter
dimension at most

```text
2c+c+c=4c.                                         (2)
```

Origin-fixing automorphisms, target Mobius changes, sign quotienting, and
the finite prime-subgroup restrictions alter constants or pass to subloci;
they do not create `Theta(c^2)` parameters.

## Theorem 2: finite-field entropy excludes arbitrary matrix advice

The same conclusion can be counted directly over `F_p`. There are at most
`p^(2c+O(1))` ordered section pairs and degree-`c` line-bundle choices, at
most `p^c` selected value tuples, and at most `N^c=p^(c+o(c))` shift tuples.
Consequently the number of represented actual-family instances is at most

```text
p^(4c+o(c))=2^(O(c*log p))=2^(O(c*log c)).         (3)
```

Two arbitrary Boolean matrices contain

```text
2*c^2 bits
```

and have `2^(2c^2)` choices. Since `c*log c=o(c^2)`, no injective encoding
of all R32 matrix instances into canonical `(g,C,Q)` data exists for growing
`c`.

This does not refute a reduction from a structured matrix subclass, nor does
it prove a fast histogram algorithm. It proves that the arbitrary R32 family
requires an additional advice channel of size

```text
Omega(c^2) bits.                                   (4)
```

## Canonicalization guard

The explicit list `Z` has `c^2` points. Its arbitrary ordering has

```text
log_2((c^2)!)=Theta(c^2*log c)
```

bits, more than enough to hide the matrices forbidden by (3)-(4). The same
problem occurs in arbitrary occurrence order, fiber order, path identifiers,
run order, or backpointer choice.

A valid harness must therefore derive and verify canonical forms:

1. projective point encodings use one fixed complete chart and byte order;
2. values in `C`, points in every fiber, signed shifts, and selected points
   use deterministic sorted encodings;
3. path starts use sorted point encodings and offsets increase by the frozen
   `q_0` step;
4. run records sort by signed shift, source start, and translated path data;
5. one source backpointer is the lexicographically first valid occurrence;
6. no permutation, duplicate occurrence, or unused metadata field is read by
   the compiler unless its derivation and entropy are charged.

Canonicalization does not make the algorithm fast. It keeps the actual
low-moduli exception honest.

## Theorem 3: split finite-algebra joint-spectrum formulation

Let

```text
H_C(Y)=product_(a in C) (Y-a),
w=H_C composed with g.
```

At the reduced full-fiber boundary, the selected divisor is

```text
D=(w)_0=sum_(P in Z) [P], degree(D)=c^2.           (5)
```

Its coordinate algebra is split finite etale:

```text
A=H^0(D,O_D)=product_(P in Z) F_p,
dim_(F_p)(A)=c^2.                                  (6)
```

Write `e_P` for its primitive idempotents. Multiplication by the restriction
of `g` is the diagonal operator

```text
M_g(e_P)=g(P)*e_P.                                 (7)
```

For a signed shift `q`, let

```text
U_q:e_P -> e_(P+q)
```

on the domain where both points lie in `Z`, and zero elsewhere. This is a
partial permutation operator. R31 represents all `U_(plus or minus q)` by
`O(c^2)` aligned run records in total, rather than by `Theta(c^3)` nonzero
matrix entries.

On the complete source projector

```text
Pi_q=U_q^*U_q*U_(-q)^*U_(-q),
```

the three commuting diagonal observables are

```text
M_0=M_g,
M_(+,q)=U_q^* M_g U_q,
M_(-,q)=U_(-q)^* M_g U_(-q).                       (8)
```

For a projective key `(a,{b,d})`, its exact multiplicity is the trace of the
joint spectral projector of (8) on `Pi_q A`, summed over `q`. One primitive
idempotent in that projector is an exact coupled source.

Thus the R31 residual is equivalently:

```text
compute the joint spectral measure of one degree-c multiplication operator
under c run-compressed partial translation conjugates in a split algebra of
dimension c^2.                                     (9)
```

Equation (9) is an FFE-style target: it retains the finite algebra, exact
idempotents, translations, multiplicities, and source recovery. It does not
replace them with field traces that lose joint support.

## Standard representation costs

In the primitive-idempotent basis, applying all observables in (8) visits
every complete source and costs `Theta(c^3)`. Writing the partial
permutations explicitly also costs `Theta(c^3)` entries; R31 is the exact
improvement for that representation.

In a dense polynomial or Riemann-Roch basis, one multiplication operator has
dimension `c^2`; dense matrices are much worse. A characteristic polynomial
of one scalar linear combination of (8) may encode one shift's keys, but `c`
separate degree-`c^2` polynomials restore `c^3` coefficients, and random
linear combinations do not by themselves preserve unordered projective
charts, integer counts, or one source.

A product, norm, or resultant over all source/shift pairs has represented
degree `c^3`. Taking a radical uses the promised `O(c^2)` distinct keys only
after the degree-`c^3` object has been constructed unless a new output-
sensitive joint-spectrum operation is supplied. This is the same
representation boundary as R13-R15, now with R31's partial permutations
made explicit.

## Surviving low-moduli operation

The matrix lower-bound control and the low-moduli escape together leave one
precise positive target:

```text
a batched joint-minimal-polynomial or joint-spectral algorithm that consumes
the O(c)-parameter map g, the decomposed divisor H_C composed with g, and the
O(c^2) R31 run package, and emits the O(c^2) exact complete keys, integer
counts, and one primitive-idempotent source in c^(9/4+o(1)) work and state.
```

It must exploit a theorem specific to multiplication by a degree-`c`
elliptic function and commuting translations. Treating `M_g` as an arbitrary
diagonal matrix returns to R32 and would imply `omega<=2.25`.

## Controls and boundaries

1. The moduli bound is for a fixed elliptic curve; allowing the curve to vary
   adds only constant dimension.
2. Counting uses prospective algebraic inputs, not arbitrary explicit point
   order or advice.
3. Low entropy does not imply low arithmetic or elimination complexity.
4. The split-algebra formulation assumes the reduced full-fiber boundary;
   ramified and partial fibers require primary components and selected
   projectors, not deletion.
5. The partial translation operators are exact only on represented domains;
   they are not algebra automorphisms of all `A`.
6. No joint-spectrum algorithm, adaptive restriction package, fresh target,
   R10 index, density, rank, factor logs, blind descent, unrestricted lower
   bound, Shoup improvement, or breakthrough is supplied.

## Deduplication

- R10 owns factorized pullbacks and exact queried coefficients.
- R13-R15 own represented norms, moments, and translation correlations.
- R28 owns common sub-divisors and residual Abel sums.
- R31 owns run-compressed partial translations.
- R32 owns generic matrix-multiplication hardness.
- R33 adds the actual-family moduli/entropy bound, canonicalization guard,
  and finite-algebra joint-spectrum target.

## Scoped disposition

```text
actual family algebraic parameters: O(B)
actual family entropy over p=B^(5+o(1)): O(B log B) bits
arbitrary matrix entropy: Theta(B^2) bits
hidden list-order advice: forbidden by canonicalization
split finite algebra dimension: B^2
all partial translations in run form: O(B^2) records
standard joint-spectrum traversal: B^3
output-sensitive elliptic joint-spectrum compiler: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Construct or refute the low-moduli joint-spectrum operation in (9). Work
directly with the decomposed divisor `H_C composed with g`, R28 residual
divisors, and R31 partial translations; emit exact complete keys, counts, and
one canonical source below `B^(9/4+o(1))`. Then require path-compatible
restrictions, fresh target, R10, rank, logs, and descent. Reject arbitrary
diagonal-label advice, noncanonical point order, degree-`B^3` norms, or field
traces that lose joint sources.
