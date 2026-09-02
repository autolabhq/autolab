# P1553 block common-right-factor compiler gate R16

## Classification

- Owner: existing P1553/P1515/IDEA-057/IDEA-195 non-Cartesian Kummer
  intertwiner and source-router frontier; no P1554.
- Evidence: exact function-field reduction and conditional compiler theorem;
  no run.
- Status: `DRAFT_REVIEW_REQUIRED_CONDITIONAL_POSITIVE`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`, `independent-review-required`.
- Cryptanalytic result: no relation campaign, factor-log solve, blind descent,
  Shoup-bound improvement, or ECDLP breakthrough.

R13-R15 leave one operation: compress the complete Kummer branch image on a
dense endpoint support crossed with the fifth deck without materializing the
`S*L=B^3` source grid. R16 separates two kinds of cross-endpoint component.

Degree-one components come only from the finite deck automorphism group of
`f=psi composed with x`. On a prime-order subgroup they relate an endpoint
only to a constant-size origin-fixing automorphism orbit, at most three Kummer
classes in characteristic greater than three. They cannot provide a
polynomial compression factor.

The non-birational exception is constructive. Partition endpoints by their
common `psi` value. If every block admits one supplied common right factor of
the complete fifth-branch maps, and the frozen fifth deck actually fills its
finite quotient fibers, the image, exact counts, one joint source, and all
dyadic source restrictions can be built in

```text
O~(S + S*L/d)
```

work and state. This reaches `B^(9/4+o(1))` at `d>=B^(3/4-o(1))`, exactly the
surviving R13 range. It is the first typed positive compiler at that exponent
in this lane.

No such prime-field block factor, deck-saturation theorem, compact identity
certificate, fresh-target action, or R10 query index is supplied. Standard
isogeny and Lattes factors fail the finite-deck compression premise because
they are injective on the prime-order subgroup. The result is therefore a
conditional algorithm and a sharper experiment-free target, not positive
ECDLP evidence.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R15 translation-correlation producer | `ea007be0b237b91127e6a42501873f914ad31d6d4fd4393c3ae6b85005893b9b` |
| P1553 R15 parent report | `9aa2b9b7fd78e7b6378a01f8a7011ce67c5d273e1cf3bc08c8b873358f763f92` |
| P1553 R14 tensor-trace compiler gate | `da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac` |
| P1553 R14 independent red team | `44c05cdfac2b994d25889e104e610bd37f635f1b97d1a5e5f4956363358c3d8f` |
| P1553 R13 erased-image gate | `78ff09226852f140a8c33ce5ba98106ed62007b7126673d3f3af2c49d08a2cc5` |
| P1553 R12 high-degree provenance gate | `999159b7f3240ad6cc78034bc68ef6f9c27bfb3a00f88ff2ffc38cc58e9fe181` |
| P1553 R11 Cartesian Kummer rigidity gate | `6c79b486bfa4cfd14674033a438db3d91ddf7bbe4d2c4aaf309f1a0706f0df4e` |
| IDEA-057 prime-order composable-bucket theorem | `524a59c1728bcbea804ac4be42ace5a965b68a6332e85d941829b89e04fc4225` |
| IDEA-195 non-Cartesian S3 intertwiner hypothesis | `deferred corpus owner; no promoted artifact hash` |

R15 is still independently unreviewed. R16 therefore inherits and explicitly
reopens every R15 sign, chart, Fourier, and algorithm-interface check. Neither
draft may be promoted until independent review binds their final hashes.

## Frozen complete branch maps

Work over `k=F_p`, `p>3`, with a cyclic prime-order subgroup `G=<P>` of odd
order `N`, `N!=p`, and

```text
B=N^(1/5),
S=|U|=Theta(B^2),
L=|I_5|=Theta(B),
B^(3/4-o(1))<=d<=B,
2d<N.
```

Let

```text
f=psi composed with x:E->P^1,    deg(f)=m=2d.
```

The map is separable because `m<p` in the screened range. For a signed
endpoint `A` and a fifth Kummer class `q=x(Q)`, define the complete unordered
branch map

```text
F_A(q)=[f(A+Q)]+[f(A-Q)] in Sym^2(P^1).
```

It is well-defined on the fifth Kummer line. On a regular affine chart it is
represented by trace and norm

```text
T_A(q)=f(A+Q)+f(A-Q),
V_A(q)=f(A+Q)f(A-Q).
```

All pole, infinity, tangent, identity, and denominator-zero strata use a
constant number of saturated tagged charts. Every factorization below must
hold on the complete chart family, not only after dropping a denominator.

Partition the endpoint support by the first complete-key coordinate:

```text
H_z={A in U:psi(x(A))=z},
m_z=|H_z|<=d,
sum_z m_z=S.
```

Cross-block image keys are distinct because their first coordinate `z` is
distinct.

## Theorem 1: degree-one cross components are deck orbits

For signed `A`, put

```text
iota_A(X)=2A-X
```

and define the ordered branch map

```text
Gamma_A:E->P^1 x P^1,
Gamma_A(X)=(f(X),f(iota_A(X))).
```

The unordered branch map is obtained by quotienting the target by coordinate
swap. Let

```text
D_f={rho in Aut(E_bar):f composed with rho=f}
```

be the geometric deck automorphism group. Since `f` is separable,

```text
|D_f|<=deg(f)=m,
[-1] in D_f.
```

Suppose a direct or coordinate-swapped cross-endpoint fiber product of
`Gamma_A` and `Gamma_A'` has an irreducible component whose two projections
to `E` have degree one. Its normalization is the graph of an automorphism
`rho:E->E`. Comparing the two coordinates, in either orientation, produces
`rho,sigma in D_f` satisfying

```text
iota_A'=sigma composed with iota_A composed with rho^(-1).       (1)
```

Write every elliptic automorphism uniquely as

```text
rho=t_R composed with u,
sigma=t_S composed with v,
u,v in Aut(E,0).
```

The linear part of the right side of (1) is `-v*u^(-1)`. Equality with the
linear part `[-1]` of `iota_A'` forces `u=v`, and the translation part gives

```text
2A'=2u(A)+R+S.                                             (2)
```

The presence of `[-1]` in `D_f` bounds the translation parts. For
`rho=t_R u`, the commutator

```text
rho[-1]rho^(-1)[-1]=t_(2R)
```

lies in the translation subgroup `K=D_f intersect E`, so `2R` and `2S` have
order dividing `|K|<=m`. Hence

```text
T=A'-u(A)
```

is killed by `4|K|`. It is also `N`-torsion because `A,A' in G` and every
origin-fixing automorphism preserves `E[N]`. Since `N` is odd prime and
`N>m`, the two torsion orders are coprime, so `T=0`. Therefore

```text
A'=u(A).                                                   (3)
```

For `p>3`, `Aut(E_bar,0)` has order `2`, `4`, or `6`. Because `[-1]` acts
trivially on Kummer classes, a fixed endpoint has at most three Kummer-class
partners arising from every degree-one common component. Requiring also
`psi(x(A'))=psi(x(A))` can only shrink this orbit.

Thus graph/deck components provide constant-factor symmetry only. Any
polynomial cross-endpoint component compression must use a component of
higher projection degree, equivalently a non-birational branch map or a
nontrivial common intermediate cover. This is a scoped classification, not a
bound on all zero-dimensional cross collisions.

## Common-right-factor interface

Because the fifth input is the Kummer line `P^1_q`, every common intermediate
function field is rational by Luroth's theorem. For each nonempty block `H_z`,
the candidate supplies:

1. a rational map

   ```text
   pi_z:P^1_q->P^1_r,       degree e_z>=1;
   ```

2. for every `A in H_z`, complete tagged rational maps

   ```text
   G_(A,z):P^1_r->Sym^2(P^1)
   ```

   such that the exact all-chart identity

   ```text
   F_A=G_(A,z) composed with pi_z                         (4)
   ```

   holds;
3. a charged certificate for (4), with verification cost `v_z`;
4. the exact image of the frozen fifth occurrence deck under `pi_z`, including
   integer multiplicities and one fifth-occurrence backpointer per image;
5. exact evaluation circuits for every `G_(A,z)` on that finite image.

Let

```text
J_z=|pi_z(I_5)|.
```

The algebraic degree `e_z` alone does not imply `J_z<=L/e_z`. An arbitrary
list may select one point from each geometric fiber and have `J_z=L`.
Finite-deck saturation is therefore an explicit, separately certified input:

```text
J_z=O(L/e_z).                                             (5)
```

This distinction is decisive on the prime-order subgroup.

## Field-direction guard: right factors use a compositum

For each endpoint define the branch-coordinate field

```text
K_A=k(T_A,V_A) subset k(q).
```

The direction of the inclusion is contravariant. If
`F_A=G_(A,z) composed with pi_z`, then

```text
K_A subset k(pi_z) subset k(q).
```

Consequently the smallest field that can contain every branch-coordinate
field is their compositum

```text
C_z=k(K_A:A in H_z).
```

By Luroth's theorem, `C_z=k(h_z)` for a rational map `h_z`. Every common
right factor satisfies

```text
C_z subset k(pi_z) subset k(q),
[k(q):C_z]=deg(pi_z)*[k(pi_z):C_z].                       (field-direction)
```

Thus `h_z` is itself the maximum-degree common right factor, and the degree
of every proposed `pi_z` divides `[k(q):C_z]`. If `C_z=k(q)`, no nontrivial
common right factor exists. Computing only

```text
intersection_(A in H_z) K_A
```

tests a different object: common left composites, meaning rational functions
that can be written from every branch output. It neither constructs nor
certifies a common right factor.

The scalar control `X(q)=q^2`, `Y(q)=q^3` makes the distinction exact:

```text
k(X) intersection k(Y)=k(q^6),
k(X,Y)=k(q).
```

The maps have a nonconstant common left composite but no nontrivial common
right factor. Any harness route that accepts the intersection alone is a
false positive. The same field direction applies to the vector-valued branch
map through its coordinate field `k(T_A,V_A)`.

### Pairwise birational kill certificate

The compositum direction gives a small exact rejection witness. For two
distinct endpoints in one proposed block, define

```text
C_(A,A')=k(T_A,V_A,T_A',V_A')
```

and let

```text
n_(A,A')=[k(q):C_(A,A')].
```

If a common right factor `pi_z` of degree `e_z` exists, then

```text
C_(A,A') subset k(pi_z),
n_(A,A')=e_z*[k(pi_z):C_(A,A')].                         (pair sieve)
```

Therefore `e_z` divides every paired-map degree in the block. In particular,
one pair with `n_(A,A')=1` proves that the paired complete branch map

```text
q |-> (F_A(q),F_A'(q))
```

is birational onto its image and rejects every nontrivial common right factor
for the full block. A pair with `n_(A,A')<e_z` rejects the proposed degree.

An exact certificate eliminates `q'` from the four complete equalities for
trace and norm at `A` and `A'`, saturates the diagonal and every projective
denominator chart, and proves the generic off-diagonal fiber degree. An
affine trace-only gcd, an unsaturated resultant, or a sampled finite-deck
collision is not a pairwise certificate.

## Theorem 2: polynomial psi cannot supply a growing block

Suppose `psi` is a polynomial of degree `d`. Then `f=psi composed with x`
has its unique pole at the elliptic identity `O`, of order `2d`. Let
`H_infinity` be the projective hyperplane in
`Sym^2(P^1)=P^2` consisting of branch divisors that contain infinity. For a
nonidentity odd-order endpoint `A`, the complete branch map has effective
pullback divisor

```text
F_A^*(H_infinity)=2d*[x(A)].                              (6)
```

Indeed, above `q=x(A)` one signed lift has `A-Q=O` and the other has
`A+Q=O`; the Kummer quotient is unramified at `+-A` because `A` is not
2-torsion. Every other lift keeps both translated points away from `O`.
Using the projective branch hyperplane prevents a zero of the finite branch
from cancelling this pole, which could happen in an affine norm coordinate.

If a common factor of degree `e_z>1` satisfies

```text
F_A=G_(A,z) composed with pi_z,
```

then pulling back `G_(A,z)^*(H_infinity)` through `pi_z` must give (6).
Consequently `pi_z` has one point above the relevant outer hyperplane
preimage and is totally ramified with index `e_z` at `x(A)`.

A degree-`e_z` rational map `P^1->P^1` has total ramification

```text
sum_P (e_P-1)=2e_z-2.
```

Each totally ramified point contributes `e_z-1`, so one map has at most two
such points. Distinct Kummer endpoints give distinct `x(A)`. Therefore

```text
e_z>1  implies  m_z<=2.                                  (7)
```

The balanced requirement `m_z=Theta(d)` is impossible as `d` grows. This
closes polynomial `psi`, including monomials and polynomial interpolation
maps, before coefficient or finite-deck costs. It does not close rational
maps with two or more pole fibers, where a common factor can distribute
ramification across several preimages.

## Theorem 3: every value fiber must have linear support

The pole count is only one coordinate of an invariant statement. For every
geometric output value `c in P^1`, let

```text
r_c=|support(psi^(-1)(c))|.
```

Let `H_c` be the projective hyperplane in `Sym^2(P^1)` consisting of branch
divisors that contain `c`. The support `D_c` of `f^*(c)` on `E` is
`[-1]`-stable and has at most `2r_c` points. For every endpoint `A`, the
support of `F_A^*(H_c)` on the fifth Kummer line is contained in

```text
P_(A,c)={x(R-A):R in D_c},       |P_(A,c)|<=2r_c.
```

If `F_A=G_(A,z) composed with pi_z`, every point in the support of
`G_(A,z)^*(H_c)` pulls back to a complete fiber of the degree-`e_z` map
`pi_z`. At least one such outer fiber exists because the pullback divisor has
degree `2d`. Its ramification defect is at least

```text
max(0,e_z-|P_(A,c)|)>=max(0,e_z-2r_c).
```

A fixed Kummer point `q=x(Q)` belongs to `P_(A,c)` for at most `4r_c`
signed endpoints: for each `R in D_c`, the equation `x(R-A)=x(Q)` gives
`A=R-Q` or `A=R+Q`. Summing the local defects over one block and using
Riemann-Hurwitz for `pi_z:P^1->P^1` gives, for every `c`,

```text
m_z*max(0,e_z-2r_c) <= 4r_c*(2e_z-2)
                     = 8r_c*(e_z-1).                     (value-fiber bound)
```

Equivalently, in both cases `r_c<e_z/2` and `r_c>=e_z/2`,

```text
r_c >= m_z*e_z/(2m_z+8(e_z-1)).                          (12)
```

Consequently, if `m_z=Theta(d)` and `e_z=Theta(d)`, then

```text
min_(c in P^1) r_c=Omega(d).
```

This is invariant under every output Mobius map `M`. Replacing `psi` by
`M composed with psi` applies the automorphism `Sym^2(M)` to every complete
branch output, preserves all right-factor and finite-deck premises, and only
relabels which fiber is called infinity.

Thus every family with even one value having `o(d)` distinct preimages fails
the balanced block gate, including every output-Mobius transform of a
polynomial or power map. The unbounded cyclic and dihedral Galois quotient
families also fail: their standard branch fibers have respectively one and at
most two distinct preimages. Exceptional tame Galois quotient degrees are
bounded. A survivor must be a fiber-uniform, non-Galois rational map with
linear support in every value fiber. This remains a necessary condition, not
a construction or an unrestricted lower bound.

## Theorem 4: conditional output-sensitive block compiler

Assume the complete interface above. Fast univariate multipoint evaluation
computes `pi_z` on all fifth entries in `O~(L+e_z)` operations. Hashing exact
projective outputs gives their integer occurrence weights and one source.

Since `F_A=G_(A,z) composed with pi_z` and `deg(F_A)=O(d)`, a separable
factor has

```text
deg(G_(A,z))=O(d/e_z)
```

on each constant chart. Evaluating it on the `J_z` attained quotient labels
costs `O~(J_z+d/e_z)` per endpoint. Aggregating the complete key
`(z,G_(A,z)(r))` retains exact geometric and occurrence multiplicities and
one jointly coupled endpoint/fifth source.

The total setup work is

```text
W_block=O~(S + sum_z [
  L + e_z + m_z*(J_z+d/e_z) + v_z
]).                                                       (8)
```

The retained key and source state is

```text
M_block=O~(S + sum_z [J_z+m_z*J_z]).                      (9)
```

Equations (4) prove containment: every source pair belongs to one endpoint
block and its branch key is generated from the exact attained `pi_z` image.
No moment-prefix stopping rule is used.

For the balanced passing regime

```text
number of blocks=Theta(S/d),
m_z=Theta(d),
e_z=Theta(d),
J_z=O(L/d),
sum_z v_z=O~(S+S*L/d),                                   (10)
```

equations (8) and (9) become

```text
W_block,M_block=O~(S+S*L/d).                              (11)
```

At dense support and `d>=B^(3/4-o(1))`, this is at most

```text
B^(9/4+o(1)).
```

This is an exact conditional algorithm, not only a representation count.

## Dyadic counts and one-source replay

The compiler extends to every canonical dyadic restriction without restoring
`S*L` state.

For fifth-deck restrictions, each occurrence contributes its `pi_z` label and
backpointer to `O(log L)` ancestors for each endpoint block. The total is

```text
O~((number of blocks)*L)=O~(S*L/d).
```

Endpoint restrictions use the inherited pair-occurrence dictionary: for a
canonical pair rectangle it returns the exact weight and one pair backpointer
for each attained endpoint `A`. Given a queried complete key `(z,D)`, inspect
only the at most `m_z` endpoints in `H_z` and the roots of

```text
G_(A,z)(r)=D
```

in the attained `pi_z` histogram of the requested fifth node. The balanced
factor has degree `O(d/e_z)=O(1)`, so this costs `O~(m_z)=O~(d)<=B^(1+o(1))`
per keyed child query and returns one common pair/fifth backpointer. The
explicit full-image table of all `(A,r)` keys already has
`sum_z m_z*J_z=O(S*L/d)` entries.

Exact keyed child counts are additive integers, and a positive child retains
one common endpoint/fifth source. Complete projective verification is still
mandatory after replay. R16 does not claim that a fresh target can be turned
into the queried key `(z,D)` without the missing target-action interface.

Arbitrary adaptive subsets outside the canonical dyadic trees are not
claimed. A target-dependent repartition or target-trained quotient is charged
as fresh advice.

## Positive and negative controls

### Isogeny and Lattes control

If `phi:E->E'` is an isogeny and `psi` is its induced Kummer map, then

```text
phi(A+-Q)=phi(A)+-phi(Q)
```

gives a global common right factor `pi(q)=x(phi(Q))`. This is the exact
algebraic positive control for (4).

On the target prime-order subgroup, every isogeny of degree below `N` is
injective. Hence the finite fifth deck has

```text
J_z=L,
```

not `L/e_z`, and endpoint `psi` blocks have size one. The control fails (10)
and supplies no compression.

On a composite-torsion source with an `e_z`-point kernel intersecting the
deck, `m_z=e_z` and `J_z=L/e_z` can both hold. The compiler then passes its
state, source, and branch identities but fails the mandatory prime-order gate.

### Arbitrary finite list control

A high-degree interpolation map can force chosen endpoint or fifth labels to
collide. It does not establish (4), (5), or a compact verifier. Listing every
finite source-to-output value is the excluded `S*L` advice route.

Polynomial interpolation is additionally closed as a growing balanced block
factor by Theorem 2: its one pole forces more totally ramified points than a
degree-`e_z` rational `pi_z` can have.

More generally, Theorem 3 closes every map with any value fiber of support
`o(d)`. Merely postcomposing a polynomial or power map so that infinity has
`Theta(d)` preimages does not evade the gate: its original exceptional fiber
is still tested through the corresponding projective hyperplane.

### Degree-one deck control

Origin-fixing curve automorphisms may create exact branch symmetries, notably
on `j=0` or `j=1728` curves. Theorem 1 retains them and charges their constant
Kummer orbit. They cannot supply the `B^(3/4)` average compression required
by the passing block regime.

## Constructor and verifier boundary

Known generic common-right-factor algorithms may expand degree-`d`
coefficients separately for every endpoint. The obvious identity check of
all `F_A=G_A composed with pi_z` coefficients costs up to

```text
sum_z m_z*d=S*d,
```

which exceeds setup in the surviving degree range. R16 therefore does not
hide verification inside a supplied factor. A positive construction must
give one of:

1. a global or block-circuit identity with total proof and verification cost
   bounded as in (10);
2. a batched polynomial identity check whose represented input and soundness
   are fully charged;
3. a structural theorem, such as an explicit addition law, that proves every
   block identity from `O~(S+SL/d)` data; or
4. another exact containment certificate with the same bound.

Random spot checks alone do not promote the theorem. They may be used only in
a later preregistered toy falsification contract with an independent exact
verifier.

## Fresh target, R10, and complete ECDLP boundary

The block compiler constructs the target-independent endpoint/fifth branch
image and its canonical dyadic restrictions. It does not yet define how a
fresh masked target acts on block labels without rebuilding the partition or
supplying target advice.

It also does not solve R10's query-sensitive rank-two multiplicative
convolution. A complete product histogram may still have `B^4` support while
only `B` shifted coefficients are requested. A valid successor must either
derive those coefficients from the same block factors or provide a separately
charged exact index, including zero strata and one occurrence source.

No R16 route supplies relation density, independent signed row rank,
repeated-column aggregation, factor-base logarithms, or identical
scalar-blind descent. The inherited accounting remains

```text
lambda=max(s,1+d_loss+kappa,2)/5,
mu=max(s_m,w,2)/5.
```

The conditional block compiler establishes only `s,s_m<=9/4` in base-`B`
units under (10). It supplies none of `d_loss`, `kappa`, target workspace,
linear algebra, or descent.

## Scoped disposition

1. Every degree-one cross-endpoint common component is a deck double-coset
   graph and relates endpoints only through a constant origin-fixing
   automorphism orbit.
2. Polynomial `psi` cannot supply a growing nontrivial block factor: every
   endpoint would force a distinct total-ramification point and `m_z<=2`.
3. The value-fiber bound forces `Omega(d)` distinct preimages for every output
   value of a balanced common factor. It is output-Mobius invariant and closes
   every map with one sparse fiber, including disguised polynomial, power,
   cyclic-Galois, and dihedral-Galois families.
4. Polynomial-scale component compression for the remaining rational maps is
   forced into a non-birational/common intermediate-cover lane.
5. A common right factor per `psi` block gives an exact constructor with work
   and state stated by (8) and (9).
6. Balanced degree, finite-deck saturation, and compact identity verification
   reduce that cost to `O~(S+SL/d)`, which fits the R13 setup cap.
7. Dyadic counts and one jointly coupled source fit the same bound.
8. Standard prime-order isogeny/Lattes maps satisfy the algebraic factor but
   fail endpoint and fifth-deck compression through injectivity.
9. No actual list-specific block factor, compact verifier, fresh-target
   action, R10 evaluator, density, rank, factor logs, or descent is supplied.

The surviving operation is now typed as

```text
an explicit prime-field, list-specific family of simultaneous common right
factors pi_z for the complete Kummer trace/norm maps on full psi-fiber blocks,
with degree and attained-image ratio Theta(d), a total O~(S+SL/d) algebraic
identity certificate, exact dyadic provenance, fresh-target action, and R10's
queried rank-two coefficients.
```

No new idea, experiment, relation row, Shoup claim, or breakthrough is
authorized.

## Deduplication

- IDEA-195 owns simultaneous non-Cartesian trace/norm descent and exact source
  inversion. R16 types its first cap-sized block-factor compiler; it allocates
  no new idea ID.
- IDEA-057 owns global composable labels, isogeny/ECFFT controls, and the
  nonhomomorphic correction residual.
- P1553 R11-R13 own Cartesian rigidity, provenance, fiber size, and erased
  image energy.
- P1553 R14 owns the trace-Hankel decoder and containment/source split.
- P1553 R15 owns the translation-correlation and sparse-convolution interface
  screen; its independent review remains pending.

## Primary-source boundary

- Chalcraft and Fryers,
  [Kummer structures](https://arxiv.org/abs/0806.0409),
  supplies the complete two-valued Kummer-addition setting, not the proposed
  list-specific block factors.
- Pakovich,
  [On generalized Lattes maps](https://arxiv.org/abs/1612.01315),
  describes the structured rational-map family containing the isogeny/Lattes
  control; it does not supply finite-deck compression on the prime subgroup.
- Pakovich,
  [On intersections of fields of rational functions](https://arxiv.org/abs/2603.29609),
  studies intersections, common left composites, and a classified Galois
  equality case. R16 uses it as a field-direction and Galois-comparison
  source only. Its intersection theorem is not a common-right-factor theorem
  and is not transferred to the finite-field branch maps.
- Beauville,
  [Finite subgroups of PGL(2,K)](https://arxiv.org/abs/0909.3942),
  supplies the tame finite-subgroup classification used to isolate cyclic and
  dihedral as the only unbounded Galois quotient families. Here all screened
  degrees are below the characteristic; the value-fiber rejection itself is
  the elementary divisor count above.
- Ben-Sasson, Carmon, Kopparty, and Levit,
  [Elliptic Curve Fast Fourier Transform Part I](https://arxiv.org/abs/2107.08473),
  supplies the isogeny-tree arithmetic control; its target-prime-subgroup
  maps remain injective.

## Exactly one next action

Construct or refute one explicit block `H_z` with `m_z=Theta(d)` and one
rational `pi_z` of degree `Theta(d)` for which every complete trace and norm
map `F_A`, `A in H_z`, factors through `pi_z`, the frozen fifth deck has
`J_z=O(L/d)`, and the all-chart identity has `O~(d+L)` proof and verification
cost. Restrict the next construction to fiber-uniform non-Galois rational maps
with `min_c |support(psi^(-1)(c))|=Omega(d)`, and compute the compositum
`C_z=k(T_A,V_A:A in H_z)` symbolically on paper before any run. Start with
one exact paired-map degree `n_(A,A')`; a birational pair rejects the whole
block. Reject every intersection-only, deck-only, isogeny-injective, trace-only,
denominator-dropped, target-trained, or explicit-table factor. A passing block remains
`model-bound` until fresh-target action, R10, density, rank, factor logs, and
blind descent are complete.
