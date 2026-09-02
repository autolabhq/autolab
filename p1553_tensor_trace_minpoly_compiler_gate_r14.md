# P1553 tensor-trace minimal-polynomial compiler gate R14

## Classification

- Owner: existing P1553/P1551/P1533/P1510/P1514/IDEA-003/IDEA-012/
  IDEA-121/IDEA-133/IDEA-195 trace, power-projection, resultant, and
  source-replay frontier; no P1554.
- Evidence: exact split-algebra derivation and represented-route accounting;
  no run.
- Status: `REVISE_SCOPED_THEOREM`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, factor-log solve, blind descent,
  generic lower-bound violation, or ECDLP breakthrough.

R13 proved that the fifth-erased complete Kummer image has
`K=Omega(B^3/d)` states at certified dense support for `d<=B`, so an explicit
image beneath `B^(9/4)` begins only at `d=Omega(B^(3/4))`. It then asked for
one output-sensitive compiler that constructs that image without forming the
`S*L=B^3` source graph.

R14 freezes the strongest standard candidate: compute the minimal polynomial
of a separating image coordinate from tensor traces. The decoder is exact and
output-sized. If `K` is correctly certified and the required trace sequences
are supplied, `2K` moments recover the squarefree image polynomial,
geometric multiplicities, occurrence weights, and every coordinate constant
on an image fiber in `K^(1+o(1))` arithmetic. This is a genuine positive
algebraic result.

The missing operation moves rather than disappears. Every reviewed standard
constructor for those moments first represents the `S*L` quotient, a
degree-`S*L` norm/resultant, a source-sized multiplication map, or a dense
source-value vector. Aggregate source-marker moments return fiber sums, not
one jointly coupled pair/fifth source. A finite moment prefix also does not
certify that no additional image state was omitted unless a valid upper bound
or image-containment proof is supplied.

This is a compiler-grammar receipt, not an unrestricted lower bound. Compact
bivariate trace algorithms, P1510-style factorization, additive or
multiplicative separability, uniform fibers, compressed containment proofs,
and arbitrary circuits remain explicit exceptions.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R13 erased-image gate | `78ff09226852f140a8c33ce5ba98106ed62007b7126673d3f3af2c49d08a2cc5` |
| P1553 R13 parent report | `b4c1645ab58f470b38519ebb0e9bd993547a5a3d39d56a1c8eb82413ae0ff838` |
| P1553 R13 independent red team | `6ccc492fab0553ec2a808300f7bf41cd9f26043bd3e3cd64eb21f60b58df169c` |
| P1553 R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| P1553 ZR represented-algebra gate | `41a93d0f0d9c1c0b1a1adb0debf39b8a84425f0d4f7147ea5eb12c89719c2b08` |
| P1553 ZR independent red team | `0f1041d6a8aa2859e8e087e88462044f7cab76da08c8e78f0597f857838fc8a3` |
| P1551 finite-domain selector circuit gate | `5f1bd9c12ca700074c9cd327f6539bc880ec60b27431dc5f34e23b0a12f6c68f` |
| P1533 collision-multiset resultant specification | `f9eb2ae215d75cce9684e8d2fde7d4a94e6ce36a9a8787eec8a5614ea1a8f337` |
| IDEA-121 KU circuit reduction | `6fcca1d12e911f6eb2142ac96b6d0a83b6ac20db11efd06bc24c0abb7c99dc48` |
| P1514 nonlinear apolar operation theorem v2 | `b97666d65119c90eb7c63ac9df3be650af1b1e42854f613d1b39dc1eb68c50b2` |
| IDEA-133 Frobenius projector/norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |
| P1510 marked-resultant compiler | `20c7e26c55801aba57d2095254823f20de66ca499c8281ac761603391e8f0d68` |
| P1510 independent compiler audit | `e89c11c5a57ae2ac90f4d42b3d33558cb1c1ba1765d7409a7940accba3098452` |

The process remains bound to the P1436 focus-enrichment contract and the
alphaXiv workflow post
`https://x.com/askalphaxiv/status/2076737985559822734`. This is attention
guidance only, not mathematical evidence.

## Frozen tensor compiler interface

Work over `k=F_p`, `p>3`, with prime subgroup order
`N=p^(1+o(1))` and `B=N^(1/5)`. Let

```text
|U|=S=Theta(B^2),
|I_5|=L=Theta(B),
B^(3/4-o(1))<=d<=B,
n=S*L=Theta(B^3).
```

The public input includes squarefree support polynomials and exact
dictionaries for the distinct endpoint classes and fifth occurrences. The
endpoint dictionary includes positive occurrence weights `omega(u)` summing
to `B^2` and one exact signed pair backpointer per endpoint.

R13's complete projective key is

```text
Phi(u,q)=(psi(u),[f(A+Q)]+[f(A-Q)]),
f=psi composed with x.
```

Fix a valid constant-size projective chart and a scalar coordinate `a` that
separates the `K` attained complete keys. Pole and exceptional strata use
parallel saturated charts and are recombined with explicit tags.

In an affine chart, the source algebra is

```text
A=k[Z,W]/(g_U(Z),g_5(W)),       dim_k(A)=n=S*L.
```

The scalar image coordinate is an element `a in A`, but its input description
need not have `n` coefficients. On a short Weierstrass chart
`E:y^2=x^3+c_4*x+c_6`, the two branch x-coordinates are the roots of

```text
H_(u,q)(X)
  =(u-q)^2*X^2
   -2*((u+q)*(u*q+c_4)+2*c_6)*X
   +(u*q-c_4)^2-4*c_6*(u+q).
```

For `psi=P_psi/Q_psi`, the local complete image divisor is compiled by the
constant-branch resultant

```text
C_(u,q)(Y)=Res_X(H_(u,q)(X),Q_psi(X)*Y-P_psi(X)).
```

It is quadratic in `Y`. Equivalently, reduce the degree-`d` numerator and
denominator of `psi(X)` modulo `H` and take trace and norm. This gives the
complete branch coefficients as constant-many rational functions of
bidegree `O(d)` in `(u,q)`. After denominator clearing and saturation, a dense
chart description has `O(d^2)` coefficients. Thus

```text
input words = O(S+L+d^2)=B^(2+o(1)),
```

inside the setup cap. R14 does not charge an `S*L` table merely to describe
the map.

The desired compiler must return or support:

1. the squarefree image polynomial `P(T)` of degree `K<=B^(9/4+o(1))`;
2. exact geometric multiplicities and occurrence weights at every image key;
3. all complete key coordinates or an exact full-key dictionary;
4. one jointly coupled `(pair occurrence,fifth occurrence)` source;
5. a compressed proof that every source image is represented;
6. exact full-box and adaptive-child counts and source replay;
7. fresh-target action and R10's exact rank-two control.

Setup and retained state are capped by `B^(9/4+o(1))`; each fresh target,
adaptive child, and replay workspace is capped by `B^(5/4+o(1))`.

## Theorem 1: exact trace-Hankel image decoder

Let `Y=Phi(U x I_5)` and write `t_y=a(y)`. Since `a` separates the complete
keys, the `t_y` are pairwise distinct. Let

```text
m_y=#{(u,q):Phi(u,q)=y}.
```

R12 gives `1<=m_y<=8d^2`, and `8d^2<p` in the screened range. Define the
ordinary tensor-trace moments

```text
s_j=Tr_A(a^j)=sum_(y in Y) m_y*t_y^j.
```

For `0<=i,j<K`, the `K by K` Hankel matrix is

```text
H_K=(s_(i+j))=V*diag(m_y)*V^T,
V_(i,y)=t_y^i.
```

Therefore

```text
det(H_K)
  = product_y m_y * product_(y<z)(t_z-t_y)^2
  !=0 in k.
```

The minimal polynomial of multiplication by `a` on `A` is exactly

```text
P(T)=product_(y in Y)(T-t_y).
```

If `K` is correctly known or certified, the moments `s_0,...,s_(2K-1)`
determine the coefficients of `P` through the nonsingular Hankel recurrence

```text
s_(j+K)+c_(K-1)s_(j+K-1)+...+c_0*s_j=0,
P(T)=T^K+c_(K-1)T^(K-1)+...+c_0.
```

Fast Padé/Hankel methods recover `P` in `K^(1+o(1))` field operations. After
the roots are represented, the first `K` moments solve the Vandermonde system
for every `m_y`.

Occurrence weights have the same exact interface. Define the compressed
linear functional

```text
Tr_omega(h)=sum_(u,q) omega(u)*h(u,q)
```

and moments

```text
r_j=Tr_omega(a^j)=sum_y mu_y*t_y^j.
```

The same Vandermonde solve recovers the integer occurrence masses `mu_y`;
their total `B^2*L=B^(3+o(1))` is below `p`. For any complete-key coordinate
`c` that is constant on each image fiber,

```text
q_j=Tr_A(c*a^j)=sum_y m_y*c_y*t_y^j
```

recovers `c_y` after division by nonzero `m_y`. A constant number of such
sequences reconstructs the full projective key table.

This theorem is coefficient-complete and handles zero image coordinates,
since the Vandermonde determinant remains nonzero when one `t_y=0`. It is a
decoder given exact moments and a certified image bound, not a constructor for
those moments or that certificate.

## Theorem 2: finite prefixes do not certify completeness

If the actual image size is known to satisfy `K<=K_max`, then `2K_max`
correct moments contain enough information to recover the exact recurrence.
Without that promise, a recurrence fitting a finite prefix does not prove that
the true exponential sum has no additional states. Over a finite field,
smaller leading Hankel minors can vanish by cancellation when more than their
size many exponentials are present.

An exact constructor must therefore additionally supply one of:

1. a proved image-size upper bound matching the decoded recurrence;
2. an identity `P(a)=0` in the complete saturated source algebra;
3. a compressed image-containment certificate with charged verification; or
4. another exact stopping rule that excludes omitted image states.

Standard verification of `P(a)=0` reduces the bivariate expression modulo
`(g_U,g_5)` or evaluates it on all split source points. The frozen represented
route allocates `S*L` coefficients or values. R14 does not claim that every
proof system or compact identity must do so.

## Theorem 3: aggregate marker moments are not a source section

Let `b(u,q)` encode a pair/fifth source coordinate or occurrence index. The
marker moments give

```text
Tr_A(b*a^j)
  = sum_y beta_y*t_y^j,
beta_y=sum_(Phi(u,q)=y) b(u,q).
```

They recover only the fiber sums `beta_y`. Two different source multisets can
have the same sum, and independently recovered pair and fifth sums need not
name any common preimage. Higher marker moments can reconstruct a complete
fiber polynomial, but the total degree over all fibers is

```text
sum_y m_y=n=S*L.
```

Thus standard all-fiber Newton/Vandermonde source recovery restores the source
incidence payload. A set-theoretic source section of `O(K)` size exists by
R13, but Theorem 1 does not construct it.

There is an exact positive boundary. If every image fiber is simple, or a
source coordinate is already constant on each fiber, one marker sequence
does recover the source. R14 preserves uniform-fiber and family-specific
source selectors rather than treating the generic marker failure as a lower
bound.

For the actual high-degree branch map, the natural direct inverse starts from
the preimage lists of the two divisor roots under `f`. Their Cartesian join
has up to `O(d^2)` candidates; iterating a fifth deck against one preimage
list has `O(dL)` candidates. Both exceed `B^(5/4)` in the surviving degree
range unless additional structure avoids the join. These are route costs,
not lower bounds for algebraic inversion.

## Standard constructor ledger

All costs are base-field words or operations and suppress polylogarithmic
factors.

| Route | First represented object | Scoped result |
|---|---:|---|
| Split evaluation | all `a(u,q)` values | `S*L=B^3` values before moment compression |
| Eliminate fifth variable first | `Res_W(g_5(W),A(Z,W)-T*C(Z,W))` | dense `(Z,T)` rectangle `Theta(d*L^2)=B^(11/4..3)` |
| Eliminate endpoint variable first | symmetric dense `(W,T)` rectangle | `Theta(d*S^2)=B^(19/4..5)` |
| Nested norm/resultant | `Norm_A(T-a)` | degree `S*L`; radicalization comes afterwards |
| Sparse or toric resultant | multidegrees `(dL,dS,SL)` in the three coefficient groups | image-equation coefficient degree and mixed volume `S*L` |
| Canny-Emiris matrix | lattice region from the three Newton polytopes | `Theta((S+d)*(L+d))=Theta(S*L)` in the screened range |
| Quotient minimal polynomial | multiplication by `a` on `A` | source vector and action dimension `S*L` |
| Krylov/Wiedemann | one represented vector in `A` plus repeated multiplication by `a` | vector length `S*L`; a short recurrence does not shrink the supplied action |
| Modular power projection | `Tr_A(a^j)` over represented `A` | transposition preserves the source-algebra domain; standard input reduction has `S*L` coordinates |
| Bihomogeneous multiplication maps | admissible Macaulay restriction for the graph/projection | standard map is built from source/mixed-volume scale; a direct numerator stencil can expose `O(S*L*d^2)` traffic |
| Per-fifth slice plus union/lcm | `L` degree-`S` norms or source evaluations | `S*L` standard work before cross-slice deduplication |
| Dense repeated powers | coefficient rectangle for `a^j mod (g_U,g_5)` | standard allocation reaches `S*L` once `j*d` spans both quotient degrees |
| Supplied moments plus certified `K` | `O(K)` Hankel/Vandermonde data | exact passing decoder; constructor and completeness omitted |

The dense-power row uses only a representation statement. A bidegree-`O(d)`
input allocates

```text
min(S,O(jd))*min(L,O(jd))
```

coefficients for the standard dense remainder of `a^j`; this reaches the full
rectangle by `j=Omega(S/d)`. Special cancellations, low separation rank,
straight-line recurrences, and trace-only algorithms remain outside that row.

The fifth-first resultant row is an earlier standard-route failure than the
full norm. Its degree in `T` is `L` and its degree in `Z` is `O(dL)`, so a
dense coefficient representation has `Theta(dL^2)` slots. At the smallest
surviving degree `d=B^(3/4)`, this is already `B^(11/4)`, above the setup cap.
This does not exclude a compact circuit or truncated factorization of that
resultant.

The quotient and power-projection rows are the same represented-width issue
already owned by the P1553 ZR receipt, IDEA-121, IDEA-133, and P1551. R14 adds
the exact image-minimal-polynomial decoder and applies the width boundary to
the R13 erased image; it does not allocate a new mechanism owner.

## Separable positive controls

The trace interface is not vacuous. For a scalar Cartesian map with supplied
unary coordinates, exact moments can factor without touching `S*L` values.

### Multiplicative map

If

```text
a(u,q)=x(u)*y(q),
```

then

```text
s_j=(sum_u x(u)^j)*(sum_q y(q)^j).
```

Unary power-sum recurrences compute the first `2K` moments from the two factor
polynomials in `(S+L+K)^(1+o(1))` arithmetic. Weighted endpoint moments give
exact occurrence masses. Under a certified `K` promise, Theorem 1 therefore
constructs the distinct product-set polynomial and multiplicities in
output-sensitive time.

### Additive map

If

```text
a(u,q)=x(u)+y(q),
```

then

```text
s_j=sum_(r=0)^j binomial(j,r)
      (sum_u x(u)^r)*(sum_q y(q)^(j-r)).
```

Because `2K<p`, factorial-scaled exponential generating functions turn these
moments into one polynomial convolution. The same output-sensitive decoder
applies under a certified `K` promise.

For a queried additive or multiplicative key, a hash table for the `S`-point
side plus an `L`-point scan returns the exact weighted count and one joint
source in `O(L)` online work, inside `B^(5/4)`. Dyadic pair histograms supply
the analogous restricted dictionaries within `B^(2+o(1))` setup.

These are decisive positive controls. They show that compact tensor traces can
beat source enumeration when the map has a closed additive or multiplicative
moment law. They do not solve `Phi`: no such separation identity is supplied
for the complete high-degree Kummer branch key.

## P1510 and Kummer controls

P1510 similarly compiles its marked-resultant coefficient family from
pure-left, pure-right, and constant-size cross factors in output-sized work.
It proves that a family-specific multiplicative decomposition can bypass an
ambient product. R14 therefore preserves a P1510-style trace compiler for
`Phi` as the main exception.

The R12 composite-torsion quotient supplies another positive control: uniform
kernel fibers give compact image moments and a structured source inverse. It
fails the prime-order gate. Known-multiple arithmetic-progression decks also
admit compact interpolation but fail relation density and carry known logs.
Separable prime-order isogeny/Lattes/ECFFT maps remain injective on the
campaign subgroup and do not supply the required compression.

## Mandatory R10 rank-two control

The rank-two all-nonzero count is

```text
sum_(a_5)
  (D_(12,I) *_mult D_(34,I))(t_R/lambda_5(a_5)).
```

Multiplicative moment factorization can construct a complete sparse product
histogram output-sensitively only when its support size `K` itself fits and is
certified. For generic pair histograms, the complete convolution support can
be as large as `B^4`, while R10 asks only for `B` target-shifted values. A full
image compiler is then the wrong output interface. The known exact routes
remain `B^3` direct work or `p-1=B^(5+o(1))` dense character modes.

Consequently R14's trace-Hankel decoder does not pass the mandatory rank-two
control. A valid positive route needs a query-sensitive sparse convolution or
an equivalent direct evaluation, including zero strata, integer
multiplicities, rectangle identity, and one source.

## Target, child, and campaign boundary

An `O(K)` target-independent image table does not specify how a fresh target
and the remaining factor lists produce a complete key to query. Rebuilding an
image polynomial for an adaptive child can itself have `K` output, above the
online cap. A passing data structure must answer child counts and select a
source without reconstructing the child image.

No R14 route supplies relation density, independent signed row rank,
repeated-column aggregation, factor-base logarithms, or scalar-blind descent.
The inherited conditional exponents remain

```text
lambda=max(s,1+d_loss+kappa,2)/5,
mu=max(s_m,w,2)/5.
```

The trace decoder alone supplies none of `kappa`, `d_loss`, row rank, factor
logs, or descent.

## Scoped disposition

Within the frozen represented tensor-algebra grammar:

1. The complete Kummer key has an `O(S+L+d^2)` chart description; map input
   size itself fits setup.
2. Given certified `K` and exact ordinary, occurrence-weighted, and key-marker
   traces, a nonsingular Hankel/Vandermonde decoder reconstructs the exact
   image polynomial, multiplicities, weights, and full keys in `O~(K)`.
3. Standard split, norm/resultant, sparse-resultant, multiplication-map,
   Krylov, power-projection, Macaulay, per-slice, and dense-power constructors
   expose `S*L=B^3` source traffic before that decoder.
4. Finite trace prefixes do not replace image containment or a certified
   image-size bound.
5. Aggregate source markers do not choose one joint preimage on a multiple
   fiber; standard all-fiber recovery has total degree `S*L`.
6. Additive and multiplicative Cartesian maps and P1510 are exact
   output-sensitive positive controls, so no unrestricted compiler lower
   bound is claimed.
7. Fresh-target action, adaptive-child queries, R10 rank two, and the complete
   ECDLP path remain absent.

The sole surviving operation is now typed as

```text
a compact complete-chart tensor-trace compiler for Phi that computes enough
ordinary, occurrence-weighted, and key-coordinate moments to recover P and
its exact weights; certifies image containment; selects one jointly coupled
source on multiple fibers; and supports query-sensitive dyadic, fresh-target,
and rank-two evaluation, without materializing an S*L quotient vector,
resultant, multiplication map, or source-incidence family.
```

No new idea, experiment, relation row, Shoup claim, or breakthrough is
authorized.

## Primary-source boundary

- Abbott, Bigatti, Palezzato, and Robbiano,
  [Computing and Using Minimal Polynomials](https://arxiv.org/abs/1702.07262),
  supplies the finite-algebra minimal-polynomial framework; its algorithms do
  not provide the missing compact tensor trace or source selector.
- Bender, Buse, Checa, and Tsigaridas,
  [Solving bihomogeneous polynomial systems with a zero-dimensional projection](https://arxiv.org/abs/2502.07048),
  computes projections through multiplication maps and linear recurrences;
  the maps are constructed from admissible Macaulay restrictions and do not
  imply an `O(K)` constructor for this split source.
- Emiris,
  [A General Solver Based on Sparse Resultants](https://arxiv.org/abs/1201.5810),
  organizes sparse elimination by mixed volume and resultant matrices; in the
  frozen Cartesian graph, the coefficient degree in the image equation is
  `S*L`.
- Neiger, Salvy, Schost, and Villard,
  [Faster Modular Composition](https://arxiv.org/abs/2110.08354), and the
  Poteaux-Schost triangular-set algorithms are represented-quotient controls,
  not compact-circuit image or source oracles.

## Exactly one next action

Write one compact tensor-trace constructor for the actual complete-chart
Kummer coordinate `a`. It must compute the ordinary, occurrence-weighted, and
complete-key moment sequences through the certified image degree without an
`S*L` vector; return a compressed proof of `P(a)=0`; and provide one jointly
coupled source plus exact adaptive-child, fresh-target, and R10 rank-two
queries. Benchmark its symbolic coefficient support first against the exact
additive, multiplicative, P1510, composite-torsion, and generic dense controls.
Reject it at the first `S*L`, `d^2`-fiber join, `dL` scan, over-cap child
output, or unproved stopping rule; preserve any passing factorization as
`model-bound` until density, rank, factor logs, and blind descent pass.
