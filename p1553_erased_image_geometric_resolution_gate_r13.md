# P1553 erased-image geometric-resolution gate R13

## Classification

- Owner: existing P1553/P1551/P1533/P1510/P1515/IDEA-001/IDEA-003/
  IDEA-012/IDEA-057/IDEA-195 image, rigidity, resultant, and source-replay
  frontier; no P1554.
- Evidence: exact finite-set algebra and represented-route accounting; no run.
- Status: `REVISE_SCOPED_THEOREM`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, factor-log solve, blind descent,
  generic lower-bound violation, or ECDLP breakthrough.

R12 left one route open: erase the fifth label from the complete Kummer branch
state and construct the exact image plus one source without touching all
`S*B` endpoint/fifth incidences. R13 separates four facts that must not be
conflated.

1. Transposing R11's branch-rigidity theorem and counting fifth-label
   collisions gives `K=Omega(B^3/d)` at dense support for `d<=B`. Thus an
   explicit erased image beneath `B^(9/4)` still requires
   `d=Omega(B^(3/4))`.
2. The radical image, exact occurrence mass, and one source per image state
   have an `O(K)` representation when the image has `K` states.
3. Standard graph-first norm, characteristic-polynomial, multiplication-
   matrix, Fitting-module, and graph-geometric-resolution routes begin with a
   split algebra of dimension `n=S*L`; at dense support that is `B^3`.
4. None of these statements is an unrestricted construction lower bound. A
   family-specific output-sensitive compiler remains open, and P1510 is a
   concrete warning that multiplicative resultant structure can sometimes
   compile an output-sized object without expanding the ambient product.

The compact representation also does not by itself answer a fresh-target or
adaptive child query. One selected source per state cannot distinguish a
child containing a different source in the same fiber, and witnesses plus a
total-mass check do not certify that no image state was omitted.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R12 high-degree Kummer provenance gate | `999159b7f3240ad6cc78034bc68ef6f9c27bfb3a00f88ff2ffc38cc58e9fe181` |
| P1553 R12 parent report | `0b1c9869776a5d8669693d4a05eaf096659e02dbce424fbfce9dd3fff269a8f4` |
| P1553 R12 independent red team | `2a609a031cf3d5aa373f9ce458f0500c7cc51c149b309c74198a6c205b89c449` |
| P1553 R11 Cartesian Kummer rigidity gate | `6c79b486bfa4cfd14674033a438db3d91ddf7bbe4d2c4aaf309f1a0706f0df4e` |
| P1553 R11 parent report | `5f65da4f50e301067258f14f4b8e6dec91a0f096ac04e2418e4957bd649604b2` |
| P1553 R11 independent red-team report | `dec795cf4abcb14fd33b4e264dbe13eca168f504fc5557dfb6421bd04d99c14b` |
| P1553 R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| P1551 finite-domain selector circuit gate | `5f1bd9c12ca700074c9cd327f6539bc880ec60b27431dc5f34e23b0a12f6c68f` |
| P1533 collision-multiset resultant specification | `f9eb2ae215d75cce9684e8d2fde7d4a94e6ce36a9a8787eec8a5614ea1a8f337` |
| P1553 target-label common-factor gate R4 | `8cf15364c2da6830255216f3766a5d016b847d4bd012df92fd86c462ee6a9bc1` |
| P1510 global marked-resultant compiler | `20c7e26c55801aba57d2095254823f20de66ca499c8281ac761603391e8f0d68` |
| P1510 independent compiler audit | `e89c11c5a57ae2ac90f4d42b3d33558cb1c1ba1765d7409a7940accba3098452` |

The process remains bound to the existing P1436 focus-enrichment contract and
the alphaXiv post
`https://x.com/askalphaxiv/status/2076737985559822734`: spend attention on
the decisive experiment, resolve nonblocking ambiguity locally, and defer
peripheral branches. This is workflow guidance only.

## Frozen erased-image interface

Let `G=<P>` have odd prime order `N=p^(1+o(1))` over characteristic `p>3`
and put

```text
B=N^(1/5).
```

Let `U` be the distinct Kummer endpoint support of the first pair deck,
`|U|=S`, and let `I_5` be a fifth occurrence deck of `L=Theta(B)` distinct
Kummer classes. The campaign specialization uses `L=B`. A public pair
dictionary records, for every `u in U`, its exact integer occurrence weight
`omega(u)` and at least one signed pair occurrence backpointer. Thus

```text
sum_(u in U) omega(u)=B^2.
```

For a degree-`d` rational Kummer map `psi`, `f=psi composed with x`, and
representatives `A,Q`, define the complete state

```text
Phi(u,q)=(psi(u),D_(u,q)),
D_(u,q)=[f(A+Q)]+[f(A-Q)].
```

The branch-rigidity statements below assume `2d<N`, automatically true in
the screened range `d<=B` for sufficiently large `B`.

The divisor retains multiplicity, poles, identity, infinity, tangent, and
vertical strata. It lies in

```text
X=P^1 x Sym^2(P^1) isomorphic to P^1 x P^2.
```

Let `Y=Phi(U x I_5)` and `K=|Y|`. The reduced source graph has degree and
base-field dimension

```text
n=S*L.
```

For each `y in Y`, distinguish its geometric fiber size and occurrence mass:

```text
m_y  = #{(u,q):Phi(u,q)=y},
mu_y = sum_(Phi(u,q)=y) omega(u).
```

R12 gives `m_y<=8d^2`. Also `sum_y m_y=n` and
`sum_y mu_y=B^2*L=B^(3+o(1))`. Since `p=B^(5+o(1))`, every geometric or
occurrence count and both totals are strictly below `p` for the asymptotic
regime, so field residues have unambiguous integer lifts. The setup/state cap is
`B^(9/4+o(1))`; each fresh target and adaptive replay workspace is
`B^(5/4+o(1))`.

R13 freezes a **graph-first represented elimination grammar**. It admits
base-field arithmetic, complete projective charts, explicit split quotient
algebras for `U x I_5`, multiplication matrices in those algebras, standard
norm/resultant/trace/characteristic-polynomial operations on those represented
objects, standard Fitting or pushforward presentations, and graph geometric
resolutions that retain source coordinates. Every coefficient, matrix entry,
module generator, and temporary field element is charged.

The grammar does not include a free radical-image oracle, source-section
oracle, image-completeness oracle, compressed proof system, or unnamed
output-sensitive compiler. A specifically written compiler that exploits
factorization or uniform fibers is outside the negative theorem and is an
admitted escape.

## Theorem 1: transposed branch energy forces `K=Omega(B^3/d)`

For each endpoint `u`, let

```text
k_u=#{D_(u,q):q in I_5}
```

and let `C_u` be the number of unordered distinct fifth-label pairs `q,r`
with `D_(u,q)=D_(u,r)`. The complete branch divisor is symmetric:

```text
D_(u,q)=D_(q,u),
```

because `f` is even and the unordered pair contains `f(A+Q)` and
`f(A-Q)=f(Q-A)`. Therefore R11's projectively complete finite-Cartesian
rigidity theorem can be transposed. For each pair of distinct fifth Kummer
classes `q,r`, complete branch equality holds at no more than `16d` endpoint
classes `u`, including pole cases. Hence

```text
sum_(u in U) C_u <= 16d*binomial(L,2).
```

If the `k_u` divisor buckets at endpoint `u` have sizes `b_(u,j)`, then

```text
sum_j b_(u,j)^2=L+2C_u.
```

Cauchy-Schwarz first within each endpoint and then across endpoints gives

```text
k_u >= L^2/(L+2C_u),

sum_u k_u
  >= S^2*L^2/(S*L+2*sum_u C_u)
  >= S^2*L^2/(S*L+32d*binomial(L,2)).
```

A global erased key `(psi(u),D)` is counted in `k_u` for endpoints `u` lying
in one fiber of the degree-`d` map `psi`. Every projective fiber contains at
most `d` distinct points, including ramified, inseparable, and infinity
fibers. Thus `sum_u k_u<=dK` and

```text
K >= S^2*L^2 /
     (d*(S*L+32d*binomial(L,2))).
```

For certified dense support `S=Theta(B^2)`, `L=Theta(B)`, and `d<=B`, this
specializes to

```text
K=Omega(B^3/d).
```

Consequently an explicit radical image, root table, or other representation
with at least one field word per attained key can fit beneath
`B^(9/4+o(1))` only if

```text
d=Omega(B^(3/4-o(1))).
```

This strengthens R12's erased-key `SB/(8d^2)` floor by using branch rigidity
across the fifth deck. It is not the retracted claim that all cross-endpoint,
cross-fifth collision pairs number `O(d^2)`: R11 does not prove that claim,
and common components of the corresponding bivariate fiber products remain
unclassified. The theorem counts only fifth-label collisions at each fixed
endpoint and then charges the `psi` fiber.

## Theorem 2: an output-sized representation exists

Embed `X` in a fixed projective space. Choose a projective chart avoiding the
finite set `Y`, then choose a linear coordinate `ell` on that chart. A random
chart fails on at most `K/p` of the choices, and, conditional on a valid
chart, a random `ell` identifies some distinct pair with probability at most

```text
K*(K-1)/(2p).
```

When `K<=B^(9/4+o(1))`, this collision term is
`B^(-1/2+o(1))`. A valid constructor must detect bad choices, resample, and
charge the detection; the union bound proves existence but does not supply
that completeness test. No deterministic universal separator is assumed.

For an injective `ell`, write `t_y=ell(y)`. The following package has
`O(K)` base-field coefficients:

```text
P(T)       = product_(y in Y) (T-t_y),
W_geo(t_y) = m_y,
W_occ(t_y) = mu_y,
J_12(t_y)  = one exact pair-occurrence index above y,
J_5(t_y)   = one exact fifth-occurrence index above y.
```

Here `P` is squarefree of degree `K`, while the four interpolants have degree
less than `K`. The pair `(J_12(t_y),J_5(t_y))` is chosen by one deterministic
set-theoretic section of the fiber and must jointly name the same preimage of
`y`; two independently valid marginal selectors are insufficient. Additional
constant-many projective key coordinates can be
stored as degree-less-than-`K` interpolants. Equivalently, supply a sorted or
hashed table of the `K` tuples

```text
(t_y, complete_projective_key(y), m_y, mu_y, J_12(t_y), J_5(t_y)).
```

The table avoids making uncharged finite-field factorization a query step and
still occupies `O(K)` state. A returned occurrence is followed through the
public dictionaries and the original complete branch is recomputed before
acceptance.

The weighted logarithmic-derivative form is equivalent. If

```text
H_occ(T)=sum_y mu_y P(T)/(T-t_y),
```

then `H_occ(t_y)=mu_y P'(t_y)` and

```text
W_occ = H_occ*(P')^(-1) mod P.
```

Thus exact multiplicities do not intrinsically require storing all preimages.
If the root-indexed package is already supplied, a product tree and
multipoint interpolation/evaluation convert between the table and polynomial
forms in `K^(1+o(1))` field operations under standard fast polynomial
arithmetic. If only `P` is supplied, root finding is separately charged.
Full-key comparison is required: an ambient state outside `Y` may share the
same scalar projection `ell` with an image root.

This is a representation theorem, not a constructor. In particular, the
selected `J_12,J_5` values prove only that each represented root has a source.

## Theorem 3: standard graph-first elimination exposes `n=S*L`

Represent the complete divisor by a binary quadratic and saturate the
projective graph equations before passing to affine strata. Then write the
reduced source algebra as

```text
A=k[Z,W]/(g_U(Z),g_5(W)),       dim_k(A)=n=S*L,
```

with constant-many projective strata treated in parallel. Multiplication by
`ell(Phi)` on `A` has characteristic polynomial

```text
Chi_A(T)
  = Norm_A(T-ell(Phi))
  = product_((u,q) in U x I_5) (T-ell(Phi(u,q)))
  = product_(y in Y) (T-t_y)^(m_y).
```

Therefore `deg Chi_A=n`, while

```text
rad(Chi_A)=P
```

has degree `K`. Computing the degree-`n` norm or characteristic polynomial
and squarefree-reducing afterwards does not refund the degree-`n` represented
traffic. The same charge appears in the named routes:

1. An explicit multiplication matrix or split-value representation of `A`
   has at least `n` field entries before matrix overhead.
2. A graph geometric resolution separates `n` reduced source points, so its
   primitive polynomial has degree `n`.
3. The finite pushforward `Phi_* O_Gamma` has base-field dimension `n` as a
   module over the reduced image algebra. An explicit Fitting/module
   presentation therefore retains the source-fiber mass rather than only the
   radical support.
4. A standard resultant that first forms the full graph norm has degree `n`
   even if its final radical image has degree `K`.

The frozen source is reduced. If an unsaturated auxiliary presentation adds
nilpotents or embedded components, its scheme length is at least `n`; those
artifacts cannot lower the represented charge, while the reduced image still
has degree `K`.

At certified dense support `S=Theta(B^2)`, `n=Theta(B^3)`, outside both the
setup/state and fresh-target rectangles. Streaming trades retained state for
`B^3` work. This proves failure only for the frozen graph-first represented
grammar.

General geometric-elimination upper bounds polynomial in input length and
geometric degree do not turn this route ledger into a lower bound. Nor does
the graph degree imply that every image-only algorithm must pay `n`: the
radical image has degree `K`, and a family-specific compiler may avoid a full
graph representation.

## Theorem 4: completeness and adaptive replay are separate gates

The output-sized package is not self-certifying under the campaign caps.
Source sections establish

```text
roots(P) subseteq Phi(U x I_5).
```

They do not establish the reverse containment. The total checks

```text
sum_(P(t)=0) W_geo(t)=n,
sum_(P(t)=0) W_occ(t)=B^2*L
```

also do not suffice: an omitted state and a compensating false multiplicity
can preserve both totals. A standard exact completeness receipt evaluates

```text
P(ell(Phi(u,q)))=0
```

throughout the split source algebra, or proves the corresponding graph-ideal
containment. In the frozen represented grammar that returns to dimension
`n`. A specialized compressed identity or independently checkable proof is a
preserved exception, but must expose its construction and verification costs.

One source per image state is likewise insufficient for adaptive dyadic
replay. If a fiber contains distinct sources `s_0,s_1` and `J` selects `s_0`,
a queried child may exclude `s_0` and contain `s_1`. The child count is
positive even though the selected section lies outside it. Exact replay must
therefore provide one of:

1. a separately complete `(P_I,W_I,J_I)` package for every queried child;
2. an exact range-count/source-selection data structure over every image
   fiber; or
3. an implicit algebraic fiber query that constructs the child restriction
   inside `B^(5/4+o(1))` work and workspace.

Materializing all source incidences in a generic range structure costs `n`
before logarithmic overhead. Precomputing pair histograms alone can remain
`B^(2+o(1))`, as in R10, but crossing those incidences with the fifth deck
restores `B^3` generic incidence traffic. This is a standard-route charge, not
a cell-probe or data-structure lower bound.

## Mandatory rank-two control

R10's exact all-nonzero control reduces a five-list rank-two predicate to

```text
sum_(a_5)
  (D_(12,I) *_mult D_(34,I))(t_R/lambda_5(a_5)),
```

with zero strata handled separately. The two dyadic pair histograms and exact
backpointers fit in `B^(2+o(1))` state. The currently known exact evaluations
still use either `B^3` direct work or `p-1=B^(5+o(1))` multiplicative-character
modes.

An image polynomial for all triple products can have degree `B^3`; forming a
full sparse pair-convolution support can have `B^4` pair incidences. Merely
asserting an `O(K)` geometric resolution therefore does not express the
required sparse convolution query. Any negative grammar that cannot represent
this exact rank-two case is too weak to close the actual R10 route. Any
positive erased-image compiler must instantiate this control with integer
multiplicities, zero signatures, rectangle identity, and exact source replay.

## Positive and boundary controls

### P1510 output-sensitive marked resultant

P1510 constructs a complete degree-two marked resultant from pure-left and
pure-right quadratic norms plus `r^2` constant-size pair resultants. Its
independently replayed compiler uses `O(r^2 polylog r)` work and `O(r^2)`
state, proportional to its output family rather than an expanded endpoint
product. It does not solve this image problem or any ECDLP stage. It is a
decisive logical control: R13 may reject standard full-graph norm/resultant
routes, but may not claim that resultant syntax itself forces domain-degree
work.

### Composite-torsion quotient

R12's auxiliary composite-torsion quotient has explicit uniform kernel
fibers. Quotient labels and kernel-coset inversion construct an output-sized
image/index with exact verification. It passes the representation and source
interfaces but fails the prime-order campaign gate because the quotient
kernel intersects the source subgroup.

### Arithmetic-progression interpolation

R12's known-multiple progression decks admit degree-`O(B)` interpolation that
collapses the frozen grid. The image can be compact and explicitly inverted,
but the fivefold endpoint support is only `O(B)`, relation density is `O(B/N)`,
and collection needs `Omega(N)` targets under favorable independence. It is a
construction control, not a cryptanalytic candidate.

### P1533 characteristic-polynomial interface

P1533 already owns complete characteristic-polynomial, direct-resultant,
relative-norm, deterministic subdivision, and source-recovery semantics. R13
specializes its construction boundary to the fifth-erased Kummer image; it
does not create a new operation owner or idea ID.

P1515 and IDEA-057 own the list-specific rational-Kummer and finite-Cartesian
branch-rigidity theorem transposed in Theorem 1. R13 contributes only the
fixed-endpoint fifth-collision energy application to the existing P1553
erased-image route.

## Scoped disposition

Within the frozen graph-first represented elimination grammar:

1. The erased image obeys
   `K>=S^2*L^2/(d*(S*L+32d*binomial(L,2)))`; at dense support and `d<=B`,
   this is `Omega(B^3/d)`, so explicit `O(K)` state requires
   `d=Omega(B^(3/4-o(1)))`.
2. The exact radical image, geometric and occurrence multiplicities, one
   source, and complete keys admit `O(K)` storage when correctly supplied.
3. Standard norm, characteristic-polynomial, multiplication-matrix, graph
   geometric-resolution, and explicit pushforward/Fitting routes expose
   `n=S*L` represented traffic before radicalization.
4. At dense pair support this is `B^3`, above the campaign rectangle.
5. A root-indexed source section and two total-mass checks do not certify image
   completeness.
6. One selected source per state does not answer arbitrary dyadic child
   counts when a fiber has multiple sources.
7. The compact package does not implement a fresh-target query or R10's exact
   rank-two sparse multiplicative convolution.

The surviving exception is now narrower but real:

```text
an oracle-free, family-specific, output-sensitive radical-image compiler for
Phi that returns exact occurrence weights, a complete source selector, and a
compressed image-containment receipt; supports every adaptive child and fresh
target inside the online cap; and expresses R10's rank-two control, without
forming the n-point graph algebra or n source incidences. For dense support
and d<=B, its explicit-image branch begins only at d=Omega(B^(3/4)).
```

P1510 prevents closing this exception by generic resultant rhetoric. General
geometric-elimination degree bounds provide useful algorithms and vocabulary,
not the missing lower bound.

No new idea, experiment, relation row, factor-log solve, descent, Shoup claim,
or breakthrough is authorized.

## Complete campaign gate

A positive compiler remains `model-bound` until it also supplies:

```text
fresh-target and all-negative costs,
every adaptive Query2P1 child and exact occurrence-labelled replay,
complete signs and projective verification,
relation density and repeated-column aggregation,
independent signed row rank,
factor-base logarithm solve,
identical scalar-blind descent,
rho/BSGS fallback and bit complexity.
```

Using setup `B^s`, retained state `B^s_m`, fresh-target work `B^kappa`,
workspace `B^w`, and reciprocal density `B^d_loss`, the inherited conditional
exponents remain

```text
lambda=max(s,1+d_loss+kappa,2)/5,
mu=max(s_m,w,2)/5.
```

The image representation alone supplies none of `kappa`, `d_loss`, row rank,
factor logs, or descent.

## Primary-source boundary

- Giusti, Heintz, Morais, Morgenstern, and Pardo,
  [Straight-Line Programs in Geometric Elimination Theory](https://arxiv.org/abs/alg-geom/9609005),
  gives geometric-degree-sensitive zero-dimensional elimination algorithms;
  it is an upper-bound framework, not an `Omega(n)` image lower bound.
- Jeronimo, Krick, Sabia, and Sombra,
  [The computational complexity of the Chow form](https://arxiv.org/abs/math/0210009),
  likewise organizes elimination cost by input length and geometric degree;
  it does not exclude family-specific output-sensitive compilation.
- Bhargava, Ghosh, Guo, Kumar, and Umans,
  [Fast Multivariate Multipoint Evaluation Over All Finite Fields](https://arxiv.org/abs/2205.00342),
  accelerates evaluation from represented coefficients and points; it does not
  construct the missing image package or source-fiber oracle.

## Exactly one next action

Write one explicit output-sensitive compiler attempt for the frozen map
`Phi(u,q)=(psi(u),D_(u,q))`. It must take the unary support polynomials,
Kummer map, occurrence weights, and dictionaries as inputs and return
`P,W_occ,J_12,J_5` plus a compressed proof of
`Phi(U x I_5) subseteq roots(P)`. Require exact full-box and adaptive-child
counts, R10 rank-two replay, one completely verified source, and the inherited
campaign accounting in the surviving dense-support degree range
`B^(3/4-o(1))<=d<=B`. Reject the attempt at the first named intermediate over
`B^(9/4)` setup/state or `B^(5/4)` fresh-target work/workspace; preserve any
explicit multiplicative or uniform-fiber compiler that passes as
`model-bound` pending the full ECDLP path.
