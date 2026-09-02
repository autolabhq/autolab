# P1553 translation-correlation sparse-convolution gate R15

## Classification

- Owner: existing P1553/P1551/IDEA-057/IDEA-195 tensor-trace,
  nonhomomorphic-router, and Query2P1 frontier; no P1554.
- Evidence: exact algebraic reduction and algorithm-interface audit; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`, `independent-review-required`.
- Cryptanalytic result: no relation campaign, factor-log solve, blind descent,
  Shoup-bound improvement, or ECDLP breakthrough.

R14 left one compact constructor: produce the complete Kummer image moments,
containment certificate, one joint source, adaptive-child counts, fresh-target
action, and R10 rank-two answers without representing the `S*L=B^3` source
algebra. R15 tests whether output-sensitive sparse convolution supplies that
constructor.

The answer is negative for the named transplant, but for a more precise reason
than ambient group size. The complete divisor moments are weighted
three-point correlations on `(A,A+Q,A-Q)`. Their exact Fourier normal form has
two character indices. The raw pair of translates `(A+Q,A-Q)` is injective in
the source pair `(A,Q)`, so a sparse sumset routine for only `A+Q` does not
retain the second branch or a joint source. Integer-index sparse-convolution
algorithms also rely on additive indices and nonnegative, cancellation-free
coefficients that are absent from the field-valued moment problem.

This does not prove that compact correlations are impossible. It closes only
ordinary one-dimensional convolution, full Fourier tables, explicit
translate-pair support, and globally composable bucket transfers. A
list-restricted nonhomomorphic correlation sketch, a family-specific Kummer
identity, compressed containment, and query-sensitive sparse coefficient
evaluation remain open.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R14 tensor-trace compiler gate | `da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac` |
| P1553 R14 parent report | `ec5bc6425a29575468f78fad1803412dc9a69d6a69e581e5deaad7160cb7de8e` |
| P1553 R14 independent red team | `44c05cdfac2b994d25889e104e610bd37f635f1b97d1a5e5f4956363358c3d8f` |
| P1553 R13 erased-image gate | `78ff09226852f140a8c33ce5ba98106ed62007b7126673d3f3af2c49d08a2cc5` |
| P1553 R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| IDEA-057 prime-order composable-bucket theorem | `524a59c1728bcbea804ac4be42ace5a965b68a6332e85d941829b89e04fc4225` |
| P1551 finite-domain selector circuit gate | `5f1bd9c12ca700074c9cd327f6539bc880ec60b27431dc5f34e23b0a12f6c68f` |

The process remains bound to the P1436 focus-enrichment contract and the
alphaXiv workflow post
`https://x.com/askalphaxiv/status/2076737985559822734`. This is workflow
guidance, not mathematical evidence.

## Frozen translation interface

Work in a non-anomalous cyclic elliptic subgroup `G=<P>` of odd prime order
`N`, over a field of characteristic different from `N`. Put `B=N^(1/5)`.
Let `E` be the signed endpoint support above the `S=Theta(B^2)` Kummer
endpoint classes and let `Q` be the signed fifth support above `L=Theta(B)`
fifth classes. Constant sign factors are suppressed because every admitted
chart retains and verifies them.

For the degree-`d` rational Kummer map define

```text
h(R)=psi(x(R)).
```

On one regular affine chart, the complete unordered branch divisor has trace
and norm coordinates

```text
tau(A,Q)=h(A+Q)+h(A-Q),
nu(A,Q)=h(A+Q)*h(A-Q).
```

Together with `h(A)`, these determine the complete local key on that chart.
Pole, tangent, identity, and infinity strata are handled by a constant number
of separately tagged saturated charts. They do not change the source-product
or correlation dimensions below.

Fix a chart-separating affine scalar

```text
a(A,Q)=c_0*h(A)+c_1*h(A+Q)+c_2*h(A-Q)
         +c_3*h(A+Q)*h(A-Q).
```

The coefficients may include public chart tags or denominator-clearing
factors. A different constant-degree encoding of `(h(A),tau,nu)` only adds a
constant number of the same correlation families.

Let `e(A)` be the exact endpoint occurrence weight and `w(Q)` the fifth
occurrence weight. The ordinary and occurrence-weighted moments required by
R14 have the common form

```text
s_j=sum_(A in E,Q in Q) e(A)w(Q)*a(A,Q)^j.
```

## Theorem 1: exact three-point correlation normal form

Expanding the four terms of `a^j` expresses every moment as a public linear
combination of correlations

```text
C_(r,s,t)
  =sum_(A in G,Q in G)
     E_r(A) W(Q) H_s(A+Q) H_t(A-Q),

E_r(A)=1_E(A)e(A)h(A)^r,
W(Q)=1_Q(Q)w(Q),
H_s(X)=h(X)^s.
```

Only source-supported terms contribute. Literal multinomial expansion has up
to `Theta(j^3)` distinct `(r,s,t)` families at moment order `j`; this count is
a cost of that expansion, not a lower bound against a compact generating
function.

The two translated arguments are not independent decoration. Because `N` is
odd,

```text
X=A+Q,
Y=A-Q
```

has the exact inverse

```text
A=(X+Y)/2,
Q=(X-Y)/2.
```

Therefore the raw translate-pair support

```text
{(A+Q,A-Q):(A,Q) in E x Q}
```

has exactly `|E|*|Q|` entries before applying `h` and erasing sources. An
output-sensitive routine for the one-dimensional sumset `E+Q` may compress
the first coordinate, but it neither constructs the paired second coordinate
nor returns one source for a complete divisor image.

This is the first exact loss in the ordinary sparse-sumset transplant.

## Theorem 2: exact two-frequency Fourier control

Let `G_hat` be the character group over a splitting field and use

```text
f_hat(chi)=sum_(x in G) f(x)*chi(x)^(-1),
f(x)=N^(-1)*sum_(chi in G_hat) f_hat(chi)*chi(x).
```

Substitution into Theorem 1 gives

```text
C_(r,s,t)=N^(-2)*sum_(alpha,beta in G_hat)
  H_s_hat(alpha) H_t_hat(beta)
  E_r_hat((alpha*beta)^(-1))
  W_hat((alpha*beta^(-1))^(-1)).
```

The formula is exact. It has two frequency variables because the two branch
translations use `+Q` and `-Q`. Since squaring is a bijection on the odd-order
dual group, the change of variables

```text
gamma=alpha*beta,
delta=alpha*beta^(-1)
```

is also bijective; it permutes the `N^2` summands but does not collapse them.

There are genuine one-frequency controls. If `s=0` or `t=0`, the transform of
`H_0=1` is supported only at the trivial character, and the formula reduces
to an ordinary correlation. If `h(A+Q)` and `h(A-Q)` factor through a common
small quotient or one branch coordinate is already constant on every source
fiber, the effective support can collapse. Those are admitted positive
exceptions.

The standard exact routes are therefore:

| Route | First charged object | Scoped result |
|---|---:|---|
| Direct supported correlation | all `(A,Q)` incidences | `S*L=B^3` work |
| Raw translate-pair support | all `(A+Q,A-Q)` pairs | exactly `S*L` pairs |
| Full character evaluation | one or two full character tables | `N` or `N^2` modes, and nontrivial EC characters require scalar indexing |
| Literal moment expansion | `Theta(j^3)` correlation types at order `j` | over cap by the required moment range |
| One-branch correlation | one frequency after a trivial transform delta | exact positive control, incomplete for a general divisor key |
| Supplied compact correlation oracle | only requested correlations or moments | preserved open route |

No unrestricted lower bound is claimed for arithmetic circuits, sparse
Fourier support promised as input, low-rank transforms, or family-specific
identities.

## Theorem 3: composable hash transfer does not create a small EC index

The output-sensitive sparse nonnegative-convolution algorithms screened here
operate on known integer indices. Their modular folds preserve index addition;
nonnegative coefficients ensure that output support cannot disappear through
cancellation.

Transferring that interface to the prime-order EC subgroup would require a
public target-independent bucket law whose equality is preserved by group
addition. The bound IDEA-057 theorem applies: the bucket fibers are cosets of
a subgroup, so on prime-order `G` the label is constant or injective. A
nonconstant exact alphabet therefore has at least `N` states.

Hashing serialized point coordinates is allowed, but it is not composable:
from hashes of `A` and `Q` one cannot derive the hash of `A+Q` without either
the points, a charged repair table, or a new field-specific correction law.
The theorem does not exclude that new nonhomomorphic operation.

The moment problem also violates the nonnegative-convolution promise.
Although geometric and occurrence multiplicities are positive integers below
`p`, the correlation coefficients contain powers of field values `h(R)` and
can cancel in `F_p`. The image size `K` is not the support size of an ordinary
group convolution. Thus output-sensitive support discovery cannot be invoked
merely from `K<=B^(9/4+o(1))`.

## Mandatory R10 multiplicative-convolution control

R10 asks for `B` shifted values of

```text
(D_(12,I) *_mult D_(34,I))(z)
```

on `F_p^*`, with exact nonnegative integer coefficients, zero strata,
rectangle identity, and one occurrence backpointer. Here the coefficient
promise does match sparse nonnegative convolution, but the index interface
and output interface still differ.

1. A full multiplicative convolution can have `B^4` nonzero outputs, while
   R10 requests only `B` target-shifted coefficients. An algorithm
   output-sensitive in the full support size is then over cap.
2. Integer-index folding requires discrete-log indices or homomorphic
   quotients of `F_p^*`. A quotient of order `m` is available when suitable
   divisors of `p-1` are supplied, for example through
   `x -> x^((p-1)/m)`. A generic-prime algorithm cannot assume a divisor chain
   in the required range.
3. Bucket coefficients alone do not identify an exact product value or one
   source when many field elements share a bucket. Exact lifting must charge
   the collision lists or an additional source-recovery identity.

Passing controls remain explicit. Known scalar logs turn multiplicative
labels into known exponent indices. Smooth supplied divisors of `p-1` permit
FFT or quotient buckets. If the complete convolution support itself fits the
setup cap, an output-sensitive sparse convolution may be useful. None of
these controls supplies the generic-prime, query-sensitive R10 evaluator.

## Algorithm transplant ledger

| Candidate | Exact useful statement | Missing P1553 interface |
|---|---|---|
| Sparse nonnegative convolution | near-linear in integer-index output support | no EC scalar index; field-valued moments cancel; complete branch is two-frequency |
| Sparse polynomial multiplication | output-sensitive from explicit exponent supports | exponents are unavailable EC logs; translate pairing and target action absent |
| Multivariate multipoint evaluation | near-linear in represented coefficients plus points | the `S*L` source points or full coefficient rectangle are already represented |
| Sparse Fourier recovery | output-sensitive when transform samples and sparsity promises are available | no compact sampler for the required EC correlations and no certified Fourier sparsity |
| Full EC character transform | exact diagonalization of translations | `N=B^5` characters and nontrivial character evaluation requires scalar indexing |
| Coordinate hash plus repair | arbitrary compact buckets | repair must preserve both branches, counts, joint source, children, and fresh targets |

The comparison with sparse convolution is interface-level. It does not claim
that the cited algorithms fail on their own integer-index problems.

## Source, containment, child, and target boundary

The Fourier formula computes an aggregate scalar only when all required
transforms are supplied. It does not produce:

1. a proof that every complete image key is present;
2. one jointly coupled `(pair occurrence,fifth occurrence)` source;
3. exact counts and a source after every adaptive support restriction;
4. a compact update under a fresh target translation; or
5. the requested R10 coefficients without full support construction.

The invertible raw change `(A,Q)<->(A+Q,A-Q)` shows why storing both branch
sources restores `S*L` incidences. Applying `h` can merge those pairs, but
recovering one preimage after the merge is exactly the source-selection
problem, not a consequence of sparse sumset enumeration.

No R15 route supplies relation density, independent signed row rank,
repeated-column aggregation, factor-base logarithms, or scalar-blind descent.
The inherited conditional exponents remain

```text
lambda=max(s,1+d_loss+kappa,2)/5,
mu=max(s_m,w,2)/5.
```

## Scoped disposition

Within the frozen sparse-convolution and Fourier grammar:

1. Complete Kummer moments reduce exactly to three-point correlations on
   `(A,A+Q,A-Q)`.
2. Their exact Fourier representation has two character variables; ordinary
   one-dimensional convolution is only a boundary case.
3. Listing both raw translates is source-injective and has `S*L` entries.
4. Prime-order globally composable EC buckets are constant or injective;
   arbitrary coordinate hashes require a new charged correction law.
5. Integer output-sensitive convolution does not transfer through
   field-valued cancellation or unknown scalar indices.
6. R10 needs queried coefficients, not a potentially `B^4` full support.
7. Known-log, smooth-quotient, one-branch, and small-output controls remain
   valid, so no unrestricted convolution or circuit lower bound is claimed.

The surviving operation is now typed as

```text
a query-sensitive, list-restricted, nonhomomorphic elliptic
translation-correlation sketch for the actual complete Kummer divisor. It
must emit the R14 moment and containment package and the R10 queried
coefficients, preserve exact integer counts and one jointly coupled source,
and update under dyadic restrictions and fresh targets without scalar logs,
N character modes, S*L translate pairs, or a full convolution support.
```

No new idea, experiment, relation row, Shoup claim, or breakthrough is
authorized.

## Primary-source boundary

- Bringmann, Fischer, and Nakos,
  [Deterministic and Las Vegas Algorithms for Sparse Nonnegative Convolution](https://arxiv.org/abs/2107.07625),
  supplies near-output-linear algorithms for nonnegative integer-index
  convolution; it does not provide unknown-log EC indices or the two-frequency
  correlation oracle.
- Arnold and Roche,
  [Output-sensitive algorithms for sumset and sparse polynomial multiplication](https://arxiv.org/abs/1501.05296),
  works from explicit integer or Laurent-monomial supports; it does not retain
  the paired elliptic translates or source provenance.
- Bhargava, Ghosh, Guo, Kumar, and Umans,
  [Fast Multivariate Multipoint Evaluation Over All Finite Fields](https://arxiv.org/abs/2205.00342),
  is near-linear in represented coefficient and point input; it does not
  compress the `S*L` source grid supplied to the frozen route.
- Shoup,
  [Lower Bounds for Discrete Logarithms and Related Problems](https://www.shoup.net/papers/dlbounds1.pdf),
  is the generic-group baseline only. R15 does not claim to extend or beat it.

## Exactly one next action

Attempt one query-sensitive nonhomomorphic translation-correlation sketch on
the complete chart. It must compute the required moment families and R10's
`B` shifted coefficients from the two unary supports, certify containment,
and return one joint source under dyadic and fresh-target updates. Reject the
candidate at the first scalar-log index, globally composable bucket, `N`
character table, `S*L` translate-pair list, full-support output, field-cancelled
support heuristic, or uncharged repair/source table. In parallel on paper,
classify common components of the cross-endpoint branch fiber product; a
passing component identity is the only currently identified way to collapse
the two-frequency correlation without violating the positive controls.
