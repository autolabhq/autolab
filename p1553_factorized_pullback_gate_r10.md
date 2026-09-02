# P1553 factorized pullback gate R10

## Classification

- Owner: existing P1515/P1534/P1536/P1553/IDEA-089/IDEA-195/
  IDEA-197/IDEA-198/IDEA-250/IDEA-266 exact-query frontier; no P1554.
- Evidence: coordinator theorem screen and independent red team; no run.
- Status: `REVISE_SCOPED_THEOREM`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, complete index calculus,
  factor-log solve, blind descent, generic lower-bound violation, or ECDLP
  breakthrough.

R9 reduced one-source routing to exact restricted root counting. R10 freezes
the separated predicate and asks whether that scalar count can be contracted
without emitting the `B^3` triple deck. It gives a coefficient-complete
moment formula, a universal separated-feature obstruction, an exact rank-at-most-one
passing control, and a sharper rank-two multiplicative-convolution control.

None of those statements closes the actual Semaev image. The remaining object
is the nonlinear pullback of a pair-side orthogonality count through a
factorized three-list map. Constructing or excluding that pullback inside the
caps remains the research problem.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R9 projector-trace gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| P1553 R9 parent report | `40ef9120033e21a16cd256fa5d7a3e4a3f035ef00f6f4fed5601246a477446f5` |
| P1553 R9 independent red team | `14178e88bb4356075302d49e93ecfa103c69cf52f46d9b91e231373e4b132bc4` |
| P1515 R1-R11 independent audit | `7e7609716f87b1b4df5ffc77406a912ad0303cc309ec1b84be42ebcc0d09539e` |
| P1553 determinant-value channel audit | `5073e39388792ea9cd8a4f7a1fe19f33f2799e59aa85f898148c9712bb963669` |
| P1536 Frobenius-projector/norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |

## Frozen interface

Let `N=p^(1+o(1))`, `B=N^(1/5)`, and retain the R9 assumptions, occurrence
labels, dyadic boxes, sign closure, finite-size threshold, and exact source
replay. For a target `R`, freeze a separated presentation

```text
F_R(a_1,...,a_5)
  = sum_(rho=1)^r c_rho(R) product_(i=1)^5 q_(rho,i)(a_i)
  = <U(a_1,a_2), V_R(a_3,a_4,a_5)>.
```

The two vectors have coordinates

```text
U_rho(a_1,a_2) = q_(rho,1)(a_1) q_(rho,2)(a_2),
V_(R,rho)(a_3,a_4,a_5)
  = c_rho(R) product_(i=3)^5 q_(rho,i)(a_i).
```

For `I=I_1 x ... x I_5`, the required scalar is

```text
C_(R,I) = sum_(a in I) (1-F_R(a)^(p-1)) mod p.
```

Under R9's checked bound `32 B^4<p`, its canonical representative is the
exact occurrence-labelled root count. Zero pair or triple signature vectors
must be counted before any projective normalization; projectivizing zero
silently deletes legal orthogonality incidences.

The caps remain

```text
target-independent setup and state    B^(9/4+o(1)),
fresh-target time and workspace        B^(5/4+o(1)).
```

## Coefficient-complete moment contraction

Let

```text
Delta_(p-1,r) = {alpha in N^r : |alpha|=p-1},
b_alpha = multinomial(p-1; alpha) mod p,
S_(i,I_i)(alpha)
  = sum_(a_i in I_i) product_(rho=1)^r
      q_(rho,i)(a_i)^(alpha_rho).
```

The multinomial theorem gives the exact identity

```text
C_(R,I)
  = product_i |I_i|
    - sum_(alpha in Delta_(p-1,r))
        b_alpha c(R)^alpha product_(i=1)^5 S_(i,I_i)(alpha).
```

Every `b_alpha` is nonzero modulo `p`: in characteristic `p`, all factorials
appearing here have arguments below `p`. The represented channel count is

```text
M = |Delta_(p-1,r)| = binomial(p+r-2,r-1).
```

For fixed `r>=2`, `M=p^(r-1+o(1))=B^(5(r-1)+o(1))`, so literal
moment-simplex construction is over cap. This is a complete exact formula,
not a rank or arithmetic-circuit lower bound. Distinct composition terms may
coincide after restricting the unary functions, and quotient reduction may
cancel them.

## Universal orthogonality obstruction

For `r>=2`, define the full-domain kernel

```text
K(u,v) = 1[u dot v=0],    u,v in F_p^r.
```

Choose

```text
u_t=(1,t,0,...,0),
v_s=(-s,1,0,...,0),       s,t in F_p.
```

Then `u_t dot v_s=t-s`, so the resulting `p x p` submatrix is the identity.
Consequently any separated identity valid for every `u,v in F_p^r`,

```text
K(u,v) = sum_(eta=1)^A f_eta(u) g_eta(v),
```

must have `A>=p=B^(5+o(1))` over any field. This excludes a universal exact
Fermat or separated-feature representation within the state cap.

The theorem does not apply merely because the actual predicate is an inner
product. The pair signatures `U(I_1 x I_2)` and target-dependent triple
signatures `V_R(I_3 x I_4 x I_5)` can lie on much smaller nonlinear images.
No identity minor on the full ambient space is a lower bound for their
restricted pullback.

## Exact rank-at-most-one passing control

If target specialization leaves at most one nonzero separated term,

```text
F_R(a) = c_R product_i q_i(a_i),
```

then for `c_R!=0`,

```text
C_(R,I)
  = product_i |I_i|
    - product_i #{a_i in I_i : q_i(a_i)!=0}.
```

If `c_R=0`, the predicate has rank zero, every tuple is a root, and its exact
count is known directly from the integer box sizes; a field trace must not be
used to decode a box size that can exceed `p`. On an actual full-size R9 box,
this branch is incompatible with the `32 B^4` geometric root bound once
`product_i |I_i|>32 B^4`; it can survive only on sufficiently restricted
boxes. Unary zero-count dyadic trees and one first-zero pointer per node use
`B^(1+o(1))` setup/state and `B^(o(1))` online replay. This is a genuine
passing boundary for the frozen grammar. It is not evidence that the
nonidentity `S_6` predicate has effective rank at most one.

## Exact rank-two control

Suppose at most two candidate terms remain. If either coefficient vanishes,
use the rank-at-most-one control. Henceforth assume `c_1(R)c_2(R)!=0`. On the
all-nonzero coordinate stratum put

```text
x_i=q_(1,i),     y_i=q_(2,i),
lambda_i=x_i/y_i,
t_R=-c_2(R)/c_1(R).
```

Every unary zero pattern is handled before division. For each coordinate the
four disjoint states are

```text
(x_i,y_i)=(0,0), (0,nonzero), (nonzero,0), (nonzero,nonzero).
```

Thus `4^5` constant status patterns suffice. Off the all-nonzero stratum, a
pattern contributes exactly when both `product_i x_i` and `product_i y_i`
vanish; its multiplicity is a product of unary status counts. On the
all-nonzero stratum,

```text
F_R(a)=0  iff  product_(i=1)^5 lambda_i(a_i)=t_R.
```

For a canonical dyadic pair rectangle, restricted to labels with
`x_i y_i!=0`, define the exact multiplicity histogram

```text
D_(12,I)(s)
  = #{(a_1,a_2) in I_1 x I_2 : lambda_1(a_1)lambda_2(a_2)=s}.
```

Store an occurrence backpointer for every nonempty bucket. Then

```text
C^x_(R,I)
  = sum_(a_3,a_4,a_5)
      D_(12,I)(t_R/(lambda_3 lambda_4 lambda_5)).
```

Equivalently, with the analogous `D_(34,I)` restricted to labels with
`x_i y_i!=0`,

```text
C^x_(R,I)
  = sum_(a_5)
      (D_(12,I) *_mult D_(34,I))(t_R/lambda_5(a_5)),
```

where `*_mult` is multiplicative convolution on `F_p^*`. Every occurrence
pair belongs to only `O(log^2 B)` dyadic ancestor rectangles, so explicit
pair histograms for all dyadic pair rectangles occupy `B^(2+o(1))` entries.

This reduction is exact and source-faithful, but the currently supplied exact
evaluators still cost either `B^3` direct work or
`p-1=B^(5+o(1))` multiplicative-character modes. Any proposed sparse
convolution index must preserve integer
multiplicities, zero strata, queried-rectangle identity, and an occurrence
backpointer. R10 neither supplies such an index nor proves it impossible.

## Route audit

| Route | First charged object | Scoped disposition |
|---|---:|---|
| Literal Fermat moments | `M=binomial(p+r-2,r-1)` channels | exact and coefficient complete; over cap for fixed `r>=2` |
| Explicit pair plus triple scan | `B^2` pair entries and `B^3` triple incidences | exact standard route; online over cap |
| Polynomial partitioning | standard implementation still emits `B^3` triple points | no cap-sized exact pullback supplied |
| Matrix multiplication | `B^3` triple rows or `B^5` pair-triple outputs | output/input materialization over cap |
| Named explicit pole-deck/rational-GF expansion | `B^3` pole deck or `M` coefficient simplex | the inherited exact expansion restores an over-cap object; compressed rational circuits remain open |
| Multivariate multipoint evaluation | represented `M` coefficients and `B^3` points | the input representation is already over cap |
| Exact character/FFT convolution | `p-1=B^(5+o(1))` multiplicative-character modes | exact dense route over cap |
| Natural tensor-network cuts | bond dimension `p`, `M`, or explicit `B^3` deck | named contractions over cap; no general circuit bound |
| Current kSUM indexing bounds | preprocessing/space outside `B^(9/4)` for the needed exact batch | no instantiation of the required regime |

The multivariate multipoint-evaluation comparison uses Bhargava et al.,
[Fast Multivariate Multipoint Evaluation Over All Finite Fields](https://arxiv.org/abs/2205.00342): its fast bound starts from an explicitly represented
coefficient vector and evaluation points. The indexing comparison uses the
parameterized theorems of Dinur and Golovnev,
[Improved Time-Space Tradeoffs for 3SUM-Indexing](https://arxiv.org/abs/2512.04258),
and the preprocessed-universe model of Kirkpatrick et al.,
[Preprocessed 3SUM for Unknown Universes with Subquadratic Space](https://arxiv.org/abs/2602.11363).
Those results do not supply the required `B^(9/4)`-setup, `B^(5/4)`-query
source-faithful evaluator. This is a parameter mismatch, not a lower bound.

## Surviving actual-image object

For a pair rectangle define

```text
h_I(v) = sum_(x in I_1 x I_2) delta_0(U(x) dot v).
```

The exact restricted count is

```text
C_(R,I)
  = sum_(y in I_3 x I_4 x I_5) h_I(V_R(y)),
```

with zero signatures and the R9 integer no-wrap condition handled explicitly.
The live exception is therefore not ambient orthogonality. It is whether the
nonlinear pullback `h_I o V_R` can be represented and summed directly from
the three unary lists without materializing its `B^3` values or source
incidences.

A legitimate negative result must bind a frozen factorized-circuit grammar
and prove its lower bound on the actual signature image. A legitimate positive
result must return exact full-box and queried-child counts, preserve source
backpointers through the leaf, and meet both caps. The rank-two convolution is
a mandatory control: a grammar that cannot express it is too weak to close
the general route.

## Complete campaign accounting

One target count is not an ECDLP algorithm. Any promoted construction must
also charge:

```text
relation collection and exact signed row density,
independent-column rank and repeated-column aggregation,
factor-base logarithm solve,
one-source dyadic replay including zero children,
all source signs and complete projective verification,
identical scalar-blind target descent,
field and bit complexity, persistent state, and target workspace.
```

An explicit batch of `B` known targets can materialize `B^3` triple records
and `B^3` target-minus-pair records, giving `B^(3+o(1))=N^(0.6+o(1))` time
and `B^3` state in the direct sort/hash implementation. This improves the
per-target `B^4=N^0.8` relation baseline but remains over the local caps;
single-target blind descent remains `B^3=N^0.6`.

For a hypothetical accepted count backend, write its setup work as `B^s`,
persistent state as `B^s_m`, fresh-target work as `B^kappa`, fresh-target
workspace as `B^w`, and reciprocal accepted-density loss as `B^d`.
Conditional on one independent signed row per accepted target, the favorable
full campaign accounting is

```text
lambda = max(s,1+d+kappa,2)/5,
mu     = max(s_m,w,2)/5.
```

The constant-density control sets `d=0`. These formulas are conditional
bookkeeping, not proved density or rank.
Meeting the local count caps would still require the full path above before
any generic-ECDLP complexity statement.

## Deduplication

- P1515 owns the explicit `2|3` pair/triple route, target batch/indexing
  variants, and the factorized-semijoin/nonlinear implicit-batch residual.
- P1534 owns quotient-kernel and balanced source extraction semantics.
- P1536 owns projector moments and the standard `B^5`/`B^3` controls.
- P1553 owns exact counts, Query2P1, dyadic replay, signs, and source labels.
- The P1553 determinant-value audit already established that a fixed-target
  root count is below `p`; R9 added the dyadic multiplicity assumptions and
  replay theorem, so R10 allocates no new owner for no-wrap counting.
- IDEA-089/195/197/198/250/266 own the related idempotent, representation,
  quotient, carry, Frobenius, and dynamic-source-tree hypotheses.

R10 allocates no new idea ID. Its moment identity, ambient identity-minor
obstruction, and rank-at-most-one/rank-two controls are receipts under the existing
owners.

## Exactly one next action

Prove or refute exact summation of `h_I(V_R(a_3,a_4,a_5))` in one frozen
factorized-circuit grammar for the actual separated `S_6` signatures, for the
full box and every adaptively queried dyadic child, without emitting `B^3`
values or source incidences. Require the grammar to express the exact rank-two
sparse multiplicative-convolution control; separate zero signatures; preserve
integer multiplicities and occurrence backpointers; and enforce
`B^(9/4+o(1))` setup/state plus `B^(5/4+o(1))` target time/workspace. Charge
replay, signs, verification, columns, density, rank, factor logs, and identical
blind descent. A negative result closes only that frozen grammar; a positive
count result remains model-bound until the complete ECDLP path is proved.
