# P1553 integer-valued quotient gate R8

## Classification

- Owner: existing P1553/P1534/P1536/IDEA-049/IDEA-198 quotient and carry
  frontier; no P1554.
- Evidence: coordinator theorem screen and independent red team; no run.
- Status: `REVISE_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, rank theorem for the elliptic
  predicate, factor-log solve, scalar-blind descent, Shoup-bound improvement,
  or ECDLP breakthrough.

R7 left open a target-symbolic quotient-lattice module for the exact R6
five-label predicate. R8 tests whether falling-factorial, binomial/Newton, or
CRT/evaluation coordinates supply that module.

They do produce a real correction: an integer-valued Newton formulation can
avoid treating the ordinary monomial coefficient height of an interpolant as
an invariant obstruction. They do not compress the split finite-grid algebra
or construct the target-fresh nonlinear objects needed for exact negative and
positive answers. A new rank control from Shi's modular multiplication matrix
also proves that constant rank modulo `p` alone cannot guarantee a low-rank
canonical bounded remainder or carry. That control is not an elliptic-family
lower bound.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R7 tensorized small-root gate | `b36870eeb0c7c0a53e6d1714d623629c522f4c58cd308a072e33ea6046a06615` |
| P1553 R7 parent report | `014d7bb61d9749e313e26179c242f321f5d474be8c7af84b48ecffa0fd8c6a25` |
| P1534 quotient-kernel independent audit | `6a2c96f41552f91ab6d6ddc4801d6e4f958cf5845f6f81676de7f4db89653c53` |
| P1536 Frobenius-projector/norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |
| IDEA-198 carry-state source-unranking hypothesis | `834211bd8e26e3d421df0749653a1bc55b786aab56a698edb9cd71013a0c058a` |

Primary literature controls:

- H. Shi, "The rank of the modular multiplication matrix," arXiv:2607.10763v1,
  12 July 2026: <https://arxiv.org/abs/2607.10763>.
- T. Chinburg, B. Hemenway, N. Heninger, and Z. Scherr,
  "Cryptographic applications of capacity theory," arXiv:1605.08065:
  <https://arxiv.org/abs/1605.08065>.
- K. Ryan, "Solving Multivariate Coppersmith Problems with Known Moduli,"
  EUROCRYPT 2025 artifact: <https://artifacts.iacr.org/eurocrypt/2025/a13/>.

## Frozen interface

Let `N=p^(1+o(1))`, `B=N^(1/5)`, and let the five sign-closed restricted
x-class decks have sizes `s_i<=B`. Put

```text
Gamma = product_i {0,...,s_i-1},
D = |Gamma| = product_i s_i = Theta(B^5)
```

on full-size nodes. R6 supplies

```text
F_R(z_1,...,z_5)
  = S_6(X_1(z_1),...,X_5(z_5),x(R)) mod p.
```

The admitted query must return one bounded root or an exact no-root
certificate, recover a constant-many sign branch, and return one verified
signed x-class-labelled source. It must remain correct at every mandatory
dyadic restriction. The caps are

```text
target-independent setup and state    B^(9/4+o(1)),
fresh-target time and workspace        B^(5/4+o(1)).
```

## Newton and CRT positive control

For one label deck of size `s`, use

```text
Q(z) = (z)_(s) = product_(a=0)^(s-1) (z-a).
```

The monomial and falling-factorial bases are related by integral
unitriangular Stirling matrices. The Newton evaluation matrix

```text
E_(a,d) = binomial(a,d),  0<=a,d<s,
```

is also lower unitriangular with determinant one. Every integer-valued
function on the consecutive labels has the exact expansion

```text
f(z) = sum_(d=0)^(s-1) Delta^d f(0) binomial(z,d).
```

Because `s<=B<p`, every `d!` and every Lagrange denominator is a unit modulo
`p^m`. Binomial/Newton coordinates therefore preserve modular divisibility.
The integer-valued CRT idempotents can be written

```text
ell_a(z) = binomial(z,a) binomial(s-1-z,s-1-a),
ell_a(b) = delta_(a,b)  for 0<=a,b<s.
```

The fivefold evaluation transform is a tensor product of invertible
one-dimensional transforms. It preserves supplied CP rank and gives

```text
A_(p^m)
  = (Z/p^m Z)[z_1,...,z_5]/(Q_1,...,Q_5)
  ~= (Z/p^m Z)^Gamma.
```

This corrects an overstrong reading of R7's ordinary monomial-height gate.
An integer-valued lattice can avoid clearing all factorial denominators, so
the `B^(Omega(B))` ordinary-coefficient scale is not by itself a rejection.
If the basis is cleared back into `Z[z]`, the factorial index and its
`B^(Theta(B))` scale return and must be charged. Either way, the quotient rank
is still `D=Theta(B^5)`: evaluation diagonalizes the state but does not
compress it.

For a fixed polynomial `h`, a basis change alone also cannot prove the needed
evaluation bound. If `phi` is a basis and `W` is an invertible coefficient
transform, then

```text
|h(a)| <= ||W c(h)||_2 ||W^(-T) phi(a)||_2.
```

For diagonal `W`, `W^(-T)=W^(-1)`. The coefficient norm and the dual
evaluation norm transform together. A new short-vector theorem must therefore
specify both the integer-valued norm and the evaluation functional, not only a
smaller-looking coefficient vector.

## Exact unit-or-witness semantics

Over `F_p`, the split quotient is

```text
A_p = F_p[z_1,...,z_5]/(Q_1,...,Q_5) ~= F_p^Gamma.
```

Let `f_R=Ev(F_R)` and `Z_R={a in Gamma : f_R(a)=0}`. Then:

- `Z_R` is empty exactly when `F_R` is a unit in `A_p`; an exact negative
  certificate is an `H_R` satisfying `F_R H_R=1` in `A_p`.
- The exact zero-support projector is
  `chi_R=1-F_R^(p-1)=sum_(a in Z_R) e_a`.
- For a singleton `Z_R={a*}`, five perfect source equations exist. Define
  `H_i(a)=(a_i-a_i*)/f_R(a)` off the root and `H_i(a*)=0`. Then
  `H_i F_R=z_i-a_i*` in `A_p`.

These are branch certificates, not a uniform constructor. `F_R^(-1)` exists
only on empty fibers, while the displayed `H_i` presuppose the unknown
singleton `a*`. On a positive multi-root dyadic node neither object supplies
the branch decision. Uniform exact controls are

```text
G_R=F_R^(p-2),
chi_R=1-F_R^(p-1),
F_R G_R=1-chi_R,
F_R chi_R=0.
```

An implementation must decide `chi_R=0` versus `chi_R!=0` exactly and preserve
a positive child. Constructing these controls is exact Query2P1/source
recovery; their standard CRT representation has `B^5` coordinates. A failed
LLL search is not an empty-fiber certificate. Short pre-quotient vectors may
lie in the domain ideal; nonzero short quotient vectors may instead be
supported away from `Z_R`.

## Canonical bounded-lift rank control

Shi proves that the real rank of the `p` by `p` least-residue modular
multiplication matrix, and of its zero-row/zero-column deletion, is

```text
floor((p-1)/2) + k(p) = (p+1)/2
```

for an odd prime `p`, where `k(p)=1` counts the proper positive divisors.
Since `B=p^(1/5+o(1))`, for large `p` there is a nonsingular `B` by `B`
minor

```text
R_(i,j) = least_residue_p(a_i b_j).
```

Modulo `p`, this minor is the outer product `(a_i)(b_j)` and has rank one.
Over the integers let

```text
U_(i,j)=a_i b_j,
K=(U-R)/p.
```

Then `rank_Q(U)=1`, `rank_Q(R)=B`, and

```text
rank_Q(K) >= rank_Q(R)-rank_Q(U) >= B-1.
```

Let `T=R tensor 1 tensor 1 tensor 1`. Its flattening across mode one versus
modes two through five has rank `B`, while a rank-`B` matrix decomposition of
`R` gives a `B`-term CP decomposition. Hence

```text
rank_(CP,Q)(T)=B.
```

The analogous carry tensor has CP rank at least `B-1`. Modulo `p`, `T` has CP
rank one. Invertible factorwise Q-linear Newton transforms applied before
modular reduction preserve these flattening ranks. A CRT transform followed
by canonical modular recentering is nonlinear over Q and is not covered.

This is an exact falsification control for the universal inference

```text
constant modular CP rank
  => low-rank canonical bounded remainder and carry.
```

Any exact CP representation of this synthetic least-nonnegative control has
`A>=B`. Substitution into R7's specific `B^2 A^2` setup and `B A^2` online
pairwise model gives `B^(4+o(1))` and `B^(3+o(1))`. These are imported
implementation costs, not intrinsic lower bounds: the explicit minor has
`B^2` entries and its norm can be contracted directly in `B^(2+o(1))`
arithmetic.

The scope is narrow. Shi supplies no rank theorem for the centered R6 elliptic
tensors, and does not prove that an actual R6/S_6 deck contains this minor. The
control does not rule out another centered or specially chosen `O(p)`-width
lift, an elliptic-specific cancellation, shared-correlation algorithms, or a
non-CP implicit module. It therefore cannot close the elliptic branch or
support a generic-group or Shoup lower-bound claim.

## First missing object

After the Newton correction, the first standard over-budget target-fresh
objects are the bounded remainders and carries

```text
R_(k,R) = ctr_(p^k)(tilde(F_R)^k),
K_(k,R) = (tilde(F_R)^k-R_(k,R))/p^k.
```

The modulus must be `p^k`, not `p`. Explicit evaluation costs `B^5`; constant
modular rank supplies neither a cap-sized construction nor a rank bound for
these nonlinear objects. A valid exception remains: an actual elliptic-family
factorization or cancellation module with fully charged integer heights, LLL
bit costs, and an exact unit-or-zero-divisor interface. Under R7's naive
pairwise CP-Gram implementation its total represented rank must satisfy
`A<=B^(1/8+o(1))`; another implementation must prove the frozen direct caps.

Ryan's 2025 methods optimize explicit multivariate shift spaces. The smaller
graph-selected alternatives and multivariate recovery remain heuristic; the
artifact does not provide the missing tensor-native exact inverse or source
locator. The univariate capacity-theory result for binomial auxiliaries is a
useful warning but is not promoted to a multivariate impossibility theorem.

## Campaign consequence

For setup/state exponents `s,s_m`, query/workspace exponents `kappa,w`,
constant relation density, and `Theta(B)` independent rows, the conditional
ECDLP exponents are

```text
lambda = max(s,1+kappa,2)/5,
mu     = max(s_m,w,2)/5.
```

At the full allowed caps, both can be `0.45`. The narrower `mu=0.40` ledger
requires persistent module and pair state to stay at `B^2`; it does not follow
from the `B^(9/4)` state cap. Standard `B^3` source recovery gives `B^4`
relation collection and `B^3` descent, above rho.

No current object supplies constant verified relation density, `Theta(B)`
independent rows, factor-log completion, or identical scalar-blind descent.

## Deduplication

- IDEA-049 owns bounded integral transducers and small-root source extraction.
- IDEA-198 owns carry-state encoding and source-unranking failure.
- P1534 owns the exact quotient kernel and primitive-idempotent source issue.
- P1536 owns the constant-rank `S_6` predicate versus projector/norm-jet
  distinction.
- P1553 owns exact restricted Query2P1, sign filtering, and source replay.

R8 is a basis-and-lift audit inside these owners. It receives no new idea ID,
P1554, experiment, fixture, run, or breakthrough claim.

## Disposition

```text
REVISE_SCOPED_CONTROL__NEWTON_INTEGER_VALUED_AND_CRT_IDENTITIES_EXACT__ORDINARY_HEIGHT_NOT_INVARIANT__EMPTY_IFF_UNIT__UNIFORM_PSEUDOINVERSE_PROJECTOR_IDENTITIES_EXACT__SHI_GIVES_AN_ARBITRARY_B_BY_B_LEAST_NONNEGATIVE_MINOR_WITH_Q_RANK_B_MOD_P_RANK_ONE_AND_RAW_CARRY_RANK_AT_LEAST_B_MINUS_1__FIVE_MODE_SYNTHETIC_CONTROL_HAS_CP_RANK_B__NEWTON_LINEAR_TRANSFORMS_PRESERVE_Q_FLATTENING_ONLY_BEFORE_RECENTERING__CRT_RECENTERING_NOT_RANK_PRESERVING__A_GE_B_EXACT_FOR_CONTROL__B4_B3_ONLY_IMPORTED_NAIVE_CP_COSTS__NO_INTRINSIC_RUNTIME_LOWER_BOUND__CENTERED_ELLIPTIC_REMAINDER_UNRESOLVED__INVERSE_AND_SINGLETON_MULTIPLIERS_BRANCH_INCOMPLETE__UNIFORM_MULTIROOT_UNIT_OR_ZERO_DIVISOR_ROUTER_UNSUPPLIED__STANDARD_RECOVERY_B5_OR_B3__NO_P1554__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

For every frozen R6 dyadic subbox and fresh known-log or blind-masked target,
prove or refute one branch-complete representation of `G_R=F_R^(p-2)` and
`chi_R=1-F_R^(p-1)`, or an explicitly equivalent unit-or-zero-divisor
interface, that certifies `chi_R=0` on empty fibers, preserves a positive child
when `chi_R!=0`, and after `O(log B)` restrictions returns one verified signed
occurrence-labelled source with repeated columns aggregated. Charge
target-independent construction and state, target specialization, every
failed child, exact zero testing, output, integer heights, and Gram/LLL bit
complexity; use `p^k` recentering only for lattice powers actually invoked.
Under R7's naive CP-Gram route require `A<=B^(1/8+o(1))`; otherwise prove the
direct `B^(9/4)` setup/state and `B^(5/4)` online/workspace bounds. Any
flattening obstruction closes only the explicitly named lift and
representation.
