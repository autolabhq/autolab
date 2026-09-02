# P1553 Cartesian Kummer rigidity gate R11

## Classification

- Owner: existing P1553/P1515/IDEA-001/IDEA-057/IDEA-195 endpoint-router
  and list-specific intertwiner frontier; no P1554.
- Evidence: coordinator theorem derivation and independent theorem sidecars;
  no run.
- Status: `REVISE_SCOPED_THEOREM`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, factor-log solve, blind descent,
  generic lower-bound violation, or ECDLP breakthrough.

R10 left the nonlinear pullback of the exact pair-side count through the
factorized `S_6` triple map. R11 first removes a semantic ambiguity: on
sign-closed x-class decks, exact existence can be counted directly as a
signed endpoint coefficient. That scalar is not the same multiplicity as the
Fermat-projector x-root count, but it has the same zero set, remains below
`p`, and supports the same dyadic source replay.

The endpoint coefficient is already the P1553/P1551 `2|3` residual; changing
from the projector to signed convolution does not construct it. R11 then
closes one actual-image subgrammar left by R10 and IDEA-057: every
branch-oblivious rational Kummer label of degree `d=o(B)` fails to merge even
two distinct Kummer source classes across a full Cartesian second deck. A
degree `Theta(B)` list-specific map, correction-carrying map, or non-Cartesian
recursive support remains open.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| P1553 R10 parent report | `acf820dac41bded67af083931e4284702cdf75d5c9fd8e121c21dc57bfc3bf97` |
| P1553 R10 independent red team | `3a810ae62318ddac77edcacf364e28454473919e956cce488352d1511e484670` |
| P1553 R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| P1551 finite-domain selector circuit gate | `5f1bd9c12ca700074c9cd327f6539bc880ec60b27431dc5f34e23b0a12f6c68f` |
| IDEA-057 prime-order composable-bucket theorem | `524a59c1728bcbea804ac4be42ace5a965b68a6332e85d941829b89e04fc4225` |
| IDEA-057 auxiliary-isogeny router gate | `f8abb802b5052f614a6500c722083d3ebbd45d6e54353686ff3283ffa27a88b7` |
| IDEA-057 list-restricted branch-locus gate | `e4972a1c6f4fb796a9efa556c745c4d4d5e4b00e5637e9fca9f3828abb4db120` |
| IDEA-001 exact spectral rank/density gate | `e572713a3910ef6a3e31ac360123aa8b5135c75d4210cbfb579e6831e9746fca` |
| P1515 R1-R11 independent audit | `7e7609716f87b1b4df5ffc77406a912ad0303cc309ec1b84be42ebcc0d09539e` |

## Frozen endpoint interface

Let `G=<P>` have odd prime order `N=p^(1+o(1))` and put
`B=N^(1/5)`. For color `i`, a canonical dyadic x-class deck `I_i` stores a
representative `P_(i,a)`, its global factor-base column, and its occurrence
backpointer. Require `P_(i,a)!=O`, no repeated x-class within one color, and
`R!=O` for comparison with the `S_6` projector. A global x-class may repeat
across colors, but every cross-color copy remains a separate occurrence until
verified relation columns are aggregated. Define the sign-closed occurrence
measure in the integral group algebra

```text
U_(i,I_i) = sum_(a in I_i) ([P_(i,a)]+[-P_(i,a)]).
```

The exact signed count for target `R` is

```text
M_(R,I) = coeff_[-R] product_(i=1)^5 U_(i,I_i).
```

It counts occurrence labels and normalized source-sign vectors satisfying

```text
R + sum_i sigma_i P_(i,a_i) = O.
```

Let `z_(R,I)` be the R9 projector count of x-class tuples for which at least
one normalized sign vector exists. The two counts are different, but in the
odd prime subgroup

```text
z_(R,I) <= M_(R,I) <= 16 z_(R,I),
M_(R,I) <= 16 min_j product_(i!=j)|I_i|.
```

For a fixed x-tuple, the first four signs determine the required fifth signed
point, so at most `2^4=16` normalized sign vectors contribute. Conversely,
every projector root has at least one such vector. The frozen R9 condition
`32 B^4<p` therefore prevents signed-count wrap as well as x-root-count wrap.

If color `j` has maximum within-color x-class multiplicity `kappa_j`, replace
the last bound by

```text
M_(R,I) <= 16 min_j kappa_j product_(i!=j)|I_i|
```

and recheck the finite-size no-wrap threshold. The rest of R11 uses the
within-color-unique interface.

Exact signed counts are additive under a dyadic split. Query the full box,
then one child at each level; a zero child sends replay to its positive
complement. At most

```text
1 + sum_i ceil(log_2 |I_i|) <= 1+5 ceil(log_2 B)
```

calls isolate one x-class occurrence tuple. Enumerate its `2^5` signs, verify
with complete projective addition, and only then aggregate repeated global
columns. Thus `M_(R,I)` is a branch-complete alternative scalar, not a faster
constructor.

## Actual-image convolution normal form

Put

```text
D_12 = U_(1,I_1)*U_(2,I_2),
D_345 = U_(3,I_3)*U_(4,I_4)*U_(5,I_5).
```

Then

```text
M_(R,I)
  = sum_(s in G) D_12(s) D_345(-R-s).
```

This is the actual signed `2|3` pullback. The kernel

```text
K_R(s,t)=1[s+t=-R],    s,t in G,
```

is an `N x N` permutation matrix and has rank exactly `N` over every
coefficient field. Pulling it back through full pair and triple addition gives
the existing endpoint-support rank theorem. On finite decks the rank is only
the number of matched attained endpoints, so this full-domain identity does
not lower-bound a nonlinear list-specific circuit.

The signed convolution, permutation rank, and source replay are already owned
by P1551/P1553 and IDEA-001. R11 allocates no new owner for them. In
particular, replacing the R10 Fermat projector by this endpoint coefficient
returns to the exact Query2P1 residual rather than solving it.

R10's effective-rank-two multiplicative-convolution control remains valid and
must be expressible by any proposed general pullback grammar. The theorem
below removes only rational branch-oblivious Kummer compression; it is not a
negative result for the rank-two control or every arithmetic circuit.

## Rational Kummer branch model

Work over a field of characteristic different from `2` and `3`. Let

```text
x:E -> P^1
```

be the Kummer quotient and let `psi:P^1->P^1` be a nonconstant rational map
of degree `d`. Write

```text
f = psi o x,       deg(f)=2d.
```

For a signed point `A in G` and an auxiliary signed point `Q`, the complete
unordered branch image is the effective degree-two divisor

```text
B_A(Q) = [f(A+Q)] + [f(A-Q)]  on P^1.
```

In an affine target chart away from its poles, it is determined by

```text
T_A(Q) = f(A+Q)+f(A-Q),
V_A(Q) = f(A+Q)f(A-Q).
```

A branch-oblivious merger of distinct Kummer classes `x(A)` and `x(A')`
requires

```text
psi(x(A))=psi(x(A')),
B_A(Q)=B_(A')(Q)
```

for every admitted second-list point `Q`. Equality of only the trace is not
enough; both coefficients, multiplicity, poles, and exceptional branches are
part of the frozen grammar.

## Finite-Cartesian rigidity theorem

Assume

```text
A' != A and A' != -A,
2d < N.
```

Each of the four translated functions

```text
f(A+Q), f(A-Q), f(A'+Q), f(A'-Q)
```

has a pole divisor of degree `2d`. Hence at most `8d` distinct `Q` values lie
in their combined pole support. Away from those values, put

```text
Delta_T = T_A-T_(A'),
Delta_V = V_A-V_(A').
```

The pole divisor of either nonzero rational function has degree at most `8d`.
It therefore has at most `8d` distinct zeros. If complete branch equality
holds at more than `8d` regular `Q` values, both differences vanish as
rational functions.

In the function field `k(E)`, equality of the two monic branch polynomials

```text
(Z-f(A+Q))(Z-f(A-Q))
  = (Z-f(A'+Q))(Z-f(A'-Q))
```

and unique factorization into linear factors imply one of

```text
f(A+Q)=f(A'+Q),
f(A+Q)=f(A'-Q)
```

identically in `Q`. In the swapped case, with `X=A+Q`, evenness of
`f=psi o x` gives

```text
f(X)=f(A+A'-X)=f(X-(A+A')).
```

Thus `f` is invariant under translation by one of the nonzero points `A-A'`
or `A+A'`. That point has order `N`. The invariant map factors through the
degree-`N` quotient by the generated subgroup, so `N` divides `deg(f)` and in
particular forces

```text
deg(f) >= N,
```

contrary to `2d<N`.

Therefore a pair of distinct Kummer classes can have equal complete branch
images at no more than

```text
16d
```

distinct second-list points when pole cases are admitted. If the deck is
certified disjoint from all four pole supports, the sharper bound is `8d`.
In particular, a Cartesian second deck of size `B>16d` admits no nontrivial
branch-oblivious merger.

This is a finite-list theorem. It strengthens IDEA-057's one-map symbolic
check to every rational `psi` with `d<B/16`, while preserving degree-linear,
correction-carrying, and non-Cartesian exceptions.

## Support-retention consequence

Partition a size-`B` first deck into `psi` fibers of sizes `m_h<=d`. Then

```text
sum_h binomial(m_h,2) <= B(d-1)/2.
```

Every colliding pair of distinct Kummer classes can share a branch-oblivious
output on at most `16d` second-list points. Hence the unconditional number of
pair-collision/second-input incidences is at most

```text
8 d(d-1) B = O(d^2 B).
```

Retaining `Theta(B^2)` genuinely compressed first-merge occurrences therefore
requires at least `d=Omega(B^(1/2))`.

The stronger `O(dB)` occurrence bound and `d=Omega(B)` requirement hold only
for one frozen indivisible state per entire `psi` fiber: no `Q`-dependent
subdivision, selective preimage removal, or correction may distinguish fiber
members. A correction can restore exactness, but it also restores source state
and lies outside that stronger branch. Neither support count is a proof of
relation density or independent row rank; later stages can only be credited
after those are separately established.

## Linear-degree boundary

The dependence on `B` cannot be deleted by a pure finite-grid identity
argument. On a cyclic subgroup with `N>4B+1`, take

```text
L={x(P),x(2P),...,x(BP)}.
```

All Kummer branches from `L x L` lie among the `O(B)` coordinates
`x(kP)` for `0<=k<=2B`. Let `Omega_aff` contain every finite affine input and
branch coordinate used on this grid and define

```text
psi_c(Z)=Z+c product_(omega in Omega_aff)(Z-omega),   c!=0.
```

Every nonconstant polynomial fixes infinity projectively, so this
degree-`O(B)` map fixes the complete finite grid and transports all of its
branches. The coordinate `x((2B+1)P)` is finite, distinct from `Omega_aff`
under `N>4B+1`, and is moved by every `c!=0`; hence the map is not a global
Kummer transporter. It is injective on the frozen input grid and supplies no
compression, but it proves that finite Cartesian transport alone cannot
globalize every degree-`Theta(B)` map.

A useful linear-degree successor must do more: merge sources, preserve a
large pair support, construct its coefficients inside setup, carry bounded
corrections through all recursive stages, and invert every accepted branch.
Interpolating a finite table without those properties is source advice, not a
router.

## Route disposition

| Route | R11 decision |
|---|---|
| Signed endpoint coefficient | exact branch-complete scalar; already P1551/P1553 Query2P1 |
| Full endpoint separation | exact rank `N`; already IDEA-001 and not a nonlinear-circuit bound |
| Global complete Kummer quotient | constant or injective on prime subgroup; already IDEA-057/P1523 |
| Any branch-oblivious rational map with `d<B/16` | finite-Cartesian merger impossible |
| Constant-degree ECFFT local maps | included in the preceding scoped negative |
| General support-changing rational merge | at most `O(d^2 B)` collision/second-input incidences; useful range starts at `d=Omega(B^(1/2))` |
| Indivisible whole-fiber state | at most `O(dB)` compressed occurrences; useful range starts at `d=Omega(B)` |
| Degree `Omega(B^(1/2))` through `O(B)` list-specific map | open; finite interpolation alone gives no compression |
| Preimage correction state | open; must charge retained source identity and branch growth |
| Non-Cartesian recursive support | open; requires a Hilbert-function or constructive theorem |
| Arbitrary nonlinear multirow or arithmetic circuit | open |

The primary structural comparison is Chalcraft and Fryers,
[Kummer structures](https://arxiv.org/abs/0806.0409), which reconstructs a
non-4-torsion Kummer structure from its group. The ECFFT comparison is
Ben-Sasson, Carmon, Kopparty, and Levit,
[Elliptic Curve Fast Fourier Transform Part I](https://arxiv.org/abs/2107.08473).
Those results do not claim the finite-list source router screened here.

## Complete campaign accounting

A linear-degree transporter or a retained non-Cartesian support is still only
a first merge. Promotion requires one typed path that charges:

```text
target-independent map and correction construction,
all retained source and branch states,
full-box and negative-child queries,
signed occurrence backpointers and complete verification,
accepted-target density and independent signed row rank,
factor-base logarithm solve,
identical scalar-blind target descent,
field and bit work, state, and target workspace.
```

The caps remain `B^(9/4+o(1))` setup/state and `B^(5/4+o(1))`
fresh-target work/workspace. No R11 theorem supplies the missing relation
campaign or changes the Pollard-rho comparison.

## Deduplication

- P1553/P1551 own the signed endpoint coefficient, Query2P1, and dyadic source
  replay.
- IDEA-001 owns exact endpoint-support separation rank and its density tradeoff.
- IDEA-057/P1523 own global composable labels, prime-order noncollapse, and the
  canonical ECFFT/Kummer branch screens.
- P1515 owns nonlinear implicit target batches and the surviving list-specific
  source router.
- IDEA-195 owns the non-Cartesian `S_3` intertwiner/source-router residual.

R11's finite-Cartesian degree bound is a versioned refinement under those
owners. It creates no new idea ID, experiment, contract, or research-status
promotion.

## Exactly one next action

Derive or refute one exact source-faithful Kummer transporter on a frozen
non-Cartesian support over the full surviving degree range
`d_psi=Omega(B^(1/2))` through `O(B)`. It must merge a positive fraction of
source classes while retaining `Theta(B^2)` first-stage pair occurrences,
express the R10 rank-two sparse-convolution control, bound every correction
and provenance state within `B^(9/4+o(1))`, answer every full-box and
adaptive-child target query within `B^(5/4+o(1))`, and return one verified
signed occurrence source. A negative result must prove a Hilbert-function,
branch-growth, or provenance lower bound for the frozen grammar; a positive
first merge remains model-bound until density, independent rank, factor logs,
and identical blind descent are complete.
