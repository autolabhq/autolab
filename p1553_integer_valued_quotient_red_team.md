# Independent P1553 integer-valued quotient R8 red-team transcript

Reviewer: `019f7ca1-e1be-7941-97be-290005295446`
Record: coordinator transcription of the independent theorem-only response
Evidence: read-only review of the final producer draft; no run

## Terminal verdict

```text
REVISE_SCOPED_CONTROL__NEWTON_INTEGER_VALUED_AND_CRT_IDENTITIES_EXACT__ORDINARY_HEIGHT_NOT_INVARIANT__EMPTY_IFF_UNIT__UNIFORM_PSEUDOINVERSE_PROJECTOR_IDENTITIES_EXACT__SHI_GIVES_AN_ARBITRARY_B_BY_B_LEAST_NONNEGATIVE_MINOR_WITH_Q_RANK_B_MOD_P_RANK_ONE_AND_RAW_CARRY_RANK_AT_LEAST_B_MINUS_1__FIVE_MODE_SYNTHETIC_CONTROL_HAS_CP_RANK_B__NEWTON_LINEAR_TRANSFORMS_PRESERVE_Q_FLATTENING_ONLY_BEFORE_RECENTERING__CRT_RECENTERING_NOT_RANK_PRESERVING__A_GE_B_EXACT_FOR_CONTROL__B4_B3_ONLY_IMPORTED_NAIVE_CP_COSTS__NO_INTRINSIC_RUNTIME_LOWER_BOUND__CENTERED_ELLIPTIC_REMAINDER_UNRESOLVED__INVERSE_AND_SINGLETON_MULTIPLIERS_BRANCH_INCOMPLETE__UNIFORM_MULTIROOT_UNIT_OR_ZERO_DIVISOR_ROUTER_UNSUPPLIED__STANDARD_RECOVERY_B5_OR_B3__NO_P1554__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Accepted basis control

On each consecutive label deck, the falling-factorial and monomial bases are
related by integral unitriangular Stirling transforms. The binomial evaluation
matrix

```text
E_(a,d)=binomial(a,d)
```

is lower unitriangular. Since every deck size is below `p`, all factorial and
Lagrange denominators are units modulo `p^m`. The integer-valued Newton and CRT
identities in R8 are therefore exact and preserve the modular value problem.

This accepts R8's correction to R7: ordinary monomial coefficient height is
not invariant under a legitimate integer-valued formulation. Clearing the
factorials back into `Z[z]` restores their scale. Neither choice compresses the
fivefold quotient, whose abstract rank is `Theta(B^5)`.

For an invertible coefficient transform `W`, the correct dual-norm inequality
is

```text
|h(a)| <= ||W c(h)||_2 ||W^(-T) phi(a)||_2.
```

Only for diagonal `W` may the inverse transpose be written as `W^(-1)`.

## Accepted exact-query control

In the split finite-grid algebra, the root fiber is empty exactly when `F_R`
is a unit. The exact identities

```text
G_R=F_R^(p-2),
chi_R=1-F_R^(p-1),
F_R G_R=1-chi_R,
F_R chi_R=0
```

give a uniform pseudoinverse and zero-support projector. They do not give a
compact constructor. An empty-fiber inverse is branch-specific, and the
displayed singleton multipliers presuppose the unknown source. A positive
multi-root dyadic node still needs an exact zero test and a sound positive
child choice.

LLL failure is not a no-root certificate. Short pre-quotient vectors may lie
in the domain ideal, while nonzero short quotient vectors may be supported
away from the zero fiber. Exact decision, useful shortness, and source recovery
remain distinct gates.

## Accepted Shi control

Shi's theorem gives real rank `(p+1)/2` for the least-nonnegative modular
multiplication matrix when `p` is an odd prime. Hence it contains a
nonsingular `B` by `B` minor `R` for the R8 asymptotic range. The same minor
has rank one modulo `p`. Relative to the raw integer outer product `U`, its
carry `K=(U-R)/p` obeys

```text
rank_Q(K) >= B-1.
```

For `T=R tensor 1 tensor 1 tensor 1`, the mode-one versus modes-two-through-
five flattening has rank `B`, and a rank-`B` matrix decomposition supplies a
`B`-term CP decomposition. Thus `rank_(CP,Q)(T)=B`; modulo `p`, its CP rank is
one. The analogous carry tensor has CP rank at least `B-1`.

Invertible factorwise Q-linear transforms before modular reduction preserve
that flattening rank. A CRT transform followed by canonical modular
recentering is nonlinear over Q and is not covered.

The control therefore rejects only the universal inference from low modular
rank to a low-rank canonical bounded lift or carry. It is synthetic: Shi gives
no rank theorem for the centered R6 elliptic tensors.

## Cost correction and remaining exception

Any exact CP representation of the synthetic least-nonnegative control has
`A>=B`. Formally substituting this into R7's pairwise
`B^2 A^2`/`B A^2` accounting gives `B^4` setup and `B^3` online work. These are
costs of that implementation, not intrinsic lower bounds. The explicit minor
has only `B^2` entries and admits direct norm contraction in `B^(2+o(1))`
arithmetic.

The live exception is a branch-complete, elliptic-specific unit-or-zero-
divisor representation. Under R7's naive CP-Gram implementation it needs
`A<=B^(1/8+o(1))`; another exact representation need only prove the direct
`B^(9/4)` setup/state and `B^(5/4)` online/workspace caps.

Standard CRT decision uses `B^5` coordinates and standard balanced source
recovery uses `B^3`. Neither relation density, independent row rank, factor
logs, nor scalar-blind descent follows from the R8 controls.

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
