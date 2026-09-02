# Independent P1553 factorized-pullback gate R10 red-team transcript

Reviewer: `019f7ca1-e1be-7941-97be-290005295446`
Record: coordinator transcription of the independent theorem-only response
Evidence: read-only review; no run

## Terminal verdict

```text
ACCEPT__EXACT_MOMENT_SIMPLEX_IDENTITY__UNIVERSAL_P_BY_P_IDENTITY_MINOR_ONLY__ACTUAL_S6_IMAGE_REMAINS_OPEN__RANK_AT_MOST_ONE_CONTROL_PASSES__RANK_TWO_REQUIRES_4_POW_5_ZERO_STRATA__ALL_NONZERO_RANK_TWO_IS_SPARSE_MULTIPLICATIVE_CONVOLUTION__CANONICAL_DYADIC_PAIR_STATE_B2__STANDARD_MOMENT_TRIPLE_CHARACTER_MPE_AND_EXPLICIT_RATIONAL_ROUTES_OVER_CAP__NO_GENERAL_CIRCUIT_LOWER_BOUND__NO_NEW_IDEA__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

The first draft required corrections to its zero-stratum partition, rank-zero
branch, dense-character count, campaign baseline, and cost-variable names.
The producer applied all corrections. The reviewer matched the final producer
hashes and returned `ACCEPT` with no remaining exact defect.

## Accepted moment theorem

For

```text
Delta_(p-1,r)={alpha in N^r: |alpha|=p-1},
b_alpha=multinomial(p-1;alpha) mod p,
S_(i,I_i)(alpha)
  = sum_(a_i in I_i) product_rho q_(rho,i)(a_i)^(alpha_rho),
```

the exact restricted projector count is

```text
C_(R,I)
  = product_i |I_i|
    - sum_(alpha in Delta_(p-1,r))
        b_alpha c(R)^alpha product_i S_(i,I_i)(alpha).
```

Every multinomial scalar is nonzero modulo `p`, because all factorial
arguments are below `p`. A zero specialized coefficient can kill a complete
channel, but it does not make the multinomial scalar zero. Literal construction
uses

```text
M=binomial(p+r-2,r-1)=B^(5(r-1)+o(1))
```

channels for fixed `r>=2`. This closes only the literal moment-simplex
representation; quotient coincidences and cancellations remain possible.

## Accepted universal obstruction

The full kernel `K(u,v)=1[u dot v=0]` on `F_p^r`, `r>=2`, contains the
identity submatrix indexed by

```text
u_t=(1,t,0,...,0),
v_s=(-s,1,0,...,0).
```

Therefore every separated identity valid on the full ambient domain has at
least `p` terms. The theorem does not transfer automatically to the restricted
nonlinear pair and target-triple images of the actual `S_6` predicate, and it
is not an arithmetic-circuit lower bound.

## Accepted low-rank controls

If target specialization leaves at most one nonzero separated term, unary
zero counts give the exact restricted integer count within `B^(1+o(1))`
setup/state and `B^(o(1))` replay. The `c_R=0` case is rank zero, not rank one:
the direct count is `product_i |I_i|`. On a full-size actual R9 box it is
incompatible with the `32 B^4` root bound when that product exceeds the bound.

With two nonzero coefficients, each coordinate has four disjoint states:

```text
(x_i,y_i)=(0,0), (0,nonzero), (nonzero,0), (nonzero,nonzero).
```

The `4^5` status patterns are handled before division. An off-ratio pattern
is a root exactly when both `product_i x_i` and `product_i y_i` vanish. On the
all-nonzero stratum,

```text
F_R=0  iff  product_i lambda_i=t_R.
```

This is exact sparse multiplicative convolution of two occurrence-counted
dyadic pair histograms, sampled at `B` target-scaled points. Each pair belongs
to `O(log^2 B)` canonical dyadic ancestor rectangles, so the total occurrence
state is `B^(2+o(1))`. The statement covers the frozen canonical dyadic trees,
not arbitrary subsets or arbitrary intervals.

No cap-sized exact sparse-convolution evaluator is supplied. Direct evaluation
costs `B^3`; a dense multiplicative-character transform uses
`p-1=B^(5+o(1))` modes.

## Accepted route and campaign scope

The named literal moment, explicit pair/triple, standard polynomial
partitioning, explicit matrix, explicit pole-deck/rational-GF, represented
multivariate-evaluation, dense-character, and natural tensor contractions all
expose a charged object above a local cap. The claims reject those named
representations only. They do not reject compressed rational circuits, the
actual-image pullback, or all implicit arithmetic circuits.

An explicit batch of `B` known targets uses `B^(3+o(1))` time and `B^3` state
in the direct sort/hash implementation. This is better than `B^4` independent
per-target work but remains over cap. Single-target blind descent remains
`B^3`.

For setup `B^s`, state `B^s_m`, fresh-target work `B^kappa`, fresh-target
workspace `B^w`, and reciprocal accepted-density loss `B^d`, the favorable
conditional campaign bookkeeping is

```text
lambda = max(s,1+d+kappa,2)/5,
mu     = max(s_m,w,2)/5.
```

The constant-density control sets `d=0`. One independent signed row per
accepted target, relation density, matrix rank, factor logarithms, and blind
descent remain assumptions or unsupplied stages, not results.

## Ownership

P1515 owns explicit/batch indexing and the nonlinear implicit-semijoin
residual. P1534 owns balanced source semantics. P1536 owns projector moments
and the standard controls. P1553 owns exact dyadic counts, signs, and source
replay. The determinant-value audit already owns fixed-target no-wrap
counting, while R9 owns its dyadic assumptions. The named IDEA entries retain
their residue, representation, quotient, carry, Frobenius, and dynamic-source
variants. R10 creates no new idea ID.

## Exactly one next action

Prove or refute exact, mask-compatible summation of
`h_I(V_R(a_3,a_4,a_5))` in one frozen factorized-circuit grammar for the
actual separated `S_6` signatures, on the full box and every adaptive dyadic
child, without emitting `B^3` values or source incidences. Require the grammar
to express the exact rank-two sparse multiplicative-convolution control;
handle all zero signatures; preserve integer multiplicities and occurrence
backpointers; enforce `B^(9/4+o(1))` setup/state and `B^(5/4+o(1))`
target time/workspace; and charge replay, signs, verification, columns,
density, rank, factor logs, and identical blind descent. A negative closes
only the frozen grammar, while a positive count remains model-bound until the
complete ECDLP path is proved.
