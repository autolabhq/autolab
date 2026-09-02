# P1553 heavy-key source and target-support gate R36

## Classification

- Owner: existing P1553/IDEA-195 erased-image, source-replay, and
  target-density frontier; no new idea ID.
- Evidence: exact Las Vegas source-section theorem and deterministic target
  support counting; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_POSITIVE_SOURCE_NEGATIVE_DENSITY_GATE`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: a globally heavy captured key has an exact unscanned
  source in expected `O(B)` work, but the both-branch selected-support model
  reaches only `O(B^4)` targets in a group of order `B^(5+o(1))`. No current
  target router, Shoup-bound improvement, or ECDLP breakthrough follows.

R35 asked for a marked source-fiber locator over all fifth shifts. At the
root, the requested operation is easier than exact source-fiber counting:
the center value fixes a degree-`B` fiber, while a high-support key occurs on
`Omega(B)` of the `B` fifth shifts. Uniform rejection sampling therefore
returns a fully coupled unscanned source in expected `O(B)` trials.

This positive section does not rescue the candidate. The same both-branch
containment that creates the compressed image forces each target-visible
endpoint `P+q` and `P-q` into one `B^2`-point set `Z`. Combining it with the
other signed pair endpoint deck gives at most `O(B^4)` possible known-log
targets. Multiplicity above one endpoint cannot enlarge that support.

The remaining target action is a marked translated-intersection query between
two `B^2`-point sets, one represented by R31's `O(B)` translation paths. In
unstructured form this is 3SUM-Indexing at `n=B^2`. Current general algorithms
and the elementary path-block index miss the campaign parameters, but known
lower bounds do not prove a polynomial exclusion. R36 is therefore a scoped
route reduction, not a data-structure lower bound.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R13 erased-image gate | `78ff09226852f140a8c33ce5ba98106ed62007b7126673d3f3af2c49d08a2cc5` |
| R30 signed-partition packing gate | `8876faa2b65f6a44410495209617b50e99cc781caaf73a2f4cb8b359b294006f` |
| R31 near-period run compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R35 shift-slice support/density gate | `e72c406ee3997eaac86b44d757a34793d3678a129331ef7be17aa09ae05ee0de` |
| R35 bundle hash list | `b56ea2610fc5406772a4aae1f43edca8a822e26363ef4c4d16f53756b550bde2` |
| R35 staging receipt | `fbe37828749337a1a2850b53e42d695c393bd61b73861fa2934900a5a5e6a159` |

## Frozen selected-support interface

Let `G=<P_0>` have odd prime order

```text
N=B^(5+o(1)).
```

Retain R35's represented data

```text
g:E->P^1, degree(g)=c=B,
C subset P^1(k), |C|=c,
Z={P in G:g(P) in C}, |Z|=c^2,
Q subset G minus {O}, |Q|=c.
```

For `q in Q`, put

```text
H_q=Z intersect (Z-q) intersect (Z+q),
kappa_q(P)=(g(P),g(P+q),g(P-q)).                 (1)
```

For an oriented key `y=(a,b,d)`, define

```text
F_a^G={P in G:g(P)=a},
m(y)=#{(P,q):P in H_q and kappa_q(P)=y},
t(y)=#{q in Q:exists P in H_q with kappa_q(P)=y}. (2)
```

Since `g` has degree `c`, including ramification multiplicity,

```text
|F_a^G|<=c.                                       (3)
```

The R35 heavy-support dictionary targets keys satisfying

```text
t(y)>=tau*c                                       (4)
```

for a fixed `tau>0`. Every retained source keeps signed pair and fifth
occurrence backpointers, and every output is recomputed with complete group
addition before acceptance.

## Theorem 1: heavy keys have an expected-linear root source section

Given a captured oriented key `y=(a,b,d)` satisfying (4), repeat:

1. sample `P` uniformly from the stored list `F_a^G`;
2. sample `q` uniformly from all of `Q`;
3. accept exactly when `P in H_q` and `kappa_q(P)=y`.

The candidate rectangle has size at most `c^2`. Every represented shift in
`t(y)` contributes at least one source, so

```text
m(y)>=t(y)>=tau*c.                                (5)
```

One trial succeeds with probability at least `tau/c`; consequently the
expected number of trials is at most

```text
|F_a^G|*|Q|/m(y)<=c/tau=O_tau(B).                 (6)
```

Every trial uses constant-many dictionary lookups, additions, and evaluations
of stored labels. Conditioned on acceptance, the result is uniform over the
oriented source fiber. The algorithm is Las Vegas: it has no false output,
and the unordered branch quotient changes only a constant orientation factor.

R35's random slice discovery can omit a heavy key, but it cannot create a
false key. For a captured low-support key, (6) is not promised; a timeout is
an omission rather than a certificate. For a certified heavy key, the
algorithm reaches unscanned fifth labels without a `B^3` source table.

This proves a root source section only. It does not return exact `m(y)`, prove
that a key is heavy, or answer every adversarial dyadic child. R9 explicitly
distinguishes exact restricted counting from a direct empty-or-source router:
once a complete source is returned, its five occurrence labels already choose
their dyadic leaves and no replay descent is needed. A fresh-target operation
must still identify an extendible captured key before Theorem 1 can be used.

## Theorem 2: both-branch containment loses one target-density exponent

Fix one normalized sign pattern. Let

```text
u=sigma_1*A_1+sigma_2*A_2,
v=sigma_3*A_3+sigma_4*A_4,
q=sigma_5*A_5,
R=u+v+q.                                          (7)
```

The first signed pair endpoint is `P=u`. A complete selected source has

```text
w_+=P+q in Z,
w_-=P-q in Z.                                     (8)
```

Whichever fifth orientation is used by (7), its target-visible endpoint is
one of `w_+` or `w_-`. Thus every target produced by this selected stratum is

```text
R=v+w,       v in V_sigma, w in Z,                (9)
```

where `V_sigma` is the signed endpoint support of the other two lists. It has
at most `B^2` occurrence endpoints. Therefore

```text
|V_sigma+Z|<=|V_sigma|*|Z|<=B^4.                 (10)
```

There are only a constant number of normalized sign patterns. For a uniformly
random known-log target in `G`, the success probability of all compressed
both-branch sources is consequently

```text
Pr[target is covered]<=O(B^4/N)=B^(-1+o(1)).      (11)
```

This is a support bound, not an expectation under independent sources.
Multiple `(P,q)` witnesses above the same `w`, repeated complete keys, and an
exact all-shift source locator do not enlarge `V_sigma+Z`. Hence R35's sampled
fifth-deck loss is not the only density charge: even restoring every fifth
shift leaves

```text
d_loss>=1-o(1).                                   (12)
```

More generally, if the union of target-visible endpoints has size `B^rho`,
then

```text
d_loss>=max(0,3-rho)-o(1).                        (13)
```

Constant target density requires `rho>=3-o(1)`. The R30-R35 complete
containment mechanism has `rho=2`.

The theorem is scoped to rows whose useful source sign is owned by the
both-branch selected stratum. A separately constructed mixed mechanism may
use a target-visible branch outside `Z`; it must expose that branch, preserve
its complete key and source, and charge its image rather than inheriting
R30's ownership for free.

## Corollary 3: the current online cap is above rho after density

Use R10's favorable bookkeeping

```text
lambda=max(s,1+d_loss+kappa,2)/5.                 (14)
```

Even granting `s<=9/4`, one independent row per accepted target, exact root
source replay from Theorem 1, and all missing algebra, (12) gives

```text
lambda>=max(s,2+kappa,2)/5.                       (15)
```

At the inherited direct online cap `kappa=5/4`, this is

```text
lambda>=13/20=0.65.                               (16)
```

To beat rho within this selected-support route, the average fresh-target
decision must satisfy the much stronger condition

```text
kappa<1/2-o(1),                                   (17)
```

while also returning a marked key and preserving independent row density.
An `O(B)` heavy-key source replay is paid only after a successful target and
does not by itself violate (17), because only `Theta(B)` accepted rows need
replay. The target decision on `Theta(B^2)` attempted known-log targets is the
new dominant operation.

## Theorem 4: the fresh target projects to marked 3SUM-Indexing

Before checking the complete key and its source fiber, every valid target
router must solve the necessary projection

```text
given fresh R, find (v,w) in V_sigma x Z with v+w=R, or return empty. (18)
```

Both sets have size at most

```text
n=B^2.                                             (19)
```

For arbitrary stored sets in an abelian group, (18) is the standard
3SUM-Indexing query: preprocess two lists and, for a fresh challenge, return a
pair adding to it. The elementary endpoints are:

```text
store the two lists:       S=O(n),   T=O(n),
store every pair sum:      S=O(n^2), T=O(1).       (20)
```

In campaign units, these are respectively

```text
(S,T)=(B^2,B^2) and (B^4,1).                      (21)
```

Neither meets setup `B^(9/4)` and the density-adjusted query target
`B^(1/2-o(1))` simultaneously.

The 2019 Fiat--Naor-based tradeoff gives, for `0<delta<1`,

```text
S=O(n^(2-delta/3)), T=O(n^delta).                 (22)
```

Its smallest stated space in that range is `n^(5/3)=B^(10/3)`, already over
setup. The 2025 application-dependent improvement gives

```text
S=O~(n^(2.5-delta)), T=O~(n^delta), 0<=delta<=1, (23)
```

whose smallest stated space is `n^(3/2)=B^3`, also over setup. These
comparisons are controls for the unstructured projection, not exclusions for
the elliptic path family.

Known adaptive cell-probe lower bounds are only logarithmic in this general
regime and do not prove the polynomial tradeoff needed here. R36 therefore
does not infer a lower bound from the failure of (20)-(23).

## Theorem 5: the direct path-block index still misses the rectangle

R31 decomposes `Z` into `O(B)` paths under one public shift `q_0`, with total
path length `B^2`. For an integer block length `h`, preprocess

```text
V_sigma+[0,h)*q_0.                                (24)
```

This uses `O(B^2*h)` marked entries. Split every R31 path into consecutive
blocks of length at most `h`. For a query `R`, test the translated anchor of
each block against (24), retaining the path, block offset, and pair source.
This is exact and costs

```text
S_path=O(B^2*h),
T_path=O(B^2/h+B).                                (25)
```

The setup cap forces `h<=B^(1/4+o(1))`, hence

```text
T_path>=B^(7/4-o(1)).                             (26)
```

Reaching (17) through this schedule would require `h>B^(3/2)`, which makes
the represented state exceed `B^(7/2)` before duplicate removal and is far
outside setup. Hashing every path point is just the `T=B^2` endpoint of (25).

Equation (25) is a route-specific blocking tradeoff, not a lower bound for a
nonlinear path index, an algebraic `g` evaluator, or a target-batched circuit.
The complete target query is harder than (18): after finding `(v,w)`, it must
identify a captured complete key above `w`, invoke the exact source section,
respect all occurrence restrictions, and verify (7).

## Harness obligations and controls

1. The source sampler uses the oriented key. The unordered branch quotient is
   handled by trying the constant orientation family and verifying the exact
   signed source.
2. Ramified fibers only decrease the number of represented center points in
   (3); they do not weaken the expected-work bound.
3. The sampler is exact conditional on termination but does not certify the
   R35 high-support promise or exact multiplicity.
4. Target-support multiplicity is not target-support cardinality. Duplicate
   source rows at one `w` do not improve (10).
5. The signed target bound unions a constant number of pair endpoint supports;
   repeated occurrence labels can only reduce the number of group targets.
6. A mixed selected/unselected branch is outside Theorem 2 only if its useful
   target-visible endpoint is explicitly outside `Z` and its complete marked
   image is constructed within the caps.
7. The 3SUM-Indexing comparison is a projection and current-algorithm audit,
   not an unconditional reduction from arbitrary instances to the structured
   elliptic family.
8. No exact global key multiplicities, fresh-target marked index, R10 queried
   coefficients, independent rank, factor-base logs, scalar-blind descent,
   unrestricted lower bound, Shoup improvement, or breakthrough is supplied.

## Operation-level deduplication

- R9 owns exact count-to-source self-reduction and explicitly leaves direct
  empty-or-source routers as a separate positive interface.
- R10 owns target-query campaign accounting and the rank-two control.
- R13 owns erased-image multiplicities and graph-first source degree.
- R30 owns near-total both-branch selected containment.
- R31 owns the path representation of `Z`.
- R35 owns heavy-key discovery and sampled-fifth density.
- R36 adds the expected-linear heavy-key root source section, the all-shift
  target-support density bound, and the marked path-3SUM target projection.

## Result boundary

```text
heavy captured key root source: exact Las Vegas, expected O(B)
unscanned fifth labels: reached by root source sampler
exact source multiplicity and arbitrary child counts: absent
both-branch target-visible endpoint support: B^2
compressed target support with other pair deck: O(B^4)
random known-log target success: at most B^(-1+o(1))
density loss: at least 1-o(1)
campaign exponent at kappa=5/4: at least 13/20
sub-rho target-query requirement: kappa below 1/2
fresh-target projection: marked path-structured 3SUM-Indexing, n=B^2
direct path-block schedule: S=B^2*h, T=B^2/h+B
fresh target, R10, rank, logs, descent: absent
Shoup-bound improvement: false
breakthrough: false
```

## Primary-source boundary

The unstructured indexing controls use:

- Kopelowitz and Porat, *The Strong 3SUM-INDEXING Conjecture is False*,
  <https://arxiv.org/abs/1907.11206>.
- Dinur and Golovnev, *Improved Time-Space Tradeoffs for 3SUM-Indexing*,
  <https://arxiv.org/abs/2512.04258>.
- Chung and Larsen, *Stronger 3SUM-Indexing Lower Bounds*,
  <https://arxiv.org/abs/2203.09334>.

They neither analyze R31's elliptic translation paths nor supply the marked
complete-key/source interface. The alphaXiv/X instruction bound in R13 remains
workflow guidance only; it does not alter any theorem or cost.

## Exactly one next action

Construct or refute a marked fresh-target index for the R31 path family.
Given a fresh known-log `R`, it must find an extendible captured complete key,
the pair endpoint `v`, and enough data to invoke Theorem 1, with target-
independent setup/state at most `B^(9/4+o(1))` and average target work below
`B^(1/2-o(1))` after the unavoidable density loss. It must beat the explicit
path-block schedule (25), preserve all sign and occurrence backpointers, and
then pass R10, independent-rank, factor-log, and scalar-blind descent gates.
