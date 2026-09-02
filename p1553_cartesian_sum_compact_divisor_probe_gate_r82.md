# P1553 Cartesian-sum compact divisor probe R82

## Classification

- Owner: existing P1553 R3/R11/R14/R81 compact-endpoint frontier; no P1554
  and no new idea ID.
- Evidence: eight prospectively frozen scalar-blind addition-pushforward
  factor bases on four prime-order relation-scale toy families, eight matched
  hash-to-curve controls, exact group source replay, and verifier-separated
  density/rank controls.
- Status: `CARTESIAN_SUM_COMPACT_S4_PASS__FULL_SOURCE_RHO_FAIL`.
- Labels: `exact-finite`, `prospective`, `scalar-blind-source`,
  `local-positive`, `verifier-dlp-separated`, `novelty-unverified`.
- Cryptanalytic result: one exact local S4 compiler passes both caps, but no
  admitted known-RHS relation source, factor-log solve, identical target
  descent, Shoup improvement, or ECDLP breakthrough follows.

R81 rejected complete one-dimensional multiplicative coordinate cosets
because their triple endpoint images remained cubic. R82 leaves that grammar.
It freezes two independent scalar-blind point decks and uses their elliptic
addition image as the factor base:

```text
|A|=u=B^(2/5+o(1)),
|C|=v=B^(3/5+o(1)),
F={F_ij=A_i+C_j},       |F|=uv=B.                              (1)
```

The addition-pushforward divisor has only `u+v` input leaves and one
two-input correspondence gate before its `B` output points. Its exact triple
endpoint representation factors as `3F=3A+3C`. This gives a genuine passing
local compiler with `B^(9/5)` state and `B^(6/5)` endpoint-query work.

The compression does not survive the full five-factor relation. It becomes
the colored ten-sum `5A+5C=R`. The best explicit equality split has
`B^(13/5)` work and `B^(12/5)` state, and generic collision search is
`B^(5/2)=N^(1/2)`. The local positive is therefore retained without promoting
the pipeline.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R81 complete multiplicative-x coset report | `e556efa7c1e639f76915f152ebdcc3d00db2a932d8ef207ae8653c94117f026f` |
| R14 tensor-trace minimal-polynomial compiler gate | `da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac` |
| R11 Cartesian Kummer rigidity gate | `6c79b486bfa4cfd14674033a438db3d91ddf7bbe4d2c4aaf309f1a0706f0df4e` |
| R31 registry containing R3 Query2P1 gate | `0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f` |
| Bound R3 Query2P1 gate | `b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e` |

## Prior-art boundary

Semaev's [summation-polynomial construction](https://eprint.iacr.org/2004/031)
is the field-equation interface for the eventual colored ten-sum. R82 does
not claim summation polynomials or Cartesian point sums as novel.

Wagner's
[generalized birthday algorithm](https://people.eecs.berkeley.edu/~daw/papers/genbday.html)
uses staged constraints in an addition-compatible representation. R82 does
not credit hash-prefix filtering on elliptic-curve encodings: a cyclic
prime-order subgroup has no proper nontrivial subgroup quotient, and an
x-coordinate or encoding prefix is not preserved by group addition.

Shoup's
[generic-group lower bound](https://www.shoup.net/papers/dlbounds1.pdf)
is the nonclaim boundary. The measured `N^(1/2)` generic collision route
matches it. A future summation-polynomial or FFE route would be
representation-specific and would require its own full source-to-descent
proof.

## Frozen divisor

Every family uses

```text
E/F_p: y^2=x^3+1,
p=2 mod 3,
#E(F_p)=p+1=6q,
q prime.                                                        (2)
```

For each of two fixed offsets, SHA-256 domain separation chooses `u` points
for `A` and `v` points for `C`. Each candidate x-coordinate is lifted,
cofactor-cleared into the order-`q` subgroup, and assigned a canonical sign.
Only identity, duplicate, or noninjective output degeneracies are rejected.
No scalar label or endpoint-support outcome enters selection.

The frozen divisor circuit is

```text
D_A=sum_i[A_i],
D_C=sum_j[C_j],
D_F=add_*(D_A x D_C)=sum_(i,j)[A_i+C_j].                       (3)
```

All eight maps are injective and omit the identity. Optional materialization
costs `B` group additions, inside setup. The construction is not a
one-dimensional field coset or known scalar orbit.

## Exact S4 compiler

For an ordered factor triple,

```text
F_i1j1+F_i2j2+F_i3j3
 = (A_i1+A_i2+A_i3) + (C_j1+C_j2+C_j3).                        (4)
```

Build exact point-keyed dictionaries for `3A` and `3C`, retaining one atom
triple per endpoint. Store both dictionaries. To query endpoint `T`, scan the
smaller `3A` support and look up `T-a` in the `3C` dictionary. The returned
atom triples may be paired positionwise into three factor-base columns and
replay exactly.

The asymptotic receipt is

```text
3A construction and query scan        u^3 = B^(6/5+o(1)),
3C construction and retained state    v^3 = B^(9/5+o(1)),
allowed setup/state                    B^(9/4+o(1)),
allowed fresh work/workspace           B^(5/4+o(1)).             (5)
```

Thus `9/5<9/4` and `6/5<5/4`. No `B^3` convolution is retained. Across the
four scales, exact endpoint supports are:

| `u,v,B` | `|3A|` | `|3C|` | verifier-only `|3A+3C|` |
|---:|---:|---:|---:|
| `2,3,6` | 4 | 10 | 40 |
| `3,5,15` | 10 | 35 | 350 |
| `3,6,18` | 10 | 56 | 560 |
| `4,7,28` | 20 | 84 | 1,680 |

Both offsets agree in this support table. Every sampled accepted endpoint
returns and verifies one exact scalar-blind source. This is a real local
positive, not an oracle placeholder.

## Column geometry

The nominal factor logs obey

```text
ell_ij = alpha_i+gamma_j.                                      (6)
```

For every rectangle,

```text
F_ij+F_i'j'-F_ij'-F_i'j=O.                                    (7)
```

Adjacent rectangles span the public kernel with exact dimension

```text
B-u-v+1=(u-1)(v-1).                                            (8)
```

Only `u+v-1` log directions are meaningful because
`alpha_i -> alpha_i+d` and `gamma_j -> gamma_j-d` leave every `ell_ij`
unchanged. On every frozen instance, verifier-only known-RHS rows reach
exact projected rank `u+v-1`; adding the public rectangle rows reaches all
`B` nominal columns:

| `u,v,B` | meaningful rank | rectangle rank | combined rank |
|---:|---:|---:|---:|
| `2,3,6` | 4 | 2 | 6 |
| `3,5,15` | 7 | 8 | 15 |
| `3,6,18` | 8 | 10 | 18 |
| `4,7,28` | 10 | 18 | 28 |

BSGS supplies the right-hand scalars only for this prospective audit. The
candidate does not know those scalars and receives no algorithmic factor-log
credit until a valid known-RHS source query exists.

## Density controls

Exact weighted multiset convolution verifies all `B^5` ordered five-factor
occurrences. Candidate five-sum support densities range from `0.001538` to
`0.007670`; matched random factor bases range from `0.011911` to `0.022017`.

The candidate loses a finite combinatorial factor because atom permutations
share endpoints, but not an asymptotic exponent:

```text
number of unordered atom source pairs
  <= C(u+4,5) C(v+4,5)
  = Theta(u^5 v^5)
  = Theta(B^5).                                                 (9)
```

The toys therefore retain relation-scale endpoint supply and full
prospective meaningful rank. They do not provide the algorithm needed to
hit a chosen known right-hand side.

## Full source obstruction

A five-factor relation or identical target descent asks for

```text
R=sum_(k=1)^5 F_ikjk
 =sum_(k=1)^5 A_ik + sum_(k=1)^5 C_jk.                         (10)
```

This is a ten-list equality problem with five `A` and five `C` positions.
For an explicit meet-in-the-middle split containing `r` A positions and `s`
C positions, the list exponents are

```text
L=(2r+3s)/5,
R=5-L.                                                         (11)
```

The best integer split is `(r,s)=(2,3)` or its complement:

```text
explicit work exponent     max(L,R)=13/5=2.6,
explicit state exponent    min(L,R)=12/5=2.4.                  (12)
```

Both exceed their caps. A generic low-memory collision route can restore the
balanced birthday exponent, but only

```text
B^(5/2)=N^(1/2),                                               (13)
```

which is Pollard-rho scale and still far above the `B^(5/4)` fresh-target
limit.

The local `3A+3C` membership oracle does not compose around this problem.
Enumerating pair endpoints and invoking it restores over-cap traffic. A
Wagner tree would need proper addition-compatible quotient projections; none
is supplied. The direct Semaev equation is `S_11=0` on ten atom coordinates
and the target. Standard split quotient or resultant routes expose
source-scale degree `u^5v^5=B^5`; no subcap FFE solver or source selector is
supplied.

## Campaign boundary

The full obligations remain

```text
setup/state                         B^(9/4+o(1)),
fresh-target work/workspace         B^(5/4+o(1)),
known-RHS source without DLP labels,
independent meaningful row rank,
verified factor logs,
identical scalar-blind target descent,
total field and bit cost below rho.                             (14)
```

R82 passes the frozen scalar-blind factor base, compact divisor SLP, exact
local S4 compiler, local caps, prospective density, matched controls, and
public/prospective rank checks: 11 of 17 obligations. It fails the full
source, algorithmic known-RHS rank, factor logs, descent, generic-prime
theorem, and Shoup-improvement obligations.

The negative is scoped. It closes Cartesian addition-pushforward factor bases
under explicit equality joins, generic collision search, and
quotient-free generalized-birthday accounting. It is not a lower bound
against a representation-specific summation-polynomial or FFE filtration
with exact provenance.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_cartesian_sum_compact_divisor_probe_r82.py` | `7380bff3175625016affee4703b0b0f2867a28113f72eef2d90614ed57ffef07` |
| `p1553_cartesian_sum_compact_divisor_probe_report_r82.json` | `ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832` |
| `frozen_compact_divisor_factor_base.json` | `b5d90a713e6d2e0d092ac9019fcc7fc262df9a1874d550ba1782271ff66465b5` |
| `divisor_slp_to_s4_endpoint_compiler.json` | `4987a01b3214a0fc5b822f52bd3ee80e405a406599be9067fb013dc2b3050629` |
| `prospective_density_and_matched_rank_controls.json` | `1a727680b6fc2644c167861bc515588dbc712c7470a9923067f55a230131e195` |
| `scalar_blind_query_source_cost_ledger.json` | `37df840a3025d2b3226cb403a764af6f4cb3e8baa88e25d1254123f52c389086` |
| `verified_factor_logs_and_identical_descent.json` | `fb6bb7367b1f75c17d9b9ce9abb690ff01ea581f13f525c8302323ed6da1dd8b` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_cartesian_sum_compact_divisor_probe_r82.py` | `4e5ec3da8794a55517be02be8fbf20b79f2abe088b9fa09fdff59cbbbe230247` |

Targeted replay:

```text
Ran 5 tests in 0.005s
OK
instances=8 s4_pass=True full_work_B=2.6 lane_admitted=False
```

## Disposition

```text
REJECT_CARTESIAN_SUM_FULL_PIPELINE_ONLY__COMPACT_ADDITION_PUSHFORWARD_FACTOR_BASE__EXACT_B6O5_BY_B9O5_TRIPLE_COMPILER__B6O5_TRIPLE_QUERY_PASSES__PUBLIC_RECTANGLE_KERNEL_EXACT__PROSPECTIVE_MEANINGFUL_RANK_PASSES__FULL_RELATION_IS_5A_PLUS_5C__BEST_EXPLICIT_JOIN_B2P6_WORK_B2P4_STATE__GENERIC_COLLISION_B2P5_EQUALS_RHO__NO_QUOTIENT_WAGNER_CREDIT__NO_SUBCAP_FFE__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Derive or refute one addition-compatible field-coordinate filtration for the
colored `5A+5C` source. Every partial filter must compose under elliptic
addition, retain exact source backpointers, avoid verifier DLP labels, fit
`B^(9/4)` setup/state and `B^(5/4)` fresh work/workspace, and pass the same
construction on a matched random-deck control.
