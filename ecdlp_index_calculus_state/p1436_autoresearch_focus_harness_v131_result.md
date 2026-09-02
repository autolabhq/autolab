# P1436 autoresearch harness V131 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V131 binds R182 as the 118th closed frontier lane. R182 admits an exact
target-divisor subresultant identity and closes standard represented remainder,
Newton-trace, inverse-certificate, triangular-norm, and target-half-GCD routes
at scale `nN=B^(7/2)`. It routes the first experiment to an SLP-direct
candidate-only determinant-zero or resultant-mod-`U` interface at priority 326.

## R182 Result

After public target deduplication, unique elliptic targets partition into at
most two x-injective charts. Each chart has a squarefree root polynomial `W`
and a curve interpolant `V_T`. The R181 signed kernel restricts to

```text
R_P(u) = A_P(u) + V_T(u)*B_P(u) mod W(u).
```

For squarefree `W`, the following conditions are equivalent:

```text
gcd(W,R_P) is nonconstant
R_P is a nonunit in F[u]/W
Norm(R_P) = Res(W,R_P) = 0.
```

R182 also includes an adversarial duplicate-plus-opposite-point control. Its
two unique targets share one x-coordinate, force exactly two charts, preserve
the direct kernel values, and produce the same candidate roots under norm and
gcd semantics.

## Exact Controls

R182 replays nine R181 controls and adds held-out seed `18206` on all three
curve families:

```text
control count:                                  12
R181 replay / held-out controls:              9 / 3
selected divisor degree sum:                    404
raw / unique retained target sum:           80 / 79
ordinary target chart count:                     12
remainder coefficient slots:                   2,962
remainder nonzero coefficients:                2,962
candidate / noncandidate source components: 253 / 151
source-target gcd-degree sum:                    409
unit inverse certificates:                      151
inverse certificate coefficient slots:        1,130
Newton trace projections:                     2,962
```

Every ordinary remainder body is fully dense and every remainder matrix has
full target rank. Direct and remainder evaluations agree exactly. Newton power
sums reconstruct every norm, every norm-zero test equals the subresultant gcd
test, all unit inverses verify, and all candidate factors replay their R181 or
fresh R174 comparator. These finite controls receive no asymptotic or attack
credit.

## Cost Boundary

```text
n selected degree:                         B^(9/4)
N target degree:                           B^(5/4)
candidate output:                          B^(3/4)
desired total work:                        B^(9/4)
represented remainder body nN:             B^(7/2)
N degree-n Newton trace outputs:            B^(7/2)
noncandidate inverse-certificate body:      B^(7/2)
represented triangular norm dimension:      B^(7/2)
standard target-variable half-GCD:           B^(7/2)
rho proxy:                                   B^(5/2)
```

The represented standard routes therefore remain above rho. This is a scoped
cost audit, not a general lower bound against compact straight-line programs,
resultants, circuits, RAM algorithms, or data structures.

R182 passes 24 of 35 obligations. An SLP-direct candidate-only resultant,
deterministic hash-to-curve transfer, factor logs, identical target descent,
and all attack flags remain open or false.

## V131 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v118`.
- Bound and closed frontier lanes: 118.
- First focus: `s66_fused_factored_dual_chow_outer_norm_mod_u`.
- First-focus priority: 326.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.
- The alphaXiv autoresearch source snapshot and bounded-critical-set method
  remain bound in the generated note.

## Verification

- R182 and harness tests: 162 passed, 6 subtests passed in 63.08 seconds.
- Full ECDLP suite: 1,283 passed, 10 subtests passed in 859.73 seconds.
- R182 clean replay: all six generated outputs are byte-identical.
- V131 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R182: 107 receipts, 2,141 recursive path/hash bindings,
  zero mismatches, missing paths or rounds, and duplicate rounds.
- `git diff --check`: passed.

## R182 Hashes

- Producer: `0058a28e4c337df97cec7b2e87bb5cc16e18a5b47029ab72d12b852405875e15`
- Report: `9a38150e9e7e0e41f455b000b4cda390aaff4f428c2828d5ffeed8a9781f7f5d`
- Frozen interface: `29291879484cf68bc16b83e2341818a611bb0c529c53e135f33bc181f30d462e`
- Cost ledger: `0eb0f05465fc072c72b6cccc24105d420431eb53162a6d5042b135bdcb454121`
- Replay: `bec170f0a0148e269b28c0499f7a8c65b20a3270a8f8dbfc41a0724d3e71e9ba`
- Controls: `850099e9c6f538baea1f94b7d87dafbe8c2df7a83b3dcf9d47be9b233639b77f`
- Applicability ledger: `5f1700e9b00d6756daa58d38f36002e7f0dee704632f223de8f4bc55cfce620c`
- Tests: `37e7ced472a7eb2244e1539cb3248fd2e170bdec343a89a69b93ce9e05d39a7e`
- Gate: `960fbcdcae0867bc8b02786f81af9eb68ed337d2dfff9019e62c61248ca303fd`
- Parent: `d83ffeed2d66b78ec40a7b14e3a89fd5a8940775d21f5c297c6105454032ba66`

## V131 Hashes

- Harness: `c296cfa2ed015b8adf25c79c4613c93ff8d9d3e390360dd16ea6e20f73566fea`
- Harness tests: `e1b69e9d4f12f906f08ae50607703f497c32e020ee92bfef8956e9a7eeaabe17`
- Focus report: `b669ef5a8b4d311e63e86d62f89b81bbfa70aa66cb77e37b424de5f0b9ae51ca`
- Note: `3fd87d4609368dc72ff55b4854176b4069754c14738635e6a10e6f25b76751d2`
- Evidence inventory: `3570105229babae49926f45c714890a04f555bdbbd486ae78b0b16b1ab82d754`
- Replay plan: `75e3aeb53a9edbf4187f0dcc0c86ede222ce391c7ae19d9468398616228cf9fb`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, asymptotic rank theorem, general resultant or circuit lower bound,
or breakthrough was produced. Exact finite controls, held-out replays,
literature bindings, and verifier passes do not receive unconditional attack
credit.

## Next Action

Construct or refute one SLP-direct determinant-zero or resultant-mod-`U`
algorithm. Consume the compact R181 signed kernel and deduplicated target
charts, and emit only `G_1` in softly `O(n+N)` work without the `nN` remainder
matrix, `N` degree-`n` Newton traces, noncandidate inverse certificates, or a
represented triangular norm. Test a transposed single-functional minimal-
polynomial or subresultant certificate on seed `18206` and one new divisor
family. Reject hidden `nN` projection outputs, candidate oracles, and unit-cost
determinant or resultant calls.
