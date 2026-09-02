# P1436 autoresearch harness V132 result

Date: 2026-08-02

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V132 binds R183 as the 119th closed frontier lane. R183 admits an exact nested
Fermat-projector multiplicity and weighted-Prony locator interface, closes
source-blind target traces as label-incomplete, and charges represented
projector and A-linear Krylov routes at `nN=B^(7/2)`. It routes the first
experiment to an SLP-direct nested projector-moment operator at priority 327.

## R183 Result

For each source point `P` and squarefree x-injective target chart
`B_r=F[u]/W_r`, let `R_P` be the R182 target remainder and define

```text
m_P = sum_r Tr_Br(1-M_RP^(p-1)).
```

Fermat's identity on the split chart components gives

```text
m_P = sum_r deg gcd(W_r,R_P).
```

Because total target degree is below the characteristic, `m_P` is nonzero
exactly on candidate source components. With candidate count `c`, the moments

```text
s_k = sum_P m_P x(P)^k,  0 <= k < 2c
```

have Hankel rank `c`; Berlekamp-Massey and coefficient reversal recover the
exact candidate locator `G_1`. This is an exact downstream interface, not a
compact construction of those moments from the R181 line-product SLP.

## Exact Controls

R183 replays all twelve R182 controls and adds a strided source-divisor family
at seed `18307`:

```text
control count:                                      13
R182 replay / new divisor controls:             12 / 1
selected source degree sum:                         411
unique target degree sum:                            84
candidate / noncandidate source components:    256 / 155
target-gcd multiplicity sum:                        414
weighted moment count:                              512
represented source-target value slots:            2,997
distinct target-value spectrum degree sum:        2,839
full-spectrum source rows / source rows:       298 / 411
```

All projector traces, gcd degrees, direct group-law multiplicities, candidate
supports, Hankel ranks, Berlekamp-Massey orders, and candidate locators agree
exactly. The new control forces two target charts and is checked by an
independent elliptic group-law comparator. Finite enumeration receives no
asymptotic or attack credit.

An exact `F_101` row-swap collision preserves all twelve tested unweighted
global target power sums but changes the locator from `X-1` to `X-2`. Thus a
source-blind scalar target trace does not preserve source labels.

## Cost Boundary

```text
n selected source degree:                     B^(9/4)
N target degree:                              B^(5/4)
c candidate degree:                           B^(3/4)
2c moment output and Prony recovery:           B^(3/4)
candidate-target verification:                    B^2
desired compact setup:                        B^(9/4)
represented tensor projector:                 B^(7/2)
A-linear N-term target Krylov output:          B^(7/2)
rho proxy:                                     B^(5/2)
```

R183 passes 19 of 28 obligations. The output contract and downstream work are
below rho, but no SLP-direct constructor realizes the moments at that cost.
Deterministic generic-prime transfer, factor logs, identical target descent,
and every attack claim remain open or false.

## V132 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v119`.
- Bound and closed frontier lanes: 119.
- First focus: `s66_fused_factored_dual_chow_outer_norm_mod_u`.
- First-focus priority: 327.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.
- The alphaXiv autoresearch source snapshot and bounded-critical-set method
  remain bound in the generated note.

## Verification

- R183 and harness tests: 161 passed, 6 subtests passed in 33.97 seconds.
- Full ECDLP suite: 1,296 passed, 10 subtests passed in 1,429.29 seconds.
- R183 clean replay: all six generated outputs are byte-identical.
- V132 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R183: 108 receipts, 2,166 recursive path/hash bindings,
  zero mismatches, missing paths or rounds, and duplicate rounds.
- `git diff --check`: passed.

## R183 Hashes

- Producer: `202a28eade6a3d532c4f9f8cf6ae3e02c9a74da9bccc436069917204e511b53f`
- Report: `a2e746fa5159a3684d3f6913888ad9c6a1fd7983fcb9abbacc17e6d4ee5baae5`
- Frozen interface: `8d020353811cf7cf47e7950fe14f7da1df74eafb7c904f7e0abcfd875ef48c4b`
- Cost ledger: `c91e54b65c5068a3cada5b3f2237ec1f1228989d2115f22cfc2fd183cec27a6b`
- Replay: `2473d0295858506f0ec687c03831c5b345f726323e59486ef98f17b070ba168e`
- Controls: `9575764c7e662210b9f41ffe2783be3f6b95926f7b3d534e361401704b63366f`
- Applicability ledger: `652e2a17148613d57d729200a3297948e9cebc898ce07992482a5bf3927e3be8`
- Tests: `c44e54238b0599e30211a4035ed19858104272c11bfb866d2528e63343fac60e`
- Gate: `4544744eccc2050390b69bc377d6188e6e66b28c226c2840d0f607bc9330e67d`
- Parent: `28f0bf2f9f380944bf92ad2349eabd7fe230cf0e9d18574f3b09a84ac88a983c`

## V132 Hashes

- Harness: `0c9a70b92b22d0f6cb2342125fbf423c1a9d308a8e8f9e8f5e172e8095fd6a30`
- Harness tests: `f058d103c9fcf2de6f446227db34af991df7d1c3258dcc89e0073835c3b59ea9`
- Focus report: `8c015c2934a9925449cb0c501199e7d7eaabe229ef4c127fd72eaad33705f1c8`
- Note: `46963062df11719d485b946d4e75506b5902d9f2caf15495506819cd3a7b86e5`
- Evidence inventory: `b6137af1a340c58a48692c4e4a5e87ad011d9ac0dca5dd072aa1214804c90972`
- Replay plan: `b104d1f281f5dd568cf5c3d5d7d8bf4e8eef376bf9a57a046089e30ef4d4f0ee`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, asymptotic nonlinear-trace or circuit theorem, or breakthrough
was produced. Exact finite controls, locator recovery, literature bindings,
and verifier passes do not receive unconditional attack credit.

## Next Action

Construct or refute one SLP-direct nested projector-moment operator. From
compact `U,V`, target charts, and the R181 signed line-product SLP, emit the
`2c` scalars

```text
Tr_A(X^k Tr_B(1-M_R^(p-1)))
```

in softly `O(n+N+c)` work without representing `R`, the Fermat projector, an
A-valued target Krylov sequence, or any `nN` tensor body. Test seed `18308` and
a second strided divisor family; reject source-blind traces and candidate
oracles.
