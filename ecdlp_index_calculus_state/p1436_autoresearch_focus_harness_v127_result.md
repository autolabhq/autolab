# P1436 autoresearch harness V127 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V127 binds R178 as the 114th closed frontier lane. It closes marked Fitting as
a distinct algorithmic lane and routes the first experiment back to
`s66_fused_factored_dual_chow_outer_norm_mod_u` at priority 318.

## R178 Result

For each selected point `P`, define the signed row norm

```text
C_h(P) = product_(Q in D) h(P+Q),
```

where `h` is the R176 principal signed-incidence witness. Its zero set on the
squarefree selected divisor consists exactly of points with at least one
signed target incidence. Interpolation modulo `U` gives the aggregate element
`C_h`, and the candidate factor is

```text
G_1 = gcd(U,C_h).
```

All six controls verify that `G_1` is byte-identical to both the R174 signed
dual-Chow candidate factor and the R177 marked-locator gcd.

## Fitting Filtration

Let

```text
mu(P) = #{Q in D : h(P+Q)=0}
G_r(A) = product_(mu(P) >= r) (A-x(P)).
```

Then the R177 marker factors exactly as

```text
L(A) = det(AI-X_1 | ker K)
     = product_P (A-x(P))^mu(P)
     = product_(r>=1) G_r(A).
```

The global threshold degrees are

```text
deg G_1: 140
deg G_2:  71
deg G_3:  23
deg G_4:   7
sum:      241
```

Only `G_1` supplies distinct ECDLP candidates. The other 101 degrees encode
repeated incidences but add no roots, labels, backpointers, or independently
verified relations.

## Exact Controls

Across three curves and two seeds:

```text
selected pair evaluations:                 8,922
signed row norms:                            202
aggregate quotient-ring output slots:        202
nonzero aggregate coefficients:              202
distinct candidate roots:                    140
incidence multiplicity sum:                  241
maximum incidence multiplicity:                4
```

Every aggregate interpolation is exact and fully dense. Every row-norm zero
set equals the R177 candidate set, every first threshold equals the R174/R177
candidate factor, and every threshold product equals the R177 marker. These
finite computations receive no asymptotic credit.

## Cost Boundary

```text
selected divisor input n:                       B^(9/4)
target witness input N:                         B^(5/4)
distinct candidate factor:                      B^(3/4)
marked multiplicity output:                     B^(3/4)
represented aggregate output:                   B^(9/4)
explicit signed target grid nN:                  B^(7/2)
R177 pair algebra n^2:                          B^(9/2)
R177 explicit marker interpolation:              B^6
conditional nonlocal signed norm total:          B^(9/4)
R163 label/backpointer postprocessing:             B^2
rho proxy:                                       B^(5/2)
```

For ECDLP, constructing the full marker and constructing the signed aggregate
factor have the same desired softly `O(n+N)` envelope because reading `U,V`
already costs `Theta(n)`. R177 is therefore a multiplicity-refined certificate,
not a separate asymptotic primitive.

The unresolved object remains the nonlocal signed elliptic translate product
or fused dual-Chow outer norm that emits `C_h mod U` or `G_1` without an `nN`
target grid, `N` dense quotient-ring elements, or `n^2` pair/Fitting state.

R178 passes 15 of 22 obligations. It admits the signed row-norm identity,
Fitting filtration, exact factor equality, finite controls, and mechanism-level
deduplication only. Constructor admission, lane admission, rho improvement,
Shoup improvement, and breakthrough flags remain false.

## V127 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v114`.
- Bound and closed frontier lanes: 114.
- First focus: `s66_fused_factored_dual_chow_outer_norm_mod_u`.
- First-focus priority: 318.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R178 focused tests: 18 passed in 69.020 seconds.
- Harness tests: 144 passed in 0.772 seconds.
- Full ECDLP suite: 1,222 passed in 706.640 seconds.
- R178 clean replay: all six JSON outputs byte-identical through the focused
  deterministic-bundle test.
- V127 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R178: 103 receipts, 2,043 recursive path/hash bindings,
  zero mismatches, missing paths or rounds, and duplicate rounds.

## R178 Hashes

- Producer: `5aa01661614943739d8dc4ee623def8df94a0ecbc6119f344cee7f88b4b24312`
- Report: `679eeeeb8031fa4b4121d2004f5778629c5f16f116dcf8b10b3603a2ad63d2a6`
- Frozen interface: `a575b4bbf47f3d43ada0fc4411d352f795a87261caae7ba388f7e427b46325a4`
- Cost ledger: `999972250b94afa500eaa1e93285d70d662192232cd2c6ae9527a36ddf8ca606`
- Replay: `b241914e4cdfd7b14e66fba051399dabfd84b1248be31fab0db57714bad6acae`
- Controls: `ca239fb08f96845781ea8a7be8237fc0bf0bc66e1ed2918cb6ac9a1d55437716`
- Equivalence ledger: `4c9d6cb8b5a7d32747e13542f42b3a3a382b04d97cc6caa709cd6aa706dd7c66`
- Tests: `ae71be82ca4f0113d27459b6360de65b1c5312e84ff9951899960a6be196ea65`
- Gate: `bf0071ea97bd82f97b18c2f944f039153b5711ab9f3ace4b0aa910b805df9462`
- Parent: `12cf8c48f4d0c3d5973c8d07dc85f92eb7459aae4952942c09a39075d4ccceaf`

## V127 Hashes

- Harness: `bbfd5faca558610869e881b4ee73556eb405a31d6b9beaf46337cc65a608d481`
- Harness tests: `3c9d8c51e925e70b5f21a59369b7f0b1a288bc8faecf978108325673ad5c6728`
- Focus report: `66bce2a5cb6f6d2bbe94ad4c2cd3e58572573f84f85427451056338750b66855`
- Note: `af0a7d649187b62476a286df5992fa96fd647aecf1a2c12486883441647fe9ef`
- Evidence inventory: `4996de51daef5d2e2124e448f673d709784c8d76189fcd8403893e366814fd00`
- Replay plan: `e2fd9970a5f766c63ca34a3e21d0daaa4bc2ecdadb2b0fb2d1946f2aa58bdd19`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact filtrations, finite controls,
candidate-factor equality, verifier passes, and mechanism-level deduplication
receive no unconditional attack or circuit-lower-bound credit.

## Next Action

Construct or refute the unified nonlocal signed elliptic translate-product
operator on compact `U,V` and target-divisor or principal-witness inputs. Emit
`C_h mod U` or `G_1=gcd(U,C_h)` in softly `O(n+N)` work. Reject `nN` target
grids, `N` dense quotient elements, `n^2` pair/Fitting state, the full marked
determinant, candidate inversions, and unit-cost norm, resultant, multipoint,
root, count, marginal, rank, source, DLP, or generic locator oracles.
