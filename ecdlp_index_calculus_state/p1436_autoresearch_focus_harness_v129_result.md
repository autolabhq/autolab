# P1436 autoresearch harness V129 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V129 binds R180 as the 116th closed frontier lane. R180 closes literal
target-factor streaming, successive D5 zero tests, directed evaluation of the
same computation tree, and standard D5 half-GCD as below-rho constructors for
the R174/R178 aggregate. The first experiment remains
`s66_fused_factored_dual_chow_outer_norm_mod_u`, narrowed to a one-shot
monogenic or bounded-bidegree compiler, at priority 322.

## R180 Result

For every retained target factor `g_j`, R180 works in

```text
A = F[X]/(U),  deg U = n,
```

and verifies `gcd(U, product_j g_j) = G_1`. Candidate components can be
retired when a factor is an actual nonunit. Every component of `U/G_1` is a
unit for every target factor, however, so it survives all `N` factor steps.
For `c = deg G_1`, every factor order therefore incurs at least

```text
(n-c)N
```

component-factor visits in the literal tree. This is not a lower bound for a
one-shot monogenic, bivariate modular-composition, transposed, or custom
arithmetic-circuit construction.

## Exact Controls

Across three curves and two seeds:

```text
selected divisor degree sum:              202
target factor count sum:                    40
candidate degree sum:                      140
noncandidate survivor degree sum:           62
target-factor zero incidences:             241
materialized factor-residue slots:       1,486
finite signed pair evaluations:          68,326
natural component-factor visits:           869
exact optimal component-factor visits:     762
component-factor lower-bound sum:           466
```

All target-factor products recover the exact R174 and R178 candidate-factor
hashes. Every target-zero union equals the candidate root set, every recorded
split is an actual nonunit, every final survivor factor is a unit, and all six
factor orders are optimized exactly by dynamic programming. Materialized
finite controls receive no asymptotic credit.

## Cost Boundary

```text
n:                                      B^(9/4)
N target factors:                       B^(5/4)
c = deg G_1:                            B^(3/4)
n-c:                                    Theta(B^(9/4))
literal factor stream:                  B^(7/2)
successive D5 zero tests:               B^(7/2)
directed evaluation of the same tree:   B^(7/2)
standard D5 half-GCD:                   B^(9/2)
generic algebraic modular composition:  B^3.02175
rho proxy:                              B^(5/2)
```

The D5 product-algebra and directed-evaluation interfaces retain either the
`N`-step tree or the polynomial-degree factor. The cited 2026 generic
algebraic modular-composition exponent `O(n^1.343)` also remains above rho at
`n = B^(9/4)`. Finite-field near-linear bit complexity remains open only after
an exact one-shot representation is constructed without emitting `N` residue
elements.

R180 passes 18 of 28 obligations. It admits the factor decomposition, all six
candidate replays, exact optimal early splits, the noncandidate-survivor
invariant, and the cited interface specializations. Constructor admission,
lane admission, factor logs, target descent, rho improvement, Shoup
improvement, and breakthrough flags remain false.

## V129 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v116`.
- Bound and closed frontier lanes: 116.
- First focus: `s66_fused_factored_dual_chow_outer_norm_mod_u`.
- First-focus priority: 322.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.
- The alphaXiv autoresearch source snapshot and bounded-critical-set method
  remain bound in the generated note.

## Verification

- R180 and harness tests: 162 passed, 6 subtests passed in 14.16 seconds.
- Full ECDLP suite: 1,253 passed, 10 subtests passed in 638.10 seconds.
- R180 clean replay: all six generated outputs are byte-identical.
- V129 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R180: 105 receipts, 2,094 recursive path/hash bindings,
  zero mismatches, missing paths or rounds, and duplicate rounds.
- `git diff --check`: passed.

## R180 Hashes

- Producer: `d7925463992a6275aa05e7401bcde60dd5f4959b3a426892d561b78a6502ac60`
- Report: `93aff1e86ceab11135757789baf02497c5b4915843061125aa351854423bb760`
- Frozen interface: `e3bd7aa983330fa315552dd8813cb58efec30633dbae370586edb2a43edb2231`
- Cost ledger: `038215322dcc394d4231f234092e08e365aab6622138570fbf7eee35e1ae79cc`
- Replay: `f136b663b81f712849552881eb78336e2b1bf39800377027fdcd222f6c2fa17d`
- Controls: `a632faa9922d552106d8099f0efd98bb2114e81ed7c1cf7d235cd78a7cf28318`
- Applicability ledger: `8bacbab2eeffb69cb3e3a4f2dd0f3a02bda992a9e584a648e2d9a41cb72b9c9b`
- Tests: `e34e4067069d9288bc57c74dfa38ef5f558e5eda8e1c44137ffb61b9a3ba0f41`
- Gate: `b1e56b8a374d2b95a04f278e5c47910d072b937c67cbaea3f6dab7caee6d44d1`
- Parent: `a11bb3fea860ebbceb57d43d1aff2e115867f8667143c7fe78ee8a272c144270`
- D5 reference: `d2e265e13f585b9a3c9d69d27c8fa948d811a9de89b513d6983c3b8e8b87f565`
- Directed-evaluation reference: `20b6959dd71e3bd16e0c072f300b07a0c3714d013a28b1b0ea03955083df5da1`

## V129 Hashes

- Harness: `4151ce07c0247250544d93c4b8fd16c08a542e51c5d94d7096fa88901852bcee`
- Harness tests: `0ae30b9984906eb419f34b67cdc6ce315a4e1bb9945592b705ac66cf3cbc1cbb`
- Focus report: `b9f2cfe1d7eea97d12eb355cdd5fc394e87e00db3747e3e9f725dc91bb06759f`
- Note: `9f1e187a5757f46ee4b167ef7582716e1c0ed8245e7c300c80551f8366fced05`
- Evidence inventory: `6d1ae6919076c6a19b3c8f15035023c4c6603a285615f1828a6278a03dc50c93`
- Replay plan: `9ccd4e14ec4b8a56f253498bd206c11422422e1753c0c7a7d1909842b84ee3f4`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, circuit lower bound, or breakthrough was produced. The exact
finite controls, literature bindings, verifier passes, and standard-interface
cost exclusions receive no unconditional attack credit.

## Next Action

Construct or refute one one-shot monogenic compiler for the signed elliptic
translate product. From compact `U,V` and the degree-`N` target Miller SLP,
derive `H,a` with `C_h = H(a) mod U`, or a bounded-bidegree
`G(X,a(X)) mod U`, in softly `O(n+N)` preprocessing without emitting `N`
residue elements. Apply a charged finite-field modular-composition algorithm
and verify `G_1` on held-out divisors. Reject a disguised `N`-step
product-algebra tree, `nN` coefficients, `n^2` pair state, candidate
inversions, and unit-cost composition, norm, resultant, root, count, marginal,
rank, source, DLP, or generic locator oracles.
