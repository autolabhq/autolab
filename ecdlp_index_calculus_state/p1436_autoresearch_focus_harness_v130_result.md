# P1436 autoresearch harness V130 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V130 binds R181 as the 117th closed frontier lane. R181 closes tautological
post-construction monogenic composition, explicit exact bounded-source-degree
scalar kernels on the controls, the canonical dense bidegree body, flattened
selected-target norms, and full composed-resultant output. It routes the first
experiment to a gcd-equivalent output-sensitive elliptic composed resultant
modulo `U` at priority 324.

## R181 Result

For fixed selected `P`, every tangent-corrected R174 factor is linear in target
coordinates `(u,v)`. Multiplication over the `n` selected points and reduction
by the curve equation gives the exact normal form

```text
K_P(u,v) = A_P(u) + v*B_P(u).
```

Every residue `C_h` already has `C_h=H(a) mod U` by taking `a=X` and
`H=C_h`. This identity gives no constructor: modular composition starts only
after the coefficients of `H` are supplied.

On all nine controls, the canonical kernel has pole order `3n`, exactly `3n`
dense coefficient slots per source point, and full source rank `n`. Its values
on a fixed public `n`-target scan also have rank `n` and equal independently
reconstructed R180 target factors entry for entry. Thus an exact scalar form

```text
K(a(P),T) = sum_(i=0)^d a(P)^i*c_i(T)
```

requires `d >= n-1` on these finite controls. This is not an asymptotic rank or
circuit lower bound and does not constrain arbitrary target-and-source-
dependent units in a gcd-only construction.

## Exact Controls

R181 replays six development controls and three controls on held-out seed
`18104`:

```text
control count:                              9
development / held-out controls:          6 / 3
selected divisor degree sum:                303
retained target count sum:                   60
canonical coefficient slots:            40,149
canonical nonzero coefficients:          40,149
desired signed incidence zeros:          13,383
charged scan pair evaluations:          658,473
```

All 13,383 desired incidence zeros are exact. Six candidate factors replay
R180, and the three held-out factors replay fresh R174 executions. Finite rank,
density, and target scans receive no asymptotic or attack credit.

## Cost Boundary

```text
n selected degree:                          B^(9/4)
N target degree:                            B^(5/4)
candidate output:                           B^(3/4)
postcompiled degree-n modular composition:  B^(9/4)
canonical explicit bidegree body:            B^(9/2)
N explicit degree-n factors:                 B^(7/2)
flattened selected-target algebra:            B^(7/2)
full composed-resultant output degree:        B^(7/2)
rho proxy:                                    B^(5/2)
```

Near-linear finite-field composition or triangular-algebra norm computation is
near-linear in represented input dimension. Flattening the selected and target
algebras has dimension `nN`. The complete special composed resultant likewise
has output degree `nN`. Neither standard represented route crosses rho.

R181 passes 21 of 33 obligations. Compact high-degree SLPs, gcd-equivalent
unit normalization, an output-sensitive resultant modulo `U`, deterministic
hash-to-curve transfer, factor logs, target descent, and attack flags remain
open or false.

## V130 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v117`.
- Bound and closed frontier lanes: 117.
- First focus: `s66_fused_factored_dual_chow_outer_norm_mod_u`.
- First-focus priority: 324.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.
- The alphaXiv autoresearch source snapshot and bounded-critical-set method
  remain bound in the generated note.

## Verification

- R181 and harness tests: 161 passed, 6 subtests passed in 51.03 seconds.
- Full ECDLP suite: 1,268 passed, 10 subtests passed in 633.45 seconds.
- R181 clean replay: all six generated outputs are byte-identical.
- V130 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R181: 106 receipts, 2,117 recursive path/hash bindings,
  zero mismatches, missing paths or rounds, and duplicate rounds.
- `git diff --check`: passed.

## R181 Hashes

- Producer: `c96a8283da297d78f3ba16d038fe6422026f09a0663d4ecacf9a290936dc178b`
- Report: `c80947aab90d88b402eb4fd42de5917b310e25455a0a7c59fc4e95b09b9f7be6`
- Frozen interface: `5244a753b4ceb16f8cbd7f271f41bb93af6cadd58fe73152c20d65225eac1356`
- Cost ledger: `77d018d104916d41686620cc0c880881d5919de7361aee4dbf6ae95f3b4f7be1`
- Replay: `860d5e366227116c8e1d3b757b498336ae08a6141640b5f480240b4fc4853b12`
- Controls: `3b85e19a38058ff9f3f0d0f79238075b5f0dc32e18ba4d38da3e0a1fa82c6a6f`
- Applicability ledger: `35157589f6109dc1443a0497f8f087fd6c0338b61b97950a7cd75073940339a3`
- Tests: `9c064a5039e027d7692729089877bfeeeed447d0b1c7fb44b1d247cc5c69766a`
- Gate: `24d801e02466f07a7a7e4036773a2fb362eeaae13bd5dccc947cf51d616f7469`
- Parent: `0bf5b1b722557ec0fe6ae5fb03297db40a9dd35df0dd88956d90955992832f4d`
- Poteaux-Schost reference: `587f302dd16c724d1be6a4b629a46a684a0c35389dbc22ba45641dba54de6f32`
- Special-resultants reference: `19db312c68f997949db342a050df568a009a340dd5026e4584e2c4fa59fcc375`
- Kedlaya-Umans reference: `93bd1f77b762f49bcae017d5c12ceccef38c67a956810873105cbce083634377`

## V130 Hashes

- Harness: `c8d9262c5ab3b40953ae3a87d3462cebed76d9a39086e1de62576a438cdcc53a`
- Harness tests: `e8c449618a6a1f5c614c7c681ae348ac720d1a75a60520ade40f47637495ef1e`
- Focus report: `5cae8b9cc703f935e78dccd9a7bfa4950b270d72bbcc2a2e9af77ac7163e21b9`
- Note: `2ace4d08ef405a886516a078d41e3634c75afd38d4050fd9b74344d618fe72fd`
- Evidence inventory: `ef67e0aa049710b972ddc8a1bb078aed3d98eecede85e9b8cd821e859129e6c7`
- Replay plan: `cac06c120ecbdf027193076b3ef2e631b9dcee1b1894d3dee86edfc392c08a96`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, asymptotic rank theorem, circuit lower bound, or breakthrough was
produced. The exact finite controls, held-out replays, literature bindings,
verifier passes, and standard represented-route exclusions receive no
unconditional attack credit.

## Next Action

Construct or refute one gcd-equivalent output-sensitive elliptic composed
resultant modulo `U`. Accept the `O(n)` signed line-product SLP and `O(N)` target
divisor, discard target-dependent units, and emit `G_1` in softly `O(n+N)` work
without constructing the canonical `3n^2` coefficient body, an `nN` tensor
element, or the full degree-`nN` composed resultant. Test direct transposed
power projections or subresultant traces modulo `U` on seed `18104` and one new
divisor family. Reject candidate oracles, unit-cost norms, and post-construction
modular composition.
