# P1436 autoresearch harness V128 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V128 binds R179 as the 115th closed frontier lane. It closes direct x-adic
and per-component CRT uses of truncated resultants for the R178 operator, then
routes the first experiment to the narrower factored arbitrary-squarefree
dynamic-evaluation form of
`s66_fused_factored_dual_chow_outer_norm_mod_u` at priority 320.

## R179 Result

The Moroz-Schost interface computes

```text
Res_y(P,Q) mod x^k
```

in softly `O(dk)` field operations for bivariate input degree at most `d`,
under the paper's stated characteristic conditions. R178 instead requires the
signed aggregate

```text
C_h mod U,
```

where `U` is an arbitrary squarefree polynomial of degree `n`.

For each root `a` of `U`, the pair

```text
0, (X-a)^n
```

has the same residue modulo `(X-a)^n` but different residues modulo `U`.
Therefore one order-`n` local truncation does not determine the required
arbitrary-modulus output.

Because the selected `U` is squarefree and split, its quotient algebra is the
product of `n` order-one local fields. The two direct adaptations are:

```text
one local expansion:     d >= n, k = n, softly O(n^2)
n squarefree CRT calls:  d >= n, k = 1 each, softly O(n^2)
```

R179 closes these two standard applications only. A factored D5/dynamic-
evaluation, transposed, or custom arithmetic-circuit constructor is not
refuted.

## Exact Controls

Across three curves and two seeds:

```text
selected divisor degree sum:                 202
target factor count sum:                      40
squarefree CRT components:                   202
signed pair evaluations:                   8,922
single-expansion degree-precision total:    8,922
CRT degree-precision total:                 8,922
represented target-Chow coefficient slots:   204
```

Every R178 aggregate replays exactly, every selected `U` passes
`gcd(U,U')=1`, all 202 order-one residues reconstruct the 202 aggregate slots,
and every local-power alias witness is exact. These finite computations
validate the identities and accounting only; they receive no asymptotic or
attack credit.

## Cost Boundary

```text
selected divisor input n:                       B^(9/4)
target factor input N:                          B^(5/4)
required output:                                B^(9/4)
desired factored total:                         B^(9/4)
one order-n x-adic application:                 B^(9/2)
n order-one CRT applications:                   B^(9/2)
represented target dual-Chow body:              B^(5/2)
standard nN target grid:                        B^(7/2)
rho proxy:                                      B^(5/2)
```

Expanding the `N` target factors reaches rho before constructing the outer
norm. R179 therefore preserves only an operator that keeps `U,V` and the
target factors factored, shares work across every CRT component, and emits
`C_h mod U` or `G_1` in softly `O(n+N)` total work.

R179 passes 11 of 20 obligations. It admits the published-interface scope,
the squarefree/local mismatch, exact CRT replay, and standard-route costs.
Constructor admission, lane admission, factor logs, target descent, rho
improvement, Shoup improvement, and breakthrough flags remain false.

## V128 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v115`.
- Bound and closed frontier lanes: 115.
- First focus: `s66_fused_factored_dual_chow_outer_norm_mod_u`.
- First-focus priority: 320.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.
- The alphaXiv autoresearch source snapshot and bounded-critical-set method
  remain bound in the generated note.

## Verification

- R179 focused tests: 13 passed in 27.300 seconds.
- Harness tests: 145 passed in 0.790 seconds.
- Full ECDLP suite: 1,236 passed in 609.286 seconds.
- R179 clean replay: the deterministic bundle test rebuilds all six outputs
  twice and requires exact equality.
- V128 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R179: 104 receipts, 2,067 recursive path/hash bindings,
  zero mismatches, missing paths or rounds, and duplicate rounds.
- `git diff --check`: passed.

## R179 Hashes

- Producer: `00210a51d1ee5dd3f8154eeab57c5909309711c3558a0699ae9ffa0cf0b271e9`
- Report: `3f68214851d59a68369f45cf7cc9b1c1eed768f0129cc218834fb3e25d4ac4e5`
- Frozen interface: `ef327c0c0dad05eb697a5e99bce2d8f37533fe07dc747ee8c5082652cb9edcc6`
- Cost ledger: `6315b3766fd611229bbb8394e4dd135f084e203ec823f3c712d9e08f5d7d73b7`
- Replay: `8d3b4e7af0f28fc0e5626ce14717fb450f54474d1e28c7c1bb6a5697794a5f06`
- Controls: `07bda95159541b61d43b307e85f756ec6b7239273b73103414ef8ff52125bcbc`
- Applicability ledger: `037891a103819194d1c8866a779ee98fbaa554fbdd53bed8576e78c888ef4769`
- Tests: `1e700b72e6934753cd939c9bcf472d9653f4e62a09e7a5af8c3f36a81f7e215e`
- Gate: `87ae37e53a7f10af580a58287f2d83848dee20ff8f5f8e157d74174b39a21791`
- Parent: `84e4d0b6acab02415dabb6dcd917028618748318b3544868cfaaf1593792ee4c`

## V128 Hashes

- Harness: `88ee88e6952608835e734bed0f1b65fa28d94697b124d580da5970d53429fc60`
- Harness tests: `4a1865b2c6a6354de72ba2a78f57ec47a4c2ded1a38cdb613b1646c5ddb517b0`
- Focus report: `e880800ab615ec8ee0bd95ee0f880943a279119bb0675bc68466de12d9812f57`
- Note: `67bdd4c3938e8b08d002355b233f8a3ad20599dff5cced6c029dd20de027b9de`
- Evidence inventory: `abe80c296385785759abd4b929012259d5e235a0240833361e3691b2589ceb95`
- Replay plan: `56b0c2f3f1e6808430411c025db4fbc9dc4376a0f31651370c57f264f9ec56b6`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. The published-resultant
applicability audit, exact finite controls, verifier passes, and standard-route
cost exclusions receive no unconditional attack or circuit-lower-bound credit.

## Next Action

Construct or refute one factored arbitrary-squarefree dynamic-evaluation
operator. Keep `U,V` and all `N` target factors factored, split `U` only by
charged gcds at actual nonunits, share elimination work across every CRT
component, and emit `C_h mod U` or `G_1` in softly `O(n+N)` total work. Reject
one order-`n` x-adic surrogate, `n` local resultants, `N` quotient-ring
elements, `N^2` coefficient expansion, `nN` or `n^2` grids, candidate
inversions, and unit-cost resultant, norm, multipoint, root, count, marginal,
rank, source, DLP, or generic locator oracles.
