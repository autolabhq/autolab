# P1436 autoresearch harness V120 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V120 binds R171 as the 107th closed frontier lane and routes the first
experiment to `s63_nonlocal_batched_elliptic_leaf_translate_product` at
priority 304.

## R171 result

R171 instantiates the compact generalized Miller witness left abstract by
R167. For a zero list of size `m`, the balanced tree has exactly `m-1`
tangent/chord line merges, logarithmic depth, and divisor

```text
sum_i [Z_i] - [sum_i Z_i] - (m-1)[O].
```

Six numerator and denominator tree pairs use 80 total merges with maximum
depth four. At 192 safe points, every compact tree equals the corresponding
dense R167 Riemann-Roch witness up to one nonzero scalar. The tree quotient
replays every selected endpoint and exactly the 140 R167 candidate roots.

For a generic shift, 1,222 admissible line-level Weil-reciprocity identities
hold exactly; 58 pole or nonunit rows are recorded without inversion. At every
selected endpoint `P`, `f0(-P)=U(x(P))=0`, so individual line ratios cannot be
specialized independently. Numerator and denominator origin factors must
cancel symbolically first.

After that signed cancellation, every internal partial sum, origin factor,
shared completion zero, and public auxiliary correction leaf cancels. Exactly
the original target factors survive: 40 across the six controls. Thus balanced
streaming lowers depth and live memory, but node-local or leaf-local work is
still `nN=B^(7/2)`.

This is a scoped negative for those streaming grammars, not an arithmetic-
circuit, RAM, cell-probe, elliptic-resultant, or generic-group lower bound.

## Cost boundary

```text
balanced Miller-tree state:             B^(5/4)
live endpoint value vector:             B^(9/4)
node-local line-norm streaming:          B^(7/2)
telescoped target-leaf evaluation:       B^(7/2)
raw signed-grid tree expansion:          B^(23/4)
represented aggregate output:            B^(9/4)
expected candidate factor:               B^(3/4)
signed verification:                     B^2
rho proxy:                               B^(5/2)
```

The output and live state fit below rho; the tested constructor work does not.
The surviving primitive is a nonlocal point-list-to-product operator that
fuses all elliptic leaf translates in softly `O(n+N)` work.

Miller and Enge justify the line recurrence and factored representation.
Moroz-Schost truncated resultants begin from represented bivariate inputs, and
Bhargava et al. multipoint evaluation begins from a represented coefficient
vector plus explicit points. Neither supplies the missing elliptic divisor-
list compilation or moving all-pairs fusion.

R171 passes 22 of 29 obligations. Lane admission, rho improvement, Shoup
improvement, and breakthrough flags remain false.

## V120 routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v107`.
- Bound and closed frontier lanes: 107.
- First focus: `s63_nonlocal_batched_elliptic_leaf_translate_product`.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R171 focused tests: 15 passed in 27.02 seconds.
- R171 plus harness tests: 152 passed in 26.43 seconds.
- Full ECDLP suite: 1,099 passed in 276.30 seconds.
- R171 clean replay: all six JSON outputs byte-identical.
- V120 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R171: 96 receipts, 1,859 recursive path/hash bindings, zero
  mismatches, missing paths or rounds, and duplicate rounds.

## R171 hashes

- Producer: `684ab4cf92398b7a7789b8e429356ee712ba255e65f5d172cee1c4ebbe58d39e`
- Report: `12c34212fa3e07321ec7e9524a32a845233b420eae28f16e2dc3abfebd4efa92`
- Frozen interface: `a80e1d9cbb7893a0a70ee04c95cd4ca2632f45181c9b72d645bd1afaf15abaef`
- Cost ledger: `9e9fe2ae4e5ec8e905b60b56227be57e86f0cc10de63e0c0917dc7b5837ebb76`
- Replay: `76a793283ce4df296d0ead1dc6826bab40835cefdfcbc809e481bbbe377141be`
- Controls: `d54c4d4db51abab1ccf4d01294bee945d240d637dd9db4a0078a523bef8aa300`
- SLP/cancellation ledger: `e322d78df66e9edec2fb39b412a98b23077b7648548f4154f30b7a166a5bcc7f`
- Tests: `85df97c68496b5ce8beb3c207ab2e6b0bf0a51486be80183e6e92d02d76e74e5`
- Gate: `0f862466fa7816c438c6ef8526e66e350b1091c02f2e631d5b28a4105c10d680`
- Parent: `3fee4660f7b558116299174194657da9464435f402a9f3fb289489474fbb1f93`

## V120 hashes

- Harness: `015d9c8d022f243a6fc5f867ae7973da133e3e0f15c0ad957eb1e1caaea68a3e`
- Harness tests: `3dccaa450553cb45525385fb6ad01d6299def6db85cfc170e9419bd5953bd038`
- Focus report: `ce5c9e2d6785b6a3319189e940a97c7e2e13725e30872a5b2ba9d945675281af`
- Note: `55ec9c50b5eb302977381cc62ab81b80a78e89e782c06038558327eafd4182a8`
- Evidence inventory: `c89df87d5da7bf9b880b8ef65f98d3e585de4e6519d1a0015ec3b575e17a8023`
- Replay plan: `8f51603de2667d96d1632638febc9a518aaca2734e7562f7d958750c7313aa24`

## Claim boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact tree construction,
telescoping, finite controls, and verifier passes receive no asymptotic attack
or lower-bound credit.

## Next action

Seek a nonlocal elliptic batch transform on the two compact divisors rather
than another line or leaf streaming law. It must compile the target list and
emit `product_j U(x(T_j-P)) mod U` in softly `O(n+N)` work without visiting
the `n`-by-`N` pair grid or assuming represented resultant/multipoint inputs
for free.
