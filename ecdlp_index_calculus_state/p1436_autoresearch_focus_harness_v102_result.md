# P1436 autoresearch focus harness V102 result

Date: 2026-07-29

## Result

V102 binds R153 as the 89th closed frontier lane and routes the highest
priority action to
`s45_reverse_only_signed_marker_density_transfer`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R153 replaces the A deck by a public deterministic known-log family closed
under negation and replaces C by `C union -C`. These changes preserve the
M6 exponents: `|A|=B^(1/12+o(1))` and
`|C|=B^(3/4+o(1))`, up to constant factors.

Let `mu(s)` be the multiplicity of shift `s` in the symmetric A6 batch,
let `U` be the indicator of the inversion-closed C deck, and let
`K=U^(*5)`. Then `mu(-s)=mu(s)` and `K(-g)=K(g)`. For C atoms `a,b`,
R153 freezes

```text
D_s(a,b) = 6 mu(s) K(s+a-b)
c_s(a)   = mu(s) U^6(s+a).
```

All eight finite controls verify the exact identities

```text
D_s^T = D_(-s),
sum_b D_s(a,b) = 6 c_s(a),
sum_b D_s(a,b) ell_b - c_s(a) ell_a = c_s(a) s.
```

Thus, if a reverse-adjoint signed marker operator is constructed for every
shift, its opposite-shift action also supplies the forward block action.
A separate tangent operator would not be required.

This is only an interface reduction. The current controls expose a rank
deficit rather than evidence for log recovery: the eight stacked ranks are
`[4, 0, 0, 0, 0, 0, 0, 0]`, and none is full C-log rank. The uniform
density estimate predicts `B^(3/4+o(1))` relations across
`B^(5/4+o(1))` structured rows, but that estimate is model-bound and is not
transferred to the deterministic family.

R153 supplies no reverse marker circuit, signed internal FFE elimination
DAG, targetable density theorem, structured generic-prime rank theorem,
factor logs, or identical target descent. It consumes no candidate DLP,
root, count, marginal, rank, or source oracle. No Pollard-rho improvement,
Shoup improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R153 tests: 12 passed.
- Full harness tests: 119 passed.
- Full ECDLP suite: 807 passed.
- R153 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R153: 78 receipts, 1,456 bindings, 0 mismatches.
- V102 frontier preflights: 89 provided, 89 closed.
- R153 obligations: 16 of 27 passed; lane admission false.
- R153 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V102 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R153 producer: `fa55c47f6b0c8a4a6071580c3ba6c519cdb7150d692ee16253fdf8150694e646`
- R153 report: `0e5c8feb4498a16b02fd1ffb5e2f7df4a64d20097c62f35d99abef84a120afce`
- R153 frozen interface: `81aa786dc9431f6e0bf83de1a43a2030996e06c73faf0f2900fe8ade4e19754c`
- R153 cost ledger: `0a2c48b7604bcef168acb850ab6de69a09d927c4d8558a3c2d8ee7ae60ffd859`
- R153 source replay: `6e66783f3ea159ad0b522b46075fe11e0c20b1c82428ed11aa31b0a72fb96f26`
- R153 controls: `abf91eff4a023d400fdb56caac7927a496cf83b5cba205536118cfdc4ef00f62`
- R153 logs/descent: `2bfb012acdbac35964775c8fa1ae38d519c3e9ebddcc81b969c791b97e86a569`
- R153 tests: `5d62c7b1be2f5267ae2d66a931ae6a7034ff2ebc6b4c97bc179587b2f7a47a79`
- R153 gate: `afaeede72a11b0ae870a0551f9d5ec5bbdf94099f53dede5b5327fbfb9b0bf6d`
- R153 parent: `1d46ec932cf58564ebbbccfdee888b7ffacd482d47e7077a1b33feb4dd382c87`
- harness: `cda2735ef70fecb1185003013a36f07d932935ca327ca2515f5de5dd54724fd1`
- harness tests: `a4482af34a12731a10584798579adc9b547925c2cb8ab547e04a1dfe97094468`
- report: `7a71a8d9981d63a6f635df747b4ae97ada64987d4414dce66198d03e1c552c85`
- note: `13e8789ee9ea1b1cf6b0963f302763e3d4fceab990552659e8e8f978c33265c4`
- inventory: `2da1ded4e9b1f20df3bd9ba1e4b07bf7e3c36969ead6297fabc78a5e0224da36`
- replay plan: `150060855f8d1661d79fc6ae75418ed15d9b83181de4a444d7a2ece15a7f79f1`
