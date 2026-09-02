# P1436 autoresearch focus harness V86 result

Date: 2026-07-29

## Result

V86 binds R137 as the 73rd closed frontier lane and routes the highest
priority action to
`s29_three_plus_five_plus_nonzero_torus_c5_selector`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R137 applies multiplicative Cauchy-Davenport in the prime-order pairing
image. If a structured color has `m` atoms, then its all-five-from-color
accepted subset satisfies

```text
|A^5| >= min(q, 5m-4).
```

At `m=B^(3/4+o(1))=q^(3/20+o(1))`, this gives a deterministic
`B^(3/4+o(1))` positive-support floor without computing a discrete
logarithm.

A nonzero represented binomial has at most one root in the prime-order
group. R136 requires the root union along the all-nonzero path to cover
either the complete positive support or its complement. Every exact
structured binomial zero-test tree therefore needs path depth at least

```text
5m-4 = B^(3/4+o(1)),
```

which misses polylogarithmic arbitrary-target work regardless of the
expanded support of the path product.

Under the frozen uniform-random support model only, the support has
`q^(3/4+o(1))` points. Kelley root bounds then force depth exponents in
`B` of `15/4`, `5/4`, and `5/12` for nodes with at most two, three, and
four modes, respectively. The three- and four-mode statements receive no
structured-factor-base credit, and the exponent gap closes at five
modes.

All eight pairing decks and 30 active colors replay exactly. Singleton
colors have one fivefold product and two-atom colors have six, attaining
`5m-4` in every control. A selected product is in C5, its inverse is
outside C5, and both follow the same all-nonzero path through linear
factors rooted at the remaining color products. These finite controls
receive no asymptotic credit.

R137 closes all structured binomial zero-test trees and model-bound trees
through four modes per node. Structured three-plus-mode nodes,
five-plus-mode low-SLP circuits, and nonzero-value Frobenius-coordinate
branches remain open. No source index, relation-rank construction,
factor logs, identical target descent, Pollard-rho improvement, Shoup
improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R137 tests: 10 passed.
- Full harness tests: 103 passed.
- Full ECDLP suite: 617 passed.
- R137 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R137: 62 receipts, 1,121 bindings, 0 mismatches.
- V86 frontier preflights: 73 provided, 73 closed.
- R137 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V86 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R137 producer: `0120bb27e2fcc271d197f2eaae4b2adc5fd6ca65590973e9fce35c8f7221fbb1`
- R137 report: `9e52400ba4ff17fe8d07e2f48f00e59ab8dd977ae9cc21a9a09b119c6cc8588d`
- R137 frozen interface: `e8cf340e380f99d57d2db08e24bdaa507f902d7fd93df02b6773603e43f310e0`
- R137 cost ledger: `45f178e841b3234607e7ae5706384f2553171be74de4a544e6eecc3973ae2ae5`
- R137 source replay: `38cd8666637b5e7e3a6864a6177a160bc27d6b1395840960b423321db8fbd3d4`
- R137 controls: `9a3e9e71b2fd984d21feac2c1650adfd8cee3559a01ee62893f280c868ca1d31`
- R137 logs/descent: `7e9d38cb8e3a6aec6d851837644ea6a813e9a0bae275bf4da6f0926140a3c6d6`
- R137 tests: `fb58d28c0d56125173b86c8eb2ce44aa11352fb76e271ae1a3b88487dc3d7a05`
- R137 gate: `88669b1d4a1177c747a3bddb2cbdcaa22352a3c705048417d07f0a80a6bb246e`
- R137 parent: `6000356936813d690d767e7952301a46c6738a460941ece6c6312a4a24fc5a47`
- pinned DeVos PDF: `924bac0a6b5a1e9379e38a53f3ab78a6ad129beb6fddb12396c8275403b8c511`
- harness: `07ea9d1e55ff7ff4b90f1d0d20ca89660aa90a5c0680626b9fa0cf950c15f7c5`
- harness tests: `28192f728d6847d91955145b8cd6f12e0871610184bd01aa33707307a24ba0e1`
- report: `5b01afe83c5ecc7db7456bedbdcf4150c3689d9a0a76ea86280b93238562edce`
- note: `d385eaf724760cbbf08da60f9f30d891df023e3541bca57be9dd9143c6256b48`
- inventory: `201fa34bf71464511b92b36b9b1129d2f26f6160b0828179bfb791f2319e297d`
- replay plan: `1324a99a6460ed27c674900c6d4531b7f8836cadcc1a22b83b86e3e181beabf5`
