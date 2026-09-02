# P1436 autoresearch focus harness V106 result

Date: 2026-07-29

## Result

V106 binds R157 as the 93rd closed frontier lane and routes the highest
priority action to
`s49_public_short_relation_rank_reverse_ffe_descent`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R157 transfers the R156 singleton-row construction from scalar labels to
public prime-order curve points. For a signed C6 coefficient vector `v`,
every unordered source with that vector is obtained by distributing

```text
k = (6-||v||_1)/2
```

cancellation pairs among the `d` inversion pairs. The exact fiber size is
`binomial(k+d-1,d-1)`. If the feasible signed coefficient map
`v -> sum_j v_j C_j` is injective, an endpoint is singleton exactly when
`||v||_1=6`.

All 24 hash-to-curve controls satisfy the coefficient-map injectivity
hypothesis, the fiber formula, and the singleton criterion. Relations are
discovered by public point equality. Their rows contain only integer source
multiplicities, and the right-hand sides come from known-log A6 shifts. No
C scalar labels, BSGS verifier, DLP oracle, or root, count, marginal, rank,
or source oracle is consumed.

Twenty-three of 24 controls attain full signed C-log rank. All 23 recover
every C factor log and verify the recovered values by public scalar
multiplication. Full-rank counts by logarithmic factor are `[7, 8, 8]`.
The lone failure has dimension three but only two projectively distinct
rows; it has no residual algebraic nullity. No control remains dependent
once projective row count is sufficient.

The finite verifier explicitly enumerates C6 endpoints. This costs
`B^(9/2+o(1))`, exceeding both the `B^(9/4)` setup cap and the `B^(5/2)`
Pollard-rho proxy. The factor-log successes therefore receive no attack
credit. An asymptotic short-relation rank theorem, reverse signed FFE
operator, and identical target descent remain open. No Pollard-rho
improvement, Shoup improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R157 tests: 14 passed.
- Full harness tests: 123 passed.
- Full ECDLP suite: 864 passed in 103.242 seconds.
- R157 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R157: 82 receipts, 1,534 bindings, 0 mismatches.
- V106 frontier preflights: 93 provided, 93 closed.
- R157 obligations: 15 of 24 passed; lane admission false.
- R157 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V106 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R157 producer: `0f022ee1c90e5a2feb61f213f71e3b1345015b89611bcf5d738dd2242426cba9`
- R157 report: `50cff20e88c953f76a373747a4f11f0dd21a36a0e526bf25c5bbfdaa8997b7d7`
- R157 frozen interface: `efb0d428c915f93b7c03057486c91d4263d695a536af12204e04fd136f0ff708`
- R157 cost ledger: `ac54b8756ce08ecdd3e659f1f60bb969ca7db07ba19f8b630e0b3719628466f9`
- R157 source replay: `cc1e0613c6c7f493400fd403e840a000fbde60770e6526e127e79d329dc8d3de`
- R157 controls: `4d78e859ed61f7e4ac7d65f381a027e32f89e8e89a0e673e27daac8702bb1deb`
- R157 logs/descent: `e8ce99ff9582fe5b90fee2bebc12d4641a203b1df2828513c1db8232568c493f`
- R157 tests: `d39dcdff1ec2f809ea275a28066f6f5f2cb0420f578f3f8c13837e05f38836d2`
- R157 gate: `ea7757fa1adfb06c8d80c5be9224419220a7f96570f069613e0631de609a38af`
- R157 parent: `200f0a17b190385aa6f7e5c3ef5610b77631f1ac6df8b048df4e06a7a0cb23f2`
- Harness: `c83600c783b23b29cec334756897f85f44a40b8575bbbb4574743de5ef3f57b6`
- Harness tests: `c2536505c2c5dac9b5570d2abac0dd3b279fa886d0b76b79f673307ee1144368`
- Report: `117b49e980548c914c0cd4653ceb313dac8ea3e082e53ffc10a5cab084739e58`
- Note: `0eed45d6278c4fcebe16a5df443d089066de9654835d9a5406c3e215b65e9071`
- Inventory: `a90560e7ebd2594bf574561de517dd333cd9d6b82c381d55cf678d857633497a`
- Replay plan: `ee9ea4d53cd86393d5f69fb6cad4d18df220d9938f5e9a2a9d69be4134bae753`
