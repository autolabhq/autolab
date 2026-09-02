# P1436 autoresearch focus harness V100 result

Date: 2026-07-29

## Result

V100 binds R151 as the 87th closed frontier lane and routes the highest
priority action to
`s43_weight_parametric_bidirectional_marker_operator`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

Let `Z(w)` be the vector of aggregate known-target fiber counts as a
function of logarithmic A/C atom weights. The aggregate marginal matrix is
the Jacobian

```text
M = D_log Z(1).
```

For an atom direction `x`, `Mx` is the first-order coefficient of
`Z(w_a(1+epsilon*x_a))`. For a row contraction `lambda`, `M^T lambda` is
the logarithmic gradient of the scalar function

```text
sum_i lambda_i Z_i(w).
```

The two actions satisfy the exact certificate

```text
lambda dot (Mx) = x dot (M^T lambda).
```

All eight actual finite R144 systems pass this identity and replay their
selected full-rank systems, factor logs, and shifted descents. Their
matrices and labels are verifier-only.

Baur-Strassen supplies constant-factor nonscalar overhead for
differentiating one complete scalar circuit. Wiedemann accepts a matrix
through operator applications. Thus, conditional on one frozen
scalar-blind circuit applying both `M` and `M^T` in `B^(5/4+o(1))` work,
the `B^(3/4)`-dimensional matrix-free solve costs

```text
B^(5/4+3/4+o(1)) = B^(2+o(1)),
```

below both the `B^(9/4)` setup cap and `B^(5/2)` rho scale.

The condition is decisive: Baur-Strassen differentiates setup too. It does
not by itself provide a weight-independent setup or reusable division-safe
tangent and adjoint state. R151 therefore supplies an exact interface
reduction and conditional envelope, not the missing marker-count circuit,
generic-prime rank, candidate factor logs, or descent.

Primary references are pinned locally:

- Baur-Strassen PDF:
  `828e8c9eb30af0089c48b1a5bcac3990bc0e21d3b68d4e343490b0fdaa38ecae`
- Wiedemann PDF:
  `8ec0b8a8b35c02bd4a84236129ab991fdbcca4b4feb8332c22b08dbdefc1218a`

No candidate DLP, root, Fourier, recurrence, algebra-state, count,
marginal, rank, or source oracle is supplied. No generic transfer,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R151 tests: 11 passed.
- Full harness tests: 117 passed.
- Full ECDLP suite: 782 passed.
- R151 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R151: 76 receipts, 1,411 bindings, 0 mismatches.
- V100 frontier preflights: 87 provided, 87 closed.
- R151 obligations: 16 of 25 passed; lane admission false.
- R151 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V100 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R151 producer: `cf632dcddc7b2bc55982ddcf5bdf24ed6af91904d13493233a9f320675b93e5b`
- R151 report: `3a8f5134ac731fb1481c358148986acdf355bfe1c8370d7e8c00d6a9ad95d14e`
- R151 frozen interface: `3bff6dff925cc884ee3753004ce81b5774348bba75c3558c7f31eb0b7f825d2d`
- R151 cost ledger: `810f77f445eb7438a9421cb3f865a385d5dabaf09eea94de281c787d442b06bb`
- R151 source replay: `5dec8cda3eab0a81d2dfd84d1e8a140778f1cc69187b3f1af1fa543cfd90c792`
- R151 controls: `5b64b07215b75e16e52e136e81e58ae988a13446331fc5499f67a68f661f33c7`
- R151 logs/descent: `9570ebf6009e680ef1d0840b775f56c7bc0021fd7219b240a84b5ada0752e0b6`
- R151 tests: `9c68d8c9b17c7fc9e85f0f9e0ffe6416050b342bf8e4439e9e50b36385323be6`
- R151 gate: `be163b44b889c67206a9a1eb73a02a6df911afc2b018f7848b145403285ad903`
- R151 parent: `016abc9cbaa7d1de107eb742d13607a35983e429fcc70f0475b83af8c5a36d0f`
- harness: `163a75f275afd7512b784546424a14d698b7c72eecf18158d4bddaed58b96abe`
- harness tests: `00b8e4eff61f38579b1da5a1fd1a3db7210b61f5776b918eb3a746d57a1e8220`
- report: `0f7e29fe263a9345040ae6dd57958864d389f604b94adea3d4bd971d14803354`
- note: `e148fe86a5773d47bcf1ae5a47738d494c9def880ce06393ca4692f8b75a6b24`
- inventory: `589e3c6d8be3c50f921bf661c66b2bf12c39e44d02da2a11e744bc7f2e16c8aa`
- replay plan: `63cceb48b49d4abc06d9896f52ec9666667500b2da59944fb1b00121849475c9`
