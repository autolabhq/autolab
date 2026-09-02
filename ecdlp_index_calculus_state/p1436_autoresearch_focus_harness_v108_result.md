# P1436 autoresearch focus harness V108 result

Date: 2026-07-29

## Result

V108 binds R159 as the 95th closed frontier lane and routes the highest
priority action to
`s51_batched_positive_c6_reverse_ffe_source_locator`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R159 replaces the R158 A6/projective relation-event process with direct
known-scalar targets. For factor-base dimension `d=B^(3/4)`, the positive
unordered C6 source universe has

```text
M = binomial(d+5,6) = Theta(d^6) = B^(9/2).
```

For iid uniform labels in a prime cyclic group of order `q=B^5`, the
expected number of distinct-source endpoint collision pairs is at most
`M(M-1)/(2q)`. Consequently, at least `M/2` endpoints are unique except
with probability at most `2M/q=O(B^(-1/2))`. Global coefficient-map
injectivity remains false and is not required.

For each factor-base column `j`, R159 chooses an independent uniform
diagonal shift `s_j` and queries positive-C6 sources at

```text
tG + [s_j]C_j
```

for independent known scalars `t`. With

```text
T = ceil(8q log(d)/M) = B^(1/2+o(1))
```

queries per column, the conditional probability that any column is
uncovered is at most `d^(-3)=B^(-9/4)`. The complete target batch has
`dT=B^(5/4+o(1))` points.

If `v_j` is the first unique source returned for column `j`, its relation
row is `v_j-s_j e_j` with known right-hand side `t_j`. The selected source
is independent of `s_j`, and the relation matrix is

```text
R = V - diag(s_1,...,s_d).
```

Conditional on `V`, `det(R)` is a nonzero multilinear polynomial with
leading monomial `(-1)^d product_j s_j`. Its full-rank failure probability
is at most `d/q=B^(-17/4)`. This closes the iid zero-coverage and full-rank
theorems without a sparse-matrix contiguity assumption or rank oracle.

Conditioning iid labels to be nonzero and pairwise distinct up to sign
gives the ideal rejection-sampled hash-to-curve law. The conditioning
event fails with probability `O(d^2/q)=O(B^(-7/2))`, so the theorem
transfers exactly to that ideal sampler. Pseudorandomness of a fixed
deterministic hash-to-curve instantiation remains unproved.

After factor logs are solved, the identical positive-C6 mechanism applied
to `Q+tG` yields

```text
log_G(Q) = sum_i v_i log_G(C_i) - t.
```

All 12 public-curve controls cover every column, attain full rank, recover
and publicly verify every factor log, and verify identical target descent.
No candidate DLP, root, count, marginal, rank, or source oracle is used.

The controls explicitly enumerate the positive-C6 endpoint map at
`B^(9/2+o(1))`, above the `B^(9/4)` setup cap and the `B^(5/2)`
Pollard-rho proxy. A batched reverse FFE unique-source locator for the
`B^(5/4)log B` targets is not constructed. Therefore no unconditional
algorithm, Pollard-rho improvement, Shoup improvement, or ECDLP
breakthrough is claimed.

## Verification

- Focused R159 tests: 16 passed.
- Full harness tests: 125 passed.
- Full ECDLP suite: 898 passed in 102.731 seconds.
- R159 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R159: 84 receipts, 1,572 bindings, 0 mismatches.
- V108 frontier preflights: 95 provided, 95 closed.
- R159 obligations: 20 of 29 passed; lane admission false.
- R159 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V108 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R159 producer: `8889acea5284ee2ca70d147f35c90d08d3e3612f9749984d6c10856c74efa448`
- R159 report: `869f04cbcfa5f918e2d96000508cb02c3ffb1266cbd9e0001f2c14d5a0ae7860`
- R159 frozen interface: `c5df20238ff3107b384d26c33b882b51754089bd4ff517c0220b1dc67307fc43`
- R159 cost ledger: `b42f8c6818a92fe269ac9642dca1225faf0a38266824550f0768be47640a00e3`
- R159 source replay: `08a247a446a0c7d1d0ca8abe0016b6ec825165e911341fdb0a54be9a5e1685de`
- R159 controls: `d95dad07fbe68c1f4196535e817a998391e15abd50f918a2b7777ca6a8a07e32`
- R159 logs/descent: `4a9258ebff17de5b77d9be2db0c0943188e134077e7ea1e93d9069d5982d5871`
- R159 tests: `d8ec783bc8b0c2827d40251364d1bce2d495aeda67c804ee9ab604b4ad60da43`
- R159 gate: `58b37a59c6f2a983ab2a0a73bdba444fab667aa24eff91eb49f448453a91fda9`
- R159 parent: `7dc129664690bf788df3671a36c47d0f47abf94bd331bcad5a368fd9ff5040c0`
- Harness: `634f664324a9c3ba27f3b95bd87fa45ed46e3cdc0186873abed614358411b0c4`
- Harness tests: `3d16c2b0ef81fb361a99ea50fb85ed8e6c8cc9dfa1ef15eb4fb0a4de13e8b277`
- Report: `30177edb3c1cad61755ca12388943aea97cdd4d45ec081d42b3200a9fcd38306`
- Note: `7f3b68dd55fec7adfc85b109be8764885f010d753fe47cdd2b6d05d789f14941`
- Inventory: `b6d9515e39670367c05c0554e94eea48de7fafbb919c599119f59eba983dd4c1`
- Replay plan: `31b0cdb3d4538994dcfe991324086bb0b036110593492cdd4a28facbdbfc5ffe`
