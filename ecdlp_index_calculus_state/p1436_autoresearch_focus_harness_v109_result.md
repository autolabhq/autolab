# P1436 autoresearch focus harness V109 result

Date: 2026-08-01

## Result

V109 binds R160 as the 96th closed frontier lane and routes the highest
priority action to
`s52_coordinate_specific_s7_reverse_ffe_source_locator`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R160 asks whether the missing R159 positive-C6 source locator can remain an
encoding-invariant generic-group algorithm. Given a prime-order DLP challenge

```text
Q = [x]G,
```

sample public `a_i in F_q` and nonzero `b_i in F_q`, and define

```text
C_i = [a_i]G + [b_i]Q.
```

For every fixed `x`, uniform `a_i` makes the `C_i` iid uniform. Rejecting zero
and equal-up-to-sign encodings gives the conditioned ideal factor-base law.
Each returned positive-C6 source for `tG+[s_j]C_j` supplies the exact row

```text
v-s_j e_j
```

with known right-hand side `t`. The admitted R159 theorem gives all-column
coverage and full rank with probability `1-o(1)`. Solving for factor logs
`ell_i` then recovers

```text
x = (ell_i-a_i)/b_i mod q.
```

With `q=B^5`, the assumed locator setup and dense solve cost
`B^(9/4+o(1))=q^(9/20+o(1))`; the complete target batch costs
`B^(5/4+o(1))=q^(1/4+o(1))`. Therefore an encoding-invariant locator at the
requested caps would solve generic DLP in `q^(9/20+o(1))`, contradicting
Shoup's `Omega(q^(1/2))` classical generic-group lower bound.

This excludes only locators simulable from opaque encodings and generic group
operations. It is not a lower bound on coordinate-aware summation-polynomial,
resultant, finite-field elimination, or FFE circuits. R116 already supplied
the C3+C3 source reduction, and R148 already charged the `B^(7/2)` scan and
`B^(9/2)` pair-table endpoints; R160 does not relabel those prior results as
new.

All six public-curve generic-embedding controls cover every column, attain
full rank, recover and publicly verify every factor log, recover and publicly
verify the embedded DLP, and verify identical positive-C6 descent. The finite
locator explicitly scans the C3 list for every target, with asymptotic batch
cost `B^(7/2)`, so the controls receive no attack credit.

No coordinate-specific source locator, deterministic hash-to-curve transfer,
unconditional generic-prime family algorithm, Pollard-rho improvement, Shoup
improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R160 tests: 16 passed.
- Full harness tests: 126 passed.
- Full ECDLP suite: 915 passed in 108.977 seconds.
- R160 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R160: 85 receipts, 1,600 bindings, 0 mismatches, no missing
  or duplicate rounds.
- V109 frontier preflights: 96 provided, 96 closed.
- R160 obligations: 17 of 24 passed; lane admission false.
- R160 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V109 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R160 producer: `c303909641700a141714fde1ef473bb6abe4331ab9cfaf0e53161aa13b1fe7b1`
- R160 report: `8b061717b7822ce969433b17003f00fb2e8e4644305cda007b70ca2213f4f48c`
- R160 frozen interface: `868b534a8684cbd63cf72d320102af5d6bc568510bcd64b44698ce5cbea4ca50`
- R160 cost ledger: `834312ee1a28f7dd7418bd09050af8c39ccbbc816731b7cc00fbfebc9f3814ab`
- R160 replay: `8bcfcc063412cdc3609748ea4843daa01dacef622433468ab201d10db2e2fce7`
- R160 controls: `84b61b9ef84ffbcc88acbf0f24cd41be48d3aa0f495c1de18e2ac2611b83fafa`
- R160 logs/descent: `3a2980834f0b87e5c68d6ffa5b7cab03da27f1fbaaf8098c1bfe00001a0ee196`
- R160 tests: `d568f454e0808498bc43e54b357b29a708252cc42cf53a9e2f0dfdce55a76633`
- R160 gate: `e4d13e79feb831e0d65e734170def4b5e79a6f53edaeb8c49c9bead4e739ec92`
- R160 parent: `9089a77246d129094800f1c941a92d355a85d2ad0d1c36c86fb2a3a5bcdc67c2`
- Shoup primary PDF: `89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3`
- Harness: `4f236e87c4a1ef8f70f2142ea8914855e927ce562f021091fc9911d66a886224`
- Harness tests: `4aee9505e4716f614a6d3680e57a306ec0bf217dc81e7e542e1bedca71085c75`
- Report: `84bf14c68503d479d6fdbbd9dbc8db98562e4e7d409f7ab91b163b17e5cd20ec`
- Note: `dffea522fd441f3a4b813fd4c918c30781e7a9fc32d097f8413a02cd1feb21f2`
- Inventory: `6385e9a1876fa1e70ed2ef13f00573c8ca218eb7cd0ffe428a92330d68af4af8`
- Replay plan: `3a3cd656dbfad8f955c8a1f31a9e882067fe8424eaafd0720d6efa04f0c9fdee`
