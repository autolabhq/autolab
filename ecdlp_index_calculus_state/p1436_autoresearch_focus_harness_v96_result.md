# P1436 autoresearch focus harness V96 result

Date: 2026-07-29

## Result

V96 binds R147 as the 83rd closed frontier lane and routes the highest
priority action to
`s39_shared_transposed_multi_target_valuation_marker`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R147 retains all ordered `C3` occurrences as repeated roots of

```text
P_occ(X) = product (X - cayley_parameter(C3 endpoint)).
```

Its degree is `|C|^3=B^(9/4+o(1))`. Pairing two occurrence roots through
the Cayley group law gives a formal pair resultant of degree
`|C|^6=B^(9/2+o(1))`, but that resultant is never materialized. Its local
valuation at target `tau` is exactly the ordered `C6` fiber count.

All repeated-root multiplicities replay on eight actual controls. Each
control has twelve exact positive valuations and six exact zero valuations,
with no candidate root or DLP oracle and no asymptotic credit.

The Moroz-Schost truncated-resultant bound is then charged only as a
standard componentwise or direct-product route. For bounded local order:

```text
one target:             B^(9/4+o(1))
one known-target batch: B^(11/4+o(1))
full relation stream:   B^(7/2+o(1)) = N^(7/10+o(1)).
```

This closes independent per-target truncation but is not a lower bound for
a shared transposed circuit, modular-composition data structure, RAM, or
cell-probe model. The remaining experiment must share work across the
complete `B^(5/4)` target stream and emit exact counts plus `A/C` marginals
without replaying degree-`B^(9/4)` work per target.

No shared valuation index, marginals, factor logs, identical target descent,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R147 tests: 11 passed.
- Full harness tests: 113 passed.
- Full ECDLP suite: 734 passed.
- R147 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R147: 72 receipts, 1,323 bindings, 0 mismatches.
- V96 frontier preflights: 83 provided, 83 closed.
- R147 obligations: 15 of 25 passed; lane admission false.
- R147 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V96 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R147 producer: `cbd08abbdb85025c2353be2253284d79896c6a1709690aa31d01a90622e1a49f`
- R147 report: `a419f994907d392b8d8cec7a3af6dc0f3a67769350be7c1044e3de4320a69977`
- R147 frozen interface: `0e1c41e3c0ddcb6808977a52d50be72c1a2cef36e186599d3bc5337a793cb26b`
- R147 cost ledger: `dbb23343d92b1d757bce760fefeed04aa47719779a4f153fe8a2c6b25dac2c10`
- R147 source replay: `30e839872cd09e34d12224662c3e3988c5cb478377da1632eb2a4f0c985ef643`
- R147 controls: `d61405a35dc3efec0c9c49dd3be7159718f052ea7daa601ed56d64a1628fdf65`
- R147 logs/descent: `4389c787e70da680929bb59faaf1550f22d435f253e8f002dc0d50eb1df2f31c`
- R147 tests: `da3d847481dd5bc38386f4f50ecb3c9d933e8e6ec0afee0b5c697813e2174a75`
- R147 gate: `7398ae8894fc40f710f76e2ec0e6f31726f283ca6fb9153f5258e0561537a6fe`
- R147 parent: `a1fd7d4c9c4b0f3b75b97c6e726187bda9db382c1f58910374009dc322f7b7cf`
- Moroz-Schost PDF: `160c68cfbb413ca27352a064cbf2d27f7ad4ed6a210c3d6ead2770e00204b709`
- harness: `32d3b1623c2b00b0bcedb2a7a0b521133f8ff178363d63bd529bc71be2c69a07`
- harness tests: `072c0b0c0fa1afa815d1a8e05ec2279ebe5b296214dd133a9a9462340b9fb9b2`
- report: `7bcac78bf6f40d16097bba6c7ba129a54ac9bf168305ec1c553712b38a850a1d`
- note: `da0f1b7ae42afe4e8880b08429515176bcad68367da78c213fe36bab9261f2bb`
- inventory: `acbdd7b358b98de45ad9f32390c92638d61a70610cdb9819518de8878f8b9ec9`
- replay plan: `81dec54d160ed0ff710ad51b31e22dec3db5659a46315f819281e4d6b8823382`
