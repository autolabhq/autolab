# P1436 autoresearch focus harness V76 result

Date: 2026-07-29

## Result

V76 binds R127 as the 63rd closed frontier lane and routes the highest
priority action to `s19_cap_tight_singleton_c3_target_router`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R127 parameterizes a balanced correction hash by `H=B^gamma` buckets. The
optimistic represented work for one C2-C3 bucket resultant is

```text
B^(9/4-gamma).
```

All-pair evaluation costs `B^(9/4+gamma)`. Routing through
`R=B^rho` pairs costs

```text
B^(rho+9/4-gamma).
```

Even ideal quotient-style `R=H` routing remains `B^(9/4)` per target.
Polylogarithmic query requires `rho<=gamma-9/4`; with nonempty balanced C3
buckets this isolates

```text
gamma = 9/4,
rho   = 0.
```

The surviving corner is therefore a cap-tight `B^(9/4)` C3
singleton-bucket index with an implicit constant-pair arbitrary-target
router and source locator. R127 does not construct that router.

Twenty-four actual coordinate-hash controls verify all-pair and
target-bucket-routed membership, one empty query per control, C2+C3 source
replay, and total symbolic bucket-resultant degree `|C2||C3|`. No
pairing-image discrete logarithm is consumed and finite controls receive no
asymptotic credit.

The tradeoff covers independent represented resultants, not shared
transposed evaluation, adaptive probes, or general data structures. No
Pollard-rho or Shoup improvement and no ECDLP breakthrough is claimed.

## Verification

- Focused R127 tests: 10 passed.
- Full harness tests: 93 passed.
- Full ECDLP suite: 507 passed.
- R127 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R127: 52 receipts, 922 bindings, 0 mismatches.
- V76 frontier preflights: 63 provided, 63 closed.
- R127 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V76 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R127 producer: `4d4d88bbd2f0f98f1eca9f2986d6ebbb50f5fef7a3af9cc8a5a9c4cd9e802a7b`
- R127 report: `373a69c5a6a0dcdb4b12f68d1a1bd71b74f4c8e1971329dba64f103fe4ae138f`
- R127 frozen interface: `c91e94d84a12e920d0b59619b730a9e81b4ff2aa9bc8fb46014a2f39594fad4a`
- R127 cost ledger: `2c1cf3b67084f01265dfde52c9c32e328bb7e30be06385f4752af8f8b5d9d45f`
- R127 source replay: `293fc34625268ea3cc34f7e6bdd28b8f46ced80f5f00855855639525f0be38e7`
- R127 controls: `536b8b366e6705ae4c7a7f5c8b46630b85a3b6dd827040202cb0c726d99ae758`
- R127 logs/descent: `7d3b8d62d4b77dcdbd92e1a2ec84df10dd0a106832ec6ee08ef022967ad6ee13`
- R127 tests: `f0fba2610d8f21b0559cc67504fb78d3be12710fdfa5003e01ad35a95fab8127`
- R127 gate: `729bb912d2db0bd518375bdbce0f7cf0fa5d068af854224b8306a7c8be6db4d8`
- R127 parent: `2b052667b1ed2687ce048c4059c13ee9be4804fc096f310bb537eaf0a10776e1`
- harness: `5d9a8becdfd0a6edfb981938fd84d2ff4c1e3502d5c3c3a83f6640bd39df4534`
- harness tests: `f52057271fe2805772483c500e058de110d8be0deebac7b5e74de629b7445845`
- report: `97d347e633b816267e4473c8177ae4ea728573e02d0cffde988c428420122d2c`
- note: `8ace9af99c76e630334abd69e45985117141f465f350a7ca58ec6f482d95ca14`
- inventory: `929bc48e32fa65aeeba4cd44bc250efba881671cef6cabf97453ca051c7ec73d`
- replay plan: `3a1433080e8751dc06b22142d145ca08647c33503f217e4454291de4751917b3`
