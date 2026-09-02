# P1436 autoresearch focus harness V73 result

Date: 2026-07-29

## Result

V73 binds R124 as the 60th closed frontier lane and routes the highest
priority action to `s16_coupled_nonlinear_torus_c5_zero_test`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R124 rewrites ordered C5 counts as translated inner products:

```text
u^(*5)(y) = sum_x u^(*3)(x) u^(*2)(y-x).
```

For a fixed C2 kernel, any universal linear sketch of arbitrary C3 vectors
with linear exact-count decoding needs at least the rank of the kernel
circulant. Over characteristic zero, a proper nonempty binary deck at prime
order has nonzero Fourier coefficients in every mode by the prime
cyclotomic-polynomial criterion. The C2 circulant therefore has full rank

```text
q = B^(5+o(1)),
```

above the `B^(9/4+o(1))` setup cap.

Five controls verify direct five-tuple counts, translated C3-C2 queries,
full rational circulant rank, and selected full finite-field spectra. These
finite controls receive no asymptotic credit.

The theorem is deliberately scoped. It does not cover nonlinear
preprocessing specialized jointly to `(u^(*2),u^(*3))`, nonlinear
membership-only decoding, adaptive data structures, or general arithmetic
circuits. Those coupled nonlinear routes remain open. No Pollard-rho or
Shoup improvement and no ECDLP breakthrough is claimed.

## Verification

- Focused R124 tests: 10 passed.
- Full harness tests: 90 passed.
- Full ECDLP suite: 474 passed.
- R124 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R124: 49 receipts, 858 bindings, 0 mismatches.
- V73 frontier preflights: 60 provided, 60 closed.
- R124 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V73 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R124 producer: `c3aabfe9d2ce8553c3facdc7fcf2f8b669b09ec48a6144772068f264e4ab012d`
- R124 report: `99df2561ba0a07506f36c67d7152c299c6954272508eca1035ea3cdbe061dc6d`
- R124 frozen interface: `129aac9bf7868c3305ef38d2af4443d2c6e8227f75f54b8b1aa09a1b4a323334`
- R124 cost ledger: `b8b5e44c76b7abb85b9f32acf08a1ebc00ce6a5729710975f7eb3362d40b906f`
- R124 source replay: `162d813653f83c782ec69334d6876f1bf6bf2cab9a4d1a9d87e1ff768110e094`
- R124 controls: `87011644cd14261d3f0afd45d367a18a39b484cdfc9b06a91de1e4812e07dfc2`
- R124 logs/descent: `dc170b78fc8460a65709fb3d3537d56571159f68e72c8b9b24c225b3c15b0e7a`
- R124 tests: `1fc02a49f7e9b6b745d9882fe087f948c23d17b365b8de7b77d861c650eac973`
- R124 gate: `30a3ce3036bab96731430a662cde3ded8e63b9fe1326942a5091148b3e939938`
- R124 parent: `eb31ae1bccf81431d27bcef8d6fe27ac3269f077bcf115eecab61526bc7a752a`
- harness: `bdfadf52972ea3332fe46bd2cab6595cbe5d04804c73366da43e2d479e5f964b`
- harness tests: `65e56653ff2c16779664ad272e035cadbf098f67460245bde17b80927511b19f`
- report: `def0332f4f42f905bd6d6894a26d749b6d47a8e85571dc5fc8efb1fe99d2b04f`
- note: `4f313e0f7f7e76a9b52a25301cf960cd51f429cc616afebbd9777d4309ee157c`
- inventory: `615017f7382e3e2ff0da03426acf26a842198144938be83d0904e13284e3c019`
- replay plan: `f58267f35e0bb196d5059512e35adcb6120437a607e78ae611a6e41e6d4ed4a5`
