# P1436 autoresearch focus harness V75 result

Date: 2026-07-29

## Result

V75 binds R126 as the 62nd closed frontier lane and routes the highest
priority action to
`s18_implicit_adaptive_torus_c5_hash_correction_circuit`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R126 analyzes arbitrary nonhomomorphic hashes with represented product-law
corrections. If `L_(a,b)` lists the distinct C2-C3 products in bucket pair
`(a,b)`, then

```text
union_(a,b) L_(a,b) = C5
```

and consequently

```text
sum_(a,b) |L_(a,b)| >= |C5|.
```

A globally deduplicated exact product dictionary still has exactly `|C5|`
keys. Under the inherited iid-distinct-support theorem this is
`B^(15/4+o(1))`, above the `B^(9/4+o(1))` setup cap.

Twenty-four actual R82 controls use field-coordinate hashes at moduli 2, 3,
and 5. Every correction union equals the canonical C5 support, and every
represented correction product has an exact C2+C3 source backpointer.
These controls consume no pairing-image discrete logarithms and receive no
asymptotic credit.

The result covers represented correction targets only. Implicit correction
circuits, adaptive probes, and nonlinear nonlisting certificates remain
open. No Pollard-rho or Shoup improvement and no ECDLP breakthrough is
claimed.

## Verification

- Focused R126 tests: 10 passed.
- Full harness tests: 92 passed.
- Full ECDLP suite: 496 passed.
- R126 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R126: 51 receipts, 900 bindings, 0 mismatches.
- V75 frontier preflights: 62 provided, 62 closed.
- R126 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V75 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R126 producer: `6540a8847cc02206879e217a00b342fafedbe87a31ecac80e24702c3b5a91f64`
- R126 report: `d7af982e0a35bded47611f2e2d50d2a602a26b6e86b8af94f5781963593c5dee`
- R126 frozen interface: `0c5455ab77affc5250ef21184b73035db93d519447deb5911657e6d3ba2fdcbd`
- R126 cost ledger: `b4a63f593f162eb47d9ce0bb27db10f336993e8e7b1679d0690bf9ab9b2e3de5`
- R126 source replay: `af4e1d8ad163de0c9919dee4a29510e307747b95ae553b413dbd4d9aea1cbc23`
- R126 controls: `f1dc25662a9707ab08f1b90f05c280610bb7659a8af2d1565182c8159bcfb132`
- R126 logs/descent: `de11144ba29fa6728ddd30843fc5de128c974cc4be25a67ae964dcffd4210da2`
- R126 tests: `f44864b2d1df41819e21f787ae9d9b8d37360a8736027f7af22d3e89263ab3f5`
- R126 gate: `31de425cd111ddc6d06d33b4e81818c964932be4d18a80fb154164c62e8bab5b`
- R126 parent: `3dd1924eac95dd48cd2ab78a5a6edec5887c960a86c6770a9a34b8850ea3dff7`
- harness: `3c68f4e6e096f8009a8a259d4d65c710d927864ac968ef2f76dddde36eb42e6d`
- harness tests: `9c4975409ff8ee36d5a63f40dfd9c9e7d74afaf7c513d0a6655ab9e563f4048b`
- report: `8715ec0a1cf28cb3434b4d32e4994bd573f83e27a0fdd1a37ea608230aaef81e`
- note: `8bc1b7dab566c33695ad5f356570e0c6d91e8fc5484bb632f4d2cd15a56e2b11`
- inventory: `ab4c7376bd517f65031270f17ec88e65c7c0af66fbdc29848ba2f1e34e50b753`
- replay plan: `08636c10343eb5dbaa66a54d2e835f5ccaf8c0a6811b5144532e2e1fdd8a5f57`
