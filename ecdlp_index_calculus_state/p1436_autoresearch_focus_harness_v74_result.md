# P1436 autoresearch focus harness V74 result

Date: 2026-07-29

## Result

V74 binds R125 as the 61st closed frontier lane and routes the highest
priority action to
`s17_nonhomomorphic_adaptive_torus_c5_fingerprint`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R125 proves a prime-order obstruction for pure product-preserving
fingerprints. Every homomorphism from `mu_q` has trivial or singleton
kernel. Equivalently, the directly computable power map

```text
z -> z^k
```

is constant for `k=0 mod q` and a permutation for every other `k`.
A constant map has false positives on arbitrary empty targets. An exact
nontrivial map retains the complete image

```text
q = B^(5+o(1)),
```

above the `B^(9/4+o(1))` setup cap for a full table. Any finite tuple of
homomorphisms has the same dichotomy: it is trivial if every component is
trivial and injective otherwise.

Prime controls at orders 5, 7, 11, and 13 verify the dichotomy and exactness
behavior. Composite controls at orders 6, 8, and 12 expose intermediate
quotient images, confirming that the obstruction comes from prime order.
Finite controls receive no asymptotic credit.

The theorem does not cover nonhomomorphic or adaptive fingerprints.
Such a route must expose and charge correction data because hashed products
are not determined by the hashes alone. No Pollard-rho or Shoup improvement
and no ECDLP breakthrough is claimed.

## Verification

- Focused R125 tests: 10 passed.
- Full harness tests: 91 passed.
- Full ECDLP suite: 485 passed.
- R125 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R125: 50 receipts, 879 bindings, 0 mismatches.
- V74 frontier preflights: 61 provided, 61 closed.
- R125 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V74 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R125 producer: `28a74fc7699f34569a30c3b378ef80985b8ff09bd01aed497df41a98c80160e4`
- R125 report: `c6f2bd31d83d2b959bd07477d0697f2e91180a5f65e02236a3c32cffd6c43a9b`
- R125 frozen interface: `be8bc523f1970c07bcc1e2285070051061c1e6dd189da717d879b539012e88ff`
- R125 cost ledger: `095ae32e7b927c2639de8cd02ff6818229a5026065d49407a0f42e3b0fea828e`
- R125 source replay: `63c51d0728f52b0618c6afdf73a3ffda070282d7fb79663b79fceb85effcc8c0`
- R125 controls: `28da34281db3b0ea782b9b5bb8b89c39eaabb902f22cec499113d4b3cc1ba2b9`
- R125 logs/descent: `750c96ae91a642278df899d845db00d26e70858d74af22dc7b51260584303c1a`
- R125 tests: `fbd5c8156f9c233a33834331e348b7809c55e69813bc99158d15b70a3e12c997`
- R125 gate: `b4d19d79a68076f8850afd7c079c7f25bb108ba9aee853330aecb508d9d11155`
- R125 parent: `29d29a6e507839ee60964fd49cae14ed3ee9dab348c961a99586acd2ea48298d`
- harness: `6d1c61e1b24aa17f378576038c4e5661c1ee3c4a6c06d5c709aabde310493a75`
- harness tests: `8e3bd54091f87f57a3964f8d1c5af68e6f32181cfc0a94960666cece49dafbf9`
- report: `c402b837caa1d1c1292340dac6ba5e6bda6b090c4b8d3aafd9e2906dec11fc45`
- note: `9e3cce2a2d1f3b127fd96c92457d51a890f85187a698ee7d4bba28d4b5f03ce3`
- inventory: `679ffa3cecb8faf54ac189de2df0b4ae76998b4d4f160c9c51b00472743d8c14`
- replay plan: `ac29eff4b8f7bea770fcea132915fd4ffa7c6a8c2daf1e97ca953c4bebdc51ed`
