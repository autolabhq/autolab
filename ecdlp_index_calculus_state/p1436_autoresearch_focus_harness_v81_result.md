# P1436 autoresearch focus harness V81 result

Date: 2026-07-29

## Result

V81 binds R132 as the 68th closed frontier lane and routes the highest
priority action to
`s24_asymmetric_frobenius_torus_c5_selector_predicate`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R132 isolates a field-specific obstruction that does not require scalar
labels. In every actual pairing family,

```text
p = -1 mod q,
z^p = z^(-1)
```

for order-`q` targets. If `f` has coefficients in `F_p`, then

```text
f(z^(-1)) = f(z^p) = f(z)^p.
```

Consequently, polynomial zero outcomes are invariant under inversion. The
same holds for numerator-zero and denominator-zero outcomes of univariate
base-field rational predicates. A deterministic DAG using only those
Boolean outcomes must follow the same path on `z` and `z^(-1)`.

All eight actual pairing decks provide exact counter-witnesses to such a
locator: every positive C5 target has an inverse outside the complete C5
support. Every active color target also has an empty inverse. The exact
full-support and color-support annihilators all require at least one
extension-field coefficient, and five sample base-field polynomials per
control replay the Frobenius identity and inversion-invariant zero outcomes.
No candidate discrete logarithm is used.

For comparison only, a uniformly random `M`-subset of an odd prime-order
group has expected inverse overlap `M^2/q`. At the inherited scales this is
`B^(5/2+o(1))`, a `B^(-5/4+o(1))` fraction of the
`B^(15/4+o(1))` support. This comparator is model-bound and receives no
candidate or asymptotic credit.

The result closes only univariate base-field zero/definedness DAGs on the
exact inversion-disjoint controls. Asymmetric `Fp2` coefficients,
Frobenius-aware coordinate predicates in `(z,z^p)`, nonzero-value tests,
and general circuits remain open. No source index, relation-rank
construction, factor logs, identical target descent, Pollard-rho
improvement, Shoup improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R132 tests: 10 passed.
- Full harness tests: 98 passed.
- Full ECDLP suite: 562 passed.
- R132 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R132: 57 receipts, 1023 bindings, 0 mismatches.
- V81 frontier preflights: 68 provided, 68 closed.
- R132 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V81 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R132 producer: `9235668bed218df40c60ea774673b03bc2d55236cd1c04ab3ef813ad61e0baad`
- R132 report: `54e0396f8fcbed3fb41893af1c2d3612f81f26b5ff44dcbc0784299a7ff96f17`
- R132 frozen interface: `efc22a83ab20a27b490204acdb67a8bda351d774caf2c186cda9f04cdd0ae56f`
- R132 cost ledger: `437e3a1165d17d6f85caa4e246ea9769012df2915113675249bfcc68502fbe4f`
- R132 source replay: `0e7f17f5c0c04cbdcf1a430f36114343b2b786cf3cb85a9ca586afeb549471e9`
- R132 controls: `f0ee93f6ddfef5395abc88a8f03b31a325504a14bf04b7d7a246d9efec3f53ab`
- R132 logs/descent: `f85f4135fe44d2aebbe61191dff04e2e40aff4a07d45118c752758c8425dfed3`
- R132 tests: `296daa244226d7d3a41da492f97cf1ea291280af6bb9e0374e0b366df0e4d099`
- R132 gate: `dd20f3188596712929c9715d1dbf27782a41a5be8900f727f948def2a7947332`
- R132 parent: `958b63522081adcba52bdfbdbaa982356d00d67098ea4c298e53227af1b6e447`
- harness: `3c65a898c292b1438ee920c6c808332c6f7d5fec9d489b085ae5a280aea97194`
- harness tests: `2a7e9c8d45c6131987fb9cfa812c59f4d9289a71ce1ac97375f2032c6f533b28`
- report: `e43b9a946d231f77c962cd4678a17d9b6e0b5fad3cc3a7556b2369059a36f8cb`
- note: `2a06f0107414088b73ca71d2d16739785094c9c7eda2c83808f917e351c54973`
- inventory: `1bcace2fc33205c5b4db23393fb74ad143796c43f131a32cc1ec10f97abb6055`
- replay plan: `5aae33aa4da64ed2a8641dad483096d834ab6c379b15920395c6d7f0622874c3`
