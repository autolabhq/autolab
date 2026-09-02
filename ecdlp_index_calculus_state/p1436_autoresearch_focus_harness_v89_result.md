# P1436 autoresearch focus harness V89 result

Date: 2026-07-29

## Result

V89 binds R140 as the 76th closed frontier lane and routes the highest
priority action to
`s32_subcap_mobius_claw_or_nonzero_torus_c5_selector`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R140 shows that the R139 order-two full-spark theorem is sharp at three
columns. For normalized progression rows `{0,1,3,4}` and Fourier modes
`{0,1,a,b}`, write `y=z^a` and `w=z^b`. Eliminating the first two
columns factors the four-minor determinant numerator as

```text
(y-w)(w-z)(w-1)(y-1)(y-z)
  * (wy+wz+w+yz+y+z).
```

Away from repeated and trivial modes, singularity is therefore
equivalent to the fractional-linear claw

```text
w = T_z(y) = -(yz+y+z)/(y+z+1).
```

The fractional-linear matrix for `T_z` squares to
`(z^2+z+1)I`, so the map is an involution for prime `q>3`.
Order-two Frobenius also gives `T_z(y)^p=T_z(y)^-1` whenever `y` and
`z` have norm one. Thus the claw remains in the full norm-one group.

All 12 available actual six-point color progressions admit bounded
known-mode claws. Every corresponding four-by-four minor has rank
three, every kernel coefficient is nonzero, and every kernel polynomial
vanishes on exactly progression rows `{0,1,3,4}` among the six checked
sources. This deterministically refutes a universal four-column
full-spark extension on the structured controls.

The claw does not yet supply a selector. Computing `w=T_z(z^a)` does not
reveal the integer mode `b` needed to evaluate `X^b` on arbitrary
targets. Recovering it uses a field-subgroup DLP, while the standard
known-mode collision or generic DLP baseline costs
`q^(1/2)=B^(5/2)`, above the `B^(9/4)` setup cap. That is a charged
generic baseline, not a lower bound against every structured method.
The finite scans and constant four-of-six root sets receive no
asymptotic credit.

R140 is a self-contained campaign result; global literature novelty is
unverified. A sub-`q^(9/20)` structured mode claw, direct hidden-mode
evaluation without DLP, asymptotically dense four-mode atom roots, and
nonzero-value Frobenius-coordinate selectors remain open. No source
index, relation-rank construction, factor logs, identical target
descent, Pollard-rho improvement, Shoup improvement, or ECDLP
breakthrough is claimed.

## Verification

- Focused R140 tests: 10 passed.
- Full harness tests: 106 passed.
- Full ECDLP suite: 650 passed.
- R140 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R140: 65 receipts, 1,180 bindings, 0 mismatches.
- V89 frontier preflights: 76 provided, 76 closed.
- R140 breakthrough, Shoup-improvement, and rho-improvement claims:
  false.
- V89 source breakthrough, promotion, below-rho, and Shoup-pressure
  gates: false.

## Hashes

- R140 producer: `a849f79dea39041de547b473c0d4fc28e6221bbb6e004738028c7f4515422373`
- R140 report: `3971132c94034f6c7f03bb7b77e83212fe805d9539b96c77a10e14176dca527c`
- R140 frozen interface: `7930da7bdd6f51254ab2a23aacf12113c0c85fd2857aec8919caeb30d69f1a1f`
- R140 cost ledger: `2cd3c7708e4f83af6f688e10ce27e06198e6aaade546b0e0ef15423777137af8`
- R140 source replay: `0661fab8f891080107a03e95981fea0170e54cbe8f53487db40fcfa9f0a9a3e2`
- R140 controls: `dbe9feac7384b1d6048b11d9a7a86e2afdb91adc339481132183e85f57ea1a75`
- R140 logs/descent: `b7d7c9f1ecf51a349dea0c610c4a8a3fee6d000708285685a4d27d9b12eeecda`
- R140 tests: `8e5daa679de1739e71c9670c5b775c2179d833748730fe897ba7eb8b8e36a585`
- R140 gate: `33c8b530802974a36b06b0032c17794f7f77bf454d41dd4a2e2415d671ccfe66`
- R140 parent: `03ae39a6fde97e95581fefccf64a86cc6adc29ff2126ffc2e72bfd7f6d136ec5`
- harness: `5b79eab11a2de1ee415acacd9baa5c01b142469d793ab6b3b5695bf9cafc844d`
- harness tests: `8f30a23badade2e21e42bf46ccb6c37638c827f7a9fb9f22c91e788ea04708c2`
- report: `cdb06cf11b04df7882614be05d99b9fb64118a13a5f99c9c4c734c74e317c4ac`
- note: `c32c947b38a8d36c04867d80623442643d70e3b23b0ccc00291e9c1b18dedae1`
- inventory: `21c568811b2d738ee9d7f491221d7c30d59ba77de83f019be6f449ba8ba07995`
- replay plan: `b1eed3461b57133f0e5ebfa442af9a0b4a8f7927d0991a6d7dcab01dfa126ab5`
