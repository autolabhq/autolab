# P1436 Autoresearch Focus Harness V51 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R101: actual divisor-image local oracles

R101 replaces R100's universal-set question with the actual R84 compact atom
decks. It constructs exact scalar-blind local count/source oracles for both
structured side images.

For `2A+3C`, a weighted unordered-`3C` endpoint dictionary has
`B^(9/5)` state and an unordered-`2A` query scan costs `B^(4/5)`. For
`3A+2C`, a weighted unordered-`2C` dictionary has `B^(6/5)` state and an
unordered-`3A` query scan costs `B^(6/5)`. Both fit the direct
`B^(9/4)` setup/state and `B^(5/4)` online caps.

All attained endpoints in 16 actual side instances match direct multiset
counts and replay one source. Blind, identity, repeated-atom, and synthetic
collision controls are exact.

This does not solve the fresh-target intersection

```text
find ell in (2A+3C) such that T-ell is in (3A+2C).
```

Enumerating either full side costs `B^(13/5)` or `B^(12/5)`. R101 provides
neither a subcap two-sided join nor an exact jointly coupled `5A+5C` source.

```text
R101 producer  854df1da576b07cc3dd63b1249a42a72d56708ef8eb43c4ef37478f2558e747d
R101 report    050d1093fc34611f502a3f7b3ebf6fc632b3aeb9d03718f922e99baa2fe985c9
R101 gate      3e12e4ab70edba64076d27b68d39ab34307ca6d9ff08252432759673b74a9ac5
R101 parent    14beac87b86d3664fd439c82663e444aa379035d292c60873523f0d6c9c67d6d
R101 tests     a38fab611425e5b15bf65021cc7cdfeb2fb964b5ec738ce42301c1b507ad4b6a
```

## Harness routing

V51 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v51.json`

SHA-256:

`c3f2de6486fc483f5283afcd2778a722a1ec61ccbe0b46b1c6787a7fd1aadd0d`

Schema: `ecdlp.p1436_autoresearch_focus_report.v43`.

Thirty-seven provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_two_sided_implicit_join
```

The next experiment must combine the two admitted local oracles under target
translation without enumerating either side, while returning exact integer
multiplicity and one jointly coupled source.

V51 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              c2bc9489ac86dfd7090f399b58014055b3c1f7808e9a21a84c7d1aeb15acbb0f
harness tests        797f5dc8552b5420d634d02e3262ce46ed89bb305a79b148de952ecc12c17409
V51 focus note       0d4d0b05b375f1dea1acb46b3e0d41d30b8af8ae278a1fdc84625d11d90a9386
V51 FFE inventory    e7e4d691e6de4346fa15a5d394cfa87fadd231888d9a110b3605c79fd4793d3f
V51 FFE replay       a6233e5e1cfb0f70e4757965385c982e77d62beec0c47deb79f36b01de9b6267
```

## Verification

- R101 targeted tests: 7 passed.
- Harness tests: 67 passed.
- Full ECDLP task suite: 272 tests passed.
- Twenty-seven R76-R101 and harness Python modules compiled.
- Twenty-six R76-R101 parent YAML receipts parsed.
- All 386 declared input/artifact hashes matched.
- No R101 parent or nested JSON artifact sets `breakthrough=true` or
  `shoup_bound_improvement=true`.
- A clean R101 rerun reproduced all six generated JSON artifacts
  byte-for-byte.
- V51 has 37 closed lanes, `promotion_allowed=false`, and the expected
  two-sided implicit-join action.
