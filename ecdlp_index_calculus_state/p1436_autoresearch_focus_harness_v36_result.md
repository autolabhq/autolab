# P1436 Autoresearch Focus Harness V36 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R86: coloured sparse-moment screen

R86 instantiates a five-colour first norm jet on the R82 compact factor base

```text
F_(a,c) = A_a + C_c.
```

Flattened factor-index residues modulo five give a target-independent
colouring represented by at most five `A`-residue by `C`-residue rectangles
per colour.

Across eight frozen instances, a supplied first norm jet at a simple
coloured fiber recovers all five factor indices, all five `A,C` atom-index
pairs, and the target group element exactly. Empty, multiple, and synthetic
nonreduced controls take the expected first-jet branches.

The control is verifier-only. It enumerates the coloured source quotient and
uses batch-BSGS endpoint labels, so it receives no candidate relation credit.
The standard multigraded quotient and norm have dimension or degree `B^5`.
Using the Cartesian-sum structure exposes a sharper `B^2` five-`A` side and
`B^3` five-`C` side. The `B^3` side exceeds both the `B^(9/4)` setup cap and
the `B^(5/4)` fresh-work cap.

R86 closes supplied-jet decoding and the standard full-quotient,
tensor/value-vector, norm/resultant/projector, and explicit five-slot
atom-product constructors only. A compositional, jet-preserving
addition-pushforward intertwiner remains open.

```text
report  b93c1e581a7953cf1a9a86d3ff4220f5c2a92b2ea3e0ed7076c57d8896ee1173
gate    21d52b3d5f04fa408d0d9a4ef229f9169f05f4a31aa858e21ff252ef497f9b44
parent  4a5de72859057c1773bcd6e2f85faeb3871d4f3f9361f762a43912c27c86a1f3
```

## Harness routing

V36 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v36.json`

SHA-256:

`ab65c24e864da7933767e9c763465192fc670335835976dca33f919880f2b352`

Schema: `ecdlp.p1436_autoresearch_focus_report.v28`.

Twenty-two hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_jet_preserving_addition_pushforward_intertwiner
```

The next experiment must derive the coloured first norm jet from public
compact `D_A,D_C`, without DLP labels, supplied moments, `B^3` five-`C`
support, or the `B^5` coloured quotient. It must fit the direct setup and
fresh-work caps and return the jointly coupled factor and atom source through
reduced, nonreduced, signed, infinity, and exceptional fibers.

V36 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R86 producer         89c540fed698ea70fab5f57166f1147384201903c8f149ff90fed40bf6ee604f
R86 tests            c6fee01ef821f1180af7a3f6692db9ffb88405cc4f301e8ce657b24231887f24
harness              db06e08befbe1b1f69ce3c4b8f550fafa0fc5d8597068a54fda02d78b9d79826
harness tests        88b7860f2c891c86c0941cab70911a3fe2368884c349cc52efc56dc2e1bbfb26
V36 focus note       336fe70bef8d896a92c953f51847ae7291000e5cb77b3a63aa58520d62f41440
V36 FFE inventory    f0ad9be6c87b0d8dd3ca08a36cb4376f87a6ac662a354a370c4a9251c4c4c1b3
V36 FFE replay       cc4d77b22f0b5e6be0cee13987745b11570dc0feeeb7d15a20f5ce28f9c6e6d6
```

## Verification

- R86 targeted tests: 5 passed.
- Harness tests: 52 passed.
- Full ECDLP task suite: 157 tests passed.
- Twelve R76-R86 and harness Python modules compiled.
- Eleven R76-R86 parent YAML receipts parsed.
- All 129 declared input/artifact hashes matched.
- Every parent receipt has `breakthrough=false`.
- A clean R86 rerun reproduced all six generated JSON artifacts
  byte-for-byte.
- V36 has 22 closed lanes, `promotion_allowed=false`, and the expected
  jet-preserving addition-pushforward action.
