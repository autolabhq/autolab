# P1436 Autoresearch Focus Harness V37 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R87: addition-pushforward jet screen

R87 proves the exact Pontryagin norm identity

```text
P_(A*C)(T) = product_(a in A) P_C(T-a).
```

Its derivative also depends on `P_C` and `P_C'` at every shifted target
`T-a`. A first norm jet at only the final target is therefore not the
standard compositional state.

The exact marked counterexample is stronger. Over `F_101`, two split cubics
with five valid marker-deformation channels have the same full local state

```text
[N, dN/dt, dN/ds_1, ..., dN/ds_5]
  = [73,70,1,2,3,4,5],
```

but composition with the same two-point left divisor gives scalar first jets
`[37,51]` and `[77,73]`.

The standard exact replacement is the translated remainder

```text
P_C(T-X) mod P_A(X).
```

R87 proves that, in characteristic above the degree, translations of a monic
polynomial span the full polynomial space. Modulo degree-`m` `P_A`, the
translated remainder orbit therefore has dimension `m`. Four exact controls
attain ranks `3,5,7,9`.

For the five-`A` versus five-`C` split, `deg(P_A)=B^2` and
`deg(P_C)=B^3`. The explicit `C` polynomial misses setup, while a fresh
translated remainder or value vector has `B^2` words and misses the online
cap.

A positive 32-by-243 synthetic endpoint control recovers one unique
ten-choice source by a degree-one translated gcd. It materializes both
polynomials and source dictionaries, so it receives no candidate credit.

R87 preserves the fixed factor-index marker subfamily, black-box scalar
resultants, implicit half-gcd/source localization, and unrestricted
arithmetic circuits as scoped exceptions.

```text
report  f10ba663867815c9ee0b1234f4d9dee698d450a3a7171336d36f3e328ea2333a
gate    16d635add67bc64d63d5870663f68ce37e35a21fa4feb436c428b7afbe6ed565
parent  dc7f257eb16b124577bd9db898dd6341498f2d567ff580f4cf72f6da5dca36f2
```

## Harness routing

V37 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v37.json`

SHA-256:

`0e8a2230a7a75d70468f315d2523159bd0c8eb107f6530f1ab2d0747aef88606`

Schema: `ecdlp.p1436_autoresearch_focus_report.v29`.

Twenty-three hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_black_box_translated_resultant_gcd_localizer
```

The next experiment must consume compact five-slot `D_A,D_C` and compute
and localize a translated common root without materializing the `B^3`
five-`C` characteristic polynomial, the `B^2` translated remainder, or a
`B^2` evaluation vector. It must fit the direct caps, return a jointly
coupled source, and reject all multiplicity and projective exceptional
strata.

V37 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R87 producer         d822035ed8e13a8ba0c987d5eab5a2a06bff251ebc32479ea358626ef038cf6b
R87 tests            41f86fe36a9c0ae9f8f4331b4a0dde2f697d81ff967775068c629414a910e21c
harness              8212ab88354344278700c090109caf31fafc995df40cc4f752206fa0b85b73b8
harness tests        bd53541013af97de97240902d25f0f6e66ff0a6bc243a2764a07d45f843f92f6
V37 focus note       ae74b5dd5529dbf58c1a019fab37aade9b4fbf16f5dba903b79fe9d161f8626e
V37 FFE inventory    288c8a1ce1a771a9d4cc7b80b8de36ec61ae7c6cc58950066aae2b8c6d09abb1
V37 FFE replay       7af647292e765b42246a22c5f2a77555f6c12f83981c530efc5ba20349f66052
```

## Verification

- R87 targeted tests: 5 passed.
- Harness tests: 53 passed.
- Full ECDLP task suite: 163 tests passed.
- Thirteen R76-R87 and harness Python modules compiled.
- Twelve R76-R87 parent YAML receipts parsed.
- All 143 declared input/artifact hashes matched.
- Every parent receipt has `breakthrough=false`.
- A clean R87 rerun reproduced all six generated JSON artifacts
  byte-for-byte.
- V37 has 23 closed lanes, `promotion_allowed=false`, and the expected
  black-box translated-resultant/gcd action.
