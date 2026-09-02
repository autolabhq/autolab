# P1436 autoresearch focus harness V77 result

Date: 2026-07-29

## Result

V77 binds R128 as the 64th closed frontier lane and routes the highest
priority action to `s20_low_slp_piecewise_torus_c5_selector`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R128 studies a rational selector

```text
R(Y) = A(Y) / D(Y)
```

that returns a C2 factor for every positive C5 target. For a fixed C2
output `x`, the polynomial `A-xD` has at most `d` roots when the selector
degree is `d`. Since one C2 value can cover at most one translated C3
fiber, a complete selector requires

```text
d >= ceil(|C5| / |C2|) = B^(9/4)
```

under the inherited iid-distinct-support exponents. Dense coefficients fit
the `B^(9/4+o(1))` setup cap exactly, but dense arbitrary-target evaluation
costs `B^(9/4)` and misses the polylogarithmic query cap. An explicit
target-to-C2 table costs `B^(15/4)` state.

All eight actual canonical selector controls have domain exactly equal to
the distinct C5 support. Their unique interpolation polynomials have degree
`|C5|-1`, evaluate exactly, return replayable C2+C3 sources, and include
empty-target controls. No pairing-image discrete logarithm is consumed and
finite controls receive no asymptotic credit.

The degree theorem is not an arithmetic-circuit lower bound: repeated
squaring can produce high degree with logarithmic multiplication count.
High-degree low-SLP, compact piecewise, and adaptive selectors therefore
remain open. No relation-rank construction, factor logs, identical target
descent, Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough
is claimed.

## Verification

- Focused R128 tests: 10 passed.
- Full harness tests: 94 passed.
- Full ECDLP suite: 518 passed.
- R128 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R128: 53 receipts, 943 bindings, 0 mismatches.
- V77 frontier preflights: 64 provided, 64 closed.
- R128 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V77 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R128 producer: `50ddb43a677d114b534ed55eb877f589ceadf481d29bf5c819323d2add6848c3`
- R128 report: `400968376068f669ee972aaed88303ce24ee12529fa69bfdc9e7a3890ccdef8f`
- R128 frozen interface: `6cd1af34f4e2c6c5f0539b08653927b51930fb1b22779f5d6f560e50e589e8bf`
- R128 cost ledger: `e375179be8e4bcc39205e17a6ea57d20bb83d27da405f82a7f6a765d1536eddf`
- R128 source replay: `96a1e44774730df55d4ae28cec543ad9eefba4d601a65e6fd7a39f95423a95c9`
- R128 controls: `ee1afcd3fe2aa702305da00bd447113b83c596043cb019d8fff52866f7261d02`
- R128 logs/descent: `dd8702872a282ab88e93d70f8f8185db52e6623ed26be7bee9f442d609479bf3`
- R128 tests: `a78bd8ed02849a150ea7a30ffc24ccd345d12927c9178e246a2c65b30335d7d7`
- R128 gate: `6033e5f5bf8fa0b9934efde304d4036a01fddf8deede30255a936b406722977d`
- R128 parent: `1c12e3a51d749946801381a9519bc781288b4b8dde85e9d5e4a3c436224c9818`
- harness: `0b895833b4f8335aa958a41e08e23581b7933af5d3d7e83683d9fe6b59921c9a`
- harness tests: `1052e42383d6c5df180d5074ffe5766a068d5464ea2c1b45edfb3b150c7b9292`
- report: `86084defbe7896233b7c3ab340cc461cba4a2d94fd296fb13f57bafa405292a0`
- note: `53f8820e3c6459e87e22ef2a2ddcaaac44795ed604f2a57bd5d5dbb8b89e2d26`
- inventory: `47432e1789c7fb3ba6779177c973c73324068b7c5d259a4250ffe12e34571e53`
- replay plan: `40f8b62e063da41f490ded47dd9f15bbf01a43200147db46f35475b50a0b8ceb`
