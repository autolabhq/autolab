# P1436 autoresearch harness V117 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V117 binds R168 as the 104th closed frontier lane and routes the highest
priority experiment to
`s60_denominator_aware_elliptic_cauchy_trace_mod_u` at priority 298. The
askalphaxiv source post remains integrated in the harness methodology:

`https://x.com/askalphaxiv/status/2076737985559822734?s=46`

## R168 result

R168 applies the invariant derivation

```text
D = 2y d/dx + (3x^2+a) d/dy
```

to the exact R167 identity. Since `(Dg/g) dx/(2y)=dg/g`, every zero of
multiplicity `m` becomes a simple logarithmic pole with residue `m`.

For the regularized R166 candidate product `G`, all candidate multiplicities
are between one and `N`, while `N<p`. No candidate can disappear through
characteristic-`p` residue cancellation. The candidate factor is therefore
the denominator support of `DG/G` on the selected signed divisor.

Differentiating R167 gives

```text
Dlog G(P)
  = Dlog correction(P)
    + sum_(Q in S union -S) Dlog h(Q+P)
    - 2n Dlog h(P).
```

This replaces the multiplicative elliptic resultant with an additive elliptic
Cauchy trace of the compact `Dh/h` witness. A valid constructor must preserve
the denominator, Fitting, or subresultant factor at candidate nonunits. A
generic-point value trace formed by inverting `h` receives no locator credit.

## Finite controls

Six controls verify:

```text
candidate poles:                       140
translated zero occurrences:           241
maximum candidate multiplicity:          4
true-orientation occurrences:           241
opposite-orientation occurrences:         0
public P=T pole corrections:              6
```

Every direct derivative `-2yU'(x)` and every corresponding translated R167
numerator derivative is nonzero. Direct and swapped multiplicities agree at
every selected endpoint, every residue is nonzero, and the 140 pole roots are
exactly the R167 candidate roots.

Each inherited batch contains one public `P=T` rational pole of order `2n`.
The signed `U,V` test detects all six; semantic regularization replaces each
factor by one and its logarithmic derivative by zero.

## Cost boundary

```text
compact h and Dh/h state:              B^(5/4)
public U,V target-equality prefilter:   B^(9/4)
direct n-by-N logarithmic table:        B^(7/2)
raw swapped 2n-by-n trace table:        B^(9/2)
standard tensor quotient state:        B^(9/2)
expected candidate denominator degree: B^(3/4)
signed candidate verification:         B^2
rho proxy:                              B^(5/2)
```

The compact witness and equality prefilter are below rho. The explicit direct,
swapped, and tensor routes are not. The surviving primitive is a
denominator-aware transposed elliptic Cauchy trace modulo `U`, strictly below
`B^(5/2)` total work and preferably `B^(9/4+o(1))`.

R168 passes 24 of 32 obligations. Lane admission, rho improvement, Shoup
improvement, and breakthrough flags are false.

## V117 routing

The harness schema is `ecdlp.p1436_autoresearch_focus_report.v104`.

- Bound frontier preflights: 104.
- Closed frontier lanes: 104.
- First focus: `s60_denominator_aware_elliptic_cauchy_trace_mod_u`.
- Natural full-rank cells: 0 of 1.
- Natural verified-log cells: 0 of 1.
- Natural below-rho cells: 0 of 1.
- Promotion allowed: false.

The decisive test freezes `U,V,h,Dh/h`, all logarithmic corrections, and the
public equality removals. It rejects `nN` or `n^2` pair tables, tensor-quotient
materialization, candidate inversion, generic-point-only values, omitted
multiplicity residues, and unit-cost trace or resultant oracles.

## Verification

- R168 focused tests: 16 passed.
- R168 plus harness tests: 150 passed, 6 subtests passed in 13.14 seconds.
- Full ECDLP suite: 1,049 passed, 10 subtests passed in 195.74 seconds.
- R168 clean replay: all six JSON outputs byte-identical.
- V117 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R168: 93 receipts, 1,787 recursive path/hash bindings, zero
  mismatches, zero missing paths or rounds, and zero duplicate rounds.
- Syntax, JSON/YAML parsing, whitespace, parent-hash, and no-promotion
  validation: passed.

## R168 hashes

- Producer: `2bf35a1e4624d0832b710d36d25cdb50d5ff855793ed3bd3a80a42a3809b7400`
- Report: `82fbb81b136af357d4a78e5ef7bf29e7ce97002b8a95a7c93b42cc0bda7ed2c3`
- Frozen interface: `150dc58ef81575191a86497a43ceef092603c8661339acaa587f91a3f9d43645`
- Cost ledger: `48bf265d3c6b19f72b0a242e63dec7aae75ac2026f641468d0e235c30f2d1412`
- Replay: `0b8d9231a701f5e85001245ffdd91e13a9bd79fc34162fa8b34109c6f0a76bae`
- Controls: `e348e8a4b17d41775cbb6fed917f62f0c24b2fecec37abc5c59c9f9c85d490ab`
- Trace ledger: `12731e6627c812951b35f53fccfa1610a763f3bf14c745544ff255affbf8660d`
- Tests: `05d7ec8b9add06c4ea452ed1f61e8c8e83113593c8e5c24af354ece36af293df`
- Gate: `7063a5f6ba686e38682ab6e29f386ab56e7fb56097e288949cce73e130b2fcae`
- Parent: `390b4ec16a24b79742901f4d8a9894a23bf387edc80998010a0fc0128dc4d68c`

## V117 hashes

- Harness: `56c42826d5fd4318094560aed17881f1594b63ae9bbaf9574168002ad3fa610e`
- Harness tests: `dbff5483bcbbc333ded529d1b44a03c2c0ec810a0eaa0f148b1eed783a8e2ebc`
- Focus report: `aa7497946d1482c884f16c1eaadd95531889581de783a38d11ef3059eb135314`
- Note: `f4e7f79e226e732e334344581bcb7794b9cae6b5c057c87489213d6d6b44e21c`
- Evidence inventory: `263f4f35112ec5ebf1b183cb7743b8e9ecc8c4550f36a1959b5f70b1cfb7cfc8`
- Replay plan: `a7f3ebb802e07bd4d2ed50251a5301920fc4c9d9e0ac92672a369c23b72933df`

## Claim boundary

No generic-prime ECDLP algorithm was produced. No Pollard-rho or Shoup lower
bound improvement was demonstrated. Compact divisor state, logarithmic
linearization, finite residues, verifier passes, and harness progress receive
no asymptotic attack credit.

## Next action

Model the corrected trace as a structured subresultant/Fitting problem over
`F_p[X]/(U)`. Measure whether its Sylvester-like or Cauchy-like operator has
bounded displacement rank under the elliptic addition law, while preserving
candidate nonunits and forbidding explicit tensor state.
