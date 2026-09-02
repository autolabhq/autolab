# P1436 autoresearch harness V116 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V116 binds R167 as the 103rd closed frontier lane and routes the highest
priority experiment to `s59_slp_elliptic_resultant_mod_u` at priority 296.
The askalphaxiv source post remains captured in the harness methodology:

`https://x.com/askalphaxiv/status/2076737985559822734?s=46`

## R167 result

R167 replaces the arbitrary target list in R166 by one compact
principal-divisor quotient. For retained targets `T_j`, it constructs

```text
div(h) = sum_j [T_j] - [A] - sum_i [R_i]
```

from two degree-`N+1` Riemann-Roch witnesses with one shared zero. The common
zero and equal poles at `O` cancel. A generalized Miller line SLP represents
this target divisor in `B^(5/4+o(1))` state.

For `f_0(Q)=U(x(Q))`, the corrected Weil-reciprocity identity is

```text
product_j f_0(T_j-P)
  = f_0(A-P) product_i f_0(R_i-P)
    product_(Q in S union -S) h(Q+P) / h(P)^(2n).
```

Candidate-zero rows are handled by specialization of the
denominator-cleared identity, not labelled as literal disjoint-support Weil
reciprocity. All auxiliary corrections and `h(P)` values are required to be
units on the selected divisor.

This admits compact target-divisor state and an exact elliptic-resultant
interface. It does not admit a fast resultant restriction modulo `U`.

## Finite controls

Six controls across three curve families and seeds 16001/16002 verify:

```text
selected endpoints checked:          202
candidate-zero specializations:      140
disjoint-support rows:                 62
direct target evaluations:          1,486
raw swapped h evaluations:         17,844
```

Every direct product equals its corrected reciprocity value. Both completed
zero lists sum to `O`; all witness nullspaces are one-dimensional with full
pole order; `h(O)` is evaluated from leading local coefficients. Exactly one
inherited `denominator_exception` target per batch is dropped from the
rational-function control. Finite enumeration receives no attack credit.

## Cost boundary

At the campaign caps:

```text
n = B^(9/4)
N = B^(5/4)
compact target-divisor SLP state = B^(5/4)
direct target table = nN = B^(7/2)
raw swapped table = 2n^2 = B^(9/2)
standard represented elliptic resultant = B^(7/2)
rho proxy = B^(5/2)
```

The raw reciprocity swap is worse than the direct table. A standard represented
resultant remains one exponent above rho. The only surviving constructor is an
output-sensitive elliptic-resultant or tame-symbol remainder modulo `U`, below
`B^(5/2)` total work and preferably `B^(9/4+o(1))`, without `nN`, `n^2`, or
degree-`Theta(nN)` intermediates.

R167 passes 24 of 32 obligations. Lane admission, rho improvement, Shoup
improvement, and breakthrough flags are false.

## V116 routing

The harness schema is `ecdlp.p1436_autoresearch_focus_report.v103`.

- Bound frontier preflights: 103.
- Closed frontier lanes: 103.
- First focus: `s59_slp_elliptic_resultant_mod_u`.
- Natural full-rank cells: 0 of 1.
- Natural verified-log cells: 0 of 1.
- Natural below-rho cells: 0 of 1.
- Promotion allowed: false.

The decisive test freezes `U,V`, the generalized Miller line SLP, anchor,
auxiliary poles, and all correction units. It rejects hidden target tables,
raw swapped tables, represented degree-`nN` resultants, omitted tame-symbol
semantics, and unit-cost resultant or restriction oracles.

## Verification

- R167 focused tests: 16 passed.
- R167 plus harness tests: 149 passed, 6 subtests passed in 8.81 seconds.
- Full ECDLP suite: 1,032 passed, 10 subtests passed in 185.87 seconds.
- R167 clean replay: all six JSON outputs byte-identical.
- V116 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R167: 92 receipts, 1,763 recursive path/hash bindings, zero
  mismatches, zero missing paths or rounds, and zero duplicate rounds.
- Syntax, JSON/YAML parsing, whitespace, parent-hash, and no-promotion
  validation: passed.

## R167 hashes

- Producer: `ba57052082668daf38027b928d88b0eca210510e2f6a6784ce3e0e8b481e1cd3`
- Report: `1d2f009525ef9d54a0f538d3a0a8cefe8451d4d97933b91b31ed1bb5c0b47e3c`
- Frozen interface: `47d3d39017faa9adc1e79abc939c658c51272097dcb8620e72db51ce7543f492`
- Cost ledger: `a84dc45c92d9a17ef9e9b73ddad426a8fe5111e72b6cd974fd5968841ebd10c3`
- Replay: `cf983cfdb3428503634d84a5a69305b71ac70e44338c37acb24e646e5593f6d0`
- Controls: `f78db8e5e5195727087552c8d8a122037eb1f0a0c5d469594ef08a070c093f83`
- Resultant ledger: `95a42cb9d499012b676d96a77ca7c8a066e0cc6ade79e6cf9d0dd9ccfb674247`
- Tests: `8c098e9a820ea7967cc700544aa64cc419750806c67e7c3dca395f271ae505e9`
- Gate: `41c4869bb231dfa2d0de1bb0d280aa34bd95903ed425792b80a84c762b845e3b`
- Parent: `0952a627b3bd5acd95e622533627f54b339605f92e9187be68d72431e7931db7`

## V116 hashes

- Harness: `6cccec3561a05e8c1a33341b34854aaf8befa58dc430c951bc500936c62ca2e8`
- Harness tests: `1856d808204b3d1d03ed0759fbe8edb4b94d1ef441fcf31c39d4be2e69acb253`
- Focus report: `6c9b837be2942d1adb4694fbba0100dc85cd5692eb8f0c9e663350e1f3eb2046`
- Note: `081c6d7cde00748739a3c66ed4180949dc926b72944c255f84113cb1e31119a7`
- Evidence inventory: `643907addd940ff3c4a85de16271365ce8d03ce4d5973e847e79beb8dcc75c79`
- Replay plan: `ede237e9e8fa3ccd249b3c22c81ca9ed90483772316e7de92a67f04665a94b2e`

## Claim boundary

No generic-prime ECDLP algorithm was produced. No Pollard-rho or Shoup lower
bound improvement was demonstrated. The compact divisor witness, exact finite
controls, source recovery interfaces, verifier passes, and harness progress
receive no asymptotic attack credit.

## Next action

Test whether generalized Miller line factors induce a reusable low-displacement
operator in `F_p[X]/(U)` under transposed modular composition or half-GCD. Any
candidate must emit the denominator-cleared resultant remainder below
`B^(5/2)` with all correction units and signed candidate verification charged.
