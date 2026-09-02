# P1436 autoresearch focus harness V90 result

Date: 2026-07-29

## Result

V90 binds R141 as the 77th closed frontier lane and routes the highest
priority action to `s33_nonlinear_sextic_mobius_source_router`. The
report remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R141 changes the status of the R140 Mobius claw without promoting it to
a source index. In Cayley coordinates

```text
X(t) = (1 + u*t)/(1 - u*t),  u^2 = d,
```

the claw `w=T_z(x)` is equivalent to

```text
d*(t_x*t_z + t_x*t_w + t_z*t_w) = 3.
```

For a full norm-one group of order `p+1=6q`, the nonzero-value label

```text
chi_z(x) = T_z(x)^q
```

lies in the six roots of unity and is evaluable with one Mobius
transform plus logarithmic exponentiation. It does not require a
discrete logarithm.

For fixed `x != x^-1`, equality
`chi_z(x)=chi_z(x^-1)` is the trivial-coset condition for a
fixed-degree rational function of `z` whose divisor has a simple
component, so it is not a sixth power. The classical Weil
square-root character-sum bound then gives equality on
`q/6+O(sqrt(q))` parameters. A union bound supplies an
`O(log |S|)` fixed parameter family separating every element of an
inversion-disjoint support `S` from its inverse.

This is an asymptotic inverse-separation result, not a C2 source router.
The corresponding translated-linear approach is also too large:
fixed-conductor character-sum cancellation bounds each Fourier
coefficient by `O(sqrt(q))`, while Parseval forces `Omega(q)` nonzero
coefficients. Materializing that translation orbit therefore costs
`Omega(q)=B^(5+o(1))` state, above the `B^(9/4+o(1))` setup cap.
This boundary applies only to translation-invariant linear
convolution/sketches of the six labels. It is not a nonlinear circuit,
RAM, or cell-probe lower bound.

All eight actual positive controls are injective and inversion-disjoint.
Greedy signatures from the frozen parameter deck separate every
positive from its inverse with at most four parameters. Every actual
C2-by-C3 character matrix has full row rank, and every multiplicative
defect realizes all six cosets. Eight synthetic prime-order controls
have Fourier support at least `q-1`. These finite controls receive no
asymptotic algorithmic credit.

The remaining question is whether a nonlinear, nontranslation data
structure can compose the sextic labels into an exact C2 source router
without storing the linear orbit. No source index, relation-rank
construction, factor logs, identical target descent, Pollard-rho
improvement, Shoup improvement, or ECDLP breakthrough is claimed.
Global literature novelty remains unverified.

## Verification

- Focused R141 tests: 10 passed.
- Full harness tests: 107 passed.
- Full ECDLP suite: 661 passed.
- R141 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R141: 66 receipts, 1,199 bindings, 0 mismatches.
- V90 frontier preflights: 77 provided, 77 closed.
- R141 obligations: 16 of 26 passed; lane admission false.
- R141 breakthrough and Shoup-improvement claims: false.
- V90 source breakthrough, promotion, below-rho, and Shoup-pressure
  gates: false.

## Hashes

- R141 producer: `5cdbf62193929e3fc8ed3634763f1d9eb2ce7d2561b752e3ac81006d423e2f49`
- R141 report: `9b3705ae5515c8f700875e31766a7e9dcb1c5150ac880ae3c0ef197e07388173`
- R141 frozen interface: `07dedf03d33d83e729e12a52f5302bbb6cc03c508e54d8ec82a21bf0e1f59a0c`
- R141 cost ledger: `62856c871858208554c515949994390af29071079a1200d9903ca7c6cac9274c`
- R141 source replay: `c783712f1b583f05f45a7b8864ff8cc34214225db92e4331958eac845d3c5b36`
- R141 controls: `8a8689012a8e41f3815e203b72fd0e942f7d0f2771a320dfd0034e303a604f8f`
- R141 logs/descent: `f0da0ce91e958076bd365b09adbaa91197c78ad761b7e68c6ea262b4882c7458`
- R141 tests: `3e900cf9ae19b4b32eb058886329760d40afdf053357358e97f49de4d0199b9a`
- R141 gate: `5874757459e62c4e7e9dd4f68f9109cc66262b2ed7cae6501c82e1be1d550f0f`
- R141 parent: `c6fa4314e6d0981e2b14c40d92ed40813c3276bb5068b06c6cf3107cd8dd331a`
- harness: `b8bb2503e11acb33224e89c7fdbc297eaea2ba4125fd71bc385b4e2b404f37da`
- harness tests: `848b951c513a3ea3a0649c14ae23b2509579ce8491b048ff2f4df97d35bb969f`
- report: `e8cf92df495bb47e7568566886594795a17cba354d78e7b65054c3e9cc323b39`
- note: `0a6a024b4b415953242c05db651824cb6ea79db00b33f721f13fa4b234c19f21`
- inventory: `e521e504b527faab87041c1e841037337130d25e1f271a708d722470f36fab02`
- replay plan: `23034faf011a84a4a1102be551a2229dac115f65a63872bd1419c34bf7242064`
