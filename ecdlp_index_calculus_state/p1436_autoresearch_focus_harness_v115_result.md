# P1436 autoresearch focus harness V115 result

Date: 2026-08-01

## Result

V115 binds R166 as the 102nd closed frontier lane and routes the highest
priority action to
`s58_output_sensitive_kummer_translate_product_remainder`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R166 sets the R165 global randomizer to zero. The hard function becomes

```text
f_0(Q) = U(x(Q)),
Phi_0(P) = product_j U(x(T_j-P)).
```

Its zero divisor is `S+(-S)`. A zero therefore represents either a true
selected-sign C3 endpoint or an opposite-sign endpoint. No true signed match
is lost, and the signed y-coordinate dictionary removes every opposite-sign-
only root after candidate factorization.

The fixed function uses only `U`, but translating it to a selected signed
point `P` still requires `V`, or the equivalent signed y side table. R166
preserves that `B^(9/4)` input state; it removes the y residual from the
product zero test, not the sign data from elliptic translation.

For a planted weight-six target `v.C`, a false-sign equation has coefficient
form `v-p+q`. Its coefficient sum is six, so it is nonzero and has l1 norm at
most twelve. Under independent uniform cyclic C labels, every fixed equation
has probability `1/q`; the planted false-pair expectation is `B^(1/4)` and the
full unplanted-target bound is `B^(3/4)`. Expected candidate verification is
therefore `B^2`, below the `B^(5/2)` rho proxy. Deterministic hash-to-curve
transfer is not proved and receives no credit.

The six inherited finite batches contain 241 true signed pairs and zero
opposite-sign pairs. Six 4,096-target matched-density controls contain 100
true and 89 opposite-sign pairs, against a combined iid expectation of about
85 per orientation. A forced `T=P-Q` control creates six opposite-sign
candidate roots and no true roots; signed verification removes all six.

The standard represented Kummer translate product still has zero and pole
divisor degrees `2nN=B^(7/2)`, above rho. R166 does not supply an
output-sensitive product remainder modulo `U`, a deterministic hash transfer,
an unconditional generic-prime algorithm, a Pollard-rho improvement, a Shoup
improvement, or an ECDLP breakthrough.

## Verification

- Focused R166 tests: 16 passed.
- Full harness tests: 132 passed, 6 subtests passed.
- Full ECDLP suite: 1,015 passed, 10 subtests passed in 255.34 seconds.
- R166 clean replay: all six JSON outputs byte-identical.
- V115 clean replay: note byte-identical; three JSON outputs identical after
  removing `generated_at` timestamps.
- Parent audit R76-R166: 91 receipts, 1,740 recursive path/hash bindings, zero
  mismatches, no missing paths, and no missing or duplicate rounds.
- V115 frontier preflights: 102 provided, 102 closed.
- R166 obligations: 24 of 31 passed; lane admission false.
- R166 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V115 natural full-rank, verified-log, below-rho, promotion, and Shoup gates:
  false.

## Hashes

- R166 producer: `c524eca7bdb586e8e68e34da515da301766668f6f0b7bd6808e005b2b77e893a`
- R166 report: `94ee650b1302d231d946fc02f14d1c12b62a63bd9fba240a9d6a93e4bb986285`
- R166 frozen interface: `e413a3f78adf508fbc22e2b05e4b2d1d5986871473de0cafa78ba89f5f2fb21a`
- R166 cost ledger: `3b6ce95aa27864c5fe475ea36d745afba2108189138ade240f2f1ee4cbd43857`
- R166 replay: `b459ab51185d403d0de91b5ff1700d43138e3e2e7281f4cdfe38ff0d1a0760ee`
- R166 controls: `72a28dbed3e056b4f1cab123546e786ae2ebc058bd012338afc7bde3ff6d9fc8`
- R166 Kummer/sign ledger: `ff79f8c98a284f4a71cd3c529e65d11b76b77af216eb33089d50245a91fa6e5a`
- R166 tests: `39570b95fbc355095a5f494ae091889d11d11b92b753b3caf0f74c5b5e63a751`
- R166 gate: `e98cd1ba5b73c591dae5882e9058fafbd6e548df3c2ae929563181bc84920d9c`
- R166 parent: `a10abdaa1f1c075ab0c0b77a709a1757d675f0369e9d7c51f276cb25e1c7c502`
- Harness: `b7828d6deb448b5bb66c386cbf396f4f0b4a7cc7bbc4f9e16874cf1a515334ad`
- Harness tests: `746cfc8a46e8017f125393dacab665360acd56d87aa0469369e080c393d7808a`
- Report: `2de9ec36ca7f7f56737374cee53935ee7323de72ac2b6b54403f7a58350bd289`
- Note: `7bea2468fd80a573009b84d4d9f4bad223c12b8b28831b1c692263a2ebcc8687`
- Inventory: `fa96d52e7fa29b88796d6d7a6d10a1f6b8068bd9d5c8638cd1fa746fc3dd9f67`
- Replay plan: `72e22528049d1c18405c8d22fe93aac69dee2e9c4c65189ad353943da10659d7`

## Next action

Construct or refute
`gcd(U,product_j U(x(T_j-P)))` below `B^(5/2)`, preferably in
`B^(9/4+o(1))`, without expanding the degree-`2nN` divisor or the `n` by `N`
value table. Preserve `V`, or the equivalent signed y side table, as the
translation input and reuse it for the output-sized post-verifier. Separately
prove deterministic hash-to-curve transfer for the signed-difference density
bound.
