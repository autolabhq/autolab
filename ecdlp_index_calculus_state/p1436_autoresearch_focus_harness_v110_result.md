# P1436 autoresearch focus harness V110 result

Date: 2026-08-01

## Result

V110 binds R161 as the 97th closed frontier lane and routes the highest
priority action to
`s53_target_batched_signed_divisor_modular_composition_gcd`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R161 instantiates the coordinate-specific escape left open by R160. For the
signed C3 endpoints with distinct x-coordinates, it constructs

```text
U(X) = product_P (X-x(P)),   V(x(P)) = y(P),   U | V^2-(X^3+aX+b).
```

For a target `T=(u,v)` and formal endpoint `P=(X,V(X))`, it computes in
`F_p[X]/U`

```text
lambda = (v+V)/(u-X)
phi    = lambda^2-u-X
psi    = lambda(u-phi)-v.
```

Outside the explicitly split roots `u=X`, `(phi,psi)=T-P`. Positive C3+C3
sources are therefore exactly the roots of

```text
gcd(U, U(phi), psi-V(phi)).
```

For a target with one positive-C6 coefficient source, every split is a
size-three submultiset of six occurrences, so the source gcd has degree at
most `binomial(6,3)=20`. This makes source extraction output-sensitive once
the gcd exists, but it does not make modular composition cheap.

The persistent divisor has degree `B^(9/4)`. Even granting optimistic
quasi-linear composition, evaluating two compositions and two gcds
independently for each of `B^(5/4)` targets costs `B^(7/2)`, above Pollard
rho. The surviving hypothesis is a many-inner target-batched composition/gcd
adjoint with `B^(5/4+o(1))` complete-batch work and no per-target
degree-`B^(9/4)` materialization.

All six public-curve controls construct exact signed divisors, recover every
preregistered unique positive-C6 relation/descent source, reject an empty
target, and exercise the denominator-exception split. Observed maximum source
gcd degrees are `6, 6, 14, 10, 14, 14`. The controls consume no candidate DLP,
root, count, marginal, rank, or source oracle and receive no attack credit.

No many-target composition algorithm, deterministic hash-to-curve transfer,
unconditional generic-prime family algorithm, Pollard-rho improvement, Shoup
improvement, or ECDLP breakthrough is claimed. Novelty of the signed-divisor
quotient/gcd formulation is also unverified.

## Verification

- Focused R161 tests: 16 passed.
- Full harness tests: 127 passed.
- Full ECDLP suite: 932 passed in 228.061 seconds.
- R161 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R161: 86 receipts, 1,625 bindings, 0 mismatches, no missing
  paths, and no missing or duplicate rounds.
- V110 frontier preflights: 97 provided, 97 closed.
- R161 obligations: 18 of 26 passed; lane admission false.
- R161 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V110 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R161 producer: `da25fe5203e243746879ecfc699da912b299ea0781f72df222cdb322bba00e80`
- R161 report: `7e73325212e9d8c3cf42ae46cce0a5c1bdebea9784b9bf5c30761c006cb8bedd`
- R161 frozen interface: `8c9c3981cbf9f1529547c61cc9673e778079beba86a306eef1ac54979d2520b7`
- R161 cost ledger: `511aeedfbb72a268e6bef87178d156cc700eccbfc49c4b7a8441e473d0461b4b`
- R161 replay: `96dee1fb3ab5423a2f5463827135e6d70b7c124c7b5171a49b3a048eb28b08c3`
- R161 controls: `6baa61c223de901691e08771da7f8af860a21195565d40be87c9662209cfa955`
- R161 logs/descent: `70d9dc60c96d4b54f107904f31e4a6800b943fb060b2d4e4b894587da21f69d1`
- R161 tests: `661edbc66d4dda3179a9829e59a2ddef5e816f11497a1a334513a1ae46676a38`
- R161 gate: `0ca8660b9554c39e2c90a8f29cee05554b9fb3adce0c4c2a1038e739b3888bcd`
- R161 parent: `67fc0a45c5493011d0d14188150c08402aaf78539c966a8840ca11e723134429`
- Semaev primary PostScript: `991f85d58ab68551a229266d03c2f88a5fc42e81b2a5f8f4432937bcceff16df`
- Harness: `54b3c9c65ad9330a79c89649f1e534d6c99fcec68f8fe4fdce5c79f51d337836`
- Harness tests: `254fc76347501d5b63c406cba60c43410ced97203970c300687a1481503c2f0d`
- Report: `cacf24d02c9e67acba2850cf9cfabde3267d1a2c6265209a6030fb8596167937`
- Note: `f9b0d230dcfcc6efa217ac4c04bf73a0d80cb596b177eba2ac61b73de3a521f0`
- Inventory: `003b5fc456b7338f3144973489fc430d29bb0be77dad5e65c25ffffde4d42497`
- Replay plan: `7793ae835b5ea2111359eb795bbad6559f588364107d31917b88d33c825ac346`
