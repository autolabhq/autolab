# P1436 Autoresearch Focus Harness V28 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R78: actual-S6 nonlinear TT projector

R78 freezes a target-specialized value-first nonlinear grammar:

```text
T(i1,...,i5)
  = Res_z(S4(x1,x2,x3,z), S4(x_R,x4,x5,z)),
Z = 1 - T^(p-1).
```

Every intermediate is represented as an exact tensor train, and the Fermat
projector uses binary Hadamard powering.

On secp256k1, P-256, P-384, and P-521, both target types and every
`B=3,4,5,6` have exact raw TT ranks

```text
[B, B^2, B^2, B].
```

The mandatory center core therefore contains `B^5` entries before Fermat
projection. The first Hadamard square retains those ranks.

Every blind final zero mask is identically zero. Every forced mask is one-hot,
has TT rank one, returns a verified source, and conserves every dyadic child
count. That tiny final representation receives no constructor credit because
the value-first path has already crossed `B^5` state and work.

R78 closes only this exact TT/Hadamard grammar. Scalar-only straight-line
resultant or norm circuits that never represent tuple values remain open.

Report SHA-256:

`e590002c433c6504725d5ab7ff1dba97ad8c15400bf0117846742da7359c5e60`

Gate SHA-256:

`9149c3b903d91a2156125a2c61896dc1111eaeffafdf0536c89a50b7e0e84ff9`

Parent receipt SHA-256:

`3310e7fc082ed0da9dd2d71eda411e6f1ec154d0d6e75d1239da2665275ee6a0`

## Harness routing

V28 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v28.json`

SHA-256:

`e1f67c44704ceb34d22ff33b0646c359197b145fba0f725e71c98f47b6adb1b6`

Schema: `ecdlp.p1436_autoresearch_focus_report.v20`.

Fourteen hash-bound lanes are closed. The top frontier is:

```text
s6_scalar_only_black_box_nested_norm
```

Every scalar, product, and norm node must now expose its type, state, and work.
No node may hide a tuple tensor, TT core, coefficient cube, quotient algebra,
degree-`B^2` suffix object, or outcome oracle. The program must retain R76
exact count, duplicate/multiple-root semantics, blind zero, one source, and
every dyadic child inside the direct caps.

V28 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. All alphaXiv-derived
autoresearch guidance checks pass.

## Verification

- Full ECDLP task suite: 115 tests passed.
- Eight changed Python modules compiled.
- R76, R77, and R78 parent YAML receipts parsed.
- All 26 locally declared parent input/artifact paths and hashes passed.
- Archived R2 and R3 hashes remain bound through the present R31 registry.
- All three parent receipts have `breakthrough=false`.
- V28 diagnostic-only and nonpromotion checks passed.
- `git diff --check` passed after final freeze.
