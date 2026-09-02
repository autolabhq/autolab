# P1436 Autoresearch Focus Harness V27 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R77: target-translation orbit rank

R77 reduces the oriented singleton part of R76 to cyclic group convolution:

```text
g = f_4 * f_5,
count(R) = (f_1 * f_2 * f_3 * f_4 * f_5)(-R).
```

For a universal exact linear target-translation sketch, the span of all
translated pair queries has rank equal to the number of nonzero
group-character Fourier coefficients of `g`.

R77 grants scalar labels and exact finite-field characters as a diagnostic
advantage. It tests the four R70 prime-order toy groups, multiplicative-x and
independent hash decks, and `B=2,3,4,6,8`. All 40 pair kernels have full
ambient translation-orbit rank `q`. All exact convolution-theorem and integer
branch-mass controls pass.

The tested target sequences contain 1,801 exact zeros, so sparse outcome
support does not yield a missing linear query mode. The result closes only
universal linear shift-equivariant sketches. It does not reject nonlinear,
target-specialized, implicit-resultant, or Las Vegas representations.

Report SHA-256:

`73f66184fa53a2d43397a915c4249a41c5687cbcd1d5d16cc3e4ff47cf254787`

Gate SHA-256:

`45324f816cc159032cb6c2ac97c0a2f52a618c409e276616521b4e894cb46b55`

Parent receipt SHA-256:

`a3996cc9a28d2b77195684440d66213586e2b54359c84fc7fc798e82ac17df65`

## Harness routing

V27 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v27.json`

SHA-256:

`60f23a78a3567ce1606ac1fb638655ced38c8c702a18b009ff278e34206c17bc`

Schema: `ecdlp.p1436_autoresearch_focus_report.v19`.

Thirteen hash-bound lanes are closed. The top frontier is:

```text
s6_nonlinear_target_specialized_nested_resultant
```

The next grammar must operate on the actual S4 deck factors without group
characters, `B^3` prefix values, or a degree-`B^2` target polynomial/root
list. It must still return the R76 exact count, multiple-root correction,
blind zero, one source, and every dyadic child inside `B^(9/4)` setup/state
and `B^(5/4)` fresh-target work/workspace.

V27 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. All alphaXiv-derived
autoresearch guidance checks pass.

## Verification

- Full ECDLP task suite: 110 tests passed.
- Six changed Python modules compiled.
- R76 and R77 parent YAML receipts parsed.
- All 14 locally declared parent input/artifact paths and hashes passed.
- Archived R2 and R3 gate hashes are bound through the present R31 registry.
- Both parent receipts have `breakthrough=false`.
- V27 diagnostic-only and nonpromotion checks passed.
- `git diff --check` passed after final freeze.
