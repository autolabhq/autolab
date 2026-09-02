# P1436 autoresearch harness V118 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V118 binds R169 as the 105th closed frontier lane and routes the highest
priority experiment to
`s61_fraction_free_elliptic_fitting_subresultant_mod_u` at priority 300. The
askalphaxiv source post remains integrated in the harness methodology:

`https://x.com/askalphaxiv/status/2076737985559822734?s=46`

## R169 result

R169 packages the R168 denominator-aware trace as the scalar resolvent

```text
chi_P(lambda)
  = product_(Q in S union -S) (h(Q+P) + lambda).
```

All inherited witness denominators are units on the selected divisor. The
lambda-adic valuation of `chi_P` at zero therefore equals the R168 candidate
multiplicity. This preserves candidate zero-divisor information without
inverting `h` at a candidate nonunit.

For nonzero regularizing lambda, R169 tests the differentiated kernel

```text
K_lambda(P,Q)
  = Dlog((F_num + lambda F_den) / F_den)(Q+P).
```

Six controls over three curves and signed-divisor degrees 10, 35, and 56
verify exact monic degree-`2n` pencils, exact candidate roots and
multiplicities, and full row rank of the regularized kernel. All tested
ordinary diagonal `x/y` Sylvester-minus, Sylvester-plus, and Stein
displacements also have full row rank. Powers one through four of the
`x`-Sylvester displacement remain full row rank.

A two-seed sweep over witness degrees two through eight on the largest curve
adds fourteen controls; the raw kernel and ordinary `x`-Sylvester displacement
both have rank 56 in every row.

This is a scoped negative for those finite operators, not a lower bound. A
custom elliptic companion displacement and fraction-free Fitting/subresultant
algorithm remain open.

## Cost boundary

```text
compact h and logarithmic witness state:       B^(5/4)
regularized n-by-2n kernel matrix:              B^(9/2)
full-rank displacement generator state:        B^(9/2)
generic pencil coefficients or samples:        B^(9/2)
preferred fraction-free output work:           B^(9/4)
rho proxy:                                     B^(5/2)
```

Generic interpolation requires `2n+1` values per selected endpoint, so its
`Theta(n^2)` sample or coefficient state is above rho. A full-rank diagonal
generator has the same problem. The surviving primitive is a fraction-free
elliptic Fitting/subresultant denominator modulo `U`, strictly below
`B^(5/2)` total work and preferably `B^(9/4+o(1))`.

R169 passes 21 of 29 obligations. Lane admission, rho improvement, Shoup
improvement, and breakthrough flags are false.

## V118 routing

The harness schema is `ecdlp.p1436_autoresearch_focus_report.v105`.

- Bound frontier preflights: 105.
- Closed frontier lanes: 105.
- First focus: `s61_fraction_free_elliptic_fitting_subresultant_mod_u`.
- Natural full-rank cells: 0 of 1.
- Natural verified-log cells: 0 of 1.
- Natural below-rho cells: 0 of 1.
- Promotion allowed: false.

The decisive test freezes `U,V,F_num,F_den`, their invariant derivatives, all
public corrections, and the exact scalar-resolvent specialization. It rejects
`2n+1` lambda samples per endpoint, full-rank diagonal generators, `nN` or
`n^2` state, tensor materialization, candidate inversion, generic-lambda-only
values, and unit-cost Fitting or subresultant oracles.

## Verification

- R169 focused tests: 16 passed in 32.12 seconds.
- R169 plus harness tests: 151 passed, 6 subtests passed in 32.82 seconds.
- Full ECDLP suite: 1,066 passed, 10 subtests passed in 221.99 seconds.
- R169 clean replay: all six JSON outputs byte-identical.
- V118 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R169: 94 receipts, 1,812 recursive path/hash bindings, zero
  mismatches, zero missing paths or rounds, and zero duplicate rounds.
- Syntax, JSON/YAML parsing, whitespace, parent-hash, and no-promotion
  validation: passed.

## R169 hashes

- Producer: `213b8d9ca7b900adef6af241efac79dbe8ca01a2283b3cd166ba56d8cfd7d36f`
- Report: `6f9123923396d0c6478486c9a669cb73ff759db00fa8997954f5b4a35d11ce86`
- Frozen interface: `df7d50d0bf60835d4513950c0961c65cb8901c20afe3b4dcea75a5a9e488cbf5`
- Cost ledger: `bab32516614e139ed989a6d7186aa1daed08c8e2efb2815d1b273c3b1a685a1f`
- Replay: `f6538b5474971bb9dbdf594b3efd8eb3c47bf202b518942b35871e8407a27044`
- Controls: `954308ce9d715fb0d904297923b0bb7316e7d1ff625831368a261e55daba5e5b`
- Pencil ledger: `cc36dc603f9c28da1194a1b699a71f670566a40aad2af12038d8f18131648d18`
- Tests: `fe149185aa0eddc72b896d48bf3ceadd05c6e5ab9502b9b809c4ab396badc2ac`
- Gate: `7aba9b4c16a35dc7db1f1ad5953ef50e597a882644260c028d8b6d7a412d185f`
- Parent: `90778d88d7c268d7fdc5c4a5257a350e8f3b15cff5f527537bf9ca7710248e78`

## V118 hashes

- Harness: `99754c0043b9958a93cd742b51d54b1493275e3365acdc5dd2d19a7b9c576f9b`
- Harness tests: `8e90019da523aaa736587eeebe98c8c07dd7763ddd40ab8855d7847e440c9d9a`
- Focus report: `d6dd63c2102c27950c01f6de26aad24cc4d38cb7871b52a339dc9b91e7a4069f`
- Note: `a5ba218a956b9e770d63e362e63e611c64c8af46774486a86bb4b384de09f944`
- Evidence inventory: `30bc01a700ca894b14f0847621ba42ffef4028eec714a77a578214fcda48b912`
- Replay plan: `e53cce3666a2fb1c883a044bcc60d069fcd85775cf4039b655bc4d8041bd5289`

## Claim boundary

No generic-prime ECDLP algorithm was produced. No Pollard-rho or Shoup lower
bound improvement was demonstrated. Exact scalar pencils, finite full-rank
controls, verifier passes, and harness progress receive no asymptotic attack
or lower-bound credit.

## Next action

Derive a fraction-free subresultant or Fitting presentation directly in the
elliptic coordinate algebra modulo `U`. The first decisive checkpoint is an
explicit arithmetic DAG whose intermediate state and work remain below rho;
a custom elliptic companion displacement is viable only with a proved
low-rank generator and fully charged application cost.
