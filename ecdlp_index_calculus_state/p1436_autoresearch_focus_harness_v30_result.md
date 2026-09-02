# P1436 Autoresearch Focus Harness V30 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R80: batched nested-norm compiler

R80 compiles the R79 `B^5` scalar leaves by separating each actual-S4
relation into a triple prefix and target-plus-pair suffix. Balanced product
trees have degrees `O(B^3)` and `O(B^2)`. Their polynomial gcd plus identity
tags decides existence, and gcd-guided tree descent returns a relation source.

Exact tuple count still requires the R76 alternating endpoint-subset
histograms. A synthetic repeated/multiple-root control separates:

```text
exact tuple count       4,
product gcd degree      3,
naive singleton count   6.
```

Across secp256k1, P-256, P-384, and P-521 at `B=3,4,5`, all 24 product-gcd
existence decisions are exact, every blind count is zero, every forced count
is one, and every positive product-tree source verifies by signed group law.

This is a strict improvement over the R79 leaf program, but it misses the
direct caps:

```text
                         achieved       required
setup state              B^(3+o(1))     B^(9/4+o(1))
fresh-target work        B^(3+o(1))     B^(5/4+o(1))
fresh-target workspace   B^(2+o(1))     B^(5/4+o(1))
```

In particular, reducing the degree-`B^3` prefix product modulo a fresh
degree-`B^2` suffix product costs `B^(3+o(1))`; it is not charged as a
unit-cost norm, resultant, or gcd.

Report SHA-256:

`936537fb78908dd2916bf6fa5b2091f336b9a47217a1ff787b068ae0491992c5`

Gate SHA-256:

`1ff3641688f4f0e13fd64f83aa540ea429a4164e0d0741b64b38acd804d7fb01`

Parent receipt SHA-256:

`80b4aec794d8c9a5b69f0ec6b8d98da447a26b6f8d8d232d326718092c222593`

## Harness routing

V30 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v30.json`

SHA-256:

`5279a1e8b537f01d5e3bc05f2002733182887702adfb1896970e3899b81ef492`

Schema: `ecdlp.p1436_autoresearch_focus_report.v22`.

Sixteen hash-bound lanes are closed. The top frontier is:

```text
s6_structured_factor_base_endpoint_compression
```

The next experiment must freeze one scalar-blind structured factor-base
geometry before outcomes. Its triple-endpoint representation must use at
most `B^(9/4+o(1))` setup/state and its fresh query plus source return at
most `B^(5/4+o(1))`. The same frozen construction must provide prospective
relation density, matched random rank controls, verified factor logs, and
identical fresh-target descent.

V30 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Verification

- Full ECDLP task suite: 125 tests passed.
- Twelve changed Python modules compiled.
- R76 through R80 parent YAML receipts parsed.
- All 49 locally declared parent input/artifact paths and hashes passed.
- All five archived gate hashes remain bound through the current R31
  registry.
- All five parent receipts have `breakthrough=false`.
- V30 has 16 closed lanes, `promotion_allowed=false`, and the expected top
  structured-factor-base action.
- `git diff --check` passed after final freeze.
