# P1436 Autoresearch Focus Harness V31 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R81: full multiplicative-x cosets

R81 tests the complete-coset residual left open by R70 and R80. On four
prime-order relation-scale toy families, it freezes four complete
multiplicative coordinate cosets before outcomes, applies the exact curve
lift mask and cofactor map, and compares each result with two matched
hash-to-curve controls.

The coordinate domain remains sparse:

```text
product_(x in g^e H) (X-x) = X^d-(g^e)^d.
```

That sparsity does not survive elliptic addition. Across all 16 cosets:

```text
unordered triple -> endpoint-set map injective     16/16,
distinct endpoint-set count                        C(B+2,3),
root-support exponent range                        2.7304..2.8870,
candidate root-support fitted slope                3.0175,
matched-control root-support fitted slope          2.9825,
root supports inside B^(9/4) cap                   0/16.
```

The toys retain ordinary density and known-RHS rank behavior: 13/16
candidate cosets and 27/32 controls reach full sampled relation rank, and
candidate mean density differs from controls by only `-0.0079`. The
compression failure is therefore not explained by a starved relation
pipeline.

Verifier-only BSGS labels replay all endpoint, count, source, and rank
statements. Their `B^(5/2+o(1))` work exceeds the setup cap and receives no
algorithmic credit.

Report SHA-256:

`e556efa7c1e639f76915f152ebdcc3d00db2a932d8ef207ae8653c94117f026f`

Gate SHA-256:

`cb3c26ad300ee02e357ef1892c0335e519acd3c29e509f15efb79bf48a806086`

Parent receipt SHA-256:

`61a1f469378d46cbedc3765a189852740ba216b7c1706f68db11819a80376b14`

## Harness routing

V31 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v31.json`

SHA-256:

`dbfeac4eae47864bda169619b2958d0856fc2b717aeb4e5fe17198c3407404b0`

Schema: `ecdlp.p1436_autoresearch_focus_report.v23`.

Seventeen hash-bound lanes are closed. The top frontier is:

```text
s6_compact_divisor_factor_base_endpoint_compiler
```

The next experiment must freeze a scalar-blind compact divisor or
straight-line factor-base description and compile it into triple S4
endpoints without materializing `C(B+2,3)` keys. It must prove at most
`B^(9/4+o(1))` setup/state and `B^(5/4+o(1))` fresh query/source work, then
pass prospective density, matched full rank, verified factor logs, and the
identical fresh-target descent.

V31 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Verification

- Full ECDLP task suite: 130 tests passed.
- Fourteen changed Python modules compiled.
- R76 through R81 parent YAML receipts parsed.
- All 60 locally declared parent input/artifact paths and hashes passed.
- All six archived gate hashes remain bound through the current R31
  registry.
- All six parent receipts have `breakthrough=false`.
- V31 has 17 closed lanes, `promotion_allowed=false`, and the expected top
  compact-divisor action.
- `git diff --check` and explicit trailing-whitespace checks passed after
  final freeze.
