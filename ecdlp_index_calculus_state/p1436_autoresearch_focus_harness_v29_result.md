# P1436 Autoresearch Focus Harness V29 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R79: scalar-only nested norm

R79 removes the `B^5` TT state from R78. Every leaf is the actual scalar

```text
Res_z(S4(x1,x2,x3,z), S4(x_R,x4,x5,z)),
```

and every internal node is a scalar product over one deck. The root is zero
exactly when at least one tuple is a relation. Fermat indicators give the
exact tuple count, preserve duplicate occurrences, and count a tuple with
multiple common roots once.

Direct `B=3` controls pass on secp256k1, P-256, P-384, and P-521. The 32 R78
instances retain blind zero, forced count one, verified source, and dyadic
child conservation.

Depth-first streaming uses at most six live scalars, but:

```text
blind failed-zero work      B^5 leaves,
forced exact-count work     B^5 leaves,
cached source-tree state    1+B+...+B^5.
```

Thus streaming meets workspace but misses fresh-target work; caching misses
state. R79 closes only the explicit scalar-leaf SLP. A batched norm compiler
with a proved sub-`B^5` implementation remains open.

Report SHA-256:

`b5bd22b7256d80f5fa475517699c42f2201a1c8c6209718ea38ab832146a8929`

Gate SHA-256:

`55cd1e2185243cba0672d1b2374b126b9f34698b89772558a6e6bd857aafc2cb`

Parent receipt SHA-256:

`1829ffdb7938c645e70c1036f78903623c2ba2252222be5e070e672ed2878663`

## Harness routing

V29 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v29.json`

SHA-256:

`9ca84ce12d1d32736ab96fc03558083d32af9735d4ab93cf5bc73a2cc8318881`

Schema: `ecdlp.p1436_autoresearch_focus_report.v21`.

Fifteen hash-bound lanes are closed. The top frontier is:

```text
s6_batched_nested_norm_node_compiler
```

The next compiler must expose every resultant, subproduct, remainder,
modular-composition, transposed, count, source, and child operation at field
level. It must first prove a strict asymptotic improvement over `B^5`, without
materializing forbidden `B^3`/`B^2` bodies or treating a norm as unit cost,
before comparison with the direct caps.

V29 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. All alphaXiv-derived
autoresearch guidance checks pass.

## Verification

- Full ECDLP task suite: 120 tests passed.
- Ten changed Python modules compiled.
- R76 through R79 parent YAML receipts parsed.
- All 38 locally declared parent input/artifact paths and hashes passed.
- Archived R2 and R3 hashes remain bound through the present R31 registry.
- All four parent receipts have `breakthrough=false`.
- V29 diagnostic-only and nonpromotion checks passed.
- `git diff --check` passed after final freeze.
