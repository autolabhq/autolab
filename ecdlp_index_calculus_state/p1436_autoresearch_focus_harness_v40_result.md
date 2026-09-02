# P1436 Autoresearch Focus Harness V40 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R90: nonlocal moment/Hankel translation screen

R90 instantiates one target-independent nonlocal nonlinear state rather than
leaving that exception abstract. For five C decks it freezes the endpoint
exponential-moment series and five fixed-marker series:

```text
G(z)   = product_i sum_j exp(c_ij z)
G_i(z) = weighted_i(z) product_(h!=i) unweighted_h(z).
```

Deck insertion is truncated-series multiplication. Target translation is
exactly `G_target(z)=exp(target*z)G(-z)`. Both identities match direct
finite-field moments on the frozen control.

Radix deck sizes 2, 3, and 4 give 32, 243, and 1024 distinct endpoints. The
norm and all five marker channels have Berlekamp-Massey complexity equal to
the full endpoint count on every control:

```text
32:   [32, 32, 32, 32, 32, 32]
243:  [243, 243, 243, 243, 243, 243]
1024: [1024, 1024, 1024, 1024, 1024, 1024]
```

Full Newton reconstruction recovers the annihilator, and the five marker
polynomials recover all 32 source tuples on the size-two positive control.
The semantics are therefore exact when the full state is supplied.

With `C=B^(3/5)`, exact moment/Hankel order is `C^5=B^3`. Even optimistic
quasi-linear updates exceed the `B^(9/4)` setup/state cap and `B^(5/4)`
fresh-work cap. This closes only exponential moments, Newton identities, and
Hankel/Padé reconstruction. Non-moment source-reporting indices and
representation-changing FFE identities remain open.

```text
report  02bece6fe25e335bd061eec784e237b6d9d2ab57f34bd8dd823217a137a56c2b
gate    b1618f6a354b995db01fbbc7aeeb69df6ebb5248b5c558bc4c72d87ce523897b
parent  ddde408f6f728446af72670a61d73ac5ee18f3843a4f2a20d09bf4b1be8099d3
```

## Harness routing

V40 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v40.json`

SHA-256:

`539ab388fef65f981c017675541caf1e2764fb9e6e0b10143cde11450c03708d`

Schema: `ecdlp.p1436_autoresearch_focus_report.v32`.

Twenty-six hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_unequal_list_subfunction_inversion_index
```

The next experiment must instantiate the `B^2` five-A versus `B^3` five-C
query as an explicit finite-field subfunction-inversion index. It must charge
preprocessing, query, source reporting, memory, random coins or
derandomization, and success amplification. It must fit `B^(9/4)` setup/state
and `B^(5/4)` fresh work/workspace, with exact signed and exceptional source
replay.

V40 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R90 producer         80e42f70a3540380b63b9db51462587780cc990852b9ac82580cdfe0baf2857a
R90 tests            cec0c14160f7923d5cceaa7c075350014c32ff37b441362313ac9b276383cf04
harness              0e1849dc4c633f1b611c403c8179adba1d8c2d8d84fea5c112dea005e755dfe4
harness tests        9a4056d8da0ecaf174481af2ca3874601295b1f16c16501d44b9b1137a1159e2
V40 focus note       c3e5bffef280e456b5600e004b9ededc773122f5376aa16aecd4c555098d3ade
V40 FFE inventory    0924b460ce2662f68da8fd3326cc79dbeb727d527b0b221b635e3a95e055c367
V40 FFE replay       fd914cde8a20139ffe4bb1645f4ccaea60a6ca99a98a95b42a4edd318799fa94
```

## Verification

- R90 targeted tests: 5 passed.
- Harness tests: 56 passed.
- Full ECDLP task suite: 181 tests passed.
- Sixteen R76-R90 and harness Python modules compiled.
- Fifteen R76-R90 parent YAML receipts parsed.
- All 186 declared input/artifact hashes matched.
- Every parent receipt and nested result has `breakthrough=false`.
- A clean R90 rerun reproduced all six generated JSON artifacts
  byte-for-byte.
- V40 has 26 closed lanes, `promotion_allowed=false`, and the expected
  unequal-list subfunction-inversion action.
