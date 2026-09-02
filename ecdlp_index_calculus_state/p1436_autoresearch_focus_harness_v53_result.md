# P1436 Autoresearch Focus Harness V53 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R102: exact two-sided baseline

R102 constructs the best setup-eligible direct join. A canonical `4A+1C`
prefix dictionary uses `B^(11/5)` state; a fresh `1A+4C` suffix scan uses
`B^(14/5)` work. Boundary indices preserve exact unordered integer counts and
one jointly coupled source.

All eight actual all-target histograms and first sources match direct
enumeration. Exhausting all 34 nontrivial atom-count splits confirms
`B^(14/5)` is the best direct query exponent under `B^(9/4)` setup.

The source and group exponents are both `B^5`. Ideal independent pruning from
`B^(14/5)` to `B^(5/4)` needs exponent `31/20`; density-one survival requires
the same repetition exponent and restores `B^(14/5)`.

```text
R102 producer  3fae17c49683586178518a0df8d377c8f553bd131539f1bab579c0f55d5b0742
R102 report    5f16489b5f0535df6c9728edfed5f2f83641a4b9c30fd2609db52a73f46d40e7
R102 gate      1af82597762525110baf31cec7b4e7b96693a730270f7e1bd6556906e5c5f56c
R102 parent    9a965158bc96e51dd8daea1abb86f5552b75dac94998633ec97b22f8b9a3ad67
R102 tests     073cc397a7237ce6da7dff810d3aadb63a8fdaada5f1fea97258b9ccd7b535b5
```

## R103: target-forced S3/FFE filter

For regular affine `L,T`, R103 proves exactly that
`S3(x(L),X,x(T))` has roots `x(T-L)` and `x(T+L)`. All actual true pairs
survive. A sign-complete synthetic control shows all four signed points pass
the x-filter while only `T-L` is the desired join.

Pointwise local-oracle composition costs `B^(16/5)` best. The canonical
query remains `B^(14/5)`. Explicit endpoint x-polynomials and materialized
base-field FFE factor lists retain `B^(12/5)` or larger bodies.

R103 closes these standard realizations only. A sign-resolved
target-specialized S3/FFE pushdown before partial endpoint emission remains
open.

```text
R103 producer  9ebba96331379a99abaa4596772855a884a62137828edefd5f6a2d81478152dd
R103 report    587d8283716ee8568aab1576376b57b786f412c7c16d4e3942d271b276a5173c
R103 gate      63e155a40d7762e1bc362d72610a91c075f5075b35997e680e74c82c29215aa2
R103 parent    315fa9d6511bb95fa6647413537ad2d2d62f6f44815dcd4adb397194dd7e3ab0
R103 tests     5ac18c568661a64eeba7ac6219ed6a224b3d0195491d875fdec46bc38e246975
```

## Harness routing

V53 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v53.json`

SHA-256:

`23d188c7bb4f3e8941886b9a46686ff539f1c131b9f85e5d4d6e803a299b5e06`

Schema: `ecdlp.p1436_autoresearch_focus_report.v45`.

Thirty-nine provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_compact_preendpoint_s3_ffe_pushdown
```

R104 must push a sign-resolved target specialization through compact
`D_A,D_C` state before any `4A+1C` or `1A+4C` endpoint list, polynomial
root body, or provenance leaf is emitted.

V53 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              60f213216a88c39fdf7e3c34ce27554d281ea86f1e86ab4e3d58e2ad2b409fb8
harness tests        a1ea49bb21944630a1a930293e9fdd7d94693eb0c91c5cc0d52ee7fd8d565fb4
V53 focus note       2da4617b24286d569574dbbac24856428a1db307f84f15e8e11046a7b07a43c9
V53 FFE inventory    bc84f46dd5bb0277649389288b2bfb7f4e01b2ae6f35517d6fee3520578fabfd
V53 FFE replay       92858083b6b5471b491e9948424db51317c974d50eb4d079e27809ea5797ba18
```

## Verification

- R102 targeted tests: 7 passed.
- R103 targeted tests: 7 passed.
- Harness tests: 69 passed.
- Full ECDLP task suite: 288 tests passed.
- Twenty-nine R76-R103 and harness Python modules compiled.
- Twenty-eight R76-R103 parent YAML receipts parsed.
- All 430 declared input/artifact hashes matched.
- No R102/R103 parent or nested JSON artifact sets `breakthrough=true` or
  `shoup_bound_improvement=true`.
- Clean R102 and R103 reruns reproduced all generated JSON artifacts
  byte-for-byte.
- V53 has 39 closed lanes, `promotion_allowed=false`, and the expected
  compact pre-endpoint S3/FFE pushdown action.
