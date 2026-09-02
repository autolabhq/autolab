# P1436 Autoresearch Focus Harness V38 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R88: black-box resultant localizer screen

R88 separates source localization from translated-resultant construction.
Given an exact subset zero-test oracle, one acceptance call and five balanced
child tests localize the unique endpoint among 32 left occurrences:

```text
1 + ceil(log2 32) = 6 calls.
```

The complementary endpoint and all ten source choices replay exactly. The
control reads a degree-243 right polynomial and a 243-word source dictionary,
so it proves conditional semantics only.

On the same accepted target, multiplication by `P_C(T-X)` in the 32-root
left quotient has one zero eigenvalue, 32 distinct eigenvalues, and scalar
Krylov Hankel rank 32. Its scalar recurrence therefore has full quotient
linear complexity on this control.

For quotient dimension `B^2`, a block width `B^alpha` has the optimistic
tradeoff

```text
iterations       B^(2-alpha)
stored blocks    B^(2+alpha).
```

The online cap requires `alpha>=3/4`, while the setup cap requires
`alpha<=1/4`. No tested width meets both. This treats each quotient matvec as
unit cost, so it is already favorable to the candidate.

R88 also reconciles the standard materialized half-gcd, nested-norm, and
coefficient routes with R80 and P1513. It preserves a coefficient-free
fixed-marker scalar resultant recurrence, non-Krylov determinant identities,
and unrestricted arithmetic circuits as scoped exceptions.

```text
report  d73e017c731a54c6913aeaa94e6b5c6d54ca3757f56be14e8ca8ee5524031de1
gate    90f0980bdeb51f540cd233f207218361cc14beb8a899c246418410d36ef43d56
parent  df5fba3dc2f9306541e86ed32d4237bcf4b8aab30d31c60430566bf2f0963c1e
```

## Harness routing

V38 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v38.json`

SHA-256:

`392dd397457d505c417d0901c83d6bebcc7ae49e5caddf4542b95a608ed39c66`

Schema: `ecdlp.p1436_autoresearch_focus_report.v30`.

Twenty-four hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_coefficient_free_fixed_marker_resultant_recurrence
```

The next experiment must derive zero, multiplicity, and all five right-source
markers directly from compact `D_A,D_C`, without `P_C` coefficients, a
`B^2` quotient/value vector or Krylov block, an endpoint dictionary, or a
unit-cost determinant oracle. It must fit the direct caps and replay every
projective exceptional source stratum.

V38 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R88 producer         bd67b19a13e6e27bd4b2315aef5374d68d44d52f03a9b63bff3cefb6d5b12874
R88 tests            5e7a9ffefda911a68525b3059981a22a6e859590990c7f48d606766537e7a74b
harness              5cb8c30d095425f3202a0d17df5ec5ef45b9cd54d374c37d3ad6cb1a245421a9
harness tests        b76e5ae41e87159ddea94ef7950f8fee36d7aa93bafeb3847576a9225f3a4fe7
V38 focus note       26fc6540128feb4383ee701bd1d6780fcb7c64c29757f3aea0a2970b0e0a03a4
V38 FFE inventory    cd155e13f38ba9dd2de7f9c0e9ab866343be377c24b14966527d573f02cfc9ea
V38 FFE replay       cfd1bd4b931e9234cfc8cdb6a0cc556b285437410c7a4b19d5761b2dd45ee28f
```

## Verification

- R88 targeted tests: 5 passed.
- Harness tests: 54 passed.
- Full ECDLP task suite: 169 tests passed.
- Fourteen R76-R88 and harness Python modules compiled.
- Thirteen R76-R88 parent YAML receipts parsed.
- All 157 declared input/artifact hashes matched.
- Every parent receipt has `breakthrough=false`.
- A clean R88 rerun reproduced all six generated JSON artifacts
  byte-for-byte.
- V38 has 24 closed lanes, `promotion_allowed=false`, and the expected
  fixed-marker scalar-recurrence action.
