# P1436 Autoresearch Focus Harness V50 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R100: succinct aggregate digit trie

R100 freezes the universal family of all `D`-element subsets of `F_p` and
requires exact membership plus one occurrence or bottom.

There are `binomial(p,D)` possible subsets. Different subsets need different
persistent states because a query in their symmetric difference must receive
different answers. Therefore an `F_p`-word state needs at least

```text
log_p binomial(p,D) >= D(1-log_p D)
```

words. Under `p=Theta(B^5)` and `D=Theta(B^(12/5))`, this is at least
`13D/25`, retaining exponent `B^(12/5)` and missing the `B^(9/4)` setup cap.

Exact finite state counts and field-word capacities are replayed for
`(p,D)=(17,4),(31,5),(61,7),(127,10)`. Explicit radix tries have `D`
terminal records; binary Patricia tries retain `D` leaves and `D-1` branch
nodes. Duplicate occurrence return also retains all occurrence payloads.

The structured interval family is an exact two-word positive control for
membership, bottom, and source index. It proves that the universal
information bound is not a lower bound on every structured endpoint image.

R100 therefore closes universal arbitrary-set tries only. It does not claim
an entropy bound or rule out a short leaf-free summary for the actual R84
`3A+2C` endpoint image.

```text
R100 producer  168f462c50aa89be7e513d21fe9f34f6080bf04922c192306997805029b88bd9
R100 report    fc7281ba084824963da550dfb8f034baa4f6d133305e18af11684e2be61154dd
R100 gate      653ad0f51c9ec6f88a89769ca757f45ae70432cb947148d447eda968286a2e50
R100 parent    f8ab3fba3270ac00c80503c8bbcf4a03b98a23dbfc7ce5648c4336aea447f26e
R100 tests     a93d4aa57d877c64e95b1193d7c16fa0811e7720444b625af75b4cd1393334b1
```

## Harness routing

V50 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v50.json`

SHA-256:

`0728f433e2b56676f5aeadd0c3780e23af8910bdbf1a0abc290918bb3caa5c92`

Schema: `ecdlp.p1436_autoresearch_focus_report.v42`.

Thirty-six provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_actual_divisor_image_entropy_merge
```

The next experiment must analyze the reachable `3A+2C` image directly from
`D_A,D_C`: either construct a subcap leaf-free summary and source query or
exhibit an injective reachable high-entropy family. Arbitrary subsets and
unrelated structured controls receive no credit.

V50 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              7fce947e0696dcc9bd8f4ccbb716dcb1b102acf6ad3a915d86e93ff23c0ac2be
harness tests        ef011e19031704ebaa45ea2f8bb38acdafef196bfbd6956c0d5dc23a8a7f5cfb
V50 focus note       ae8eb8fd629ae7904f9ed16f71a4528a1a8a6a43c8bb82502b26460433b961d9
V50 FFE inventory    5205759b6d633fd324c561f877706404a61808977aaa8539a26925595ce0b552
V50 FFE replay       dc4826cfa62fbc769c7c13fbc7499c095664c08fd55a3698b83e51b59d0582af
```

## Verification

- R100 targeted tests: 7 passed.
- Harness tests: 66 passed.
- Full ECDLP task suite: 264 tests passed.
- Twenty-six R76-R100 and harness Python modules compiled.
- Twenty-five R76-R100 parent YAML receipts parsed.
- All 365 declared input/artifact hashes matched.
- No R100 parent or nested JSON artifact sets `breakthrough=true` or
  `shoup_bound_improvement=true`.
- A clean R100 rerun reproduced all six generated JSON artifacts byte-for-byte.
- V50 has 36 closed lanes, `promotion_allowed=false`, and the expected actual
  divisor-image entropy/merge action.
