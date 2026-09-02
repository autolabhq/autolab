# P1436 Autoresearch Focus Harness V49 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R99: multi-edge digitized equality projector

R99 gives the R98 escape hatch its proper positive credit. With supplied
canonical binary digits,

```text
EQ(x,y) = product_i (1-(d_i(x)-d_i(y))^2)
```

computes exact equality through `ceil(log2 p)` width-two channels. This holds
for every pair over `p=5,7,11,13,17`.

The flattened channel capacity remains at least `p`, so the construction is
compatible with R98's rank theorem. More importantly, the small-edge equality
gadget does not construct an aggregate index.

Every nonempty radix-digit fiber has an exact polynomial indicator of degree
`p-1`. R99 verifies all interpolations, but does not turn degree into a
circuit lower bound.

The frozen standard constructors cost

```text
full digit/fiber tables and root lists   Theta(p log p) = B^5 polylog(B),
sourcewise occurrence digit traffic      Theta(D log p) = B^(12/5) polylog(B).
```

Both miss the direct caps. Duplicate occurrence count, one dyadic occurrence,
and blind bottom are exact when every source digit vector is supplied.

This is a scoped negative against materialized digit/fiber tables and
sourcewise extraction. A succinct aggregate digit trie derived directly from
the compact A/C divisor circuits remains open.

```text
R99 producer  3497a069b3bb5b2590537344d68ca0d1cefd0995de6cd739d8f79b352c862e23
R99 report    dac6fbf38357e640bb174860df16117fdd3461717cb3b2ceac4e76ec7f77707c
R99 gate      a7e69ad7cf661f09c679039c5ed9874940eca5432d7ca91dc980f97005bfe560
R99 parent    94adc8bf8f97be2bfcb8c08953dd6634a9356ad534c7b96f3faedc35bb95b7bb
R99 tests     f87d268b013369a594ed4abf7b5bdbc46254f42f1b6212df791601a245abdef5
```

## Harness routing

V49 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v49.json`

SHA-256:

`c24033bdef648cf49ccb2ae4775e8dd82866c35d4717eb814ed36aa98fee5f9d`

Schema: `ecdlp.p1436_autoresearch_focus_report.v41`.

Thirty-five provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_succinct_aggregate_digit_trie
```

The next experiment must freeze a leaf-free digit/fiber summary and merge law
compiled from the compact divisor circuits. Supplied pairwise digits,
sourcewise scans, and `p`-size tables receive no further credit.

V49 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              7ce2f9bd05d67b7d4439cc0405e43148eb26718e1a6d5c9d92560bb67b5c8556
harness tests        d0a18469ecc187bc6a50c3b1cc4ce633c38538b1b6cb89973d73d2a75aeb61eb
V49 focus note       d34cfd297cc49cc30215e325186b3d7fea7be777084df4077f196d57a3ca1829
V49 FFE inventory    0e1731135ebadb68abb64be1e1341e89908839552bb160bd3d46a2ae5f8c6ed6
V49 FFE replay       e54abb689b7480701251cc9b5ee2f9b6d6750462a6a899b4befc2825a5db76f9
```

## Verification

- R99 targeted tests: 7 passed.
- Harness tests: 65 passed.
- Full ECDLP task suite: 256 tests passed.
- Twenty-five R76-R99 and harness Python modules compiled.
- Twenty-four R76-R99 parent YAML receipts parsed.
- All 346 declared input/artifact hashes matched.
- No R99 parent or nested JSON artifact sets `breakthrough=true` or
  `shoup_bound_improvement=true`.
- A clean R99 rerun reproduced all six generated JSON artifacts byte-for-byte.
- V49 has 35 closed lanes, `promotion_allowed=false`, and the expected
  succinct aggregate digit-trie action.
