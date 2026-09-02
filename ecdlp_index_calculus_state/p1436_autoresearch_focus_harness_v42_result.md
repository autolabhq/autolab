# P1436 Autoresearch Focus Harness V42 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R92: compact elliptic subfunction-map screen

R92 instantiates public `MAP1`, `MAP2`, `f_d`, and endpoint `TR` semantics on
the fully enumerated prime-order curve

```text
E/F_101: y^2 = x^3 + 7x + 4
#E(F_101) = 97.
```

The field and group orders are prime, the discriminant is nonzero, the full
addition table is closed, and a nonidentity point has exact order 97.

Four projective rational maps return every endpoint exactly, including
infinity:

```text
h=x       : D=49, L=2
h=y       : D=63, L=3
h=x+y     : D=71, L=3
h=x+2y    : D=60, L=3.
```

Every map satisfies its rational-degree fiber bound and `D*L>=97`. Five
matched deterministic random partitions also satisfy the coverage law.

For an asymptotic endpoint space `N=B^5`, any deterministic endpoint
partition with `D=B^d` fibers and largest fiber `L=B^ell` obeys

```text
d + ell >= 5.
```

Dinur and Golovnev's Theorem 4.1 builds advice of size

```text
S = soft-O(L^(3/2-delta) D + Aux + L^delta)
T = soft-O(L^delta).
```

Charging the stored `L^delta` randomness term, the best theorem-certified
setup point even without an online cap is

```text
ell=10/3, delta=1
S = B^(10/3).
```

Deleting that term as an optimistic free-randomness control still leaves
`B^(5/2)` setup. With fresh work capped at `B^(5/4)`, the optimum is

```text
ell=5/4, delta=1
D = B^(15/4)
S = B^(35/8)
T = B^(5/4).
```

The endpoint translator is not a five-A plus five-C source translator. An
optimistic one-source-per-endpoint control is exact only through an explicit
`B^5` dictionary.

This closes direct endpoint partitions using `D` independently preprocessed
generic subfunctions. It is not a lower bound on jointly compressed,
overlapping correspondences or a representation-changing Semaev/FFE
construction.

```text
paper   e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c
report  6604e1ac8e15c9048d06262a4116596ff793d4805901df8be16d67c2f170c24c
gate    f4e5cfe45f25721bd11e60fab31a49866f38fe690b65b832d0b2ba7e20e78ad6
parent  3945fa7856e3c4e028596e636279c2cbee4cb4fa618940519e6543e13ef2c299
```

Primary reference:
<https://arxiv.org/abs/2512.04258v2>.

## Harness routing

V42 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v42.json`

SHA-256:

`3d29da42a7566d90ecbe9d4f8e9e3f6bd3867ce8c75f55f80e56f55b4640f04c`

Schema: `ecdlp.p1436_autoresearch_focus_report.v34`.

Twenty-eight provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_shared_semilinear_incidence_correspondence
```

The next experiment must freeze one target-dependent family of overlapping
`S3`/`S4` incidence charts and a single shared semilinear operator. It must
demonstrate joint state below `B^(9/4)`, fresh work/workspace below
`B^(5/4)`, and exact projective 5A+5C source unranking, without decomposing
into independent fiber tables or using endpoint/source dictionaries.

V42 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R92 producer         4f3724031a14de218ddda89120d32518ceb7b0430baf147b95a049227c8f8332
R92 tests            a39f0f93eccb8a1a6451f5717101ceeea48598706e301c04077bbea523a9635d
harness              870a1e58b9f20c02c6e971fb1b95466f4b04c59a728171330e9aac0a72f0900b
harness tests        625673929b6eeed19693758e86fb1b32f72150e636e340b3831de557627eee40
V42 focus note       ec91c40c281be04824d75ea06e0deef7ab33cb6c8a78fea0ef0869e2dbcce4d6
V42 FFE inventory    5e1b53026c370196f88a5f36a3afbb3f26eba69df89ae03f03d453918a26a3d7
V42 FFE replay       065f0f97fe5a16dcdb311849d08b006f49ff9cf8260da1616203ce052cc9be4b
```

## Verification

- R92 targeted tests: 8 passed.
- Harness plus R92 focused tests: 66 passed.
- Full ECDLP task suite: 198 tests passed.
- Eighteen R76-R92 and harness Python modules compiled.
- Seventeen R76-R92 parent YAML receipts parsed.
- All 220 declared input/artifact hashes matched.
- Every parent receipt and nested result has `breakthrough=false`.
- A clean R92 rerun reproduced all six generated JSON artifacts
  byte-for-byte.
- V42 has 28 closed lanes, `promotion_allowed=false`, and the expected shared
  semilinear incidence-correspondence action.
