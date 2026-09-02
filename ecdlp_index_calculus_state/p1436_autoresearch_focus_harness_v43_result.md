# P1436 Autoresearch Focus Harness V43 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R92: endpoint subfunction framework

R92 realizes exact public `MAP1`, `MAP2`, `f_d`, and endpoint `TR` semantics
for four rational maps on the fully enumerated prime-order curve

```text
E/F_101: y^2=x^3+7x+4,
#E(F_101)=97.
```

Every projective endpoint, including infinity, is returned exactly. For a
`B^5` endpoint space, the unavoidable partition coverage relation `D*L>=B^5`
combined with Dinur and Golovnev's Theorem 4.1 gives:

```text
all stored theorem advice, no online cap   B^(10/3)
optimistic free-randomness control         B^(5/2)
online-compatible setup minimum            B^(35/8)
online work at that point                  B^(5/4).
```

The maps return endpoints, not five-A plus five-C source tuples. An explicit
one-source-per-endpoint translator costs `B^5`.

This closes direct endpoint partitions using independently preprocessed
generic subfunctions, not joint or representation-changing constructions.

## R93: shared semilinear incidence

R93 constructs the surviving joint operator exactly. For quadratics

```text
f(z)=a*z^2+b*z+c,
g(z)=d*z^2+e*z+f,
```

the resultant factors as

```text
Res(f,g)=Ver2(a,b,c)^T K Ver2(d,e,f),
Ver2(a,b,c)=(a^2,ab,ac,b^2,bc,c^2),
```

where the frozen `6x6` matrix `K` has exact rank six over `F_101`. The
identity holds on all `78^2` frozen S3-chart pairs.

For pair-chart counts

```text
10, 21, 36, 55, 78,
```

the raw resultant matrix always has rank six, but the exact zero-incidence
matrix has ranks

```text
7, 18, 29, 53, 78.
```

At the largest scale the nonlinear zero projector is full rank. Every affine
`S4=0` has an exact signed four-point source, with zero predicate mismatches;
both proper-subsum and full-only branches are present.

At the full R84 root split, a constant-width feature per source still emits

```text
B^(12/5) and B^(13/5)
```

source-bearing rows. The rank-six operator compresses comparison arithmetic,
not row construction, indexing, or source unranking. Infinity, nonreduced
multiplicity, and full 5A+5C source return remain absent.

```text
R92 report  6604e1ac8e15c9048d06262a4116596ff793d4805901df8be16d67c2f170c24c
R92 gate    f4e5cfe45f25721bd11e60fab31a49866f38fe690b65b832d0b2ba7e20e78ad6
R92 parent  3945fa7856e3c4e028596e636279c2cbee4cb4fa618940519e6543e13ef2c299
R93 report  f33c764ebbf491e55d7b8342fda9df8b29038beef8cdb06662c343ac415b7d4b
R93 gate    15d58598c875b8ffe89fb1c7d551cd2adfc15cc8f7215a5f3242987305a2b2ad
R93 parent  df5b4b6cc6305a0304be9f8166179e692cee221f41343bf7ee8b51bf77960d2d
```

The Theorem 4.1 source bound in R92 is from Dinur and Golovnev,
<https://arxiv.org/abs/2512.04258v2>.

## Harness routing

V43 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v43.json`

SHA-256:

`6f9569fda9cebe28c78603a57d0a4775fbb4048443f0c73e3fecd444eb8989c8`

Schema: `ecdlp.p1436_autoresearch_focus_report.v35`.

Twenty-nine provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_implicit_veronese_hyperplane_source_index
```

The next experiment must act on the implicit Cartesian A/C parameterization
before `B^(12/5)` feature-row emission. It must answer the exact bilinear-zero
query, return a coupled source, fit `B^(9/4)` setup and `B^(5/4)` fresh
work/workspace, and cover infinity, proper subsums, tangencies,
multiplicities, and blind-zero controls.

V43 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R93 producer         9a82761a02cdb43fb0def4e959074eaa4e8e152dcea52901eec47bb1ab55c162
R93 tests            033bfbd6a7b515d6a21b7f757fbd2571c543af1651c5276036bf3d3b1ce8e3ba
harness              9c4a3f0b8ecd79eb13ade0a5c7d6691f1b0ba8d214bf0725bf49c3edef55560b
harness tests        1f9c8f1e58e1d0125bd1754a9de7c1093bbe549db71a77fdc1fdaf4062824d6c
V43 focus note       fa06611144ba3f2a7f3ae5d99e2f1b38475d9a22d1356f73a9034818ba0a9bb7
V43 FFE inventory    a96e5b4063aa13e3fe67e454e187f35b961241d76838780fb0fc9cfc59a79704
V43 FFE replay       7143fd5e5a481935094c1340b55862d2328ca4beebedb10c4590bc1808ac666a
```

## Verification

- R92 targeted tests: 8 passed.
- R93 targeted tests: 7 passed.
- R93 plus harness focused tests: 66 passed.
- Full ECDLP task suite: 206 tests passed.
- Nineteen R76-R93 and harness Python modules compiled.
- Eighteen R76-R93 parent YAML receipts parsed.
- All 237 declared input/artifact hashes matched.
- Every parent receipt and nested result has `breakthrough=false`.
- Clean R92 and R93 reruns each reproduced all six generated JSON artifacts
  byte-for-byte.
- V43 has 29 closed lanes, `promotion_allowed=false`, and the expected
  implicit Veronese hyperplane source-index action.
