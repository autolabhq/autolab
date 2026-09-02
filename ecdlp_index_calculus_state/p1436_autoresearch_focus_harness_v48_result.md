# P1436 Autoresearch Focus Harness V48 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R98: nonlinear tensor-tower trace

R98 freezes a one-bond A/C tensor grammar

```text
K(u,v) = sum_(r=1)^w L_r(u) R_r(v)
```

with arbitrary nonlinear local encoders. On the monic-quadratic subfamily

```text
A_u(z)=z^2+u, C_v(z)=z^2+v,
Res(A_u,C_v)=(v-u)^2,
```

the resultant zero projector is exactly the `F_p` equality kernel. Its matrix
is `I_p`, so every exact one-bond representation has width at least `p`.
One-hot encoders show the bound is tight.

The identity, rank, and width-`p` factorization are exact for
`p=3,5,7,11,13`. Restricted kernels on `4,8,16,32` distinct messages over
`F_101` have the corresponding full ranks.

Under `p=Theta(B^5)`, the full bridge costs `B^5`. A restriction to
`D=Theta(B^(12/5))` distinct messages would still require width `D`, outside
the direct caps. R98 does not prove that the actual EC divisor image contains
that many distinct messages.

Duplicate occurrence count, blind bottom, and toy dyadic source replay are
exact. Collapsing equal bridge values loses an occurrence.

This is a scoped negative, not a lower bound on multi-edge digitized
encodings, nonalgebraic lookup, or a smaller actual EC divisor image.

```text
R98 producer  41d98744ff051a6dd259323e7a1298aea77a14b2d12d7cf0149b6eb592297a1e
R98 report    cb322e28620a4634ffac0c474d772c06a26010f10b3dc32c97c9e75637a90dfa
R98 gate      7acd34a7e3286edd658089a7d258688f9e327ffce03a185c762eee4eade06f3c
R98 parent    7d22c98b12c77a46c619acb7450336af7afcb315189ec2e81b5f73671c600f1a
R98 tests     bff539b87d73b10390c2a619471c1e1d714d88d6f8fb5e2db720f542baf42120
```

## Harness routing

V48 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v48.json`

SHA-256:

`e8653804bcb987db78b9d3a2efefa31ea9f9e4bf3a384de51d910aa4088ebc91`

Schema: `ecdlp.p1436_autoresearch_focus_report.v40`.

Thirty-four provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_multiedge_digitized_equality_projector
```

The next experiment must freeze a small-channel algebraic extractor and charge
its injectivity, cut capacity, lookup/interpolation state, target transition,
and occurrence-complete source reporting.

V48 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              e2ce149e6da143e26d2916ffacab99858bbe7b485d01fe53fa480e624d4b2cc7
harness tests        6ae9fd6c83f6bc585499136c61c4d418d13774c362a5f65126164efcd9d22473
V48 focus note       a1004b4016abe69c913fc499f0d14e540b5a3c411b1ba82ea0c8775328e35ef2
V48 FFE inventory    56ec8f7f6884e28f9f5cbacd3ac2d41563be60bc10c7e0e3c2c69b4860421d69
V48 FFE replay       06c25ff34871d172a1ff1b8249e01e4b94a80004f9a841d0734f31203746e526
```

## Verification

- R98 targeted tests: 8 passed.
- Harness tests: 64 passed.
- Full ECDLP task suite: 248 tests passed.
- Twenty-four R76-R98 and harness Python modules compiled.
- Twenty-three R76-R98 parent YAML receipts parsed.
- All 328 declared input/artifact hashes matched.
- No R98 parent or nested JSON artifact sets `breakthrough=true` or
  `shoup_bound_improvement=true`.
- A clean R98 rerun reproduced all six generated JSON artifacts byte-for-byte.
- V48 has 34 closed lanes, `promotion_allowed=false`, and the expected
  multi-edge digitized equality-projector action.
