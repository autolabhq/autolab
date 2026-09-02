# P1436 Autoresearch Focus Harness V45 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R94: implicit Veronese index

R93's rank-six quadratic-resultant kernel gives an exact hyperplane value
`H`. R94 identifies its zero query as

```text
delta_0(H) = 1 - H^(p-1),
```

the R9 Fermat-projector router specialized to the Veronese kernel. On frozen
pair-chart counts

```text
10, 21, 36, 55, 78,
```

the projector ranks are

```text
7, 18, 29, 53, 78.
```

Direct counts followed by dyadic descent return one exact signed affine source,
but use fewer than two full scans rather than changing the exponent. The
smaller R84 source side has `B^(12/5)` values, so materialized features,
direct scanning, and the bound all-output multipoint evaluator miss the
`B^(9/4)` setup or `B^(5/4)` online caps.

The multipoint comparison uses the output-sensitive upper bound in
Bhargava-Ghosh-Guo-Kumar-Umans,
<https://arxiv.org/abs/2205.00342>. It is not a lower bound on a zero-only
aggregate algorithm.

## R95: aggregate moment recurrence

For quadratic coefficient triples `u=(a,b,c)` and `v=(d,e,f)`,

```text
Res(u,v)
  = (a*f-c*d)^2 - (a*e-b*d)*(b*f-c*e).
```

R95 constructs the exact coefficient-moment contraction for
`Res(u,v)^(p-1)`. After quotienting the raw rank-six symmetric-power
coordinates by all quadratic Veronese relations, the canonical moment vector
has exact dimension

```text
binomial(2p,2) = p(2p-1).
```

At

```text
p = 3, 5, 7, 11, 13, 17, 19, 29,
```

the exact coefficient-pairing ranks are

```text
15, 45, 91, 231, 325, 561, 703, 1653,
```

full at every frozen scale. This finite sweep is not an asymptotic rank
theorem.

An `F_11` no-wrap box verifies the aggregate integer count, monic squarefree
blind zero, duplicate occurrence multiplicity, and one dyadically recovered
zero source. Under `p=Theta(B^5)`, however, the canonical moment state is

```text
Theta(p^2) = Theta(B^10),
```

far outside both P1515 caps. This closes the all-monomial moment contraction,
not modular Frobenius traces, character sums, streaming nonlinear circuits,
or source-reporting indices.

```text
R94 report  a9966f1fb407e720ce1e9aaafc82ab4a38eade9f338cda287cb737445d099df0
R94 gate    66d2004d1b4dfa63ac69f60397da45e379dfca4bb8db12ede0647f028aaf517d
R94 parent  9059deb6ac6a42660f73bee6d6b1b800b2be827b17965a59994941cbc2fd646a
R95 report  50441deecd5fadcb05d30c09532d2b5fe6e8893ea2530243c20a52761e0a1742
R95 gate    6645a20b942f4c86abafed39fa14538b1bb80edee3ffa94a2e7d914e12c0c763
R95 parent  2c28724fe73c6d32197100ac256f18ebc719ded3a7be8653774814d61c77d746
```

## Harness routing

V45 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v45.json`

SHA-256:

`cf1382e9291abfbc2c6ef26f875a4bdb6554c13e3aeee13ae6bb29e972f983ac`

Schema: `ecdlp.p1436_autoresearch_focus_report.v37`.

Thirty-one provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_modular_frobenius_trace_recurrence
```

The next experiment must give explicit quotient/trace state, Frobenius
transition, range restriction, integer lifting, and exact coupled-source
replay. Merely invoking `H^p=H`, pointwise Fermat powering, or a quotient trace
does not supply a contraction.

V45 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R94 producer         607cf20cb9091611d4d6085b7fc8046c35b5d64ed59817d4db2289d2b3ee0661
R94 tests            d131bc16217cf76d94e959a3a8487b89477ad8416942ad18caa27707de37dea3
R95 producer         494ed8e091bd965d235dfb5b3b350be2f9493eaeb1bc676229116259f59126f9
R95 tests            9dfb78178847468f5dea08c1279e83cc11a13bdf8e8c46ea4030889eb32b638b
harness              7ccf3f196538eff3dd3f0d5ca019d8cb04f25207f5e25428fe215777f7462050
harness tests        1861db558dce7287c7e7f679ccdfe43d5d7b848092bed7c1a74e9141d2e7c0f5
V45 focus note       a428d0c65be1aca3a3c465bd4ac15e915cbfbbb473d6e7668fb1b9917bac9764
V45 FFE inventory    e443468bc68da189a0ea19b384ff89b9ef65ed07ab56bdf8a27b15135378892a
V45 FFE replay       5f1200025b4406a749b1574b3ba9d564451c94c94b45f1caa112b5352cacce20
```

## Verification

- R94 targeted tests: 7 passed.
- R95 targeted tests: 7 passed.
- R94, R95, and harness focused tests: 75 passed.
- Full ECDLP task suite: 222 tests passed.
- Twenty-one R76-R95 and harness Python modules compiled.
- Twenty R76-R95 parent YAML receipts parsed.
- All 272 declared input/artifact hashes matched.
- No parent receipt or nested JSON artifact sets `breakthrough=true` or
  `shoup_bound_improvement=true`.
- Clean R94 and R95 reruns each reproduced all six generated JSON artifacts
  byte-for-byte.
- V45 has 31 closed lanes, `promotion_allowed=false`, and the expected modular
  Frobenius trace action.
