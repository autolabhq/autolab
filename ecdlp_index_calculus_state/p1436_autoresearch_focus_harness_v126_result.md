# P1436 autoresearch harness V126 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V126 binds R177 as the 113th closed frontier lane and routes the first
experiment to `s69_output_sensitive_marked_fitting_locator` at priority 316.

## R177 Result

Let `T_D` be the reduced pair algebra of the R176 selected divisor `D x D`.
Let `K` be multiplication by the principal signed-incidence value

```text
k(P,Q) = h(P+Q),
```

and let `X_1` be multiplication by `x(P)`. R177 defines the global marked norm

```text
F(A, lambda)
  = det(K + lambda (A I - X_1))
  = product_(P,Q) (h(P+Q) + lambda (A-x(P))).
```

If `M` is the number of signed incidence pairs, then exactly

```text
ord_lambda F = M
```

and

```text
[lambda^M] F
  = pdet(K) product_(h(P+Q)=0) (A-x(P)).
```

After monic normalization, this coefficient is

```text
det(A I - X_1 | ker K).
```

Thus its roots are the candidate left coordinates with pair-incidence
multiplicities. Since `U` is squarefree, one gcd with `U` returns exactly the
distinct R176 roots. This eliminates the R175 adaptive subset-query tree from
the exact algebraic interface.

## Exact Controls

Across six controls:

```text
pair-algebra dimension sum:                 8,922
kernel-dimension and incidence sum:           241
marker-polynomial degree sum:                 241
candidate-factor degree sum:                  140
distinct R176 roots recovered:                140
maximum candidate incidence multiplicity:       4
marker interpolation samples:                 247
truncated lambda updates:              28,788,282
```

For every control, all coefficients below lambda degree `M` vanish. The
interpolated degree-`M` coefficient exactly equals the pseudodeterminant product
formula, and its candidate gcd exactly matches R176. These finite controls are
fully charged and receive no asymptotic credit.

## Cost Boundary

```text
selected divisor degree n:                    B^(9/4)
target witness degree N:                      B^(5/4)
candidate marker degree M:                    B^(3/4)
standard pair-algebra dimension n^2:          B^(9/2)
full bivariate marked-norm body:               B^9
explicit symbolic lambda truncation state:    B^(3/2)
explicit pair scan:                           B^(9/2)
explicit marker interpolation work:            B^6
conditional output-sensitive Fitting total:   B^(9/4)
R163 label/backpointer postprocessing:          B^2
rho proxy:                                    B^(5/2)
```

The output polynomial is small enough, but no below-rho constructor for it has
been supplied. Standard pair-algebra, generic matrix-pencil, full determinant-
body, and explicit interpolation routes remain above rho. These are standard-
route cost boundaries, not arithmetic-circuit lower bounds.

The surviving primitive is a fraction-free output-sensitive marked Fitting or
subresultant operator that accepts compact `U,V,h`, computes

```text
M = dim ker K
det(A I - X_1 | ker K)
```

in softly `O(n+N+M)` total work without representing `n^2` pair state, the
full lambda/A coefficient body, or an `M^2` marker grid.

R177 passes 15 of 22 obligations. It admits the global norm identity, exact
lambda valuation, restricted-kernel characteristic polynomial, candidate gcd,
and six finite controls only. Constructor admission, lane admission, rho
improvement, Shoup improvement, and breakthrough flags remain false.

## V126 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v113`.
- Bound and closed frontier lanes: 113.
- First focus: `s69_output_sensitive_marked_fitting_locator`.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R177 focused tests: 19 passed in 37.948 seconds.
- Harness tests: 143 passed in 0.736 seconds.
- Full ECDLP suite: 1,203 passed in 576.384 seconds.
- R177 clean replay: all six JSON outputs byte-identical through the focused
  deterministic-bundle test.
- V126 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R177: 102 receipts, 2,014 recursive path/hash bindings,
  zero mismatches, missing paths or rounds, and duplicate rounds.

## R177 Hashes

- Producer: `6aad043e4279175f5b7838eb536b003108b11a0c278df11f9b0737db75ffe923`
- Report: `d37a2f5a79a41150127d6f1be540522d06c1cb02553e76a43936d58995e95394`
- Frozen interface: `101a2dd4eccc198acbdecd7c76219c1a268a9061815434db6c8f3a0583a0bac7`
- Cost ledger: `dc4649daa8069f7bb393929f6cf433b6610a875f644293829a9b9be3f1971a74`
- Replay: `41ce8a59968d8fbc493d39e9fdf55aec6a8c916ae46a9a834e76671b5b07d782`
- Controls: `f8a1cb00056fec06b7b54db6c648a9ed4d615c3b808595ad79d98d7583926feb`
- Marker ledger: `d5d4f83e2e145c13f93bb9fa0f8132850b6bfb6bdc3fa58ad5b3779acff40980`
- Tests: `8d8b17d48c39842597abdc9d58c6cd135a8b2c1f8224a60d3a2ff86a49483717`
- Gate: `a670f43be024d898f67d37d5a0cb6f26aa033406633ee4017ab6a340f0c35c5c`
- Parent: `179d8a8010fd231e6bce5b868185876f30180b41b60c91ec2c5c7e64b440a538`

## V126 Hashes

- Harness: `869890f579ca98b5d019838fe94e9301c669148a4ff1e50e10f98d86a4e30774`
- Harness tests: `bce656b7b8a86ca71cb26257f1cca71fd83107886478d4b3f669280ac654e809`
- Focus report: `f5c79795ac7d7222e1916a61b830990779dd75278b897d4224e121257a7a35a1`
- Note: `9e625c49c1b9d06e5833a8a524e8abffe91453818d06a1e470f23eea60b93b3e`
- Evidence inventory: `9f6ec13179ad09f2c53f75ad7eedab7e87f020672993f6be5443cdf7f8979ab5`
- Replay plan: `140a78bc70d292f844ed5b5bb02dfb10e2e6083fe8499fa8ed8be45f5fa57689`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact determinant identities,
finite controls, gcd recovery, verifier passes, and represented-route negatives
receive no unconditional attack or circuit-lower-bound credit.

## Next Action

Construct or refute a fraction-free output-sensitive marked Fitting or
subresultant arithmetic DAG on compact `U,V,h` inputs. Reject explicit `n^2`
pair or tensor state, generic `n^2` matrix pencils, the full lambda/A
determinant body, `M^2` marker interpolation, candidate inversions, and
unit-cost Fitting, kernel, resultant, root, count, marginal, rank, source, DLP,
or generic locator oracles.
