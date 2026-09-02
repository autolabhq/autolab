# P1553 M6 matrix-free marginal Jacobian-Krylov gate R151

## Claim boundary

R151 tests whether a surviving V99 marker-count circuit must materialize
the complete aggregate marginal matrix before factor-log linear algebra.

It proves an exact matrix-free interface reduction and derives a
conditional `B^2` linear-algebra envelope. It does not construct the
required bidirectional marker circuit or its reusable derivative state.

Classification:

```text
AGGREGATE_MARGINAL_MATRIX_IS_LOG_WEIGHT_JACOBIAN__FORWARD_DIRECTION_GIVES_MX__REVERSE_SCALAR_CONTRACTION_GIVES_MT_LAMBDA__WIEDEMANN_ACCEPTS_OPERATOR_ACTIONS__CONDITIONAL_B5O4_APPLY_TIMES_B3O4_ITERATIONS_EQUALS_B2_BELOW_SETUP_AND_RHO__BAUR_STRASSEN_DOES_NOT_PRESERVE_OFFLINE_ONLINE_STATE__BIDIRECTIONAL_WEIGHT_PARAMETRIC_COUNT_CIRCUIT_OPEN__NO_GENERIC_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Jacobian interface

Let `Z(w)` be the vector of aggregate known-target fiber counts as a
function of logarithmic A/C atom weights. At `w=1`, the aggregate marginal
matrix is

```text
M = D_log Z(1).
```

For any atom direction `x`,

```text
Mx = coefficient of epsilon in
     Z(w_a*(1+epsilon*x_a)).
```

Thus `Mx` is one forward directional derivative of the count-vector
circuit.

For any row contraction `lambda`,

```text
M^T lambda
  = D_log (sum_i lambda_i Z_i)(1).
```

Thus `M^T lambda` is one reverse gradient of a scalar-contracted count
circuit.

The two actions carry the exact bilinear certificate

```text
lambda dot (Mx) = x dot (M^T lambda).
```

All eight R144 actual control systems satisfy this identity exactly over
their subgroup fields. Their selected full-rank systems, factor logs, and
shifted descents also replay exactly. The matrices and labels are
verifier-only and receive no candidate or asymptotic credit.

## Primary bounds

Baur and Strassen prove constant-factor nonscalar overhead for computing a
scalar rational function together with all first derivatives:

```text
Walter Baur and Volker Strassen,
The Complexity of Partial Derivatives,
Theoretical Computer Science 22 (1983), 317-330,
doi:10.1016/0304-3975(83)90110-X.
```

Wiedemann gives finite-field linear algebra using a matrix only through
operator applications:

```text
Douglas H. Wiedemann,
Solving Sparse Linear Equations Over Finite Fields,
IEEE Transactions on Information Theory 32(1) (1986), 54-62,
doi:10.1109/TIT.1986.1057137.
```

R151 pins local copies of both papers and uses only the stated interface
and asymptotic bounds.

## Cost envelope

At the selected six-factor vertex:

```text
meaningful log dimension:          n = B^(3/4+o(1))
known-target row count:                B^(3/4+o(1))
A6 markers per row:                    B^(1/2+o(1))
complete scalar marker batch:          B^(5/4+o(1))
explicit marginal matrix entries:      B^(3/2+o(1)).
```

Suppose one frozen scalar-blind, weight-parametric circuit applies both
`M` and `M^T` to arbitrary vectors in

```text
B^(5/4+o(1))
```

work. Wiedemann uses `B^(3/4+o(1))` operator applications, so the
conditional solve costs

```text
B^(5/4+3/4+o(1)) = B^(2+o(1)).
```

This is below both:

```text
setup cap: B^(9/4+o(1))
rho:       B^(5/2+o(1)).
```

The final residual must be checked exactly. Structured generic-prime rank,
failure probability, field, extension, chart, integer-lift, memory, and
bit costs remain charged conditions.

## Offline/online boundary

Baur-Strassen differentiates a complete circuit. If that circuit contains
`B^(9/4)` deck-dependent setup, a fresh reverse pass differentiates or
replays the setup as well.

Therefore Baur-Strassen alone does not produce either:

- a weight-independent frozen setup; or
- reusable division-safe tangent and adjoint state.

One of those objects is required before the `B^(5/4)` operator-application
cost receives candidate credit.

## Admission

Sixteen of twenty-five obligations pass. The Jacobian identities,
matrix-free bilinear certificate, eight finite operator replays, two
primary-source bindings, explicit `B^(3/2)` matrix-output charge, and
conditional `B^2` solve envelope are admitted.

No bidirectional marker-batch operator, reusable derivative state,
generic-prime rank/density theorem, factor logs without verifier matrices,
identical candidate descent, or ECDLP algorithm is admitted.

Disposition:

```text
ADMIT_MATRIX_FREE_MARGINAL_JACOBIAN_INTERFACE__ADMIT_CONDITIONAL_B2_KRYLOV_LINEAR_ALGEBRA__DO_NOT_INFER_OFFLINE_ONLINE_DERIVATIVE_STATE_FROM_BAUR_STRASSEN__PRESERVE_WEIGHT_PARAMETRIC_MARKER_CIRCUIT__NO_GENERIC_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct one scalar-blind weight-parametric marker-batch circuit whose
frozen setup is weight-independent or carries reusable division-safe
tangent and adjoint state. It must apply both `M` and `M^T` in
`B^(5/4+o(1))` work without forming marginal rows, `q` modes, `q` targets,
or C3 occurrences. Then run an exact-residual matrix-free solve and
shifted descent without a DLP, root, Fourier, recurrence, algebra-state,
count, marginal, rank, or source oracle.
