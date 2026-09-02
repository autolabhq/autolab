# P1553 M6 symmetric-shift reverse-only marginal gate R153

## Claim boundary

R153 tests whether inversion symmetry can remove the separate
forward-tangent requirement from the R152 signed marker DAG.

It proves exact opposite-shift transpose and count identities. It also
records that the eight actual finite stacked systems have ranks `0..4` and
none is full rank. No reverse marker circuit or rank transfer is supplied.

Classification:

```text
KNOWN_LOG_A_DECK_AND_INVERSION_CLOSED_C_DECK_PRESERVE_EXPONENTS__A6_SHIFT_SET_SYMMETRIC__C5_KERNEL_EVEN__MARGINAL_BLOCK_DS_TRANSPOSE_EQUALS_D_MINUS_S__ROW_SUM_RECOVERS_6_COUNT__KNOWN_SHIFT_RELATIONS_EXACT__EIGHT_FINITE_STACKED_RANKS_ZERO_TO_FOUR_AND_NONE_FULL__REVERSE_ONLY_OPERATOR_REDUCTION_EXACT__TARGETABLE_DENSITY_SIGNED_FFE_REVERSE_MARKER_CIRCUIT_GENERIC_RANK_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH
```

## Symmetric family

Replace the small A deck by public known-log scalar multiples of `G`,
closed under negation:

```text
|A| = B^(1/12+o(1)).
```

The scalars are generated deterministically and their points are obtained
by public scalar multiplication. No DLP is used.

Replace the compact C deck by its public inversion closure:

```text
C <- C union -C,
|C| = B^(3/4+o(1)).
```

This changes only a constant factor. The Cartesian factor base remains

```text
F = A+C,
```

and each factor logarithm is a known A scalar plus one unknown C log.

## Reverse-only identity

Let `U` be the indicator of the inversion-closed C deck and

```text
K = U^(*5).
```

Both are even:

```text
U(-g)=U(g),
K(-g)=K(g).
```

The ordered A6 shift multiplicity `mu` is also even:

```text
mu(-s)=mu(s).
```

For each known shift `s` and C atoms `a,b`, define

```text
D_s(a,b) = 6 mu(s) K(s+a-b),
c_s(a)   = mu(s) U^(*6)(s+a).
```

Evenness gives

```text
D_s^T = D_(-s).
```

Because the shift batch contains both `s` and `-s`, a reverse action for
the opposite block supplies every forward action. A separate forward
tangent through the elimination setup is unnecessary.

Counts are row sums:

```text
sum_b D_s(a,b) = 6 c_s(a).
```

The subgroup characteristic is greater than six, so no independent count
oracle is needed once the block operator exists.

## Known-RHS relations

The public target for row `(s,a)` is

```text
[s]G + C_a.
```

Writing `ell_b=log_G(C_b)`, aggregate relation semantics give

```text
sum_b D_s(a,b) ell_b - c_s(a) ell_a
  = c_s(a) s.
```

The right side is public because `s` is a constructed scalar. Thus the
corrected block

```text
H_s = D_s - diag(c_s)
```

is a known-RHS system on C logs only.

## Density and rank

The scaling is:

```text
|A|^6 = B^(1/2+o(1))
|C|^6 = B^(9/2+o(1))
q     = B^(5+o(1)).
```

There are

```text
B^(1/2) shifts times B^(3/4) C targets
  = B^(5/4)
```

structured rows. Under a uniform endpoint model, the expected number of
aggregate relations is `B^(3/4)`, matching the C-log dimension.

This is model-bound only. The eight actual controls have stacked ranks:

```text
4, 0, 0, 0, 0, 0, 0, 0.
```

No control is full rank. The finite result neither validates nor refutes
asymptotic rank, but it forbids treating the uniform count as a transfer
theorem.

## Cost envelope

Symmetric closure preserves all exponents. If one reverse-adjoint marker
batch costs

```text
B^(5/4+o(1)),
```

then the R151 matrix-free solve remains conditionally

```text
B^(2+o(1)),
```

below setup and rho. R153 does not supply that reverse operator.

## Admission

Sixteen of twenty-seven obligations pass. The known-A construction,
inversion-closed C construction, C5 and A6 evenness, opposite-shift
transpose identity, reverse-to-forward action, row-sum counts, known-RHS
relations, exponent ledger, and finite rank deficit are admitted.

No full-rank finite control, targetable asymptotic density/rank transfer,
signed reverse marker circuit, factor logs without verifier labels,
identical descent, or ECDLP algorithm is admitted.

Disposition:

```text
ADMIT_SYMMETRIC_REVERSE_ONLY_OPERATOR_REDUCTION__ADMIT_FINITE_RANK_DEFICIT__DO_NOT_CREDIT_UNIFORM_DENSITY_MODEL__REQUIRE_TARGETABLE_MULTISCALE_RANK_AND_SIGNED_REVERSE_FFE_CIRCUIT__NO_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct only the reverse-adjoint signed marker operator for the
inversion-closed C deck and symmetric known-A6 shift batch. Its
geometry-only setup must fit `B^(9/4+o(1))` and one complete reverse batch
`B^(5/4+o(1))`; do not build a separate forward tangent. Because the
finite ranks are never full, add a preregistered multiscale targetable
density and rank transfer. Freeze signed FFE pivots, integer counts,
exact-residual factor logs, and identical descent without DLP, root,
count, marginal, rank, or source oracles.
