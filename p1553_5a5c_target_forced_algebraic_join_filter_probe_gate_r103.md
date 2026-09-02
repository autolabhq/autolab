# P1553 5A/5C Target-Forced Algebraic Join Filter Gate R103

Date: 2026-07-29

Status: `S3_IDENTITY_EXACT__STANDARD_REALIZATIONS_OVER_CAP`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can a necessary Semaev/FFE relation preserve every true `L+R=T` source and
reduce the R102 two-sided join below the direct caps?

R103 freezes the affine x-coordinate invariant, `S3(x(L),x(R),x(T))=0`,
its sign branches, and pointwise, aggregate-polynomial, and materialized FFE
realizations before outcomes.

## Exact regular factorization

For affine regular `L,T`, the quadratic in `x(R)` has roots

```text
x(T-L) and x(T+L).
```

Equivalently, up to leading coefficient `(x(L)-x(T))^2`,

```text
S3(x(L),X,x(T))
  = (X-x(T-L))(X-x(T+L)).
```

All regular root and coefficient factorizations replay exactly on 16 actual
positive/blind target controls. Every true source survives.

The x-coordinate loses signs. A synthetic control contains the four points

```text
T-L, -(T-L), T+L, -(T+L).
```

All four pass `S3`; exactly `T-L` is the desired join. Point-level replay is
still required.

## Charged realizations

Pointwise composition is only a constant number of translated local-oracle
queries per enumerated endpoint:

```text
enumerate 2A+3C, query 3A+2C: B^(19/5)
enumerate 3A+2C, query 2A+3C: B^(16/5).
```

The R102 canonical split remains `B^(11/5)` state and `B^(14/5)` fresh
query work. The constant-degree `S3` test does not change either exponent.

Explicit left and right x-polynomials have degrees `B^(13/5)` and
`B^(12/5)`, above setup. A fresh canonical suffix polynomial has degree
`B^(14/5)`, above the online cap. Since all endpoint x-roots lie in the base
field, materialized FFE factorization produces the same number of linear
factors and source payloads.

This closes these standard representations only. It does not close a compact
target-specialized S3/FFE pushdown before partial endpoint emission.

## Admission

Passed obligations: `21/32`

Target-forced S3 identity admitted: `true`

Full lane admitted: `false`

Missing gates include compact pre-endpoint pushdown, exact in-cap integer
count and coupled source, projective/degree-drop/tangent/proper-subsum charts,
known-RHS rank, factor logs, identical descent, and a complete Shoup
comparison.

## Exactly one next action

Construct or refute one sign-resolved target-specialized S3/FFE pushdown
acting directly on `D_A,D_C` before `4A+1C` or `1A+4C` endpoint emission.
Freeze its recurrence and projective charts; require exact count and one
coupled source inside both caps with no root or verifier oracle.
