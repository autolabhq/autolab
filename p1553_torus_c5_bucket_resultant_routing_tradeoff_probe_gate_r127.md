# P1553 torus C5 bucket-resultant routing tradeoff gate R127

## Claim boundary

R127 closes balanced bucket-polynomial correction circuits that evaluate
bucket-pair product resultants independently. It covers all-pair queries,
quotient-style target routing through one bucket pair per left bucket,
represented symbolic bucket resultants, and represented dense
target-bucket routing tensors.

It does not cover shared transposed evaluation across resultants, an
implicit constant-pair router at the cap-tight singleton-C3 point, adaptive
cell probes, nonlinear non-resultant certificates, or general arithmetic
circuits and data structures. It supplies no complete source index, rank,
factor logs, identical descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
BALANCED_H_BGAMMA_BUCKET_RESULTANT_PAIR_WORK_B9O4_MINUS_GAMMA__ALL_H2_QUERY_B9O4_PLUS_GAMMA__QUOTIENT_H_PAIR_QUERY_B9O4__POLYLOG_REQUIRES_GAMMA9O4_RHO0_EXTREME__SYMBOLIC_OUTPUT_B15O4__ACTUAL_COORDINATE_HASH_ROUTING_AND_SOURCE_CONTROLS_EXACT__CAP_TIGHT_C3_SINGLETON_IMPLICIT_O1_ROUTER_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Balanced tradeoff

Let the hash have

```text
H = B^gamma
```

balanced buckets. The represented C2 and C3 bucket-polynomial degrees are

```text
deg C2_bucket = B^(3/2-gamma),
deg C3_bucket = B^(9/4-gamma).
```

Granting optimistic quasi-linear resultant work in the larger represented
degree, one bucket-pair query costs

```text
B^(9/4-gamma).
```

Evaluating all `H^2` pairs costs

```text
B^(9/4+gamma).
```

If target routing selects `R=B^rho` bucket pairs, the query cost is

```text
B^(rho+9/4-gamma).
```

Even an ideal quotient or Latin-square routing rule has `R=H`, one right
bucket for every left bucket, and therefore costs `B^(9/4)`.

Polylogarithmic query requires

```text
rho <= gamma-9/4.
```

Since a nonempty C3 bucket model has `0<=gamma<=9/4` and `rho>=0`, the only
cap-compatible corner is

```text
gamma = 9/4,
rho   = 0.
```

This is a cap-tight `B^(9/4)` C3 singleton-bucket index with an implicit
`O(1)`-pair arbitrary-target router and source locator. R127 does not
construct that router.

## Represented routing costs

Precomputing symbolic bucket resultants has total output degree

```text
sum_(a,b) deg(P2_a) deg(P3_b)
  = |C2| |C3|
  = B^(15/4),
```

above the setup cap.

A dense represented target-bucket routing tensor has `H^3` state. Fitting
that tensor under `B^(9/4)` forces `gamma<=3/4`, where even one routed
resultant has exponent at least `3/2`. Sparse or formulaic routing is not
covered by this dense-tensor charge.

## Actual controls

Twenty-four controls use all eight R82 pairing decks and coordinate hashes
at moduli 2, 3, and 5. They verify:

- all-bucket-pair product-resultant membership on every positive product and
  one empty subgroup target;
- exact target-bucket routing through the observed correction tensor;
- replay of a C2+C3 source for every positive routed hit;
- equality of total symbolic bucket-resultant degree and `|C2||C3|`;
- no pairing-image discrete logarithm is consumed.

Finite controls receive no asymptotic credit.

## Scope limits

The exponent formulas charge independent represented resultants. A shared
transposed circuit might reuse work across routed bucket pairs. More
importantly, the extreme singleton-C3 point would meet the frozen setup and
query caps if an exact implicit constant-pair router and source locator
existed. R127 preserves both possibilities and proves no general
cell-probe, RAM, or circuit lower bound.

## Admission

Twelve of twenty obligations pass. The bucket-routing semantics and scoped
resultant tradeoff are admitted. The extreme router, complete five-source
index, known-RHS rank, logs, identical descent, Pollard-rho improvement,
Shoup improvement, and breakthrough obligations remain false.

Disposition:

```text
ADMIT_ACTUAL_BUCKET_RESULTANT_ROUTING_AND_SOURCE_SEMANTICS_ONLY__REJECT_ALL_PAIR_QUOTIENT_STYLE_SYMBOLIC_AND_DENSE_TENSOR_GRAMMARS_AT_FROZEN_CAPS__PRESERVE_GAMMA9O4_RHO0_IMPLICIT_SINGLETON_C3_ROUTER__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Primary sources

- Moroz and Schost, *A Fast Algorithm for Computing the Truncated
  Resultant*, <https://arxiv.org/abs/1609.04259>.
- Bhargava, Ghosh, Guo, Kumar, and Umans, *Fast Multivariate Multipoint
  Evaluation Over All Finite Fields*,
  <https://arxiv.org/abs/2205.00342>.

## Exactly one next action

Construct or refute the isolated extreme survivor: a
`B^(9/4+o(1))`-state C3 singleton-bucket index with an implicit `O(1)`-pair
arbitrary-target router that returns matching C2 and C3 sources without
field DLP. Shared transposed work is allowed, but every routing cell, empty
certificate, source adjoint, rank, log, identical-descent, memory,
field-operation, and bit cost must be explicit.
