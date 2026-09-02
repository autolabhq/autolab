# P1553 M6 random-diagonal known-target rank gate R159

Date: 2026-07-29

## Scope

R159 replaces the R158 A6/projective relation-event process with direct
known-scalar target queries. It proves zero factor-base coverage and full
relation rank for an iid cyclic-label model, transfers that theorem to the
corresponding conditioned ideal hash-to-curve sampler, and replays the
result on public curve points.

This is a theorem and finite implementation gate, not a complete ECDLP
algorithm. The required batched positive-C6 reverse FFE source locator is
not constructed.

## Positive C6 support

For `d` positive factor-base representatives, the unordered six-source
universe has

```text
M = binomial(d+5,6) = Theta(d^6).
```

For iid uniform labels in a prime cyclic group of order `q`, any two
distinct coefficient vectors collide with probability `1/q`. The expected
number of collision pairs is at most `M(M-1)/(2q)`. A non-singleton fiber
of size `f` contributes `f` bad sources and `binomial(f,2)` collision
pairs, with `f <= 2 binomial(f,2)`. Therefore

```text
Pr[unique endpoint count < M/2] <= 2M/q.
```

At `d=B^(3/4)` and `q=B^5`, this failure probability is
`O(B^(-1/2))`. Global injectivity remains false; only a constant-fraction
unique endpoint support is needed.

## Direct target coverage

For each factor-base column `j`, choose an independent uniform
`s_j in F_q`. Query the positive-C6 locator at

```text
tG + [s_j]C_j
```

for independent known uniform scalars `t`. Conditional on the C deck and
`s_j`, every query target is uniform. If at least `M/2` endpoints are
unique, then

```text
T = ceil(8q log(d)/M)
```

queries leave some column uncovered with probability at most

```text
d exp(-MT/(2q)) <= d^(-3).
```

The all-column target count is

```text
dT = B^(5/4+o(1)).
```

This proves zero uncovered columns with high probability in the iid model;
it does not merely prove a vanishing uncovered fraction.

## Random diagonal rank

Let `v_j` be the first unique positive-C6 source returned for column `j`.
The public point identity

```text
sum_i v_ji C_i = t_j G + [s_j]C_j
```

gives the factor-log row

```text
r_j = v_j - s_j e_j
```

with known right-hand side `t_j`.

Uniform `t_j` makes the success event and selected source independent of
`s_j`. Conditional on the selected source matrix `V`, the relation matrix
is

```text
R = V - diag(s_1,...,s_d).
```

Its determinant is a nonzero multilinear polynomial with leading monomial

```text
(-1)^d product_j s_j.
```

The finite-field polynomial zero bound gives

```text
Pr[rank(R) < d | coverage and V] <= d/q
                                  = B^(-17/4).
```

No sparse-random-matrix contiguity assumption, projective opposite-row
quotient, or rank oracle is used.

## Conditioned sampler

Conditioning iid group labels to be nonzero and pairwise distinct up to
sign gives the ideal rejection-sampled hash-to-curve factor-base law. The
conditioning event fails with probability `O(d^2/q)=O(B^(-7/2))`, so the
iid probability theorem transfers with only the reciprocal conditioning
probability factor.

This is an exact statement for the ideal random-oracle sampler.
Pseudorandomness of a fixed deterministic hash-to-curve instantiation is
not proved and receives no generic-prime transfer credit.

## Identical descent

Once factor logs are solved, query the same positive-C6 locator at

```text
Q + tG.
```

A unique source `v` gives

```text
log_G(Q) = sum_i v_i log_G(C_i) - t.
```

The query count is `Theta(B^(1/2)log B)`, and the source mechanism is
identical to relation collection. This closes the descent reduction
conditional on the same missing locator.

## Finite controls

The preregistered grid contains three public prime-order curve families,
two deterministic factor-base offsets, and two seeds, for 12 controls.

- All positive-C6 endpoint maps are injective on these finite controls.
- All factor-base columns are covered by public point equality.
- All 12 random-diagonal relation matrices have full rank.
- All 12 recovered factor-log sets verify by public scalar multiplication.
- All 12 identical positive-C6 target descents verify publicly.
- No candidate DLP, root, count, marginal, rank, or source oracle is used.

The finite implementation explicitly enumerates the positive-C6 endpoint
map. Its `B^(9/2+o(1))` cost exceeds both the `B^(9/4)` setup cap and the
`B^(5/2)` Pollard-rho proxy. The finite successes receive no attack or
asymptotic credit.

## Admission

Admit:

- the positive-C6 unique-support probability bound;
- zero all-column coverage for `B^(5/4)log B` known targets;
- the random-diagonal full-rank theorem;
- exact transfer to the conditioned ideal sampler;
- the identical positive-C6 descent reduction;
- the 12 finite public-curve correctness controls.

Do not admit:

- deterministic hash-to-curve pseudorandomness;
- a batched positive-C6 unique-source locator;
- a reverse-only signed marker or FFE operator;
- an unconditional setup or online cost;
- a generic-prime ECDLP algorithm;
- a Pollard-rho or Shoup improvement;
- an ECDLP breakthrough.

## Next action

Construct a public batched reverse FFE locator that returns unique
positive-C6 sources for `B^(5/4)log B` arbitrary targets using at most
`B^(9/4+o(1))` setup and `B^(5/4+o(1))` work. Then replay the admitted
random-diagonal factor-log solve and identical descent without explicit
endpoint enumeration.
