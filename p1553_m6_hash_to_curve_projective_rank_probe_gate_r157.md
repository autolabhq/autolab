# P1553 M6 hash-to-curve projective rank gate R157

## Claim boundary

R157 transfers the R156 singleton-row experiment from verifier-known scalar
labels to public prime-order curve points. Relations are discovered by curve
point equality, rows contain only public integer multiplicities, and recovered
factor logs are verified by scalar multiplication.

Classification:

```text
L1_SINGLETON_FIBER_THEOREM__PUBLIC_GROUP_POINT_EQUALITY_RELATION_DISCOVERY__HASH_TO_CURVE_FINITE_RANK_AND_FACTOR_LOG_TRANSFER__NO_DLP_OR_LABEL_ORACLE__ASYMPTOTIC_INJECTIVITY_RANK_REVERSE_FFE_AND_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH
```

## Singleton theorem

Let the inversion-closed C deck be

```text
C = {+C_1,-C_1,...,+C_d,-C_d}.
```

An unordered six-source has signed coefficient vector
`v=(v_1,...,v_d)`. Its minimum number of atoms is `||v||_1`. Every other
source with the same vector is obtained by adding cancellation pairs
`(+C_j,-C_j)`. If

```text
k = (6-||v||_1)/2,
```

the exact number of unordered sources with vector `v` is

```text
binomial(k+d-1,d-1).
```

Consequently, if the map

```text
v -> sum_j v_j C_j
```

is injective on feasible signed C6 vectors, a C6 endpoint has exactly one
unordered source if and only if `||v||_1=6`.

All 24 curve controls satisfy the injectivity hypothesis, the fiber formula,
and the singleton criterion exactly.

## Public relation rows

For singleton source `v`, signed target `sigma*C_j`, and known A6 shift `S`,
the public point equality

```text
sum_i v_i C_i = S + sigma*C_j
```

gives the signed row

```text
r = v - sigma*e_j
```

and relation

```text
sum_i r_i log_G(C_i) = log_G(S) mod q.
```

The row is formed without C scalar labels. The A deck consists of
deterministic known-log inversion pairs, so every A6 right-hand side is known
by construction. R157 does not invoke BSGS, a DLP oracle, or any root, count,
marginal, rank, or source oracle.

All public group relation identities, opposite-row identities, and
projectively deduplicated right-hand sides replay exactly.

## Finite controls

The frozen grid contains four prime-order curve families, two independent
hash-to-curve offsets, and logarithmic factors `2,4,8`, for 24 controls.
The A-pair count is selected before relation discovery as the smallest count
whose maximum support occupancy reaches

```text
ceil(log_factor * ln(C_pair_count)).
```

Full-rank counts by logarithmic factor are:

```text
factor 2: 7 of 8
factor 4: 8 of 8
factor 8: 8 of 8
```

Twenty-three controls reach full signed C-log rank. All 23 recover every C
factor log without a DLP oracle and verify the result by public scalar
multiplication.

The only failure has dimension three, covers all three columns, and has two
projectively distinct rows of rank two. Its deficit is row supply, not
residual algebraic dependency. No control has residual nullity after
projective row count is sufficient.

## Cost boundary

The finite verifier explicitly enumerates every unordered C6 endpoint. With
`|C|=B^(3/4+o(1))`, this costs

```text
B^(9/2+o(1)),
```

which exceeds both the `B^(9/4)` setup cap and the `B^(5/2)` Pollard-rho
proxy. The finite factor-log successes therefore receive no attack credit.

The intended conditional envelope remains:

```text
A6 shift support:         B^(1/2+o(1))
signed C-log dimension:   B^(3/4+o(1))
relations:                B^(3/4+o(1)) log B
reverse row batch:        B^(5/4+o(1)) log B
conditional solve:        B^(2+o(1))
```

Crossing from the explicit `B^(9/2)` control to this envelope still requires
a reverse signed FFE operator. No such operator is supplied.

## Admission

Fifteen of twenty-four obligations pass. The cancellation-pair fiber theorem,
conditional singleton criterion, public group relation construction, 24
finite controls, 23 full-rank factor-log recoveries, public verification, and
full explicit cost charge are admitted.

No asymptotic hash-to-curve coefficient-map injectivity theorem, short
relation rank theorem, reverse signed FFE operator, identical target descent,
generic-prime algorithm, rho improvement, or Shoup improvement is admitted.

Disposition:

```text
ADMIT_L1_SINGLETON_THEOREM_AND_FINITE_PUBLIC_GROUP_FACTOR_LOG_TRANSFER__CHARGE_EXPLICIT_C6_ENUMERATION_B9O2__REQUIRE_ASYMPTOTIC_INJECTIVITY_AND_RANK_THEOREM_REVERSE_SIGNED_FFE_AND_IDENTICAL_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Prove a generic-prime high-probability injectivity and full-rank theorem for
the public short-relation coefficient system. Then instantiate the reverse
signed FFE operator within `B^(9/4)` setup and `B^(5/4+o(1))` batch work and
perform identical target descent.
