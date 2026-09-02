# P1553 R107 Non-Character Algebraic Target-Norm Resultant Gate

## Claim boundary

This is a scoped negative for explicit coefficient, Sylvester/subresultant,
quotient-free cofactor, and sparse Macaulay resultant grammars. It is not a
general arithmetic-circuit lower bound and does not rule out a compact
factored elliptic lambda-ring, Chow-form, or rational recurrence.

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or breakthrough is claimed.

## Exact root-interface gate

Assign asymptotic source weights `2/5` to each A occurrence and `3/5` to
each C occurrence. Exhausting all 34 nontrivial root partitions of the
canonical `5A+5C` source gives a minimum explicit resultant interface

```text
B^(12/5) versus B^(13/5).
```

The four minimizers are `0A+4C`, `2A+3C`, `3A+2C`, and `5A+1C`, up to
complement. Thus every explicit binary root resultant, coefficient body,
subresultant vector, or quotient-free cofactor body has dimension at least
`B^(13/5)`, above both the `B^(9/4)` setup cap and `B^(5/4)` online cap.

For coefficient-form bivariate inputs, Moroz and Schost compute an order-`k`
truncated resultant in softly `O(kd)` operations for degree `d`. At constant
marker-jet order this still charges the frozen `B^(13/5)` degree body. This
is an upper-bound/body accounting result for the standard representation,
not an unconditional lower bound on a compact arithmetic circuit.

Sparse Macaulay formulas express the resultant as a quotient of
determinants. The explicit mixed source body has `B^5` factors, and neither
determinant receives unit-cost oracle credit.

Primary sources:

- https://arxiv.org/abs/1609.04259
- https://arxiv.org/abs/math/0107181

## Canonical-weight gate

An unrestricted `2A+3C` versus `3A+2C` resultant does not equal the
canonical source norm. Its target vanishing order is

```text
sum_(s:endpoint(s)=T) w_split(s),
```

where `w_split(s)` is the number of bounded two-submultisets of the five A
indices times the number of bounded three-submultisets of the five C
indices.

All 80 actual and matched query controls satisfy this identity exactly.
The split weight is globally nonconstant on every frozen deck. The two
actual double fibers have:

```text
canonical count 2, weights 15 and 12, resultant order 27
canonical count 2, weights  8 and 35, resultant order 43
```

Therefore no constant scalar normalization converts unrestricted resultant
order to canonical integer count. The ten R105 marker channels identify the
individual sources in these finite low-multiplicity controls, but generic
factorization and exact weight correction remain unsupplied.

## Multiplicity and exceptional gates

A duplicate-root `Z/11Z` control verifies the elementary resultant/convolution
jet identity. A collapsed-deck control has 2,646 canonical sources at one
target and resultant order 31,500, proving that the observed order-two
actual fibers do not supply a generic multiplicity bound.

Signed `Fp2` keys, inverse-pair identity, tangent doubling, blind targets,
and identity targets replay in the group verifier. No homogeneous
projective resultant chart is supplied to the candidate, so this receives
no constructor credit.

## Disposition

```text
BALANCED_NONCHARACTER_ROOT_RESULTANT_REQUIRES_B13O5_INTERFACE
UNRESTRICTED_RESULTANT_COUNTS_SOURCE_DEPENDENT_PARTITION_WEIGHTS
ACTUAL_DOUBLE_FIBERS_HAVE_NONUNIFORM_WEIGHTS
STANDARD_TRUNCATED_SUBRESULTANT_AND_MACAULAY_GRAMMARS_OVER_CAP
FACTORED_ELLIPTIC_LAMBDA_RING_CHOW_CIRCUIT_OPEN
```

The lane is not admitted: 15 of 27 obligations pass.

## Exactly one next action

Construct or refute one factored elliptic lambda-ring/Chow-form recurrence
for the canonical, not partition-weighted, `5A+5C` target norm. It must use
compact `D_A,D_C,T` without explicit side coefficients, subresultant
vectors, or Macaulay bodies; support all eleven marker channels,
homogeneous projective branches, generic multiplicity and integer lifting;
and fit both caps before rank, factor logs, and identical descent.
