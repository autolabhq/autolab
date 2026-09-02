# P1553 5A5C Modular Frobenius Trace Recurrence Gate R96

## Claim boundary

No generic-prime-field ECDLP algorithm, Pollard-rho improvement, Shoup-bound
improvement, relation-rank result, factor-log solve, or target descent is
claimed.

Classification:

```text
SPLIT_QUOTIENT_PROJECTOR_TRACE_EXACT__REDUCED_FROBENIUS_IDENTITY__NONREDUCED_FROBENIUS_LOSES_NILPOTENT_SOURCE_STATE__STANDARD_QUOTIENT_B12O5
```

## Exact quotient trace

For a finite source-coordinate multiset with root polynomial `F`, let

```text
A = F_p[x]/F(x)
```

and let `M_h` be multiplication by the query value `h`. Then

```text
Tr_A(1-M_h^(p-1))
```

is the Fermat-projector zero count modulo `p`, weighted by the local algebra
dimensions. If the quotient dimension is below `p`, this is also the exact
integer count.

R96 verifies this identity by exact quotient arithmetic and multiplication
matrices over `F_11`.

## Reduced versus nonreduced Frobenius

For distinct roots

```text
1, 2, 4, 7,
```

and `h=(x-1)(x-4)`, the degree-four split quotient returns trace two. Because
every root is in `F_11`, the Frobenius map on the reduced quotient is exactly
the `4x4` identity. It supplies no coordinate or state contraction.

Adding a second occurrence of root one gives the nonreduced degree-five
quotient

```text
(x-1)^2(x-2)(x-4)(x-7).
```

The projector trace is three, correctly counting both copies. Frobenius now
has rank four rather than five: it kills the nilpotent occurrence direction.
Radicalization loses one zero occurrence, while retaining the nonreduced
algebra preserves a five-dimensional source body.

A separate blind query `h=x` returns trace zero. Exact quotient traces on
dyadic source ranges recover one zero occurrence; the summed queried
dimensions remain fewer than three root bodies, but the root quotient body
itself remains mandatory in this grammar.

## Charged state

The smaller R84 root side has `B^(12/5)` source occurrences. A
source-complete standard split quotient therefore has

```text
B^(12/5)
```

basis coordinates, outside both the `B^(9/4)` setup cap and `B^(5/4)` fresh
cap. An explicit multiplication or Frobenius matrix has

```text
B^(24/5)
```

entries. A dyadic subproduct/range tree retains `B^(12/5)` coefficient state
up to logarithmic factors.

Reduced Frobenius does not compress this state. Nonreduced Frobenius loses
the multiplicity direction required for source-complete occurrence replay.

## Scope

R96 closes only the standard explicit split-quotient basis, multiplication
and Frobenius matrices, and dyadic range quotients. It does not lower-bound a
factored transposed trace, modular character sum, or another succinct
source-reporting circuit that never materializes the quotient basis.

Actual five-A plus five-C integer lifting and coupled source unranking,
infinity, proper-subsum, tangent, and projective multiplicity remain
incomplete. Known-RHS relation rank, factor logs, and identical fresh-target
descent are absent.

Thirteen of twenty-seven obligations pass.

Disposition:

```text
REJECT_STANDARD_SPLIT_QUOTIENT_FROBENIUS_TRACE_ONLY__PROJECTOR_TRACE_AND_DYADIC_TOY_SOURCE_EXACT__REDUCED_FROBENIUS_IDENTITY_NO_COMPRESSION__RADICAL_LOSES_DUPLICATE__NONREDUCED_FROBENIUS_LOSES_NILPOTENT_SOURCE_STATE__QUOTIENT_B12O5_AND_MATRIX_B24O5__FACTORED_TRANSPOSED_TRACE_OPEN__PROJECTIVE_AND_FULL_5A5C_SOURCE_INCOMPLETE__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one factored transposed projector-trace functional over
the compact A/C divisor circuits. It must compute the integer count and one
nonzero dyadic child without materializing a `B^(12/5)` quotient basis, root
polynomial, moment vector, or endpoint/source table; fit `B^(9/4)` setup and
`B^(5/4)` fresh work/workspace; and replay blind-zero, duplicate, nilpotent,
infinity, proper-subsum, tangent, and multiplicity branches without DLP labels
or verifier oracles.
