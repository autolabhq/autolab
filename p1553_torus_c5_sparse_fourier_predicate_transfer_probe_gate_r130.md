# P1553 torus C5 sparse-Fourier predicate transfer gate R130

## Claim boundary

R130 closes complex or characteristic-zero sparse-Fourier color predicates
at the frozen caps and rejects an unjustified transfer of that lower bound
to the actual finite pairing fields.

It does not close order-two finite-field Fourier predicates, non-Fourier
shared-predicate decision DAGs, high-degree low-SLP selectors, adaptive
cell probes, or general arithmetic circuits and data structures. R130
supplies no inside-cap source index, rank, factor logs, identical descent,
Shoup improvement, or ECDLP breakthrough.

Classification:

```text
TAO_SHARP_UNCERTAINTY_CLOSES_COMPLEX_SPARSE_FOURIER_PREDICATES_ONLY__ALL_FOUR_ACTUAL_PAIRING_FIELDS_HAVE_ORD_Q_CHARACTERISTIC_TWO_AND_FAIL_PRIMITIVE_CHEBOTAREV_CONDITION__GF2E10_Q11_FIVE_MODE_FIVE_ZERO_COUNTEREXAMPLE_EXACT__FINITE_FIELD_TRANSFER_UNSUPPLIED__ORDER2_FOURIER_OR_NONFOURIER_SHARED_PREDICATE_DAG_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Complex theorem

For prime `q`, Tao's sharp uncertainty theorem over complex coefficients
states

```text
|supp(f)| + |supp(fhat)| >= q + 1.
```

Consequently, a nonzero complex cyclic Fourier polynomial with `s` modes
has at most `s-1` zeros. A color predicate vanishing on
`Theta(B^(15/4))` accepted C5 targets therefore needs
`B^(15/4+o(1))` complex modes. An exact Boolean indicator supported on at
most `B^(15/4+o(1))` points of a `q=B^(5+o(1))` group has
`B^(5+o(1))` complex Fourier support.

Both represented complex grammars miss the `B^(9/4+o(1))` setup cap.
Neither receives candidate finite-field work credit.

## Transfer boundary

The sharp support inequality is equivalent to a nonvanishing-minors
property for the prime cyclic Fourier matrix. In finite characteristic,
that property is not automatic.

The pinned Emmrich-Kunis finite-field Chebotarev theorem gives a sufficient
condition in the primitive case

```text
ord_q(characteristic) = q - 1.
```

Every actual R82 pairing family instead satisfies

```text
characteristic = 6q - 1,
characteristic = -1 mod q,
ord_q(characteristic) = 2.
```

The order-`q` roots therefore live in the quadratic extension, but the
primitive sufficient condition does not apply. R130 does not infer that
the actual order-two Fourier matrices have vanishing minors; it records
that an all-minors theorem or direct predicate analysis is still required.

## Exact counterexample

R130 implements `GF(2^10)` with irreducible modulus

```text
x^10 + x^3 + 1
```

and an element `omega` of exact order 11. The five Fourier modes

```text
exponents:    0, 1, 2, 4, 7
coefficients: 844, 964, 671, 534, 1
```

vanish exactly at subgroup exponents

```text
0, 1, 2, 3, 7.
```

The corresponding `5 x 5` Fourier minor has rank 4. Thus a five-mode
finite-field polynomial has five distinct subgroup zeros, whereas the
complex sharp consequence would permit at most four. This is an exact
transfer counterexample, not an attack algorithm and not evidence that the
same minor vanishes in the actual pairing fields.

No candidate discrete logarithm is used. Verifier exponents only enumerate
the synthetic order-11 subgroup.

## Scope limits

The complex uncertainty theorem remains valid and closes complex sparse
Fourier representations. The counterexample proves that coefficient-field
transfer needs justification. It does not provide a lower bound or a
construction for the actual `ord=2` fields. Non-Fourier circuits and
finite-field low-SLP selectors remain entirely outside this gate.

## Admission

Twelve of twenty obligations pass. The complex sparse-Fourier negative and
the finite-field transfer rejection are admitted. The actual order-two
minor theorem, inside-cap shared-predicate DAG, known-RHS rank, logs,
identical descent, Pollard-rho improvement, Shoup improvement, and
breakthrough obligations remain false.

Disposition:

```text
ADMIT_COMPLEX_FOURIER_UNCERTAINTY_SCOPE_AND_EXACT_FINITE_FIELD_TRANSFER_COUNTEREXAMPLE_ONLY__REJECT_UNJUSTIFIED_COMPLEX_TO_FINITE_LOWER_BOUND__PRESERVE_ORDER2_FINITE_FIELD_FOURIER_AND_NONFOURIER_SHARED_PREDICATE_DAG__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Primary sources

- Terence Tao, *An uncertainty principle for cyclic groups of prime
  order*, <https://arxiv.org/abs/math/0308286>.
- Tarek Emmrich and Stefan Kunis, *Real and finite field versions of
  Chebotarev's theorem*, <https://arxiv.org/abs/2506.02947>.

## Exactly one next action

Construct a finite-field shared-predicate selector DAG directly, or prove a
field-specific uncertainty/minor theorem for the actual
`ord_q(characteristic)=2` pairing families. It must avoid importing
complex Chebotarev bounds without transfer, choose a valid C2 branch in
polylogarithmic arbitrary-target work, return exact C2+C3 sources or an
empty certificate, fit `B^(9/4+o(1))` state, avoid field DLP, and include
rank, logs, identical descent, memory, field-operation, and bit costs.
