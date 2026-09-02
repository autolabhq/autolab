# P1553 torus C5 base-field Frobenius predicate-DAG gate R132

## Claim boundary

R132 closes univariate base-field polynomial and rational zero-test DAGs on
the actual order-two pairing controls. It does not close extension-field
lacunary predicates, Frobenius-aware coordinate predicates, tests that use
nonzero values, adaptive cell probes, or general arithmetic circuits and
data structures.

It supplies no inside-cap source index, rank, factor logs, identical
descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
ORDER2_FROBENIUS_EQUALS_INVERSION__UNIVARIATE_BASE_FIELD_POLYNOMIAL_RATIONAL_ZERO_DAGS_ARE_INVERSION_INVARIANT__ALL_EIGHT_ACTUAL_C5_SUPPORTS_HAVE_EVERY_POSITIVE_INVERSE_EMPTY_AND_REQUIRE_EXTENSION_COEFFICIENT_ANNIHILATORS__BASE_FIELD_DAG_REJECTED_ON_ACTUAL_CONTROLS__ASYMMETRIC_EXTENSION_OR_FROBENIUS_AWARE_DAG_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Frobenius theorem

In every actual pairing family,

```text
p = -1 mod q.
```

Therefore every order-`q` subgroup element satisfies

```text
z^p = z^(-1).
```

For a polynomial `f` with coefficients in `F_p`,

```text
f(z^(-1)) = f(z^p) = f(z)^p.
```

Thus `f(z)=0` if and only if `f(z^(-1))=0`. The same statement holds for
the numerator-zero and denominator-zero outcomes of a univariate
base-field rational predicate.

Any deterministic decision DAG that branches only on these Boolean
outcomes follows the same path on `z` and `z^(-1)`. This includes
univariate trace or Dickson zero-test grammars. It does not include
coordinate circuits that explicitly consume both `z` and `z^p`.

## Exact inverse witnesses

All eight actual pairing decks are replayed directly in `Fp2`, without
candidate discrete logarithms. Every C5 product is nonzero and distinct.
For every positive target `z`, the inverse `z^(-1)` is absent from the
entire C5 support. The same is true for every active color-acceptance set.

Consequently, a base-field zero-test DAG must give the same answer to a
positive target and an empty target. It cannot be an exact membership or
source locator on these controls.

The exact support and color root-annihilator polynomials all contain an
extension-field coefficient, as required by their non-Frobenius-stable root
sets. Five sample base-field polynomials per control replay the Frobenius
evaluation identity and inversion-invariant zero outcomes exactly.

These are finite controls and receive no asymptotic credit.

## Random comparator

For a uniformly random `M`-subset `S` of an odd prime-order group,

```text
E |S intersect S^(-1)| = M^2/q.
```

At the inherited scales `M=B^(15/4+o(1))` and `q=B^(5+o(1))`, the expected
overlap is `B^(5/2+o(1))`, only a `B^(-5/4+o(1))` fraction of `S`.

This is a model-bound comparator, not a theorem about the structured
asymptotic factor base and not candidate work evidence.

## Admission

Twelve of twenty obligations pass. The scoped base-field Frobenius-DAG
negative and actual inverse witnesses are admitted. An asymmetric
extension-field or Frobenius-aware coordinate selector, inside-cap source
index, known-RHS rank, logs, identical descent, Pollard-rho improvement,
Shoup improvement, and breakthrough remain false.

Disposition:

```text
ADMIT_ORDER2_FROBENIUS_INVERSION_THEOREM_AND_EXACT_ACTUAL_POSITIVE_EMPTY_INVERSE_WITNESSES_ONLY__REJECT_UNIVARIATE_BASE_FIELD_ZERO_TEST_DAGS_ON_ACTUAL_CONTROLS__PRESERVE_ASYMMETRIC_EXTENSION_AND_FROBENIUS_AWARE_COORDINATE_DAGS__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct an asymmetric lacunary predicate over `F_(p^2)`, or a
Frobenius-aware coordinate DAG using both `z` and `z^p`. It must
distinguish the observed positive/empty inverse pairs, choose a valid C2
branch in polylogarithmic arbitrary-target work, return exact C2+C3 sources
or an empty certificate, fit `B^(9/4+o(1))` state, avoid field DLP, and
include rank, logs, identical descent, memory, field-operation, and bit
costs.
