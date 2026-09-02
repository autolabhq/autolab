# P1553 torus C5 prime-order homomorphic-fingerprint gate R125

## Claim boundary

R125 closes one product-preserving descendant of the R124 coupled nonlinear
interface: single or finite tuples of group-homomorphic fingerprints from
the prime-order pairing image.

It does not cover nonhomomorphic hashes with separately proved correction
data, adaptive probes, target-dependent maps, rational-Krylov circuits,
deck-specific perfect hashing, or general arithmetic circuits and data
structures. It supplies no source locator, rank, factor logs, identical
descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
PRIME_ORDER_TORUS_HOMOMORPHIC_FINGERPRINT_KERNEL_TRIVIAL_OR_INJECTIVE__TRIVIAL_MAP_FAILS_EMPTY_TARGETS__NONTRIVIAL_POWER_MAP_IS_DLP_FREE_PERMUTATION_WITH_IMAGE_Q_B5__FINITE_TUPLES_RETAIN_DICHOTOMY__COMPOSITE_CONTROLS_HAVE_PROPER_QUOTIENTS__NONHOMOMORPHIC_ADAPTIVE_FINGERPRINT_OPEN__NO_SOURCE_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Prime-order theorem

Let `G=mu_q` with prime `q`. A product-preserving fingerprint

```text
phi:G -> H
```

is a group homomorphism. Its kernel is a subgroup of `G`, so

```text
ker(phi) = G  or  ker(phi) = {1}.
```

The fingerprint is therefore trivial or injective. Every endomorphism of
`mu_q` is a power map

```text
z -> z^k.
```

This map is directly computable on pairing images and needs no field
discrete logarithm. When `k=0 mod q` it is constant. For every other `k`,
it is a permutation of all `q` elements.

For a finite tuple `(phi_1,...,phi_t)`, the combined kernel is the
intersection of the component kernels. The tuple is trivial when all
components are trivial and injective as soon as one component is
nontrivial.

## Exactness

The candidate regime has nonempty C5 support and arbitrary empty targets.
A trivial fingerprint maps every target to the occupied image and therefore
produces false positives on every empty target. An exact pure homomorphic
fingerprint must be injective, with image cardinality at least

```text
q = B^(5+o(1)).
```

A full image-indexed table exceeds the `B^(9/4+o(1))` setup cap. Keeping
only an injectively mapped C3 hash table preserves the standard
`B^(9/4+o(1))` setup but still requires a `B^(3/2+o(1))` C2 scan per
arbitrary target.

## Controls

Prime-order controls at `q=5,7,11,13` have both positive and empty targets.
They enumerate every power map:

- the sole trivial map has false positives and no false negatives;
- every nontrivial map is injective and preserves exact membership;
- no intermediate image cardinality occurs.

A tuple control at `q=11` verifies that any finite tuple has image size one
or eleven and is exact only in the injective case.

Composite-order controls at `q=6,8,12` have intermediate image sizes. They
confirm that proper quotient fingerprints exist when the group order has
nontrivial divisors, so the obstruction is specifically the prime-order
kernel lattice. Finite controls receive no asymptotic credit.

## Scope limits

The theorem applies only when the fingerprint itself preserves the product
law. A nonhomomorphic hash can compress the image, but then C2-C3
composition is no longer determined by the fingerprints alone. Any
candidate using such a hash must expose and charge correction data for
every product-law failure, prove deterministic exact empty semantics, and
recover five projective sources. R125 proves no lower bound for such
correction schemes, adaptive probes, or arbitrary nonlinear circuits.

## Admission

Twelve of twenty obligations pass. The prime-order homomorphism dichotomy
and scoped negative are admitted. The nonhomomorphic/adaptive
membership/source interface, known-RHS rank, logs, identical descent,
Pollard-rho improvement, Shoup improvement, and breakthrough obligations
remain false.

Disposition:

```text
ADMIT_PRIME_ORDER_HOMOMORPHIC_FINGERPRINT_TRIVIAL_OR_INJECTIVE_DICHOTOMY_ONLY__REJECT_TRIVIAL_MAP_FOR_EMPTY_TARGETS_AND_NONTRIVIAL_MAP_AT_Q_B5_IMAGE__PRESERVE_NONHOMOMORPHIC_ADAPTIVE_FINGERPRINT_WITH_CHARGED_CORRECTIONS__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Freeze one nonhomomorphic or adaptive target fingerprint for the coupled
pair `(u^(*2),u^(*3))`. Require explicit product-law correction data,
deterministic exact empty semantics, five projective source backpointers,
`B^(9/4+o(1))` total setup, polylogarithmic arbitrary-target work, no field
DLP, and complete rank, logs, identical descent, memory, field-operation,
and bit costs.
