# P1553 torus C5 Fourier/product-resultant gate R123

## Claim boundary

R123 verifies exact multiplicative Fourier and ordered product-resultant
semantics for the R122 nonoccurrence torus interface. It closes only full
mode, linear-recurrence, represented resultant, symbolic product-polynomial,
and full-grid grammars.

It supplies no inside-cap source index, known-RHS rank, factor logs,
identical descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
MULTIPLICATIVE_FOURIER_CHARACTERS_Z_TO_ZJ_REQUIRE_NO_DLP__FULL_INVERSION_Q_MODES_B5__ORDERED_MOMENT_BM_ORDER_EQUALS_INJECTIVE_C5_SUPPORT_B15O4__ORDERED_P2P3_PRODUCT_RESULTANT_EXACT__P3_SETUP_B9O4_AND_TARGET_RESULTANT_QUERY_B9O4__SYMBOLIC_P5_B15O4__TARGET_SPECIALIZED_NONREPRESENTED_FOURIER_RESULTANT_TORUS_CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Multiplicative Fourier identity

For a deck of pairing images `z_i` in `mu_q`, the ordered five-product
count is

```text
c5(y) = q^(-1) sum_(j=0..q-1)
        (sum_i z_i^j)^5 y^(-j).
```

The characters `chi_j(z)=z^j` are computable by field exponentiation and do
not require discrete logarithms.

A full inverse transform uses

```text
q = B^(5+o(1))
```

modes. Sparse Prony/Berlekamp-Massey reconstruction does not rescue the
represented route: on all eight R82 controls, the ordered moment sequence
has linear complexity equal to the distinct canonical C5 support. Under the
R119 iid model this is

```text
B^(15/4+o(1)).
```

This is a statement about Fourier/linear-recurrence representations, not
all nonlinear target circuits.

## Product-resultant identity

Let

```text
P1(X)=product_i(X-z_i).
```

For ordered product multisets,

```text
P_(a+b)(Y)
  = Res_X(P_a(X), X^deg(P_b) P_b(Y/X)).
```

Thus `P5(y)=0` is exactly ordered five-product membership. A source still
requires gcd/root isolation and occurrence backpointers.

At `|C|=B^(3/4+o(1))`:

```text
deg P2 = B^(3/2+o(1)),
deg P3 = B^(9/4+o(1)),
deg P5 = B^(15/4+o(1)).
```

`P3` fits setup exactly. A represented target-scaled `P2|P3` fast resultant
still has optimistic `B^(9/4+o(1))` work per arbitrary target, while
symbolic `P5` exceeds setup. The all-field multipoint theorem is
near-linear in its represented coefficient and output bodies and therefore
does not erase these dimensions.

Moroz-Schost truncation computes `k` resultant coefficients softly linear
in `k` times the represented degree. At a fresh target, even the
constant-order specialization retains the `P3` degree body; it is not a
unit-cost resultant oracle.

## Controls

- Eight R82 controls verify that `(sum_i z_i^j)^5` equals the
  occurrence-weighted distinct-support moment through twice the support
  length.
- Their Berlekamp-Massey order equals the distinct product count and the
  recurrence equals the support annihilator.
- A synthetic `q=11` subgroup of `F_353^*` verifies all eleven Fourier
  counts, positive and empty membership, without candidate DLP labels.
- The same control verifies that the ordered `P2|P3` resultant equals direct
  `P5` evaluation and vanishes exactly on present targets.
- Finite controls receive no asymptotic credit.

## Admission

Twelve of nineteen obligations pass. The inside-cap target-specialized
membership and source circuit, rank, logs, identical descent, Shoup
improvement, and breakthrough obligations remain false.

Disposition:

```text
ADMIT_EXACT_MULTIPLICATIVE_FOURIER_ORDERED_MOMENT_AND_P2P3_PRODUCT_RESULTANT_SEMANTICS_ONLY__REJECT_FULL_Q_MODE_PRONY_BM_REPRESENTED_TARGET_RESULTANT_SYMBOLIC_P5_AND_GRID_ROUTES_AT_FROZEN_CAPS__PRESERVE_TARGET_SPECIALIZED_NONREPRESENTED_FOURIER_RESULTANT_TORUS_CIRCUIT__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Primary sources

- Bhargava, Ghosh, Guo, Kumar, and Umans, *Fast Multivariate Multipoint
  Evaluation Over All Finite Fields*, <https://arxiv.org/abs/2205.00342>.
- Moroz and Schost, *A Fast Algorithm for Computing the Truncated
  Resultant*, <https://arxiv.org/abs/1609.04259>.

## Exactly one next action

Construct or refute one target-specialized nonrepresented torus C5 circuit
outside `q` Fourier modes, full Prony/BM state, represented `P2|P3`
resultants, symbolic `P5`, and endpoint splits. Require an explicit circuit
or data structure, exact noncancellation and empty certificates, five
projective source backpointers, `B^(9/4+o(1))` state, polylogarithmic query,
no field DLP, and complete pairing-to-descent costs.
