# P1553 S6 iterated-norm support probe R75

## Classification

- Owner: existing P1553 R9-R10 non-CP trace-contraction frontier; no P1554.
- Evidence: exact symbolic coefficient support over four standardized
  prime-order fields.
- Status: `REJECT_EXPANDED_S6_ITERATED_NORM_MOD_SUFFIX_GRAMMAR_ONLY`.
- Labels: `exact-finite`, `representation-bound`, `scalar-blind-source`,
  `novelty-unverified`.
- Cryptanalytic result: no S6 trace contraction, relation campaign,
  factor-log solve, scalar-blind descent, Shoup-bound improvement, or ECDLP
  breakthrough.

R74 found that explicit S4 residual memoization creates `B^3` distinct
prefix states. R75 tests the natural algebraic escape: keep those residuals
factored, take an iterated norm over one unary deck, and reduce only modulo
the suffix support polynomial.

The first norm already fills a trivariate coefficient cube of side `4B+1`.
Its state is therefore `(4B+1)^3=Theta(B^3)`. For `B>4`, its Z-degree `4B`
is below the suffix modulus degree `B^2`, so the proposed early reduction
changes nothing before the over-cap object appears.

This rejects the expanded coefficient grammar only. A transposed
scalar-functional algorithm that never represents the cube remains open.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R74 residual decision-diagram report | `1558482f504bd5e05b112464ee7c6734dd30ee518789735ac7abc8c305a5f740` |
| R73 resultant-valuation report | `00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915` |

## First norm

For the third public deck `I_3`, define

```text
H_(I_3)(X_1,X_2,Z)
  = product_(a_3 in I_3) S4(X_1,X_2,x(a_3),Z).    (1)
```

Each S4 factor has degree four in each displayed variable. On every frozen
curve and deck value, its coefficient support is exactly

```text
{0,1,2,3,4}^3,
support size 5^3=125.                             (2)
```

R75 constructs (2) directly from the S3-by-S3 resultant formula. Numeric
specialization in `X_1,X_2` matches R72's independent univariate S4 compiler
on all four curves.

For every tested prefix, multiplication in the actual curve field gives

```text
supp(H_(I_3))={0,...,4B}^3,
|supp(H_(I_3))|=(4B+1)^3.                        (3)
```

There are no coefficient cancellations or missing interior monomials in any
of the 20 frozen instances.

## Exact replay

The scalar-blind R72 decks are reused on secp256k1, P-256, P-384, and P-521.
The prefix support counts are identical on all four curves:

| B | support |
|---:|---:|
| 2 | 729 |
| 3 | 2,197 |
| 4 | 4,913 |
| 6 | 15,625 |
| 8 | 35,937 |

Every value equals `(4B+1)^3`. The result is finite but directly measures
coefficients in each standardized prime field, not an auxiliary lift.

## Suffix reduction boundary

The suffix side has `B^2` pair occurrences, so its literal support polynomial
has degree at least the number of distinct endpoints and at most `O(B^2)`.
The frozen grammar proposes to reduce (1) modulo a degree-`B^2` suffix
polynomial.

At the first norm stage:

```text
deg_Z H_(I_3)=4B.                                 (4)
```

For `B>4`,

```text
4B < B^2.                                        (5)
```

Therefore polynomial remainder modulo the suffix polynomial leaves (1)
unchanged. The full coefficient body in (3) appears before reduction can
discard one Z term.

The expanded grammar consequently pays

```text
first-stage state = Theta(B^3),
first-stage represented work >= Theta(B^3),       (6)
```

above the `B^(9/4+o(1))` setup/state cap. Building (1) after a target arrives
also exceeds the `B^(5/4+o(1))` online/workspace cap.

## Scope

R75 closes only `s6_iterated_norm_mod_suffix_v1` with an expanded trivariate
coefficient dictionary. It does not lower-bound:

- black-box products of S4 factors;
- transposed remainder, trace, or scalar-functional evaluation;
- straight-line circuits with cancellation not visible in coefficient
  support; or
- support-adaptive algorithms that construct only a gcd or zero certificate.

R73's duplicate multiplicity, zero strata, dyadic source, and backpointer
controls are not reached by this failed first stage. They remain mandatory.

No relation density, independent rank, factor-base logarithms, linear algebra,
or identical target descent is supplied.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_s6_iterated_norm_support_probe_r75.py` | `0bbbccda66ae1df14fa61ec58f3caa116d6558ea1d00b7ae111e63b0fbf4290a` |
| `p1553_s6_iterated_norm_support_probe_report_r75.json` | `c5f41fbb7325fe3f4e85fe6084fbaed6ab2df9d9101bd031cd1c589a95a3ba8c` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_s6_iterated_norm_support_probe_r75.py` | `0ddcc7886cd14f9e81a31bda0f01359b1865113c8cf7329997f0dc9b30f30c64` |

Targeted replay:

```text
Ran 3 tests in 1.628s
OK
families=4
full_products=True
max_support=35937
lane_admitted=False
```

## Disposition

```text
REJECT_EXPANDED_S6_ITERATED_NORM_MOD_SUFFIX_GRAMMAR_ONLY__FOUR_STANDARD_CURVES__B2_3_4_6_8__EVERY_S4_FACTOR_FULL_5_CUBE__EVERY_FIRST_NORM_PRODUCT_FULL_4B_PLUS1_CUBED_SUPPORT__B3_STATE__SUFFIX_REDUCTION_INACTIVE_FOR_B_GT4__NO_CIRCUIT_LOWER_BOUND__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Leave expanded iterated norms. Freeze a transposed scalar-functional grammar
that computes only the suffix gcd or exact zero certificate from the factored
unary S4 norm, without representing its trivariate coefficient body. It must
pass R73's duplicate multiplicity, zero-signature, dyadic-child, and source
controls inside the direct caps. Reject it if transposition merely postpones
construction of the `(4B+1)^3` cube or a `B^3` evaluation vector.
