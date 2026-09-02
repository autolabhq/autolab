# P1553 torus C5 Khatri-Rao Kruskal-rank amplification gate R135

## Claim boundary

R135 identifies the all-five-atoms-from-one-color C5 evaluation matrix
as a row-deduplicated fifth Khatri-Rao power of the corresponding atom
evaluation matrix. It proves the ordinary Kruskal-rank product inequality
over any field and uses it in two sharply separated ways:

- deterministically, atom Kruskal rank two recovers the structured
  one-through-six-mode obstruction from R134; and
- under the frozen uniform-random-deck model only, a sparse-root union
  bound gives atom Kruskal rank near `log2(q)`, which amplifies to a
  product-mode exclusion near `5*log2(q)`.

The random-model result does not transfer to the structured factor base.
R135 supplies no deterministic seventh-mode obstruction, source index,
known-RHS rank, factor logs, identical descent, Shoup improvement, or
ECDLP breakthrough.

Classification:

```text
C5_COLOR_EVALUATION_IS_FIFTH_KHATRI_RAO_POWER_OF_ATOM_MATRIX__ORDINARY_KRUSKAL_RANK_INEQUALITY_PROVED_OVER_ANY_FIELD__ATOM_KRANK_TWO_RECOVERS_DETERMINISTIC_SIX_MODE_OBSTRUCTION__UNIFORM_RANDOM_DECK_ATOM_KRANK_AMPLIFIES_TO_NEAR_FIVE_LOG2_Q_PRODUCT_MODES_WITH_OVERWHELMING_PROBABILITY__TWELVE_EXACT_FINITE_RANK_REPLAYS__NO_STRUCTURED_CREDIT_ABOVE_SIX__MULTI_PREDICATE_NONZERO_VALUE_LOW_SLP_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Pinned source and finite-field proof

The pinned primary source is Bhaskara, Charikar, and Vijayaraghavan,
*Uniqueness of Tensor Decompositions with Applications to Polynomial
Identifiability*, arXiv:1304.8087. Its Definition 3.8 and Lemma A.4
provide provenance for the Khatri-Rao product and Kruskal-rank
inequality. R135 does not import the paper's robust real singular-value
claim into a finite field.

Instead, let `A` and `B` have the same `R` indexed columns and ordinary
Kruskal ranks `k_A` and `k_B` over any field. Consider a dependence on at
most `k_A+k_B-1` columns of `A odot B`, and select a column with nonzero
coefficient. Partition the remaining columns into sets of sizes at most
`k_A-1` and `k_B-1`. Kruskal independence supplies dual functionals
that kill the respective sets without killing the selected column. The
tensor product of those functionals isolates its nonzero coefficient, a
contradiction. Therefore

```text
krank(A odot B) >= min(k_A+k_B-1, R).
```

Induction gives

```text
krank(A odot A odot A odot A odot A)
    >= min(5*(krank(A)-1)+1, R).
```

This proof is ordinary algebra over any field.

## C5 matrix identification

For color atoms `a` and exponent modes `e`, define `A[a,e]=a^e`. A row
of the fifth Khatri-Rao power indexed by `(a1,...,a5)` has entry

```text
(a1*a2*a3*a4*a5)^e.
```

This is exactly the represented-mode evaluation row for the associated
degree-five source product. Source permutations and product collisions
only duplicate rows; deleting duplicates does not change column
dependencies. The remaining rows form the all-five-atoms-from-this-color
submatrix of the full C5 color acceptance matrix. Failure to vanish on
this submatrix is sufficient to obstruct vanishing on the full support.

Every asymptotic balanced color contains two distinct atoms. Its
all-mode atom matrix has Kruskal rank at least two, so the product
submatrix has Kruskal rank at least six. This is exactly the
deterministic R134 boundary; it gives no credit at seven modes.

## Random-deck amplification

Fix `0<epsilon<1` and set

```text
T = 1 + floor((1-epsilon)*log2(q)).
```

The Kelley-adapted sparse-root bound gives root density at most

```text
rho_epsilon = 2^(-epsilon/(1-epsilon))
```

for every nonzero represented polynomial with at most `T` distinct
modes. With `p<6q`, the number of projectively normalized predicates
through `T` modes is below

```text
T*36^T*q^(3T).
```

If `m` is the smallest balanced color part, sampling without replacement
and a union bound give

```text
Pr[any color atom matrix has krank below T]
    <= 4*T*36^T*q^(3T)*rho_epsilon^m.
```

At `m=q^(3/20+o(1))`, epsilon may tend to zero slowly while
`epsilon*m` dominates `(log q)^2`. With overwhelming probability under
the frozen random-deck model, the fifth Khatri-Rao power then excludes
all represented predicates through

```text
(5-o(1))*log2(q)
```

modes. This is a model-bound negative theorem, not structured-factor-base
or candidate-algorithm credit.

## Exact controls

All eight pairing decks are replayed directly in `Fp2`. Across 30 active
finite colors, 12 have two atoms. For each available control:

- the seven-column atom matrix has Kruskal rank exactly two;
- the row-deduplicated fifth Khatri-Rao matrix has six rows and Kruskal
  rank exactly six;
- all six-column subsets are independent while all seven columns are
  dependent;
- 32 ordered source rows deduplicate to the six degree-five products;
  and
- all six sources replay in C5 and satisfy the color acceptance rule.

The 18 singleton finite colors have no two-atom witness. The small
controls also have numerical random union bounds above one. Neither gap
receives probability or asymptotic credit.

## Admission

Fourteen of twenty-two obligations pass. The finite-field Khatri-Rao
lemma, deterministic six-mode recovery, model-bound near-`5*log2(q)`
negative, and 12 exact rank controls are admitted. Structured
seven-plus-mode predicates, multi-predicate and nonzero-value DAGs,
low-SLP predicates, source indexing, rank, logs, identical descent,
Pollard-rho improvement, Shoup improvement, and breakthrough remain
false.

Disposition:

```text
ADMIT_FIELD_INDEPENDENT_KHATRI_RAO_KRUSKAL_RANK_LEMMA__ADMIT_DETERMINISTIC_SIX_MODE_RECOVERY__ADMIT_NEAR_FIVE_LOG2_Q_EXCLUSION_UNDER_UNIFORM_RANDOM_DECK_MODEL_ONLY__ADMIT_TWELVE_EXACT_FINITE_RANK_REPLAYS_WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_STRUCTURED_SEVEN_MODE_MULTI_PREDICATE_NONZERO_VALUE_AND_LOW_SLP_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Attack a selector grammar that sparse-support rank does not cover: a
multiple-small-predicate decision DAG, a nonzero-value
Frobenius-coordinate branch, or a high-expansion low-SLP predicate.
Freeze every branch and coefficient, replay exact positive and empty
paths and C2+C3 sources, fit `B^(9/4+o(1))` state and polylogarithmic
arbitrary-target work, avoid field DLP, and charge rank, logs, identical
descent, memory, field operations, and bits.
