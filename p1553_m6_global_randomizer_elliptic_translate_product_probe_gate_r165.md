# P1553 M6 global-randomizer elliptic translate-product gate R165

Date: 2026-08-01

## Scope

R165 strengthens R164 by proving that one uniform scalar randomizer suffices
for the entire target batch. Every regular factor is consequently a translate
of one fixed elliptic function, exposing more shared structure to the
remaining constructor.

The finite producer evaluates all endpoint-target pairs. That work receives
no asymptotic attack credit.

## One global randomizer

Let `a_j(P),b_j(P)` be R161's signed membership residuals for a fixed regular
pair. For one uniform `r` in `F_p`, define

```text
c_j(P) = a_j(P) + r b_j(P).
```

A true match is always zero. Each fixed nonmatch cancels for at most one
value of `r`, so its probability is at most `1/p`. Independence between
targets is unnecessary for the union bound:

```text
Pr[any regular false root] <= nN/p = B^(-3/2).
```

Linearity of expectation also preserves R164's expected false-root
verification cost. Factoring the candidate and checking direct group
identities gives an exact Las Vegas output with expected `B^2` true-root
verification.

A forced global value `r=-a/b` creates at least one correlated false root in
the finite control. The verifier removes every false root and retains the
complete R163 union.

## One fixed elliptic function

Define on the curve

```text
f_r(Q) = U(x(Q)) + r(y(Q)-V(x(Q)))
       = (U-rV)(x(Q)) + r y(Q).
```

All signed C3 endpoints are zeros. If `n=deg(U)` and `n>=3`, then `U(x)` has
pole order `2n` at `O`, while `V(x)` has pole order at most `2n-2` and `y`
has pole order 3. The monic leading term cannot cancel, so `f_r` has exact
pole order `2n` and a zero divisor of degree `2n`.

For every affine translate `Q=T_j-P`,

```text
a_j(P)+r b_j(P) = f_r(T_j-P).
```

Thus the hard norm is the single-function translate product

```text
Phi_r(P) = product_j f_r(T_j-P),
G_r = gcd(U, Phi_r mod U).
```

At the quotient components with `P=T_j`, the translate is `O`, a pole rather
than an affine C3 member; the semantic product factor is regularized to one.
The `P=-T_j` x-incidence is an ordinary tangent evaluation of the global
translation morphism. The synthetic `P+(-2P)=-P` control proves that tangent
and opposite regular orientations both make `f_r` vanish, while the equality
control maps to the pole and contributes no membership root.

## Degree and cost boundary

For arbitrary `N` public targets, an explicit representation of the global
translate product has zero and pole divisor degrees

```text
2nN = B^(7/2).
```

An ordinary product tree, explicit divisor, or coefficient-ring norm therefore
remains one exponent above Pollard rho. This is representation accounting,
not an arithmetic-circuit or data-structure lower bound. The desired output
is only a low-degree factor of the remainder modulo `U`.

Miller's algorithm compresses functions whose divisors follow a
scalar-multiplication chain. The R165 target divisor is an arbitrary public
set, and no scalar-chain representation is supplied. Miller evaluation does
not receive unit-cost translate-product credit.

Primary source:

- Victor S. Miller, *Short Programs for Functions on Curves*, 1986,
  `references/miller_weil_pairing_algorithm_1986.pdf`.

## Deduplication

R164 admits independent target randomizers, exact verification, and an
incidence split. R165 proves target independence is unnecessary and
globalizes tangent evaluation.

R113 closes a full fixed-translation orbit and ordinary nonlinear product
trees. R165 instead has arbitrary public translations of one fixed function;
it does not supply a compressed product.

R148 closes the named standard static 3SUM-indexing tradeoffs but preserves a
coordinate-aware elliptic operator. R165 is a sharper coordinate interface,
not a published indexing improvement.

## Admission

Twenty-one of twenty-eight obligations pass. Admit the global-randomizer
bound, fixed-function identity, pole order, translate-product semantics,
tangent globalization, and exact verification.

Do not admit an output-sensitive translate-product constructor,
deterministic hash-to-curve transfer, unconditional generic-prime algorithm,
Pollard-rho or Shoup improvement, or ECDLP breakthrough.

Disposition:

```text
ADMIT_ONE_GLOBAL_RANDOMIZER__ADMIT_FIXED_ELLIPTIC_FUNCTION_TRANSLATE_PRODUCT__POLE_ORDER_2N__TANGENT_BRANCH_GLOBAL__P_EQUALS_T_POLE_REGULARIZED__FORCED_CORRELATED_FALSE_ROOTS_VERIFIED_AWAY__EXPLICIT_TRANSLATE_DIVISOR_B7O2__OUTPUT_SENSITIVE_REMAINDER_OPEN__NO_RHO__NO_SHOUP_IMPROVEMENT__NO_BREAKTHROUGH
```

## Next action

Construct or refute `gcd(U,product_j f_r(T_j-P))` below `B^(5/2)`, preferably
in `B^(9/4+o(1))`, without expanding the degree-`2nN` divisor or the `n` by
`N` value table. The constructor may return only the unlabeled remainder
factor; R165 and R163 supply exact verification, labels, and backpointers.
