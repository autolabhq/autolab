# P1553 tensorized small-root lattice gate R7

## Classification

- Owner: existing P1553/P1536/IDEA-049 tensor and bounded-root frontier; no
  P1554.
- Evidence: coordinator theorem screen pending independent red team; no run.
- Status: `REVISE_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, rank theorem, factor-log solve,
  scalar-blind descent, Shoup-bound improvement, or ECDLP breakthrough.

R6 constructed a cheap modular predicate on five bounded occurrence labels,
but a standard monomial-coefficient Coppersmith lattice begins in an ambient
space of size `Theta(B^5)`. R7 screens the surviving exception: keep the
predicate in a separated coefficient-tensor representation and perform norm
certification and lattice reduction without materializing that ambient space.

The correction is substantive. A `B^5` ambient coefficient box is not by
itself a runtime lower bound. Exact Gram computations can exploit a supplied
low-rank tensor. The candidate still fails in its natural integer lift because
the modular separation does not supply the low-norm carry corrections or the
exact decision-and-witness solver required by the frozen interface.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R6 coordinate-small-root gate | `03e9234bb0e17d7cf97e85a46735748a61fefabb300278c52fef11b41755be05` |
| P1553 R6 parent report | `c4202511d97248dcf9284c63680ae9cc2cc1b75df8a99fc71422b0c0f03b4a7d` |
| P1536 Frobenius-projector/norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |
| P1553 finite-deck reporter specification | `fd12ff17055a108ef31e58b2fb813feb1b8dc8eb2950db127a7a623e69a4d77f` |
| P1553 tensor red-team report | `0a48143e4b5a25fad52200abf7d43f7094ef18af6df0348ea9aa987ecd15002d` |
| IDEA-049 hypothesis | `1a58addced8c2bb14590eb9a74252707409d01e7acb6fbab206dc12af974b1e0` |
| IDEA-049 integral-lift derivation | `cbbe0f83426067fefde0a1f440afafad3d856ad49540469219df78da5a716b4d` |

## Frozen interface

Let `N=p^(1+o(1))`, `B=N^(1/5)`, and let each restricted deck contain
nonidentity x-classes `{P,-P}` with local labels `0<=z_i<s_i<=B`. R6 supplies
degree-`<s_i` maps
`X_i(z_i)` and the modular predicate

```text
F_R(z_1,...,z_5)
  = S_6(X_1(z_1),...,X_5(z_5),x(R)) mod p.
```

An admitted query must return one bounded root or certify that none exists,
recover one of its constant-many sign branches, and return one verified signed
x-class-labelled source. The total caps remain

```text
target-independent setup and state    B^(9/4+o(1)),
fresh-target time and workspace        B^(5/4+o(1)).
```

## Modular tensor positive control

Write the fixed summation polynomial as

```text
S_6(x_1,...,x_5,x_R)
  = sum_(rho=1)^r c_rho(x_R) product_i x_i^(e_(rho,i)),
```

where `r` and all exponents are constants independent of `B`. After reducing
each univariate factor modulo the public domain polynomial

```text
Q_i(z)=product_(a=0)^(s_i-1) (z-a),
```

the coefficient tensor of `F_R` over `F_p` has the explicit CP form

```text
C_R = sum_(rho=1)^r c_rho(x(R))
        tensor_i coeff(X_i(z_i)^(e_(rho,i)) mod Q_i).
```

It therefore has CP rank at most the constant `r` and `B^(1+o(1))`
factor storage. This is stronger than merely having a short evaluation
circuit. It is also consistent with P1536: constant separation rank of a
fixed polynomial does not locate its zeros.

Choose separated integer lifts with

```text
tilde(F_R)=sum_(rho=1)^r tilde(c_rho)(x_R)
             product_i tilde(u_(rho,i))(z_i),
tilde(F_R) mod p = F_R.
```

For a Coppersmith modulus power `m`, the literal-power shifts

```text
g_(k,alpha)=p^(m-k) z^alpha tilde(F_R)^k
```

are divisible by `p^m` at every modular root. Their explicit CP-rank bound is

```text
rank_CP(g_(k,alpha)) <= binomial(r+k-1,k).
```

Monomial shifts preserve that rank. Reduction modulo the integer domain
polynomials `Q_i` can be done factor by factor over `Z`; it preserves values on
legal labels and the separated form.

For supplied CP generators `g_j=sum_a tensor_i u_(j,a,i)`, the bound-scaled
Gram entry factors exactly as

```text
G_(j,l)=sum_(a,b) product_i
          sum_d u_(j,a,i,d) u_(l,b,i,d) B_i^(2d).
```

Thus exact norms and a Gram matrix for `t` explicitly supplied generators can
be computed from one-dimensional factors, without writing `B^5` coefficients.
Gram-form LLL returns an integer combination matrix `U`; keeping
`h_j=sum_l U_(j,l)g_l` in that form bounds every output CP rank by the total
input rank

```text
A=sum_l rank_CP(g_l).
```

For the naive pairwise correlation implementation, the reconstructed costs are
`B^(2+o(1)) A^2` target-independent bit work and `B^(1+o(1)) A^2`
fresh-target Gram assembly. Both frozen caps therefore require
`A<=B^(1/8+o(1))`. Shared correlations or another exact Gram algorithm could
change this representation-scoped ceiling. This is a valid exception to R6's
full dense embedding gate, not yet a root algorithm.

## Integer-lift and carry gate

Coppersmith needs a normed integer lattice, not only a tensor over `F_p`.
The uncentered separated integer lift above is valid, so a centered low-rank
carry tensor is sufficient but not formally necessary. For the `k`th power,
entrywise centered reduction has the form

```text
ctr_(p^k)(tilde(F_R)^k)=tilde(F_R)^k-p^k K_k.
```

Reduction only modulo `p` can destroy the required `p^k` divisibility. Modular
CP rank bounds neither the integer CP/TT rank of `K_k` nor the rank of the
centered tensor.

The distinction is load bearing. For a generic full-degree deck map, a
nonzero extreme coefficient of the separated lift is attached after root-bound
scaling to a monomial of degree `Omega(B)`. Its absolute contribution is at
least

```text
B^(Omega(B)),
```

whereas `p=B^(5+o(1))`. The natural vector `p^(m-1) F_R` therefore does not
become Howgrave-Graham-short merely by increasing `m`: the common
`p^(m-1)` factor also appears in its norm. Powers and positive monomial shifts
increase the weighted degree.

Shortness therefore requires cancellations among shifts. Standard monomial
`p^k`-corrections provide them entry by entry in

```text
Z[z_1,...,z_5]/(Q_1,...,Q_5),
```

whose rank is `product_i s_i=Theta(B^5)`. A tensorized route must instead
construct a cap-sized structured shift module containing enough of these
cancellations. A low-rank `K_k` would suffice, but is not the only possible
mechanism. No such module, constructor, determinant inequality, or worst-case
rank bound is supplied. This is the first conventional over-budget object,
not a lower bound on every carry tensor or implicit lattice algorithm.

## Lattice-closure gate

Integer linear combinations add represented ranks and products multiply them.
LLL does not preserve constant rank, but retaining its outputs as combinations
of the original generators gives the explicit bound `A` above. The implicit
Gram construction rigorously certifies the norms of the combinations it
actually constructs. It still does not prove that:

- enough short vectors lie in a subspace generated by a cap-sized structured
  shift family;
- the required generator family has `A<=B^(1/8+o(1))` and bounded bit cost;
- at least five useful integer polynomials are algebraically independent; or
- their common bounded roots can be enumerated from the separated form.

Ryan's EUROCRYPT 2025 shift-selection algorithms optimize explicit
multivariate Coppersmith shift spaces and can produce large-rank lattices. The
smaller graph-selected alternatives and the final multivariate root recovery
remain heuristic. The artifact accepts expanded polynomial relations and does
not provide a circuit- or tensor-native exact decision-and-witness theorem for
this predicate.

Consequently the tensor representation removes the first *materialization*
objection and supplies exact implicit Gram-form LLL for a cap-sized input
family. It reaches a more precise missing object: a target-symbolic integer
shift module that preserves `p^m` divisibility, has total separated rank within
the caps, contains enough cancellations for five useful short outputs, and is
sufficient for exact source-relevant recovery.

## Source and campaign gate

A short polynomial is not yet a source. A passing construction must produce
enough independent integer equations, return one common root or an exact
no-root certificate in the current five-label box, recover and verify one R6
sign branch, and bound no-relation behavior. Repeating this under `O(log B)`
sign-closed x-class restrictions must remain within the same fresh-target cap.

The conventional exact recovery objects remain over budget: a full
Groebner/resultant quotient on the label domains has `B^5` states, while a
balanced finite-grid locator first materializes a source-faithful third-label
table or norm with `B^3` entries. Extraneous common roots make bisection unsound
unless the solver either exhausts them or gives an exact source-relevant
existence certificate. These are scoped costs of the named recovery routes,
not a general root-finding lower bound.

No tensor-lattice object currently supplies the online relation constructor,
constant verified relation density, `Theta(B)` independent rows, factor-log
completion, identical scalar-blind descent, or bit complexity. The favorable
conditional ledger therefore remains only

```text
pair-index setup               B^2,
B known-log relation targets   B^(9/4),
sparse factor-log algebra      B^2,
one masked descent             B^(5/4),
lambda                         0.45,
mu                             0.40.
```

Falling back to the standard `B^3` source recovery instead gives `B^4`
relation collection and `B^3` descent, so it does not beat rho.

## Deduplication

- IDEA-049 owns bounded integral transducers and small-root source extraction.
- P1536 owns the constant-rank `S_6` control and the distinction between a
  short predicate and a zero locator.
- P1553 owns the exact target-labelled existence bit, signed restrictions,
  and source replay.
- The completed P1553 tensor audit owns CP/TT pre-mask compression and the
  failure of low-rank syntax alone to aggregate exact support.

R7 refines the integer-lattice interface inside these owners. It receives no
new idea ID, P1554, experiment, fixture, or breakthrough claim.

## Disposition

```text
REVISE_SCOPED_REDUCTION__FIVE_LABEL_S6_COEFFICIENT_TENSOR_HAS_CONSTANT_MODULAR_CP_RANK__LITERAL_POWER_SHIFT_RANK_BINOMIAL_R_PLUS_K_MINUS_1_K__DOMAIN_REDUCTION_PRESERVES_SEPARATED_FACTORS_AND_VALUES__EXACT_IMPLICIT_GRAM_LLL_AVOIDS_B5_MATERIALIZATION__OUTPUT_COMBINATION_RANK_BOUNDED_BY_TOTAL_INPUT_RANK_A__NAIVE_GRAM_CAP_REQUIRES_A_LE_B1_OVER_8__MODULAR_RANK_DOES_NOT_BOUND_CENTERED_INTEGER_P_K_CARRY_RANK__NATURAL_SEPARATED_LIFT_HAS_B_TO_OMEGA_B_ROOT_SCALED_NORM_ON_FULL_DEGREE_DECKS__STANDARD_ENTRYWISE_P_K_SHIFT_MODULE_HAS_B5_RANK__NO_CAP_SIZED_STRUCTURED_CANCELLATION_MODULE__NO_FIVE_USEFUL_SHORT_POLYNOMIALS_OR_EXACT_DECISION_WITNESS_THEOREM__STANDARD_ROOT_RECOVERY_B5_OR_B3__RYAN2025_EXPLICIT_AND_HEURISTIC_AT_MULTIVARIATE_RECOVERY__IDEA049_P1536_P1553_MERGE__RANK_LOGS_DESCENT_UNSUPPLIED__NO_P1554__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Prove or refute one target-symbolic quotient-lattice module whose literal-power
shifts preserve `p^m` divisibility, have total separated rank
`A<=B^(1/8+o(1))` under the naive exact Gram route, yield five useful
Howgrave-Graham-short outputs, and support exact subset-stable
decision-and-source recovery within `B^(5/4+o(1))`; if its needed
cancellations require the full `B^5` quotient module, close this tensor-lattice
branch and preserve Query2P1.

## Primary controls

- Semaev, *Summation polynomials and the discrete logarithm problem*,
  <https://eprint.iacr.org/2004/031.pdf>.
- Howgrave-Graham, *Finding small roots of univariate modular equations
  revisited*, <https://doi.org/10.1007/BFb0054862>.
- Coron, *Finding Small Roots of Bivariate Integer Polynomial Equations: a
  Direct Approach*,
  <https://www.iacr.org/archive/crypto2007/46220372/46220372.pdf>.
- Ryan, *Solving Multivariate Coppersmith Problems with Known Moduli*,
  <https://doi.org/10.1007/978-3-031-91095-1_13> and
  <https://artifacts.iacr.org/eurocrypt/2025/a13/>.
