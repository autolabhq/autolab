# P1553 quadratic-intermediate-field obstruction R21

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 common-right-factor lane; no new
  idea ID.
- Evidence: theorem-only structural exclusion; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: a common right factor that also compresses the frozen
  prime-subgroup fifth list can serve at most seven distinct nonzero endpoints.
  No growing balanced R16 block survives. This is not an unrestricted lower
  bound, Shoup-bound improvement, or ECDLP breakthrough.

R20 closed geometrically Galois common factors by their deck transformations.
R21 does not assume that the factor is Galois or indecomposable. It moves to
the elliptic function field and uses the complete branch trace and norm to
construct one quadratic intermediate field per endpoint. Genus, divisor
classes, and the bounded elementary-`2` automorphism groups of genus-zero and
genus-one curves then close every growing block that also passes finite-list
compression.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R16 common-right-factor compiler gate | `9f7ec4ecb6821e30affce39b25b459891f6be0b945b6d64332971dd86df2237c` |
| R18 repaired generic-pair gate | `d1eadf3edf74acedf30888aff764d221d1483ca3dc8521ff293cd7aa422ca5c4` |
| R19 Galois-orbit and target gate | `b5aef048b9748763f6449e845fa7554755e12715dce695cd99bdc8f0b77df61f` |
| R20 connected-Kummer deck gate | `1fae860dfc2a783e8ee494eeadae8efedec136916f057e91f317d8df616f5527` |
| R20 parent report | `2410c2e7a1e7b344915ad03e2b75fb93cdbcee3b259348d16564343d165d13a3` |
| R20 bundle hash list | `80faf0f516860a21b1c5037c796b0ae57926cfce921d097fb4e7e45ef5fd8be7` |

## Frozen regime

Work over an algebraically closed field of characteristic greater than `3`.
Let `E` contain an odd prime-order subgroup `G=<P>` of order `N`, and let

```text
x:E->P^1_q,
f=psi composed with x:E->P^1,       deg(f)=m=2d<N.
```

Let a block `H_z subset G\{0}` have a separable common right factor

```text
pi:P^1_q->P^1_r,                    deg(pi)=e>1,
F_A=G_A composed with pi            for every A in H_z.
```

Put

```text
h=pi composed with x:E->P^1_r,
L=k(E),
K=k(h),
s=[-1].
```

Then `[L:K]=2e`, `s` fixes `K`, and every degree is below the characteristic.
For an endpoint `A`, define

```text
y_A(Q)=f(A+Q),
s(y_A)(Q)=f(A-Q).
```

The complete trace-and-norm factorization says exactly

```text
y_A+s(y_A) in K,
y_A*s(y_A) in K.                                      (1)
```

## Theorem 1: every endpoint creates a genuine quadratic field

Let

```text
M_A=K(y_A) subset L.
```

Equation (1) gives `[M_A:K]<=2`. Equality cannot be one. If `y_A in K`, then
it is fixed by `s`, so

```text
f(A+Q)=f(A-Q)
```

identically. As in R20, invariance under this reflection and under `[-1]`
forces invariance under translation by `2A`, an automorphism of order `N` of
the degree-`m` separable extension `L/k(f)`. This contradicts `m<N`.

Therefore

```text
[M_A:K]=2,
[L:M_A]=e.                                             (2)
```

Let `Y_A` be the smooth curve with function field `M_A`. The finite map

```text
alpha_A:E->Y_A
```

has degree `e`. Riemann-Hurwitz gives `genus(Y_A)<=1`. The involution `s`
descends nontrivially to `Y_A`, its fixed field is `K`, and the diagram is

```text
E --------alpha_A-------> Y_A
|                         |
| h, degree 2e            | eta_A, degree 2
v                         v
P^1_r  ================  P^1_r.
```

Moreover `y_A=g_A composed with alpha_A`, where

```text
deg(g_A)=m/e.                                           (3)
```

Thus every endpoint supplies a crossed `(e,2)` intermediate decomposition,
whether or not `pi` itself is Galois.

## Theorem 2: a genus-one intermediate fails fifth-list compression

Suppose one `Y_A` has genus one. Riemann-Hurwitz makes `alpha_A` etale, hence
after choosing origins it is a translate of a separable degree-`e` isogeny.
The descended involution satisfies

```text
alpha_A(-Q)=s_bar(alpha_A(Q)),
```

and `eta_A` is the quotient by `s_bar`.

For `Q,Q' in G`, equality `h(Q)=h(Q')` therefore implies

```text
alpha_A(Q')=alpha_A(Q)  or  alpha_A(Q')=alpha_A(-Q).
```

The isogeny kernel has order `e<N` and intersects the prime-order subgroup
trivially. Hence `Q'=Q` or `Q'=-Q`. The map `h`, equivalently `pi` on the
Kummer line, is injective on every distinct Kummer support from `G`:

```text
J=|pi(I_5)|=|I_5|=L_5.                                 (4)
```

This fails the balanced R16 requirement `J=O(L_5/e)` for growing `e`.

## Theorem 3: a genus-zero quadratic field cannot repeat

Assume `Y_A` has genus zero. If two endpoints `A,B` had

```text
M_A=M_B=M,
```

write the common map as `alpha:E->P^1`, of degree `e`. By (3), both
`y_A` and `y_B` are degree-`m/e` rational functions of `alpha`. Let `D` be
the pole divisor of `f`, of degree `m`. Then

```text
(y_A)_infinity=t_(-A)(D),
(y_B)_infinity=t_(-B)(D).
```

Each is the pullback by `alpha` of a degree-`m/e` divisor on `P^1`. All
divisors of that degree on `P^1` are linearly equivalent, so

```text
t_(-A)(D) linearly_equivalent t_(-B)(D).                (5)
```

Under `Pic^m(E)->E`, translation of a degree-`m` divisor by `-A` subtracts
`mA` from its Abel sum. Equation (5) therefore gives

```text
m(A-B)=0 in E.
```

But `A-B` lies in the prime-order group and `0<m<N`, so `A=B`. Distinct
endpoints yield distinct genus-zero quadratic fields `M_A/K`.

## Theorem 4: at most seven genus-zero endpoint fields

Let `M` be the compositum inside `L` of all the quadratic fields `M_A` from
the block. Since every `M_A/K` is quadratic and separable,

```text
Gal(M/K)=(Z/2Z)^r
```

for some `r`. Let `Z` be the smooth curve with function field `M`. The map
`E->Z` is finite and separable, so `genus(Z)<=1`.

If `Z` has genus zero, `Gal(M/K)` is an elementary-`2` subgroup of `PGL_2`.
In odd characteristic such a subgroup has order at most `4`.

If `Z` has genus one, its translation subgroup contributes at most the four
points of `Z[2]`. The image in `Aut(Z,0)` is an elementary-`2` group of order
at most `2` in characteristic greater than `3`. Therefore

```text
[M:K]<=8.                                               (6)
```

An elementary abelian group of order at most `8` has at most seven index-two
subgroups, equivalently `M/K` has at most seven quadratic subfields. Combining
this with Theorems 2 and 3 gives the dichotomy

```text
J=L_5,  or  |H_z|<=7.                                   (7)
```

The first branch fails growing finite-list compression; the second fails the
balanced endpoint requirement `|H_z|=Theta(d)`.

## Consequence for the R16 compiler

No separable common right factor of any monodromy type can simultaneously
satisfy, for growing `d`,

```text
|H_z|=Theta(d),
e=Theta(d),
|pi(I_5)|=O(L_5/e).
```

This closes Galois, decomposable non-Galois, and primitive non-Galois factors
inside the exact complete-branch common-factor interface. R18's exceptional
cross-factor envelope may still contain chart artifacts and isolated or
bounded-block components, but none can feed the balanced R16 compiler.

## Controls and boundaries

- The proof requires both complete trace and norm in (1). A trace-only factor
  does not produce the quadratic intermediate field.
- All endpoint identities are global rational identities. Finite-list-only
  interpolation tables remain excluded advice.
- The genus-one branch is not called algebraically absent; it is rejected by
  exact prime-subgroup Kummer injectivity.
- The constant `7` is a geometric upper bound before endpoint `psi`-fiber or
  chart restrictions, which can only reduce the block.
- Inseparable factors, characteristic `2` or `3`, `m>=N`, zero endpoints, and
  composite source torsion are outside the frozen theorem.
- This closes only the R16 common-factor compiler. It is not a lower bound on
  every possible implicit nonlinear router or ECDLP algorithm.
- No fresh multivalued target action, R10 index, relation density, rank,
  factor logs, or blind descent is supplied.

## Scoped disposition

1. Complete branch factorization creates one genuine quadratic intermediate
   field per nonzero endpoint.
2. A genus-one intermediate is an isogeny quotient and is injective on the
   prime-order Kummer list.
3. A genus-zero intermediate cannot serve two endpoints because translated
   pole divisors have different prime-order Picard classes.
4. The compositum of all endpoint fields is an elementary-`2` cover carried
   by a curve of genus at most one and has degree at most eight.
5. Every passing block has at most seven endpoints; no growing balanced common
   right factor survives, regardless of monodromy.

No new algorithm, relation, unrestricted lower bound, Shoup claim, or
breakthrough is authorized.

## Primary-source boundary

- Chalcraft and Fryers,
  [Kummer structures](https://arxiv.org/abs/0806.0409),
  supplies the complete two-valued Kummer setting. The quadratic-field,
  Picard-class, and genus arguments above are elementary deductions in the
  frozen function-field model.
- Beauville,
  [Finite subgroups of PGL(2,K)](https://arxiv.org/abs/0909.3942),
  supplies a classification check for the genus-zero elementary-`2` bound.
  The proof uses only that an odd-characteristic elementary-`2` subgroup of
  `PGL_2` has order at most four.

## Exactly one next action

Retire the common-right-factor compiler as a growing candidate after
independent review, then return to the R14/R15 representation frontier and
test a common-factor-free output-sensitive constructor. It must build the
actual paired branch image and one exact source within `O~(B^(9/4))`, expose
fresh multivalued target and R10 queries, and must not reintroduce a hidden
quadratic intermediate field, finite-list interpolation table, or uncharged
`S*L` source algebra.
