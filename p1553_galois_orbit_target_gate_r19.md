# P1553 Galois-orbit and fresh-target gate R19

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 common-factor and target-action
  lanes; no new idea ID.
- Evidence: theorem-only structural reduction; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: no finite-field interval incidence bound, Shoup-bound
  improvement, or ECDLP breakthrough.

R18 contracts generic full-zero-fiber interpolation to a proper closed
cross-factor envelope. R19 treats two structured survivors:

1. a shared right factor `pi:P^1->P^1` that is geometrically Galois; and
2. a quotient label on which a fresh prime-subgroup translation acts through
   one single-valued rational map.

Growing Galois factors reduce to cyclic or dihedral orbit quotients. Exact
finite-list compression forces dense orbit incidence on the fifth deck, and
almost all of that incidence lies on genus-at-least-two correspondences that
do not lift to the elliptic curve. A single-valued fresh-target action is
impossible below degree `N`, independently of whether the factor is Galois.

These are scoped obstructions. Multivalued Kummer action, non-Galois factors,
and query-specific target interfaces remain open.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R16 common-right-factor compiler gate | `9f7ec4ecb6821e30affce39b25b459891f6be0b945b6d64332971dd86df2237c` |
| R17 paired-compositum toy gate | `0ca83b9a6991fe593832519773971aa9fa17b705a7ea155f561cd18ecebc0101` |
| R18 repaired generic-pair gate | `d1eadf3edf74acedf30888aff764d221d1483ca3dc8521ff293cd7aa422ca5c4` |
| R18 repaired parent report | `a11c7987bf2c307a5dde77d3ce597a659b356ac841380084c6c289523067d97d` |

## Frozen regime

Work over the geometric closure of `F_p`, with `p>2d`, an odd prime-order
subgroup `G=<P>` of order `N!=p`, and

```text
B=N^(1/5),
L=Theta(B),
B^(3/4-o(1))<=d<=B,
e=Theta(d),
2e<N
```

in the asymptotic screened range. The last inequality follows from
`e=O(d)=O(N^(1/5))` for sufficiently large `N`; it is explicit in the
fresh-target theorem.

Let `I` be the distinct geometric support of the frozen fifth Kummer list,
`|I|=L`. Occurrence multiplicities and source backpointers remain separately
charged.

## Theorem 1: growing tame Galois factors are cyclic or dihedral

Assume a common right factor

```text
pi:P^1_q->P^1_r
```

is geometrically Galois of degree `e`. Since `e<p`, its deck group

```text
K=Aut(P^1_q,pi) subset PGL_2
```

is tame and has order `e`. The tame finite-subgroup classification gives

```text
K in {cyclic, dihedral, A4, S4, A5}.
```

The exceptional groups have orders at most 60. Therefore every unbounded
balanced Galois factor is, after input and output Mobius changes, one of

```text
cyclic:   pi(q)=q^e,
dihedral: pi(q)=q^n+q^(-n),       e=2n.
```

Because every complete branch map `F_A` factors through `pi`, pullback degree
also gives

```text
e divides 2d,
deg(G_A)=2d/e=O(1)
```

for the balanced R16 compiler. Thus all growing geometric complexity is in
the cyclic or dihedral quotient; the endpoint-dependent outer maps have
bounded degree.

## Theorem 2: finite-list compression forces dense orbit incidence

Let

```text
J=|pi(I)|
```

and write `n_y=|I intersection pi^(-1)(y)|` for the attained fiber
occupancies. If the R16 saturation premise holds with a fixed constant `C`,

```text
J<=C*L/e,
```

then Cauchy-Schwarz gives

```text
sum_y n_y^2 >= L^2/J >= eL/C,
sum_y n_y(n_y-1) >= eL/C-L.                              (orbit energy)
```

Moreover, fibers with occupancy below `e/(2C)` contain fewer than

```text
J*e/(2C)<=L/2
```

list points. Hence at least `L/2` fifth-list points belong to attained
geometric fibers containing at least `e/(2C)` points of `I`.

For a Galois cover, every fiber is a `K` orbit. Passing finite-list
compression therefore means that a constant fraction of the fifth list lies
in densely sampled cyclic or dihedral Mobius orbits. Map degree alone supplies
none of this arithmetic saturation.

## Theorem 3: growing orbit incidence is nonliftable

Let

```text
x:E->P^1_q
```

be the Kummer double cover and let `D_2` be its four-point geometric branch
set. For `kappa in K`, lift the orbit graph `q'=kappa(q)` to the normalization

```text
C_kappa={(Q,Q') in E x E:x(Q')=kappa(x(Q))}.
```

As a cover of the `q` line, this is the fiber product of two double covers
with branch sets

```text
D_2,
kappa^(-1)(D_2).
```

If `kappa` does not preserve `D_2`, put

```text
s=|D_2 intersection kappa^(-1)(D_2)|<=3.
```

The connected `(Z/2)^2` cover is branched over `8-s` points. Riemann-Hurwitz
gives

```text
2g(C_kappa)-2=-8+2(8-s),
g(C_kappa)=5-s>=2.                                       (correspondence genus)
```

If `kappa` preserves `D_2`, it lifts to an automorphism of the elliptic double
cover. The subgroup of `PGL_2` preserving four points has bounded order, at
most 24. Thus a growing cyclic or dihedral `K` has only constantly many
liftable elements.

The orbit-energy bound supplies at least `eL/C-L` ordered distinct same-fiber
pairs. The graphs of all branch-preserving elements account for at most
`24L` of them. For growing `e`, the remaining

```text
Omega(eL)
```

ordered fifth-list incidences lie on genus-at-least-two nonliftable
correspondences `C_kappa`.

R19 does not prove that a scalar interval cannot have these incidences. It
turns Galois finite-list saturation into an explicit arithmetic incidence
obligation that is absent from standard isogeny or Kummer addition structure.

## Theorem 4: no sub-N single-valued fresh-target descent

Let

```text
h:E->P^1
```

be any nonconstant separable morphism of degree `m<N`, and let `T` be a
nonzero point of the prime-order subgroup. Suppose a single-valued rational
target action exists:

```text
h(Q+T)=tau(h(Q))                                         (target descent)
```

for every geometric `Q` and some rational `tau:P^1->P^1`.

Both sides have degree `m`, so `deg(tau)=1`. Iterating `N` times gives
`tau^N=1`. If `tau=1`, then `h` is invariant under translation by `T` and
factors through the degree-`N` etale quotient `E->E/<T>`, contradicting
`m<N`. Hence `tau` has order exactly `N`.

Quotient the equivariant map by the two cyclic actions:

```text
E --------h--------> P^1
|                     |
| degree N etale      | degree N, branched at two points
v                     v
E/<T> ----h_bar-----> P^1/<tau>.
```

The natural map from `E` to the normalization of the fiber product in this
square is generically one-to-one: the translation and Mobius actions are both
faithful on generic orbits. Both covers of `E/<T>` have degree `N`, so the
square is the normalized pullback square.

The right vertical cyclic cover is conjugate to `z |-> z^N` and is totally
ramified over two values. Its pullback along `h_bar` is etale only if every
ramification index of `h_bar` above those values is divisible by `N`. But

```text
deg(h_bar)=deg(h)=m<N,
```

so no such local index exists. This contradicts the etaleness of
`E->E/<T>`.

Therefore no exact single-valued target descent exists for `m<N`.

In particular, if a quotient candidate asks for

```text
pi(x(Q+T))=tau_T(pi(x(Q)))
```

and `2deg(pi)<N`, it fails. Valid successors must retain the genuinely
two-valued Kummer action, use another explicitly charged multivalued
interface, or avoid quotient-label target descent entirely.

## Controls and boundaries

- Isogeny Kummer maps provide algebraic common factors, but their prime-order
  fifth images are injective and fail `J=O(L/e)`.
- Cyclic and dihedral `pi` pass the geometric Galois classification but still
  owe the `Omega(eL)` list incidences and every source backpointer.
- The target theorem excludes only a single-valued rational action. Standard
  Kummer addition is two-valued and remains outside that theorem.
- Non-Galois right factors have no deck-orbit interpretation and remain open.
- No R19 statement bounds arbitrary finite-field points on the genus-two to
  genus-five correspondence family.
- No relation density, rank, factor logs, R10 index, or blind descent is
  supplied.

## Scoped disposition

1. Growing tame Galois factors reduce to cyclic or dihedral quotients with
   bounded-degree outer endpoint maps.
2. R16 finite-list saturation forces dense orbit fibers and
   `Omega(eL)` exact same-fiber incidences.
3. All but `O(L)` of those incidences use nonliftable genus-at-least-two
   correspondences.
4. Any exact single-valued fresh-target descent through a quotient of degree
   below `N` is impossible.
5. Non-Galois factors, multivalued target action, and a finite-field interval
   incidence theorem remain open.

No new idea, positive algorithm, Shoup claim, or breakthrough is authorized.

## Primary-source boundary

- Beauville,
  [Finite subgroups of PGL(2,K)](https://arxiv.org/abs/0909.3942),
  supplies the tame finite-subgroup classification. The orbit-energy, genus,
  and target-descent arguments in R19 are elementary deductions in the frozen
  setting.
- Pakovich,
  [On intersections of fields of rational functions](https://arxiv.org/abs/2603.29609),
  remains a comparison source for Galois functional equations. R19 does not
  replace a common-right-factor compositum by a field intersection.
- Ahmadi and Shparlinski,
  [On the Sum-Product Problem on Elliptic Curves](https://arxiv.org/abs/0806.0640),
  prove largeness for one of two sets built from independently chosen scalar
  subsets: `x(aP)+x(bP)` and `x(abP)`. That dichotomy does not upper-bound
  `|I intersection kappa(I)|` for one consecutive Kummer interval and a
  prescribed Mobius transformation, so it does not supply the R19 incidence
  theorem.

## Exactly one next action

For the cyclic and dihedral normal forms, prove or refute a uniform bound

```text
sum_(kappa nonliftable) |I intersection kappa(I)|=o(eL)
```

for the frozen prime-subgroup Kummer interval `I`. A proof closes every
growing Galois factor; a counterexample must provide the explicit Mobius
group, dense orbit decomposition, occurrence backpointers, complete branch
factor identities, and a multivalued fresh-target/R10 interface. Keep
non-Galois factors and the full ECDLP path separate.
