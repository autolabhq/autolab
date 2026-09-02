# P1553 connected-Kummer deck obstruction R20

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 common-right-factor lane; no new
  idea ID.
- Evidence: theorem-only structural exclusion; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: every geometrically Galois common right factor in the
  frozen prime-subgroup regime has bounded degree. No non-Galois factor,
  target action, Shoup-bound improvement, or ECDLP breakthrough is supplied.

R19 reduced a growing geometrically Galois factor to cyclic or dihedral deck
orbits and charged their finite-list incidences. R20 closes the case earlier.
The complete unordered branch map cannot be invariant under a Mobius deck
transformation that fails to lift through the Kummer double cover. Connected
Kummer monodromy would collapse the branch discriminant, forcing a forbidden
order-`N` translation into the deck group of a degree-below-`N` function.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R16 common-right-factor compiler gate | `9f7ec4ecb6821e30affce39b25b459891f6be0b945b6d64332971dd86df2237c` |
| R18 repaired generic-pair gate | `d1eadf3edf74acedf30888aff764d221d1483ca3dc8521ff293cd7aa422ca5c4` |
| R19 Galois-orbit and target gate | `b5aef048b9748763f6449e845fa7554755e12715dce695cd99bdc8f0b77df61f` |
| R19 parent report | `e66c15494cf8b45480fda32deb2df3873a48d353e6e7b98f36b11d03254a742e` |
| R19 bundle hash list | `ceabb9c14112c4460519b72ea757efe8b9dde51d366865d4a64b2b6778e4da8f` |

## Frozen regime

Work over the geometric closure of `F_p`, with `p>2d`, an elliptic curve `E`,
and an odd prime-order subgroup `G=<P>` of order `N!=p`. Let

```text
x:E->P^1_q
f=psi composed with x:E->P^1,       deg(f)=m=2d<N.
```

The map `f` is separable and satisfies `f(-X)=f(X)`. Let `D_2` be the four
geometric branch points of the Kummer double cover `x`. For a nonzero endpoint
`A in G`, the complete branch map is

```text
F_A(q)=[f(A+Q)]+[f(A-Q)] in Sym^2(P^1),    q=x(Q).
```

All identities are rational identities on complete projective charts. No
finite-list sampling or denominator deletion is used.

## Lemma 1: a nonliftable Mobius graph has connected sign monodromy

For `kappa in PGL_2`, let `C_kappa` be the normalization of

```text
x(Q')=kappa(x(Q)).
```

Over the `q` line this is the fiber product of the Kummer double cover with
the pullback having branch set `kappa^(-1)(D_2)`. If `kappa` does not preserve
`D_2`, the two quadratic extensions have different branch divisors and are
distinct. Their compositum is therefore a connected biquadratic extension.
In particular, `C_kappa` is integral and has independent involutions

```text
sigma:  Q  |-> -Q,
sigma': Q' |-> -Q'.
```

The second projection `C_kappa->E`, `(Q,Q')|->Q'`, is finite and surjective.

## Lemma 2: unordered branch invariance collapses one discriminant

Assume `kappa` is a deck transformation of `F_A`, so

```text
F_A(kappa(q))=F_A(q).
```

In the function field of `C_kappa`, put

```text
u =f(A+Q),    v =f(A-Q),
u'=f(A+Q'),   v'=f(A-Q').
```

Equality of the complete unordered pairs is exactly

```text
u+v=u'+v',
uv=u'v'.
```

Thus `u'` is a root of `(X-u)(X-v)`. Since the function field is a field,

```text
(u'-u)(u'-v)=0
```

implies either `u'=u` or `u'=v` identically. Apply the independent involution
`sigma'`, which fixes `u,v` and swaps `u',v'`. In the first case it gives
`v'=u`; in the second it gives `v'=v`. Hence in either case

```text
u'=v'.
```

Surjectivity of the second projection now gives the rational identity

```text
f(A+X)=f(A-X)                                             (1)
```

for every geometric `X in E`.

## Theorem: every branch-map deck transformation lifts

Identity (1) says that `f` is invariant under the reflection

```text
iota_A(X)=2A-X.
```

It is already invariant under `[-1]`. Therefore it is invariant under their
composition

```text
iota_A composed with [-1]=t_(2A).
```

Because `A` is nonzero in an odd prime-order subgroup, translation by `2A`
has order `N`. But the deck automorphism group of the separable degree-`m`
map `f` has order at most `m`. This contradicts

```text
N<=|Aut(E/f)|<=m<N.
```

Consequently the assumption of Lemma 1 is impossible: every Mobius deck
transformation of `F_A` preserves the Kummer branch set `D_2` and lifts to the
elliptic double cover.

Now suppose a common right factor

```text
pi:P^1_q->P^1_r
```

is geometrically Galois of degree `e`. Its deck group `K` leaves every
`F_A=G_A composed with pi` invariant. For any one nonzero endpoint in the
block, the theorem places

```text
K subset Stab_PGL2(D_2).
```

The action of this stabilizer on four distinct points is faithful, so

```text
e=|K|<=24.                                                (2)
```

Thus no balanced family with `e=Theta(d)` and growing `d` can be Galois.
This excludes cyclic, dihedral, and exceptional geometric Galois factors
without any fifth-list incidence estimate.

## Relation to R19

R19's orbit-energy and genus statements remain valid necessary consequences
under their hypotheses. R20 proves that their growing nonliftable case cannot
arise from an exact common Galois right factor of even one nonzero complete
branch map in the frozen regime. The proposed `o(eL)` interval theorem is
therefore no longer required to close Galois factors.

The R19 single-valued fresh-target obstruction also remains independent and
unchanged. It still does not classify two-valued or otherwise multivalued
target interfaces.

## Controls and boundaries

- The proof uses the complete trace and norm, equivalently the complete
  unordered pair. Trace alone does not force equality of the two roots.
- The connectedness step requires `kappa(D_2)!=D_2`. Branch-preserving maps
  are retained and bounded by (2), not declared absent.
- The endpoint must be nonzero and satisfy `N>deg(f)`. Both conditions hold
  for the balanced prime-subgroup blocks under review.
- Inseparable maps and order-`p` subgroup phenomena are outside the frozen
  regime.
- A non-Galois right factor has no deck group of size equal to its degree, so
  this theorem does not classify non-Galois or decomposable exceptional
  components.
- No finite-list compression construction, compact all-chart certificate,
  fresh multivalued target action, R10 index, density, rank, factor logs, or
  blind descent is supplied.

## Scoped disposition

1. A nonliftable Mobius graph gives a connected biquadratic Kummer fiber
   product with independent sign involutions.
2. Exact unordered branch invariance on that connected curve collapses the
   translated branch discriminant.
3. The collapse forces order-`N` translation invariance of a separable map of
   degree below `N`, a contradiction.
4. Every Galois common-factor deck group embeds in the four-branch Kummer
   stabilizer and has degree at most `24`.
5. Every growing balanced Galois common factor is closed. Genuinely
   non-Galois factors remain open.

No new algorithm, relation, Shoup claim, or breakthrough is authorized.

## Primary-source boundary

- Chalcraft and Fryers,
  [Kummer structures](https://arxiv.org/abs/0806.0409),
  supplies the two-valued Kummer-addition setting. The connected quadratic
  compositum and root-swap argument above are elementary deductions for the
  complete branch map.
- Beauville,
  [Finite subgroups of PGL(2,K)](https://arxiv.org/abs/0909.3942),
  remains the classification source used in R19. R20 needs only the stronger
  elementary fact that the stabilizer of four distinct points injects into
  `S_4`.

## Exactly one next action

Classify the positive-dimensional non-Galois components inside R18's proper
cross-factor envelope. Start with a decomposable factor `pi=pi_2 composed
with pi_1`: compute the Galois closure action of the paired complete branch
field, separate chart artifacts, and either reduce a normal intermediate
stage to the degree-`<=24` theorem above or exhibit one explicit primitive
non-Galois factor with balanced endpoint and fifth-list saturation, a compact
all-chart identity certificate, and exact source backpointers. Keep target
action, R10, relation rank, factor logs, and blind descent explicitly charged.
