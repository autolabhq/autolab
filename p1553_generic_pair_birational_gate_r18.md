# P1553 generic paired-map birational gate R18

## Classification

- Owner: existing P1553/IDEA-195 common-right-factor lane; no new idea ID.
- Evidence: theorem-only generic geometric exclusion, informed by exact R17B.
- Status: `DRAFT_REVIEW_REQUIRED_GENERIC_NEGATIVE`.
- Labels: `theorem-only`, `generic`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: no prime-subgroup density theorem, finite-deck
  compression, Shoup-bound improvement, or ECDLP breakthrough.

R16 proves that a shared right factor for complete endpoint branch maps is
controlled by their coordinate-field compositum. R17B rejects one
fiber-uniform degree-three map because one endpoint pair is birational onto
its image. R18 shows that this behavior is generic for the full-zero-fiber
interpolation family in every screened degree.

The conclusion is a generic no-go theorem, not an unrestricted lower bound.
Every surviving construction must lie in an explicit proper algebraic
common-factor envelope and must also satisfy both finite-list saturation
gates.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R16 common-right-factor compiler gate | `9f7ec4ecb6821e30affce39b25b459891f6be0b945b6d64332971dd86df2237c` |
| R16 parent report | `0442c740131216004d191137a248d9b4f524fcbd4d16226464d0fa79a6f65bc2` |
| R17 paired-compositum toy gate | `0ca83b9a6991fe593832519773971aa9fa17b705a7ea155f561cd18ecebc0101` |
| R17B exact report | `f5c9257d77a3c87efcf00655a5ec04735eebbacc3f5b7377df3266d9ed2cd154` |

R17B is evidence for the proof grammar only. No finite-degree computation is
used as proof of the asymptotic statement below.

## Interpolation family

Work over an algebraically closed field `k` of characteristic zero or
characteristic `p>2d`. Let

```text
E:y^2=x^3+alpha*x+beta
```

be nonsingular. Choose `d>=3` signed non-2-torsion endpoints
`A_1,...,A_d` with distinct Kummer coordinates

```text
a_i=x(A_i),
h(t)=prod_(i=1)^d (t-a_i).
```

For a polynomial `g` of degree at most `d`, define

```text
psi_g(t)=h(t)/g(t),
```

on the open parameter set

```text
g(a_i)!=0 for every i,
gcd(g,h)=1,
max(deg h,deg g)=d.
```

Then `psi_g^(-1)(0)` is the complete endpoint block
`{a_1,...,a_d}` for every parameter in the family.

For endpoint `A_i`, let

```text
F_(i,g)(q)=
  [psi_g(x(A_i+Q))]+[psi_g(x(A_i-Q))]
```

be the complete unordered trace/norm branch map on `q=x(Q)`.

## Theorem 1: a local witness makes the polynomial pair birational

Fix two endpoints `A_1,A_2` and put

```text
b_plus =x(A_2+A_1),
b_minus=x(A_2-A_1).
```

Assume

```text
b_plus!=b_minus,
b_plus,b_minus not in {a_1,...,a_d},
h(b_plus)!=h(b_minus),
h'(b_plus)!=0.                                           (local witness)
```

The odd-order prime-subgroup setting automatically keeps `A_1` and
`A_1+A_2` away from 2-torsion when the selected Kummer endpoints are distinct
and nonopposite.

Specialize to `g=1`, so `psi_1=h` is polynomial. R16's complete projective
pole calculation gives

```text
F_(1,1)^*(H_infinity)=2d*[a_1].
```

Suppose `F_(1,1)` and `F_(2,1)` shared a right factor `pi` of degree `e>1`.
The displayed divisor forces `pi` to be totally ramified at `q=a_1`.
Therefore every map factoring through `pi`, including `F_(2,1)`, must be
ramified at `a_1`.

On a local signed lift at `Q=A_1`, the two branches of `F_(2,1)` are

```text
h(x(A_2+Q)),
h(x(A_2-Q)).
```

The first derivative is nonzero: translation is etale, both Kummer maps are
unramified at the chosen non-2-torsion points, and `h'(b_plus)!=0`. The two
branch values are distinct by the local witness, so passage from the ordered
pair to trace and norm has invertible differential. Hence the complete
unordered map `F_(2,1)` is unramified at `a_1`, a contradiction.

Thus

```text
k(F_(1,1),F_(2,1))=k(q),
```

and the polynomial paired map is birational onto its image.

## Theorem 2: the local-witness endpoint locus is nonempty and open

Fix generic `A_1,A_2` so `b_plus` and `b_minus` are distinct from each other
and from `a_1,a_2`. Vary the remaining roots `a_3,...,a_d` over the
configuration space of distinct points avoiding this finite set.

Each failure in the local witness is an algebraic equation in the remaining
roots. Neither

```text
h(b_plus)-h(b_minus)
```

nor

```text
h'(b_plus)
```

is identically zero: varying one remaining root changes the two evaluation
functionals independently, while the other roots can be kept away from all
forbidden values. The configuration space is irreducible, so the intersection
of the complementary nonempty opens is nonempty.

Therefore the endpoint tuples admitting a polynomial birational-pair witness
contain a nonempty Zariski-open set for every `d>=3`. Every selected Kummer
coordinate has elliptic lifts over the algebraic closure. This geometric
statement does not prove that a prescribed prime-subgroup interval meets the
open set with the density needed by an algorithm.

## Theorem 3: generic denominators preserve pairwise birationality

For variable `g`, write the four complete paired equalities in independent
Kummer variables `q0,q1`:

```text
T_(1,g)(q0)=T_(1,g)(q1),
V_(1,g)(q0)=V_(1,g)(q1),
T_(2,g)(q0)=T_(2,g)(q1),
V_(2,g)(q0)=V_(2,g)(q1).
```

After cross-multiplication and saturation by the fixed projective chart
denominators, every equation contains the diagonal `q0-q1`. Their bidegrees
are bounded as functions of `d`.

Divide the universal diagonal from the four cross forms. For each possible
positive off-diagonal bidegree, tuples having a common factor of that
bidegree form a closed set: use the projective multiplication map from the
candidate common factor and four cofactor spaces. The finite union over
possible bidegrees is a closed set `X_cross`.

At `g=1`, Theorem 1 supplies no off-diagonal common factor. Hence

```text
X_cross={g:the four divided cross forms have a common off-diagonal factor}
```

is a proper Zariski-closed subset of the denominator parameter space. Every
genuine paired map of degree greater than one belongs to `X_cross`, because
its generic off-diagonal fiber component meets a complete finite chart and
divides all four cross equalities. The converse can fail on special
denominator or chart strata, so R18 does not claim that the genuine
nonbirational locus itself is closed.

The complement of `X_cross` intersects the dense open set `deg(g)=d`, as well
as the endpoint nonvanishing and coprimality opens. Therefore a generic
degree-`d` rational denominator gives

```text
k(F_(1,g),F_(2,g))=k(q).
```

The full endpoint block then has no nontrivial simultaneous common right
factor.

## Consequences and controls

1. Generic interpolation is not a construction for R16. It is rejected before
   finite endpoint or fifth-deck saturation is considered.
2. A surviving factor must force every selected endpoint pair into the proper
   closed common-factor envelope, not merely arrange a full zero fiber.
3. Output Mobius changes preserve paired-map degree and do not evade the
   theorem.
4. R17B is one exact point in the generic-negative behavior, not the proof.
5. Isogeny/Lattes controls remain special common-factor points, but fail
   prime-subgroup finite-list compression by injectivity.

## Remaining exceptional envelope

R18 does not classify the proper closed set `X_cross`, separate genuine fiber
components from denominator/chart artifacts inside it, or prove that the
genuine nonbirational locus is closed. The genuine part may contain:

- isogeny and Lattes decompositions;
- maps with nontrivial functional decompositions or finite deck groups;
- special non-Galois common intermediate covers;
- endpoint tuples satisfying addition-dependent identities;
- inseparable or wild maps outside `p>2d`, which are excluded from this gate.

Nor does R18 prove an arithmetic incidence bound showing that the prime-order
endpoint support avoids `X_pair`. A list-specific construction may deliberately
choose coefficients in the exceptional locus. Such a construction still owes
the complete R16 factor identity, compact verification, both finite-list
saturation bounds, fresh-target action, R10, density, rank, factor logs, and
blind descent.

## Scoped disposition

- Generic full-zero-fiber interpolation maps do not supply R16's common
  factor.
- The first paired-map obstruction is asymptotic in degree but geometric and
  generic only.
- The candidate frontier contracts to explicit special points of a proper
  off-diagonal common-factor envelope.
- No unrestricted circuit, incidence, or ECDLP lower bound is claimed.
- No new idea, run, Shoup claim, or breakthrough is authorized.

## Exactly one next action

Classify the positive-dimensional components of the closed envelope
`X_cross` under the complete Kummer trace/norm equations, first separating
genuine common fiber components from denominator and chart artifacts. Prove
that each genuine component is isogeny/Lattes, finite-deck/decomposable, or
fails one of the two finite-list saturation gates; otherwise extract one
explicit non-Galois component with its compact factor certificate. Keep
prime-subgroup arithmetic, fresh target, R10, density, rank, logs, and descent
charged separately.
