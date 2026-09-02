# P1553 paired-compositum toy gate R17

## Classification

- Owner: existing P1553/IDEA-195 common-right-factor lane; no new idea ID.
- Evidence: one exact degree-three toy over `F_193`.
- Status: `DRAFT_REVIEW_REQUIRED_NEGATIVE_TOY`.
- Labels: `toy`, `exact`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: no asymptotic theorem, relation campaign, Shoup-bound
  improvement, or ECDLP breakthrough.

R16 corrected the field direction for common right factors. For endpoint
branch-coordinate fields

```text
K_A=k(T_A,V_A) subset k(q),
```

a shared right factor is controlled by the compositum, not the intersection.
For a pair `A,A'`, a degree-`e` shared factor forces

```text
e divides [k(q):k(T_A,V_A,T_A',V_A')].
```

Therefore one paired map birational onto its image is an exact rejection
certificate for the whole proposed block. R17 executes that certificate on
one fixed fiber-uniform non-Galois degree-three map.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R16 common-right-factor compiler gate | `9f7ec4ecb6821e30affce39b25b459891f6be0b945b6d64332971dd86df2237c` |
| R16 parent report | `0442c740131216004d191137a248d9b4f524fcbd4d16226464d0fa79a6f65bc2` |
| R17 original contract | `d75c2f8f6c2bc7f24cff28f3e3c220dda854d8f2a5b80c43cb14334759abfb56` |
| R17 original script | `1e407b89427e6cd9121e0123608878c6bc8bbf7b609981684eeb1d3634896bfe` |
| R17 failed receipt | `1dc7231ac321a99a15c10a017315cd01f15a450464af8f0c50095f88463e99a6` |
| R17A repair contract | `5f79e07b53594999d41a120a148111641229c0ae28f25082c57b35d59031d246` |
| R17A repair script | `e8cdcb9f2f1c6b84f83d38ecfcb09a32354753d6c24b1b0288d40d9504f778da` |
| R17A failed receipt | `29a4aa3227e9e9751e8953ef188839e1f54db068d63f805c071e71852b875a2e` |
| R17B repair contract | `7da9feeeca5baf490d75233af1e14e63ccb426ee325a9487e1d5ffa877f50ff7` |
| R17B repair script | `7fca161667dbc638efb34e3e78f684d6e8b1c8cfa5158aaec6ff0b334f16ce6a` |
| R17B exact report | `f5c9257d77a3c87efcf00655a5ec04735eebbacc3f5b7377df3266d9ed2cd154` |

R17 and R17A each consumed their single allowed run and failed only while
serializing the already-computed report. They carry no mathematical decision.
R17B is a versioned reporting-only repair and consumed one successful run.
The failed receipts remain part of the evidence.

## Frozen fixture

The exact fixture is

```text
k=F_193,
E:y^2=x^3+2x+3,
#E(k)=206,
P=(1,44),
ord(P)=103.
```

Take `d=3`, endpoint scalars `1,2,3`, and Kummer coordinates

```text
x(P)=1,
x(2P)=184,
x(3P)=62.
```

Define

```text
h(t)=(t-1)(t-184)(t-62),
g(t)=t^3+5t+7,
psi(t)=h(t)/g(t).
```

The denominator is nonzero on the three endpoints, so `psi^(-1)(0)` is the
full three-point block. The derivative numerator is

```text
h'(t)g(t)-h(t)g'(t)
 =54t^4+55t^3+7t^2+16t+44,
```

which is squarefree. Infinity is unramified because `deg(h-g)=2`. Hence every
geometric value fiber has at least two distinct points, and this degree-three
map is not Galois. It passes the finite-degree analogue of R16's
fiber-uniformity screen.

## Exact branch construction

For `a=x(A)` and `q=x(Q)`, let `X_+`, `X_-` be the two Kummer addition roots
`x(A+Q)`, `x(A-Q)`. Their trace and norm are exact rational functions of
`a,q`. Reducing `h(X)` and `g(X)` modulo

```text
X^2-(X_++X_-)X+X_+X_-
```

gives the complete trace and norm of

```text
{psi(X_+),psi(X_-)}.
```

The script checks these reduced formulas against direct elliptic additions at
six fixed endpoint/fifth pairs. Both trace and norm are retained.

For endpoints `P` and `2P`, introduce independent Kummer variables `q0,q1`
and cross-multiply the four equalities

```text
T_P(q0)=T_P(q1),
V_P(q0)=V_P(q1),
T_2P(q0)=T_2P(q1),
V_2P(q0)=V_2P(q1).
```

Every cross polynomial has total degree 11 and contains the diagonal
`q0-q1`. Their exact common gcd over `F_193` factors as

```text
q0-q1
```

with multiplicity one and no off-diagonal factor. A generic off-diagonal
fiber component cannot lie entirely in an affine denominator stratum; it
would meet the dense chart used by the cross equations. Thus the paired map
is birational onto its image and

```text
k(T_P,V_P,T_2P,V_2P)=k(q).
```

The fixed candidate has no nontrivial shared right factor even before finite
fifth-deck saturation, target action, or R10 are considered.

## Scoped disposition

1. The R16 compositum direction and pairwise rejection sieve work on one
   exact complete trace/norm fixture.
2. The fixed fiber-uniform non-Galois degree-three candidate is rejected by a
   birational endpoint pair.
3. The result does not prove that generic degree-`d` pairs are birational.
4. It does not exclude special asymptotic fiber-uniform maps or non-Galois
   common factors.
5. It supplies no finite-list endpoint saturation, fifth-deck compression,
   fresh-target action, R10 index, relation density, rank, factor logs, or
   blind descent.
6. R17 and R17A reporting failures are preserved and are not counted as
   negative mathematical trials.

No new idea, positive experiment, Shoup claim, or breakthrough is authorized.

## Exactly one next action

For the interpolation family

```text
psi(t)=prod_(i=1)^d(t-a_i)/g(t),
```

with a full endpoint zero fiber and generic degree-`d` denominator, prove or
refute that one endpoint pair has compositum `k(q)` on a Zariski-open set of
coefficients. Use the universal four-equation off-diagonal resultant or a
local ramification witness; charge coefficient construction and do not infer
an asymptotic theorem from R17B alone.
