# P1553 5A+5C jet-preserving addition-pushforward gate R87

## Classification

- Owner: existing P1515/P1536/P1553/R82-R86 field-router lane.
- Evidence: exact marked-norm counterexample, a translation-orbit theorem,
  four finite rank controls, and an exact translated-gcd source replay.
- Status:
  `TARGET_LOCAL_FIRST_NORM_JET_NONFUNCTORIAL__EXPLICIT_TRANSLATED_REMAINDER_FULL_B2`.
- Cryptanalytic result: no compact public-input scalar resultant, known-RHS
  rank, factor logs, blind descent, Shoup-bound improvement, or ECDLP
  breakthrough.

R87 narrows the compositional exception left by R86. A first-order norm jet
at the final target is not itself a state that can be propagated through
addition. Exact composition asks for translated jets. The standard exact
replacement is the translated polynomial remainder modulo the left endpoint
polynomial, whose explicit target orbit has full `B^2` dimension.

This is a scoped representation result. It is not a lower bound against the
fixed factor-index marker subfamily, an implicit scalar resultant/half-gcd
algorithm, or an unrestricted arithmetic circuit.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R86 report | `b93c1e581a7953cf1a9a86d3ff4220f5c2a92b2ea3e0ed7076c57d8896ee1173` |
| R86 gate | `21d52b3d5f04fa408d0d9a4ef229f9169f05f4a31aa858e21ff252ef497f9b44` |
| P1536 norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |
| P1515 compressed-navigator gate | `dadcadf45bdea910f0a12e904bdfe32c4a517b0756ef08148de75fb39929e3e5` |
| P1515 local-separator trichotomy | `dec667b097bcaefdf4c54091b2a9fa7757db5a65efe5b36e6ac15a6ff11a435a` |

All five hashes are verified before the producer runs.

## Exact composition identity

For effective scalar endpoint divisors `A` and `C`, let

```text
P_C(T) = product_(c in C) (T-c).
```

Their Pontryagin addition pushforward satisfies

```text
P_(A*C)(T) = product_(a in A) P_C(T-a).
```

Differentiating gives

```text
P'_(A*C)(T)
  = sum_a P'_C(T-a) product_(b!=a) P_C(T-b).
```

Thus exact composition naturally consumes the translated first jets
`j^1_(T-a)(P_C)`, not only `j^1_T(P_C)`.

## Marked non-functor witness

Over `F_101`, take

```text
P roots = {1,18,24}
Q roots = {2,5,23}
T       = 0
A       = {0,1}.
```

The two monic split cubics have

```text
j^1_T(P) = j^1_T(Q) = [73,70].
```

Five constant marker-deformation polynomials `1,2,3,4,5` are realized for
each split cubic by exact Lagrange root perturbations. Consequently the full
marked target-local states agree:

```text
[N, dN/dt, dN/ds_1, ..., dN/ds_5]
  = [73,70,1,2,3,4,5].
```

After addition pushforward by `A`, however,

```text
j^1_T(A*P) = [37,51]
j^1_T(A*Q) = [77,73].
```

No universal operator receiving only that local marked first jet can compose
every split marked norm with every nontrivial left divisor.

The witness allows arbitrary public root-marker weights. It does not prove
the same collision inside R86's narrower fixed factor-index weight family.
That family could only escape by using its marker channels as additional
global translated-state advice; such an identity remains unsupplied.

## Translation-orbit theorem

Let `P` be monic of degree `n` over a field of characteristic `p>n`.
Translations of `P` span every polynomial of degree at most `n`: successive
finite differences recover a triangular basis with nonzero leading
coefficients. If `A(X)` is monic of degree `m<=n+1`, reduction modulo `A`
therefore gives

```text
span_T { P(T-X) mod A(X) } = K[X]/(A),
```

of dimension exactly `m`.

R87 verifies ranks `3,5,7,9` for `(m,n)=(3,5),(5,8),(7,11),(9,14)` over
`F_1009`. This theorem closes explicit linear remainder-orbit compression,
not nonlinear scalar computation.

For the R82 five-`A` versus five-`C` split:

```text
deg(P_A) = B^2
deg(P_C) = B^3.
```

Materializing `P_C` misses the `B^(9/4)` setup cap. Materializing
`P_C(T-X) mod P_A(X)` or its translated evaluation vector costs `B^2` fresh
words, above the `B^(5/4)` online cap.

## Exact source control

A deterministic five-by-two `A` deck and five-by-three `C` deck over
`F_1000003` produce:

```text
32 A occurrences
243 C occurrences
7,776 full occurrences.
```

At a frozen multiplicity-one target, R87 computes

```text
gcd(P_A(X), P_C(T-X)).
```

The gcd has degree one, recovers the unique `A` endpoint, recovers the
complementary `C` endpoint, and replays all ten source choices exactly. The
final norm has zero constant term and nonzero first derivative.

This positive control materializes both endpoint polynomials and enumerates
source dictionaries. It receives no candidate relation credit.

An empty target has gcd degree zero. A target with two distinct matches has
gcd degree two. A repeated `C` root can leave a degree-one squarefree gcd,
but its norm and first derivative both vanish, confirming that the first jet
is still required for the nonreduced branch.

Actual elliptic projective, signed, infinity, tangent, and exceptional-chart
source replay remains unsupplied.

## Scope and nonclaim

R87 closes:

- universal propagation from only a target-local marked first norm jet;
- explicit translated `C` value vectors;
- explicit `P_C(T-X) mod P_A(X)` linear remainder orbits;
- the materialized `B^3` characteristic-polynomial route.

R87 does not close:

- the fixed factor-index marker subfamily by a separate identity;
- black-box scalar resultant or half-gcd computation;
- implicit source localization without a `B^2` remainder;
- complete elliptic/Semaev exceptional-chart source recovery;
- relation rank, factor logs, identical descent, or Shoup improvement.

Six of 16 admission obligations pass. The lane is not admitted and the
breakthrough flag is false.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Producer | `d822035ed8e13a8ba0c987d5eab5a2a06bff251ebc32479ea358626ef038cf6b` |
| Main report | `f10ba663867815c9ee0b1234f4d9dee698d450a3a7171336d36f3e328ea2333a` |
| Frozen intertwiner interface | `17da8d76443ff32d9e603293888e321da139b5881c86a14ea9bffbbd4f1079b9` |
| Slotwise composition receipt | `e92a9997b3cca52f5b76105cb73bb94b0edce1ff92e7aa91fe5e96603a66ef00` |
| Target/source replay | `59b29d077fe60e37bd6c7c24471f305fe96392f3685e23d79e1d73fa3cc63fce` |
| Exceptional controls | `140445bb3acbd687125c375fd425d1560aa1c08c51277acf993e8ab3c0a0fe9d` |
| Factor-log/descent receipt | `f81bcda053f2d840d23bcab637e826148e5ee3fc9e3a96dbde712b78ca9c3b9c` |
| Unit test | `41f86fe36a9c0ae9f8f4331b4a0dde2f697d81ff967775068c629414a910e21c` |

## Exactly one next action

Construct or refute one black-box translated resultant/gcd source localizer
for `P_A(X)` and implicit `P_C(T-X)`. It must consume compact five-slot
`D_A,D_C`, avoid materializing the `B^3` `C` polynomial and `B^2` translated
remainder/evaluation orbit, specialize one fresh target inside `B^(5/4)`,
return the unique jointly coupled source, and reject multiple, nonreduced,
signed, infinity, and exceptional fibers.
