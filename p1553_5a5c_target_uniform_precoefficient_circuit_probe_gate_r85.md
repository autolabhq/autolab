# P1553 5A+5C target-uniform pre-coefficient circuit gate R85

## Classification

- Owner: existing P1539/P1553/R82-R84 Cartesian-sum lane.
- Evidence: exact theorem, eight finite controls, composite-order positive
  control, and prior-route cost reconciliation.
- Status:
  `FIXED_TARGET_EQUIVARIANT_QUOTIENT_TRIVIAL__STANDARD_PRECOEFFICIENT_GRAMMARS_OVER_CAP`.
- Cryptanalytic result: no known-RHS rank, factor logs, blind descent,
  Shoup-bound improvement, or ECDLP breakthrough.

R85 closes one broader mechanism than R83. A fixed label map need not be a
homomorphism: if every group translation induces an exact action on its
labels, its fibers are nevertheless subgroup cosets. A prime-order group
therefore permits only an injective label map or a constant map. It has no
proper fixed compressing target-equivariant quotient.

The result is not a lower bound against target-specialized nonlinear
circuits.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R84 report | `c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b` |
| R84 gate | `4e23024a1a5971a52d6664be678fd095f506814297e61b0f8992076e643e3661` |
| P1512 linear-Chow handoff | `d725027f381770686eb972b1acfcf9c370f17254725f05526d338dea2f213f25` |
| P1513 direct-KU handoff | `27c8f1f15fd0c3b81ebe2008aa96db12417c3f6612c5c151212206dcba388dcc` |
| P1514 apolar handoff | `16edd92f80a515f645d29577cea951859c4a56b45c65cd4931cf3874f83e48c7` |

All five hashes are verified before the producer runs.

## Equivariant-fiber theorem

Let `G` act on itself by translations and let `pi:G->Y` be surjective.
Suppose that for every `t in G` there is a map `tau_t:Y->Y` satisfying

```text
pi(t+g)=tau_t(pi(g))                 for every g in G.
```

Define `g~h` when `pi(g)=pi(h)`. The displayed identity makes `~`
translation invariant. Let `H` be the equivalence class of zero. If
`a,b in H`, translating `a~0` and `b~0` by `-b` shows `a-b~0`; hence
`H` is a subgroup. Translation invariance then makes every fiber one coset
of `H`.

For prime-order `G`, either:

```text
H={0},  so pi is injective; or
H=G,    so pi is constant.
```

No homomorphism assumption on `pi` is used.

This theorem applies to a fixed exact label quotient with action for every
fresh target. It does not apply to target-specialized labels, a
false-positive filter with charged verification, a restricted target set,
or a circuit that never exposes a quotient.

## Exact controls

Each of the eight R82 instances retains an injective `3A+2C` endpoint map.
R85 tests six fixed labels on a symmetric 49-point subgroup sample and 12
targets:

- the full point key has exact target action and source biconditionality but
  retains every right endpoint;
- the constant key has exact action and maximal compression but cannot
  identify a source;
- the x-coordinate key has the expected `P,-P` collisions and fails induced
  target action;
- SHA-256 encoding buckets modulo 2, 4, and 8 compress but fail induced
  target action and source biconditionality.

The x-coordinate control has 264 action conflicts on every family.
Hash controls have 244-432 conflicts, depending on field and modulus.
The full key retains 24, 150, 210, or 560 labels, exactly the corresponding
right endpoint support. No tested proper map has both exact target action and
a source biconditional.

The composite-order positive control

```text
Z/808Z -> Z/8Z
```

passes all 652,864 target/point action checks and has kernel size 101. This
confirms that the interface accepts a real proper quotient when the group
has one.

## Standard circuit reconciliation

Every explicit source-bearing binary contraction of five `A` leaves of
weight `2/5` and five `C` leaves of weight `3/5` has a root split. The best
balance is

```text
B^2.6 by B^2.4.
```

Thus even its smaller explicit payload exceeds both the `B^(9/4)` setup cap
and `B^(5/4)` fresh cap.

R85 also binds the closest independently audited/deferred circuit work:

- P1512 closes source-labelled scalar-linear Chow/Tate atomizers because
  determinant multiplicity pays the full source cycle;
- P1513 closes standard shared-norm, dense fiber-product, algebraic
  modular-composition, and direct KU encodings; a supplied common-factor
  decoder does not locate that factor;
- P1514 closes supplied-moment decoders as constructors, direct source
  moments, materialized/fully enumerated 2+3 joins, and the cited dense
  Macaulay route.

The P1514 boundary explicitly leaves a target-specialized sparse or
multihomogeneous moment constructor open. R85 preserves that exception.

## Scope and nonclaim

R85 closes:

- fixed exact target-equivariant compressing labels on a prime-order group;
- explicit source-bearing binary convolution nodes;
- the bound P1512 scalar-linear atomizer grammar;
- the bound P1513 standard shared-norm/KU representations;
- the bound P1514 standard moment and dense-Macaulay constructors.

R85 does not close:

- target-specialized nonlinear circuits;
- sparse multihomogeneous moment recurrences;
- false-positive labels with fully charged replay;
- arbitrary FFE/Semaev or arithmetic circuits.

No standard admitted route supplies relation rows inside the caps. Rank,
factor logs, and identical target descent remain absent.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Producer | `6caffd6809e63707c3527f1ff2d29d01e2cfce188436c48972149250eaf7445b` |
| Main report | `2a94e5b2807a327cc2de5a35ec8c30c5961065cdc374a7ffd3090a3626e67f78` |
| Frozen circuit interface | `9b158823d2ddf5f50d855c8336d9b1fa1ab427ec3f0cebb6c15082a3b224375a` |
| Node/payload ledger | `80d6baa73b6d054f86f3dde19c49583b9123629847d0505cc7bf75bb2b7722c9` |
| Fresh-target replay | `972e007bb04ccbc8e41d34726e154a423fd49e978ff66322be7ded8c5bcde799` |
| Matched controls | `a782ab497a972e632631e575e9d7ee30180e0a8d9592473730835d4e42a57790` |
| Factor-log/descent receipt | `34eef1ba106b389129416be8042f7f8df1cf78182fb797da640c1efdff3630e1` |
| Unit test | `03cb1c13287d81c8bd8eaf7c2e8c74224f844976c6c2eb54a3d3b9c1d47b3487` |

## Exactly one next action

Construct or refute one target-specialized sparse multihomogeneous moment
recurrence for the `5A+5C` fiber. Derive its moments directly from compact
`D_A,D_C` without a fixed label quotient or supplied oracle, fit `B^(9/4)`
setup and `B^(5/4)` fresh work, and recover one jointly coupled source on
every accepted fiber, including nonreduced and exceptional strata.
