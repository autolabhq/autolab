# P1553 5A+5C sparse multihomogeneous moment recurrence gate R86

## Classification

- Owner: existing P1536/P1553/R82-R85 coloured-norm lane.
- Evidence: exact finite norm-jet controls on eight frozen instances, a
  constant-rectangle public colouring, and a charged constructor ledger.
- Status:
  `SUPPLIED_COLORED_NORM_JET_EXACT__STANDARD_MULTIGRADED_CONSTRUCTOR_PRODUCT_DIMENSION`.
- Cryptanalytic result: no public-input moment constructor, known-RHS rank,
  factor logs, blind descent, Shoup-bound improvement, or ECDLP breakthrough.

R86 instantiates the first-order coloured norm-jet decoder from the P1536
audit on the compact Cartesian-sum factor base `F={A_i+C_j}` from R82. The
decoder is exact when its jet is supplied, but the current producer obtains
that jet only after verifier DLP labels and full coloured-tuple enumeration.
It therefore receives no candidate or algorithmic relation credit.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R85 report | `2a94e5b2807a327cc2de5a35ec8c30c5961065cdc374a7ffd3090a3626e67f78` |
| R85 gate | `bb052ad21fde0f8f35dd56e014a6ab3a1ba1e148fd697fb80af230e4d3c4029a` |
| P1536 norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |
| R14 tensor-trace/minpoly gate | `da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac` |
| P1514 apolar-moment handoff | `16edd92f80a515f645d29577cea951859c4a56b45c65cd4931cf3874f83e48c7` |

All five hashes are verified before the producer runs.

## Coloured norm-jet control

Number flattened factor indices modulo five. Each colour is a union of at
most five rectangles of the form

```text
{A indices in one residue class} x {C indices in one residue class}.
```

Thus the colouring has a compact target-independent description over the
R82 factors. For one factor from each colour and verifier-only endpoint
labels `ell(z)`, form the conceptual product

```text
N(t,s) = product_z(
    ell(z)-t-s_0(index(z_0)+1)-...-s_4(index(z_4)+1)
).
```

At a target with one simple coloured source tuple, `N=0`,
`dN/dt` is nonzero, and each ratio

```text
(dN/ds_i)/(dN/dt)
```

recovers the corresponding flattened factor index, up to the fixed sign
convention. The factor index then recovers both atom indices because
`F_(a,c)` is flattened lexicographically.

All eight frozen instances recover the five factor indices, five pairs of
atom indices, and the target group element exactly. Empty fibers have
nonzero norm. Six instances contain genuine multiple fibers and have zero
norm and zero first jet at a maximum-multiplicity target; the two size-six
instances have no multiple fiber, so that branch is recorded as vacuous.
A synthetic double-root control also has zero norm and zero first
derivative.

These calculations use batch-BSGS endpoint labels. They verify the decoder
semantics only.

## Constructor ledger

For five coloured factor decks of size `Theta(B)`, the standard source
quotient has dimension and norm degree

```text
B^5.
```

A full value vector, companion-tensor matvec, Fermat projector, norm, or
resultant therefore materializes a `B^5` object before producing the
constant-size jet.

Using the R82 decomposition gives a sharper but still negative split. Five
coloured `A` choices have support exponent

```text
5*(2/5) = 2,
```

while five `C` choices have support exponent

```text
5*(3/5) = 3.
```

The `B^2` side fits the `B^(9/4)` setup cap. The `B^3` side does not, and
neither side supplies fresh-target work within `B^(5/4)`. The compact
colouring is therefore not itself a compact jet constructor.

This closes only the standard full quotient, explicit tensor/value-vector,
norm/resultant/projector, and explicit five-slot atom-product realizations.
It is not a lower bound against a compositional addition-pushforward
intertwiner or an unrestricted target-specialized arithmetic circuit.

## Finite receipts

| `B` | colour sizes | coloured quotient | distinct labels | max multiplicity |
|---:|---|---:|---:|---:|
| 6 | 2,1,1,1,1 | 2 | 2 | 1 |
| 15 | 3,3,3,3,3 | 243 | 21 | 30 |
| 18 | 4,4,4,3,3 | 576 | 400 | 6 |
| 28 | 6,6,6,5,5 | 5,400 | 2,694 | 12 |

Each row is replayed at two frozen offsets. The largest colouring uses four
rectangles per colour.

## Scope and nonclaim

R86 establishes:

- an exact supplied coloured first-norm-jet decoder;
- a compact five-colour description compatible with `F=A+C`;
- exact simple, empty, multiple, and nonreduced semantic controls;
- the `B^5` standard quotient and `B^2`-by-`B^3` atom-support charges.

R86 does not establish:

- a scalar-blind public-input jet constructor;
- a complete Semaev projective source biconditional;
- signed and infinity exceptional-chart replay;
- known-RHS relation rank or factor logarithms;
- identical fresh-target descent;
- a generic-prime algorithm or Shoup-bound improvement.

Eight of 18 admission obligations pass. The lane is not admitted and the
breakthrough flag is false.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Producer | `89c540fed698ea70fab5f57166f1147384201903c8f149ff90fed40bf6ee604f` |
| Main report | `b93c1e581a7953cf1a9a86d3ff4220f5c2a92b2ea3e0ed7076c57d8896ee1173` |
| Frozen moment interface | `a47df6a12ff98ca151c7f70b477a5a42a1836ce3d4b38ee337fa259dd0199ebd` |
| Support/regular-degree ledger | `af55a80d867ed0bb9712f67b8d1333ae2eb62f0687b0141be156daf02835923b` |
| Moment/flat-extension replay | `ef320f91426856ec3426140cd51e15b47f7f9701dd021dfe448275e061bb83c6` |
| Source/exceptional-fiber receipt | `78aede61add2035070b935a33bf0cca752ecafd7de48c59ffbe1aaf176410d8e` |
| Factor-log/descent receipt | `aebecc3a76b402a01667b18e88569e9c63a8c01cdf853bed3aa922d15f09000d` |
| Unit test | `c6fee01ef821f1180af7a3f6692db9ffb88405cc4f301e8ce657b24231887f24` |

## Exactly one next action

Construct or refute one jet-preserving addition-pushforward intertwiner for
`F=A+C`. It must propagate the coloured first norm jet from compact
`D_A,D_C` through five factor slots without forming the `B^3` five-`C`
support or `B^5` source quotient, specialize a fresh target inside
`B^(5/4)`, and return the jointly coupled factor and atom source with
complete exceptional-chart replay.
