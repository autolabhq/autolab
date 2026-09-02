# P1553 list-only mass-collision toy gate R22

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 common-factor-free image-router
  lane; no new idea ID.
- Evidence: exact finite-field search plus fresh held-out interval.
- Status: `REJECT_TOY_NO_TRANSFER`.
- Labels: `toy`, `exact`, `model-bound`, `novelty-unverified`.
- Cryptanalytic result: no transferable twofold complete-key compression, no
  constructor, no Shoup-bound improvement, and no ECDLP breakthrough.

R21 closed every growing exact common-right-factor block. R22 tested the
remaining finite-list escape: tune a degree-eight fiber-uniform rational map
on one frozen endpoint/fifth list and retain one exact source per observed
complete trace-and-norm key, without claiming a global factor.

The training search found a visible but sub-threshold effect. The best of
4,096 denominators compressed 160 sources to 96 keys (`1.67x`), below the
preregistered `2x` gate. A disjoint fifth-scalar interval then reduced the
effect to `1.07x` on the 145 regular sources and exposed 15 denominator-pole
sources. The trained collision pattern does not transfer.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R21 quadratic-intermediate-field gate | `ad8d9e83b919ea92c98eefcb657831f01378b6760d49b2aaea0a892d6ed52be1` |
| R21 parent report | `5ff7c41e4d1e48877e8cfcef93c642e4ea52c531eac2c0d4ba67b3c5b71fb5cf` |
| R21 bundle hash list | `362b820cc46124e01e3c0d5f9e07cf6fac2287fd09b78160bb040e3e1ac2d64b` |
| R22C search contract | `12b42e35f38df4642a89214b93ffdfee9f5f68af48a74390cd7e7a02668960c5` |
| R22 search source | `48de8132d5c1538eca7b61cbe44b129a4649bfd200a7954f67ecd2bf0eefbc14` |
| R22C exact report | `40ef33016a6e12b802ee7c40183c853e4d180d89b7993abf85f1d6e6c48a5af8` |
| R22D holdout contract | `9dabdc2bbec1c204359fc95ee27c6ff0ce0e1ffb20b12fe7dcfb10541dc54ab2` |
| R22D holdout source | `0c97097abe93f973611a27df808a85763445d5c1bee8b29ea642ca91b6a463a1` |
| R22D exact report | `b32b017c2d126e537891c5891cb31d5720caeee660eb1d715d0fbe1e04190302` |

## Frozen experiment

The exact toy used

```text
F_193,
E:y^2=x^3+2x+3,
P=(1,44), order(P)=103,
d=8.
```

The eight endpoint scalars were `1,...,8`. The numerator of `psi` was the
monic product of their eight x-coordinate factors, so every endpoint belonged
to one exact `psi=0` block. The training fifth scalars were `9,...,28`.

R22C evaluated 4,096 deterministic pseudorandom monic degree-eight
denominators, using seed `155322`. A denominator was admitted only when it was
coprime to the numerator, separable as a rational map, and nonzero on every
training branch point. Every source key was the exact finite affine pair

```text
(psi(x(A+Q))+psi(x(A-Q)),
 psi(x(A+Q))*psi(x(A-Q))).
```

The acceptance gate required at most 80 distinct keys from 160 sources,
equivalently at least `2x` compression.

After the search, the selected denominator was frozen. R22D evaluated it with
no retuning on the disjoint fifth interval `29,...,48`. Its transfer gate
required at least `1.5x` compression and no poles.

## Preserved failed runs

Three versioned failures occurred before the successful R22C report:

| Version | Failure | Preserved stderr SHA-256 |
|---|---|---|
| R22 | Sage cache was not writable; experiment did not start | `e6e829e6ea5cdaf6da893b329ad9bf2838ffa740b2e58d8d96e647d6f92f6ebb` |
| R22A | Sage integer seed was not a native Python seed; experiment did not start | `05b04dc9773bce46d89b97c83b291c1de8577faf2c7b2616d58479eb5b1f2220` |
| R22B | Search completed but JSON serialization failed | `c495fa84b1fd4524dcfb238505ec767764bd264ee6c9365e5075f8ef9a0b075e` |

R22C changed only JSON serialization relative to R22B. Its successful report
was reproduced byte-for-byte. R22D was also reproduced byte-for-byte.

## Exact training result

The selected denominator was

```text
x^8 + 46*x^7 + 182*x^6 + 123*x^5 + 188*x^4
    + 41*x^3 + 100*x^2 + 140*x + 147.
```

The exact metrics were

```text
sources                         160
distinct complete keys           96
compression ratio              1.6666666667
collision excess                 64
maximum fiber                      8
sum of squared fiber sizes       412
minimum geometric fiber support    6
```

The identity-`psi` control had 160 distinct keys and maximum fiber one. A
uniform random-key control in the `193^2` key universe has expected distinct
count approximately 159. The candidate therefore created real list-specific
collisions, but did not pass the frozen twofold gate.

Nine repeated training keys had norm zero. They carried 34 sources and 25 of
the 64 collision excess. This is the expected special stratum where one branch
hits the deliberately planted endpoint-zero fiber. The other collisions were
finite-list equalities selected jointly with the denominator.

Even on the trained source distribution, a dictionary built from 96 uniform
source samples has expected covered source mass about `0.6805`. This is a
training-distribution statement only; it does not certify a fresh-target or
held-out query distribution.

## Exact fresh holdout

On fifth scalars `29,...,48`, the selected denominator produced

```text
nominal sources                  160
regular evaluated sources        145
denominator-pole sources           15
distinct complete keys           136
compression ratio              1.0661764706
collision excess                   9
maximum fiber                      2
zero-branch collision excess       0
```

The holdout failed both preregistered requirements: compression was below
`1.5x`, and the regular training chart did not remain regular. The large
training fibers disappeared; no held-out fiber had more than two sources.

## Interpretation

1. Finite-list denominator search can manufacture more complete-key collisions
   than random occupancy on one small interval.
2. The strongest observed effect missed the training acceptance gate.
3. The effect did not transfer to a disjoint fifth interval and introduced
   unhandled pole strata.
4. A sampled dictionary on the trained keys is not a target-independent image
   constructor or a fresh-query router.
5. Further search over denominators on the same list would be target-trained
   advice, not asymptotic evidence.

This rejects only the frozen random-denominator toy family. It is not a lower
bound against every list-restricted circuit, correlation sketch, or adaptive
probabilistic relation collector.

## Missing ECDLP path

R22 supplies no asymptotic image bound, all-chart containment proof, source
mass transfer theorem, adaptive-child index, fresh multivalued target action,
R10 queried coefficients, relation density, independent rank, factor-base
logs, or scalar-blind descent. No R22 datum authorizes a Shoup or breakthrough
claim.

## Exactly one next action

Do not tune another denominator on this source list. Derive one
list-independent common-factor-free identity for the complete three-point
correlation or its generating function, then preregister a multi-interval
all-chart test whose parameters are frozen before every endpoint and fifth
holdout. Reject it at the first pole-dependent chart, `S*L` representation,
training-only collision gain, missing joint source, or absent fresh-target and
R10 interface.
