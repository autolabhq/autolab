# P1436 Autoresearch Focus Harness V35 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R84: explicit marked source section

R84 constructs exact scalar-blind `F_(p^2)` endpoint keys, radical side
polynomials, and packed source interpolants for

```text
(2A+3C) + (3A+2C) = T.
```

Across eight fixtures, every side source replays, every direct `5A+5C`
target is present in the split join, and all 32 sampled coefficient gcds
match the exact point intersection. The exhaustive audit covers 1,146,360
side pairs and 67,801 distinct targets.

The representation is over cap. Its prospective generic side coefficient
degrees are `B^2.6` and `B^2.4`; the smaller explicit side exceeds the
`B^(9/4)` setup cap, and fresh target translation exceeds `B^(5/4)`.
P1510's output-sensitive exception remains open before coefficient emission.

```text
report  c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b
gate    4e23024a1a5971a52d6664be678fd095f506814297e61b0f8992076e643e3661
parent  ff6e88d5075e36bd9353a1ec325558643035ddd5ac65107074d540a84230612c
```

## R85: pre-coefficient circuit screen

R85 proves that a fixed label map with exact induced action for every
translation has subgroup-coset fibers. On a prime-order group it is therefore
injective or constant, even when the label map is not a homomorphism.

The finite controls agree:

- the full point key is exact but uncompressed;
- the constant key is equivariant but not source-biconditional;
- x-coordinate and hash buckets compress only by losing exact target action
  or source biconditionality;
- `Z/808Z -> Z/8Z` passes all 652,864 action checks with kernel size 101.

R85 also binds the P1512 scalar-linear, P1513 shared-norm/KU, and P1514
standard moment/Macaulay scoped negatives. It preserves target-specialized
sparse multihomogeneous moments as the surviving mechanism.

```text
report  2a94e5b2807a327cc2de5a35ec8c30c5961065cdc374a7ffd3090a3626e67f78
gate    bb052ad21fde0f8f35dd56e014a6ab3a1ba1e148fd697fb80af230e4d3c4029a
parent  da7154bea3be91942bcb37692e9ca161724b3081765524bfcc43a5b34c55285c
```

## Harness routing

V35 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v35.json`

SHA-256:

`f51cb3f2b362e479b9dd8a58e860eeac4e7955fd5b5c783a8c1b95b06b5ecd83`

Schema: `ecdlp.p1436_autoresearch_focus_report.v27`.

Twenty-one hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_sparse_multihomogeneous_moment_recurrence
```

The next experiment must derive every target-specialized moment directly
from compact `D_A,D_C`, without supplied moments, a fixed label quotient,
dense Macaulay coordinates, or a materialized 2+3 deck. It must fit the
direct caps and recover a jointly coupled source through reduced,
nonreduced, signed, infinity, and exceptional fibers.

V35 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              ba025425e3abd8ba089bdc958f7d874b0f43add80e541fb1d672f3d6db41ae69
harness tests        dceafdd83b87926fe5e628fefd24bfece988805f53c682be9bd7571ee003e256
V35 focus note       0c8bcbfe9da02a0859cd3130c841377d6b4c758a16600cc598a66b764c674768
V35 FFE inventory    a3c454cc380cff26902c5abd27defead71f0eae69710df991605f7a594e1fb28
V35 FFE replay       c3004dc2728d858d09f32a1dd20f107029818b0d71d196f2de1af2c453d38bbb
```

## Verification

- R84/R85 targeted and harness tests: 58 passed.
- Full ECDLP task suite: 151 tests passed.
- Eleven R76-R85 and harness Python modules compiled.
- Ten R76-R85 parent YAML receipts parsed.
- All 115 declared input/artifact hashes matched.
- Every parent receipt has `breakthrough=false`.
- Clean R84 and R85 reruns reproduced all 12 generated JSON artifacts
  byte-for-byte.
- V35 has 21 closed lanes, `promotion_allowed=false`, and the expected sparse
  moment-recurrence action.
- `git diff --check` and explicit trailing-whitespace checks passed.
