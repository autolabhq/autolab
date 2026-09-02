# P1553 5A+5C fixed-marker scalar recurrence gate R89

## Classification

- Owner: existing P1515/P1536/P1553/R87-R88 field-router lane.
- Evidence: exact equal-size fixed-marker collision, six translation-orbit
  ranks, three shift-support controls, and exact multiplicity branches.
- Status:
  `FIXED_MARKER_LOCAL_JET_NONFUNCTORIAL__EXPLICIT_SHIFT_RECURRENCE_REACHES_C4_B2P4`.
- Cryptanalytic result: no nonlocal compressed translation state, relation
  rank, factor logs, blind descent, Shoup improvement, or breakthrough.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R88 report | `d73e017c731a54c6913aeaa94e6b5c6d54ca3757f56be14e8ca8ee5524031de1` |
| R88 gate | `90f0980bdeb51f540cd233f207218361cc14beb8a899c246418410d36ef43d56` |
| R87 gate | `16d635add67bc64d63d5870663f68ce37e35a21fa4feb436c428b7afbe6ed565` |
| P1536 norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |
| P1515 local-separator trichotomy | `dec667b097bcaefdf4c54091b2a9fa7757db5a65efe5b36e6ac15a6ff11a435a` |

## Fixed-marker collision

Over `F_17`, R89 freezes two systems of five equal two-element decks. Marker
weight in every slot is exactly the source choice index plus one, matching
the R86 interface.

At `T=0`, both systems have the same nonzero local state:

```text
[N, dN/dT, dN/ds_1, ..., dN/ds_5]
  = [4,2,13,13,13,5,15].
```

At `T=1`, their states are:

```text
[0,12,5,5,5,10,5]
[0,0,0,0,0,0,0].
```

The first translated fiber is simple; the second is multiple or nonreduced.
Therefore the fixed seven-scalar local jet is not a congruence for target
translation, even on equal-size decks and away from a zero local norm.

This removes R87's arbitrary-marker caveat for target-local state. It does
not rule out a nonlocal target-independent state.

## Translation ranks

For a five-by-two control over `F_1009`, the norm polynomial has degree 32
and every fixed-marker derivative polynomial has degree 31. Modulo a
degree-16 polynomial:

```text
norm translated-remainder rank       16
five marker translated ranks         16,16,16,16,16.
```

All six channels fill the quotient. This follows from the translation-span
theorem for a nonzero polynomial in characteristic above its degree. It
closes explicit linear translation sketches, not nonlinear circuits.

## Shift-support growth

Deterministic controls with deck sizes 2, 3, and 4 have prefix occurrence
sizes `d,d^2,...,d^5`. Sizes 2 and 3 are collision-free. Size 4 retains

```text
4,16,64,255,1020
```

distinct endpoints against `4,16,64,256,1024` occurrences, with minimum
occupancy above 99%.

At scale, one C deck has size `B^(3/5)`. The explicit translated-evaluation
recurrence therefore reaches:

```text
C^1  B^0.6
C^2  B^1.2
C^3  B^1.8
C^4  B^2.4
C^5  B^3.
```

It first exceeds the `B^(9/4)` setup cap at slot four and also misses the
`B^(5/4)` fresh-work cap.

## Scope

R89 closes only:

- the target-local seven-scalar fixed-marker state;
- explicit shifted-evaluation propagation;
- explicit linear translation-remainder sketches.

It leaves open a nonlocal nonlinear target-independent translation sketch,
non-Krylov arithmetic circuits, and representation-changing elliptic/FFE
identities. No projective exceptional source inverse, rank, logs, descent, or
generic-prime speedup is supplied.

Six of 16 obligations pass. The lane is not admitted and `breakthrough` is
false.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Producer | `a7dcd1f5f55484e5190cacc97671df02fdd95f109a9c4a504b3ab5059e8044d3` |
| Main report | `50a075dd73d3c5298be00efdbf5186734ddf694ed14bbb38cb37f890815a3b63` |
| Frozen recurrence | `2609f1c28989e7ea35f927b38152c3cbd757270b77cff841055dae53a9299dce` |
| Transition ledger | `8b62be2f52d53b40ceca2ae66714d1b52b05229f4c46022bd6d5ecdf24553702` |
| Norm/marker replay | `2f81d3e7cc49693baf1744cc8c11f2eb69b0c7cce2592394ff919918c7c512ed` |
| Exceptional controls | `fcbada85eb78563828c197e22de20a0de19d335367dab8b084a358ae86f89298` |
| Factor-log/descent receipt | `bfd897bc3a7cbf4a604c295d7221642dc3c8ca56c3b72da074c5ffd5ec2db1b3` |
| Unit test | `9deafd859521493ad46fe575f25ce456650744cc8e8fed7151587452a355248b` |

## Exactly one next action

Construct or refute one nonlocal nonlinear translation sketch for the fixed
norm/marker family. Freeze its target-independent state and deck-update law;
prove state at most `B^(9/4)`, fresh translation and source return at most
`B^(5/4)`, no shift-value or coefficient/quotient table, and exact reduced,
multiple, nonreduced, signed, infinity, tangent, and exceptional replay.
