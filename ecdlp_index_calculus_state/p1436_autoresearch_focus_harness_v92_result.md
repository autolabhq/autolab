# P1436 autoresearch focus harness V92 result

Date: 2026-07-29

## Result

V92 binds R143 as the 79th closed frontier lane and routes the highest
priority action to `s35_transposed_ffe_relation_span`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R143 first derives the exact sextic-character product defect. For `t=x*y`,

```text
delta_z(x,y) = chi_z(t)/(chi_z(x)*chi_z(y))
```

is the `q`th power of

```text
-(t+x*z+x)*(x+z+1)*(t*z+t+z)
/
((t+z+1)*(t*z+t+x*z)*(x*z+x+z)).
```

This is a fixed-degree rational function of the unknown C2 operand `x`,
with two numerator and two denominator roots. Every defined value replays
exactly on all eight frozen controls, but the identity does not locate `x`.

R143 also proves a group-theoretic boundary. If `G` has prime order and a
total law `F` satisfies

```text
sigma(x*y) = F(sigma(x), sigma(y))
```

for every pair, equality of signatures is a group congruence. Its classes
are cosets of a subgroup, so `sigma` is constant or injective. A nonconstant
fixed tuple of sextic labels with `6^k < |G|` therefore cannot support a
total label-only product law.

The exact finite controls agree with this boundary. Two controls have no
deterministic C2-by-C3 composition for any nonempty deck-parameter subset.
In the other six, the smallest deterministic table has one entry for every
C2-by-C3 operand pair. All eight fail the finite `floor(B^(9/4))` table
comparator. These finite results receive no asymptotic credit.

Finally, an ideal projected four-list merge with factor-base size
`n=N^beta` and projection size `N^mu` must satisfy

```text
mu <= 3*beta - 1
```

to emit `N^beta` explicit rows for `N^beta` factor-log unknowns. Its merge
work is at least `N^(1-beta)`, while explicit output costs `N^beta`. The
envelope `max(beta,1-beta)` bottoms out at exponent `1/2`. At the campaign
value `beta=9/20`, the ideal explicit merge costs `N^(11/20)`.

This does not bound compressed linear algebra or implicit relation-span
operators. The surviving route is a target-batched transposed
summation-polynomial/FFE operator that supplies known-RHS rank, factor logs,
and identical target descent below the square-root exponent without
materializing the explicit pair merge or row body.

No source locator, relation-rank construction, factor logs, identical target
descent, Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R143 tests: 11 passed.
- Full harness tests: 109 passed.
- Full ECDLP suite: 685 passed.
- R143 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R143: 68 receipts, 1,237 bindings, 0 mismatches.
- V92 frontier preflights: 79 provided, 79 closed.
- R143 obligations: 15 of 23 passed; lane admission false.
- R143 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V92 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R143 producer: `715c34ecaf4fd839a8c30d8b43886bad734d96f9113d0c1bed3cde31ed08b00d`
- R143 report: `d7d2ef76a1a0a8579ca543152e5d621e66a8e46723987896b45a8cc47feb950b`
- R143 frozen interface: `ad81801828ee4e88e7117bc3bd06d62ce237cc54bb24e2a90537f0857aebc588`
- R143 cost ledger: `2229360cdcd9d625b9c93746f53c973393d4bd005637296c3b3558f8c62bc7f5`
- R143 source replay: `e793128490b03e22b8d532f3cbbc298583a206fe0a8a206b41615fb99d20402a`
- R143 controls: `82f41b2215f9261e38533d65128b6cdc2e107b9aa496cc01086514a1d3d94e9e`
- R143 logs/descent: `abe05d9d06e23008f46ac9c040affa57e5bb488886ec925bec0de81143c3c271`
- R143 tests: `4d7573f630148b687a59a27fec9cb95f1e902e36d0e8ac5dc9d24bf09481a5d6`
- R143 gate: `e17719b1f8ef38c13c02c682f4a51b072ef3f47f23633cf29f3499a8e4ec6653`
- R143 parent: `91d76c1b26396e28a2f7199d196ae1bdd294641491dbdfed418d2840e60cc6c9`
- harness: `f9e032cc5f4a519c679ce34c156088e62ef5a6111a30cc081b45b4e4ebd66fb9`
- harness tests: `21d27a83b7dd2ed046c39886d71ce874117b6daf0ecb4f6182574daa12d41d5e`
- report: `0ab7e666db88dfc1a8dfd15d25277ed1e0ce08d3ee28ec578a0b1b0e1ce0d841`
- note: `b58fdfe08f6e9f31e877abc4878e3b382fef7a41d080881811afa2d63b5c470d`
- inventory: `ced9352f7885293c3d68afc72b039b1bb6ec32984402ffed591d5e282f869abc`
- replay plan: `a2799d728319e0f5b63de4956fde74ee1ce77cb07fd70f5f803a4643e68159e8`
