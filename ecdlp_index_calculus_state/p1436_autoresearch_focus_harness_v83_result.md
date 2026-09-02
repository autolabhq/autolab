# P1436 autoresearch focus harness V83 result

Date: 2026-07-29

## Result

V83 binds R134 as the 70th closed frontier lane and routes the highest
priority action to
`s26_seven_mode_multi_predicate_torus_c5_selector`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R134 identifies a deterministic obstruction inside the structured C5
factor base. If `x` and `y` are two distinct atoms from one color, then
all six degree-five sources

```text
x^(5-j) y^j,  j=0,...,5
```

are accepted by that color. Their targets are the six distinct points

```text
z_j = x^5 (y/x)^j.
```

Because `y/x` is a nonidentity element of the prime-order subgroup, it has
exact order `q`. For a represented `t`-mode predicate with `t<=6`,
evaluation on the first `t` progression points is a Vandermonde matrix
with distinct nodes `(y/x)^(e_i)`. Its determinant is nonzero, so no
nonzero at-most-six-mode polynomial can vanish on the full progression.
The same obstruction applies separately to a represented rational
numerator-zero or denominator-zero set.

Every asymptotic balanced color has `Theta(B^(3/4))` atoms and therefore
contains the required pair. This is a deterministic theorem about the
structured factor base; it does not use R133's random-deck model or any
candidate discrete logarithm.

The eight actual pairing controls contain 30 active finite colors. Twelve
have two atoms and replay exact six-target progressions, exact C5 source
tuples, color multiplicity five, prime-order ratios, and nonzero sample
Vandermonde determinants. The remaining 18 singleton colors have no
two-atom witness and receive no asymptotic credit.

R134 closes only one represented zero or pole set through six modes.
Seven-or-more-mode predicates, combinations of several small predicates,
nonzero-value Frobenius-coordinate branches, high-expansion low-SLP
predicates, and general circuits remain open. No source index,
relation-rank construction, factor logs, identical target descent,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R134 tests: 10 passed.
- Full harness tests: 100 passed.
- Full ECDLP suite: 584 passed.
- R134 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R134: 59 receipts, 1062 bindings, 0 mismatches.
- V83 frontier preflights: 70 provided, 70 closed.
- R134 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V83 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R134 producer: `42be98b69e5ba120e79d75d52b66e85719e4e81e59e0a6ba9f5dfb1fd804420c`
- R134 report: `29b274cce5b88242f31e07906147157a44a3e783774413a230a1c4e4a679d9cf`
- R134 frozen interface: `2f9fb995e8a0b6821365c7ee404130cc1d6d58006e59012806a2105ffc2e3829`
- R134 cost ledger: `bd223baa1e454f7b9b09cf5bb4b5c733bc94f46daa1e853c8cfd2ace4c7454a6`
- R134 source replay: `384774470b2d92c2eaade2038ec1eb753e263f756e3a2c32ec1fd858b4082e04`
- R134 controls: `dfc7c9d0f13d21cd3bee1bba5a0dd7d727ef5df0fb7969ff66d9384293ea0a56`
- R134 logs/descent: `4784ebaa2d16e9dbc7fad24588a7c924b32f4a3efbc5633540d33f8fdb45a859`
- R134 tests: `1b71034e755e072967fff685cba29d65240e62b585090c64ac6702ca88379ad5`
- R134 gate: `161eca40da4f5fd1a861892a628168f10176b592f786913f24d07bf37f634f7c`
- R134 parent: `40c2423dc8939c25335465091840cc13e3a08e4f60dc9ff7e5ad556090e98dc3`
- harness: `252dc757c5d628960764b119fdde4cba564ee3295ccb29fa23cbaae1a7fc1e55`
- harness tests: `e2289d7057b7fb6b90c7cd1334f32063b09a9598860eeb7f34b5dbf98cd1912a`
- report: `52f8b7059bbe937c7c69063222ed2eb497a0459eee745673a0c787b23ec9edf0`
- note: `232bb4e8b37f7eaf729b17242762655bdb917fe4de6927d44b26262bec79a5dc`
- inventory: `2ef56fc9bc85e1971c83743ae6403d5a0f357d363bf85bf9f4f950e227ba04f1`
- replay plan: `beac817652021b8871fb68184c86546a36e08a71254a7a91482b658fd9e078d8`
