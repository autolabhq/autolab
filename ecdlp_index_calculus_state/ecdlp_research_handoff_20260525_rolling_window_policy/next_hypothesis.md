# Next Hypothesis

The public same-signature window policy found strong original-seed positives,
but fresh shared-challenge transfer failed even when row, scout, and filter
schedules were frozen. A first three-seed stability matrix showed that fresh
seeds can preserve one-target public signature families while losing the
cross-target alignment needed for compact two-equation witnesses. The next
hypothesis is that the useful object is not a salt window by itself, but a
challenge-invariant algebraic signature class inside the
summation-polynomial/FFE materialization stream.

Concrete next steps:

1. Extend the challenge-seed stability matrix beyond fresh-transfer-a/b/c,
   keeping row/scout/filter fixed and only the shared challenge varied.
2. Track per-target term signatures before verification: degree partition,
   leaf support, signed coefficient pattern, selected leaf rank, and public
   filter score.
3. Compare the original two-target signatures against fresh one-target
   signatures to isolate the missing cross-target invariant.
4. Penalize dense one-target repetition unless a compatible signature family
   also appears on the second target under the same public schedule.
5. Promote only signatures that survive on both targets across multiple
   challenge seeds before verifier-backed pair derivation.
6. If no same-signature family survives, switch the generator from
   hash-seeded salt windows to algebraic row synthesis: choose FFE/summation
   polynomial constraints first, then search rows that realize those public
   constraints on both fixed targets.
7. Keep Pollard-rho accounting attached to every verifier-backed pair, and
   keep row materialization cost visible.

Promotion rule:

Do not call this a breakthrough until a public, challenge-invariant selector
produces below-rho verifier-backed witnesses on fresh challenge seeds and
preserves explicit Pollard-rho accounting.
