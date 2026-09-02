# Next Hypothesis

Freeze the exact marginal fourth-leaf trigger, then repeat the route on a fresh
window.

Recommended test:

1. Keep the total3/total4 public stress selector, pre-factor FFE gate, Sage
   factorization, and prefactor unique-leaf root policy.
2. Generate the public selector without a per-challenge cap, then select bundles
   from the larger public pool.
3. Re-run the pre-factor gate and root policy on the uncapped public source,
   not the earlier capped selector.  The current uncapped root policy expands
   the 200-207 gate from 7 to 15 preserving below-rho anchors.
4. Freeze the current declared fourth-leaf trigger before inspecting a fresh
   window: target-scoped sanity trigger `22050.cf1@11731`, base leaf `[90]`,
   added leaf `[34]`, total4-over-total3 marginal lift only.  Also track the
   more general audit rule `added_leaf=34`, but treat it as retrospective until
   it repeats.
5. For each candidate row/leaf surface, compute public pre-replay descriptors
   that can plausibly correlate with the eventual linear form: row schedule key,
   salt, selected leaf index pattern, root-hyperplane root, factor index,
   top-k, selector mode, and public root-cost profile.
6. Pair promising total3 assemblies with same-row total4 lifts, then retain
   exactly the marginal row/leaf surfaces whose leaf set strictly extends the
   total3 partner.
7. Build same-challenge bundles by maximizing predicted form diversity, not just
   anchor/root/row diversity.
8. Replay and report relation count, unique form count, duplicate count, rank,
   derived secret, public-key verification, retained-case ops/rho, and
   selected-bundle unique scan ops/rho.
9. Audit whether the useful fourth leaves are predictable from public features
   such as extra leaf index, row salt, root-policy root, selected hit-root
   ambiguity, and row schedule key.

Success criterion:

A fresh window should produce a public-key-verified same-challenge group using
only public candidate selection, public FFE/root-policy fields, and a
predeclared fourth-leaf trigger, with exact marginal selected-bundle ops/rho
below 1.0.  The current exact marginal 200-207 result does this for two
transfers: secret 4688 at selected-bundle ops/rho 0.71532847 and secret 9718 at
selected-bundle ops/rho 0.68613139.  The narrowed declared-trigger replay keeps
only four `[90] -> [34,90]` rows and still verifies both groups, with retained
case ops/rho between 0.34306569 and 0.35766423.

Failure criterion:

If the exact marginal result does not repeat on a fresh window, treat the 200-207
recoveries as a useful microscope for the fourth-leaf mechanism rather than a
general algorithm.  If it repeats but still depends on signature-provided leaves,
replace that dependency with a public fourth-leaf trigger before claiming a
candidate index-calculus speedup.
