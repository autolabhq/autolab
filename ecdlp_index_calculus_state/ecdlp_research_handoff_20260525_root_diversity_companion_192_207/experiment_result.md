# Experiment Result

New probe:

- `tasks/ecdlp_index_calculus/ffe_public_root_diversity_companion_selector_probe.py`

192-199 artifacts:

- Selector: `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_selector_total3_total4_public_stress_192_199.json`
- Replay: `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_stress_192_199.json`

192-199 result:

- Public root-policy candidate surfaces: 4.
- Retained surfaces: 8 total, with 4 root-policy anchors and 4 public companions.
- Selected source cases: 11 across 4 target/transfer challenges.
- Replay verified 4/11 retained cases and 1/2 relation challenge groups.
- The hard-target group `67.a1@9803`, transfer 196, reached rank 2 and derived
  secret 303.
- Context errors: 0.

200-207 artifacts:

- Selector: `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_selector_total3_total4_public_bounded_200_207.json`
- Replay: `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_bounded_200_207.json`

200-207 result:

- Public root-policy candidate surfaces: 7.
- Retained surfaces: 22 total, with 7 root-policy anchors and 15 public
  companions.
- Selected source cases: 20 across 5 target/transfer challenges.
- Replay verified 0/20 retained cases and 0/4 relation challenge groups.
- Context errors: 0.
- Challenge-group max rank was 2, but no group derived the public key.
- `22050.cf1@11731`, transfer 201, had two unique forms and rank 2, but
  `derived=false`.
- Hard-target `67.a1@9803` groups stayed non-derived; transfer 203 and 207
  reached only rank 1, and transfer 201 produced no relations.

Comparison to retained-root-only replay on 200-207:

- Retained-root-only replay already had 0 verified cases, 0 verified groups,
  challenge-group max rank 2, and relation count sum 16.
- Complete companion retention also has 0 verified cases and 0 verified groups.
- The companion run retained more surfaces and raised retained ops/rho, but it
  did not add useful independent forms beyond the existing boundary.

Interpretation:

The 192-199 sanity check confirms the new bridge did not break the known
public-selected hard-target derivation.  The 200-207 boundary is now sharper:
the FFE/root surface selector is still healthy, but simply retaining complete
public companion assemblies does not solve relation-form independence.  The next
branch should predict or select public relation-form independence directly,
rather than adding more companion surfaces around the same root anchors.

Uncapped public-source extension:

- Full 192-199 public selector:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_belowrho_full_selector_192_199.json`
- Full 192-199 selector replay:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_belowrho_full_192_199.json`
- Full 200-207 public bounded selector:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_200_207.json`
- Full 200-207 cap-6 selector:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_selector_total3_total4_public_bounded_full_cap6_200_207.json`
- Full 200-207 cap-6 replay:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_bounded_full_cap6_200_207.json`
- Full 200-207 cap-8 replay:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_bounded_full_cap8_200_207.json`
- Full 200-207 paired total4 cap-4 selector:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_selector_total3_total4_public_bounded_full_pair_cap4_200_207.json`
- Full 200-207 paired total4 cap-4 replay:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_bounded_full_pair_cap4_200_207.json`
- Full 200-207 uncapped gate:
  `ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_public_bounded_full_200_207.json`
- Full 200-207 uncapped Sage factors:
  `ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_public_bounded_full_200_207.json`
- Full 200-207 uncapped root policy:
  `ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total3_total4_public_bounded_full_200_207.json`
- Full 200-207 uncapped root-policy paired cap-4 replay:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_bounded_full_policy_pair_cap4_200_207.json`
- Full 200-207 marginal paired cap-4 selector:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_selector_total3_total4_public_bounded_full_policy_pair_cap4_marginal_200_207.json`
- Full 200-207 marginal paired cap-4 exact replay:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_bounded_full_policy_pair_cap4_marginal_exact_200_207.json`

Uncapped result:

- The full 200-207 bounded selector exposes 180 public cases.  Of these, 68
  intersect the 7 public root-policy anchor surfaces.
- With cap 4 per target/transfer challenge, replay still verifies 0 cases and
  0 groups.
- With cap 6, replay verifies 1/30 cases and 1/4 same-challenge groups.
- Cap-6 verified group: `22050.cf1@11731`, transfer 201, rank 4, four unique
  forms, derived secret 4688.
- The verified cap-6 case is `top_k=4`,
  `fixed_global_cap3_ow1_hw3_lw0_sw0_cw0_aw0`,
  `mode_cost_low_term_support_total4`, leaves
  `salt166:[90]`, `salt175:[90]`, `salt163:[34,90]`.
- The verified cap-6 case has ops/rho 1.02189781, so this is a fresh public
  derivation but not yet a below-rho speedup certificate.
- With cap 8, replay verifies 2/40 cases and the same transfer-201 group.
- With public total3-to-total4 lift pairing enabled, cap 4 is enough: replay
  verifies 1/20 cases and 1/4 same-challenge groups.
- Paired cap-4 verified group: `22050.cf1@11731`, transfer 201, rank 4, four
  unique forms, derived secret 4688, with only two duplicate forms.
- Paired cap-3 still fails, so the current minimal successful public bundle is
  cap 4 with explicit total4 lift pairing.
- The full 192-199 below-rho replay now verifies two hard-target groups:
  `67.a1@9803` transfer 196 secret 303, and transfer 199 secret 8903.
- Re-running the pre-factor gate on the full 200-207 public source materializes
  54 surfaces and selects 15 pre-factor gate surfaces.
- Sage factors all 15 selected surfaces; all 15 have preserving root-scan
  factors below rho.
- The uncapped root policy selects 15/15 preserving and false-positive-free
  anchors, with policy ops/rho min 0.47445255, mean 0.60669197, max 0.744.
- Using the uncapped root-policy anchors with paired cap 4 verifies 5/32 cases
  and 3/7 same-challenge groups.
- Verified uncapped-root-policy groups:
  `22050.cf1@11731` transfer 201 secret 4688,
  transfer 204 secret 9718, and transfer 205 secret 10141.
- The raw verified 200-207 candidates are still all over rho.  The best label
  seen in the full public source has ops/rho 1.01459854.

Marginal total4 exact-retention result:

- `tasks/ecdlp_index_calculus/ffe_single_hit_root_relation_replay_probe.py` now
  honors exact bridge-provided `row_leaf_keys` when present, instead of only
  retaining by row-level `surface_id`.
- This matters because the same row surface can appear as `[90]` in one case
  and `[34,90]` in a total4 lift.  Surface-only replay leaks the `[90]` scan
  into marginal-only tests.
- Marginal selector rule: pair public total3 assemblies with same-row public
  total4 lifts, then retain only the row surfaces whose leaf set strictly
  extends the total3 partner.
- Exact marginal replay verifies 1/19 retained cases and 2/6 same-challenge
  groups.
- Retained-case ops/rho: min 0.32116788, mean 0.39994468, max 0.488.
- Verified marginal groups:
  `22050.cf1@11731` transfer 201 secret 4688, and transfer 204 secret 9718.
- Public selected-bundle cost for transfer 201: two marginal scans, 98 ops
  against rho 137, ops/rho 0.71532847.  Only the `[34,90]` row on salt163
  emits the two useful forms; the selected `[65,79]` row on salt169 is a
  no-event public companion.
- Public selected-bundle cost for transfer 204: two marginal `[34,90]` scans
  on salts 167 and 176, 94 ops against rho 137, ops/rho 0.68613139.
- Transfer 205 remains a useful near miss: source replay derives secret 10141,
  but marginal-only retention leaves only rank-1 pieces.

Public fourth-leaf trigger audit:

- New probe:
  `tasks/ecdlp_index_calculus/ffe_public_fourth_leaf_trigger_audit.py`
- Audit artifact:
  `ecdlp_index_calculus_state/ffe_public_fourth_leaf_trigger_audit_total3_total4_public_bounded_full_policy_pair_cap4_marginal_exact_200_207.json`
- Trigger bridge:
  `ecdlp_index_calculus_state/ffe_public_fourth_leaf_trigger_bridge_total3_total4_public_bounded_full_policy_pair_cap4_marginal_base90_add34_22050_200_207.json`
- Trigger replay:
  `ecdlp_index_calculus_state/ffe_public_fourth_leaf_trigger_replay_total3_total4_public_bounded_full_policy_pair_cap4_marginal_base90_add34_22050_200_207.json`
- The retrospective public-feature audit scored 19 exact marginal row records
  across 8 target/transfer challenges.
- The simplest scored rule covering both verified marginal groups is
  `added_leaf=34`; the mechanistic declared trigger was narrowed to
  `target=22050.cf1@11731`, base leaf `[90]`, added leaf `[34]`.
- That declared trigger selects 4 exact marginal rows across transfers 201,
  204, and 205.  Three of the four rows emit relations; the selected-row
  relation precision is 0.75.
- Replaying only the declared-trigger bridge verifies both same-challenge
  groups: transfer 201 derives secret 4688, and transfer 204 derives secret
  9718.
- Trigger replay cost stays below rho on every retained case: min ops/rho
  0.34306569, mean 0.35036496, max 0.35766423.
- The trigger replay has 0 context errors, 4 retained relations, and 2/2
  verified challenge groups.
- This is still a 200-207 retrospective trigger audit.  It replaces the
  surface-only leakage and narrows the public fourth-leaf hypothesis, but it
  is not yet a fresh-window validation.

Updated interpretation:

The failure was not complete companion retention in itself; it was the earlier
per-challenge public-source cap plus a selector that did not explicitly keep
total4 lifts.  Once the public candidate generator is uncapped, a root-diverse
public bundle recovers a fresh 200-207 transfer key.  Pairing public total3
assemblies with same-row total4 lifts compresses the win from cap 6 to cap 4.
Re-running the FFE/root-policy stage on the uncapped source expands the fresh
result to three transfer keys.  Exact marginal total4 retention then converts
two of those transfers into below-rho selected-bundle recoveries.  The
fourth-leaf trigger audit narrows the strongest current mechanism to the
`[90] -> [34,90]` lift on `22050.cf1@11731`, and replaying only that declared
trigger still recovers both verified secrets below rho.  The next obligation is
to freeze this trigger before looking at a fresh total3/total4 window, then
test whether a public fourth-leaf rule can repeat without using
signature-provided leaf labels from the same window.
