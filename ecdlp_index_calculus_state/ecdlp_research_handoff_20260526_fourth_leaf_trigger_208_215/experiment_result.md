# Experiment Result

New probe:

- `tasks/ecdlp_index_calculus/ffe_public_fourth_leaf_trigger_bridge_probe.py`

Fresh 208-215 artifacts:

- Stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_208_215_probe.json`
- Public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_208_215.json`
- Pre-factor gate:
  `ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_public_bounded_full_208_215.json`
- Sage factors:
  `ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_public_bounded_full_208_215.json`
- Root policy:
  `ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total3_total4_public_bounded_full_208_215.json`
- Root-diversity marginal selector:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_selector_total3_total4_public_bounded_full_policy_pair_cap4_marginal_208_215.json`
- Root-diversity marginal replay:
  `ecdlp_index_calculus_state/ffe_public_root_diversity_companion_replay_total3_total4_public_bounded_full_policy_pair_cap4_marginal_exact_208_215.json`

Public source result:

- 258 public bounded cases under the `1.5*rho` cap.
- 10 verifier-positive source labels, but no below-rho source-label recovery at
  the stress stage.
- Best selected public case by cost is below rho but not verified:
  `22050.cf1@11731`, transfer 210, leaves `[8,56,90]`, ops/rho 0.3649635.

FFE result:

- Pre-factor gate selected 10/64 materialized surfaces.
- Sage factored all 10 selected surfaces.
- The prefactor unique-leaf root policy recovered public-zero roots for 10/10
  surfaces, all preserving, all false-positive-free, and all below rho.
- Root-policy ops/rho: min 0.50364964, mean 0.58106278, max 0.72.
- One selected surface has preserving full-remainder FFE ops/rho below rho:
  0.87591241.

Frozen trigger result:

- Direct public trigger bridge for `target=22050.cf1@11731`, base leaf `[90]`,
  added leaf `[34]` found 8 marginal rows across transfers 210, 213, and 214.
- Exact replay stayed below rho but did not derive:
  relation count sum 2, challenge-group max rank 1, verified groups 0.
- Retained ops/rho range for the frozen trigger: 0.32116788 to 0.37956204.

Exploratory controls:

- `base [90] -> added [8]` on `22050` selected 30 rows across 6 transfers.
  Replay reached challenge-group max rank 3 and relation count sum 6, but
  verified groups remained 0.
- `base [79] -> added [65]` on `22050` selected 18 rows across 6 transfers.
  Replay reached rank 3 on transfer 210, but verified groups remained 0.
- All public marginal total4-over-total3 rows on `22050` selected 60 cases
  across 7 transfers.  Replay reached rank 6 on transfer 210 and relation count
  sum 10, but still verified 0 groups.

Closest fresh positive:

- Two full source cases on transfer 211 derive secret 4273 with rank 3 and
  relation count 3.
- Those full source cases cost ops/rho 1.05839416, just over rho.
- The retained marginal row in the same case costs ops/rho 0.35766423 and rank
  1; the missing two base `[90]` rows are what push the full source case over
  rho but make the relation system derive.

Transfer-211 FFE diagnostic:

- Diagnostic Sage factorization:
  `ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_public_bounded_full_208_215_transfer211_deriving_surfaces.json`
- Explicitly factored the three deriving full-source surfaces on transfer 211:
  salts 166, 167, and 169.
- All three have nontrivial Sage factors and preserving root-scan candidates
  with root-scan ops/rho 0.33576642.
- This is not yet a usable compression win: all three diagnostic candidates
  have `selected_root_pair_count=0`, `selected_valid_root_leaves=0`, and
  `selected_missed_leaves=1`.  In other words, the surfaces are FFE-cheap but
  vacuous for the selected base leaf under the current gate.

Corrected exact-profile transfer-211 audit:

- New wrapper:
  `tasks/ecdlp_index_calculus/ffe_sage_factor_exact_profile_subset_probe.py`
- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_transfer211_deriving_total4_public_bounded_208_215.json`
- The wrapper avoids row-level `surface_id` contamination by cloning and
  materializing one exact row/leaf profile at a time.
- Exact profiles:
  - salt166 leaves `[8,90]`
  - salt167 leaves `[90]`
  - salt169 leaves `[90]`
- All three exact profiles have preserving nonvacuous Sage factors:
  `selected_root_pair_count=1`, no missing selected root pairs, and below-rho
  per-row factor root-scan charges.
- The best preserving factor on all three profiles has the same public linear
  fingerprint:
  `[(0,0,4058), (0,1,1), (1,0,6110)]`, i.e. the same
  `6110*b + c + 4058` factor over `GF(11731)`.
- Per-row best preserving factor surface ops/rho are 0.39416058, 0.37956204,
  and 0.37956204.  Per-row root-scan ops/rho are 0.48905109, 0.52554745, and
  0.51824818.
- This corrects the earlier diagnostic boundary: the base-row factors are not
  vacuous.  They are public, shared, linear, and relation-bearing.
- They still do not close the speedup gap.  The raw full-source derivation costs
  145 ops / 137 rho = 1.05839416.  Charging the shared 3-monomial linear factor
  once across the three rows gives 152 ops / 137 rho = 1.10948905 under the
  current surface-charge model, so the exact-profile FFE explanation is
  structurally cleaner but not cheaper than the raw near miss yet.

Public linear-factor gate replay:

- New probe:
  `tasks/ecdlp_index_calculus/ffe_public_linear_factor_gate_replay_probe.py`
- Transfer-211 exact replay artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_transfer211_shared_factor_total4_public_bounded_208_215.json`
- All-transfer `22050` replay artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_shared_factor_all_transfers_22050_public_bounded_208_215.json`
- The public factor gate keeps exactly the leaf-90 profiles on salts 166, 167,
  and 169 and rejects the dead leaf 8.
- It still derives secret 4273 with rank 3 and relation count 3.
- Measured replay cost improves from 1.05839416 rho to 1.05109489 rho, but
  remains above rho.
- In the all-transfer `22050` sweep, the gate was nonempty on 84/120 public
  cases and below rho on 44/120 cases, but the below-rho cases did not verify.
- Verified gate count was 4/120, all equivalent transfer-211 leaf-selector
  variants, and the best verified gate cost remained 1.05109489 rho.

Public x-match orientation audit:

- New probe:
  `tasks/ecdlp_index_calculus/ffe_public_linear_factor_xmatch_orientation_audit.py`
- Fresh 208-215 artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_shared_factor_all_transfers_22050_208_215.json`
- Backtest 200-207 artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_shared_factor_all_transfers_22050_200_207.json`
- Preceding-window 192-199 artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_shared_factor_all_transfers_22050_192_199.json`
- The audit exposes every x-match after the public linear-factor gate and tests
  public orientation rules before relation verification.
- On fresh 208-215, the exploratory public rule
  `salt_parity_even_35_odd_24` preserves the transfer-211 derivation and cuts
  the selected x-match count from 6 to 3.  Measured replay drops from
  144/137 to 138/137 = 1.00729927 rho, still just above rho.
- With an explicit shared identical leaf charge, the same 208-215 rule becomes
  a charged-model below-rho derivation at 136/137 = 0.99270073 rho.  With both
  shared leaf and shared hit-root charges, it is 134/137 = 0.97810219 rho.
- This is not a direct measured speedup yet: the below-rho 208-215 result
  depends on a charged amortization model for the identical leaf-90 work.
- Backtest on 200-207 is stronger: the same public linear factor plus
  `candidate_pos_min` finds measured below-rho verified derivations on transfer
  205, best 135/137 = 0.98540146 rho, deriving secret 10141.
- The preceding 192-199 stress source gives 0 verified rules for this factor,
  so the mechanism is not universal across all adjacent windows.

Fresh later-window validation:

- Fresh 216-223 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_216_223_probe.json`
- Fresh 216-223 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_216_223.json`
- Preregistered shared-factor/orientation replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_shared_factor_all_transfers_22050_216_223.json`
- The fresh stress source chose `fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0`
  as best policy, with 28 verified stress-leaf labels, 27 verified stress-row
  labels, no below-rho stress-leaf labels, and best stress-leaf ops/rho
  1.01459854.
- The public bounded selector emitted 161 public cases, 33 signature-verified
  labels, selected min ops/rho 0.94160584, and selected mean ops/rho
  1.1443199.
- For `22050.cf1@11731`, the fresh selector emitted 92 public cases over
  transfers 219-223.  Its cheapest public case was transfer 222, top_k 4,
  salts 162/164/166, leaves `[90,90,90]`, ops/rho 0.94160584, but it had
  rank 0 and was not verified.
- Replaying the preregistered factor
  `6110*b + c + 4058 == 0 (mod 11731)` on the fresh `22050` window verified
  8 public orientation rules and derived secret 6198 on transfer 219, but no
  verified rule was below rho.
- Best fresh measured replay was 143/137 = 1.04379562 rho.
- Best fresh shared-leaf charged replay was 141/137 = 1.02919708 rho.
- Best fresh shared-leaf plus hit-root charged replay was
  139/137 = 1.01459854 rho.
- The only best-rule families that verified in this fresh later window were
  `all` and `term_shape:2+2`; the exploratory
  `salt_parity_even_35_odd_24` and the 200-207 `candidate_pos_min` measured
  rule did not validate as below-rho fresh later-window wins.
- This is a negative result for the preregistered shared leaf-90 linear-factor
  package.  The relation signal persists, but the package does not beat rho on
  216-223 under either measured replay or the explicitly marked charged models.

Leaf-79 exact-profile discovery and unseen validation:

- Exact-profile leaf-79 artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_total3_public_bounded_216_223.json`
- In-window leaf-79 factor replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_factor1976_all_transfers_22050_216_223.json`
- Equivalent in-window line replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_factor270_all_transfers_22050_216_223.json`
- Unseen 224-231 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_224_231_probe.json`
- Unseen 224-231 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_224_231.json`
- Unseen 224-231 leaf-79 factor replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_factor1976_all_transfers_22050_224_231.json`
- The six fresh 216-223 leaf-79 exact profiles all have nontrivial preserving
  Sage factors with below-rho root-scan charges.  Minimum preserving root-scan
  cost is 0.33576642 rho, minimum preserving surface cost is 0.34306569 rho,
  and no preserving full-remainder candidate is below rho.
- The apparent 216-223 factors `1976*b + c + 9884` and
  `270*b + c + 2514` are not independent factor families.  They are two public
  lines through the same monic leaf point `(b,c)=(9485,5625)`, so they select
  the same leaf-79 profiles in the replay.
- In-window on 216-223, the leaf-79 gate plus public `candidate_pos_min`
  orientation gives a measured below-rho derivation: 135/137 = 0.98540146 rho,
  deriving secret 4437 on transfer 223 with rank 2 and relation count 2.
  Shared-leaf and shared-leaf plus hit-root charges are 0.97080292 and
  0.95620438 rho.
- This in-window result is a strong lead but not an independent validation,
  because the leaf-79 line was discovered from the same 216-223 exact profiles.
- The next unseen window, 224-231, has 212 public bounded cases and 26
  signature-verified labels.  Its best stress source policy is
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, with two below-rho verified
  stress-leaf labels and best stress-leaf ops/rho 0.608, but the cheapest
  public bounded cases are rank-0 and unverified.
- Preregistering `1976*b + c + 9884` on unseen 224-231 still derives a real
  `22050` secret, 9344, on transfer 227 with rank 3 and relation count 3, but
  it does not beat rho.  Best measured replay is 144/137 = 1.05109489 rho,
  best shared-leaf replay is 142/137 = 1.03649635 rho, and best shared-leaf
  plus hit-root replay is 140/137 = 1.02189781 rho.
- The 224-231 verified rules are `all` and `term_shape:2+2`; the
  `candidate_pos_min` compression that made 216-223 measured-below-rho does
  not validate on the unseen window.
- Follow-up 224-231 exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_transfer227_total3_public_bounded_224_231.json`
- The three transfer-227 leaf-79 profiles all have preserving below-rho
  root-scan factors.  Minimum root-scan cost is 0.33576642 rho, minimum surface
  cost is 0.34306569 rho, and minimum preserving remainder cost is still
  1.44525547 rho.
- The exact-profile factors do not give a single nonvacuous common factor
  across all three transfer-227 rows.  The line `270*b + c + 2514` appears on
  salt164 and salt165, but salt174's preserving factors miss the selected
  root pair.  This shows that the old `1976*b + c + 9884` line is best viewed
  as a public selector for a repeated monic leaf point, not as a universal
  resultant component in the next window.
- The 224-231 x-match overhead is now explicit: salt164 has one valid relation
  behind two public x-matches, salt165 has two valid relations behind four
  public x-matches, and salt174 contributes no relation after the leaf-79 gate.
  No tested public rule separates exactly those three relation-bearing
  x-matches yet.

Interpretation:

The exact `[90] -> [34,90]` fourth-leaf trigger did not reproduce as a
below-rho key recovery on 208-215.  This is a real negative for the narrow
200-207 trigger.  However, the FFE/root-policy layer itself generalized cleanly:
every fresh gate-selected surface had a preserving below-rho public root policy.
The fresh boundary is now sharper: transfer 211 exposes a shared public linear
factor on all three deriving rows, and that factor is strong enough to replay
the derivation after removing the dead leaf.  A public x-match orientation rule
gets the fresh measured cost to 1.00729927 rho, and a clearly marked shared-leaf
charge crosses below rho at 0.99270073.  The 200-207 backtest shows the same
factor family can produce measured below-rho verification, but 192-199 gives no
verification.  The next attempt should validate the shared-leaf amortization
mechanism on a fresh later window, not declare a general index-calculus
speedup from the charged 208-215 model alone.

That later-window validation has now been run on 216-223, and it is negative
for this exact shared factor/orientation package: best measured replay is
1.04379562 rho, best shared-leaf replay is 1.02919708 rho, and best
shared-leaf plus hit-root replay is 1.01459854 rho.  The next boundary should
move away from the transfer-211 leaf-90 factor as a complete package and ask
whether the fresh later-window leaf-79 relation rows have their own public
factor family.

The leaf-79 follow-up partly answers that: leaf-79 can produce an in-window
measured below-rho replay on 216-223, but the frozen line/orientation package
does not remain below rho on unseen 224-231.  The reusable signal is not simply
the line equation; it is the repeated monic leaf point plus a still-missing
public orientation rule that can reduce two x-matches per selected row to the
relation-bearing one without looking at verifier labels.

The transfer-227 exact-profile pass sharpens the next problem: the algebraic
factor layer keeps finding cheap public components, but the measured replay is
lost in x-match orientation and dead selected rows.  The next experiment should
mine public x-match features across the 216-223 success and 224-231 failure,
freeze one orientation/drop-row rule, and validate only on a new window.

Public rule-miner and later support-rule validation:

- New rule-miner script:
  `tasks/ecdlp_index_calculus/ffe_public_xmatch_orientation_rule_miner.py`
- The x-match audit now accepts public `where:` rules over compact metadata,
  including candidate/scheduled position, support set, term shape, salt
  modulus, and row x-match count.
- Label-ranked miner artifact:
  `ecdlp_index_calculus_state/ffe_public_xmatch_orientation_rule_miner_leaf79_216_231.json`
- The miner's cleanest label-only rule,
  `where:candidate_eq_scheduled&salt_mod2=1`, selected no invalid training
  x-matches but failed verifier replay on both 216-223 and 224-231.  This is a
  useful guardrail: label precision alone is not enough; a frozen rule must
  still derive.
- Verifier-backed training diagnostic:
  `where:candidate_eq_scheduled|support=0+5&scheduled_trial=1&salt_mod2=0`
  derives on both training windows, but remains above rho measured: 1.0 rho on
  216-223 and 1.02189781 rho on 224-231.
- Fresh 232-239 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_232_239_probe.json`
- Fresh 232-239 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_232_239.json`
- Frozen/manual rule replay on 232-239:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_manual_rule_all_transfers_22050_232_239.json`
- Default leaf-79 replay on 232-239:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_factor1976_all_transfers_22050_232_239.json`
- Exact-profile 232-239 transfer-233 artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_transfer233_total3_public_bounded_232_239.json`
- The frozen/manual training rule did not validate on 232-239: 0 verified
  rules over 80 `22050` source cases.
- The default leaf-79 gate still derived on 232-239, producing secret 6214 on
  transfer 233 with rank 2 and relation count 2.  It stayed above rho measured
  at 139/137 = 1.01459854, with shared-leaf plus hit-root charged cost
  135/137 = 0.98540146.
- The 232-239 exact profiles again showed cheap preserving public factors:
  all three transfer-233 leaf-79 profiles had below-rho root-scan and surface
  charges, but no preserving full-remainder candidate below rho.
- After inspecting 232-239 x-match metadata, the post-hoc simple support rule
  `where:support=0+5&scheduled_trial=1` explained the 232-239 derivation at a
  measured 135/137 = 0.98540146 rho, deriving secret 6214.  It also finds a
  measured 135/137 derivation on 216-223, deriving secret 3607.
- That support rule is not universal: it fails with 0 verified rules on
  224-231.
- Clean holdout 240-247 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_240_247_probe.json`
- Clean holdout 240-247 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_240_247.json`
- Clean holdout support-rule replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_support05_rule_all_transfers_22050_240_247.json`
- Clean holdout default leaf-79 replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_factor1976_all_transfers_22050_240_247.json`
- On 240-247, both the support rule and the default leaf-79 gate produced
  0 verified rules over 104 `22050` source cases.  This is a real negative for
  treating leaf-79 plus support `0+5`/scheduled-trial-1 as a general
  index-calculus speedup.
- The side-branch `67.a1@9803` target-cap1 stress signal recurred on 240-247:
  best stress-leaf ops/rho 0.536 in the stress probe, and best public bounded
  source cost 0.424, but the public bounded best case is rank 0 and unverified.

Current boundary after 240-247:

The leaf-79 branch now has a repeatable but intermittent pattern.  It can
produce measured below-rho recoveries under the public support rule on 216-223
and 232-239, but it fails on 224-231 and 240-247.  The algebraic FFE layer
continues to expose cheap preserving factors when the relation signal exists.
The missing component is no longer just an orientation rule; it is a public
activation predictor that tells when the leaf-79 gate is worth applying, or a
pivot to the recurring `67.a1@9803` target-cap1 below-rho stress branch.

Public activation gate and fresh 248-255 validation:

- New activation miner:
  `tasks/ecdlp_index_calculus/ffe_public_leaf79_activation_rule_miner.py`
- Activation miner artifact:
  `ecdlp_index_calculus_state/ffe_public_leaf79_activation_rule_miner_support05_216_247.json`
- Concise max-clause-2 activation artifact:
  `ecdlp_index_calculus_state/ffe_public_leaf79_activation_rule_miner_support05_216_247_max2.json`
- The x-match orientation audit now supports a public case-level
  `--activation-rule` that filters whole candidate cases before verifier
  derivation.  The activation features use public metadata only: transfer,
  policy, leaf selector, row x-match counts, support counts, scheduled-trial
  counts, factor-zero profile count, and row-salt residue counts.
- Frozen activation rule from 216-247:
  `activate:row_xmatch_counts=0,2,2&salt_mod3_residue0_count=1`
- Frozen orientation rule:
  `where:support=0+5&scheduled_trial=1`
- Training replay artifacts:
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_support05_activation_top_rule_216_223.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_support05_activation_top_rule_224_231.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_support05_activation_top_rule_232_239.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_support05_activation_top_rule_240_247.json`
- The frozen activation rule activates 4 cases on 216-223 and 4 cases on
  232-239, reproducing the measured below-rho support-rule derivations at
  135/137 = 0.98540146 rho.  It activates 0 cases on the negative 224-231 and
  240-247 windows.
- Fresh 248-255 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_248_255_probe.json`
- Fresh 248-255 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_248_255.json`
- Fresh 248-255 activation replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_support05_activation_top_rule_248_255.json`
- The fresh 248-255 stress probe had no below-rho stress-leaf labels for the
  best `22050` policy; best stress-leaf ops/rho was 1.01459854.  The public
  bounded selector emitted 209 total cases and 104 `22050` cases.
- The frozen activation-plus-support package validated on fresh 248-255:
  it activated 8 `22050` cases and verified 4 measured below-rho rules on
  transfer 255.  The best verified replay derives secret 5983 with rank 2,
  relation count 2, selected x-match count 2, and measured cost
  135/137 = 0.98540146 rho.  The explicitly marked shared-leaf and
  shared-leaf-plus-hit-root charged costs are 0.97080292 and 0.95620438 rho.

Current boundary after 248-255:

This is the strongest leaf-79 result so far: a public activation rule trained
on 216-247 and frozen before the next window produced a fresh measured
below-rho recovery on 248-255.  It is still not a general ECDLP index-calculus
algorithm.  The activation rule is simple and public, but it has only one
fresh holdout win after being selected from four adjacent windows.  The next
obligation is to repeat this exact frozen package on additional later windows
and inspect the transfer-255 exact profiles to understand whether the
activation feature corresponds to a stable FFE/summation-polynomial mechanism
or just a local row-salt coincidence.

Unretrained later holdouts and transfer-255 exact profiles:

- Fresh 256-263 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_256_263_probe.json`
- Fresh 256-263 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_256_263.json`
- Frozen 256-263 activation replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_support05_activation_top_rule_256_263.json`
- Fresh 264-271 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_264_271_probe.json`
- Fresh 264-271 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_264_271.json`
- Frozen 264-271 activation replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_support05_activation_top_rule_264_271.json`
- The frozen package did not activate on either later holdout:
  0 activated `22050` cases out of 80 on 256-263, and 0 activated `22050`
  cases out of 128 on 264-271.
- Transfer-255 filtered exact-profile source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_248_255_leaf79_transfer255_win_only.json`
- Transfer-255 exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_transfer255_total3_public_bounded_248_255.json`
- The exact-profile wrapper now has a replay-materializer fallback for cases
  whose rows are absent from the older worktree direct-source defaults; the
  successful run used the live `/Volumes/Volume/autolab/...` bank, config,
  direct, and transfer sources.
- All three transfer-255 leaf-79 exact profiles have nontrivial preserving Sage
  factors with below-rho root-scan and surface charges.  Minimum preserving
  root-scan cost is 0.33576642 rho, minimum preserving surface cost is
  0.34306569 rho, and no preserving full-remainder candidate is below rho.
- The two support-rule relation-bearing rows are salts 165 and 172.  Both have
  the same preserving public linear factor
  `270*b + c + 2514 over GF(11731)`, the same line seen in the earlier
  leaf-79 family.  Salt161 has preserving linear factors but no selected root
  pair and contributes no selected support-rule relation.

Current boundary after 256-271:

The activation-gated leaf-79 package has one genuine fresh measured below-rho
holdout win on 248-255, two subsequent no-activation misses on 256-263 and
264-271, and a clean exact-profile explanation for the 248-255 win: the two
relation-bearing rows share the old `270*b + c + 2514` public line.  This
keeps the FFE/summation-polynomial route alive, but the activation rule now
looks sparse rather than broadly predictive.  The next best move is not to
retrain immediately; it is to characterize the public row-salt/support
conditions under which the shared `270*b + c + 2514` line produces exactly two
support `0+5` scheduled-trial-1 relations below rho.

Factor-line mechanism audit:

- New mechanism audit:
  `tasks/ecdlp_index_calculus/ffe_public_factor_line_mechanism_audit.py`
- Audit artifact:
  `ecdlp_index_calculus_state/ffe_public_factor_line_mechanism_audit_leaf79_line270_216_271.json`
- Direct-line replay artifacts:
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_216_223.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_224_231.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_232_239.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_240_247.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_248_255.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_256_263.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_264_271.json`
- Direct-line audit artifact:
  `ecdlp_index_calculus_state/ffe_public_factor_line_mechanism_audit_leaf79_line270_direct_replay_216_271.json`
- The audit joins public activation/orientation replay artifacts with exact
  Sage profile artifacts and checks the fixed line
  `270*b + c + 2514` against support-rule relation rows.
- Replaying the frozen activation/support package directly through
  `270*b + c + 2514` reproduces the earlier equivalent-line behavior:
  measured below-rho verified recoveries on transfers 222, 233, and 255, and
  0 activated cases on 224-231, 240-247, 256-263, and 264-271.
- Across the 216-271 replay artifacts it inspected 700 public orientation
  cases, 16 activated cases, 12 verified cases, and 12 verified measured
  below-rho cases.  Collapsing duplicate leaf-selector variants gives 3 unique
  verified support systems.
- Across the exact-profile artifacts it inspected 15 exact leaf-79 surfaces.
  8 surfaces have a preserving `270*b + c + 2514` candidate, and 7 of those
  have a selected-root-pair candidate.
- All 3 unique verified support systems are mechanism matches: transfers
  222, 233, and 255 all have exactly two valid support `0+5`,
  scheduled-trial-1 rows, and those rows are exactly the rows where the
  `270*b + c + 2514` line has selected root pairs.
- The negative 224-231 exact-profile transfer 227 is now explained more
  cleanly: the line appears on salts 164 and 165, but only salt164 has a
  selected root pair.  That window never activates under the public activation
  rule and never forms the two-relation below-rho support system.

Current boundary after mechanism audit:

The FFE mechanism is no longer just "leaf 79 sometimes works."  The reusable
algebraic object is the public line `270*b + c + 2514`, and measured below-rho
recoveries occur in the audited data exactly when that line contributes two
selected-root rows that the public support rule selects as valid relations.
The open problem has narrowed to a public predictor for this two-row line
event on farther unseen windows.  Until that predictor validates beyond the
adjacent windows, this remains a strong mechanism lead rather than a completed
index-calculus algorithm.

Farther 272-287 public holdout:

- Fresh 272-279 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_272_279_probe.json`
- Fresh 272-279 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_272_279.json`
- Frozen direct-line 272-279 replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_272_279.json`
- Fresh 280-287 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_280_287_probe.json`
- Fresh 280-287 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_280_287.json`
- Frozen direct-line 280-287 replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line270_support05_activation_top_rule_280_287.json`
- Updated direct-line mechanism audit:
  `ecdlp_index_calculus_state/ffe_public_factor_line_mechanism_audit_leaf79_line270_direct_replay_216_287.json`
- The broad frozen public selector still finds low-cost 22050 candidates:
  272-279 has a best public-selected case at transfer 277 using leaves
  `[19,34,90]`, salt 167, top_k 4, and cost 0.32846715 rho; 280-287 has a
  best public-selected case at transfer 285 using leaves `[8,56,90]`,
  salt 174, top_k 12, and cost 0.32846715 rho.
- The stricter line-270 activation does not fire on the farther holdout:
  0 activated cases out of 144 target cases on 272-279, and 0 activated cases
  out of 68 target cases on 280-287.
- The updated 216-287 direct-line mechanism audit now inspects 912 public
  orientation cases, still has only 16 activated cases and 12 measured
  below-rho verified cases, and still has exactly 3 unique mechanism-matched
  support systems on transfers 222, 233, and 255.

Current boundary after 272-287:

The public low-cost FFE selector remains productive on later windows, but the
specific `270*b + c + 2514` two-row support mechanism does not extend to
272-287 under the frozen activation rule.  That demotes the line-270 family
from "likely global predictor" to "mechanistically clean local family."  The
best next move is to mine a new factor-line or support-event family from the
fresh 272-287 public below-rho candidates, especially the transfer-277
`[19,34,90]` and transfer-285 `[8,56,90]` rows, while keeping the line-270
audit as a template for proving any new candidate.

Exact profiles for the fresh 272-287 candidates:

- Transfer-277 exact profile:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_transfer277_total3_public_bounded_272_279.json`
- Transfer-285 exact profile:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_transfer285_total3_public_bounded_280_287.json`
- Exploratory shared-line public replays:
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line224_exploratory_272_279.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line224_exploratory_280_287.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line350_exploratory_272_279.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line350_exploratory_280_287.json`
- Both exact profiles materialized successfully with Sage 10.9 and live
  `/Volumes/Volume/autolab` materializer sources.  Each has one nontrivial
  preserving degree-1 factor candidate with root-scan cost 0.35036496 rho and
  surface cost 0.37226277 rho, but both have `selected_root_pair_count = 0`,
  `selected_missed_leaves = 3`, and `verified_case_count = 0`.
- Transfer 277 has 22 degree-1 resultant factors; transfer 285 has 20.  They
  share five linear factors, including `224*b + c + 3252` and
  `350*b + c + 5190`.  The old `270*b + c + 2514` line appears in transfer
  277 but not transfer 285.
- Unfiltered public orientation replays for the shared `224*b + c + 3252` and
  `350*b + c + 5190` lines over both 272-287 windows produced no verified
  rules and no measured below-rho recoveries.

Current boundary after 272-287 exact profiles:

The later windows still expose cheap degree-1 FFE structure, and some factors
repeat across unrelated low-cost candidates.  The missing ingredient is not
factorization; it is selected-root alignment.  The next experiment should mine
for public conditions that predict selected-root rows among the common
degree-1 factors, or broaden exact profiling to nearby candidate rows until a
two-row support system appears.

Target-cap1 broad-selector validation on 288-303:

- Fresh 288-295 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_288_295_probe.json`
- Fresh 288-295 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_288_295.json`
- Fresh 296-303 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_296_303_probe.json`
- Fresh 296-303 public source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_296_303.json`
- Exact-profile samples:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_verified_selected_root_mining_288_295.json`
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_verified_selected_root_mining_296_303.json`
- Public line replays:
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line4745_exploratory_288_295.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line4745_exploratory_296_303.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line586_exploratory_288_295.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line586_exploratory_296_303.json`
- The same frozen target-cap1 row selector remains the best stress policy on
  both windows, with verified below-rho stress leaves at 0.39416058 rho.
- The bounded public selector gives verified below-rho `22050` recoveries in
  both windows:
  - 288-295 has 4 verified below-rho `22050` cases.  Best is transfer 292,
    salt 167, leaves `[8,56,90]`, top_k 12, rank 2, relation count 2, at
    0.39416058 rho.
  - 296-303 has 4 verified below-rho `22050` cases.  Best is transfer 302,
    salt 165, leaves `[65,79,90]`, top_k 16, rank 2, relation count 2, at
    0.39416058 rho.
- The exact profiles for transfer 292 and transfer 302 all verify and all have
  preserving degree-1 Sage factors below rho under root-scan and surface costs.
  Transfer 292 uses line `4745*b + c + 3236`; transfer 302 uses line
  `586*b + c + 3197`.  In each window the total3 and total4 variants share the
  same selected-root line.
- Public replay of `4745*b + c + 3236` over 288-295 verifies 8 below-rho rules;
  best measured replay cost is 0.37956204 rho and derives secret 1663.
- Public replay of `586*b + c + 3197` over 296-303 verifies 15 below-rho rules;
  best measured replay cost is 0.35036496 rho and derives secret 10867.
- Cross-window specificity check: `586*b + c + 3197` finds no verified rules on
  288-295.  `4745*b + c + 3236` does find verified rules on 296-303, but its
  best case is a different target-cap3/global-style support system at transfer
  299, not the target-cap1 transfer-302 row.

Current boundary after 288-303:

The broad target-cap1 public selector is now the strongest route: it has
fresh verified below-rho `22050` recoveries on 272-279, 288-295, and 296-303,
with no verifier-informed selector retraining for those windows.  The mechanism
is not the old line-270 family.  It looks like a per-window degree-1 FFE line
event selected by stable low-term support patterns: `[8,56,90]` around
transfers 276/292, and `[65,79,90]` around transfer 302.  The next algorithmic
step is to freeze a public predictor for these target-cap1 rows and their
degree-1 line events, then validate beyond 303.

Frozen target-cap1 predictor and unseen 304-319 validation:

- Predictor script:
  `tasks/ecdlp_index_calculus/ffe_public_target_cap1_event_predictor.py`
- Calibration artifact:
  `ecdlp_index_calculus_state/ffe_public_target_cap1_event_predictor_v1_calibration_272_303.json`
- Fresh validation artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_304_311_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_312_319_probe.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_304_311.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_312_319.json`
  - `ecdlp_index_calculus_state/ffe_public_target_cap1_event_predictor_v1_validation_304_319.json`
- Exact-profile artifacts:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_predictor_selected_304_311.json`
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_predictor_selected_312_319.json`
- Public line replay artifacts:
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line4059_predictor_304_311.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line4059_predictor_312_319.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line6202_predictor_304_311.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line6202_predictor_312_319.json`
- Predictor v1 selects only public fields: target `22050.cf1@11731`, policy
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, single-row low-term
  total3/total4 cases, top_k in `{4,7,12,16}`, and the frozen leaf signatures
  `[8,56,90]`, `[8,34,56,90]`, `[8,54,56,90]`, `[65,79,90]`,
  `[8,65,79,90]`, and `[19,34,89,90]`.  Verifier labels are reported only
  after public selection.
- On calibration/selection material 272-303, predictor v1 selects 27 public
  systems and 9 are verified below rho after selection.
- On unseen 304-319, predictor v1 selects 8 public systems and 7 are verified
  below rho after selection.  Best selected fresh case is transfer 310, salt
  165, leaves `[8,56,90]`, top_k 7, at 0.39416058 rho, rank 2 and relation
  count 2.
- The full public-bounded sources independently show 4 verified below-rho
  `22050` cases on 304-311 and 10 on 312-319.
- Exact profiles of the selected fresh systems verify all requested cases and
  preserve selected roots through a single degree-1 line in each window:
  transfer 310 uses `4059*b + c + 5157`; transfer 319 uses
  `6202*b + c + 10586`.
- Public replay of `4059*b + c + 5157` over 304-311 verifies 8 below-rho rules;
  best measured replay cost is 0.37956204 rho and derives secret 5959.
- Public replay of `6202*b + c + 10586` over 312-319 verifies 30 below-rho
  rules; best measured replay cost is 0.35036496 rho and derives secret 2826.
  This line is recurrent: it was already the selected-root line for transfer
  276, then reappears on transfer 319.
- Cross-window line specificity remains useful: `4059*b + c + 5157` has no
  verified rules on 312-319, and `6202*b + c + 10586` has no verified rules on
  304-311.

Current boundary after 304-319:

The campaign now has a frozen public target-cap1 predictor with an actual
unseen-window validation.  It is still not a complete index-calculus algorithm,
because the degree-1 line is discovered by exact confirmation after row
selection.  But the public row/leaf event is no longer merely post-hoc: it
selects fresh rows beyond 303 with high verifier-backed precision, and the
exact line replay derives secrets well below rho.  The next missing layer is a
public predictor for the per-window degree-1 line, or an amortized confirmation
argument that charges exact line discovery honestly while staying below rho.

Predictor v1 failure on 320-335 and target-67 split:

- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_320_327_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_328_335_probe.json`
- Public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_320_327.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_328_335.json`
- Predictor artifacts:
  - `ecdlp_index_calculus_state/ffe_public_target_cap1_event_predictor_v1_validation_320_335.json`
  - `ecdlp_index_calculus_state/ffe_public_target_cap1_event_predictor_v2_target67_top4_calibration_272_319.json`
  - `ecdlp_index_calculus_state/ffe_public_target_cap1_event_predictor_v2_target67_top4_validation_320_335.json`
- Exact-profile and public replay artifacts:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_predictor_v2_328_335.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line945_predictor_v2_328_335.json`
- The strict `22050.cf1@11731` v1 predictor fails the next held-out window:
  it reads 279 public-bounded cases from 320-335 and selects 0 systems.
- This is a target-specific failure, not a complete loss of the target-cap1
  mechanism.  The frozen public row selector remains the best stress policy in
  both new windows: 320-327 has 2 verified below-rho leaf cases with best
  0.664 rho, and 328-335 has 2 verified below-rho leaf cases with best
  0.568 rho.
- The positive split is `67.a1@9803`.  A narrow target-67 v2 predictor freezes
  the pre-320 public signature `[0,1,2,3]` at top_k 4.  On 272-319 calibration
  it selects 3 systems and all 3 are verified below rho.  On held-out 320-335
  it selects one system: transfer 330, salt 206, top_k 4, leaves
  `[0,1,2,3]`; it is verified below rho at 0.568 rho with rank 2 and relation
  count 2.
- Sage exact profiling of that selected transfer-330 row finds a preserving
  degree-1 factor `945*b + c + 952` over GF(9803).  The exact profile keeps the
  charged root-scan path below rho at 0.768 rho, with surface-only cost
  0.624 rho.
- Public replay through `945*b + c + 952` derives secret 255 on transfer 330.
  The measured oriented replay cost is 0.544 rho, with rank 2, relation count
  2, and one verified below-rho public rule.

Current boundary after 320-335:

The `22050` v1 branch must be split or retired until it reappears; it did not
select anything on 320-335.  The broader target-cap1 mechanism did survive on a
new target-specific branch: `67.a1@9803`, signature `[0,1,2,3]`, top_k 4.  This
branch has pre-320 calibration evidence and a held-out transfer-330 recovery,
including exact degree-1 FFE confirmation and public line replay below rho.
It is still not a full algorithmic speedup because the line `945*b + c + 952`
was extracted by exact factorization after row selection; the next missing
piece is a public line predictor or a charged line-confirmation argument that
continues to beat rho on the next `67.a1@9803` held-out windows.

Target-67 v2 failure on 336-351 and `22050` v1 reactivation:

- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_336_343_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_344_351_probe.json`
- Public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_336_343.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_344_351.json`
- Predictor artifacts:
  - `ecdlp_index_calculus_state/ffe_public_target_cap1_event_predictor_v2_target67_top4_validation_336_351.json`
  - `ecdlp_index_calculus_state/ffe_public_target_cap1_event_predictor_v1_regression_336_351.json`
- Exact-profile and public replay artifacts:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_leaf79_predictor_v1_regression_336_343.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line3093_predictor_v1_336_343.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_line3093_crosscheck_344_351.json`
- Target-67 v2 does not survive the next strict holdout: on 336-351 it reads
  350 public-bounded cases and selects 0 systems for the frozen
  `67.a1@9803`, top_k 4, `[0,1,2,3]` rule.
- The raw target-cap1 stress signal remains split by window.  On 336-343,
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0` is again the best row policy,
  with 6 verified below-rho leaf cases and best cost 0.37226277 rho.  On
  344-351, target-cap1 has 0 verified stress rows/leaves and target-cap3/global
  take over above rho.
- The original `22050.cf1@11731` v1 predictor naturally reactivates on
  336-343.  Across 336-351 it selects 5 systems, all on transfer 341; 2 are
  verified below rho after public selection.
- The best verified v1-selected case is transfer 341, salt 174, top_k 7,
  leaves `[8,34,56,90]`, at 0.37226277 rho with rank 2 and relation count 2.
  A second verified case uses leaves `[19,34,89,90]` at 0.40875912 rho.
- Sage exact profiling of the two verified transfer-341 rows finds a preserving
  degree-1 factor only for `[8,34,56,90]`: `3093*b + c + 5884` over GF(11731).
  Its charged root-scan path is 0.53284672 rho and its surface-only cost is
  0.42335766 rho.  The `[19,34,89,90]` row verifies at the public-bounded layer
  but has no preserving selected-root factor in this exact-profile test.
- Public replay through `3093*b + c + 5884` derives secret 5689 on transfer
  341 at 0.35036496 rho.  It verifies three public orientation rules
  (`all`, `candidate_pos_min`, and `term_shape:2+2`), all with rank 2 and
  relation count 2.
- Cross-window replay of `3093*b + c + 5884` on 344-351 activates 84 source
  cases but verifies 0 rules, so the line is not a broad replay artifact.

Current boundary after 336-351:

The target-67 v2 branch is now a local positive rather than a stable forward
predictor.  The older `22050` v1 branch, however, reactivated on the next
half-window with a new transfer-341 degree-1 line and a measured public replay
at 0.35036496 rho.  The campaign now has a more nuanced periodic/clustered
picture: target-cap1 windows alternate between `22050` v1, target-67 v2, and
holes.  The next useful step is not another single-branch claim; it is a
pre-window branch scheduler that predicts which target/signature family is
active before exact FFE line confirmation.

Public branch scheduler and unseen 352-367 validation:

- New scheduler:
  `tasks/ecdlp_index_calculus/ffe_public_branch_scheduler.py`
- Calibration artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_calibration_272_351.json`
- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_352_359_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_360_367_probe.json`
- Fresh public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_352_359.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_360_367.json`
- Scheduler validation artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_352_367.json`
- Exact-profile artifacts:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_branch_scheduler_selected_352_359.json`
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_branch_scheduler_selected_360_367.json`
- Public line replay artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_branch_scheduler_line6110_352_367.json`
- The scheduler freezes four branch definitions from pre-352 evidence and
  selects only from public fields: target, policy, row selector, top_k,
  low-term leaf signature, leaf selector, selected row/leaf counts, row salts,
  and public cost.  Verifier labels are summarized only after selection.
- Calibration on 272-351 reads 1847 cases and selects 20 public systems:
  16 selected systems are verifier-positive and below rho.  The strongest
  calibration case is transfer 341, leaves `[8,34,56,90]`, at 0.37226277 rho.
- On unseen 352-367, the scheduler reads 414 public-bounded cases and selects
  only three systems.  All three are verifier-positive below rho:
  - `22050` transfer 353, top_k 12, leaves `[8,56,90]`, salt 173,
    0.39416058 rho.
  - `22050` transfer 363, top_k 12, leaves `[8,56,90]`, salt 165,
    0.43065693 rho.
  - `67` transfer 359, top_k 4, leaves `[0,1,2,3]`, salt 204,
    0.64 rho.
- Exact Sage profiling confirms a nonvacuous preserving degree-1 factor for
  the transfer-353 `22050` system:
  `6110*b + c + 4058` over GF(11731).  Its exact surface-only charge is
  0.4379562 rho and its conservative root-scan charge is 0.51824818 rho.
- The transfer-363 `22050` system verifies at the row/leaf relation layer but
  has no preserving Sage factor in this exact-profile pass.  The target-67
  transfer-359 system similarly verifies at the row/leaf relation layer but
  has no preserving factor in this pass.
- Public replay through `6110*b + c + 4058` on transfer 353 derives secret
  5675 with rank 2 and relation count 2.  The best public orientation rule
  `salt_parity_even_35_odd_24` charges 48/137 = 0.35036496 rho.

Current boundary after 352-367:

The public branch scheduler validated on an unseen 16-transfer window: it did
not need verifier labels to choose the three systems, and every selected system
was below-rho verified.  One of those systems has the full mechanism chain:
public branch selection, exact FFE linear-factor confirmation, and public
orientation replay below rho.  The remaining gap is still the same honest one:
the degree-1 line is found by exact confirmation after public row selection.
The next useful step is either to build a public line predictor for the
branch-selected rows or to measure a line-confirmation charge that keeps the
full selected path below rho across multiple windows.

Forward 368-383 scheduler abstention:

- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_368_375_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_376_383_probe.json`
- Fresh public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_368_375.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_376_383.json`
- Scheduler validation artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_368_383.json`
- Both stress windows are holes for the frozen target-cap1 branches.  The best
  stress policy is `fixed_global_cap3_ow1_hw3_lw0_sw0_cw0_aw0` in both
  windows; target-cap1 has 0 verified stress rows/leaves.
- The public-bounded conversion emits 264 cases total: 152 from 368-375 and
  112 from 376-383.  None match the frozen single-row branch signatures under
  the preregistered target-cap1/target-67 scheduler.
- The unchanged scheduler abstains: 0 selected cases, 0 verifier labels, and
  no forced branch choice.
- The cheapest public-bounded cases in these windows are rank-0 leaf-90 triples
  at 0.94160584 rho, so there is no hidden below-rho derivation lost by the
  abstention under the current branch set.

Current boundary after 368-383:

The 368-383 result is a clean scheduler-control negative.  It supports the
branch-scheduler framing: the scheduler can validate on active windows and
abstain on holes.  The next positive work is either a fresh active-window
search beyond 383 or a new branch family for global/target-cap3 windows, but
the current global/target-cap3 evidence is above rho and rank-0 at the public
bounded layer.

Forward 384-399 scheduler failure and target-67 v3 discovery:

- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_384_391_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_392_399_probe.json`
- Fresh public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_384_391.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_392_399.json`
- Scheduler validation artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_384_399.json`
- Post-hoc target-67 exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer395_posthoc_392_399.json`
- Post-hoc target-67 public replay artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line3394_posthoc_392_399.json`
- Target-cap1 reactivates in the raw stress summaries after the 368-383 hole:
  384-391 has 4 verified below-rho stress-leaf labels, and 392-399 has 12.
- The frozen scheduler does not validate on 384-399.  It selects two `22050`
  systems on transfer 395, both salt 167:
  - `[8,34,56,90]`, top_k 7, total4 selector, 0.39416058 rho, rank 0.
  - `[8,56,90]`, top_k 7, total3 selector, 0.3649635 rho, rank 0.
- The actual verifier-positive target-cap1 single-row systems in 392-399 are
  on `67.a1@9803`, transfer 395, salt 206, with signatures such as
  `[1,5,6]`, `[5,6,10]`, and `[0,1,5,6]`.
- Post-hoc exact Sage profiling of the target-67 `[1,5,6]` row finds a
  preserving degree-1 factor:
  `3394*b + c + 711` over GF(9803).  Surface-only cost is 0.576 rho and
  root-scan cost is 0.768 rho.
- Public replay through `3394*b + c + 711` derives secret 6208 on transfer 395
  with rank 2, relation count 2, and best measured orientation cost
  56/125 = 0.448 rho.
- This is a post-hoc branch discovery, not a preregistered success.  The
  candidate v3 rule is `67.a1@9803`, top_k 7, target-cap1 row selector,
  total3 signature `[1,5,6]`.

Current boundary after 384-399:

The branch scheduler is useful but insufficient: it can abstain on holes, but
the old `22050` branches false-positive when a new target-67 signature becomes
active.  The next clean test is to add the target-67 v3 branch only as a
frozen candidate and validate it on 400-415.  If it fails, the scheduler needs
an activation/anti-rank0 filter before adding more branches.

Target-67 v3 forward check on 400-415:

- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_400_407_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_408_415_probe.json`
- Fresh public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_400_407.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_408_415.json`
- Augmented scheduler artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_400_415_augmented_v3.json`
- The augmented run keeps the four earlier branches and adds only:
  `67_v3_156|67.a1@9803|7|1,5,6|mode_cost_low_term_support_total3`.
- The 400-415 stress windows do not reactivate target-cap1.  Both halves are
  target-cap3/global dominated above rho; target-cap1 has 0 verified stress
  leaf labels.
- Public-bounded conversion emits 378 cases total.  The augmented scheduler
  abstains: 0 selected cases, 0 verifier labels, and no false-positive v3 row.

Current boundary after 400-415:

The target-67 v3 branch remains a post-hoc local discovery from 392-399, not a
validated forward predictor.  Its 400-415 abstention is still useful because it
does not overselect in a target-cap3/global window.  The scheduler story is now
three-part: active-window validation on 352-367, clean hole abstentions on
368-383 and 400-415, and a false-positive old-branch failure on 384-399 that
exposes the need for public activation or anti-rank0 filtering.

Public branch-family support and 416-447 forward checks:

- Scheduler update:
  `tasks/ecdlp_index_calculus/ffe_public_branch_scheduler.py`
- The scheduler now supports:
  - public branch precedence via
    `--suppress-branch-when-active suppressed|activator`
  - branch signatures of the form `contains:5,6`
  - an optional branch salt set as the eighth branch field
- Activation replay artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_activation_replay_392_399_v3_precedence.json`
- Broad replay artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_all_272_415_augmented_v3_precedence.json`
- Pair-family retrospective artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_400_431_pair56_s206_precise.json`
- Fresh 416-431 artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_416_423_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_424_431_probe.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_416_423.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_424_431.json`
  - `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_416_431_augmented_v3_no_precedence.json`
  - `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_416_431_augmented_v3_precedence.json`
- Fresh 424-431 exact/replay artifacts:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer427_posthoc_424_431.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line1119_posthoc_424_431.json`
- The branch-family syntax expresses a candidate that was hidden by exact
  signatures: `67_pair56_s206|67.a1@9803|7|contains:5,6|mode_cost_low_term_support_total3|||206`.
- Historical scan of that family has three hits:
  - transfer 267, `[1,5,6]`, salt 206, rank 0, not verified.
  - transfer 395, `[1,5,6]`, salt 206, verified below rho.
  - transfer 427, `[4,5,6]`, salt 206, verified below rho.
- Because the pair-family was recognized after seeing 424-431, the transfer
  427 result is a retrospective branch-family confirmation, not a clean
  preregistered validation.
- Exact profiling of the transfer-427 `[4,5,6]` row finds a preserving
  degree-1 line `1119*b + c + 7180` over GF(9803).  Surface-only cost is
  0.624 rho and root-scan cost is 0.768 rho.
- Public replay through `1119*b + c + 7180` derives secret 4698 with rank 2,
  relation count 2, and measured orientation cost 70/125 = 0.56 rho.
- The 416-431 augmented scheduler with exact v3 abstains on both windows:
  0 selected systems from 392 public-bounded cases.
- The pair-family scheduler over 400-431 selects the transfer-427 row and it is
  verified below rho, while abstaining on 400-415 and 416-423.

Forward 432-447 frozen pair-family check:

- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_432_439_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_440_447_probe.json`
- Fresh public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_432_439.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_440_447.json`
- Scheduler artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_432_447_pair56_s206_precise.json`
- Exact-profile artifacts:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_v2_transfer437_432_439.json`
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer437_neighbor_432_439.json`
- The frozen pair-family branch does not repeat on 432-447.  Instead, the old
  target-67 v2 branch reactivates on transfer 437: `[0,1,2,3]`, salt 208,
  top_k 4, at 0.656 rho with rank 2 and relation count 2.
- Exact Sage profiling of the selected `[0,1,2,3]` profile finds no preserving
  resultant factor.  Neighboring verified total3 profiles `[1,2,3]` and
  `[2,3,6]` also have no preserving factor in this pass.

Current boundary after 432-447:

The branch scheduler has another forward row-selector success on target-67 v2,
but not every verified row has a usable linear-factor compression.  The pair
`{5,6}`/salt206 family is now the best line-backed target-67 mechanism, but it
still needs a clean future validation because the family itself was recognized
after inspecting the 424-431 row.  The next experiment should freeze both
target-67 v2 and the pair-family branch, validate on 448-463, and exact-profile
only selected rows so line-backed and relation-only wins remain separated.

Fresh 448-463 frozen scheduler and line-family activation check:

- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_448_455_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_456_463_probe.json`
- Fresh public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_448_455.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_456_463.json`
- Frozen scheduler artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_448_463_pair56_s206_precise.json`
- Exact-profile/replay artifacts:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_branch_scheduler_selected_448_455.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8741_forward_448_455.json`
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_pair56_salt204_transfer449_posthoc_448_455.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line3394_forward_448_455.json`
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_pair56_salt208_transfer458_control_456_463.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line1119_control_456_463.json`
  - `ecdlp_index_calculus_state/ffe_public_line_family_activation_miner_target67_known_lines_328_463.json`
- The 448-455 stress window reactivates target-cap1: 8 verified below-rho
  stress-leaf rows, best 0.64 rho.  The 456-463 stress window is a hole for
  target-cap1 leaf recovery: 0 target-cap1 stress-leaf wins, with the best
  policy global/target-cap3 and above rho at 1.336.
- Public-bounded conversion emits 217 cases for 448-455 and 220 for 456-463.
- The frozen scheduler over both sources selects four rows, all from 448-455:
  - three old `22050` rows at transfer 451, salt 175; all are below-rho and
    rank/relation-positive but not public-key verified.
  - one target-67 v2 row at transfer 449, leaves `[0,1,2,3]`, salt 204,
    0.648 rho, rank 4, relation count 4, public-key verified.
- The frozen `67_pair56_s206` branch does not fire on 448-463.
- Exact profiling of the four scheduler-selected rows materializes all four
  profiles.  It finds preserving below-rho root-scan factors for three
  profiles.  The verified target-67 v2 row has line
  `8741*b + c + 499` over GF(9803), surface-only cost 0.704 rho, and root-scan
  cost 0.872 rho.
- Public replay through `8741*b + c + 499` activates one case but verifies no
  rule.  The factor selects one zero leaf, emits at most rank 1 under existing
  public orientation rules, and does not derive the key.
- A post-hoc salt-relaxed pair-family diagnostic finds a fresh verifier-positive
  target-67 row at transfer 449: leaves `[2,5,6]`, salt 204, top_k 7,
  0.704 rho.  Exact profiling shows it reuses the old transfer-395 line
  `3394*b + c + 711`; surface-only cost is 0.752 rho and root-scan cost is
  0.928 rho.
- Public replay through `3394*b + c + 711` on transfer 449 activates one case
  but verifies no rule.  The factor selects leaf 5 and emits no verified
  relation under the current public orientation rules.
- A 456-463 control row, transfer 458, leaves `[3,5,6]`, salt 208, is not
  public-key verified even though it is cheap at 0.48 rho.  Exact profiling
  shows it reuses the old transfer-427 line `1119*b + c + 7180`; surface-only
  cost is 0.528 rho and root-scan cost is 0.744 rho.  Public replay through
  that line also verifies no rule.
- The line-family activation miner compares target-67 line successes
  (`945`, `3394`, `1119`) against fresh line-present failures (`8741`,
  `3394`, `1119`).  It keeps public x-match event features separate from
  verifier labels.  In this small set, successful replays have 6, 8, or 9
  xmatches and label rank 2; failures have 3 or 6 xmatches and label rank 0 or
  1.  This is not yet a rule, but it is the right feature surface for the next
  preregistered activation test.

Current boundary after 448-463:

The frozen branch scheduler still has row-selection value: it found a fresh
target-67 v2 verifier-positive row on 448-455 while abstaining on 456-463.  But
fresh line-present cases show that degree-1 factor recurrence is not enough.
The target-67 line families `3394` and `1119` recur beyond their discovery
windows, yet the public orientation layer can fail to harvest enough relations.
The next clean step is to mine a public activation rule over factor-gated
x-match event shapes, freeze it using the known target-67 successes/failures,
and validate on 464-479 without adding new post-hoc branch signatures.

Fresh 464-479 line-family activation validation:

- Rule-miner script:
  `tasks/ecdlp_index_calculus/ffe_public_line_family_activation_rule_miner.py`
- Frozen rule artifact:
  `ecdlp_index_calculus_state/ffe_public_line_family_activation_rule_miner_target67_known_lines_328_463.json`
- Fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_464_471_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_472_479_probe.json`
- Fresh public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_464_471.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_472_479.json`
- Scheduler/exact/replay artifacts:
  - `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_464_479_pair56_s206_precise.json`
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_branch_scheduler_selected_472_479.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line9481_activation_472_479.json`
- The rule miner reads the line-family activation comparison artifact and
  enumerates public `activate:` rules over the existing DSL.  The frozen best
  rule is:
  `activate:top_k=7&xmatch_count>=6|xmatch_count>=8`.
  On the known target-67 line-family set it has 3 true positives, 0 false
  positives, and 0 false negatives.  This rule was frozen before inspecting
  the 464-479 validation artifacts.
- Both fresh validation halves are active target-cap1 windows:
  - 464-471: 8 verified below-rho stress-leaf rows, best 0.624 rho.
  - 472-479: 6 verified below-rho stress-leaf rows, best 0.4379562 rho.
- Public-bounded conversion emits 204 cases for 464-471 and 283 cases for
  472-479.
- The unchanged frozen branch scheduler reads 487 cases and selects exactly one
  row: `67_v2_0123`, transfer 473, leaves `[0,1,2,3]`, salt 206, 0.568 rho,
  rank 2, relation count 2, public-key verified.
- Exact profiling of that selected row finds a preserving degree-1 line
  `9481*b + c + 5654` over GF(9803).  Surface-only cost is 0.624 rho and
  conservative root-scan cost is 0.776 rho.
- Public replay through `9481*b + c + 5654`, gated by the frozen activation
  rule, activates the case.  The rule's top-k 4 clause requires at least 8
  public xmatches; the selected row has exactly 8.
- The replay derives secret 7675.  The best verified public orientation rule is
  `candidate_pos_min`, rank 2, relation count 2, measured cost
  56/125 = 0.448 rho.  The `all` rule also verifies at 68/125 = 0.544 rho.

Current boundary after 464-479:

The target-67 line-family activation hypothesis has one clean fresh validation:
public branch selection, exact FFE degree-1 line confirmation, frozen public
activation, and measured public orientation replay all stay below rho and
derive the secret.  The honest remaining gap is line prediction or line
confirmation charge: the line `9481*b + c + 5654` was found by exact
factorization after public row selection, not predicted directly.  The next
experiment should keep the same activation rule frozen and validate again on
480-495, while measuring how often exact line confirmation remains below rho
for the scheduler-selected target-67 rows.

Fresh 480-495 frozen-rule validation:

- New stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_480_487_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_488_495_probe.json`
- New public-bounded artifacts:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_480_487.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_488_495.json`
- Scheduler and bucket-report artifacts:
  - `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_480_495_pair56_s206_precise.json`
  - `ecdlp_index_calculus_state/ffe_public_branch_scheduler_window_report_480_495_pair56_s206_precise.json`
  - `tasks/ecdlp_index_calculus/ffe_public_branch_scheduler_window_report.py`
- The activation rule remained frozen:
  `activate:top_k=7&xmatch_count>=6|xmatch_count>=8`.
- Stress window 480-487 is a target-cap1 hole: the fixed target-cap1 policy
  has zero verified rows and zero verified leaves.  The best policy is
  target-cap3, with 28 verified leaves but no below-rho leaves; best leaf cost
  is 1.08029197 rho.
- Stress window 488-495 has four target-cap1 row-level verified cases, but no
  below-rho leaf recoveries.  The best policy is global-cap3, with 16 verified
  leaves, no below-rho leaves, and best leaf cost 1.01459854 rho.
- Public-bounded conversion emits 160 cases for 480-487 and 198 cases for
  488-495.
- The unchanged frozen branch scheduler reads 358 bounded cases and selects
  three cases, all on old 22050 branches in 488-495:
  - `22050_v1_8345690`, transfer 492, top-k 7, leaves `[8,34,56,90]`,
    salt 173, public cost 0.40875912 rho, verifier rank 0.
  - `22050_v1_85690`, transfer 493, top-k 12, leaves `[8,56,90]`,
    salt 174, public cost 0.32846715 rho, verifier rank 0.
  - `22050_v1_85690`, transfer 492, top-k 7, leaves `[8,56,90]`,
    salt 173, public cost 0.40145985 rho, verifier rank 0.
- The scheduler selects no target-67 cases and no verifier-backed below-rho
  cases.  This is a branch-reach failure window, not a replay-rule test:
  exact target-67 line profiling was not run because there was no selected
  target-67 row to profile.
- The bucket report records 150 public target-67 bounded candidates, but only
  four target-cap1 below-rho candidates.  All four are verifier-negative:
  transfer 494, salt 208, top-k 16, leaves `[3,9,17]` at 0.424 rho with rank
  0, and leaves `[3,9,10,17]` at 0.52 rho with rank 1.

Current boundary after 480-495:

The 480-495 run is a clean falsification of branch reach for this exact frozen
scheduler window.  It does not falsify the x-match activation rule, because no
target-67 selected row reached the exact-line/replay stage.  The rule should
remain frozen; the next useful work is not to add another signature, but to
separate the missing pieces: public target-67 branch reach, public prediction
of the degree-1 FFE line, and a charged exact-line confirmation model.

Target-67 line-stage confirmation audit:

- New audit script:
  `tasks/ecdlp_index_calculus/ffe_target67_line_stage_audit.py`
- New audit artifact:
  `ecdlp_index_calculus_state/ffe_target67_line_stage_audit_328_495.json`
- Inputs:
  - target-67 exact-profile artifacts from transfers 330, 395, 427, 437,
    449, 458, and 473.
  - target-67 public line replay artifacts for lines `945`, `3394`, `1119`,
    `8741`, and `9481`.
  - the 480-495 branch-reach bucket report.
- The audit separates three charges:
  public row selection, exact degree-1 line confirmation/root scan, and public
  orientation replay.
- It finds 12 target-67 exact-profile surfaces.  Eleven are row-verified.
- Seven surfaces have preserving degree-1 FFE lines, and all seven preserving
  line root scans are individually below rho.  Root-scan costs range from
  0.744 rho to 0.928 rho, with mean 0.80342857 rho.
- Four line-present rows replay successfully and recover the secret:
  - transfer 330, salt 206, leaves `[0,1,2,3]`, line
    `945*b + c + 952`, replay secret 255, line+replay additive cost
    1.312 rho.
  - transfer 395, salt 206, leaves `[1,5,6]`, line
    `3394*b + c + 711`, replay secret 6208, line+replay additive cost
    1.216 rho.
  - transfer 427, salt 206, leaves `[4,5,6]`, line
    `1119*b + c + 7180`, replay secret 4698, line+replay additive cost
    1.328 rho.
  - transfer 473, salt 206, leaves `[0,1,2,3]`, line
    `9481*b + c + 5654`, replay secret 7675, line+replay additive cost
    1.224 rho.
- Three line-present rows fail replay:
  - transfer 449, salt 204, leaves `[0,1,2,3]`, line
    `8741*b + c + 499`.
  - transfer 449, salt 204, leaves `[2,5,6]`, line
    `3394*b + c + 711`.
  - transfer 458, salt 208, leaves `[3,5,6]`, line
    `1119*b + c + 7180`.
- Five row-verified surfaces have no preserving line, including the transfer
  437 target-67 v2 hit.
- Preserving factor indices range from 0 to 22:
  `[0,1,2,7,8,22]`.  The preserving line is not consistently the first,
  last, or a fixed low-index factor.
- Critically, no replay-success row stays below rho under the naive additive
  charge `line confirmation + replay`; the minimum line+replay cost is
  1.216 rho, and the minimum row+line+replay cost is 1.744 rho.
- The audit computes a concrete amortization target: every replay-success row
  would have `line confirmation / n + replay < rho` at `n = 2`.  If the public
  row-selection cost is also added, the only observed finite break-even is
  `n = 33`; the other replay-success rows already have `row + replay >= rho`
  before line confirmation is charged.

Current boundary after the line-stage audit:

The target-67 route now has repeated stagewise positives, but not an honest
end-to-end below-rho algorithm.  Public row selection, preserving degree-1 line
confirmation, and orientation replay can each be below rho in isolation.  When
exact line confirmation is charged additively, the recovery no longer beats
rho.  The next algorithmic step must therefore either predict the line before
exact factorization or amortize the line-confirmation/root-scan charge across
at least two replay-eligible rows, with row-selection cost kept explicitly
separate.  Adding another row signature would not address this specific gap.

Target-67 repeated-line amortization probe:

- New probe script:
  `tasks/ecdlp_index_calculus/ffe_target67_repeated_line_amortization_probe.py`
- New probe artifact:
  `ecdlp_index_calculus_state/ffe_target67_repeated_line_amortization_probe_328_495.json`
- The probe groups preserving target-67 degree-1 lines from the line-stage
  audit and keeps two ledgers:
  - strict `success_only_amortized_rows`, which can support an ECDLP recovery
    amortization claim.
  - diagnostic `line_present_amortized_rows`, which shares line confirmation
    over failed rows too and is mechanism evidence only.
- It finds five public line families, two repeated line-present families, and
  zero repeated line families with the required two replay-success rows.
- The repeated line-present families are:
  - `1119*b + 1*c + 7180 mod 9803`: transfer 427 succeeds and recovers secret
    4698, while transfer 458 is line-present but replay-failing.
  - `3394*b + 1*c + 711 mod 9803`: transfer 395 succeeds and recovers secret
    6208, while transfer 449 is line-present but replay-failing.
- Strict success-only accounting finds zero amortized below-rho line families.
  The two repeated line-present families fall below rho only if the failed row
  also shares the line-confirmation charge, so this is not an ECDLP speedup.

Fixed-line future checks for the repeated target-67 lines:

- New fixed-line replay artifacts:
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line3394_fixed_480_487.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line3394_fixed_488_495.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line1119_fixed_480_487.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line1119_fixed_488_495.json`
- The frozen activation rule remained
  `activate:top_k=7&xmatch_count>=6|xmatch_count>=8`.
- Source-case counts were 4, 5, 4, and 5 respectively.  All four artifacts have
  `activated_case_count = 0` and `verified_rule_count = 0`.

Current boundary after repeated-line amortization:

The target-67 line reuse mechanism is real enough to steer the next search, but
the strict `n = 2` amortization target is not yet observed.  The two repeated
lines `1119` and `3394` each pair one replay-success row with one replay-failure
row, and fixed-line replay for those exact lines does not extend to the 480-495
future window.  The next useful experiment is therefore either to find future
rows where the same public line has two replay-successes, or to learn a public
degree-1 line predictor that selects fresh lines before exact factorization.

Fresh 496-511 target-67 top-16 branch and line probe:

- New fresh stress artifacts:
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_496_503_probe.json`
  - `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_504_511_probe.json`
- New public-bounded sources:
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_496_503.json`
  - `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_504_511.json`
- The 496-503 stress window is not a target-cap1 hole: the frozen target-cap1
  row selector has 4 verified below-rho stress-leaf cases, best 0.704 rho.
  The public-bounded source emits 194 cases and 47 verifier-positive labels.
- The old frozen scheduler abstains on 496-503, but a post-hoc target-67
  top-16 branch selects two verifier-backed below-rho transfer-503 cases:
  - `67_top16_71013_s204`: leaves `[7,10,13]`, salt 204, 0.704 rho,
    rank 3 with 3 relations.
  - `67_top16_671013_s204`: leaves `[6,7,10,13]`, salt 204, 0.712 rho,
    rank 3 with 3 relations.
- New exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer503_top16_496_503.json`
- Both transfer-503 profiles have the same preserving degree-1 line:
  `8022*b + c + 5592 mod 9803`, factor index 20.  Root-scan charges are
  0.928 rho for `[7,10,13]` and 0.936 rho for `[6,7,10,13]`; surface FFE
  charges are 0.752 and 0.768 rho.  The full remainders remain over rho.
- New replay artifacts:
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8022_transfer503_total3_496_503.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8022_transfer503_total4_496_503.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8022_transfer503_total3_all_496_503.json`
  - `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8022_transfer503_total4_all_496_503.json`
- The frozen activation rule
  `activate:top_k=7&xmatch_count>=6|xmatch_count>=8` activates zero cases for
  line `8022`.  Diagnostic `activation-rule all` activates one case per
  profile, but still verifies zero orientation rules.  The factor-zero leaf is
  leaf 13, with `xmatch_count = 4`; this is an orientation/replay failure, not
  just an activation-threshold miss.
- Updated line-stage artifacts:
  - `ecdlp_index_calculus_state/ffe_target67_line_stage_audit_328_511.json`
  - `ecdlp_index_calculus_state/ffe_target67_repeated_line_amortization_probe_328_511.json`
- The 328-511 line-stage audit now has 14 target-67 surfaces, 13 row-verified
  surfaces, 9 preserving lines, 4 replay successes, and 5 replay failures.
  Every preserving line root scan is still below rho, but no replay-success row
  beats rho under additive line+replay accounting.
- The repeated-line amortization probe over 328-511 now sees six line families
  and three repeated line-present families: `1119`, `3394`, and the new `8022`.
  There are still zero repeated lines with two replay-success rows, and zero
  strict success-only amortized below-rho families.  The new `8022` line repeats
  only because two leaf profiles from the same transfer-503 row share it; both
  are replay failures.
- The post-hoc top-16 target-67 branch was frozen and tested on 504-511:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_holdout_target67_top16_504_511.json`
  abstains, selecting zero cases.  Do not promote the top-16 target-67 branch.
- As a side observation, the old non-target-67 scheduler does validate on
  504-511:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_504_511_pair56_s206_precise.json`
  selects three `22050` cases, all verifier-backed below rho, best 0.39416058
  rho.  This is useful scheduler evidence but not a target-67 line-prediction
  win.

Current boundary after 496-511:

The 496-503 window adds a new below-rho target-67 row-selection and preserving
line-confirmation example, but it also strengthens the honest blocker: public
line discovery alone is insufficient when orientation replay does not derive
the secret.  The top-16 `[7,10,13]` target-67 signature is post-hoc and fails
the immediate 504-511 holdout.  The next target-67 work should mine a public
orientation predictor from success-vs-failure x-match features, or find two
independent replay-success rows for the same line; it should not widen the
branch scheduler around transfer 503.

Target-67 public orientation rule probe:

- Mined public x-match orientation rules from the four known line-present
  target-67 replay successes:
  `ecdlp_index_calculus_state/ffe_public_xmatch_orientation_rule_miner_target67_successes_328_511.json`
- The miner's frozen rule is `where:candidate_eq_scheduled`.  It looked
  attractive in-sample because it selected one valid relation from every known
  success with only five invalid relations total.
- Replayed that frozen rule with `activation-rule all` on all nine known
  line-present cases: the four replay successes and five replay failures from
  the 328-511 line-stage audit.  New replay artifacts are the
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_*where_candidate_eq_scheduled*.json`
  files.
- Result: all 9 cases activated, but 0 of 9 verified a public key and 0 of 9
  beat rho.  This includes 0 of the 4 original replay successes.
- Added the feature joiner:
  `tasks/ecdlp_index_calculus/ffe_target67_orientation_feature_probe.py`
  and output:
  `ecdlp_index_calculus_state/ffe_target67_orientation_feature_probe_328_511.json`
- The feature probe shows why the rule fails: every replay-success row has two
  valid x-matches total, while `where:candidate_eq_scheduled` keeps only one
  valid relation and reaches rank 1, not rank 2.  The failures have at most one
  valid x-match under the same public slice.  The next orientation search must
  be pair/rank-aware; a one-relation public orientation rule is not enough.
- Added a rank-aware rule miner:
  `tasks/ecdlp_index_calculus/ffe_public_xmatch_orientation_rank_rule_miner.py`
  and output:
  `ecdlp_index_calculus_state/ffe_public_xmatch_orientation_rank_rule_miner_target67_328_511.json`
- The rank-aware miner evaluates already-replayed public rules by public-key
  verification, not by single-relation hits.  Its retrospective best rule is
  `all`: 4 true positives, 0 false positives, 0 false negatives over the nine
  line-present cases, with verified ops 0.512, 0.544, 0.544, and 0.560 rho.
  `candidate_pos_min` recovers only the two 0.448-rho successes; the mined
  `where:candidate_eq_scheduled` rule recovers none.
- Added a line-success activation miner for the rank-aware baseline:
  `tasks/ecdlp_index_calculus/ffe_target67_line_success_activation_rule_miner.py`
  and output:
  `ecdlp_index_calculus_state/ffe_target67_line_success_activation_rule_miner_328_511.json`
- Training on line-present cases through transfer 463 and validating only on
  later line-present cases gives the same public activation rule as before:
  `activate:top_k=7&xmatch_count>=6|xmatch_count>=8`.  Training score is
  3 true positives, 0 false positives, 0 false negatives.  Later-transfer
  validation score is 1 true positive, 0 false positives, 0 false negatives:
  it accepts the transfer-473 success and rejects both transfer-503 line-8022
  failures.

Current boundary after orientation-rule probing:

The target-67 mechanism still has real stagewise structure: below-rho public
row signatures, below-rho preserving degree-1 line root scans, and four public
line-gated replay recoveries.  The new negative result rules out the simple
`candidate_eq_scheduled` orientation shortcut.  The rank-aware baseline says
that keeping all x-matches after a public line gate is currently the honest
orientation rule to freeze.  The old activation rule also survives the transfer
503 negative when judged as a line-present success filter.  The next credible
path is therefore public line prediction or line-confirmation amortization,
then applying this frozen activation-plus-all-replay package to fresh
line-present holdouts.

Target-67 public line prediction audit:

- Added a public feature dictionary audit:
  `tasks/ecdlp_index_calculus/ffe_target67_public_line_prediction_audit.py`
  and output:
  `ecdlp_index_calculus_state/ffe_target67_public_line_prediction_audit_328_511.json`
- The audit trains public feature atoms from preserving-line cases and asks
  whether later preserving lines can be uniquely predicted before exact FFE
  factorization.  Feature atoms include exact leaf signatures, top-k, leaf
  counts, x-match counts, scheduled-trial counts, and public leaf subsets.
- Across cutoffs 431, 463, and 495, no unique public feature dictionary
  predicts any later preserving line.  Summary counts are all zero for unique
  line matches, including replay-success matches.
- With cutoff 431, the repeated public pair `{5,6}` is diagnostic but
  ambiguous: it contains both `3394*b + c + 711` and
  `1119*b + c + 7180`.  It contains the true line for the later transfer-449
  and transfer-458 replay failures, but it cannot choose a single line and it
  does not produce a recovery.
- Exact signature reuse is actively misleading in this window: the public
  signature `[0,1,2,3]` maps from line `945*b + c + 952` in training to
  `8741*b + c + 499` on the transfer-449 failure and
  `9481*b + c + 5654` on the transfer-473 success.
- The transfer-503 line `8022*b + c + 5592` has no prior public dictionary
  support and remains a post-hoc line-confirmation example.

Current boundary after line-prediction probing:

The activation-plus-all replay package is now the best frozen target-67
line-present package, but the line itself is not predictable from simple public
feature dictionaries.  The next experiment needs a stronger line predictor
that uses actual FFE/summation-polynomial residual structure, or an
amortization route where exact line confirmation is shared across at least two
replay-success rows for the same line.

Target-67 residual factor-cloud audit:

- Added a post-factorization residual factor-cloud audit:
  `tasks/ecdlp_index_calculus/ffe_target67_residual_factor_cloud_audit.py`
  and output:
  `ecdlp_index_calculus_state/ffe_target67_residual_factor_cloud_audit_328_511.json`
- The audit asks whether the preserving degree-1 factor is obvious inside the
  Sage resultant factor cloud by public heuristics such as factor order,
  coefficient size, centered coefficient size, or coefficient norm.
- It covers all 14 line-stage surfaces and all 9 line-present cases.  The
  preserving factor indices are spread across
  `[0,1,2,7,8,20,20,22,22]`; replay-success rows alone use `[0,2,7,22]`.
- The best simple factor-cloud heuristic is `min_constant`, but it hits only
  2 of 9 line-present cases and only 1 of 4 replay-success cases.  `first`,
  `last`, `min_b`, `max_b`, and norm-based choices all hit at most one
  replay-success case.

Current boundary after residual factor-cloud probing:

The preserving line is not hidden behind a trivial factor order or coefficient
ranking rule, even after exact factorization.  This strengthens the next
obligation: the line predictor needs richer algebraic residual features before
full factorization, for example quotient/remainder coefficient fingerprints,
root-locus residues, or transfer-stable low-degree substructures.

Target-67 axis-root line-lift audit:

- Added a pre-bivariate-factorization line-lift audit:
  `tasks/ecdlp_index_calculus/ffe_target67_axis_root_line_lift_audit.py`
  and output:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_line_lift_audit_328_511.json`
- Mechanism: specialize the resultant on the c-axis to get roots
  `R(0,c)=0`.  For each axis root `c=-k` and selected leaf monic point
  `(b0,c0)`, lift the line slope as `m=-(c0+k)/b0`, then test whether
  substituting `c=-m*b-k` makes the resultant identically zero.  This happens
  before full bivariate factorization.
- The audit rematerialized all 14 target-67 line-stage surfaces from live
  AutoLab state.  Candidate-line counts were
  `[1,1,1,1,1,2,2,2,2,2,3,3,3,5]`, mean 2.071429 and max 5.
- It recovered all 9 known preserving lines and all 4 replay-success
  preserving lines.  On the 4 replay-success surfaces the lifted candidate set
  is singleton each time: lines `945`, `3394`, `1119`, and `9481`.
- The five no-preserving-line surfaces still emit candidate lines, so the lift
  is a line-enumeration stage, not a complete recovery filter.  It must be
  combined with the frozen activation-plus-all replay package and costed on a
  fresh holdout.

Current boundary after axis-root probing:

This is the strongest target-67 line-prediction lead so far: the preserving
line can be recovered before full bivariate Sage factorization by axis-root
specialization plus selected-leaf line lifts, and replay-success cases have a
single lifted line.  It is still retrospective and not yet costed below rho;
the next test is to freeze this lift with
`activate:top_k=7&xmatch_count>=6|xmatch_count>=8` plus all-xmatch replay on a
fresh target-67 line-present holdout, while charging axis-root enumeration and
line-substitution tests explicitly.

Target-67 axis-root lift cost audit:

- Added a proxy cost ledger:
  `tasks/ecdlp_index_calculus/ffe_target67_axis_root_lift_cost_audit.py`
  and output:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_lift_cost_audit_328_511.json`
- Cost model: `line_lift_ops = axis_root_count + candidate_line_test_count`.
  The candidate tests are selected-leaf slope lifts plus line-substitution
  divisibility checks.  This is a root/test-count proxy, not an optimized
  polynomial-operation benchmark.
- The ledger uses generic rho 125 for target 67.  Across the nine
  line-present cases, line-lift cost values are
  `[0.672,0.8,0.8,0.864,0.88,0.92,0.96,1.0,1.0]` rho.
- On the four replay-success cases, the axis-root lift alone is below rho in
  every case, with costs `[0.672,0.864,0.88,0.92]` rho, and all four candidate
  line sets are singleton.
- Per-row `line_lift + replay` is not below rho for any replay-success case:
  the four combined costs are 1.232, 1.312, 1.368, and 1.424 rho.
- With strict `n=2` reuse, all four replay-success cases fall below rho under
  the proxy lift+replay ledger.  The amortized costs are 0.88, 0.896, 0.908,
  and 0.984 rho; each case has break-even reuse count 2.
- This does not include row-selection cost and does not claim brute-force
  c-axis root scanning over the field is below rho.  A real implementation
  should charge univariate root finding/factorization and line-substitution
  tests explicitly.

Current boundary after axis-root costing:

The axis-root lift is now a plausible pre-factor line enumerator rather than a
complete speedup claim.  It beats rho alone on every known replay-success
target-67 surface under the simple root/test-count proxy, but the full
per-row lift+replay package still needs either fresh singleton-candidate
success or at least two strict replay-success rows sharing the charged lift.
The next run should freeze three pieces together: the axis-root lift,
`activate:top_k=7&xmatch_count>=6|xmatch_count>=8`, and all-xmatch replay.

Fresh 512-519 target-67 holdout and line-945 diagnostic:

- Generated the next total3/total4 fresh stress window:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_512_519_probe.json`
- Public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_512_519.json`
- Frozen branch scheduler:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_512_519_pair56_s206_precise.json`
- Window report:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_window_report_512_519_pair56_s206_precise.json`
- The frozen target-cap1 row selector had no fresh 512-519 reach:
  target-cap1 selected 0 public target-67 candidates, verified 0, and had 0
  below-rho cases.  The frozen multi-branch scheduler abstained cleanly with
  zero selected cases and zero old-branch false positives.
- The best stress policy in this window is target-cap3, not target-cap1:
  `fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0` has 24 verified stress-leaf
  cases, but none below rho; best stress-leaf cost is 1.01459854 rho.
- The public bounded source still contains 112 target-67 candidates.  The best
  target-67 target-cap3 candidate is transfer 517/top-k 4 at 1.224 rho, but it
  is rank 0 and verifier-negative.

Fresh transfer-514 exact-profile diagnostic:

- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer514_targetcap3_512_519.json`
- Selected the verifier-positive, over-rho target-cap3 transfer-514 case:
  target 67, top-k 4, leaf selector `mode_cost_low_term_support_total3`, rows
  salt208/salt203/salt207, selected leaf 1.  Source public cost is 1.336 rho.
- Sage exact-profile factorization materialized all three row/leaf profiles.
  All three have nontrivial degree-1 preserving factors with below-rho
  factor-root-scan costs: 0.432, 0.656, and 0.664 rho.
- Two rows, salt203 and salt207, have a selected root pair on the preserving
  factor.  The salt208 factor misses the selected leaf, so it is mechanism
  evidence but not a relation-bearing selected-root row.
- Added a general exact-profile axis-root probe:
  `tasks/ecdlp_index_calculus/ffe_target67_axis_root_exact_profile_probe.py`
  and output:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer514_targetcap3_512_519.json`
- Axis-root line lift on those three fresh exact profiles emits candidate-line
  counts `[0,1,1]`.  On the two selected-root-positive rows, the candidate set
  is singleton and the same line appears both times:
  `945*b + 1*c + 952`.
- Fresh line-lift proxy costs are 0.288, 0.336, and 0.352 rho.  The zero-line
  row is exactly the row whose preserving factor missed the selected leaf.
- Replayed line 945 on the transfer-514 target-cap3 public source:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line945_transfer514_targetcap3_512_519.json`
- The line-gated replay derives secret 7260 with rank 2 under public
  `candidate_pos_min`, selecting two x-matches, but it is not below rho:
  measured cost is 159/125 = 1.272 rho.  The all-xmatch rule also verifies at
  1.336 rho.

Current boundary after 512-519:

This is a fresh structural positive and a speedup negative.  The axis-root
mechanism generalized to a new transfer window and rediscovered the old line
945 as a singleton on two relation-bearing selected-root surfaces, and public
line replay derives the key.  However, the row package is target-cap3 and
over-rho, the frozen target-cap1 scheduler has no activation opportunity, and
line replay remains over rho.  The next target-67 blocker is not line recovery
alone; it is public below-rho branch reach or strict amortization of the
line-lift plus replay costs across relation-bearing rows.

Fresh 520-527 branch-reach and repeated-line diagnostic:

- Generated the next total3/total4 fresh stress window:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_520_527_probe.json`
- Public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_520_527.json`
- Frozen branch scheduler:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_520_527_pair56_s206_precise.json`
- Window report:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_window_report_520_527_pair56_s206_precise.json`
- The stress window is still a speedup negative.  The best stress policy is
  target-cap3 with 16 verified leaf cases, zero below rho, and best leaf cost
  1.03649635 rho.  Target-cap1 has 2 verified stress rows and 1 below-rho row,
  but zero verified leaf packages.
- The public bounded source contains 244 cases and 124 target-67 candidates.
  Unlike 512-519, target-cap1 reaches target 67: it emits 4 below-rho
  target-cap1 target-67 public candidates, all at transfer 520/salt208/top-k
  16.  All four are verifier-negative with rank 0.
- The frozen multi-branch scheduler abstains with zero selected cases.  The
  window report therefore refines the blocker from branch reach to
  rank-positive branch reach: target-cap1 can now reach cheaply, but not with a
  usable relation system in this window.

Fresh transfer-520 target-cap1 rank-0 exact-profile diagnostic:

- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer520_targetcap1_rank0_520_527.json`
- The cheap target-cap1 case is transfer 520, salt208, top-k 16, leaves
  `[4,12,17]`, public cost 0.496 rho, rank 0, verifier-negative.
- Sage exact-profile factorization still finds a nontrivial preserving
  degree-1 factor with below-rho root scan cost 0.704 rho and surface cost
  0.544 rho.  Only one of the three selected leaves contributes a selected root
  pair; two selected leaves are missed.
- Axis-root exact-profile output:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer520_targetcap1_rank0_520_527.json`
- Axis-root line lift predicts a singleton line
  `1966*b + 1*c + 2774` at 0.736 rho, but the line only covers selected leaf
  17.
- Public replay through line 1966:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line1966_transfer520_targetcap1_rank0_520_527.json`
- The line activates the source case but verifies zero rules and derives no
  key.  This is a clean rank/leaf-coherence negative, not a line-discovery
  failure.

Fresh transfer-527 target-cap3 repeated-line diagnostic:

- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer527_targetcap3_520_527.json`
- The verifier-positive over-rho case is transfer 527, top-k 7, rows
  salt202/salt204/salt208, leaf 5.  Source public cost is 1.368 rho.
- All three exact profiles have below-rho degree-1 preserving factor root
  scans.  The two relation-bearing rows, salt202 and salt204, have selected
  root pairs; salt208 misses the selected leaf.
- Axis-root exact-profile output:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer527_targetcap3_520_527.json`
- Axis-root lift emits candidate-line counts `[1,1,0]` by row and recovers the
  already-known line `3394*b + 1*c + 711` as a singleton on the two
  selected-root-positive rows.  Lift costs are 0.352, 0.336, and 0.384 rho.
- Public replay through line 3394:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line3394_transfer527_targetcap3_520_527.json`
- The line-gated replay derives secret 9642 under `candidate_pos_min`, with
  rank 2 and two selected x-matches, but again costs 159/125 = 1.272 rho.  The
  all-xmatch rule also verifies at 1.368 rho.

Current boundary after 520-527:

The axis-root line lift now has two fresh post-511 repeated-line positives:
line 945 on transfer 514 and line 3394 on transfer 527.  Both derive secrets
through public line replay, and both remain over rho at 1.272 measured.  The
new target-cap1 case shows that public below-rho target-cap1 reach can happen,
but rank-0/relation-empty rows are not enough.  The next target-67 gate is a
public target-cap1 row package that is rank-positive and relation-bearing, or
an unchanged repeated-line package with at least two replay-success rows so
the axis-root lift cost can be amortized honestly.

Fresh 528-535 target-cap1 recovery and line-8741 replay:

- Generated the next total3/total4 fresh stress window:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_528_535_probe.json`
- Public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_528_535.json`
- Frozen branch scheduler:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_528_535_pair56_s206_precise.json`
- Window report:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_window_report_528_535_pair56_s206_precise.json`
- The best stress policy is now target-cap1:
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, with 2 verified below-rho
  leaf cases and best stress cost 0.712 rho.  Target-cap3 still verifies, but
  only above rho; global selected no verified cases.
- The public bounded source contains 148 cases, 88 target-67 candidates, and 3
  signature-verified labels overall.  Target-cap1 has 12 target-67 candidates;
  all 12 are below rho and 2 labels verify.
- The clean verifier-positive target-cap1 case is target `67.a1@9803`,
  transfer 528, salt208, top-k 7, leaves `[0,1,2,4]`, policy
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, leaf selector
  `mode_cost_low_term_support_total4`.  The source cost is 0.712 rho with rank
  4, relation count 4, and `public_key_verified=true`.
- Lower-cost target-cap1 candidates at 0.52 to 0.60 rho exist, but they are
  rank 1 or rank 2 and verifier-negative.  This keeps relation rank and
  verifier positivity in the gate, not just cost.
- The old frozen scheduler selected zero cases, so this is a branch-taxonomy
  miss rather than a row-selector miss.
- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer528_targetcap1_verified_528_535.json`
- Exact profiling verified the case but found no preserving Sage degree-1
  factor candidate under the current preserving-factor criterion; the full
  remainder cost was 4.0 rho.
- Axis-root exact-profile output:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer528_targetcap1_verified_528_535.json`
- Axis-root emitted three candidate lines at 1.12 rho and no
  selected-root-positive preserving-factor surface:
  `4136*b + 1*c + 261`, `8098*b + 1*c + 5337`, and
  `8741*b + 1*c + 499`.
- Public replay through line 8741 with the frozen activation rule:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8741_transfer528_targetcap1_verified_frozen_activation_528_535.json`
- The frozen activation fires through `xmatch_count>=8`; replay derives secret
  574 under `all` at 0.544 rho, rank 2, relation count 2.  A diagnostic
  `where:candidate_pos=2` replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8741_transfer528_targetcap1_verified_candidate2_528_535.json`
  derives the same secret at 0.448 rho, but that orientation rule is post-hoc
  until it validates on future cases.
- A post-hoc branch
  `67_v4_0124_s208|67.a1@9803|7|0,1,2,4|mode_cost_low_term_support_total4|fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0|target_cap1_ow1_hw3_lw0_sw0_cw0_aw0|208`
  selects this case below rho:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_posthoc_target67_v4_0124_s208_528_535.json`.

Fresh 536-543 target-cap1 replication and line-8142 replay:

- Generated the next fresh stress window:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_536_543_probe.json`
- Public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_536_543.json`
- Frozen branch scheduler:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_536_543_pair56_s206_precise.json`
- Window report:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_window_report_536_543_pair56_s206_precise.json`
- The best stress policy again is target-cap1:
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, with 8 verified below-rho
  leaf cases and best stress cost 0.56 rho.
- The public bounded source contains 204 cases, 84 target-67 candidates, and
  46 signature-verified labels overall.  Target-cap1 has 8 target-67
  candidates; all 8 are below rho and all 8 verify.
- The best target-67 target-cap1 case is transfer 537, salt202, top-k 12,
  leaves `[0,6,10]`, policy `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, leaf
  selector `mode_cost_low_term_support_total3`.  The source cost is 0.56 rho
  with rank 2 and relation count 2.
- The old frozen scheduler selected zero cases.  The 528 post-hoc v4 branch
  also selected zero cases here:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_536_543_with_v4_0124_s208.json`.
  Exact leaf signatures are therefore drifting; the stable signal is broader
  target-cap1 plus low-term total3/total4 family reach.
- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer537_targetcap1_verified_536_543.json`
- Exact profiling finds a preserving degree-1 factor.  Root scan costs 0.784
  rho, surface cost is 0.608 rho, and the full remainder cost is 3.192 rho.
- Axis-root exact-profile output:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer537_targetcap1_verified_536_543.json`
- Axis-root recovers a singleton line `8142*b + 1*c + 4278` with line-lift
  cost 0.8 rho and one selected-root-positive surface.
- Public replay through line 8142 with the frozen activation rule:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8142_transfer537_targetcap1_verified_frozen_activation_536_543.json`
- The frozen activation fires through `xmatch_count>=8`.  Replay verifies three
  rules below rho.  The best rule is `candidate_pos_min`, deriving secret 5938
  at 0.448 rho with rank 2, relation count 2, and two selected x-matches.
  `all` and `term_shape:2+1+1` also derive the same secret at 0.544 rho.
- A post-hoc branch
  `67_v5_0610_s202|67.a1@9803|12|0,6,10|mode_cost_low_term_support_total3|fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0|target_cap1_ow1_hw3_lw0_sw0_cw0_aw0|202`
  selects exactly this case:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_posthoc_target67_v5_0610_s202_536_543.json`.

Current boundary after 528-543:

The target-67 blocker has moved.  Consecutive fresh windows now have
verifier-backed below-rho target-cap1 recoveries: transfer 528 at 0.712 rho
and transfer 537 at 0.56 rho from the public source, with line-gated replays at
0.544 rho and 0.448 rho respectively.  The old scheduler misses both because
the exact target branches are too narrow, and the exact v4 leaf signature from
528 fails on 536-543.  The next honest step is to mine and freeze a
family-level target-cap1 low-term branch over policy, low-term total3/total4
leaves, rank/relation positivity, and line-gate features, then validate it on
fresh 544-551.  Axis-root lift plus replay is still not below rho when lift
cost is charged per row for 528 or 537, so direct public source cost, replay
cost, and charged line-lift cost must remain separate ledgers.

Family-level target-67 target-cap1 selector:

- Added a reusable public family selector:
  `tasks/ecdlp_index_calculus/ffe_public_target67_family_branch_predictor.py`
- The frozen family rule is target `67.a1@9803`, policy
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, row selector
  `target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, low-term total3/total4 leaf modes,
  below-rho public cost, rank at least 2, relation count at least 2, selected
  row count 1, leaf 0 present, and either total4 support or leaf span at least
  10.  Verifier labels are reported after selection; `public_key_verified` is
  not used by the selector.
- Calibration artifact:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_calibration_528_543.json`
- On 528-543 the family selector selects 10 cases, all 10 verifier-backed
  below rho, with zero false positives.
- Added a rank-first per-transfer refinement, still using public relation
  system shape:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_calibration_528_543.json`
- On 528-543 the rank-first refinement selects 2 cases, both verifier-backed
  below rho, one per positive transfer.

Fresh 544-551 family validation control:

- Generated stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_544_551_probe.json`
- Public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_544_551.json`
- Family validation:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_validation_544_551.json`
- Rank-first family validation:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_544_551.json`
- This is a precision control.  Target-cap1 has 8 below-rho target-67 public
  candidates in the public source, but all are verifier-negative and mostly
  rank 0 or rank 1.  The frozen family selector and the rank-first refinement
  both abstain with zero selected cases and zero false positives.  A broad
  diagnostic selector without the leaf/rank gate selects 8 cases and verifies
  zero, confirming that abstention avoided a cheap false-positive hole.

Fresh 552-559 family validation and line-3379 replay:

- Generated stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_552_559_probe.json`
- Public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_552_559.json`
- The stress window is target-cap1 positive again: target-cap1 is the best
  policy with 4 verified below-rho leaf cases and best stress cost
  0.4379562 rho.
- Family validation:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_validation_552_559.json`
- The broad family selector selects 12 target-67 target-cap1 cases, with 2
  verifier-backed below-rho labels and 10 false positives.
- Rank-first family validation:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_552_559.json`
- The rank-first refinement selects exactly one case: transfer 559, salt204,
  top-k 4, leaves `[0,1,2,3]`, leaf selector
  `mode_cost_low_term_support_total4`, public source cost 0.712 rho, rank 4,
  relation count 4.  It is verifier-backed below rho with zero false
  positives.
- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer559_targetcap1_family_552_559.json`
- Exact profiling verifies the case but finds no preserving degree-1 factor
  under the current Sage preserving-factor criterion.  The full remainder cost
  is 2.6 rho.
- Axis-root exact-profile output:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer559_targetcap1_family_552_559.json`
- Axis-root emits three candidate lines at 0.84 rho:
  `945*b + 1*c + 952`, `3379*b + 1*c + 6949`, and
  `4136*b + 1*c + 261`.  There is no selected-root-positive preserving-factor
  surface under the exact-profile criterion.
- Frozen activation replay through those lines:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line945_transfer559_targetcap1_family_frozen_activation_552_559.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line3379_transfer559_targetcap1_family_frozen_activation_552_559.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line4136_transfer559_targetcap1_family_frozen_activation_552_559.json`
- Lines 945 and 4136 do not activate.  Line 3379 activates under the unchanged
  activation rule and derives secret 2952 below rho.  The best rule is the now
  forward-tested `where:candidate_pos=2` orientation at 0.464 rho; `all` and
  `term_shape:2+1+1` also verify at 0.56 rho.

Fresh 560-567 family validation control:

- Generated stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_560_567_probe.json`
- Public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_560_567.json`
- Rank-first family validation:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_560_567.json`
- This is another clean control, not a target-67 positive.  The stress window
  has target-cap1 below-rho leaf hits, but the public bounded source has zero
  target-67 target-cap1 candidates.  The rank-first target-67 family selector
  abstains with zero selected cases.

Current boundary after 544-567:

The target-67 family branch is now a real scheduler-level lead: it calibrates
cleanly on 528-543, abstains on a verifier-negative target-67 target-cap1 hole
in 544-551, selects a fresh verifier-backed transfer-559 case in 552-559, and
abstains on 560-567 when the target-cap1 positives move to `22050` rather than
target 67.  The rank-first per-transfer refinement is the cleaner branch
candidate: 2/2 verified on calibration, 0/0 in the 544 and 560 controls, and
1/1 verified on 552-559.  Line 3379 gives a fresh line-gated replay success at
0.464 rho, and `candidate_pos=2` is no longer only a 528 post-hoc diagnostic;
it has one later target-cap1 validation.  The charged axis-root lift plus replay
ledger still remains above rho per row for 559 (0.84 + 0.464), so the honest
claim is a public family branch plus line-gated below-rho replay, not a fully
charged line-prediction speedup.

Fresh 568-575 branch miss and nonzero-triple replay:

- Added a compact branch audit:
  `tasks/ecdlp_index_calculus/ffe_public_target67_family_branch_audit.py`
- Corrected per-window rank-dedup artifacts:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_528_535.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_536_543.json`
- Strict rank-dedup audit through 568-575:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_575.json`
- Through 528-575, the strict leaf-zero rank-first branch is precision-clean
  when it selects: 3 selected, 3 verifier-backed below rho, 0 false positives.
  It also records one family-rule miss on 568-575.
- Fresh 568-575 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_568_575_probe.json`
- Fresh 568-575 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_568_575.json`
- The best target-67 target-cap1 public case is transfer 569, salt206,
  top-k 12, leaves `[8,9,10]`, at 0.648 rho with rank 2 and relation count 2.
  It verifies the public key, but the strict leaf-zero family branch abstains
  because the leaf tuple has no zero.
- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer569_targetcap1_568_575.json`
- Exact profiling materializes the row cleanly and finds nontrivial resultant
  factors, but no preserving degree-1 Sage factor under the current preserving
  criterion.
- Axis-root artifact:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer569_targetcap1_568_575.json`
- Axis-root lift emits two candidate lines at 0.64 rho:
  `1296*b + 1*c + 3303` selecting leaf 10 and
  `2632*b + 1*c + 6506` selecting leaf 8.
- Public linear-factor gate replay:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_line1296_transfer569_targetcap1_568_575.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_line2632_transfer569_targetcap1_568_575.json`
- Line 1296 derives secret 2272 and verifies below rho at 0.576 with one
  selected leaf, rank 2, and relation count 2.  Line 2632 is below rho by cost
  but does not verify.
- A no-leaf-zero broad rank-dedup diagnostic captures this window and is clean
  retrospectively through 528-575:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_575.json`
  reports 4 selected, 4 verifier-backed below rho, and 0 false positives.
  This was not promoted to a claim because it was widened after seeing the
  568-575 miss.

Fresh 576-583 falsification and orientation boundary:

- Fresh 576-583 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_576_583_probe.json`
- Fresh 576-583 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_576_583.json`
- Strict rank-dedup validation:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_576_583.json`
- Broad no-leaf-zero rank-dedup validation:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_576_583.json`
- The fresh window falsifies both scheduler variants as standalone recovery
  selectors.  The strict leaf-zero branch selects transfer 582, salt206,
  top-k 12, leaves `[0,5,8,9]`, cost 0.592 rho, rank 2, relation count 2, but
  it is verifier-negative.  The broad branch selects two verifier-negative
  cases: transfer 582 leaves `[0,8,9]` at 0.584 rho and transfer 580 leaves
  `[6,7,8,16]` at 0.632 rho.
- Strict audit through 528-583:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_583.json`
  reports 3 selected verifier-backed below-rho cases, 1 selected false
  positive, one 568-575 family-rule miss, and precision 0.75 when selected.
- Broad audit through 528-583:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_583.json`
  reports 4 selected verifier-backed below-rho cases, 2 selected false
  positives, and precision 2/3 when selected.
- False-positive exact profiles:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_false_576_583.json`
- The transfer-580 false positive has a preserving degree-1 Sage factor with
  below-rho surface and root-scan proxy costs, but the public source case still
  does not verify.  Therefore cheap linear factors are not sufficient; the
  orientation/replay layer must be part of the branch boundary.
- False-positive axis-root artifact:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_false_576_583.json`
- The false positives emit candidate-line counts `[2,2,3]` with line-lift
  costs 0.896, 1.12, and 1.16 rho.
- False-positive line orientation artifacts:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line5612_transfer582_false_total3_576_583.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line7570_transfer582_false_total3_576_583.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line2793_transfer580_false_total4_576_583.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line5658_transfer580_false_total4_576_583.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8142_transfer580_false_total4_576_583.json`
- All five false-positive candidate lines activate but verify zero public
  orientation rules.  This gives the next boundary: target-cap1 low-term rank
  shape and even cheap line factors can be false, while x-match orientation
  consistency still separates the known successful line-gated cases from the
  576-583 false positives in this sample.

Current boundary after 568-583:

The strict rank-dedup family selector is no longer a precision-clean standalone
scheduler.  It remains useful as a target-cap1 low-term candidate generator,
but 576-583 proves that rank/relation positivity plus leaf-zero support can
select verifier-negative rows.  The broad no-leaf-zero split captures the
568-575 nonzero-triple replay but also adds false positives immediately on
576-583.  The most actionable signal is now a two-stage package: use the
target-cap1 low-term family only to propose candidate rows, then require a
public axis-root line and x-match orientation rule that survives a false-positive
feature audit.  The next concrete work item is to build that orientation
feature table over 528-583, with 528/537/559/569 as successes and 580/582 as
hard negatives, then validate the frozen rule on 584-591.

Orientation feature rule and 584-591 forward check:

- Added a public axis/orientation feature miner:
  `tasks/ecdlp_index_calculus/ffe_target67_axis_orientation_feature_rule_miner.py`
- Training artifact:
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_miner_528_583.json`
- Training rows are the below-rho line-gated successes 528, 537, 559, and 569,
  plus hard negatives from transfer 580 and 582.
- The mined public rule is:
  `activate:top_k<=12&xmatch_count>=8`
- Under measured-below-rho labels it selects all 4 successes and rejects all
  5 hard negatives.  The rule search excludes transfer ids, salts, line
  coefficients, verifier labels, and `valid_relation` labels.
- Fresh 584-591 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_584_591_probe.json`
- Fresh 584-591 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_584_591.json`
- The fresh window is target-cap1 positive, but for `22050`, not for target 67:
  target-cap1 has 4 verified below-rho leaf hits, best 0.43065693 rho, while
  the target-67 family generators select zero cases.
- Fresh family validation artifacts:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_584_591.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_584_591.json`
- Updated audits through 584-591:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_591.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_591.json`
- The strict branch audit now has 8 windows: 3 verified selections, 1 selected
  false positive, 2 no-target67-targetcap1 abstentions, one verifier-negative
  target67-targetcap1 abstention, and the 568-575 family-rule miss.  Precision
  when selected remains 0.75.
- Since target 67 had no target-cap1 candidate, the forward orientation gate
  could not validate on the intended branch.  A target-cap3 diagnostic remains
  useful: transfer 586, top-k 12, salts 203/201/202, leaf 9, verifies the
  public key at source cost 1.496 rho.
- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer586_targetcap3_584_591.json`
- All three transfer-586 target-cap3 profiles have preserving below-rho
  root-scan factors.  Two are selected-root-positive; one is preserving but
  misses the selected root under the current criterion.
- Axis-root artifact:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer586_targetcap3_584_591.json`
- Axis-root line lift emits low-cost public lines: line-lift costs are 0.304,
  0.352, and 0.352 rho.  Line `1427*b + c + 7108` appears on salts 203/201;
  line `7515*b + c + 142` appears on salts 201/202.
- Frozen-gate replay artifacts:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line1427_transfer586_targetcap3_frozen_gate_584_591.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line7515_transfer586_targetcap3_frozen_gate_584_591.json`
- Both lines activate under the frozen `top_k<=12 & xmatch_count>=8` rule and
  both derive secret 9387, but best measured replay is 1.336 rho and best
  shared-leaf plus hit-root charged replay is 1.304 rho.  This is a
  line-family/orientation validation, not a below-rho speedup.
- Frozen-rule validation artifact:
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_frozen_validation_584_591.json`
- Under measured-below-rho labels, the frozen rule selects two 584-591
  over-rho target-cap3 line cases and zero below-rho successes.  This means the
  rule is insufficient as a speedup gate unless the candidate generator is
  restricted back to target-cap1 or a cost ceiling is added.

Current boundary after 584-591:

The new orientation feature rule is a good hard-negative filter for the
target-cap1 cases seen through 583, but it is not yet a general speedup gate.
On the first forward window it has no target-cap1 target-67 opportunity, and on
the target-cap3 diagnostic it activates real line-gated secret recovery that
still costs above rho.  The next refinement should freeze an additional public
cost/policy guard before the orientation rule is promoted: either keep it
target-cap1-only for the scheduler branch, or add a public source/replay cost
ceiling that rejects the 586 target-cap3 over-rho rows while preserving the
528/537/559/569 below-rho rows.

V2 row-capacity gate and 592-599 forward check:

- Updated the feature miner to include public `policy` and
  `source_ops_over_rho_millirhos` in the rule vocabulary:
  `tasks/ecdlp_index_calculus/ffe_target67_axis_orientation_feature_rule_miner.py`
- V2 training artifact:
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_miner_v2_528_591.json`
- Training rows include the four below-rho successes 528/537/559/569, the
  five 580/582 hard negatives, and the two 586 target-cap3 over-rho line
  recoveries.  Under measured-below-rho labels, the mined best rule is:
  `activate:row_xmatch_count_max>=8&top_k<=12`
- The v2 rule selects all four measured-below-rho successes and rejects all
  seven non-speedup rows.  The planned policy guard and source-cost guard also
  score 4 true positives, 0 false positives, and 0 false negatives:
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_v2_policy_guard_eval_528_591.json`
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_v2_cost_guard_eval_528_591.json`
- Added `row_xmatch_count_min` and `row_xmatch_count_max` to the public
  activation features emitted by:
  `tasks/ecdlp_index_calculus/ffe_public_linear_factor_xmatch_orientation_audit.py`
- Fresh 592-599 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_592_599_probe.json`
- Fresh 592-599 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_592_599.json`
- The fresh window is strongly target-cap1 positive overall: 16 verified
  below-rho target-cap1 leaf hits and 9 verified target-cap1 row hits, best
  0.40145985 rho.  For target 67, however, the cheapest target-cap1 public
  candidates are rank-0 and verifier-negative.
- Fresh target-67 family validations:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_592_599.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_592_599.json`
- Both strict and broad target-67 target-cap1 candidate generators abstain
  with zero selected cases.  This is a clean rank/relation guard control:
  target-67 target-cap1 reach exists, but the public rank/relation shape is
  not credible.
- Rolling audits through 592-599:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_599.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_599.json`
- The strict audit now covers nine windows: 3 selected verified-below-rho
  windows, 1 selected false-positive window, 2 no-target67-targetcap1
  abstentions, 2 target67-targetcap1-unverified abstentions, and the 568-575
  family-rule miss.  Precision when selected remains 0.75.
- Target-cap3 diagnostic exact profiles:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer595_targetcap3_592_599.json`
- Transfer 595, top-k 12, leaf 12 on salts 209/204/202 verifies target 67 at
  public source cost 1.368 rho.  Exact profiling finds preserving below-rho
  root-scan factors on all three rows, but one row still has no selected-root
  recovery under the current criterion.
- Axis-root diagnostic:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer595_targetcap3_592_599.json`
- Axis-root lift emits repeated singleton public line
  `2716*b + c + 4800` on salts 209 and 204.  Line-lift proxy costs are 0.288,
  0.448, and 0.336 rho across the three exact profiles.
- V2 replay artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line2716_transfer595_targetcap3_v2_gate_592_599.json`
- The v2 activation gate blocks the transfer-595 target-cap3 diagnostic:
  activated case count 0, verified rules 0.
- V2 block validation over 586 and 595 target-cap3 diagnostics:
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_v2_block_validation_584_599.json`
- The frozen v2 rule has zero false positives on the three over-rho diagnostic
  line cases while still preserving the four below-rho training successes in
  the 528-591 training table.

Current boundary after 592-599:

The candidate stack is now better separated into stages.  The target-67
target-cap1 rank/relation generator abstains on fresh rank-0 holes, and the
v2 axis/orientation gate rejects target-cap3 repeated-line recoveries that
derive secrets but remain over rho.  This is progress toward a public
index-calculus-style pipeline, but still not a completed speedup: the latest
fresh target-67 below-rho wins are absent, while the useful fresh target-67
line structure is over-rho or blocked.  The next forward test should keep the
v2 rule frozen and validate on 600-607, looking specifically for a target-67
target-cap1 candidate that passes both rank/relation generation and the v2
orientation gate.

600-607 forward validation:

- Fresh 600-607 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_600_607_probe.json`
- The best frozen policy is again target-cap1: 2 verified below-rho leaf hits,
  4 verified below-rho row hits, and best leaf cost 0.568 rho.  Target-cap3
  and global-cap3 produce verifier positives but no below-rho leaf hits.
- Fresh 600-607 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_600_607.json`
- The public bounded table has 208 cases and 42 verifier-positive labels.
  For target 67, the cheapest target-cap1 case is the familiar rank-0,
  verifier-negative 0.424-rho row, but the rank/relation-aware target-cap1
  family selector finds a stronger case:
  transfer 607, salt 206, top-k 4, leaves `[0,1,2,3]`, total4, rank 2,
  relation count 2, verified below rho at public source cost 0.568.
- Strict and broad family predictor artifacts:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_600_607.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_600_607.json`
- Both predictors select exactly the same target-67 target-cap1 case and have
  1 selected verified-below-rho case, 0 selected false positives.
- Rolling audits through 600-607:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_607.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_607.json`
- Strict rank-dedup now has 10 windows, 5 selected cases, 4 selected
  verified-below-rho cases, and 1 selected false positive; precision when
  selected is 0.8.  Broad now has 7 selected cases, 5 selected
  verified-below-rho cases, and 2 selected false positives; precision when
  selected is about 0.714.
- Exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer607_targetcap1_600_607.json`
- Sage materializes the exact selected profile and finds a preserving
  degree-1 resultant factor on
  `67.a1@9803:uniform:256:salt206`.  The full remainder costs 2.456 rho, but
  the preserving factor has surface FFE cost 0.624 rho and root-scan cost
  0.76 rho.
- Axis-root artifact:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer607_targetcap1_600_607.json`
- Axis-root lift emits one singleton candidate line:
  `161*b + c + 6315`, tied to selected leaf 2, with line-lift proxy cost
  0.84 rho.
- V2 orientation replay artifact:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line161_transfer607_targetcap1_v2_gate_600_607.json`
- The frozen v2 activation rule
  `activate:row_xmatch_count_max>=8&top_k<=12` activates the case.  The
  public `candidate_pos_min` x-match rule derives secret 1752 with rank 2,
  relation count 2, and measured replay cost 0.448 rho.  The broader `all`
  rule also derives the same secret at 0.544 rho.
- Updated v2 forward feature audit:
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_v2_forward_eval_528_607.json`
- With the 607 success and the 595 over-rho blocked diagnostic included, the
  frozen v2 rule remains the best rule: 13 line-present records, 5 measured
  below-rho successes, 8 failures, and frozen-rule score 5 true positives,
  0 false positives, 0 false negatives.

Current boundary after 600-607:

This is the first clean fresh target-67 target-cap1 end-to-end hit in the
current v2 pipeline.  The public family generator selected a rank/relation
positive target-cap1 case, Sage exact profiling exposed a preserving
degree-1 factor, axis-root lift reduced the line stage to a singleton public
line, and the frozen v2 orientation gate plus public x-match rule derived the
secret below rho.  The result is still a staged candidate pipeline rather than
a completed ECDLP index-calculus algorithm: exact line confirmation and
axis-root lift remain separately charged stages, and the next obligation is a
new forward window with the same frozen generator and v2 gate.

608-615 forward validation:

- Fresh 608-615 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_608_615_probe.json`
- The frozen target-cap1 row selector remains the best policy: 6 verified
  below-rho leaf hits, 4 verified below-rho row hits, and best leaf cost
  0.568 rho.  Target-cap3 and global-cap3 again have verifier positives but
  no below-rho leaf hits.
- Fresh 608-615 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_608_615.json`
- The source has 199 public bounded cases and 10 verifier-positive labels.
  Target 67 has 115 cases, 6 verifier-positive labels, and 6 verified
  below-rho labels.  The cheap target67 decoys still exist, but the rank-aware
  branch finds transfer 612, salt206, top-k 4.
- Strict and broad family predictor artifacts:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_608_615.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_608_615.json`
- The strict branch selects the total4 row `[0,1,2,3]` at public source cost
  0.576 rho, rank 2, relation count 2, verified below rho.  The broad branch
  selects the total3 sibling `[1,2,3]` at 0.568 rho, also rank 2 and verified
  below rho.
- Rolling audits through 608-615:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_615.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_615.json`
- Strict rank-dedup now covers 11 windows, with 6 selected cases, 5 selected
  verified-below-rho cases, and 1 selected false positive.  Precision when
  selected rises to 5/6 = 0.83333333.  Broad now has 8 selected cases, 6
  verified-below-rho, and 2 false positives, precision 0.75.
- Strict exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer612_targetcap1_608_615.json`
- The strict total4 profile is a verifier-backed relation success but not a
  preserving-factor success.  Sage materializes the profile and finds
  nontrivial resultant factors, but no preserving factor candidate; full
  remainder cost is 3.864 rho.
- Strict axis-root artifact:
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer612_targetcap1_608_615.json`
- Axis-root lift emits two old public lines, `945*b + c + 952` and
  `3379*b + c + 6949`, but selected-root-positive surface count is 0 and
  line-lift proxy cost is 1.12 rho.
- Broad total3 diagnostics:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer612_targetcap1_broad_total3_608_615.json`
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer612_targetcap1_broad_total3_608_615.json`
- The broad total3 sibling has the same boundary: no preserving factor,
  full remainder cost 3.848 rho, the same two candidate lines, and zero
  selected-root-positive surfaces.
- V2 replay artifacts for the repeated lines:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line945_transfer612_targetcap1_total4_v2_gate_608_615.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line3379_transfer612_targetcap1_total4_v2_gate_608_615.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line945_transfer612_targetcap1_total3_v2_gate_608_615.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line3379_transfer612_targetcap1_total3_v2_gate_608_615.json`
- All four repeated-line replays have activated case count 0 and verified rule
  count 0 under the frozen v2 activation gate.
- Updated v2 forward feature audit:
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_v2_forward_eval_528_615.json`
- With the 608-615 line negatives included, the frozen rule remains the best
  rule: 17 records, 5 measured-below-rho successes, 12 failures, and
  frozen-rule score 5 true positives, 0 false positives, 0 false negatives.

Current boundary after 608-615:

The strict target-67 target-cap1 public family generator validated again as a
row/relation selector, but the FFE line stage did not validate on this
window.  Transfer 612 is a clean relation-layer below-rho row-selection hit,
not an end-to-end line-backed speedup.  The repeated old lines 945 and 3379
reappear as axis-root candidates but have no selected-root-positive preserving
surface and are blocked by the v2 orientation gate.  This strengthens the
stage separation: public generation is now improving, while exact/axis line
confirmation remains the bottleneck for a generalized index-calculus pipeline.

616-623 forward validation:

- Fresh 616-623 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_616_623_probe.json`
- The frozen target-cap1 row selector remains the best policy again, but this
  time the below-rho target-cap1 hit moves to `22050.cf1@11731`: 4 verified
  below-rho leaf hits, 1 verified below-rho row hit, and best leaf cost
  0.43065693 rho.
- Fresh 616-623 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_616_623.json`
- The source has 191 public bounded cases and 36 verifier-positive labels.
  The best public case is target `22050.cf1@11731`, transfer 618, salt166,
  top-k 7, total3 leaves `[8,56,90]`, source cost 0.43065693 rho, rank 2,
  relation count 2, and public-key verified.
- Target-67 strict and broad family predictor artifacts:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_616_623.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_616_623.json`
- Both frozen target-67 predictors abstain on this window.  Target 67 has
  verifier-positive over-rho diagnostics, but no target67 target-cap1 public
  selection that satisfies the strict or broad family rule below rho.
- Existing frozen multi-branch scheduler artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_616_623_pair56_s206_precise.json`
- The scheduler reactivates the old 22050 branches without using verifier
  labels: it selects two systems, one from `22050_v1_85690` and one from
  `22050_v1_8345690`, both at transfer 618/salt166, and both are
  verifier-backed below rho.  This means the fresh 22050 line-backed replay is
  not only a post-hoc hand selection from the public table.
- Rolling target-67 audits through 616-623:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_623.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_623.json`
- Strict rank-dedup now covers 12 windows and remains at 6 selected cases,
  5 selected verified-below-rho cases, and 1 selected false positive;
  precision when selected stays 5/6 = 0.83333333.  Broad remains at
  8 selected cases, 6 verified-below-rho, and 2 false positives, precision
  0.75.
- `22050.cf1@11731` total3 exact-profile artifact:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer618_targetcap1_616_623.json`
- Sage exact profiling on the total3 row finds a preserving degree-1 factor:
  root-scan cost 0.45255474 rho, surface FFE cost 0.47445255 rho, full
  remainder cost 1.7080292 rho.  The selected-root pair count is still zero,
  so the factor is a line-gate lead rather than a full selected-root-positive
  axis-root success.
- `22050.cf1@11731` total3 axis-root artifact:
  `ecdlp_index_calculus_state/ffe_axis_root_exact_profile_probe_22050_transfer618_targetcap1_616_623.json`
- Axis-root lift emits two public candidate lines through selected leaf 8:
  `4745*b + c + 3236` and `6202*b + c + 10586`; line-lift proxy cost is
  0.52554745 rho and selected-root-positive surface count is 0.
- Public linear-factor gate replay artifacts:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line4745_transfer618_targetcap1_616_623.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line6202_transfer618_targetcap1_616_623.json`
- Both lines isolate leaf 8 and derive the same verified secret 4620 with
  rank 2, relation count 2, selected row count 1, selected leaf count 1, and
  measured gate replay cost 0.41605839 rho.
- Public orientation artifacts:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line4745_transfer618_targetcap1_v2_gate_616_623.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line6202_transfer618_targetcap1_v2_gate_616_623.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line4745_transfer618_targetcap1_all_616_623.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line6202_transfer618_targetcap1_all_616_623.json`
- The frozen target67 v2 activation rule
  `row_xmatch_count_max>=8&top_k<=12` does not activate because this row has
  `row_xmatch_count_max=6`.  Under ungated public orientation, the rule
  `candidate_pos_min` derives secret 4620 at measured oriented cost
  0.37226277 rho, while `all` derives at 0.41605839 rho.
- The total4 sibling `[8,34,56,90]` repeats the same factor/axis-line
  mechanism:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer618_targetcap1_total4_616_623.json`
  `ecdlp_index_calculus_state/ffe_axis_root_exact_profile_probe_22050_transfer618_targetcap1_total4_616_623.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line4745_transfer618_targetcap1_total4_616_623.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line6202_transfer618_targetcap1_total4_616_623.json`
- The total4 exact root-scan cost is 0.45985401 rho and both public line
  replays again derive secret 4620 at 0.41605839 rho.

Current boundary after 616-623:

Target 67 abstained under the frozen strict and broad public family rules, so
the target-67 v2 claim is not extended by this window.  However, the same
frozen target-cap1 row selector plus the older public multi-branch scheduler
produced a fresh cross-target `22050.cf1@11731` line-backed recovery: public
row selection, Sage degree-1 FFE factor, public axis-root candidate lines,
public linear-factor gate replay, and public orientation replay all agree on
secret 4620 below rho.  The main caveat is line prediction and charging: the
exact/axis line stage is still a second stage, selected-root-positive count is
zero, and the target67 v2 activation rule correctly does not fire.  This
should be treated as a revived 22050 target-cap1 FFE line-gate branch that now
needs fresh line prediction or strict holdout validation on 624-631.

624-631 forward validation:

- Fresh 624-631 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_624_631_probe.json`
- The frozen target-cap1 row selector remains best: 6 verified below-rho leaf
  hits, 3 verified below-rho row hits, 4 verified rows total, and best leaf
  cost 0.47445255 rho.  Target-cap3 and global-cap3 again have verifier
  positives but no below-rho leaf hits.
- Fresh 624-631 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_624_631.json`
- The source has 188 public bounded cases and 14 verifier-positive labels.
  The cheapest 22050 and target67 public cases are decoys, so branch evidence
  comes from the frozen public branch ledgers below.
- Frozen multi-branch scheduler artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_624_631_pair56_s206_precise.json`
- The scheduler selects exactly one system, the old
  `22050_v1_19348990` branch at target `22050.cf1@11731`, transfer 625,
  salt167, top-k 4, leaves `[19,34,89,90]`.  It is verifier-backed below rho
  with source cost 0.47445255 rho, rank 3, and relation count 3.
- Target67 strict and broad family artifacts:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_624_631.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_624_631.json`
- The strict target67 family abstains.  The broad low-term diagnostic selects
  target `67.a1@9803`, transfer 628, salt206, top-k 16, total3 leaves
  `[2,3,6]`, source cost 0.528 rho, rank 2, relation count 2, and
  public-key verified.
- Rolling target67 audits through 624-631:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_631.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_631.json`
- Strict rank-dedup now covers 13 windows and remains at 6 selected cases,
  5 selected verified-below-rho cases, and 1 selected false positive;
  precision when selected stays 5/6 = 0.83333333.  Broad now has 9 selected
  cases, 7 verified-below-rho, and 2 false positives; precision is 7/9 =
  0.77777778.
- `22050.cf1@11731` exact and axis artifacts:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer625_targetcap1_624_631.json`
  `ecdlp_index_calculus_state/ffe_axis_root_exact_profile_probe_22050_transfer625_targetcap1_624_631.json`
- The scheduled 22050 profile is relation-backed but not preserving-factor
  backed: Sage finds nontrivial factors but zero preserving candidates, and
  full remainder cost is 2.35766423 rho.  Axis-root emits three candidate
  lines, selected-root-positive surface count is 0, and line-lift proxy cost
  is 0.88 rho.
- 22050 public line-gate artifacts:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line3431_transfer625_targetcap1_624_631.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line5230_transfer625_targetcap1_624_631.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line6445_transfer625_targetcap1_624_631.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line5230_transfer625_targetcap1_all_624_631.json`
- Two of the three axis lines are verifier-negative under line-gated replay.
  The middle line, `5230*b + c + 7939`, isolates leaf 89 and derives secret
  7198 below rho: gate replay costs 0.37956204 rho, and public
  `candidate_pos_min` orientation costs 0.35036496 rho.
- Target67 broad exact and axis artifacts:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer628_broad_total3_624_631.json`
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer628_broad_total3_624_631.json`
- The target67 broad total3 profile has a preserving degree-1 factor:
  root-scan cost 0.792 rho, surface FFE cost 0.576 rho, selected-root pair
  count 1, selected valid root leaves 1, and full remainder cost 4.28 rho.
  Axis-root lift emits a singleton selected-root-positive line,
  `161*b + c + 6315`, with line-lift proxy cost 0.96 rho.
- Target67 broad replay artifacts:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_line161_transfer628_broad_total3_624_631.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line161_transfer628_broad_total3_v2_gate_624_631.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line161_transfer628_broad_total3_all_624_631.json`
- Line-gated replay through `161*b + c + 6315` derives secret 955 below rho
  at 0.512 rho.  Ungated public orientation with `candidate_pos_min` derives
  the same secret at 0.448 rho.  The frozen v2 gate
  `row_xmatch_count_max>=8&top_k<=12` does not activate because this is a
  top-k 16 broad-branch row.
- Target67 broad total4 control:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer628_broad_total4_624_631.json`
- The total4 sibling `[2,3,6,16]` is relation-backed but has no preserving
  factor; the line-backed mechanism is the total3 broad profile only.

Current boundary after 624-631:

The strict target67 branch still abstains, so the original v2 claim remains
unchanged.  Two useful line-gated recoveries nevertheless appeared on the same
fresh window.  First, the old 22050 scheduler branch selected a verified
target-cap1 row and one public axis line, `5230*b+c+7939`, derived secret 7198
below rho despite the absence of a preserving-factor certificate.  Second, the
broad target67 total3 diagnostic selected a top-k 16 row with a true
singleton selected-root-positive axis line, `161*b+c+6315`, and public replay
derived secret 955 below rho.  The price of this progress is honesty: neither
result extends the frozen target67 v2 gate, and both still need a preregistered
public line/activation rule on 632-639 before they can be promoted beyond
staged FFE line-gate evidence.

632-639 forward validation:

- Fresh 632-639 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_632_639_probe.json`
- The frozen target-cap1 row selector is again the best stress policy: 24
  verified below-rho leaf hits, 5 verified below-rho row hits, 6 verified rows
  total, and best leaf cost 0.39416058 rho.  Target-cap3 and global-cap3 have
  verifier positives but no below-rho leaf hits.
- Fresh 632-639 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_632_639.json`
- The source has 235 public bounded cases and 79 verifier-positive labels.  The
  cheapest public 22050 case is transfer 636, salt173, top-k 4, leaves
  `[19,34,90]` at 0.39416058 rho; the cheapest public target67 case is
  transfer 639, salt204, top-k 7, leaves `[0,2,4]` at 0.496 rho.
- Frozen multi-branch scheduler artifact:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_632_639_pair56_s206_precise.json`
- The scheduler selects three verified below-rho `22050.cf1@11731` systems,
  all on transfer 636/salt173: `22050_v1_85690` total3 leaves `[8,56,90]`
  at 0.43065693 rho, `22050_v1_8345690` total4 leaves `[8,34,56,90]`
  at 0.4379562 rho, and `22050_v1_19348990` total4 leaves `[19,34,89,90]`
  at 0.4379562 rho.  All three have rank 3 and relation count 3.
- Target67 strict and broad family artifacts:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_632_639.json`
- The strict family selects two verified below-rho cases with no false
  positives: transfer 639/salt204/top-k 7 total4 leaves `[0,2,4,6]` at
  0.504 rho, and transfer 634/salt202/top-k 16 total3 leaves `[0,5,15]` at
  0.672 rho.  The broad diagnostic selects the matching transfer 639 total3
  leaves `[0,2,4]` at 0.496 rho plus the same transfer 634 top-k 16 case.
- Rolling target67 audits through 632-639:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_639.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_639.json`
- Strict rank-dedup now covers 14 windows, with 8 selected cases, 7 selected
  verified-below-rho cases, and 1 selected false positive; precision when
  selected is 7/8 = 0.875.  Broad covers the same 14 windows with 11 selected
  cases, 9 verified-below-rho, and 2 false positives; precision is 9/11 =
  0.81818182.
- `22050.cf1@11731` exact and axis artifacts:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer636_targetcap1_632_639.json`
  `ecdlp_index_calculus_state/ffe_axis_root_exact_profile_probe_22050_transfer636_targetcap1_632_639.json`
- The three 22050 profiles are relation-backed but not preserving-factor
  backed: Sage materializes all three profiles, finds nontrivial factors on
  all three, but zero preserving candidates.  Axis-root still emits public
  candidate lines.  The repeated line `6110*b+c+4058` isolates leaf 90; the
  sibling lines `6202*b+c+10586` and `7359*b+c+4585` are verifier-negative
  under replay.
- 22050 public line-gate and orientation artifacts:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line6110_transfer636_total3_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line6110_transfer636_top7_total4_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line6110_transfer636_top4_total4_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line6110_transfer636_total3_all_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line6110_transfer636_top7_total4_all_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line6110_transfer636_top4_total4_all_632_639.json`
- Public replay through `6110*b+c+4058` derives secret 850 below rho on all
  three scheduled 22050 profiles.  Gate replay and public orientation both
  cost 0.37956204 rho, with rank 2 and relation count 2.
- Target67 exact and axis artifacts:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_union_632_639.json`
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_union_632_639.json`
- All three target67 profiles have preserving degree-1 factors.  Root-scan
  costs are 0.688 rho for transfer 639 total3, 0.696 rho for transfer 639
  total4, and 0.896 rho for transfer 634 top-k 16 total3.  Axis-root emits a
  singleton selected-root-positive line `8098*b+c+5337` for both transfer 639
  profiles and three candidate lines for transfer 634, of which
  `562*b+c+2148` verifies under replay.
- Target67 replay artifacts:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_line8098_transfer639_total3_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_line8098_transfer639_total4_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_line562_transfer634_top16_total3_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8098_transfer639_total3_all_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8098_transfer639_total4_all_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line562_transfer634_top16_total3_all_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8098_transfer639_total3_v2_gate_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line8098_transfer639_total4_v2_gate_632_639.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_line562_transfer634_top16_total3_v2_gate_632_639.json`
- Line `8098*b+c+5337` derives secret 9577 at 0.48 rho on both transfer 639
  total3 and total4 profiles.  Line `562*b+c+2148` derives secret 650 at
  0.512 rho by gate replay and 0.448 rho under public `candidate_pos_min`
  orientation.  The frozen v2 activation rule
  `row_xmatch_count_max>=8&top_k<=12` activates zero target67 cases in this
  window, including the top-k 7 transfer 639 profiles.

Current boundary after 632-639:

The fresh window is the strongest combined structural-positive window so far:
strict target67 selection no longer abstains, broad target67 repeats a
top-k-16 line-gated success, and the old 22050 line `6110*b+c+4058` verifies
across three scheduled profiles.  The main caveat is still the same honest
boundary: the frozen target67 v2 activation gate abstains on all target67
632-639 line replays, and the 22050 branch remains axis-line-only rather than
preserving-factor-backed.  Treat this as strong FFE line-gate recovery
evidence and a better target for public trigger mining, not yet as a closed
generalized index-calculus algorithm.

640-647 forward validation and trigger-miner follow-up:

- Fresh 640-647 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_640_647_probe.json`
- The frozen target-cap1 selector has no reach in this window: 0 verified
  rows and 0 verified leaves.  Target-cap3 and global-cap3 each have 17
  verified rows, but no verified total3/total4 leaf cases and no below-rho
  leaf hits.
- Fresh 640-647 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_640_647.json`
- The source has 136 public bounded cases and zero verifier-positive labels.
  The best-looking 22050 public cost is below rho at 0.94160584, but it is
  rank 0/relation 0 and therefore a public-cost decoy.  The best target67
  public case is over rho at 1.224 and also rank 0.
- Frozen ledgers:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_640_647_pair56_s206_precise.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_640_647.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_640_647.json`
- The 22050 scheduler, strict target67, and broad target67 ledgers all abstain
  with zero selected cases and zero false positives.  Rolling target67 audits
  through 640-647 are:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_647.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_647.json`
- Strict rank-dedup now covers 15 windows and remains at 8 selected cases, 7
  verified-below-rho cases, and 1 false positive; precision when selected is
  still 7/8 = 0.875.  Broad remains 9/11 = 0.81818182 when selected.
- 22050 repeated-line negative control:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_22050_line6110_transfer646_globalcap3_total3_640_647.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_22050_line6110_transfer646_globalcap3_total3_all_640_647.json`
- The old line `6110*b+c+4058` is nonempty on the best global-cap3 decoy and
  retains the same 0.94160584 rho public cost, but it does not verify under
  gate replay or orientation.  This confirms the 640-647 22050 outcome is a
  genuine abstention/negative, not just a missed branch-scheduler alias.
- Target67 trigger-miner refresh artifacts:
  `ecdlp_index_calculus_state/ffe_target67_line_stage_audit_328_639_plus.json`
  `ecdlp_index_calculus_state/ffe_target67_orientation_feature_probe_328_639_plus.json`
  `ecdlp_index_calculus_state/ffe_target67_line_success_activation_rule_miner_328_639_train631_max3.json`
- The refreshed target67 line-stage table now has 42 surfaces, 29 preserving
  lines, 16 line-present replay successes, 9 line-present replay failures, and
  13 no-preserving-line surfaces.  No naive additive line+replay charge beats
  rho; line-confirmation amortization still needs two reuses.
- The line-success activation miner is still post-line-gate, not a degree-1
  line predictor.  Trained through transfer 631, its training-best zero-FP
  rule is
  `activate:top_k=4&scheduled1_count>=4|top_k=7&xmatch_count>=4|xmatch_count>=9`.
  That rule has 11 train true positives, 0 train false positives, and catches
  2/3 held-out 632-639 successes with 0 validation false positives.  The
  validation-favorable zero-FP rule is
  `activate:top_k=16&xmatch_count>=6|top_k=4&scheduled1_count>=4|top_k=7&xmatch_count>=4`;
  it catches all three 632-639 held-out target67 line successes, at the cost
  of one fewer old training true positive.

Current boundary after 640-647:

640-647 is a clean abstention/negative window for the three frozen ledgers.  It
does not refute the 632-639 structural line-gate evidence, but it does show
the target-cap1-dependent branch is intermittent and that the old 22050
`6110` line is not enough when it appears only on a global-cap3 decoy.  The
most useful new artifact is the refreshed target67 line-success activation
miner: it gives a concrete zero-FP validation-favorable activation rule to
freeze for the next line-present holdout, while preserving the honest caveat
that it activates after a public line gate and does not yet predict the line
before exact FFE.

648-655 forward validation and exact line boundary:

- Fresh 648-655 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_648_655_probe.json`
- The frozen target-cap1 selector again has no verified leaves, though it has
  2 verified rows.  Target-cap3 has 4 verified leaves and 25 verified rows;
  global-cap3 has 4 verified leaves and 23 verified rows.  The best verified
  leaves are still over rho at 1.304, so the frozen leaf path has no below-rho
  end-to-end win in this window.
- Fresh 648-655 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_648_655.json`
- The source has 200 public bounded cases and 8 verifier-positive labels.  All
  verifier positives are target67 transfer 653 over-rho rows under target-cap3
  or global-cap3; the best public 22050 case is below rho at 0.3649635 but is
  not public-key verified.
- Frozen ledgers:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_648_655_pair56_s206_precise.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_648_655.json`
- The 22050 scheduler, strict target67 branch, and broad target67 branch all
  abstain with zero selected cases.  Rolling target67 audits through 648-655
  are:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_655.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_655.json`
- Strict rank-dedup now covers 16 windows and remains at 8 selected cases, 7
  verified-below-rho cases, and 1 false positive; precision when selected is
  still 0.875.  Broad remains 9 verified-below-rho selections out of 11
  selected cases, with 2 false positives and precision 0.81818182.
- Target67 transfer 653 exact/profile artifacts:
  `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_target67_transfer653_targetcap3_verified_overrho_648_655.json`
  `ecdlp_index_calculus_state/ffe_target67_axis_root_exact_profile_probe_transfer653_targetcap3_verified_overrho_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_transfer653_targetcap3_line6185_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_transfer653_targetcap3_line6185_all_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_transfer653_targetcap3_line6185_activation_miner_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_transfer653_targetcap3_costtotal3_line6185_all_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_transfer653_targetcap3_costtotal4_line6185_all_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_transfer653_targetcap3_costtotal3_line6185_activation_miner_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_transfer653_targetcap3_costtotal4_line6185_activation_miner_648_655.json`
  `ecdlp_index_calculus_state/ffe_target67_line_stage_audit_transfer653_targetcap3_line6185_split_648_655.json`
- Exact Sage factorization found 6 preserving target67 surfaces.  All have
  below-rho preserving factor/root scans, with root-scan ratios from 0.432 to
  0.688, but no full-remainder FFE win.
- Axis-root lift found 4 singleton candidate-line surfaces.  The repeated line
  is `6185*b+c+2919` over GF(9803), with line-lift ratios from 0.304 to
  0.624.  The factor gate keeps the same leaf-3 monic coefficients
  `(b,c)=(239,8922)` on salts 205, 208, and 207.
- Public line replay derives the correct target67 secret 3588 under the line
  gate, but not below rho: factor-gate replay bottoms out at 1.304 rho, and
  x-match orientation bottoms out at 1.272 measured / 1.24 shared-leaf-hit-root.
  The validation-favorable post-line activation rule activates this case, but
  it does not turn it into a below-rho replay.
- Because all selected leaves share the public monic point `(b,c)=(239,8922)`,
  two canonical coordinate gates were tested without using the exact factor
  slope:
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_transfer653_targetcap3_b239_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_gate_replay_target67_transfer653_targetcap3_c8922_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_transfer653_targetcap3_b239_all_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_target67_transfer653_targetcap3_c8922_all_648_655.json`
  Both `b=239` and `c=8922` select the same leaf-3 rows and derive the same
  secret 3588 with the same over-rho costs: 1.304 by factor-gate replay and
  1.272 measured / 1.24 shared-leaf-hit-root by orientation.  This is a
  stronger pre-exact mechanism lead than the particular Sage-discovered line,
  although it still does not beat rho.
- The split line-stage audit has 6 preserving lines, 4 replay-success rows, and
  2 line-present/no-replay rows.  No additive line+replay or row+line+replay
  accounting beats rho; the minimum line+replay charge is 1.896 rho.

Current boundary after 648-655:

648-655 is not a fresh frozen-ledger success, but it is a useful exact-line
boundary.  The target67 transfer653 line is real and repeated, and the selected
leaf-coordinate point can be recovered by canonical public coordinate gates
without knowing the exact factor slope.  However, public replay cost remains
over rho.  This keeps the candidate family alive as a repeated-coordinate
amortization/pre-exact-gate problem, not as a promoted index-calculus speedup
yet.

656-663 forward validation and repeated-coordinate miner:

- Fresh 656-663 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_656_663_probe.json`
- Target-cap1 again has no reach: 0 verified rows and 0 verified leaves.
  Target-cap3 has 22 verified rows, global-cap3 has 21 verified rows, and all
  three fixed policies have 0 verified leaves and 0 below-rho rows/leaves.
- Fresh 656-663 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_656_663.json`
- The source has 172 public bounded cases and zero verifier-positive labels.
  The best 22050 case is again a below-rho public-cost decoy at 0.94160584,
  rank 0/relation 0.  The best target67 case is over rho at 1.224, rank
  0/relation 0.
- Frozen ledgers:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_656_663_pair56_s206_precise.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_656_663.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_656_663.json`
- The 22050 scheduler, strict target67 branch, and broad target67 branch all
  abstain with zero selected cases.  Rolling target67 audits through 656-663
  are:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_663.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_663.json`
- Strict rank-dedup now covers 17 windows and remains at 8 selected cases, 7
  verified-below-rho cases, and 1 false positive; precision when selected is
  still 0.875.  Broad remains 9 verified-below-rho selections out of 11
  selected cases, with 2 false positives and precision 0.81818182.
- New pre-exact repeated-coordinate miner script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_gate_miner.py`
- Coordinate miner artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_gate_miner_target67_transfer653_targetcap3_648_655.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_gate_miner_target67_632_639_fullreplay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_gate_miner_target67_640_647_fullreplay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_gate_miner_target67_648_655_fullreplay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_gate_miner_target67_656_663_fullreplay.json`
- The miner uses only public selected rows/leaves to materialize monic
  coordinates, then ranks repeated `(b,c)` coordinates before any exact Sage
  factorization.  It can replay exact-coordinate gates and the corresponding
  `b=` / `c=` axis gates for audit.
- On the 648-655 transfer653 target-cap3 slice, `(b,c)=(239,8922)` is surfaced
  from public coordinates alone.  Exact-coordinate, `b=239`, and `c=8922`
  gates all derive secret 3588, but at 1.304 rho.
- Window-level target67 coordinate replay:
  - 632-639: 115 repeated-coordinate candidates, 39 source-verified labels,
    35 coordinate-gate verified replays, 0 below rho; best verified replay cost
    is 1.336 rho.  Prominent coordinates include `(161,6976)` and
    `(1114,8506)`.
  - 640-647: 72 repeated-coordinate candidates, 0 source-verified labels, 0
    verified coordinate replays.
  - 648-655: 84 repeated-coordinate candidates, 8 source-verified labels, 8
    coordinate-gate verified replays, 0 below rho; all verified coordinate
    replays use `(239,8922)` and derive secret 3588 at 1.304 rho.
  - 656-663: 108 repeated-coordinate candidates, 0 source-verified labels, 0
    verified coordinate replays.
- Important negative/control detail: `(239,8922)` reappears in 656-663 under
  global-cap3 at transfer 658, but it is rank 0/relation 0 and does not verify.
  Therefore coordinate identity alone is not a selector.  The remaining
  algorithmic gap is a public activation rule that distinguishes verified
  repeated-coordinate windows from cheap repeated-coordinate decoys, plus an
  amortized replay charge below rho.

Current boundary after 656-663:

The frozen branch story is another abstention/negative window.  The new
progress is the pre-exact coordinate miner: it proves that some target67
line-gate recoveries can be recast as repeated public monic-coordinate gates
without exact factor leakage.  The honest limitation is now sharper: repeated
coordinates recover secrets in the 632-639 and 648-655 positive windows, but
all verified coordinate replays are still over rho and the same coordinates can
also occur in false windows.  This is a better mechanism, not yet a speedup.

664-671 forward validation and coordinate activation rule:

- Fresh 664-671 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_664_671_probe.json`
- Target-cap1 again has no reach: 0 verified rows and 0 verified leaves.
  Target-cap3 has 19 verified rows and 4 verified leaves, global-cap3 has 13
  verified rows, and all three fixed policies have 0 below-rho rows/leaves.
  The best target-cap3 leaf is still over rho at 1.05109489.
- Fresh 664-671 public bounded source:
  `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_664_671.json`
- The source has 128 public bounded cases and 4 verifier-positive labels.  The
  best 22050 case is a below-rho public-cost decoy at 0.94160584, rank
  0/relation 0.  The best target67 case is over rho at 1.224, rank
  0/relation 0.
- Frozen ledgers:
  `ecdlp_index_calculus_state/ffe_public_branch_scheduler_validation_664_671_pair56_s206_precise.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_rank_dedup_validation_664_671.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_predictor_broad_lowterm_rank_dedup_validation_664_671.json`
- The 22050 scheduler, strict target67 branch, and broad target67 branch all
  abstain with zero selected cases.  Rolling target67 audits through 664-671
  are:
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_rank_dedup_528_671.json`
  `ecdlp_index_calculus_state/ffe_public_target67_family_branch_audit_broad_lowterm_rank_dedup_528_671.json`
- Strict rank-dedup now covers 18 windows and remains at 8 selected cases, 7
  verified-below-rho cases, and 1 false positive; precision when selected is
  still 0.875.  Broad remains 9 verified-below-rho selections out of 11
  selected cases, with 2 false positives and precision 0.81818182.
- New pre-exact repeated-coordinate activation rule miner script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_activation_rule_miner.py`
- Training artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_activation_rule_miner_target67_632_663_no_exact_coord.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_activation_rule_miner_target67_632_663_with_exact_coord.json`
- The best frozen rule, even without exact coordinate features, is:
  `activate:b_minus_c_mod16=0&source_ops_millirhos>=1312|c_mod16=10&source_ops_millirhos>=1368`
- On 632-639 plus 648-655 training positives it selects 29 of 43 verifier-
  positive repeated-coordinate replays, with 0 false positives, precision 1.0,
  recall 0.6744186, minimum verified replay cost 1.304 rho, and mean verified
  replay cost 1.36303448 rho.  Allowing exact coordinate features yields the
  same top rule, so the current best rule is not simply memorizing `(239,8922)`
  or another exact coordinate.
- Fresh 664-671 repeated-coordinate replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_gate_miner_target67_664_671_fullreplay.json`
- On 664-671 the miner finds 64 repeated-coordinate candidates and replays all
  64, but there are 0 source verifier-positive labels, 0 verified coordinate
  replays, and 0 below-rho replays.  Minimum replayed public cost is 1.224 rho.
- Frozen activation evaluation:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_activation_rule_eval_target67_632_663_rule_on_664_671.json`
- The frozen rule selects 0 validation cases across 640-647, 656-663, and
  664-671, so the activation surface has zero false positives on three negative
  windows.  It still only recovers over-rho training positives, so it is a
  cleaner pre-exact filter rather than an end-to-end speedup.

Current boundary after 664-671:

The repeated-coordinate activation rule is now an honest public pre-exact
selector: it keeps the known 632-639 and 648-655 coordinate-gate positives while
abstaining on three negative windows.  That is progress on the "which repeated
coordinate should be tested" problem.  The remaining blocker is unchanged but
sharper: every verified coordinate replay remains over rho, and fresh 664-671
adds no verifier-positive target67 coordinate replay.  The next useful step is
therefore not to widen the rule blindly; it is to search for an amortizable
coordinate family or a cheaper confirmation path that can be charged below rho
after the rule has frozen.

Repeated-coordinate amortization audit through 664-671:

- New audit script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_amortization_probe.py`
- Frozen-rule activated audit:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_amortization_probe_target67_632_671_activated_axis.json`
- All-coordinate sanity audit:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_amortization_probe_target67_632_671_all_axis.json`
- The audit groups exact `(b,c)` gates plus `b=` and `c=` axis gates, then
  dedupes policy/selector/top-k aliases down to unique `(gate, transfer,
  derived secret)` recoveries before computing break-even counts.
- Activated-only audit result:
  - 87 activated gate records after including exact, `b=`, and `c=` gates.
  - 12 unique verified gate recoveries under the gate-kind-aware dedupe, but no
    direct replay below rho.
  - 9 gate groups total; exactly 3 groups have at least two unique recoveries:
    exact `(1114,8506)`, `b=1114`, and `c=8506`.
  - All three reuse candidates are the same transfer pair: transfers 632 and
    636, secrets 3531 and 7331, maximum unique replay cost 1.368 rho.
  - With only two unique recoveries, the replay charge would need at least a
    0.5380117 reusable fraction to cross rho.  A 0.5 reusable fraction would
    require three unique successes; 0.75 or 1.0 would make the observed two
    enough in the break-even model.
  - The additive source-plus-replay proxy is not feasible with a <=1.0 reusable
    fraction; the best required fraction is null / greater than one.  This
    keeps row-selection-plus-replay accounting out of claim territory.
- All-coordinate sanity audit result:
  - 1329 exact/axis gate records across 632-671, 44 gate groups, 129 verified
    gate records, and 15 unique verified gate recoveries after gate-kind-aware
    dedupe.
  - The best reuse candidate is still `(1114,8506)` and its axes, with the same
    two unique verified recoveries and the same 0.5380117 replay-charge reusable
    fraction requirement.
  - Unactivated repeated coordinates such as `(161,6976)` recur across many
    transfers but have only one unique verified recovery, so widening the rule
    would add repeated decoys rather than a stronger amortization family.

Current boundary after the amortization audit:

The coordinate path now has a concrete amortization target instead of a vague
"maybe reuse helps" note.  Under the frozen activation rule, the only observed
two-recovery family is `(1114,8506)` / `b=1114` / `c=8506`, and it still needs
more than half of the measured replay charge to become reusable before a
two-use below-rho claim is plausible.  Since no separate coordinate-confirmation
timer has been measured, this remains a break-even audit, not an ECDLP speedup.
The next technical move should be to instrument the coordinate replay path into
fixed coordinate confirmation versus per-row relation replay, or to find a
fresh third unique `(1114,8506)`-style recovery that lowers the required
reusable fraction.

Coordinate replay row-cost decomposition:

- New decomposition script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_cost_decomposition_probe.py`
- Decomposition artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord1114_8506_632_671.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord239_8922_632_671.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord1219_8787_632_671.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord161_6976_632_671.json`
- The script reruns verified exact-coordinate recoveries with full row scan
  details and evaluates all row subsets.  It does not claim subset pruning is
  public; subset wins are diagnostics until the same row-pruning rule is
  predicted before relation events.
- Across the five unique verified coordinate recoveries sampled here:
  - Full exact-coordinate replay costs remain over rho: min 1.304, max 1.384.
  - Every recovery has a verifier-informed below-rho subset.
  - Best subset costs:
    - `(1114,8506)` transfer 636, secret 7331: two rows, salts 203 and 205,
      0.928 rho versus full 1.336.
    - `(1114,8506)` transfer 632, secret 3531: two rows, salts 206 and 207,
      0.96 rho versus full 1.368.
    - `(239,8922)` transfer 653, secret 3588: two rows, salts 205 and 208,
      0.896 rho versus full 1.304.
    - `(1219,8787)` transfer 634, secret 650: one row, salt 202, 0.512 rho
      versus full 1.384.
    - `(161,6976)` transfer 633, secret 5110: one row, salt 208, 0.48 rho
      versus full 1.336.  This coordinate is not selected by the current frozen
      activation rule, but it is useful as a row-cost diagnostic.
- Row-level pattern:
  - All zero-event rows have `selected_hit_roots=0`, `selected_hit_events=0`,
    and preassociation cost 51.
  - All relation-bearing rows have `selected_hit_roots=1`; one-event rows cost
    56, 58, or 60 ops, while two-event rows cost 60 or 64 ops.
  - The aggregate row event-count distribution over these five recoveries is:
    5 zero-event rows, 8 one-event rows, and 2 two-event rows.

Current boundary after row-cost decomposition:

The algebraic coordinate gate is stronger than the raw full-replay cost made it
look: once relation-bearing rows are known, every sampled verified coordinate
recovery beats rho, and two of them can even recover from a single row with two
independent forms.  The remaining blocker has shifted from "is coordinate replay
intrinsically over rho?" to "can root-positive/relation-bearing rows be selected
publicly or tested cheaply enough before paying for dead rows?"  The exact
diagnostic signal is `selected_hit_roots > 0`, but the current cost model charges
the preassociation work needed to learn it, so it cannot be treated as a free
selector.  The next experiment should mine or implement a cheaper public
root-positive row predictor for repeated coordinates.

Public root-positive row predictor and replay:

- New row miner:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_root_positive_row_miner.py`
- New row-rule replay probe:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_row_rule_replay_probe.py`
- Row miner artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_root_positive_row_miner_target67_632_671_activated_no_exact.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_root_positive_row_miner_target67_632_671_all_no_exact.json`
- Row-rule replay artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_rule_replay_target67_632_671_activated_rule.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_rule_replay_target67_632_671_all_rule.json`
- The row miner uses `selected_hit_roots > 0` as the training/evaluation label,
  but rule predicates are limited to public row, salt, transfer, selector,
  policy, top-k, and coordinate residue features.  Final public-key
  verification is not used as a predicate.
- Activated-surface root-positive rule:
  `row_activate:b_minus_c_mod16=0&salt_mod4=3|b_mod16=10&salt_delta_to_max=1|salt_delta_from_min<=1&salt_mod2=0`
  - Training surface: 87 activated rows, 58 root-positive rows, and 58
    relation-positive rows.
  - The rule selects 47 rows with precision 1.0 and recall 0.81034483 on the
    activated training rows.
  - The frozen coordinate activation rule abstains on validation windows
    640-647, 656-663, and 664-671, so this activated row rule has no independent
    validation rows yet.
- Activated row-rule replay:
  - 29 input coordinate cases, 27 row-selected cases, 27 public-key-verified
    cases, and 27 verified below-rho cases.
  - Unique verified recoveries are:
    - `(1219,8787)` transfer 634, one selected row, secret 650,
      0.512 rho.
    - `(1114,8506)` transfer 636, two selected rows, secret 7331,
      0.928 rho.
    - `(1114,8506)` transfer 632, two selected rows, secret 3531,
      0.960 rho.
  - This is the first measured public row-pruned coordinate replay that beats
    rho under the current cost ledger, but it remains in-sample with respect to
    the activated coordinate surface.
- All-coordinate root-positive rule:
  `row_activate:b_mod5=4&salt=203|leaf_min_mod8=4&salt_delta_from_min=1|salt_mod4=3&transfer_mod16=8`
  - Across 632-671 it sees 1329 rows, 334 root-positive rows, and 210
    relation-positive rows.
  - Train split: precision 1.0, recall 0.25287356.
  - Validation split: precision 1.0, recall 0.275 over root-positive rows.
- All-coordinate row-rule replay:
  - 443 input cases, 76 row-selected cases, 12 public-key-verified cases, and
    12 verified below-rho cases.
  - The only unique verified recovery is `(1114,8506)` transfer 632, secret
    3531, at 0.960 rho; it is in the training window.
  - Validation rows are selected in 656-663 and 664-671, but each selected case
    has at most one relation and no public-key verification.  The selected
    validation diagnostics are `(429,3910)` transfer 658, `(281,3224)` transfer
    661, `(861,1638)` transfer 664, and `(429,3910)` transfer 665.

Current boundary after public row-rule replay:

The repeated-coordinate branch now has a real sub-rho mechanism: public features
can prune exact-coordinate replay down to relation-bearing rows on the 632-639
activated surface, and the replay verifies below rho without using verifier
labels inside the rule.  The honest boundary is still strict: the activated
coordinate rule abstained on the held-out validation windows, and the broader
row rule validated only root-positive row prediction, not full secret recovery.
The next step is to freeze the best row rule and run it on the next fresh
coordinate-positive window, or mine a two-row/rank-completion rule that turns
the validation root hits into enough independent relations before claiming a
fresh speedup.

Public row-pair/rank-completion miner:

- New pair-completion miner:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_row_pair_completion_miner.py`
- Pair-completion artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_pair_completion_miner_target67_632_671_all_no_exact.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_pair_completion_miner_target67_632_671_all_no_exact_clause1.json`
- Pair-rule replay artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_rule_replay_target67_632_671_pair_completion_rule.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_rule_replay_target67_632_671_sourceops1448_rule.json`
- The miner scores public row predicates at the coordinate-case level.  A case
  is relation-complete when selected rows carry at least two relation events.
  Public-key verification is deliberately left to the replay probe.
- The all-coordinate row scan over 632-671 contains 443 candidate cases:
  67 relation-complete cases, 80 one-relation cases, 334 root-positive rows,
  and 210 relation-positive rows.
- Held-out validation windows 640-647, 656-663, and 664-671 are not empty:
  they contain 244 candidate cases, 20 relation-complete cases, 40 one-relation
  cases, 160 root-positive rows, and 76 relation-positive rows.
- One-clause sanity:
  - Best rule is `row_activate:candidate_activation_selected=1`, which simply
    rediscovers the frozen coordinate activation.
  - It selects 29 train relation-complete cases with zero train incomplete
    cases, but selects zero validation cases because the coordinate activation
    abstains on validation.
  - The validation-reaching single bucket `row_activate:source_ops_millirhos=1448`
    selects two validation relation-complete cases and zero validation
    incomplete cases, but replay verifies zero public keys.  Its selected cases
    cost at least 1.296 rho and the held-out `(281,3224)` case reaches rank 2
    at 1.368 rho without deriving the secret.
- Broad pair-completion rule:
  `row_activate:leaf_index>=10&source_ops_millirhos=1344|leaf_min_mod8=4&salt_min=206|source_ops_millirhos=1336&transfer_mod3=0`
  - Train score: 11 relation-complete selections, 0 incomplete selections,
    recall 0.23404255, mean complete cost 1.34763636 rho.
  - Validation score: 16 relation-complete selections, 0 incomplete selections,
    recall 0.8, mean complete cost 1.36 rho.
  - Replay selects 27 cases total: 11 verified in train, none below rho, and
    zero verified in validation.
  - Held-out replay diagnostics:
    - 656-663: `(281,3224)` transfer 661 reaches 3 selected rows, relation
      count 2, rank 2, unique forms 2, but public-key verification is false at
      1.368 rho.
    - 664-671: `(1145,8648)` transfer 666 reaches 3 selected rows, relation
      count 2, rank 2, unique forms 2, but public-key verification is false at
      1.336 rho.

Current boundary after row-pair mining:

The validation windows do expose public row patterns with enough relation
events and rank, but relation-complete is not the same as key-consistent.  The
blocker has moved one notch deeper: root-positive row prediction and row-pair
completion can both be mined from public metadata, yet held-out rank-2 cases
still fail public-key verification and remain over rho.  The next meaningful
surface is orientation/form consistency inside the selected relation events,
or a public pre-replay test that distinguishes key-consistent forms from
rank-only decoys.

Public form-orientation audit:

- New form audit:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_form_orientation_audit.py`
- Audit artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_orientation_audit_target67_632_671_row_rules.json`
- The audit replays three row-rule surfaces and records public relation-form
  features before treating public-key verification as the label:
  activated row rule, all-coordinate root-positive row rule, and the
  pair-completion rule.
- Across these row-rule surfaces:
  - 154 selected cases have at least one relation event.
  - 70 cases are relation-complete with relation count at least 2.
  - 50/70 relation-complete cases are public-key verified.
  - 39/70 relation-complete cases are verified below rho.
  - Minimum relation-case cost is 0.512 rho and mean relation-case cost is
    1.04057143 rho.
- The strongest public form-orientation atom is `candidate_pos_span=0`:
  - It selects 50/70 relation-complete cases.
  - All 50 selected cases are public-key verified.
  - It rejects all 20 unverified relation-complete cases.
  - It preserves all 39 verified-below-rho cases.
- By row-rule surface under `candidate_pos_span=0`:
  - Activated row rule: 27 selected, 27 verified, 27 below rho, cost range
    0.512 to 0.960 rho.
  - All-coordinate root rule: 12 selected, 12 verified, 12 below rho, cost
    0.960 rho.
  - Pair-completion rule: 11 selected, 11 verified, 0 below rho, cost range
    1.336 to 1.368 rho.
- Unique below-rho wins preserved by the guard:
  - Activated `(1219,8787)` transfer 634 at 0.512 rho with
    `candidate_pos_signature=1,1`.
  - Activated `(1114,8506)` transfer 636 at 0.928 rho with
    `candidate_pos_signature=3,3`.
  - Activated/all-root `(1114,8506)` transfer 632 at 0.960 rho with
    `candidate_pos_signature=1,1`.
- Rejected hard negatives:
  - Pair-completion `(281,3224)` transfer 661 has
    `candidate_pos_signature=1,2`, relation count 2, rank 2, unique forms 2,
    but public-key verification false at 1.368 rho.
  - Pair-completion `(1145,8648)` transfer 666 has
    `candidate_pos_signature=1,3`, relation count 2, rank 2, unique forms 2,
    but public-key verification false at 1.336 rho.
- No ready 672-679 repeated-coordinate gate artifact was present in the current
  writable worktree or mounted state, so this remains a frozen-candidate
  orientation guard awaiting the next fresh coordinate-positive artifact.

Current boundary after form-orientation audit:

The coordinate path now has a three-stage candidate mechanism:
public repeated-coordinate row pruning, public relation-form orientation via
`candidate_pos_span=0`, and replay verification.  The guard cleanly separates
all current key-consistent relation systems from rank-only decoys in 632-671
and preserves the below-rho row-pruned wins.  It is not yet a fresh algorithmic
claim because `candidate_pos_span` is observed after relation-event scanning,
and the next-window gate artifact does not yet exist here.  The next promotion
test is to freeze this guard and evaluate it unchanged on the next
coordinate-positive window.

Frozen form-guard replay wrapper:

- New guard replay probe:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_form_guard_replay_probe.py`
- Guard replay artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_632_671_candidate_pos_span0.json`
- This probe makes the orientation audit operational: row rules select public
  row/leaf work, relation events are scanned, and a frozen form guard decides
  whether the event system is eligible for derivation accounting.
- Control run over 632-671 with `candidate_pos_span=0` and the activated,
  all-root, and pair-completion row rules:
  - 422 row-rule-selected cases.
  - 70 relation-complete cases.
  - 50 guard-passed relation-complete cases.
  - 50 guard-passed public-key-verified cases.
  - 39 guard-passed below-rho cases.
  - 20 guard-rejected relation-complete cases.
  - 0 rejected-but-verified cases.
  - Guard-passed cost range: min 0.512 rho, mean 0.98 rho.
- By row-rule surface:
  - Activated row rule: 31 relation-complete cases, 27 guard-passed, 27
    verified, 27 below rho, 0 rejected-but-verified.
  - All-coordinate root rule: 12 relation-complete cases, 12 guard-passed, 12
    verified, 12 below rho, 0 rejected-but-verified.
  - Pair-completion rule: 27 relation-complete cases, 11 guard-passed, 11
    verified, 0 below rho, 0 rejected-but-verified.

Current boundary after frozen guard replay:

The guard is now a reusable probe rather than a notebook observation.  It gives
the next fresh-window command a fixed selection surface: public row rule plus
`candidate_pos_span=0`, with all row-scan costs still charged.  The control run
confirms no known verified system is lost, but it is still not fresh validation.
The next required artifact is a new repeated-coordinate gate window after
664-671 or an equivalent fresh target67 coordinate-positive source.

Frozen pipeline runner control:

- New end-to-end runner:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_frozen_pipeline_runner.py`
- Control manifest:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_664_671_control_manifest.json`
- Control artifacts:
  - Public selector:
    `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_664_671_control_public_selector.json`
  - Repeated-coordinate gate:
    `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_664_671_control_coordinate_gate.json`
  - Frozen form-guard replay:
    `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_664_671_control_guard_replay.json`
- The runner promotes the current frozen branch from a sequence of manual
  commands into a reproducible pipeline: public low-term selector, repeated
  coordinate gate miner, then frozen row-rule replay under
  `candidate_pos_span=0`.
- Control over the known 664-671 stress source selected 128 public cases
  across targets `22050.cf1@11731` and `67.a1@9803`.
- For target `67.a1@9803`, the coordinate gate replayed 64 candidates.  The
  minimum replayed cost was 1.224 rho, mean replayed cost was 1.2675 rho, and
  no replayed candidate was verifier-positive or below rho.
- The frozen guard replay saw 4 relation-complete activated-rule cases, rejected
  all 4 under `candidate_pos_span=0`, and had 0 rejected-but-verified cases.

Current boundary after frozen pipeline runner control:

The promotion path is now one command per fresh stress window.  The 664-671
control is a clean negative for target 67, which is useful: it shows the frozen
guard does not turn rank-only relation events into claimed recoveries and that
the manifest preserves the selector/gate/guard summaries together.  It is still
not fresh validation because 664-671 already informed the repeated-coordinate
branch.  The next genuine promotion test is to run the same runner unchanged on
the first available post-671 coordinate-positive stress source, especially a
672-679 source if it appears.

Fresh 672-679 promotion test:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_672_679_probe.json`
- Target-67 frozen pipeline manifest:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_672_679_fresh_manifest.json`
- Target-22050 coordinate control manifest:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target22050_672_679_fresh_manifest.json`
- The fresh stress source is positive overall: the frozen
  `target_cap1_ow1_hw3_lw0_sw0_cw0_aw0` row selector has 4 verifier-backed
  below-rho leaf cases, best ops/rho 0.39416058.
- The best public 22050 case is transfer 677, salt 164, leaves `[65,79,90]`,
  rank 2, relation count 2, and public-key verified at 0.39416058 rho.
- The best public target-67 case by cost is transfer 673, salt 208, leaves
  `[3,6,10]`, at 0.424 rho, but it is rank 0 and verifier-negative.
- Target-67 repeated-coordinate gate:
  - 152 candidates replayed.
  - 8 verifier-positive replayed candidates.
  - 0 below-rho verifier-positive replayed candidates.
  - Minimum replayed cost 1.224 rho, mean replayed cost 1.26252632 rho.
  - The verifier-positive family is coordinate `(161,6976)` on transfer 677,
    deriving secret 9544 at 1.392 rho with rank 2 and relation count 2.
- The frozen target-67 row/form guard does not promote the fresh window:
  relation-case count 0 and guarded-verified count 0 under the previously
  frozen activated/all-root row rules plus `candidate_pos_span=0`.
- Target-22050 repeated-coordinate control:
  - 52 candidates replayed.
  - Minimum replayed cost 0.94160584 rho, mean replayed cost 0.96967996 rho.
  - 0 verifier-positive repeated-coordinate candidates.
  - The actual 22050 below-rho win in this fresh window is therefore not
    carried by the repeated-coordinate gate.

Fresh coordinate-161 decomposition:

- Fresh cost decomposition:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord161_6976_672_679_fresh.json`
- Combined 632-679 decomposition:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord161_6976_632_679_with_fresh.json`
- Diagnostic row-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord161_saltmod5_3_632_679_with_fresh.json`
- The fresh full `(161,6976)` replay costs 1.392 rho, but the
  verifier-informed row subset `{salt203,salt208}` derives the same secret
  9544 at 0.928 rho.
- Combining prior 632-671 and fresh 672-679 gives two coordinate-161
  recoveries:
  - transfer 633 derives secret 5110; full replay 1.336 rho, best below-rho
    subset `{salt208}` at 0.48 rho.
  - transfer 677 derives secret 9544; full replay 1.392 rho, best below-rho
    subset `{salt203,salt208}` at 0.928 rho.
- A simple public diagnostic row rule
  `row_activate:coordinate_key=161,6976&leaf_min=2&salt_mod5=3`, followed by
  `candidate_pos_span=0`, selects 16 relation-complete duplicate cases across
  the prior plus fresh artifacts.  All 16 are verified and below rho, with
  min cost 0.48 rho and mean cost 0.704 rho.

Current boundary after 672-679:

The original frozen repeated-coordinate guard failed to promote on fresh
target 67, but the failure is now more informative than a plain negative.  The
coordinate `(161,6976)` recurs in a fresh window, has a below-rho
relation-bearing row subset, and exposes a compact public candidate rule
(`leaf_min=2`, `salt_mod5=3`, `candidate_pos_span=0`).  This is not yet a
fresh algorithmic claim because the salt-mod rule was identified after seeing
the 672-679 decomposition.  It is, however, a better next frozen candidate than
the older activated/all-root row rules and should be tested unchanged on the
next post-679 coordinate-positive target-67 window.

Fresh 680-687 holdout:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_680_687_probe.json`
- Target-67 frozen pipeline manifest:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_680_687_fresh_manifest.json`
- Coord161 replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord161_saltmod5_3_680_687_fresh.json`
- The stress source was verifier-positive but over rho: target-cap3 selected
  20 verified leaf cases, 0 below rho, best 1.368 rho.
- Target-67 coordinate gate replayed 101 candidates, found 33 verified
  candidates, and found 0 below-rho verified candidates.  Best verified
  full-coordinate replays were `(757,1452)` on transfer 683 at 1.368 rho and
  `(55,7531)` on transfer 686 at 1.44 rho.
- The frozen coord161 rule did not activate on this window: input case count 0.
  This is a coordinate-absence miss, not a false positive.
- Cost decomposition exposed verifier-informed below-rho subsets:
  - `(757,1452)` derives secret 95 from salts 202 and 209 at 0.96 rho.
  - `(55,7531)` derives secret 2256 from salt 208 at 0.568 rho.
- The no-exact row miner trained on 664-671 and 672-679, with 680-687 as a
  scoring window, found
  `row_activate:salt<=201&source_ops_millirhos>=1368|salt_max=209&salt_mod5=2|salt_mod4=0&source_ops_millirhos>=1440`.
  It had validation precision 1.0 and recall 0.64179104 at row-label level on
  680-687, but replay under `candidate_pos_span=0` rejected all 29 verified
  relation cases.  Diagnostic `form_guard=all` verified those cases at 1.032
  rho, still over rho.
- Exact mined coordinate-subset replay on 680-687 is diagnostic only.  It
  confirms `(757,1452)` salts 202/209 pass `candidate_pos_span=0` and stay
  below rho at 0.96.  The cheaper `(55,7531)` salt208 subset verifies below
  rho at 0.568 but fails `candidate_pos_span=0` with candidate positions
  `[1,3,3]`.

Fresh 688-695 true holdout:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_688_695_probe.json`
- Target-67 frozen pipeline manifest:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_688_695_fresh_manifest.json`
- Row-rule holdout replays:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord161_saltmod5_3_688_695_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_miner664_680_no_exact_best_688_695_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_mined_exact_coord_subsets_688_695_fresh.json`
- The fresh stress source is positive overall: target-cap1 has 4 verified
  below-rho leaf cases and 1 verified below-rho row case, with best ops/rho
  0.43065693 on `22050.cf1@11731`.
- Target-67 repeated-coordinate gate replayed 68 candidates, found 12 verified
  full-coordinate replays, and found 0 below-rho verified full replays.  The
  best verified full replay is `(117,6119)` on transfer 695 at 1.304 rho;
  `(861,9226)` on transfer 692 costs 1.368 rho.
- Frozen row-rule holdout result:
  - coord161 selected 4 cases, relation-case count 0.
  - the 680-derived no-exact salt/cost rule selected 10 cases, relation-case
    count 0.
  - the 680 mined exact-coordinate recipes selected 8 cases, relation-case
    count 0.
- Row-label scoring of the 680-derived no-exact rule on 688-695 confirms
  overfit: validation precision 0.18181818, recall 0.02, 2 true-positive rows
  and 9 false positives.
- New 688-695 cost decompositions again expose below-rho relation-row subsets:
  - `(117,6119)` derives secret 9438 from salts 205 and 206 at 0.896 rho.
  - `(861,9226)` derives secret 8677 from salts 206 and 208 at 0.96 rho.
- Exact mined local subset replay on 688-695 is diagnostic only, but both new
  two-row subsets pass `candidate_pos_span=0`: 12/12 duplicate cases guarded,
  verified, and below rho; min 0.896 rho, mean 0.93866667 rho.

Current boundary after 688-695:

The repeated-coordinate path has a stable algebraic pattern but not yet a
public predictor.  Across 672-679, 680-687, and 688-695, full target-67
coordinate replays are over rho, while verifier-informed relation-row subsets
regularly fall below rho and often satisfy `candidate_pos_span=0`.  The
coord161 rule did not false-positive on later windows, but it missed by
coordinate absence.  The first no-exact salt/cost predictor was overfit and
failed the 688-695 holdout.  The next blocker is therefore public prediction of
the relation-bearing two-row subset, not relation algebra or form orientation.

Subset-pair miner and 696-719 forward tests:

- New miner:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_subset_pair_rule_miner.py`
- New replay wrapper:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_subset_pair_rule_replay_probe.py`
- The miner labels the best below-rho verified subset from cost-decomposition
  artifacts and searches public pair predicates.  Exact coordinate and exact
  salt features are disabled unless explicitly requested.
- Training on 672-679 and 680-687, with 688-695 held out, produced the no-exact
  pair rule
  `pair_activate:b_minus_c_mod16=3&pair_salt_delta_from_min_signature=2,4|pair_salt_index_span>=1&pair_salt_sum_mod5=1`.
  Label scoring on 688-695 was 2/2 true positives with 0 false positives.
- Replaying that frozen pair rule on 688-695 reproduced 12 guarded verified
  below-rho duplicate cases, min 0.896 rho and mean 0.93866667 rho.
- Fresh 696-703:
  - Stress source:
    `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_696_703_probe.json`
  - Target-67 coordinate gate:
    `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_696_703_fresh_coordinate_gate.json`
  - The stress source had a fresh target-cap1 below-rho source signal: 4
    verified below-rho leaf cases and 4 verified below-rho row cases, best
    0.43065693 rho.
  - Target-67 repeated-coordinate gate abstained: 52 replayed candidates, 0
    verifier-positive full replays.
  - Frozen row and pair rules selected some rows/pairs but produced no
    guarded verified relation system, so 696-703 is an abstention window for
    the repeated-coordinate branch.
- Fresh 704-711:
  - Stress source:
    `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_704_711_probe.json`
  - Target-67 coordinate gate:
    `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_704_711_fresh_coordinate_gate.json`
  - Cost decomposition:
    `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord55_7531_704_711_fresh.json`
  - Target-67 had 12 verified full coordinate replays, all over rho, best
    `(55,7531)` transfer 706 at 1.336 rho.
  - Decomposition again exposed a below-rho verified subset: salts 203/205
    derive secret 3713 at 0.928 rho.
  - The original 672/680 pair rule missed this window.  A widened no-exact
    candidate chosen after seeing 704,
    `pair_activate:b_minus_c_mod16=12&pair_salt_sum_mod8=0|b_minus_c_mod16=3&pair_salt_delta_from_min_signature=2,4|pair_salt_index_span>=1&pair_salt_sum_mod8=3`,
    replays the 704 pair below rho at 0.928.  This is post-704 and must be
    validated later.
- Fresh 712-719:
  - Stress source:
    `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_712_719_probe.json`
  - Target-67 coordinate gate:
    `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_712_719_fresh_coordinate_gate.json`
  - Cost decompositions:
    - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord55_7531_712_719_fresh.json`
    - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord757_1452_712_719_fresh.json`
  - Target-67 had 8 verified full coordinate replays, all over rho: `(55,7531)`
    transfer 718 at 1.304 rho and `(757,1452)` transfer 715 at 1.36 rho.
  - Decomposition found below-rho subsets for both:
    - `(55,7531)`, salts 204/206, secret 9797 at 0.896 rho.
    - `(757,1452)`, singleton salt209, secret 702 at 0.544 rho.
  - The original 672/680 no-exact pair rule forward-validates on this unseen
    window: replay selected `(757,1452)` salts 202/209, passed
    `candidate_pos_span=0`, derived secret 702 at 0.952 rho, and produced 4
    guarded verified below-rho duplicate cases.
  - The post-704 widened candidate also validates on 712-719, but it adds no
    extra evidence over the older rule for this specific window.

Current boundary after 712-719:

This is the first fresh target-67 repeated-coordinate pair-rule replay that
beats rho without exact coordinates or exact salts.  It is not yet a general
ECDLP break: the rule missed 704-711, and the label miner's "best subset"
metric can undercount valid below-rho replays when a non-minimal pair contains
a singleton relation-bearing row.  Still, the path has progressed from
verifier-informed subset diagnostics to a public, frozen pair predicate with a
clean forward below-rho replay on 712-719.  The next proof obligation is a
second post-712 forward validation, preferably on a window whose coordinate
family was not already seen in 680-687.

Fresh 720-727 pair-rule validation:

- Stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_720_727_probe.json`
- Target-67 frozen pipeline artifacts:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_720_727_fresh_public_selector.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_720_727_fresh_coordinate_gate.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_720_727_fresh_guard_replay.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_720_727_fresh_manifest.json`
- Decomposition and replay checks:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord161_6976_720_727_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_680_no_exact_720_727_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_704_candidate_720_727_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord161_saltmod5_3_720_727_fresh.json`
- The stress source again verified target-67 relation signals but not a
  below-rho leaf recovery under the selected global policy: 8 verified stress
  leaves, 16 verified stress rows, 0 stress-leaf below-rho cases, and best
  stress-leaf cost 1.00729927 rho.
- The target-67 coordinate gate had 72 candidates.  Four full coordinate
  replays verified, all the same coordinate `(161,6976)` on transfer 721,
  deriving secret 1767 at 1.44 rho with rank 2 and two relations.  No full
  coordinate replay was below rho.
- Cost decomposition improved the verified replay from 1.44 rho to the
  two-row subset salts 206/207 at 1.032 rho, still just over rho.  Thus this
  window is a repeated-coordinate recurrence but not a verifier-informed
  below-rho subset win.
- The original 672/680 no-exact pair rule selected 20 public pairs, but none
  were relation-bearing after the guard; guarded verified count 0 and
  guarded below-rho count 0.
- The widened post-704 no-exact candidate behaved the same on this window:
  20 selected public pairs, 0 relation pairs, 0 guarded verified replays.
- The old coord161 salt-mod row rule did not activate at all on 720-727
  (`input_case_count=0`), so it neither validates nor false-positives here.

Current boundary after 720-727:

The second post-712 validation did not materialize.  This is not a direct
failure of the pair-rule branch, because the coordinate-positive 720-727 case
does not have a below-rho verifier-informed subset for the pair rules to find.
It is a sharper speedup boundary: repeated coordinates can recur and derive
target-67 secrets, but the row-subset compression still has to clear the rho
line before public pair predicates matter.  The next useful window should be
treated as a three-way classifier: no full coordinate replay, full replay with
only over-rho subset compression, or full replay with a below-rho subset that
tests the frozen pair rules.

Fresh 728-735 repeated-coordinate and near-threshold audit:

- Stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_728_735_probe.json`
- Target-67 frozen pipeline artifacts:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_728_735_fresh_public_selector.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_728_735_fresh_coordinate_gate.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_728_735_fresh_guard_replay.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_728_735_fresh_manifest.json`
- Decomposition and replay checks:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord161_6976_728_735_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_680_no_exact_728_735_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_704_candidate_728_735_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord161_saltmod5_3_728_735_fresh.json`
- New diagnostic wrapper:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_near_threshold_audit.py`
- Near-threshold audit:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_near_threshold_audit_target67_672_728.json`
- The stress source chose `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`,
  with 4 verified below-rho stress leaves, 3 verified below-rho stress rows,
  and best stress-leaf cost 0.39416058 rho.
- The public selector carried a real below-rho `22050` source case:
  transfer 733, salt166, leaves `[8,56,90]`, rank 2, relation count 2, and
  0.39416058 rho.  The best target-67 public source case was transfer 730,
  salt204, leaves `[3,4,10]`, 0.528 rho, but rank 1 and not public-key
  verified.
- The target-67 coordinate gate had 115 candidates and 19 verified full
  coordinate replays, all over rho.  The verified coordinate was again
  `(161,6976)`, now on transfer 732, deriving secret 3693 at 1.432 rho.
- Decomposition again found exactly two relation-bearing rows, but still not a
  speedup: salts 207/208, leaf 2 on both rows, rank 2, two relations,
  128 ops versus 125 rho steps, i.e. 1.024 rho.
- The original 672/680 no-exact pair rule selected 60 public pairs and found
  zero relation pairs.  The widened post-704 no-exact candidate selected 80
  public pairs and also found zero relation pairs.
- The old coord161 salt-mod row rule activated on 19 cases but produced zero
  relation cases, so it is not the missing predictor for the recurring
  `(161,6976)` family.
- The near-threshold audit over eight decomposed target-67 recoveries reports
  6 verified below-rho subsets and 2 over-rho near misses.  Both near misses
  are the repeated `(161,6976)` family:
  - 720-727: transfer 721, salts 206/207, 129 ops / 125 rho = 1.032.
  - 728-735: transfer 732, salts 207/208, 128 ops / 125 rho = 1.024.
- Those two near misses share the same diagnostic form signature: selected
  leaf `2`, term shape `2+1+1`, and factor support `1|9|13`.  The
  candidate-position signature moved from `1|1` to `2|2`, so the stable signal
  is the leaf/form/factor-support pattern rather than a fixed candidate
  position.

Current boundary after 728-735:

The target-67 repeated-coordinate path now has two consecutive coordinate-
positive windows for `(161,6976)` that are only 3-4 operations above rho after
verifier-informed row pruning.  This is not a pair-rule validation and not a
speedup claim, but it is a sharper algorithmic target than broad salt-pair
mining.  A public residual/FFE layer that removes even one small fixed charge
per relation row, or shares a fixed leaf-2 `2+1+1` factor-support computation
across the two-row system, would cross the rho line on 728-735.  The next
experiment should therefore prioritize a public near-threshold residual audit
for the repeated `(161,6976)` form family before widening the salt-congruence
pair-rule miner again.

Residual-share audit and 736-743 classifier:

- New residual-share audit:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_residual_share_audit.py`
- Residual-share artifacts:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_residual_share_audit_target67_672_728.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_residual_share_audit_target67_672_736.json`
- Updated near-threshold artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_near_threshold_audit_target67_672_736.json`
- The residual-share ledger expands the preassociation cost formula
  `core + selected_leaf_count + 2 * selected_hit_events` for each
  relation-bearing row.  Public-before-event sharing of duplicate leaf/root
  setup is not enough to beat rho on the 720/728 near misses.
- Sharing one hit-event pass is the concrete residual target: it moves the
  728-735 near miss from 128 ops to 122 ops, below the 125-step rho baseline,
  but only moves the 720-727 near miss from 129 ops to 125 ops.  Adding the
  duplicate leaf/root share moves 720 below rho at 123 ops.  This remains a
  diagnostic cost ledger until a public FFE/summation-polynomial computation
  realizes the shared event pass before relation labels are inspected.
- Fresh 736-743 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_736_743_probe.json`
- Target-67 frozen pipeline artifacts:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_736_743_fresh_public_selector.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_736_743_fresh_coordinate_gate.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_736_743_fresh_guard_replay.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_736_743_fresh_manifest.json`
- 736-743 decomposition and replays:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord161_6976_736_743_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_680_no_exact_736_743_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_704_candidate_736_743_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord161_saltmod5_3_736_743_fresh.json`
- The 736-743 stress source again chose `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`,
  with 8 verified below-rho stress leaves, 3 verified below-rho stress rows,
  and best stress-leaf cost 0.43065693 rho.
- The target-67 coordinate gate had 56 candidates and 4 verified full
  coordinate replays.  All verified full replays were `(161,6976)` on transfer
  740, deriving secret 4924 at 1.344 rho with rank 3 and three relations.
- Cost decomposition found the first post-728 below-rho verifier-informed
  subset for this family: salts 203/209, leaf 2 on both rows, rank 2, two
  relations, 112 ops / 125 rho = 0.896 rho.
- Both pre-existing frozen pair rules miss this fair test:
  - the original 672/680 no-exact rule selected 12 public pairs and found zero
    relation pairs;
  - the widened post-704 rule selected 24 public pairs and found zero relation
    pairs.
- The old coord161 salt-mod row rule selected 4 cases and found zero relation
  cases.
- A post-736 no-exact miner run, with `w736_743_fresh` marked as the validation
  window, surfaced the simple rule
  `pair_activate:b_minus_c_mod16=1&pair_salt_index_signature=0,2`.
  Its train recall is poor (1/6 positives, 0 false positives), but it selects
  the 736 validation positive with 1/1 validation recall and 0 false positives.
  Because the miner ranks using validation score, this must be treated as a
  post-736 candidate, not as a promoted held-out rule.
- Replaying that post-736 candidate over 720/728/736 selects the 736 pair
  salts 203/209 and produces 4 guarded verified below-rho duplicate cases
  across the leaf-selector variants, all at 0.896 rho.  It finds no relation
  pairs in 720/728.

Current boundary after 736-743:

The repeated-coordinate path has now seen the same target-67 coordinate
`(161,6976)` in three consecutive fresh classifier windows: two near misses
at 1.032 and 1.024 rho, followed by a fair below-rho decomposed subset at
0.896 rho.  The old frozen pair rules are no longer adequate, but the
post-736 candidate gives a concrete next rule to freeze for a true unseen
holdout: `b_minus_c_mod16=1` plus selecting the first and third salt profiles
(`pair_salt_index_signature=0,2`).  The next proof obligation is 744-751:
replay this candidate without modification, and in parallel keep the residual
event-share ledger as a fallback if the family returns as another near miss.

Fresh 744-751 holdout of post-736 pair rule:

- Fresh 744-751 stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_744_751_probe.json`
- Target-67 frozen pipeline artifacts:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_744_751_fresh_public_selector.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_744_751_fresh_coordinate_gate.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_744_751_fresh_guard_replay.json`
- 744-751 decomposition and replays:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord1114_8506_744_751_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_680_no_exact_744_751_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_704_candidate_744_751_fresh.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_672_736_candidate_frozen_744_751_fresh.json`
- Updated audit artifacts:
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_near_threshold_audit_target67_672_744.json`
  - `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_residual_share_audit_target67_672_744.json`
- The fresh stress probe again chose `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`
  as the best public stress policy, with 4 verified below-rho stress leaves
  and best stress-leaf cost 0.45985401 rho.
- The target-67 coordinate gate had 68 candidates and 4 verified full
  coordinate replays, all over rho.  The verified coordinate changed from
  `(161,6976)` to `(1114,8506)` on transfer 744, deriving secret 6227 at
  1.44 rho with rank 3 and three relations.
- Cost decomposition reduced that recovery to salts 206/209, leaf 12, rank 3,
  three relations, and 129 ops / 125 rho = 1.032 rho.  This is a near miss,
  not a speedup.
- All three frozen pair-rule families miss this unseen holdout:
  - the original 672/680 rule selected 48 public pairs and found zero relation
    pairs;
  - the widened post-704 rule selected 44 public pairs and found zero relation
    pairs;
  - the post-736 candidate selected 20 public pairs and found zero relation
    pairs.
- The updated near-threshold audit now covers 10 verified decompositions: 7
  verified below-rho subsets and 3 over-rho near misses.  The new 744 miss has
  the same term shape family (`2+1+1`) but a different coordinate, leaf 12,
  and factor supports `1|6|12` / `1|6|13`.
- The updated residual-share ledger still has zero public-before-event models
  below rho.  For the 744 miss, duplicate-leaf sharing only reaches 128 ops,
  duplicate leaf plus hit-root reaches 127 ops, one shared hit-event pass ties
  rho at 125 ops, and only the diagnostic leaf/root/event-share combination
  crosses below rho at 123 ops.

Current boundary after 744-751:

The post-736 pair rule did not validate on the first truly unseen holdout.
That demotes it to a 736-local explanation rather than a promoted public
selector.  The stronger surviving signal is the residual FFE/summation-
polynomial compression target: repeated-coordinate recoveries continue to
create small near-rho relation systems, but public row/leaf sharing alone is
too weak.  The next useful experiment should either materialize a public
shared event-pass computation, or mine a predictor over the diagnostic
`2+1+1` form family that is frozen before the next fresh window.

Public hit-event share probe:

- New script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_hit_event_share_probe.py`
- Artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_hit_event_share_target67_672_744.json`
- The probe selects rows from public FFE hit-scan counters before relation
  labels, then audits the resulting replay and modeled shared-pass costs.
- Across the 10 decompositions, `all_hit_event_rows`, `all_hit_root_rows`, and
  `largest_leaf_hit_event_group` each select a public-key-verified replay in
  all 10 cases.  Five are already below rho without modeled sharing.
- Under a materialized duplicate leaf/root plus one shared hit-event pass,
  those three selectors put 8 of 10 verified replays below rho.  The two
  remaining misses are older three-row over-rho pockets: coord161_672 and
  coord161_736.
- On the three recent near misses:
  - 720: public hit-event rows select salts 206/207, 129 ops; leaf/root plus
    one event pass models 123 ops = 0.984 rho.
  - 728: public hit-event rows select salts 207/208, 128 ops; one event pass
    alone models 122 ops = 0.976 rho.
  - 744: public hit-event rows select salts 206/209, 129 ops; leaf/root plus
    one event pass models 123 ops = 0.984 rho.

This is the strongest current algorithmic target: the row selector is public
at the FFE hit-scan stage, and the remaining gap is now an explicit shared
work materialization problem.  It is still not an ECDLP speedup until the
shared leaf/root/event-pass computation is implemented and charged in replay.

Charged public hit-event replay:

- Updated script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_hit_event_share_probe.py`
- New charged artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_hit_event_charged_replay_target67_672_744.json`
- Run configuration:
  - public rule: `all_hit_event_rows`
  - charge policy: `share_leaf_hit_root_and_one_event_pass`
- Result:
  - 10/10 selected-row replays are public-key verified.
  - 5/10 are already below rho before shared charging.
  - 8/10 are below rho after the explicit charged accounting.
  - Mean charged cost is 0.9592 rho.
- Recent near-miss conversions:
  - 720: secret 1767, salts 206/207, baseline 129 ops, charged 123 ops
    = 0.984 rho.
  - 728: secret 3693, salts 207/208, baseline 128 ops, charged 120 ops
    = 0.96 rho.
  - 744: secret 6227, salts 206/209, baseline 129 ops, charged 123 ops
    = 0.984 rho.
- Remaining charged misses:
  - 672: baseline 174 ops, charged 164 ops = 1.312 rho.
  - 736: baseline 168 ops, charged 160 ops = 1.28 rho.

This artifact moves the branch from a loose cost ledger to an executable
public-selector/charged-replay audit.  The charged replay is still labeled
`accounting_only_not_algebraic_kernel`; the next promotion step is to implement
the shared leaf/root/event-pass computation directly inside the FFE replay path
and rerun this same audit without model-only subtraction.

Shared-pass trace materialization:

- Updated replay wrapper:
  `tasks/ecdlp_index_calculus/ffe_single_hit_root_relation_replay_probe.py`
- Updated decomposition probe:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_cost_decomposition_probe.py`
- New materialization audit:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_shared_pass_materialization_audit.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_shared_pass_materialization_audit_target67_672_744.json`
- Change:
  - repeated-coordinate decompositions now retain `selected_hit_root_values`;
  - row scans now retain `hit_event_summaries` for every x-match, not only
    accepted relation events;
  - the charged-replay artifact threads those public traces through selected
    rows.
- Result after rerunning all 10 decompositions and the charged replay:
  - 8/10 charged replays are below rho;
  - 7/10 need nonzero shared savings to cross rho;
  - 7/7 shared-savings wins now have the public hit-root IDs and all x-match
    traces needed to audit the charged shared pass;
  - 0 charged-below-rho records are missing hit-root IDs;
  - 0 charged-below-rho records are missing all-xmatch traces.

This closes the previous trace-materialization blocker.  The remaining
non-claim is narrower: the shared pass is still charged by counter subtraction
rather than by a fused FFE/summation-polynomial kernel.  Promotion now requires
moving the traced sharing into the replay computation itself and preserving the
same 7 shared-savings below-rho wins.

Trace-fused shared-pass executor audit:

- New audit script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_trace_fused_executor_audit.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_trace_fused_executor_audit_target67_672_744.json`
- The audit recomputes shared-pass work from materialized public traces instead
  of trusting the `shared_*_ops` or `charged_ops` counters as evidence.
- Cost model:
  - nonshared base work is the row preassociation scan after removing selected
    leaf, hit-root, and two x-match event-pass charges;
  - identical selected leaf signatures are charged once;
  - hit-root work is charged on the union of selected public hit-root IDs;
  - the first x-match event pass is charged on trace identities
    `(leaf_signature, scout_pos, original_trial)`;
  - the second event pass remains per selected hit event.
- Result on the same 10 verified 672-744 replays:
  - 10/10 verified replays have complete public traces;
  - 6/10 are below rho under trace-derived fused counters;
  - 5/10 need shared savings and remain below rho;
  - only 2/10 trace-derived costs exactly match the previous charged counters;
  - mean trace-fused below-rho cost is 0.85733333 rho.
- Correction to the previous charged replay:
  - the charged-counter model reported 8 below-rho replays and 7 shared-savings
    wins;
  - materialized x-match identities support only 6 below-rho replays and
    5 shared-savings wins;
  - the earlier 720 and 728 near-miss conversions no longer beat rho under
    strict event identity sharing: 720 is 127/125 = 1.016 rho and 728 is
    126/125 = 1.008 rho.
- The strongest fresh lead survives the stricter audit:
  - `coord1114_744` derives secret 6227 from salts 206/209, leaf 12;
  - baseline replay is 129/125 = 1.032 rho;
  - trace-fused replay is 123/125 = 0.984 rho;
  - the trace group has hit roots `[2716,5973]`, 8 unique event keys, and
    12 total selected hit events.

This is a better honesty boundary than the charged replay alone.  The result
is still not a low-level algebraic speedup, because the script derives work
counters from traces rather than executing a fused FFE kernel.  But it rules
out the optimistic 720/728 interpretation and keeps 744 as the immediate
kernel-implementation target.

Fused public worklist audit:

- New audit script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_worklist_audit.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_worklist_audit_target67_672_744.json`
- The audit emits the public worklist a fused leaf/root/x-match pass would
  consume, then links every accepted x-match instance back to relation
  summaries from the decomposition artifacts.
- Result on the same 10 verified 672-744 replays:
  - 10/10 records are `ready_for_kernel_implementation`;
  - 0 records have duplicate-event public-field conflicts;
  - 0 records are missing relation-summary links for accepted events;
  - 6/10 remain below rho under the fused worklist counters;
  - 5/10 below-rho records need shared savings;
  - only 1 below-rho record depends on reusable first-pass x-match events.
- That single event-reuse speedup target is the fresh 744 case:
  - `coord1114_744` has leaf signature `12`, salts 206/209, and hit roots
    `[2716,5973]`;
  - baseline replay is 129/125 = 1.032 rho;
  - fused worklist cost is 123/125 = 0.984 rho;
  - the worklist saves 1 duplicate leaf op, 1 hit-root op, and 4 first-pass
    x-match ops;
  - four duplicate public event keys are reusable:
    `(12,17,469)`, `(12,30,469)`, `(12,174,469)`, and `(12,210,469)`;
  - the accepted duplicate event `(12,174,469)` links to two row-specific
    `2+1+1` relation summaries, so first-pass sharing is public while
    relation-form emission remains a row-specific second pass.
- The old 720/728 charged wins are now explicit negative controls:
  - both are `ready_for_kernel_implementation`;
  - both have zero reusable first-pass event keys;
  - 720 remains 127/125 = 1.016 rho;
  - 728 remains 126/125 = 1.008 rho.

This is still not the final fused algebraic kernel, but it is no longer just a
counter ledger.  It gives the exact public leaf/root/event worklist the kernel
must consume and proves that the 744 target is the only current below-rho
case whose margin actually depends on first-pass x-match reuse.

Fused worklist relation-form replay:

- New replay consumer:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_worklist_replay_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_worklist_replay_target67_672_744.json`
- The consumer reads the fused public worklist, reconstructs the compact
  verifier linear forms from accepted second-pass relation summaries, derives
  the target secret by modular linear algebra, and recomputes fused work
  counters from the worklist.  It does not read `charged_ops` or `shared_*_ops`.
- The first run exposed and fixed an important order/modulus boundary:
  `67.a1@9803` has field prime 9803 but base order 9887.  The consumer now
  reads `frontier_targets.json` and uses `base_order` rather than parsing the
  target suffix as the relation modulus.
- Result on the same 10 verified 672-744 records:
  - 10/10 worklists are consumed and relation-replayed;
  - 10/10 relation-form replays derive the same secret as the upstream
    public-key-verified replay;
  - 6/10 consumed worklists remain below rho;
  - only 1 consumed below-rho worklist depends on first-pass event reuse;
  - mean consumed below-rho cost remains 0.85733333 rho.
- The fresh 744 target now has a replay-facing artifact, not just a counter
  audit:
  - `coord1114_744` consumes the public worklist;
  - relation-form replay derives secret 6227 with rank 3 over base order 9887;
  - fused work counters are recomputed as 123/125 = 0.984 rho;
  - first-pass reuse contributes 4 saved ops across 4 reused event keys.
- The 720/728 controls also consume and derive their recorded secrets, but stay
  negative under fused work counters:
  - 720 derives 1767 but costs 127/125 = 1.016 rho, with zero first-pass reuse;
  - 728 derives 3693 but costs 126/125 = 1.008 rho, with zero first-pass reuse.

This moves the branch past passive audit into replay-layer consumption of the
public worklist.  The remaining non-claim is still explicit: low-level
candidate equality and relation-form emission are supplied by the existing
scanner artifacts.  The next promotion step is to move those two pieces into
the fused scanner itself.

Fused worklist scanner-layer replay:

- New scanner consumer:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_worklist_scanner_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_worklist_scanner_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_worklist_scanner_target67_744_controls.json`
- The scanner consumer rematerializes row contexts, uses the public fused
  worklist event keys to drive candidate x-match checks, emits relation forms
  through the existing verifier predicate, and recomputes fused work counters
  from the observed scanner events.
- The charged replay artifact is used only as a locator for decomposition
  source paths.  The scanner output does not consume `charged_ops`,
  `shared_*_ops`, or accepted relation summaries as relation-form evidence.
- Three-record gate:
  - `coord1114_744` derives secret 6227 with rank 3 and public-key
    verification;
  - it observes all 12 expected second-pass instances, 8 unique first-pass
    event keys, no missing/unexpected instances, and no acceptance mismatches;
  - scanner-fused cost is 123/125 = 0.984 rho;
  - `coord161_720` derives 1767 but remains 127/125 = 1.016 rho;
  - `coord161_728` derives 3693 but remains 126/125 = 1.008 rho.
- Full 10-record result:
  - 10/10 worklists are consumed by the scanner;
  - 10/10 derive the recorded public-key-verified secret;
  - 10/10 match the worklist fused counters;
  - 6/10 are below rho;
  - only 1 below-rho record depends on first-pass event reuse;
  - mean scanner below-rho cost remains 0.85733333 rho.

This is the strongest boundary so far.  The public worklist now drives
scanner-layer x-match checks and relation-form emission instead of merely
replaying compact summaries.  The remaining non-claim is narrower: the code
still reuses the existing row context builder and verifier predicate, so the
next promotion is an optimized fused FFE/summation-polynomial kernel for the
same public worklist.

Fused candidate-point reuse audit:

- New prototype/audit script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_candidate_point_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_candidate_point_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_candidate_point_target67_744_controls.json`
- The probe rematerializes the same row contexts as the scanner consumer,
  groups row instances by public event key, audits whether every instance in a
  duplicate group has the same elliptic-curve candidate point, and then emits
  relation forms using one shared representative candidate point for
  point-consistent groups.
- The audit recomputes local candidate points for every row instance to prove
  equality, but reports those extra audit computations separately from the
  planned fused-kernel work counters.
- Three-record gate:
  - `coord1114_744` has zero candidate-point conflicts, derives secret 6227,
    and stays at 123/125 = 0.984 rho;
  - its four duplicate public event keys are point-consistent and save four
    planned first-pass candidate-point operations;
  - the accepted duplicate event `(12,174,469)` reuses representative point
    `[9251,9216]` across salts 206 and 209 and still emits both row-specific
    relation forms;
  - `coord161_720` and `coord161_728` derive 1767 and 3693 respectively but
    have zero candidate-point reuse and stay above rho.
- Full 10-record result:
  - 10/10 records verify candidate-point reuse boundaries;
  - 0/10 have candidate-point conflicts;
  - 10/10 derive the recorded public-key-verified secret;
  - 10/10 match the fused worklist counters;
  - 6/10 are below rho;
  - only 1 below-rho record depends on event-key candidate-point reuse;
  - mean below-rho cost remains 0.85733333 rho.

This moves the evidence one step below scanner scheduling.  The current branch
now proves that the 744 first-pass event reuse is not only a public keying
identity: the duplicate event-key groups really share the same candidate point
and can feed row-specific relation emission from one representative first-pass
result.  The remaining non-claim is now specifically the standalone optimized
FFE/summation-polynomial implementation of this plan.

Fused hot-path candidate-point executor:

- New executor script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_executor_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_executor_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_executor_target67_744_controls.json`
- The executor removes candidate-point equality recomputation from the planned
  hot path.  It computes one representative elliptic-curve candidate point per
  public event key and reuses that point for every row-specific relation
  predicate in the event-key group.
- The candidate-point audit artifact is used only as a post-run oracle.  It is
  not used to select row instances, relation forms, or hot-path candidate
  points.
- Three-record gate:
  - `coord1114_744` derives secret 6227, rank 3, and public-key verification;
  - it executes 8 first-pass candidate-point groups, 4 of them reused, then 12
    second-pass row instances;
  - hot-path fused cost is 123/125 = 0.984 rho;
  - the accepted duplicate key `(12,174,469)` computes representative point
    `[9251,9216]` once from salt206 and emits both row-specific relation forms;
  - `coord161_720` and `coord161_728` derive 1767 and 3693 but have zero
    reusable candidate-point groups and stay at 127/125 and 126/125.
- Full 10-record result:
  - 10/10 records verify under the hot-path executor;
  - 10/10 derive the recorded public-key-verified secret;
  - 10/10 match the worklist fused counters;
  - 10/10 match the candidate-point audit oracle;
  - 6/10 are below rho;
  - only 1 below-rho record depends on hot-path candidate-point reuse;
  - mean hot-path below-rho cost remains 0.85733333 rho.

This is now an executable Python control-flow prototype of the fused first
pass.  It still uses the existing row-context builder and verifier predicate,
but the first-pass candidate-point work itself is no longer an audit-only
claim: the hot path performs one candidate-point computation per public event
key and feeds all row-specific relation checks from that result.

Fused kernel contract:

- New contract emitter/verifier:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_contract_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_contract_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_contract_target67_744_controls.json`
- The contract lowers the hot-path executor output into a portable kernel
  target: first-pass candidate-point groups, second-pass row checks, accepted
  compact relation summaries, independent modular derivation, and work
  counters recomputed from the contract.
- Three-record gate:
  - `coord1114_744` has 8 first-pass candidate-point outputs, 12 second-pass
    row checks, 3 accepted relation summaries, rank 3, and derived secret 6227
    over base order 9887;
  - contract work is 123/125 = 0.984 rho;
  - `coord161_720` and `coord161_728` derive their secrets from the same
    contract shape but stay above rho with 12 first-pass groups and no reuse.
- Full 10-record result:
  - 10/10 contracts verify;
  - 10/10 independently derive the recorded secret;
  - 10/10 recomputed fused counters match the executor;
  - 6/10 are below rho;
  - only 1 below-rho contract depends on event-key reuse;
  - mean contract below-rho cost remains 0.85733333 rho.

This is the current best low-level handoff boundary.  A real optimized
FFE/summation-polynomial kernel can now target this contract directly: emit the
same first-pass candidate-point groups and second-pass row checks, then verify
against the contract without relying on scanner internals.

Fused kernel ABI/curve invariant probe:

- New ABI/curve guard:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_abi_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_abi_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_abi_target67_744_controls.json`
- The probe consumes the portable contract and the verifier-reduced curve.  It
  checks that every first-pass candidate point is finite, in field range, on
  the curve, and in the base-order subgroup; every second-pass row check maps
  back to a first-pass event-key group; accepted compact summaries match the
  row-specific relation check ABI; and recorded first/second-pass counters
  match the contract.
- Three-record gate:
  - 3/3 records verify at the ABI/curve boundary;
  - 32 candidate-point curve/subgroup checks and 36 second-pass row checks pass;
  - `coord1114_744` preserves the exact reuse anchor: distinct first-pass
    candidate points are `[2705, 9753]` and `[9251, 9216]`, event
    `["12", 174, 469]` reuses `[9251, 9216]` across two row instances, and
    both row-specific accepted relation checks remain present;
  - `coord161_720` and `coord161_728` remain no-reuse controls above rho.
- Full 10-record result:
  - 10/10 records verify at the ABI/curve boundary;
  - 77/77 candidate-point checks are on-curve and in the base-order subgroup;
  - 81/81 second-pass row checks match their first-pass event-key fanout;
  - 22 accepted relation summaries match accepted row checks;
  - 6/10 verified records stay below rho;
  - only 1 below-rho record depends on event-key reuse.

This narrows the promotion target again: the next implementation only has to
emit the same first-pass curve points and second-pass row-check stream from
lower-level FFE/summation-polynomial code, then pass this ABI/curve guard.

Fused kernel affine field-operation trace:

- New affine trace emitter:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_affine_trace_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_affine_trace_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_affine_trace_target67_744_controls.json`
- The trace emitter rematerializes the public worklist contexts and emits the
  generalized-Weierstrass affine addition fields for every first-pass
  candidate-point group: input points, slope numerator/denominator, denominator
  inverse, slope, intercept, and resulting candidate point.  It then checks
  every row-instance local add against the verifier formula and every
  representative output against the ABI artifact.
- Three-record gate:
  - 3/3 affine traces verify;
  - 32 first-pass groups match the ABI candidate points;
  - 36 row-instance affine adds match the verifier formula;
  - 4 reused groups are all in `coord1114_744`;
  - the critical reused event `["12", 174, 469]` computes
    `[4979, 3914] + [7537, 8016] -> [9251, 9216]` with slope numerator 4102,
    denominator 2558, denominator inverse 6741, slope 7122, and intercept 927;
    both row instances reuse that same output.
- Full 10-record result:
  - 10/10 affine traces verify;
  - 77 first-pass groups match the ABI candidate points;
  - 81 row-instance affine adds match the verifier formula;
  - 6 verified records remain below rho;
  - only 1 below-rho affine trace depends on event-key reuse.

This is the strongest current low-level target: a native FFE implementation can
now be tested against explicit field-operation traces before it is asked to
emit row-specific relation predicates or derive the secret.

Fused kernel finite-field replay:

- New verifier-independent field replay:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_field_replay_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_field_replay_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_field_replay_target67_744_controls.json`
- The replay consumes the affine trace and reruns every first-pass candidate
  point with plain modular arithmetic only.  It imports no verifier helpers and
  does not call curve arithmetic.  The emitted register stream covers the
  slope numerator, slope denominator, denominator inverse, slope, intercept,
  x/y result formulas, and explicit field operation counts.
- Three-record gate:
  - 3/3 field replays verify;
  - 32 representative first-pass instruction streams match the affine/ABI
    outputs;
  - 36 row-instance field replays are consistent with their representative
    outputs;
  - the first-pass representatives require 544 field operations:
    32 inversions, 160 multiplications, 64 additions, 256 subtractions, and
    32 negations;
  - the critical 744 reused event `["12", 174, 469]` replays to
    `[9251, 9216]` with the register prefix
    `4102 = 8016 - 3914`, `2558 = 7537 - 4979`,
    `6741 = 2558^-1`, `7122 = 4102 * 6741`, and
    `927 = 3914 - 7122 * 4979`.
- Full 10-record result:
  - 10/10 field replays verify;
  - 77 representative first-pass instruction streams match ABI outputs;
  - 81 row-instance field replays are consistent;
  - the first-pass representatives require 1309 field operations:
    77 inversions, 385 multiplications, 154 additions, 616 subtractions, and
    77 negations;
  - 6 verified records remain below rho;
  - only 1 below-rho field replay depends on event-key reuse.

This is the first verifier-independent candidate-point kernel contract in the
current branch.  The next native/FFE implementation can target this register
stream directly, then feed the unchanged ABI/contract layers.

Fused kernel native C reference:

- New native reference generator/checker:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_native_reference_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_native_reference_target67_672_744.json`
- New generated C source:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_native_reference_target67_672_744.c`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_native_reference_target67_744_controls.json`
- Control generated C source:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_native_reference_target67_744_controls.c`
- The checker consumes the verifier-independent field replay, emits a
  standalone C99 program with static first-pass instruction cases, compiles it
  with the system C compiler, and runs it.  The C program independently
  replays modular add/sub/mul/neg/inv instructions, verifies every recorded
  register value, and checks the final candidate-point outputs.
- Three-record gate:
  - C source compiled and executed with empty compiler/runtime stderr;
  - 3/3 records verify through the native reference;
  - 32 native first-pass cases verify;
  - 544 native field instructions verify with 0 failures;
  - field op counts match the field replay exactly:
    32 inversions, 160 multiplications, 64 additions, 256 subtractions, and
    32 negations;
  - `coord1114_744` contributes 8 native cases, 136 instructions, and the same
    4 reused event-key groups.
- Full 10-record result:
  - C source compiled and executed with empty compiler/runtime stderr;
  - 10/10 records verify through the native reference;
  - 77 native first-pass cases verify;
  - 1309 native field instructions verify with 0 failures;
  - field op counts match the field replay exactly:
    77 inversions, 385 multiplications, 154 additions, 616 subtractions, and
    77 negations;
  - 6 verified records remain below rho;
  - only 1 below-rho native reference depends on event-key reuse.

This is now a native-code promotion gate for the first-pass candidate-point
layer.  The remaining implementation gap is not the arithmetic contract; it is
replacing the generated reference with an optimized FFE/summation-polynomial
kernel that emits the same register stream and then feeds the unchanged
second-pass relation predicates.

Fused kernel compact native affine-add kernel:

- New compact native kernel generator/checker:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_native_kernel_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_native_kernel_target67_672_744.json`
- New generated C source:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_native_kernel_target67_672_744.c`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_native_kernel_target67_744_controls.json`
- Control generated C source:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_native_kernel_target67_744_controls.c`
- This replaces the static-instruction C reference with a reusable
  `fused_candidate_point_kernel` C function.  The generated C now stores
  operand arrays and expected registers, then computes slope numerator,
  denominator, inverse, slope, intercept, and candidate point from the operands
  inside the kernel.
- Three-record gate:
  - C source compiled and executed with empty compiler/runtime stderr;
  - 3/3 records verify through the compact native kernel;
  - 32 operand-driven native first-pass cases verify;
  - 32/32 register streams match and 32/32 candidate points match;
  - field op counts match the previous native reference exactly:
    32 inversions, 160 multiplications, 64 additions, 256 subtractions, and
    32 negations;
  - `coord1114_744` contributes 8 native-kernel cases and the same 4 reused
    event-key groups.
- Full 10-record result:
  - C source compiled and executed with empty compiler/runtime stderr;
  - 10/10 records verify through the compact native kernel;
  - 77 operand-driven native first-pass cases verify;
  - 77/77 register streams match and 77/77 candidate points match;
  - field op counts match the previous native reference exactly:
    77 inversions, 385 multiplications, 154 additions, 616 subtractions, and
    77 negations;
  - 6 verified records remain below rho;
  - only 1 below-rho native-kernel record depends on event-key reuse.

This moves the first-pass candidate-point implementation from replaying static
field instructions to executing a compact native affine-add kernel over
operands.  The next remaining optimization step is to specialize or vectorize
this kernel for the repeated-coordinate 744 event-key family while preserving
the same register stream and ABI/contract gates.

Fused kernel shared-denominator native specialization:

- New shared-denominator native kernel generator/checker:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_shared_denominator_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_shared_denominator_target67_672_744.json`
- New generated C source:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_shared_denominator_target67_672_744.c`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_shared_denominator_target67_744_controls.json`
- Control generated C source:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_shared_denominator_target67_744_controls.c`
- This specializes the compact native kernel by grouping repeated-coordinate
  cases with the same field, curve coefficients, left x-coordinate, and right
  point.  The generated C prepares `right_x - left_x` and its inverse once per
  group, then computes all row/event-specific candidate points from the
  varying left y-coordinate.
- Three-record gate:
  - C source compiled and executed with empty compiler/runtime stderr;
  - 3/3 records verify through the shared-denominator native kernel;
  - 2 shared denominator groups cover 32 native cases;
  - 32/32 register streams match and 32/32 candidate points match;
  - field op count drops from 544 to 484 compared with the compact native
    kernel: 30 fewer inversions and 30 fewer subtractions;
  - `coord1114_744` remains the below-rho event-reuse target, with 8 cases and
    the shared denominator/inverse group for right point `[7537, 8016]`:
    denominator 2558, inverse 6741.
- Full 10-record result:
  - C source compiled and executed with empty compiler/runtime stderr;
  - 10/10 records verify through the shared-denominator native kernel;
  - 6 shared denominator groups cover 77 native cases;
  - 77/77 register streams match and 77/77 candidate points match;
  - field op count drops from 1309 to 1167 compared with the compact native
    kernel: 71 fewer inversions and 71 fewer subtractions;
  - 6 verified records remain below rho;
  - only 1 below-rho shared-denominator record depends on event-key reuse.

This is the first native optimization that directly exploits the repeated
coordinate structure rather than merely replaying it.  The next implementation
step is to turn each shared group into a vector/batch lane and keep the same
register-output checks.

Fused kernel grouped-left-y batch lanes:

- New grouped-left-y native batch kernel generator/checker:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_batch_lane_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_batch_lane_target67_672_744.json`
- New generated C source:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_batch_lane_target67_672_744.c`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_batch_lane_target67_744_controls.json`
- Control generated C source:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_batch_lane_target67_744_controls.c`
- This specializes the shared-denominator kernel again: inside each
  denominator group, cases with the same left y-coordinate are one batch lane.
  The generated C computes each lane once, stores the register/candidate output,
  and verifies every event case against its lane.
- Three-record gate:
  - C source compiled and executed with empty compiler/runtime stderr;
  - 3/3 records verify through the batch-lane kernel;
  - 2 shared denominator groups collapse 32 event cases into 4 batch lanes;
  - 4/4 lane outputs match and 32/32 event cases match;
  - field op count drops from 484 to 64 compared with the shared-denominator
    kernel: 420 saved field operations;
  - `coord1114_744` has 8 event cases but only 2 lane computations:
    left y 3914 -> `[9251, 9216]` and left y 5888 -> `[2705, 9753]`.
- Full 10-record result:
  - C source compiled and executed with empty compiler/runtime stderr;
  - 10/10 records verify through the batch-lane kernel;
  - 6 shared denominator groups collapse 77 event cases into 12 batch lanes;
  - 12/12 lane outputs match and 77/77 event cases match;
  - field op count drops from 1167 to 192 compared with the shared-denominator
    kernel: 975 saved field operations;
  - 6 verified records remain below rho;
  - only 1 below-rho batch-lane record depends on event-key reuse.

This is the strongest first-pass kernel specialization so far: candidate-point
arithmetic is now amortized by denominator group and by unique left-y lane,
while the full event-case ABI remains verified.

Fused kernel batch-lane portable contract:

- New lane-amortized contract verifier:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_batch_lane_contract_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_batch_lane_contract_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_batch_lane_contract_target67_744_controls.json`
- The verifier consumes the original portable contract and the validated native
  batch-lane kernel artifact.  It preserves the original second-pass event
  fanout, accepted relation summaries, and independent modular derivation, but
  replaces first-pass candidate-point charge with native batch-lane count.
- Three-record gate:
  - 3/3 lane-amortized contracts verify and derive the recorded secrets;
  - all 3 records are below rho under lane accounting;
  - total contract work drops from 376 to 350;
  - `coord1114_744` drops from 123/125 = 0.984 rho to 117/125 = 0.936 rho;
  - 720 and 728 controls also cross below rho because their 12 event-key
    candidate cases collapse to 2 native batch lanes while preserving 12
    second-pass checks.
- Full 10-record result:
  - 10/10 lane-amortized contracts verify;
  - 10/10 independently derive the recorded secret;
  - 8/10 are below rho under lane accounting;
  - total contract work drops from 1232 to 1175;
  - mean below-rho lane cost is 0.85 rho;
  - every record preserves second-pass fanout, accepted summary count, and
    relation-derived secret.

This promotes the native batch-lane optimization back into the portable
contract layer.  The old event-key contract remains the conservative fallback,
but the new result shows the repeated-coordinate kernel has a verifier-backed
lane-amortized cost model.

Fused kernel batch-lane ABI/curve guard:

- New batch-lane ABI/curve guard:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_batch_lane_abi_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_batch_lane_abi_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_batch_lane_abi_target67_744_controls.json`
- The guard reconstructs grouped-left-y lanes from the affine trace, checks the
  native batch-lane artifact shape, verifies every source-local lane as a
  finite point on the verifier curve in the base-order subgroup, and confirms
  that the lane contract still preserves the second-pass fanout and relation
  replay secret.
- Three-record gate:
  - 3/3 batch-lane ABI records verify;
  - 4 global native lanes cover 32 event cases;
  - 6 source-local contract lanes preserve 36 second-pass checks;
  - 3/3 records remain below rho under lane accounting;
  - mean below-rho lane cost is 0.93333333 rho.
- Full 10-record result:
  - 10/10 batch-lane ABI records verify;
  - all 10 source-local lane sets are on-curve and in the base-order subgroup;
  - 12 global native lanes cover 77 event cases;
  - 20 source-local contract lanes preserve 81 second-pass checks;
  - 10/10 relation replays derive the recorded secret;
  - 8/10 records remain below rho under lane accounting;
  - mean below-rho lane cost is 0.85 rho.

This closes the current implementation-proof ladder for the repeated-coordinate
batch-lane kernel.  The old event-key contract is still the conservative
accounting floor, but the optimized lane contract now has native execution,
curve/subgroup ABI validation, preserved fanout, and independent secret replay.

Fused kernel contract comparison:

- New side-by-side contract comparison probe:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_contract_comparison_probe.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_contract_comparison_target67_672_744.json`
- Control artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_fused_kernel_contract_comparison_target67_744_controls.json`
- The comparison keeps two claim boundaries separate:
  - conservative event-key accounting: one first-pass candidate-point operation
    per public event key;
  - optimized batch-lane accounting: first-pass candidate-point work charged by
    ABI-guarded grouped-left-y native lanes.
- Three-record gate:
  - 3/3 contract comparisons verify;
  - conservative event-key below-rho count is 1/3;
  - ABI-guarded batch-lane below-rho count is 3/3;
  - lane-only below-rho promotions are exactly `coord161_720` and
    `coord161_728`;
  - total work drops from 376 event-key fused ops to 350 batch-lane fused ops.
- Full 10-record result:
  - 10/10 contract comparisons verify;
  - conservative event-key below-rho count is 6/10;
  - ABI-guarded batch-lane below-rho count is 8/10;
  - lane-only below-rho promotions are still exactly `coord161_720` and
    `coord161_728`;
  - total work drops from 1232 event-key fused ops to 1175 batch-lane fused
    ops, saving 57 operations;
  - mean event-key below-rho cost is 0.85733333 rho;
  - mean ABI-guarded batch-lane below-rho cost is 0.85 rho.

This comparison prevents the main accounting mistake: the 720 and 728 wins are
not conservative event-key wins.  They are optimized implementation claims that
require the native batch-lane artifact, the lane-amortized contract, and the
batch-lane ABI/curve guard.

Fresh 752-759 repeated-coordinate diagnostic:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_752_759_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_752_759_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_752_759_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_752_759_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_752_759_fresh_manifest.json`
- Cost decomposition:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_cost_decomposition_target67_coord429_3910_752_759_fresh.json`
- Post-hoc public row-rule diagnostics:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_rule_replay_target67_coord429_3910_saltmod3_1_752_759_fresh_diagnostic.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_rule_replay_target67_coord429_3910_globalcap3_saltmod3_1_752_759_fresh_diagnostic.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord429_3910_globalcap3_saltmod3_1_752_759_fresh_diagnostic.json`
- The fresh stress source is a direct negative for below-rho source labels:
  12 verified stress-leaf labels, zero below-rho stress-leaf labels, and best
  stress-leaf ops/rho 1.336.
- The frozen coordinate gate found a real repeated-coordinate family on target
  67, transfer 753, coordinate `(b,c)=(429,3910)`.  It replayed 136
  candidates, verified 24 public keys, but had zero verified below-rho
  recoveries; the best verified replay cost was still 1.224 rho.
- Cost decomposition of coordinate `(429,3910)` explains the miss: the full
  verified replay carries salts 205, 206, and 208 at 167 ops / 125 rho =
  1.336, while the relation-bearing subset salts 205 and 208 alone derives
  secret 7062 at 116 ops / 125 rho = 0.928.
- The broad post-hoc public row rule
  `row_activate:coordinate_key=429,3910&leaf_min=5&salt_mod3=1` selected 40
  cases, verified 16, and had 8 verified below-rho recoveries.  It still
  overselected target-cap3 variants.
- The tightened post-hoc public row rule
  `row_activate:coordinate_key=429,3910&leaf_min=5&salt_mod3=1&policy_family=global_cap3`
  selected 20 cases, verified 8, and all 8 verified cases were below rho at
  0.928.  The same 8 cases passed the `candidate_pos_span=0` form guard:
  8 guarded verified, 8 guarded below-rho, no rejected-but-verified cases.
- This is a new candidate family, not a completed speedup claim.  The row rule
  was chosen after inspecting the 752-759 decomposition.  Promotion requires a
  frozen no-peek validation on a later window or an independent public rule
  miner that predicts the same two relation-bearing rows without using current
  relation events.

Fresh 760-767 frozen row-pruning validation:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_760_767_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_760_767_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_760_767_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_760_767_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_760_767_fresh_manifest.json`
- Frozen rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_row_rule_replay_target67_coord429_3910_globalcap3_saltmod3_1_760_767_fresh_frozen.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord429_3910_globalcap3_saltmod3_1_760_767_fresh_frozen.json`
- Direct stress remains negative: best policy is
  `fixed_global_cap3_ow1_hw3_lw0_sw0_cw0_aw0`, with 8 verified stress-leaf
  labels, zero below-rho stress-leaf labels, and best stress-leaf ops/rho
  1.08029197.
- The repeated-coordinate gate selected 52 target-67 candidates, but verified
  zero public keys and had zero below-rho verified replays.  Minimum replayed
  ops/rho was 1.224.
- The exact post-hoc 752-759 coordinate `(429,3910)` does recur in 760-767,
  but on transfer 764 with salts 204, 205, and 206 instead of the
  relation-bearing 752-759 salt pair 205 and 208.  The frozen rule
  `coordinate_key=429,3910&leaf_min=5&salt_mod3=1&policy_family=global_cap3`
  therefore selects only the salt-205 row: 4 selected cases, zero verified
  cases, zero relation cases, and zero guarded below-rho cases.
- This is a clean later-window negative for promoting the 752-759 exact
  salt-mod row-pruning rule.  It does not falsify repeated-coordinate row
  pruning in general; it says the useful predicate cannot be a fixed
  coordinate plus a static salt residue.  The next useful predictor needs a
  public pair rule for relation-bearing salt pairs or a broader
  no-exact-coordinate residue feature that survives fresh windows.

No-exact-coordinate salt-pair miner:

- Pair-rule miner artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_miner_target67_positive_decomp_validation752_no_exact.json`
- Pair-rule replay artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_752_759.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_760_767.json`
- Training corpus: all available positive target-67 repeated-coordinate cost
  decompositions through 736-743, with 752-759 held out as validation.  Exact
  coordinate and exact salt features were disabled.
- The best mined public pair rule is:
  `pair_activate:b_minus_c_mod16=1&pair_salt_span>=4|b_minus_c_mod16=7&pair_salt_delta_from_min_signature=0,3|pair_salt_sum_mod8=0&transfer_mod5=1`
- Miner score:
  - train: 5/14 true positives, 0 false positives, precision 1.0, recall
    0.35714286;
  - validation on 752-759 decomposition: 1/1 true positive, 0 false positives,
    precision 1.0, recall 1.0.
- Replaying the mined pair rule on the 752-759 coordinate-gate artifact gives
  76 selected pair cases, 8 relation pair cases, 8 guarded verified pair cases,
  and 8 guarded below-rho pair cases.  All best cases derive secret 7062 at
  116/125 = 0.928 rho with the `candidate_pos_span=0` guard.
- Replaying the same pair rule on the later 760-767 coordinate-gate artifact
  abstains cleanly: 0 selected pair cases, 0 relation cases, and 0 verified
  cases.
- This is the better current branch.  It does not yet promote a new speedup,
  because the rule still needs a later-window positive.  But compared with the
  exact coordinate/salt residue, it both preserves the 752-759 below-rho
  diagnostic and avoids a 760-767 false positive without using exact
  coordinate or exact salt literals.

Fresh 768-775 no-exact pair-rule promotion test:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_768_775_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_768_775_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_768_775_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_768_775_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_768_775_fresh_manifest.json`
- No-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_768_775.json`
- Direct stress found a separate target-cap1 signal: best policy
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, 8 verified stress-leaf labels,
  all 8 below rho, and best stress-leaf ops/rho 0.528.  This is not the
  repeated-coordinate claim but is worth preserving as a scheduler cue.
- The frozen repeated-coordinate gate itself found 87 target-67 candidates and
  7 verified replays, but zero verified below-rho full-coordinate replays.
  The best verified full replay was coordinate `(161,6976)` on transfer 771,
  deriving secret 6683 at 1.328 rho.
- Replaying the already-mined no-exact pair rule
  `pair_activate:b_minus_c_mod16=1&pair_salt_span>=4|b_minus_c_mod16=7&pair_salt_delta_from_min_signature=0,3|pair_salt_sum_mod8=0&transfer_mod5=1`
  on this fresh 768-775 window gives the first later-window positive:
  32 selected pair/subset cases, 12 relation cases, 12 guarded verified cases,
  and 12 guarded below-rho cases.
- Best guarded case: coordinate `(161,6976)`, transfer 771, global-cap3,
  leaf 2, salt 208 only, `candidate_pos_span=0`, deriving secret 6683 at
  64/125 = 0.512 rho.
- The same rule also selects two-row variants such as salts 203/208 and
  202/208, deriving the same secret at 115/125 = 0.92 rho.  Across guarded
  verified cases, min ops/rho is 0.512, mean is 0.784, and max is 0.92.
- This is now a promoted candidate branch rather than a post-hoc diagnostic:
  the pair rule used no exact coordinate or exact salt literals, was mined
  before seeing 768-775, abstained on 760-767, and then recovered below rho on
  768-775.  It still needs another fresh replication and a compact accounting
  write-up before being described as a general index-calculus algorithm.

Fresh 776-783 no-exact pair-rule replication:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_776_783_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_776_783_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_776_783_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_776_783_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_776_783_fresh_manifest.json`
- No-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_776_783.json`
- Direct stress is again negative below rho: best policy
  `fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0`, 24 verified stress-leaf labels,
  zero below-rho stress-leaf labels, and best stress-leaf ops/rho 1.01459854.
- The frozen repeated-coordinate gate found 108 target-67 candidates and 36
  verified replays, but zero verified below-rho full-coordinate replays.  The
  best verified full replays on transfer 779 derive secret 4774 at 1.336 rho;
  the global minimum replayed ops/rho is 1.224, still with no below-rho
  verified recovery.
- Replaying the exact same no-exact pair rule on the fresh 776-783 window gives
  a second later-window positive: 52 selected pair/subset cases, 4 relation
  cases, 4 guarded verified cases, and 4 guarded below-rho cases.
- Best guarded case: coordinate `(542,3911)`, transfer 779,
  `fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0`, salts 203 and 206, leaf min 6,
  `candidate_pos_span=0`, rank 2, relation count 2, deriving secret 4774 at
  116/125 = 0.928 rho.  Across guarded verified cases, min and mean ops/rho are
  both 0.928.
- The evidence table is now: 752-759 held-out positive, 760-767 clean
  abstention, 768-775 fresh later-window positive, and 776-783 fresh
  later-window positive.  The claim boundary remains unchanged: full
  repeated-coordinate replay is over rho; the candidate speedup is public
  subset/pair pruning plus the event-stage `candidate_pos_span=0` form guard.

Fresh 784-791 unchanged-rule follow-up:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_784_791_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_784_791_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_784_791_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_784_791_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_784_791_fresh_manifest.json`
- No-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_784_791.json`
- Direct stress remains below-rho negative: best policy
  `fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0`, 8 verified stress-leaf labels,
  zero below-rho stress-leaf labels, and best stress-leaf ops/rho 1.08029197.
- The frozen repeated-coordinate gate selected 94 target-67 candidates, but
  verified zero public keys and had zero verified below-rho full-coordinate
  replays.  Minimum replayed ops/rho was 1.224.
- The unchanged no-exact pair rule selected 72 pair/subset cases, but produced
  zero relation pair cases, zero guarded verified cases, and zero guarded
  below-rho cases.  Sixteen selected cases pass the `candidate_pos_span=0`
  guard, but they reach only rank 1 and do not verify.
- This is not a verified over-rho false positive, but it is also not the clean
  abstention seen on 760-767.  The scheduler now has a real dead-selection cost
  boundary: it can select public pair cases in a fresh window without reaching a
  relation system.

Strict rank-lift form guard audit:

- Guard implementation update:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_form_guard_replay_probe.py`
- Strict replay artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_752_791.json`
- The old guard `candidate_pos_span=0` was vacuous on one-event systems: the
  784-791 dead cases had signatures `1` or `2`, so span was zero even though
  only one relation event existed.
- The stricter event-stage guard is
  `candidate_pos_span=0&candidate_pos_count>=2`.  It keeps the same public
  pair selector but requires at least two candidate-position events at the same
  position before a case is guard-passed.
- Replaying 752-791 under the strict guard preserves all existing wins:
  24 guarded verified below-rho cases total, with 8 on 752-759, 12 on 768-775,
  and 4 on 776-783.  Min ops/rho remains 0.512 and mean guarded ops/rho is
  0.856 across the 24 cases.
- The same strict guard rejects the 784-791 singleton dead cases: 72 selected
  pair/subset cases, 16 rank-1 singleton cases, zero guard-passed rank-1 cases,
  zero verified cases.
- This is a cleaner rank-lift boundary, not a new selection win.  The row/pair
  selector still pays for selected cases; the stricter guard prevents singleton
  relation events from being counted as guarded relation systems.

Fresh 792-799 strict-guard follow-up:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_792_799_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_792_799_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_792_799_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_792_799_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_792_799_fresh_manifest.json`
- Strict no-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_792_799.json`
- Direct stress found a separate target-cap1 below-rho signal on
  `22050.cf1@11731`: best policy `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`,
  4 verified stress-leaf labels, all 4 below rho, best ops/rho 0.43065693.
  This is not the repeated-coordinate pair-rule claim.
- The frozen repeated-coordinate gate selected 32 target-67 candidates, but
  verified zero public keys and had zero verified below-rho full-coordinate
  replays.  Minimum replayed ops/rho was 1.224.
- The strict no-exact pair rule selected 12 pair/subset cases, all rank 0, with
  zero relation pair cases, zero guard-passed cases, and zero verified cases.
  This is a clean repeated-coordinate negative for 792-799 under the stricter
  rank-lift guard.

Fresh 800-807 strict-guard replication:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_800_807_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_800_807_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_800_807_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_800_807_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_800_807_fresh_manifest.json`
- Strict no-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_800_807.json`
- Direct stress again has a separate target-cap1 scheduler signal:
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, 6 verified stress-leaf labels,
  all 6 below rho, best ops/rho 0.728.  This remains separate from the
  repeated-coordinate pair-rule claim.
- The frozen repeated-coordinate gate selected 130 target-67 candidates and
  verified 8 full-coordinate replays, but zero full-coordinate replays were
  below rho.  The best verified full replay was transfer 806, coordinate
  `(161,6976)`, deriving secret 2023 at 1.368 rho.
- The strict no-exact pair rule selected 68 pair/subset cases and recovered a
  fresh below-rho strict-guard positive: 8 relation pair cases, 4 guard-passed
  relation pair cases, and 4 guarded verified below-rho recoveries.
- Best strict guarded cases: transfer 806, coordinate `(161,6976)`,
  `fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0`, salts 201 and 207, leaf min 2,
  candidate-position signature `2,2`, rank 2, relation count 2, deriving secret
  2023 at 120/125 = 0.96 rho.
- The strict guard also rejects four relation-pair decoys on transfer 802:
  coordinate `(542,3911)`, salts 204 and 207, candidate-position signature
  `1,2`, rank 2 but public-key verification false at 0.976 rho.  This supports
  the rank-lift orientation interpretation: it is not enough to have two
  relation events; they must align to the same candidate position.

Candidate-position alignment audit, 752-807:

- Audit script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_candidate_position_alignment_audit.py`
- Audit artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_807.json`
- The audit replays the relation-bearing strict pair cases from 752-791,
  792-799, and 800-807, then records event-level candidate-position, coefficient
  support, RHS residue, salt, and term-shape features before using public-key
  verification only as a label.
- Replay integrity checks passed: 32 relation-bearing pair cases were
  rehydrated with zero context errors, zero skipped cases, zero source replay
  mismatches, and zero form-guard feature mismatches.
- Summary: 28 of 32 relation cases verify and are below rho; 4 of 32 are
  unverified rank-2 decoys.  Verified signatures are `1,1` (16 cases) and
  `2,2` (12 cases).  Unverified signatures are `1,2` (4 cases).
- The form-only separator is now explicit: `candidate_pos_aligned=1`,
  `candidate_pos_span=0`, `candidate_pos_unique_count=1`, and
  `candidate_pos_span0_count_ge2=1` occur in all 28 verified cases and zero
  unverified cases.  The negative atoms `candidate_pos_aligned=0`,
  `candidate_pos_span=1`, and `candidate_pos_signature=1,2` occur in all
  4 unverified decoys and zero verified cases.
- This strengthens the structural lead but not the claim boundary.  The guard
  is still event-stage, after public pair replay has exposed relation events;
  the next real promotion test is to freeze the same candidate-position
  alignment surface on 808-815 or connect it to a cheaper FFE/residual
  prefilter.

Fresh 808-815 strict-guard negative and 752-815 audit:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_808_815_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_808_815_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_808_815_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_808_815_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_808_815_fresh_manifest.json`
- Strict no-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_808_815.json`
- Updated candidate-position audit:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_815.json`
- Direct stress again has a separate target-cap1 scheduler signal:
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, 6 verified stress-leaf labels,
  all 6 below rho, and best stress-leaf ops/rho 0.632.  This remains separate
  from the repeated-coordinate pair-rule claim.
- The frozen repeated-coordinate gate selected 123 target-67 candidates and
  verified 11 full-coordinate replays, but zero full-coordinate replays were
  below rho.  The best verified full replay was transfer 811, coordinate
  `(1114,8506)`, deriving secret 7002 at 1.44 rho.
- The old row-rule form replay had 11 relation cases, but all 11 were rejected
  by `candidate_pos_span=0`; zero guarded verified cases and zero guarded
  below-rho cases survived.
- The strict no-exact pair rule selected 67 pair/subset cases and found 11
  rank-2 relation cases at nominal 0.96 rho, but all had candidate-position
  signature `1,2`, `candidate_pos_span=1`, and `candidate_pos_aligned=0`.
  Therefore none passed `candidate_pos_span=0&candidate_pos_count>=2`, and none
  verified.
- The combined 752-815 alignment audit rehydrated 43 relation-bearing strict
  pair cases with zero context errors and zero skipped cases.  The verified side
  remains exactly the 28 below-rho cases from 752-807, with signatures `1,1`
  and `2,2`; all 15 unverified decoys now have signature `1,2`, span 1, and
  misalignment.
- This is a useful negative replication.  It strengthens the candidate-position
  alignment interpretation, but it also shows the unchanged no-exact pair rule
  does not fire positively on every fresh window.  Promotion now needs either a
  cheaper pre-event proxy for aligned candidate positions or another disjoint
  unchanged-rule positive.

Pre-event proxy mining over 752-815 alignment audit:

- Proxy miner script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_pre_event_proxy_miner.py`
- Proxy miner artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_815.json`
- The miner consumes the 752-815 alignment audit and builds two diagnostic
  surfaces that exclude candidate-position labels, exact coordinates, and exact
  salt signatures:
  - `public_pair_no_position_no_exact`
  - `form_residual_no_position_no_exact`
- The best public negative guard is the single atom `transfer_index_mod3=1`.
  On the current 43 relation-bearing cases, it rejects all 15 unverified rank-2
  decoys, rejects zero of the 28 verified below-rho cases, and covers exactly
  the decoy transfers 802 and 811.
- Leave-one-window checks for that guard are encouraging but small: holding out
  800-807, the guard trained on the remaining windows rejects all 4 held-out
  unverified cases and no verified cases; holding out 808-815, it rejects all
  11 held-out unverified cases and no verified cases.
- Positive public atoms remain fragmented.  For example, `pair_salt_max_mod5=3`
  and `transfer_index_mod3=0` each cover 20 verified cases from 752-759 and
  768-775 with zero unverified cases, but they do not cover the 776-783 or
  800-807 positives.
- This gives a concrete next frozen proxy test:
  `reject_if transfer_index_mod3=1`, then run 816-823 with the unchanged pair
  rule and strict rank-lift guard.  The claim is only that this is a diagnostic
  pre-event negative-control candidate; it is not yet a structural explanation
  or a general selector.

Proxy guard filter audit on 752-815:

- Filter audit script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_proxy_guard_filter_audit.py`
- Filter audit artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_815.json`
- Applying the frozen reject clause `transfer_index_mod3=1` to the current
  relation-case corpus keeps all 28 verified below-rho cases, rejects all 15
  unverified rank-2 decoys, leaves zero unverified cases kept, and false-rejects
  zero verified cases.
- By window: 752-759 keeps 8/8 verified, 768-775 keeps 12/12 verified, 776-783
  keeps 4/4 verified, 800-807 keeps 4/4 verified while rejecting 4 unverified
  decoys, and 808-815 rejects 11 unverified decoys.
- This is the exact local invariant the future wrapper must preserve before
  testing 816-823.  It uses existing replay labels only for audit accounting,
  so it is not a fresh-window validation by itself.

Fresh 816-823 strict-guard calibration:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_816_823_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_816_823_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_816_823_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_816_823_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_816_823_fresh_manifest.json`
- Strict no-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_816_823.json`
- Updated alignment/proxy artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_823.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_823.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_823.json`
- Direct stress again has a separate target-cap1 scheduler signal:
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, 4 verified stress-leaf labels,
  all 4 below rho, best ops/rho 0.56.  This remains outside the
  repeated-coordinate pair-rule claim.
- The frozen repeated-coordinate gate selected 116 target-67 candidates, but
  verified zero public keys and had zero verified below-rho full-coordinate
  replays.  Minimum replayed ops/rho was 1.224.
- The old row-rule replay had 80 selected cases and zero relation cases.
- The strict no-exact pair rule selected 116 pair/subset cases, but produced
  zero relation pair cases, zero guard-passed cases, zero verified cases, and
  zero below-rho recoveries.
- The 752-823 alignment audit is relation-case unchanged because 816-823 adds
  no relation cases: it still has 43 relation-bearing strict pair cases, 28
  verified below rho, and 15 unverified rank-2 decoys.
- The proxy filter audit is also unchanged in relation-case terms:
  `reject_if transfer_index_mod3=1` keeps all 28 verified below-rho cases,
  rejects all 15 unverified decoys, keeps zero unverified cases, and rejects
  zero verified cases.
- This is not a fresh proxy challenge because there was no 816-823 relation
  system to accept or reject.  It is a useful dead-selection calibration:
  public pair selection can spend 116 selected pair cases without producing a
  relation case, while the proxy invariant remains intact.

Fresh 824-831 no-relation calibration and relation-reach audit:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_824_831_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_824_831_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_824_831_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_824_831_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_824_831_fresh_manifest.json`
- Strict no-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_824_831.json`
- Updated 752-831 audit artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_831.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_831.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_831.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_relation_reach_audit_target67_752_831.json`
- Direct stress again has a separate scheduler signal:
  `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0` has 2 verified stress-leaf
  labels, both below rho, best ops/rho 0.45985401.  This stays separate from
  the target-67 repeated-coordinate pair-rule claim.
- The frozen repeated-coordinate gate selected 156 public cases overall, with
  the target-67 slice contributing 12 candidates.  The target-67 coordinate
  replay verified zero public keys, produced zero verified below-rho full
  replays, and had minimum replayed ops/rho 1.264.
- The strict no-exact pair rule selected 16 pair/subset cases in 824-831, all
  on transfer 830, but produced zero relation pair cases, zero guard-passed
  cases, zero verified cases, and zero below-rho recoveries.
- The 752-831 candidate-position alignment audit remains relation-case
  unchanged because 816-823 and 824-831 both add no relation cases: 43
  relation-bearing strict pair cases, 28 verified below rho, and 15 unverified
  rank-2 decoys.
- The frozen proxy `reject_if transfer_index_mod3=1` is still clean on that
  relation-case corpus: it keeps all 28 verified below-rho cases, rejects all
  15 unverified decoys, keeps zero unverified cases, and rejects zero verified
  cases.  Again, this is not fresh proxy validation because the new 824-831
  window did not reach relation cases.
- A new relation-reach audit separates dead public-pair selection from
  candidate-position alignment.  Across 752-831 it counts 511 selected strict
  pair/subset cases, 43 relation-reaching cases, 28 guard-passed verified
  below-rho cases, and dead-selection windows 784-791, 792-799, 816-823, and
  824-831.
- The immediate bottleneck is now public relation reach, not only form
  orientation.  Candidate-position alignment remains the right downstream guard
  once relation cases exist; before the next promotion attempt, the pair
  selector needs a public relation-reach/dead-selection gate that excludes
  verifier outcomes, exact coordinates, and exact salts.

Fresh 832-839 branch-gap calibration:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_832_839_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_832_839_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_832_839_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_832_839_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_832_839_fresh_manifest.json`
- Strict no-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_832_839.json`
- Updated 752-839 audit artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_839.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_839.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_839.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_relation_reach_audit_target67_752_839.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_gap_audit_target67_832_839.json`
- Direct stress shifted to target-cap3/global only: target-cap1 had zero
  verified stress rows or leaves; target-cap3 had 24 verified stress-leaf
  labels with best ops/rho 1.36 and zero below-rho leaves; global-cap3 had
  8 verified stress-leaf labels with best ops/rho 1.44 and zero below-rho
  leaves.
- The frozen target-67 coordinate gate replayed 87 candidates, verified 31
  public keys, and verified zero below rho.  The verified coordinates are
  over-rho hits on `(55,7531)`, including transfer 834 at 1.36 rho.
- The old row/form replay selected 67 cases and reached 8 relation cases, but
  all 8 were rejected by the form guard and none verified.  Those relation
  cases sit at transfer 835, coordinate `(542,3911)`, with candidate-position
  signature `1,3`, rank 2, and nominal 0.552-rho row cost.
- The strict no-exact pair rule selected only 4 pair/subset cases, all on
  transfer 838 and coordinate `(161,6976)`, but reached zero relation cases.
  It therefore missed both the over-rho coordinate-gate verified branch and the
  row/form relation branch.
- The 752-839 relation-case candidate-position/proxy audits remain unchanged:
  43 strict-pair relation cases, 28 verified below rho, and 15 unverified rank-2
  decoys.  The frozen proxy `reject_if transfer_index_mod3=1` still keeps all
  28 verified below-rho cases and rejects all 15 decoys, but 832-839 adds no
  strict-pair relation case to challenge it.
- The 752-839 relation-reach audit now counts 515 selected strict pair/subset
  cases and still only 43 relation-reaching cases.  Dead-selection windows are
  784-791, 792-799, 816-823, 824-831, and 832-839.
- The actionable conclusion is sharper than a simple no-relation result: the
  current no-exact strict pair rule is no longer the right front door for the
  832-839 target-cap3/global branch.  The next algorithmic move should mine a
  public branch selector from coordinate-gate verified over-rho hits and
  row/form relation misses, then only apply candidate-position/proxy guards
  after that selector reaches relation systems.

Branch selector miner across 752-839:

- Branch selector script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_branch_selector_miner.py`
- Branch selector artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_752_839.json`
- The miner aggregates coordinate-gate candidates, old row/form replay cases,
  and strict pair-rule cases by window, transfer, and repeated coordinate.  It
  scores labels with replay/verifier outcomes but candidate atoms exclude exact
  coordinate keys, exact salt signatures, verifier labels, and
  candidate-position labels.
- Corpus: 103 public transfer/coordinate keys across 11 windows.  Label counts:
  13 coordinate-gate verified keys, 7 coordinate-gate verified keys missed by
  strict pair selection, 2 row/form relation keys, 1 row/form relation key
  missed by strict pair selection, 8 total branch-gap keys, and 36 strict-pair
  dead keys.
- Best current no-exact branch-gap selector:
  `b_mod4=3&salt_mod2_pattern=0,1,0`.
  It selects 4 branch-gap keys, 0 non-branch keys, precision 1.0, recall 0.5,
  and spans 752-759, 776-783, and 832-839.  This catches a reusable
  coordinate-gate-missed family without using exact coordinates or exact salts.
- The narrower coordinate-55 family candidates include
  `b_minus_c_mod16=12&salt_mod2_pattern=0,1,0` and
  `b_minus_c_mod16=12&salt_span=6`; each selects 3 branch-gap keys with zero
  false positives across 776-783 and 832-839.
- The row/form relation-missed branch at 832-839 has only one positive key.
  Its apparent separators, such as `b_mod16=14&salt_mod2_pattern=0,0,1`, are
  single-window diagnostics and should not be promoted until a future window
  repeats the `1,3` misalignment family.
- The strict-pair dead selector is dominated by `source_ops_millirhos=816`,
  selecting 23 strict-pair-dead keys across 10 windows with zero false
  positives in this corpus.  That is useful as a public rejection cue for the
  old pair rule, not as a positive relation selector.
- Current next frozen package for a future window:
  1. run coordinate-gate/row-form/strict-pair artifacts as before;
  2. score the frozen branch-gap selector
     `b_mod4=3&salt_mod2_pattern=0,1,0` before strict-pair promotion;
  3. keep candidate-position/proxy guards downstream only after the branch
     selector reaches relation systems.

Fresh 840-847 branch-selector validation and strict-pair refresh:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_840_847_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_840_847_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_840_847_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_840_847_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_840_847_fresh_manifest.json`
- Strict no-exact pair-rule replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_840_847.json`
- Fresh branch-gap and fixed-selector artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_gap_audit_target67_840_847.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_840_847_fixed_branch_rules.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_752_847.json`
- Updated 752-847 audit artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_847.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_847.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_847.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_relation_reach_audit_target67_752_847.json`
- Direct stress remains over-rho for the repeated-coordinate target-67 claim:
  target-cap1 has zero verified stress leaves, target-cap3 has 20 verified
  stress leaves with best ops/rho 1.00729927, and global-cap3 has 36 verified
  stress leaves with best ops/rho 1.01459854.
- The frozen coordinate gate replayed 72 target-67 candidates and verified 16
  public keys, all over rho.  The strongest coordinate-gate branch is transfer
  847 on `(55,7531)` at 1.336 rho.  The old row/form replay selected 48 cases
  and reached zero relation cases.
- The strict no-exact pair rule improved relative to 832-839: it selected 32
  pair/subset cases, reached 12 relation cases, and verified 8 public keys at
  1.032 rho on transfer 846 coordinate `(281,3224)`.  This is a tighter
  over-rho branch, not a below-rho win.
- The branch-gap audit shows one coordinate-gate verified key still missed by
  strict pair: transfer 847 `(55,7531)`.
- The frozen branch-gap selector survived the fresh check.  On 840-847,
  `b_mod4=3&salt_mod2_pattern=0,1,0` selects the single missed verified
  coordinate key, with zero false positives and zero strict-pair-dead
  selections.  The narrower
  `b_minus_c_mod16=12&salt_mod2_pattern=0,1,0` does the same; the older
  `b_minus_c_mod16=12&salt_span=6` abstains because this fresh salt span is 4.
- Across the combined 752-847 corpus there are 114 public transfer/coordinate
  keys across 12 windows: 16 coordinate-gate verified keys, 8 coordinate-gate
  verified keys missed by strict pair, 2 row/form relation keys, 1 row/form
  relation key missed by strict pair, 9 total branch-gap keys, and 38
  strict-pair dead keys.  The frozen selector
  `b_mod4=3&salt_mod2_pattern=0,1,0` now selects 5/9 branch-gap keys, zero
  false positives, and spans 752-759, 776-783, 832-839, and 840-847.
- The combined 752-847 relation-reach audit counts 547 selected strict
  pair/subset cases, 55 relation-reaching cases, 36 verified relation cases,
  and 28 verified below-rho cases.  Fresh 840-847 contributes 12 relation
  cases: 8 verified over rho on transfer 846 and 4 unverified relation cases on
  transfer 841.
- The proxy `reject_if transfer_index_mod3=1` remains clean after the fresh
  relation cases.  It rejects all 19 unverified relation cases, including the
  four new transfer-841 decoys, keeps all 28 verified below-rho cases, keeps
  the eight fresh verified-over-rho cases, and rejects zero verified cases.
- Current status: the branch selector is now a repeatable public diagnostic for
  coordinate-gate branches missed by strict pair.  It is not yet a completed
  ECDLP speedup, because the fresh strict-pair relation reach verified only
  over rho and the branch selector has not yet been converted into a cheaper
  relation-producing recovery path.

Branch-aware subset replay after the frozen selector:

- Replay support:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_subset_pair_rule_replay_probe.py`
  now accepts repeatable `--candidate-clause` filters before pair/subset
  expansion.
- Branch subset miner:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_branch_subset_rule_miner.py`
- Fresh 840-847 branch-aware all-subsets replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_840_847.json`
- Combined 752-847 branch-aware all-subsets replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_752_847.json`
- Combined 752-847 branch-aware two-row replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_752_847.json`
- Branch subset miner artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_subset_rule_miner_target67_bmod4_saltmod2_752_847.json`
- The frozen candidate clause
  `b_mod4=3&salt_mod2_pattern=0,1,0` selects 51/1049 coordinate-gate
  candidates across 752-847.
- On fresh 840-847, the branch-aware all-subsets replay selected 28 cases,
  reached 8 relation cases, verified all 8, and produced 4 below-rho
  recoveries.  Best case: transfer 847, coordinate `(55,7531)`, salts 204 and
  208, derived secret 5840, at 116/125 = 0.928 rho.  This converts the
  previous over-rho coordinate-gate branch for the same key into a below-rho
  two-row relation system.
- Across 752-847, the branch-aware all-subsets replay selected 357 cases,
  reached 86 relation cases, verified all 86, and produced 36 below-rho
  recoveries.  Minimum guarded verified cost is 0.896 rho and mean guarded
  verified cost is 1.14930233 rho.
- The cheaper two-row replay is the important contraction: it selected 153
  cases, reached 43 relation cases, verified all 43, and preserved all 36
  below-rho recoveries.  Minimum guarded verified cost is 0.896 rho and mean
  guarded verified cost drops to 0.936 rho.
- Two-row replay by window:
  752-759 has 8 verified and 8 below rho, best 0.960 rho, transfer 757,
  coordinate `(239,8922)`;
  776-783 has 24 verified and 24 below rho, best 0.896 rho, transfers 779 and
  780, coordinate `(55,7531)`;
  832-839 has 7 verified and zero below rho, best 1.032 rho, transfer 832,
  coordinate `(55,7531)`;
  840-847 has 4 verified and 4 below rho, best 0.928 rho, transfer 847,
  coordinate `(55,7531)`.
- No guarded relation case in the two-row replay failed public-key
  verification.
- The branch subset miner found clean but narrow historical selectors such as
  `pair_salt_delta_from_min_signature=0,2`, which selects 20 below-rho cases
  on 776-783 with zero false positives.  It is not promoted as the branch rule:
  the fresh 840-847 below-rho case uses `pair_salt_delta_from_min_signature=0,4`.
- Current status: the branch selector has now been turned into a measured
  below-rho fresh recovery path for 840-847 under two-row public subset replay.
  The honest claim is still a candidate algorithmic branch, not a completed
  general ECDLP index-calculus speedup, because the selector/subset package
  needs a future frozen-window validation and a final end-to-end accounting of
  candidate-filter and relation-search costs.

Fresh 848-855 frozen abstention:

- Fresh stress source:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_848_855_probe.json`
- Frozen pipeline artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_848_855_fresh_public_selector.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_848_855_fresh_coordinate_gate.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_848_855_fresh_guard_replay.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_848_855_fresh_manifest.json`
- Frozen branch-aware two-row replay:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_848_855.json`
- The fixed-selector stress summaries are fully negative on 848-855:
  target-cap1, target-cap3, and global-cap3 all have zero stress-row
  verification, zero stress-leaf verification, and zero below-rho labels.
- The target-67 public selector emits zero public bounded cases.  The
  coordinate gate therefore has zero candidates, zero replayed candidates, and
  zero verified public keys.
- The frozen branch-aware package
  `candidate_clause=b_mod4=3&salt_mod2_pattern=0,1,0`,
  `pair_rule=subset_size=2`, and
  `form_guard=candidate_pos_span=0&candidate_pos_count>=2` also emits zero
  input pair cases and zero relation cases.
- Interpretation: 848-855 is a clean abstention, not a failure.  It does not
  add a second fresh below-rho recovery window, but it also introduces no
  guarded unverified relation case and no over-rho-only branch-aware recovery.

Fresh 856-863 and 864-871 frozen abstentions:

- Fresh stress sources:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_856_863_probe.json`
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_864_871_probe.json`
- Frozen pipeline manifests:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_856_863_fresh_manifest.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_864_871_fresh_manifest.json`
- Frozen branch-aware two-row replays:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_856_863.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_864_871.json`
- Both windows repeat the 848-855 abstention pattern.  The fixed-selector
  stress summaries have zero stress-row verification, zero stress-leaf
  verification, and zero below-rho labels for target-cap1, target-cap3, and
  global-cap3.
- The target-67 public selector emits zero public bounded cases in both
  windows.  The coordinate gates therefore have zero candidates, zero replayed
  candidates, and zero verified public keys.
- The frozen branch-aware package has zero input pair cases and zero relation
  cases in both windows.
- Interpretation: 848-871 is now three consecutive frozen activation
  abstentions after the 840-847 positive.  This is not evidence of a bad
  relation form; it shows that the current public selector did not expose any
  target-67 bounded repeated-coordinate branch to test.  The next useful
  evidence needs either a wider frozen activation scan or a pre-activation FFE
  cue that predicts when target-67 repeated-coordinate rows will appear.

Batched frozen activation scan through 903:

- Activation scan script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_branch_activation_scan.py`
- Post-positive scan artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_activation_scan_target67_bmod4_saltmod2_subset2_848_903.json`
- Full local activation ledger artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_activation_scan_target67_bmod4_saltmod2_subset2_752_903.json`
- Fresh 872-903 stress windows:
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_872_879_probe.json`
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_880_887_probe.json`
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_888_895_probe.json`
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_896_903_probe.json`
- The 872-903 stress windows repeat the 848-871 shape: all fixed target-cap1,
  target-cap3, and global-cap3 summaries have zero stress-row verification,
  zero stress-leaf verification, and zero below-rho labels.
- The frozen target-67 pipeline emits zero selected public cases, zero
  coordinate-gate candidates, and zero guard-replay cases in all four windows.
- The frozen branch-aware two-row replay emits zero input pair cases and zero
  relation cases in all four windows.
- The post-positive scan over 848-903 has 7/7 manifests, 7/7 branch replay
  artifacts, zero selected public windows, zero coordinate-candidate windows,
  zero branch-clause windows, and seven abstention windows:
  848-855, 856-863, 864-871, 872-879, 880-887, 888-895, and 896-903.
- The full local activation ledger over 752-903 has 19 manifests.  It records
  12 activation windows from 752-847, 7 abstention windows from 848-903,
  1049 total coordinate candidates before the dry spell, and 51 total matches
  to the frozen branch clause before the dry spell.  Per-window branch replay
  artifacts currently exist only for 848-903 in this ledger; the earlier
  below-rho relation evidence remains in the combined 752-847 branch replay.
- Interpretation: the current frozen relation package has not been falsified
  by 848-903.  The bottleneck has moved upstream: the fixed public selector is
  no longer activating target-67 repeated-coordinate branches after 840-847.
  The next useful work is a public pre-activation cue or a wider activation
  scheduler, not more two-row replay on empty coordinate gates.

Pre-activation miner and older-window form boundary:

- Pre-activation miner script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_preactivation_miner.py`
- Wider activation scan:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_activation_scan_target67_bmod4_saltmod2_subset2_672_903.json`
- Pre-activation artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_preactivation_miner_target67_activated_752_903.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_preactivation_miner_target67_branch_clause_752_903.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_preactivation_miner_target67_activated_672_903.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_preactivation_miner_target67_branch_clause_672_903.json`
- Over 672-903, the simple public stress cue `has_target67_leaf_total3=1`
  separates all 22 activation windows from the seven dry 848-903 windows with
  zero false positives and zero false negatives.
- For the narrower branch-clause label, the public cue
  `target67_salt_mod2_pattern_0_0_1_count>=8&target67_salt_mod3_pattern_0_1_1_count>=1`
  selects 6/8 branch-clause windows with zero false positives.  It misses
  704-711 and 736-743, so it is a high-precision branch scheduler cue rather
  than a complete activation model.
- Older-window branch-aware backtest artifacts:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_680_687.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_704_711.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_728_735.json`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_736_743.json`
- The frozen branch clause reaches relation systems only in 680-687 among
  those older backtests.  The strict same-position guard rejects all 680
  relation cases, but the replay exposes a new boundary: signature `1,3,3`
  has 11 verified below-rho cases at 0.976 rho, while signature `1,1,3,3`
  has 11 verified over-rho cases at 1.032 rho.
- Form-boundary audit:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_branch_form_boundary_audit.py`
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_form_boundary_audit_target67_bmod4_saltmod2_subset2_680_847.json`
- Across 680 and 752-847, the audit records 43 strict same-position relation
  cases, all verified, with 36 below rho and minimum cost 0.896 rho.  It also
  records 22 partial-duplicate two-position relation cases from 680, all
  verified, with 11 below rho and minimum cost 0.976 rho.
- Interpretation: partial-duplicate form is an important diagnostic, but not a
  promoted guard.  A relaxed guard must first reject known misaligned decoys
  and validate on a future frozen replay.

Partial-duplicate guard extension audit:

- New audit script:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_partial_duplicate_guard_audit.py`
- New artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_partial_duplicate_guard_audit_target67_bmod4_saltmod2_680_903.json`
- The audit combines branch-aware replay relation cases from 680, 704, 728,
  736, and 752-847; strict-pair alignment cases and decoys through 752-847;
  the 832 row/form branch-gap decoy aggregate; and the seven dry 848-903
  activation-abstention windows.
- Weighted relation-case corpus: 128 cases across replay families, with 75
  verified below rho, 26 verified over rho, and 27 unverified decoys.  Form
  signatures in the corpus are `1,1`, `2,2`, `1,3,3`, `1,1,3,3`, `1,2`, and
  `1,3`.
- Baseline strict same-position guard accepts 64 verified below-rho cases, 15
  verified over-rho cases, and zero unverified cases; it rejects the 11
  verified below-rho 680 partial-duplicate cases.
- Candidate structural extension:
  `strict_same_position OR (candidate_pos_count=3 & candidate_pos_unique_count=2 & candidate_pos_max_duplicate>=2)`.
  It accepts 75 verified below-rho cases, 15 verified over-rho strict cases,
  and zero unverified cases.  It rejects all 27 unverified decoys and the 11
  verified over-rho `1,1,3,3` controls.
- Exact `1,3,3` extension gives the same retrospective result.  The structural
  version is preferred as a hypothesis because it names the form shape instead
  of hard-coding one signature.
- Broad relaxation is unsafe as a speedup guard: accepting any partial
  duplicate also admits the `1,1,3,3` verified-over-rho class.  Accepting any
  misaligned relation admits all 27 known unverified decoys.
- Interpretation: the partial-duplicate extension is now a frozen candidate
  guard for the next nonempty activation window.  It is still not an
  algorithmic promotion because the useful support is one older window.

Executable partial-count3 guard replay:

- Replay support update:
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_form_guard_replay_probe.py`
  now exposes `candidate_pos_max_duplicate` and accepts `|` alternatives in
  form-guard expressions.
- Executable replay artifact:
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_partial_count3_680_847.json`
- Executed guard:
  `candidate_pos_span=0&candidate_pos_count>=2|candidate_pos_count=3&candidate_pos_unique_count=2&candidate_pos_max_duplicate>=2`
- Running the actual branch-aware subset replay over 680, 704, 728, 736, and
  752-847 selected 222 pair cases, reached 65 relation cases, passed 54
  relation cases through the executable guard, verified all 54 guard-passed
  cases, and produced 47 below-rho cases.  The guard-passed minimum cost is
  0.896 rho and the mean cost is 0.94414815 rho.
- This is stronger than post-processing the audit because future windows can
  invoke the same guard expression directly.  It is still retrospective replay
  support, not a forward promotion: no 904+ target-67 activation artifact was
  present in the current worktree, and the known 848-903 windows remain dry.
