# Experiment Result

New probe:

- `tasks/ecdlp_index_calculus/low_term_totalk_public_stress_selector_probe.py`

192-199 public-only selector artifacts:

- Public selector: `ecdlp_index_calculus_state/low_term_total3_total4_public_stress_selector_192_199.json`
- Gate: `ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_public_stress_selector_192_199.json`
- Sage factors: `ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_public_stress_selector_192_199.json`
- Factor audit: `ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total3_total4_public_stress_selector_192_199.json`
- Root policy: `ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total3_total4_public_stress_selector_192_199.json`
- Assembly selector: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_selector_total3_total4_public_stress_192_199.json`
- Replay: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_replay_total3_total4_public_stress_192_199.json`

192-199 result:

- Raw public below-rho candidate cases: 65.
- Public selector retained 30 cases with a cap of 4 per target/transfer.
- Gate materialized 21 surfaces and selected 5.
- Sage factored 5/5 selected surfaces.
- All 110 resultant factors were quadratic root hyperplanes.
- Preserving below-rho root-scan surfaces: 4/5.
- Prefactor policy selected 4/4 eligible surfaces, all preserving and
  false-positive-free.
- Policy total ops/rho: min 0.49635036, mean 0.66059854, max 0.832.
- Assembly retained 4 surfaces and 11 public-selected source cases.
- Replay verified 4/11 cases and 1/2 same-challenge groups.
- Hard-target verified group: `67.a1@9803`, transfer 196, rank 2,
  derived secret 303.
- Hard-target retained ops/rho on verified cases: 0.528 and 0.536.

200-207 bounded public selector artifacts:

- Fresh stress: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_200_207_probe.json`
- Public bounded selector: `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_selector_200_207.json`
- Gate: `ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_public_bounded_selector_200_207.json`
- Sage factors: `ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_public_bounded_selector_200_207.json`
- Factor audit: `ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total3_total4_public_bounded_selector_200_207.json`
- Root policy: `ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total3_total4_public_bounded_selector_200_207.json`
- Assembly selector: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_selector_total3_total4_public_bounded_200_207.json`
- Replay: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_replay_total3_total4_public_bounded_200_207.json`

200-207 result:

- Raw below-rho total3/total4 stress cases: 0.
- Bounded public selector admitted candidates with ops/rho below 1.5.
- Public bounded candidate cases before cap: 180.
- Public selector retained 48 cases with a cap of 4 per target/transfer.
- Selected verifier-positive labels: 0.
- Gate materialized 44 surfaces and selected 7.
- Sage factored 7/7 selected surfaces.
- All 158 resultant factors were quadratic root hyperplanes.
- Every selected surface had a preserving below-rho root-scan factor.
- Prefactor policy selected 7/7 eligible surfaces, all preserving and
  false-positive-free.
- Policy total ops/rho: min 0.51094891, mean 0.64043379, max 0.744.
- Assembly retained 7 surfaces and 20 public-selected source cases.
- Replay verified 0/20 cases and 0/4 same-challenge groups.
- Challenge groups had relation forms but no public-key derivation:
  `22050.cf1@11731` transfer 201 reached rank 2 with two unique forms but
  `derived=false`; hard-target groups stayed rank 1.

Interpretation:

The 192-199 hard-target result survives a public-only row/leaf selector, so the
previous positive was not merely an artifact of strict signature membership.
The 200-207 follow-up shows the next boundary: the FFE first-fall/root-policy
component generalizes strongly, even rescuing bounded over-rho public rows into
below-rho root scans, but relation-form independence does not yet generalize
enough to derive the key.  The current blocker is independent relation-form
assembly, not Sage factor shape or public root selection.
