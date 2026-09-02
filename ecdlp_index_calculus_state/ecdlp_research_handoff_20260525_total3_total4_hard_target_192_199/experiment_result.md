# Experiment Result

Primary artifacts:

- Fresh stress: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_192_199_probe.json`
- Strict signature: `ecdlp_index_calculus_state/low_term_total3_total4_signature_fixed_selector_192_199.json`
- Candidate signature: `ecdlp_index_calculus_state/low_term_total3_total4_candidate_signature_fixed_selector_192_199.json`
- Gate: `ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_fixed_selector_192_199.json`
- Sage factors: `ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_single_hit_root_192_199.json`
- Factor audit: `ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total3_total4_single_hit_root_192_199.json`
- Root policy: `ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total3_total4_192_199.json`
- Assembly selector: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_selector_total3_total4_192_199.json`
- Assembly replay: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_replay_total3_total4_192_199.json`

Fresh stress and signature:

- Transfers: 192-199.
- Leaf selectors: `mode_cost_low_term_support_total3`,
  `mode_low_term_support_total3`, `mode_cost_low_term_support_total4`,
  `mode_low_term_support_total4`.
- Best frozen row selector: `target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`.
- Verified below-rho leaf cases: 14.
- Best strict ops/rho: 0.528.
- Strict target: `67.a1@9803`.
- Strict transfer set: 196, 197, 199.
- Candidate below-rho-all case count: 65.

FFE/root component:

- Pre-factor gate input cases: 14.
- Gate selected surfaces: 3/3.
- Sage factored selected surfaces: 3/3, with 73 resultant factors.
- All 73 factors are quadratic root hyperplanes.
- One surface has a preserving Sage factor.
- Best preserving surface: `67.a1@9803`, transfer 196, row salt 204.
- Best preserving surface FFE ops/rho: 0.576.
- Best preserving root-scan ops/rho: 0.68.
- Prefactor unique-leaf single-hit-root policy selected 1/1 eligible surface.
- Policy result: preserving, false-positive-free, below rho.
- Policy total ops/rho: 0.752; direct ops/rho: 0.736.

Assembly replay:

- Selector retained 1 FFE surface and 4 source cases.
- Retained target: `67.a1@9803`.
- Retained transfer: 196.
- Retained row: `67.a1@9803:uniform:256:salt204`.
- Retained leaf signatures: `[0, 2, 3]` and `[0, 1, 2, 3]`.
- Replay verified 4/4 selected cases.
- Same-challenge verified groups: 1/1.
- Challenge-group rank: 2.
- Challenge-group relation count: 2.
- Derived secret: 303.
- Retained-only ops/rho: min 0.528, mean 0.532, max 0.536.

Interpretation:

This is the first measured hard-target retained-root replay positive in this
thread: the total3/total4 public leaf family produced a below-rho
single-surface FFE root anchor on `67.a1@9803`, and verifier replay recovered a
rank-2 same-challenge derivation.  It is still a component result, because the
replay harness consumes signature-provided row/leaf keys.  The next proof
obligation is to make the total3/total4 row/leaf assembly selector standalone
and repeat on a fresh preregistered window.
