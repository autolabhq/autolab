# Experiment Result

## Fresh 160-167 Run

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_single_hit_root_policy_preregistration_160_167.json`

Artifacts:

- Stress: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_160_167_probe.json`
- Signature: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_160_167.json`
- Gate: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_160_167.json`
- Sage factors: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_single_hit_root_preregistered_160_167.json`
- Factor audit: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_single_hit_root_preregistered_160_167.json`
- Strict policy eval: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total2_160_167.json`
- Broad unique-leaf control: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_hit_root_policy_total2_160_167.json`
- Broad selected-hit-root control: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_hit_root_policy_total2_160_167.json`

Stress and signature:

- Total2 stress found 22 strict below-rho, public-key-verified cases.
- Targets: `22050.cf1@11731` and `67.a1@9803`.
- Best strict case: `67.a1@9803`, transfer 162, salt 202, leaves 3 and 4,
  at 0.552 ops/rho.
- Best `22050.cf1@11731` strict case: transfer 167, salts 172 and 175,
  leaf 90, at 0.69343066 ops/rho.
- Strict signature produced 13 distinct surface IDs.
- Pre-factor gate selected 10 surfaces.
- Every gate-selected surface had exactly one distinct pre-factor selected hit
  root, so the strict single-hit-root gate and broad all-prefactor gate select
  the same 10 surfaces in this window.

Sage factorization:

- Requested/factored selected surfaces: 10/10.
- Resultant factor candidates: 205.
- Preserving factor surfaces: 10/10.
- All factors were quadratic root hyperplanes.
- Preserving factor root-scan below rho: 10/10.
- Preserving direct/surface below rho: 10/10.
- Best preserving root-scan cost: 0.48905109 ops/rho.
- Full remainders remained above rho; best full remainder was 1.35766423
  ops/rho.

Primary policy result:

- `prefactor_unique_leaf_hit_root_first_target_hash` with
  `single_prefactor_hit_root`: 10/10 preserving, 0 false positives, 10/10
  scan-below-rho, 10/10 direct-below-rho.
- Mean evaluated root count: 1.0.
- Mean selector eval ops: 3.3.
- Scan cost: min 0.51094891, mean 0.61344818, max 0.824 ops/rho.
- Direct cost: min 0.46715328, mean 0.53441168, max 0.72 ops/rho.

Controls:

- Broad all-prefactor `prefactor_unique_leaf_hit_root_first_target_hash`:
  10/10 preserving, 0 false positives, 10/10 scan/direct below rho.
- Broad all-prefactor `prefactor_selected_hit_root_first_target_hash`:
  10/10 preserving, 0 false positives, 10/10 scan/direct below rho.
- The broad controls match the strict policy because there were no multi-root
  gate-selected surfaces in 160-167.

Interpretation:

This is a second fresh preregistered positive for the single-hit-root total2
component, and it is stronger than 152-159 by surface count. The failure mode
seen in 144-151 and 152-159 broad controls did not appear here because no
selected surface had multiple pre-factor hit roots. The promoted component is
therefore now: when a total2 surface has exactly one public pre-factor hit
root, that root selects a preserving first-fall hyperplane below rho on the
measured fresh windows.
