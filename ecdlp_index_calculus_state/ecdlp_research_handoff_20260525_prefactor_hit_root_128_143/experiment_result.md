# Experiment Result

## 128-135 Target-Hash Test

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_target_root_hash_total2_preregistration_128_135.json`

Artifacts:

- Stress: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_128_135_probe.json`
- Signature: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_128_135.json`
- Gate: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_128_135.json`
- Sage factors: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_preregistered_128_135.json`
- Factor audit: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_preregistered_128_135.json`
- Root selector: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total2_preregistered_128_135.json`
- Frozen evaluator: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_frozen_root_policy_total2_train_le87_test_128_135.json`

Result:

- Total2 stress: 4 verified below-rho leaf cases, best 0.552 ops/rho.
- Strict signature: 4 strict cases collapsed to one unique surface:
  `67.a1@9803|67.a1@9803:uniform:256:salt206|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:134:67.a1@9803`.
- Gate: 1/1 surface selected.
- Sage factorization: 27 root-hyperplane factors, 1 preserving factor.
- Preserving factor root-scan cost: 0.792 ops/rho.
- Preserving surface cost: 0.592 ops/rho.
- Full remainder remained above rho at 3.6 ops/rho.
- Preregistered `target_root_hash`: preserving and false-positive-free, but
  too late in the root order: 1.896 scan ops/rho and 1.776 direct ops/rho.

Interpretation:

`target_root_hash` is a controlled negative on 128-135. The component existed,
but the static hash order charged too many public factor probes.

## Pre-Factor Hit-Root Diagnostic

New script:

- `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_prefactor_hit_root_policy_probe.py`

Diagnostic artifacts:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_hit_root_policy_total2_112_135.json`
- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_hit_root_policy_total3_total4_120_127_negative_control.json`

Result:

- On total2 112-135, `prefactor_selected_hit_root_first_target_hash` was 6/6
  preserving, 0 false positives, 6/6 scan-below-rho, and 6/6
  direct-below-rho.
- Mean evaluated root count was 1.0.
- Mean scan cost was 0.69402433 ops/rho.
- Mean direct cost was 0.60052555 ops/rho.
- On the known total3/4 120-127 negative-control surface, the same policy
  selected a false positive. This keeps the policy total2-scoped.

## 136-143 Fresh Preregistration

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_hit_root_policy_preregistration_136_143.json`

Artifacts:

- Stress: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_136_143_probe.json`
- Signature: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_136_143.json`
- Gate: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_136_143.json`
- Sage factors: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_preregistered_136_143.json`
- Factor audit: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_preregistered_136_143.json`
- Primary policy eval: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_hit_root_policy_total2_136_143.json`
- Unique-leaf diagnostic: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_hit_root_policy_total2_136_143.json`

Stress and signature:

- Total2 stress: 8 verified below-rho leaf cases, best 0.648 ops/rho.
- Strict signature: 13 strict positive cases across 2 targets.
- Gate: 6 of 8 materialized surfaces selected.
- Selected targets: 2 surfaces on `22050.cf1@11731`, 4 surfaces on
  `67.a1@9803`.

Sage factorization:

- Requested/factored surfaces: 6/6.
- Resultant factor candidates: 139.
- All factors were quadratic root hyperplanes.
- All 6 selected surfaces had a preserving factor.
- Preserving root-scan below rho: 5/6.
- Preserving direct-root/surface below rho: 6/6.
- Best preserving scan cost: 0.48905109 ops/rho.
- Best preserving surface cost: 0.37956204 ops/rho.
- Full remainders remained above rho; best full remainder was 1.16058394
  ops/rho.

Primary pre-factor policy:

- `prefactor_selected_hit_root_first_target_hash` on 136-143:
  5/6 preserving, 1 false positive, 5/6 scan-below-rho, 5/6
  direct-below-rho.
- Failure:
  `67.a1@9803|67.a1@9803:uniform:256:salt208|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:137:67.a1@9803`
- The failure had pre-factor hit roots `[1296, 7515, 7570]`.
- Root `1296` was a false positive. Root `7515` was preserving.

Unique-leaf tie-break:

- `prefactor_unique_leaf_hit_root_first_target_hash` on 136-143:
  6/6 preserving, 0 false positives, 5/6 scan-below-rho, 6/6
  direct-below-rho.
- Mean evaluated root count: 1.0.
- Mean scan cost: 0.75274939 ops/rho.
- Mean direct cost: 0.65391727 ops/rho.
- Max scan cost: 1.064 ops/rho.
- Max direct cost: 0.928 ops/rho.

Across total2 112-143, the unique-leaf variant is:

- 12/12 preserving.
- 0 false positives.
- 11/12 scan-below-rho.
- 12/12 direct-below-rho.
- Mean evaluated root count: 1.0.
- Mean scan cost: 0.72338686 ops/rho.
- Mean direct cost: 0.62722141 ops/rho.

Interpretation:

The total2 first-fall/root-hyperplane route now has a fresh forward positive
under direct-root accounting: the pre-factor unique-leaf locator finds the
preserving hyperplane on every 136-143 selected surface with zero false
positives and direct cost below rho. It is not a scan-positive promotion
because one surface still costs 1.064 ops/rho under conservative root scan.

The total3/4 negative-control still fails, so the policy must remain
total2-scoped unless a separate public channel discriminator is added.
