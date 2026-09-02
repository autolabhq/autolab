# Experiment Result

## Fresh 152-159 Run

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_single_hit_root_policy_preregistration_152_159.json`

Artifacts:

- Stress: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_152_159_probe.json`
- Signature: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_152_159.json`
- Gate: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_152_159.json`
- Strict Sage factors: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_single_hit_root_preregistered_152_159.json`
- Strict factor audit: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_single_hit_root_preregistered_152_159.json`
- Strict policy eval: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total2_152_159.json`
- Broad Sage factors: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_all_prefactor_preregistered_152_159.json`
- Broad factor audit: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_all_prefactor_preregistered_152_159.json`
- Broad unique-leaf control: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_hit_root_policy_total2_152_159.json`
- Broad selected-hit-root control: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_hit_root_policy_total2_152_159.json`

Stress and signature:

- Total2 stress found 5 below-rho cases across 2 targets.
- Targets: `22050.cf1@11731` at transfer 159 and `67.a1@9803` at transfer
  155.
- Best individual observed cost: 0.39416058 ops/rho on `22050.cf1@11731`.
- Gate manifest had 4 broad pre-factor selected surfaces.
- Strict single-hit-root gate kept 3 surfaces and dropped the one multi-root
  `22050.cf1@11731` surface.

Strict single-hit-root result:

- Kept surfaces: 3.
- Target retained: `67.a1@9803`.
- Preserving: 3/3.
- False positives: 0.
- Scan below rho: 3/3.
- Direct below rho: 3/3.
- Mean evaluated root count: 1.0.
- Mean selector eval ops: 3.0.
- Scan cost: min 0.688, mean 0.72266667, max 0.768 ops/rho.
- Direct cost: 0.608 ops/rho on every retained surface.

Sage factorization for the strict gate:

- Requested/factored strict surfaces: 3/3.
- Resultant factor candidates: 73.
- Preserving factor surfaces: 3/3.
- All factors were quadratic root hyperplanes.
- Preserving factor root-scan below rho: 3/3.
- Preserving direct/surface below rho: 3/3.
- Full remainders remained above rho.

## Broad Gate Control

The broad pre-factor gate included 4 surfaces:

- `22050.cf1@11731|22050.cf1@11731:uniform:256:salt174|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:159:22050.cf1@11731`
- `67.a1@9803|67.a1@9803:uniform:256:salt201|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:155:67.a1@9803`
- `67.a1@9803|67.a1@9803:uniform:256:salt203|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:155:67.a1@9803`
- `67.a1@9803|67.a1@9803:uniform:256:salt205|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:155:67.a1@9803`

Broad Sage factorization:

- Requested/factored broad surfaces: 4/4.
- Resultant factor candidates: 89.
- All 89 factors were quadratic root hyperplanes.
- Preserving factor surfaces: 3/4.
- The 3 preserving surfaces were the retained `67.a1@9803` rows.
- The multi-root `22050.cf1@11731` row had 16 root-hyperplane factors and 0
  preserving factors.

Broad policy controls:

- `prefactor_unique_leaf_hit_root_first_target_hash`: 3/4 preserving, 1 false
  positive, 3/4 scan below rho, 3/4 direct below rho.
- `prefactor_selected_hit_root_first_target_hash`: 3/4 preserving, 1 false
  positive, 3/4 scan below rho, 3/4 direct below rho.
- Both controls recovered a public zero on the multi-root `22050.cf1@11731`
  surface, but that zero did not preserve the selected root pair.

Interpretation:

This is the first fresh preregistered positive for the stricter
single-hit-root total2 component. The broad gate remains a controlled negative:
it sees one extra public-zero surface, but that surface has multiple pre-factor
hit roots and no preserving factor. The result supports the rule that
multi-hit-root surfaces need a separate discriminator instead of being folded
into the promoted single-root route.
