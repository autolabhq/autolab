# Experiment Result

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_hybrid_root_policy_preregistration_112_119.json`

Code change:

- Added hybrid policies `residue16_then_global_hash_cap1` through
  `residue16_then_global_hash_cap4` to
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_first_fall_root_hyperplane_selector_probe.py`.
- Added hybrid-policy support to the frozen external-train evaluator:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_frozen_root_policy_evaluator.py`.

Generated artifacts:

- Total2 fixed-selector stress:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_112_119_probe.json`
- Total3/4 fixed-selector stress:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_112_119_probe.json`
- Total2 strict signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_112_119.json`
- Total3/4 strict signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total3_total4_signature_fixed_selector_112_119.json`
- Total2 pre-factor gate manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_112_119.json`
- Total3/4 pre-factor gate manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_fixed_selector_112_119.json`
- Total2 Sage factor subset:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_preregistered_112_119.json`
- Total2 first-fall audit:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_preregistered_112_119.json`
- Total2 root selector:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total2_preregistered_112_119.json`
- Frozen <=87 evaluator on 112-119:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_frozen_root_policy_train_le87_test_112_119.json`

Stress and signature results:

- Total2 stress best policy: `fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0`
- Total2 stress verified leaf cases: 5
- Total2 stress below-rho leaf cases: 5
- Total2 best stress cost: 0.73722628 ops/rho
- Total2 strict signature cases: 9
- Total2 strict targets: `22050.cf1@11731`, `67.a1@9803`
- Total2 strict transfer indices: 115, 116, 117, 118
- Total2 strict materialized surfaces: 7
- Total3/4 stress best policy: `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`
- Total3/4 stress verified leaf cases: 4
- Total3/4 stress below-rho leaf cases: 4
- Total3/4 best stress cost: 0.43065693 ops/rho
- Total3/4 strict signature cases: 4
- Total3/4 strict materialized surfaces: 1

Pre-factor gate:

- Total2 input cases: 9
- Total2 materialized surfaces: 7
- Total2 pre-factor gate-selected surfaces: 3
- Total2 selected surfaces:
  - `22050.cf1@11731|22050.cf1@11731:uniform:256:salt163|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:116:22050.cf1@11731`
  - `67.a1@9803|67.a1@9803:uniform:256:salt206|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:118:67.a1@9803`
  - `67.a1@9803|67.a1@9803:uniform:256:salt208|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:117:67.a1@9803`
- Total3/4 input cases: 4
- Total3/4 materialized surfaces: 1
- Total3/4 pre-factor gate-selected surfaces: 0

Sage factorization:

- Requested surfaces: 3
- Factored surfaces: 3
- Missing requested surfaces: 0
- Sage resultant factor candidates: 61
- Surfaces with nontrivial resultant factorization: 3/3
- Surfaces with preserving Sage factor candidate: 3/3
- Preserving root-scan below rho: 3/3
- Preserving surface FFE below rho: 3/3
- Min preserving root-scan cost: 0.51824818 ops/rho
- Min preserving surface FFE cost: 0.37956204 ops/rho
- Min full-remainder FFE cost: 1.896 ops/rho
- Full-remainder below-rho count: 0

First-fall audit:

- Factors: 61
- Factor shape counts: `deg1_mon3_root1`: 61
- All factors are quadratic root hyperplanes: true
- All surface factor counts match known hit roots: true
- Preserving factor count: 3

Preregistered primary policy result:

- Policy: `residue16_then_global_hash_cap1`
- External train rows: 30 gated <=87 surfaces
- 112-119 preserving count: 3/3
- 112-119 false positives: 0
- 112-119 below rho by scan cost: 2/3
- 112-119 below rho by direct-root cost: 3/3
- Mean scan cost: 0.81471533 ops/rho
- Max scan cost: 1.016 ops/rho
- Mean direct-root cost: 0.73438443 ops/rho
- Max direct-root cost: 0.952 ops/rho

The scan-only miss:

- Surface:
  `67.a1@9803|67.a1@9803:uniform:256:salt206|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:118:67.a1@9803`
- Chosen factor: `sage_resultant_factor_0`
- Chosen root: 161
- Chosen factor preserves the selected root pair: true
- Evaluated root count: 14
- Scan cost: 1.016 ops/rho
- Direct-root cost: 0.952 ops/rho

Declared controls:

- `summax_sage_low_constant_target_hash`: 3/3 preserving, 0 false positives,
  2/3 scan-below-rho, 3/3 direct-below-rho, max scan 1.032.
- `global_root_hash`: 3/3 preserving, 0 false positives, 2/3 scan-below-rho,
  3/3 direct-below-rho, max scan 1.016.
- `learned_target_zero_residue16_prior`: 3/3 preserving, 0 false positives,
  1/3 scan-below-rho, 2/3 direct-below-rho, max scan 1.104.

Diagnostic, not claim:

- Post hoc best static policy: `target_root_hash`
- Post hoc `target_root_hash`: 3/3 preserving, 0 false positives, 3/3
  scan-below-rho, 3/3 direct-below-rho, max scan 0.912.

Interpretation:

The 112-119 run is a forward positive for the cap1 hybrid under the declared
promotion boundary because all selected surfaces are preserving, false-positive
free, and below rho under direct-root accounting. The scan-only metric is still
not uniformly below rho: one selected 67.a1 surface lands at 1.016 ops/rho.

The algebraic story strengthened: strict total2 certificates appeared on two
targets, the pre-factor gate selected three surfaces, Sage factorization again
collapsed every measured resultant factor to a root hyperplane, and every
selected surface had a preserving below-rho root-scan candidate. Total3/4 also
had strict signatures, but the preregistered pre-factor gate selected none of
those surfaces, so no total3/4 factor claim is made for this window.

This is still not an end-to-end ECDLP speedup. The current claim is narrower:
a preregistered FFE first-fall/root-hyperplane policy now has a fresh-window
direct-root below-rho certificate on all gate-selected surfaces.
