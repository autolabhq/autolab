# Experiment Result

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_target_root_hash_preregistration_120_127.json`

Generated artifacts:

- Total2 fixed-selector stress:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_120_127_probe.json`
- Total3/4 fixed-selector stress:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_120_127_probe.json`
- Total2 strict signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_120_127.json`
- Total3/4 strict signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total3_total4_signature_fixed_selector_120_127.json`
- Total2 pre-factor gate manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_120_127.json`
- Total3/4 pre-factor gate manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_fixed_selector_120_127.json`
- Total2 Sage factor subset:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_preregistered_120_127.json`
- Total3/4 Sage factor subset:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_preregistered_120_127.json`
- Total2 first-fall audit:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_preregistered_120_127.json`
- Total3/4 first-fall audit:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total3_total4_preregistered_120_127.json`
- Total2 root selector:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total2_preregistered_120_127.json`
- Total3/4 root selector:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total3_total4_preregistered_120_127.json`
- Frozen <=87 evaluator on 120-127:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_frozen_root_policy_train_le87_test_120_127.json`

Stress and signature results:

- Total2 stress best policy: `fixed_global_cap3_ow1_hw3_lw0_sw0_cw0_aw0`
- Total2 stress verified leaf cases: 2
- Total2 stress below-rho leaf cases: 1
- Total2 best stress cost: 0.928 ops/rho
- Total2 strict signature cases: 1
- Total2 strict target: `67.a1@9803`
- Total2 strict transfer index: 123
- Total2 strict row salts: `[208, 205]`
- Total2 strict leaf signature: `[2]`
- Total3/4 stress best policy: `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`
- Total3/4 stress verified leaf cases: 8
- Total3/4 stress below-rho leaf cases: 8
- Total3/4 best stress cost: 0.43065693 ops/rho
- Total3/4 strict target: `22050.cf1@11731`
- Total3/4 strict transfer index: 127
- Total3/4 strict row salt: `[174]`
- Total3/4 best leaf signature: `[8, 56, 90]`

Pre-factor gate:

- Total2 input cases: 1
- Total2 materialized surfaces: 2
- Total2 pre-factor gate-selected surfaces: 2
- Total2 selected surfaces:
  - `67.a1@9803|67.a1@9803:uniform:256:salt205|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:123:67.a1@9803`
  - `67.a1@9803|67.a1@9803:uniform:256:salt208|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:123:67.a1@9803`
- Total3/4 input cases: 8
- Total3/4 materialized surfaces: 1
- Total3/4 pre-factor gate-selected surfaces: 1
- Total3/4 selected surface:
  - `22050.cf1@11731|22050.cf1@11731:uniform:256:salt174|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:127:22050.cf1@11731`

Sage factorization:

- Total2 requested surfaces: 2
- Total2 factored surfaces: 2
- Total2 Sage resultant factor candidates: 45
- Total2 preserving Sage factor candidates: 2
- Total2 preserving root-scan below rho: 2
- Total2 min preserving root-scan cost: 0.648 ops/rho
- Total2 min preserving surface FFE cost: 0.496 ops/rho
- Total2 min full-remainder FFE cost: 2.16 ops/rho
- Total3/4 requested surfaces: 1
- Total3/4 factored surfaces: 1
- Total3/4 Sage resultant factor candidates: 18
- Total3/4 preserving Sage factor candidates: 0
- Total3/4 min full-remainder FFE cost: 1.7080292 ops/rho
- Full-remainder below-rho count: 0

First-fall audit:

- Total2 factors: 45
- Total3/4 factors: 18
- Factor shape counts on both channels: `deg1_mon3_root1`
- All factors are quadratic root hyperplanes: true
- All surface factor counts match known hit roots: true
- Preserving factor count: total2 has 2, total3/4 has 0

Preregistered primary policy result across all selected surfaces:

- Policy: `target_root_hash`
- Holdout surfaces: 3
- Preserving count: 2/3
- False positives: 1
- Scan below rho: 2/3
- Direct-root below rho: 2/3
- Mean scan cost: 0.99454988 ops/rho
- Max scan cost: 1.50364964 ops/rho
- Mean direct-root cost: 0.91815085 ops/rho
- Max direct-root cost: 1.47445255 ops/rho

Target-hash total2-only diagnostic:

- Total2 selected surfaces: 2
- Total2 preserving count: 2/2
- Total2 false positives: 0
- Total2 scan below rho: 2/2
- Total2 direct-root below rho: 2/2
- Total2 chosen root on both surfaces: 161
- Total2 scan costs: 0.672 and 0.808 ops/rho
- Total2 direct-root costs: 0.592 and 0.688 ops/rho

All-channel failure:

- Failed surface:
  `22050.cf1@11731|22050.cf1@11731:uniform:256:salt174|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:127:22050.cf1@11731`
- Channel: total3/4
- Chosen factor: `sage_resultant_factor_5`
- Chosen root: 4059
- Chosen factor preserves selected root pairs: false
- Scan cost: 1.50364964 ops/rho
- Direct-root cost: 1.47445255 ops/rho

Declared controls:

- `summax_sage_low_constant_target_hash`: 2/3 preserving, 1 false positive,
  2/3 scan-below-rho.
- `global_root_hash`: 2/3 preserving, 1 false positive,
  0/3 scan-below-rho.
- `residue16_then_global_hash_cap1`: 2/3 preserving, 1 false positive,
  0/3 scan-below-rho.
- `learned_target_zero_residue16_prior`: 2/3 preserving, 1 false positive,
  0/3 scan-below-rho.

Diagnostic, not claim:

- Total2 posthoc best static policy: `summax_low_root_norm_target_hash`
- Total2 posthoc best: 2/2 preserving, 0 false positives, 2/2 scan below rho,
  max scan cost 0.712 ops/rho.

Interpretation:

The preregistered all-channel `target_root_hash` claim failed. The old
pre-factor gate admitted a total3/4 surface that factored into root
hyperplanes but had no preserving factor candidate. That one surface produced
the false positive and breaks both scan and direct-root promotion.

The narrower total2 signal is still strong: both gate-selected 67.a1 surfaces
preserved the same root 161, and `target_root_hash` scored them below rho by
scan and direct-root accounting. The next move should not be another
all-channel promotion; it should either preregister a total2-only factor route
or add a public channel gate that can reject this total3/4 non-preserving
failure mode before factorization.

This remains below an end-to-end ECDLP speedup. The measured component is a
public FFE first-fall/root-hyperplane route on total2-selected surfaces.
