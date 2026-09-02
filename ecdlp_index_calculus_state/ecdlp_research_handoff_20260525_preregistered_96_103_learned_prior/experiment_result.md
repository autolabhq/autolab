# Experiment Result

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_learned_root_prior_preregistration_96_103.json`

Generated artifacts:

- Total2 fixed-selector stress:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_96_103_probe.json`
- Total3/4 fixed-selector stress:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_96_103_probe.json`
- Total2 strict signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_96_103.json`
- Total3/4 strict signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total3_total4_signature_fixed_selector_96_103.json`
- Total2 pre-factor gate manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_96_103.json`
- Total3/4 pre-factor gate manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_fixed_selector_96_103.json`
- Total2 Sage factor subset:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_preregistered_96_103.json`
- Total2 first-fall audit:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_preregistered_96_103.json`
- Total2 root selector:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total2_preregistered_96_103.json`
- Frozen <=87 evaluator on 96-103:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_frozen_root_policy_train_le87_test_96_103.json`

Stress and signature results:

- Total2 stress best policy:
  `fixed_global_cap3_ow1_hw3_lw0_sw0_cw0_aw0`
- Total2 stress verified leaf cases: 3
- Total2 stress below-rho leaf cases: 3
- Total2 best stress cost: 0.928 ops/rho
- Total2 strict signature cases: 3
- Total2 strict target: `67.a1@9803`
- Total2 strict transfer index: 102
- Total2 strict row salts: `[201, 204]`
- Total2 strict leaf signature: `[6]`
- Total2 strict surfaces: 2
- Total3/4 verified cases: 12
- Total3/4 below-rho strict cases: 0
- Total3/4 best verified-over-rho cost: 1.336 ops/rho

Pre-factor gate:

- Total2 input cases: 3
- Total2 materialized surfaces: 2
- Total2 pre-factor gate-selected surfaces: 2/2
- Total3/4 input cases: 0
- Total3/4 gate-selected surfaces: 0

Sage factorization:

- Total2 requested surfaces: 2
- Total2 factored surfaces: 2
- Missing requested surfaces: 0
- Sage resultant factor candidates: 36
- Surfaces with nontrivial resultant factorization: 2/2
- Surfaces with preserving Sage factor candidate: 2/2
- Preserving root-scan below rho: 2/2
- Min preserving root-scan cost: 0.608 ops/rho
- Min preserving surface FFE cost: 0.496 ops/rho
- Min full-remainder FFE cost: 1.44 ops/rho
- Full-remainder below-rho count: 0

First-fall audit:

- Surfaces: 2
- Factors: 36
- Factor shape counts: `deg1_mon3_root1`: 36
- All factors are quadratic root hyperplanes: true
- All surface factor counts match known hit roots: true
- Preserving factor count: 2
- Top repeated target root: 8142, present on both surfaces

Preregistered learned policy result:

- Preregistered policy: `learned_global_zero_root_prior`
- External train rows: 30 gated <=87 surfaces
- 96-103 preserving count: 2/2
- 96-103 false positives: 0
- 96-103 below rho by scan cost: 1/2
- 96-103 below rho by direct-root cost: 1/2
- Mean scan cost: 1.064 ops/rho
- Max scan cost: 1.16 ops/rho
- Mean direct-root cost: 1.0 ops/rho
- Max direct-root cost: 1.072 ops/rho

The learned-global miss:

- Surface:
  `67.a1@9803|67.a1@9803:uniform:256:salt204|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:102:67.a1@9803`
- Chosen factor: `sage_resultant_factor_19`
- Chosen root: 8142
- Chosen factor preserves the selected root pair: true
- Scan cost: 1.16 ops/rho
- Direct-root cost: 1.072 ops/rho

Diagnostic policies, not claims:

- Post hoc best static policy: `low_root_norm`
- Post hoc `low_root_norm`: 2/2 preserving, 0 false positives, 2/2 below rho
- Post hoc `low_root_norm` mean scan cost: 0.764 ops/rho
- Post hoc `low_root_norm` max scan cost: 0.8 ops/rho
- Post hoc `low_root_norm` mean direct-root cost: 0.7 ops/rho
- Post hoc `low_root_norm` max direct-root cost: 0.712 ops/rho
- External trained `learned_target_zero_residue16_prior`: 2/2 below rho,
  max scan cost 0.92 ops/rho

Interpretation:

The 96-103 run is a mixed but useful result. The FFE/summation-polynomial
surface generation and preregistered pre-factor gate replicated on a new
window: strict total2 signatures appeared, the gate selected both surfaces,
Sage factors again collapsed entirely to root hyperplanes, and preserving
root-scan candidates were below rho.

The preregistered `learned_global_zero_root_prior` did not meet the promotion
condition because one of the two preserving choices was too late in the root
order. The failure is cost-ordering, not algebraic correctness: it still picked
the preserving root on both surfaces with no false positives.

The next policy problem is now sharper: the repeated root 8142 is easy for
`low_root_norm` and `learned_target_zero_residue16_prior` on this held-out
window, but those are diagnostic until declared before the next unseen window.
