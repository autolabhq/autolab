# Experiment Result

Artifacts:

- Total3/4 fixed-selector stress:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_88_95_probe.json`
- Total2 fixed-selector stress:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_88_95_probe.json`
- Total3/4 strict signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total3_total4_signature_fixed_selector_88_95.json`
- Total2 strict signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_88_95.json`
- Total3/4 pre-factor gate manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_fixed_selector_88_95.json`
- Total2 pre-factor gate manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_88_95.json`
- Total3/4 Sage factor subset:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_preregistered_88_95.json`
- Total2 Sage factor subset:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_preregistered_88_95.json`
- Total3/4 first-fall audit:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total3_total4_preregistered_88_95.json`
- Total2 first-fall audit:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_preregistered_88_95.json`
- Total3/4 root selector evaluation:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total3_total4_preregistered_88_95.json`
- Total2 root selector evaluation:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total2_preregistered_88_95.json`

Execution note:

The fixed-selector stress probes for 88-95 were run under Sage Python because
the live campaign dependency path still uses `zip(..., strict=True)`, which
macOS Python 3.9 cannot execute. The local total2 wrapper was also patched so
future output describes the requested fresh window instead of hardcoding
80-87.

Total3/4 fresh 88-95:

- strict verifier-backed below-rho cases: 2
- verified over-rho diagnostic cases: 4
- relation target: `67.a1@9803`, transfer 90, row salt 206, leaves
  `[1, 3, 5, 6]`
- best strict signature cost before FFE factor audit: 0.536 ops/rho
- manifest input cases: 2
- materialized surfaces: 1
- pre-factor gate-selected surfaces: 1/1
- Sage-resultant factor candidates: 22
- preserving Sage factor root-scan below rho: 1/1
- min preserving Sage root-scan cost: 0.736 ops/rho
- min preserving Sage surface FFE cost: 0.592 ops/rho
- min full-remainder FFE cost: 2.6 ops/rho
- first-fall factor audit: 22/22 factors are `deg1_mon3_root1`
- all factor counts match known hit roots: true
- best post-factor selector policy: `global_root_hash`
- selector preserving below rho: 1/1
- selector direct-root below rho: 1/1
- selector scan cost: mean 0.928, max 0.928 ops/rho
- selector direct-root cost: mean 0.88, max 0.88 ops/rho
- selector false positives: 0
- vacuous preserving surfaces: 0

Total2 fresh 88-95:

- strict verifier-backed below-rho cases: 1
- verified over-rho diagnostic cases: 0
- relation target: `22050.cf1@11731`, transfer 95, row salts
  `[163, 166]`, leaf `[79]`
- best strict signature cost before FFE factor audit: 0.70072993 ops/rho
- manifest input cases: 1
- materialized surfaces: 2
- pre-factor gate-selected surfaces: 2/2
- Sage-resultant factor candidates: 34
- preserving Sage factor root-scan below rho: 2/2
- min preserving Sage root-scan cost: 0.47445255 ops/rho
- min preserving Sage surface FFE cost: 0.37956204 ops/rho
- min full-remainder FFE cost: 1.13138686 ops/rho
- first-fall factor audit: 34/34 factors are `deg1_mon3_root1`
- all factor counts match known hit roots: true
- best post-factor selector policy: `low_root_norm`
- selector preserving below rho: 2/2
- selector direct-root below rho: 2/2
- selector scan cost: mean 0.51824817, max 0.54014599 ops/rho
- selector direct-root cost: mean 0.46715328, max 0.46715328 ops/rho
- selector false positives: 0
- vacuous preserving surfaces: 0

Interpretation:

This is the first fresh 88-95 run in the local artifacts where strict
below-rho signatures feed a pre-factor manifest, only manifest-selected
surfaces are factored with Sage, and the resulting factors are audited before
selector scoring. The root-hyperplane collapse survived on every measured
fresh surface: 56/56 factors across total3/4 and total2 have the simple
`deg1_mon3_root1` shape.

The positive signal is not yet an algorithmic ECDLP break. The best root
selector policy in this handoff is still chosen as an evaluation result on a
tiny fresh set. The stronger claim available now is that the preregistered
pre-factor gate did not wash out the fresh strict below-rho surfaces, and those
surfaces still reduce to a first-fall root-selection problem with below-rho
root-scan/direct-root costs.
