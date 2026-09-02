# Experiment Result

Artifacts:

- Manifest probe:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_preregistered_gate_manifest_probe.py`
- 80-87 manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_verified_over_rho_80_87.json`
- 72-79 calibration manifest:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_all_public_leaf_plus_fresh_72_79.json`
- 80-87 selector joined for evaluation:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total3_total4_verified_over_rho_80_87.json`
- 72-79 selector joined for evaluation:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_all_public_leaf_plus_fresh_72_79.json`

80-87 control:

- input cases: 8
- verified cases after rematerialization: 8
- materialized surfaces: 6
- pre-factor non-vacuous surfaces: 3
- pre-factor selected-hit-root proxy surfaces: 4
- pre-factor gate-selected surfaces: 3
- joined selector policy: `low_root_norm`
- joined gate public-zero-capable surfaces: 3/3
- joined gate public-zero recovered: 3/3
- joined gate chosen preserving: 3/3
- joined gate false positives: 0
- joined gate conservative scan below rho: 3/3
- joined gate direct-root below rho: 3/3
- joined gate mean scan ops/rho: 0.88533333
- joined gate max scan ops/rho: 0.96
- joined gate mean direct-root ops/rho: 0.816
- joined gate max direct-root ops/rho: 0.888

72-79 calibration:

- input cases: 45
- verified cases after rematerialization: 45
- materialized surfaces: 28
- pre-factor non-vacuous surfaces: 27
- pre-factor selected-hit-root proxy surfaces: 28
- pre-factor gate-selected surfaces: 27
- joined selector policy: `summax_sage_low_constant_target_hash`
- joined gate public-zero-capable surfaces: 27/27
- joined gate public-zero recovered: 27/27
- joined gate chosen preserving: 27/27
- joined gate false positives: 0
- joined gate conservative scan below rho: 27/27
- joined gate direct-root below rho: 27/27
- joined gate mean scan ops/rho: 0.79625196
- joined gate max scan ops/rho: 0.968
- joined gate mean direct-root ops/rho: 0.72735118
- joined gate max direct-root ops/rho: 0.848

Interpretation:

This converts the previous ex post `nonvacuous_public_zero_capable` diagnostic
into a pre-factor manifest rule that can be frozen before Sage factorization.
On both controls, the preregistered proxy gate selects exactly the non-vacuous
surface slice that later evaluates cleanly under the fixed selector.

The 80-87 result is especially useful: the all-surface control is still only
3/6 below rho by scan cost, but the pre-factor gate selects 3 surfaces and all
3 later prove public-zero-capable, preserving, and below rho under both scan
and direct-root accounting.  This is still not an ECDLP break because the bank
itself is a diagnostic verified-over-rho bank, not a fresh below-rho generator.

Conclusion:

The next run can preregister `pre_factor_nonvacuous_hit_root_proxy` before root
policy scoring.  The remaining proof obligation is fresh 80+ generation:
materialize candidate surfaces, apply this pre-factor gate, then run the frozen
root policy and compare scan/direct-root costs to Pollard-rho.
