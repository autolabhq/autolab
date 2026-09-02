# Experiment Result

Artifacts:

```text
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_rolling_baseline.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_baseline.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_min2_support_control.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_exclude_hash_control.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_train_worst075_control.json
```

Commands used the live script:

```text
/Volumes/Volume/autolab/tasks/ecdlp_index_calculus/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_guarded_portfolio_probe.py
/Volumes/Volume/autolab/tasks/ecdlp_index_calculus/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_guarded_portfolio_loo_probe.py
```

Baseline rolling portfolio:

- 11 rolling test windows.
- 64/64 selected surfaces recovered and preserved by the quadratic-root factor
  route.
- 64/64 are below rho.
- Zero quadratic false-positive surfaces.
- Worst window maximum charged cost is 0.84 rho.
- Latest window is `fresh_264_279_fixed`, with 16/16 below rho and worst cost
  0.672 rho.

Baseline leave-one-window-out:

- 12 held-out windows.
- 87/87 selected surfaces recovered and preserved.
- 87/87 are below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Worst window maximum charged cost is 0.84 rho.
- Selected policy mix: `global_fingerprint_hash` 24,
  `target_fingerprint_hash` 15, `low_degree_then_order` 20,
  `low_coeff_max` 13, `low_constant` 5, `low_coeff_sum` 2, and four
  mixed target-hash coefficient/degree policies totaling 8.

Minimum-support control:

- Requiring `min_surfaces=2` for policy eligibility preserves the full result:
  87/87 below rho, zero false positives, and zero oracle gap.
- This rules out the easiest overfit explanation where single-surface policy
  support is carrying the result.

Hash-family ablation:

- Excluding hash policies still preserves all 87 selected surfaces and has zero
  false positives, but only 86/87 are below rho.
- The single failing held-out window is `fresh_264_279_fixed`.
- The over-rho selected row is target `67.a1@9803`, transfer 268,
  `row_key=67.a1@9803:uniform:256:salt208`, selected by `low_coeff_sum` at
  1.032 rho.
- The guarded oracle for that same window chooses hash-family policies for the
  corresponding 67.a1@9803 transfer-268 rows at 0.624 and 0.672 rho.
- Interpretation: hash policies are not cosmetic; they are needed to keep the
  latest 67.a1@9803 quotient surface below rho under public selection.

Strict train-worst control:

- Setting `max_train_worst_ratio=0.75` is too restrictive.
- It leaves only 16 selected surfaces, all below rho with zero false positives,
  but misses 71 guarded-oracle below-rho surfaces and fails 11/12 held-out
  windows.
- This is a recall collapse, not a precision gain.

Conclusion:

The public-factor portfolio is stronger than a one-off oracle slice replay:
all-label LOO and the `min_surfaces=2` robustness control both match the guarded
oracle on 87/87 below-rho surfaces.  The mechanism does depend on hash-order
factor policies for at least one latest-window 67.a1@9803 case, so the next
promotion step should keep hash policies eligible and test a new materialized
window rather than tightening worst-ratio eligibility.

Non-claims:

- This is still a small-field quotient/FFE component result, not a complete
  asymptotic ECDLP index-calculus algorithm.
- The mounted live state is read-only from this sandbox; outputs were persisted
  in the local writable worktree.
- The result does not replace a fresh future-window promotion test.
