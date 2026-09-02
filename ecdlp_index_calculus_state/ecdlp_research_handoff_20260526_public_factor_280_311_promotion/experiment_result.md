# Experiment Result

This handoff supersedes the 280-295 waypoint by adding a second fresh transfer
block, 296-311, under the same frozen public-factor portfolio.

Artifacts added for 296-311:

```text
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_row_296_311_expanded_probe.json
ecdlp_index_calculus_state/low_term_total2_expanded_leaf_rescue_296_311_fixed_summary.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_296_311_fixed_probe.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_quadratic_root_fresh_296_311_fixed_probe.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_rolling_with_280_311_fixed.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_with_280_311_fixed.json
```

Fixed public compact-leaf stress, 296-311:

- Fixed public row-selector policy:
  `fixed_target_cap2_ow0_hw3_lw0_sw0_cw0_aw1`.
- 31 fixed-policy verified leaf cases.
- 12 fixed-policy below-rho cases.
- Target counts among below-rho fixed-policy cases:
  `22050.cf1@11731`: 9, `67.a1@9803`: 3.
- Best fixed-policy case is target `22050.cf1@11731`, transfer 309,
  row keys `salt169` and `salt177`, leaf index 79, at 0.69343066 rho.

Sage/resultant factor audit, 296-311:

- 12 input cases materialized 12 FFE surfaces.
- 12/12 surfaces have a preserving nontrivial Sage resultant factor candidate.
- 12/12 have preserving factor-surface cost below rho.
- 12/12 have preserving factor-root-scan cost below rho.
- 0/12 have full remainder cost below rho.
- Best preserving factor-root-scan cost is 0.33576642 rho.
- Best preserving factor-surface FFE cost is 0.34306569 rho.
- Minimum full-remainder FFE cost is 1.35766423 rho.

Direct public quadratic-root audit, 296-311:

- 12 surfaces, 13 public factor-order policies.
- Best single policy is `low_degree_then_order`.
- Best policy has 10 recovered/preserving surfaces, 8 below-rho surfaces, and
  zero false positives.
- Best-policy maximum charged cost is 1.088 rho, so the per-window single-policy
  selector is not sufficient by itself.
- Transfer, target, and rolling-forward holdout summaries are
  false-positive-free but not all below rho.

Frozen rolling guarded portfolio, 280-311 included:

- 13 rolling test windows.
- Cumulative selected surfaces: 77.
- 77/77 recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Worst window maximum charged cost remains 0.84 rho.
- Latest window is `fresh_296_311_fixed`.
- Latest window selected surfaces: 5.
- Latest window result: 5/5 recovered, preserved, below rho, zero false
  positives, no missed guarded-oracle surfaces.
- Latest window max charged cost: 0.552 rho; mean charged cost: 0.52007007 rho.
- Latest selected policy mix: `target_fingerprint_hash` 3,
  `low_coeff_sum` 1, `low_degree_then_order` 1.

Leave-one-window-out robustness, 280-311 included:

- 14 held-out windows.
- 100/100 selected surfaces recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Zero missed guarded-oracle surfaces.
- No failing held-out windows.
- Worst window maximum charged cost remains 0.84 rho.

Conclusion:

The frozen guarded public-factor portfolio now survives two new materialized
fresh blocks, 280-295 and 296-311.  The second block is especially useful
because the best per-window public policy is mixed and over-rho on some
surfaces, but the prior-calibrated guarded portfolio still selects 5/5
below-rho preserving surfaces and keeps the global LOO audit perfect at
100/100.

Non-claims:

- This is still a small-field quotient/FFE component, not a full ECDLP
  algorithm.
- Full-remainder FFE costs remain above rho.
- Hash policies remain in the frozen public selector set and need structural
  explanation before they become a mechanism claim.
