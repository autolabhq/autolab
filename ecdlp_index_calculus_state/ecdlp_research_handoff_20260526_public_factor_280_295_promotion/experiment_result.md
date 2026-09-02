# Experiment Result

Artifacts:

```text
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_row_280_295_expanded_probe.json
ecdlp_index_calculus_state/low_term_total2_expanded_leaf_rescue_280_295_fixed_summary.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_280_295_fixed_probe.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_quadratic_root_fresh_280_295_fixed_probe.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_rolling_with_280_295_fixed.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_with_280_295_fixed.json
```

Runtime notes:

- The default `/usr/bin/python3` lacked `sympy`.
- `/Users/adamburan/.cache/vllm-venvs/vllm-312/bin/python` provided `sympy`
  for public-factor quadratic-root and portfolio probes.
- Sage was available as `/usr/local/bin/sage`, reporting SageMath 10.9.
- Large source inputs were read from `/Volumes/Volume/autolab/ecdlp_index_calculus_state`;
  new outputs were persisted in the writable local worktree.

Public compact-leaf stress for 280-295:

- Fixed public row-selector policy:
  `fixed_target_cap2_ow0_hw3_lw0_sw0_cw0_aw1`.
- Stress transfer indices: 280-295.
- Fixed-policy summary has 27 verified cases and 26 below-rho cases.
- Best fixed-policy case is target `22050.cf1@11731`, transfer 292,
  row keys `salt167` and `salt171`, leaf index 8, at 0.69343066 rho.
- Target counts among below-rho fixed-policy cases:
  `22050.cf1@11731`: 18, `67.a1@9803`: 8.

Sage/resultant factor audit:

- 26 input cases materialized 11 FFE surfaces.
- 11/11 surfaces have a preserving nontrivial Sage resultant factor candidate.
- 11/11 have preserving factor-surface cost below rho.
- 11/11 have preserving factor-root-scan cost below rho.
- 0/11 have full remainder cost below rho.
- Best preserving factor-root-scan cost is 0.33576642 rho.
- Best preserving factor-surface FFE cost is 0.34306569 rho.
- Minimum full-remainder FFE cost is 1.13138686 rho.

Direct public quadratic-root audit:

- 11 surfaces, 13 public factor-order policies.
- Best policy is `low_constant`.
- Best-policy recovered/preserved/below-rho count is 8/11, with zero false
  positives and maximum charged cost 0.72992701 rho.
- Transfer and target holdout summaries are false-positive-free but not all
  below rho, so the standalone per-window selector is not promoted by itself.
- Rolling-forward internal split is clean: 2/2 below rho, preserving, recovered,
  and false-positive-free.

Frozen rolling guarded portfolio with the new window:

- 12 rolling test windows.
- Cumulative selected surfaces: 72.
- 72/72 recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Worst window maximum charged cost remains 0.84 rho.
- Latest window is `fresh_280_295_fixed`.
- Latest window selected surfaces: 8.
- Latest window result: 8/8 recovered, preserved, below rho, zero false
  positives, no missed guarded-oracle surfaces.
- Latest window max charged cost: 0.64 rho; mean charged cost: 0.54897811 rho.
- Latest selected policy mix: `global_fingerprint_hash` 3,
  `target_fingerprint_hash` 3, `low_constant` 1, `low_degree_then_order` 1.

Leave-one-window-out robustness after adding 280-295:

- 13 held-out windows.
- 95/95 selected surfaces recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Zero missed guarded-oracle surfaces.
- No failing held-out windows.
- Worst window maximum charged cost remains 0.84 rho.

Conclusion:

The public-factor portfolio survived a genuinely new materialized 280-295
window with the frozen public guard and policy eligibility rule.  The strongest
claim is now a robust small-field FFE/resultant quotient component: public
compact-leaf selection yields Sage factors, direct quadratic roots remove the
known-hit-root scan, and a public guarded portfolio matches the guarded oracle
on 95/95 LOO surfaces.

Non-claims:

- This is not yet a complete ECDLP index-calculus algorithm.
- Full-remainder FFE cost remains above rho on the 280-295 surfaces.
- Hash policies are still part of the frozen public selector mix, so their
  structural meaning should be audited before making a mechanism claim.
