# Experiment Result

This handoff extends the fresh-window public-factor promotion through
transfer block 312-327.  Unlike 280-295 and 296-311, this block exposes a raw
public-factor false positive before the guard.  The frozen guarded portfolio
still filters it out and preserves the rolling/LOO zero-gap result.

Artifacts added for 312-327:

```text
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_row_312_327_expanded_probe.json
ecdlp_index_calculus_state/low_term_total2_expanded_leaf_rescue_312_327_fixed_summary.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_312_327_fixed_probe.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_quadratic_root_fresh_312_327_fixed_probe.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_rolling_with_280_327_fixed.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_with_280_327_fixed.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260526_public_factor_280_327_guarded_filter/false_positive_guard_structural_audit.json
```

Fixed public compact-leaf stress, 312-327:

- Fixed public row-selector policy:
  `fixed_target_cap2_ow0_hw3_lw0_sw0_cw0_aw1`.
- 37 fixed-policy verified leaf cases.
- 36 fixed-policy below-rho cases.
- Target counts among below-rho fixed-policy cases:
  `22050.cf1@11731`: 25, `67.a1@9803`: 11.
- Best fixed-policy case is target `22050.cf1@11731`, transfer 326,
  row keys `salt168` and `salt173`, leaf index 8, at 0.69343066 rho.

Sage/resultant factor audit, 312-327:

- 36 input cases materialized 10 FFE surfaces.
- 10/10 surfaces have nontrivial Sage resultant factorization.
- 9/10 surfaces have a preserving Sage factor candidate.
- 9/10 have preserving factor-surface cost below rho.
- 9/10 have preserving factor-root-scan cost below rho.
- 0/10 have full remainder cost below rho.
- Best preserving factor-root-scan cost is 0.33576642 rho.
- Best preserving factor-surface FFE cost is 0.34306569 rho.
- Minimum full-remainder FFE cost is 1.05839416 rho.

Direct public quadratic-root audit, 312-327:

- 10 surfaces, 13 public factor-order policies.
- Best single policy is `global_fingerprint_hash`.
- Best policy has 7 recovered surfaces, 6 preserving surfaces, 7 below-rho
  surfaces, and 1 quadratic false-positive surface.
- The raw false-positive example is target `22050.cf1@11731`, transfer 324,
  row `22050.cf1@11731:uniform:256:salt167`, selected by
  `global_fingerprint_hash`; it is below rho at 0.67883212 but misses selected
  root pairs.
- The false-positive row has `selected_leaf_count=3`, so it is rejected by the
  frozen portfolio guard `selected_leaf_count_eq1_and_factor_zero_eq1`.
- Transfer and target holdout summaries are not false-positive-free before the
  guard.

Frozen rolling guarded portfolio, 280-327 included:

- 14 rolling test windows.
- Cumulative selected surfaces: 82.
- 82/82 recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Worst window maximum charged cost remains 0.84 rho.
- Latest window is `fresh_312_327_fixed`.
- Latest window selected surfaces: 5.
- Latest window result: 5/5 recovered, preserved, below rho, zero false
  positives, no missed guarded-oracle surfaces.
- Latest window max charged cost: 0.68 rho; mean charged cost: 0.55192993 rho.
- Latest selected policy mix: `low_constant` 3, `low_degree_then_order` 1,
  `target_fingerprint_hash` 1.

Leave-one-window-out robustness, 280-327 included:

- 15 held-out windows.
- 105/105 selected surfaces recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Zero missed guarded-oracle surfaces.
- No failing held-out windows.
- Worst window maximum charged cost remains 0.84 rho.

False-positive guard structural audit:

- Raw 312-327 quadratic-root rows contain 13 false-positive rows on 1 surface.
- 0/13 raw false-positive rows pass the frozen
  `selected_leaf_count_eq1_and_factor_zero_eq1` guard.
- Latest guarded portfolio selected rows: 5.
- Latest guarded selected false-positive rows: 0.
- Latest guarded selected surface overlap with raw false-positive surfaces: 0.
- Interpretation: the public guard is not merely decorative here; it is
  actively separating a below-rho non-preserving factor choice from the
  preserving guarded portfolio candidates.

Conclusion:

The 312-327 block is the first fresh extension in this chain that produces a
raw quadratic-root false positive, and it is therefore a more meaningful guard
test than another clean block.  The existing public selected-leaf/factor-zero
guard filters the false-positive row and keeps both rolling and LOO portfolios
perfect on selected surfaces.  The strongest current evidence is a robust
guarded public FFE/resultant quotient component with direct quadratic-root
recovery, now validated through three fresh blocks after 264-279.

Non-claims:

- This remains a small-field quotient/FFE component, not a complete ECDLP
  index-calculus algorithm.
- Full-remainder FFE costs remain above rho.
- One materialized 312-327 surface lacks a preserving Sage factor, so the
  unguarded factor bank is not universally preserving.
- Hash policies remain in the public selector set and need a structural
  explanation before becoming a mechanism claim.
