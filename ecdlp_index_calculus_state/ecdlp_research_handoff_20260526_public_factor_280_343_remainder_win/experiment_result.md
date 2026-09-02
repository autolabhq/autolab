# Experiment Result

This handoff extends the frozen guarded public-factor portfolio through
transfer block 328-343.  The block keeps the rolling and leave-one-window-out
portfolio audits clean, and it adds the first preserving Sage-factor surface
in this fresh extension chain whose full remainder FFE cost is below rho.

Artifacts added for 328-343:

```text
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_row_328_343_expanded_probe.json
ecdlp_index_calculus_state/low_term_total2_expanded_leaf_rescue_328_343_fixed_summary.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_328_343_fixed_probe.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_quadratic_root_fresh_328_343_fixed_probe.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_rolling_with_280_343_fixed.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_with_280_343_fixed.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260526_public_factor_280_343_remainder_win/false_positive_guard_structural_audit.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260526_public_factor_280_343_remainder_win/full_remainder_sparsity_audit.json
```

Fixed public compact-leaf stress, 328-343:

- Fixed public row-selector policy:
  `fixed_target_cap2_ow0_hw3_lw0_sw0_cw0_aw1`.
- 25 fixed-policy verified leaf cases.
- 23 fixed-policy below-rho cases.
- Target counts among below-rho fixed-policy cases:
  `22050.cf1@11731`: 5, `67.a1@9803`: 18.
- Best fixed-policy case is target `22050.cf1@11731`, transfer 341,
  row keys `salt174` and `salt167`, leaf index 79, at 0.70072993 rho.

Sage/resultant factor audit, 328-343:

- 23 input cases materialized 9 FFE surfaces.
- 9/9 surfaces have nontrivial Sage resultant factorization.
- 9/9 surfaces have a preserving Sage factor candidate.
- 9/9 have preserving factor-surface cost below rho.
- 9/9 have preserving factor-root-scan cost below rho.
- 1/9 has full-remainder FFE cost below rho.
- Best preserving factor-root-scan cost is 0.432 rho.
- Best preserving factor-surface FFE cost is 0.37956204 rho.
- Minimum full-remainder FFE cost is 0.96350365 rho.

First full-remainder FFE below-rho witness:

- Target: `22050.cf1@11731`.
- Transfer: 342.
- Row key: `22050.cf1@11731:uniform:256:salt165`.
- Candidate: `sage_resultant_factor_0`.
- Selected leaf count: 3.
- Surface monomials: 3.
- Full remainder monomials: 79.
- Surface FFE cost: 0.40875912 rho.
- Factor root-scan cost: 0.47445255 rho.
- Full remainder FFE cost: 0.96350365 rho.
- Structural signal: among the 328-343 preserving surfaces, this witness has
  the smallest known-hit-root count and the smallest full-remainder monomial
  count.  The full-remainder costs increase from 0.96350365 rho at 12 known
  hit roots to 4.072 rho at 29 known hit roots in this block.

Direct public quadratic-root audit, 328-343:

- 9 surfaces, 13 public factor-order policies.
- Best single policy is `global_fingerprint_hash`.
- Best policy has 8 recovered surfaces, 8 preserving surfaces, 8 below-rho
  surfaces, and zero quadratic false-positive surfaces.
- Transfer holdout and target holdout summaries remain false-positive-free,
  but they are not complete: each recovers and preserves 8/9 surfaces.
- The rolling-forward split for this fresh block is clean: 2/2 recovered,
  preserving, below rho, and false-positive-free.

Frozen rolling guarded portfolio, 280-343 included:

- 15 rolling test windows.
- Cumulative selected surfaces: 87.
- 87/87 recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Worst window maximum charged cost remains 0.84 rho.
- Latest window is `fresh_328_343_fixed`.
- Latest window selected surfaces: 5.
- Latest window result: 5/5 recovered, preserved, below rho, zero false
  positives, no missed guarded-oracle surfaces.
- Latest window max charged cost: 0.72 rho; mean charged cost: 0.55640292 rho.
- Latest selected policy mix: `low_coeff_max` 2, `global_fingerprint_hash` 1,
  `summax_low_degree_low_coeff_max_target_hash` 1, `target_fingerprint_hash` 1.

Leave-one-window-out robustness, 280-343 included:

- 16 held-out windows.
- 110/110 selected surfaces recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Zero missed guarded-oracle surfaces.
- No failing held-out windows.
- Worst window maximum charged cost remains 0.84 rho.

False-positive guard structural audit:

- Raw 328-343 quadratic-root rows contain zero false-positive rows.
- 65 public-zero rows pass the frozen
  `selected_leaf_count_eq1_and_factor_zero_eq1` guard.
- 0/65 guard-passed rows are false positives.
- Latest guarded portfolio selected rows: 5.
- Latest guarded selected false-positive rows: 0.

Conclusion:

The 328-343 block is a useful promotion, but for a different reason than the
312-327 block.  It does not stress the guard with new raw false positives;
instead it keeps the guarded portfolio perfect through another fresh transfer
block and produces the first preserving full-remainder FFE below-rho witness.
That moves the strongest component evidence from "factor surface/root scan
can be below rho" toward "the full quotient remainder can sometimes be below
rho" without yet claiming a complete ECDLP index-calculus algorithm.

Non-claims:

- This remains a small-field quotient/FFE component, not a complete ECDLP
  index-calculus algorithm.
- Only 1/9 materialized 328-343 surfaces has full-remainder FFE cost below
  rho.
- The direct public quadratic-root audit is 8/9 on this block before the
  guarded portfolio selection.
- Hash policies remain in the public selector set and still need a structural
  explanation before becoming a mechanism claim.
