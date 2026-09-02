# Experiment Result

This handoff extends the frozen guarded public-factor portfolio through
transfer block 360-375.  The block is a boundary result: the guarded
quadratic-root portfolio remains clean, but the repeated full-remainder FFE
below-rho pattern from 328-343 and 344-359 does not repeat.

Artifacts added for 360-375:

```text
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_row_360_375_expanded_probe.json
ecdlp_index_calculus_state/low_term_total2_expanded_leaf_rescue_360_375_fixed_summary.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_360_375_fixed_probe.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_quadratic_root_fresh_360_375_fixed_probe.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_rolling_with_280_375_fixed.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_with_280_375_fixed.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260526_public_factor_280_375_guarded_remainder_boundary/false_positive_guard_structural_audit.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260526_public_factor_280_375_guarded_remainder_boundary/full_remainder_boundary_audit.json
```

Fixed public compact-leaf stress, 360-375:

- Fixed public row-selector policy:
  `fixed_target_cap2_ow0_hw3_lw0_sw0_cw0_aw1`.
- 16 fixed-policy verified leaf cases.
- 8 fixed-policy below-rho cases.
- Target counts among below-rho fixed-policy cases:
  `22050.cf1@11731`: 2, `67.a1@9803`: 6.
- Best fixed-policy case is target `22050.cf1@11731`, transfer 363,
  row keys `salt173` and `salt165`, selected leaf count 5, at 0.75182482 rho.

Sage/resultant factor audit, 360-375:

- 8 input cases materialized 7 FFE surfaces.
- 7/7 surfaces have nontrivial Sage resultant factorization.
- 6/7 surfaces have a preserving Sage factor candidate.
- 6/7 have preserving factor-surface cost below rho.
- 6/7 have preserving factor-root-scan cost below rho.
- 0/7 have full-remainder FFE cost below rho.
- Best preserving factor-root-scan cost is 0.34306569 rho.
- Best preserving factor-surface FFE cost is 0.35766423 rho.
- Minimum full-remainder FFE cost is 1.208 rho.

Full-remainder boundary:

- Best full-remainder surface is target `67.a1@9803`, transfer 364,
  row `67.a1@9803:uniform:256:salt207`.
- Candidate: `sage_resultant_factor_0`.
- Selected leaf count: 1.
- Known hit-root count: 13.
- Surface monomials: 3.
- Full remainder monomials: 92.
- Surface FFE cost: 0.496 rho.
- Factor root-scan cost: 0.592 rho.
- Full remainder FFE cost: 1.208 rho.
- Interpretation: this block still has cheap factor-surface/root-scan
  quotients, but it misses the 79-monomial full-remainder profile seen on
  `22050.cf1@11731:uniform:256:salt165` in the prior two blocks.

Direct public quadratic-root audit, 360-375:

- 7 surfaces, 13 public factor-order policies.
- Best single policy is `low_constant`.
- Best policy has 4 recovered surfaces, 3 preserving surfaces, 4 below-rho
  surfaces, and 1 quadratic false-positive surface.
- Raw false positives: 13 policy rows on 1 surface.
- The false-positive surface is target `22050.cf1@11731`, transfer 363,
  row `22050.cf1@11731:uniform:256:salt165`, candidate
  `sage_resultant_factor_4`; it misses selected root pair `[8, 4745]`.
- 0/13 raw false-positive rows pass the frozen portfolio guard
  `selected_leaf_count_eq1_and_factor_zero_eq1`.

Frozen rolling guarded portfolio, 280-375 included:

- 17 rolling test windows.
- Cumulative selected surfaces: 99.
- 99/99 recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Worst window maximum charged cost remains 0.84 rho.
- Latest window is `fresh_360_375_fixed`.
- Latest window selected surfaces: 3.
- Latest window result: 3/3 recovered, preserved, below rho, zero false
  positives, no missed guarded-oracle surfaces.
- Latest window max charged cost: 0.68 rho; mean charged cost: 0.584 rho.
- Latest selected policy mix: `low_coeff_max` 2, `low_constant` 1.

Leave-one-window-out robustness, 280-375 included:

- 18 held-out windows.
- 122/122 selected surfaces recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Zero missed guarded-oracle surfaces.
- No failing held-out windows.
- Worst window maximum charged cost remains 0.84 rho.

Conclusion:

The 360-375 block keeps the guarded public-factor quotient/root-scan component
alive, but demotes the full-remainder repetition hypothesis.  The current
best interpretation is that the full-remainder win is tied to a narrower
row-envelope/salt-specific sparsity condition, not to every fresh block that
the guarded portfolio can solve below rho.

Non-claims:

- This remains a small-field quotient/FFE component, not a complete ECDLP
  index-calculus algorithm.
- 360-375 has zero full-remainder FFE below-rho surfaces.
- One 360-375 surface lacks a preserving Sage factor.
- Direct public quadratic-root selection is still false-positive-prone before
  the frozen public guard.
- The latest guarded selected-surface count drops from 9 to 3, so the guard
  still works but the fresh evidence density weakened.
