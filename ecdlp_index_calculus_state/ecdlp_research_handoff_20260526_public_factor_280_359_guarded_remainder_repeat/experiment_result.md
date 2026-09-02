# Experiment Result

This handoff extends the frozen guarded public-factor portfolio through
transfer block 344-359.  The block is a stronger follow-up than 328-343 in two
ways: it repeats the full-remainder FFE below-rho witness and it reintroduces
raw quadratic-root false positives that the frozen public guard filters out.

Artifacts added for 344-359:

```text
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_row_344_359_expanded_probe.json
ecdlp_index_calculus_state/low_term_total2_expanded_leaf_rescue_344_359_fixed_summary.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_344_359_fixed_probe.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_quadratic_root_fresh_344_359_fixed_probe.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_rolling_with_280_359_fixed.json
ecdlp_index_calculus_state/ffe_public_factor_guarded_portfolio_loo_with_280_359_fixed.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260526_public_factor_280_359_guarded_remainder_repeat/false_positive_guard_structural_audit.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260526_public_factor_280_359_guarded_remainder_repeat/full_remainder_sparsity_audit.json
```

Fixed public compact-leaf stress, 344-359:

- Fixed public row-selector policy:
  `fixed_target_cap2_ow0_hw3_lw0_sw0_cw0_aw1`.
- 31 fixed-policy verified leaf cases.
- 24 fixed-policy below-rho cases.
- Target counts among below-rho fixed-policy cases:
  `22050.cf1@11731`: 19, `67.a1@9803`: 5.
- Best fixed-policy case is target `22050.cf1@11731`, transfer 349,
  row keys `salt167` and `salt172`, selected leaf count 5, at 0.69343066 rho.

Sage/resultant factor audit, 344-359:

- 24 input cases materialized 15 FFE surfaces.
- 15/15 surfaces have nontrivial Sage resultant factorization.
- 14/15 surfaces have a preserving Sage factor candidate.
- 14/15 have preserving factor-surface cost below rho.
- 14/15 have preserving factor-root-scan cost below rho.
- 1/15 has full-remainder FFE cost below rho.
- Best preserving factor-root-scan cost is 0.33576642 rho.
- Best preserving factor-surface FFE cost is 0.34306569 rho.
- Minimum full-remainder FFE cost is 0.93430657 rho.

Repeated full-remainder FFE below-rho witness:

- Target: `22050.cf1@11731`.
- Transfer: 348.
- Row key: `22050.cf1@11731:uniform:256:salt165`.
- Candidate: `sage_resultant_factor_6`.
- Selected leaf count: 1.
- Known hit-root count: 12.
- Surface monomials: 3.
- Full remainder monomials: 79.
- Surface FFE cost: 0.37956204 rho.
- Factor root-scan cost: 0.45985401 rho.
- Full remainder FFE cost: 0.93430657 rho.

Direct public quadratic-root audit, 344-359:

- 15 surfaces, 13 public factor-order policies.
- Best single policy is `low_degree_then_order`.
- Best policy has 13 recovered surfaces, 12 preserving surfaces, 13 below-rho
  surfaces, and 1 quadratic false-positive surface.
- Raw false positives: 13 policy rows on 1 surface.
- The false-positive surface is target `22050.cf1@11731`, transfer 357,
  row `22050.cf1@11731:uniform:256:salt164`, candidate
  `sage_resultant_factor_2`; it misses selected root pair `[90, 6110]`.
- 0/13 raw false-positive rows pass the frozen portfolio guard
  `selected_leaf_count_eq1_and_factor_zero_eq1`.

Frozen rolling guarded portfolio, 280-359 included:

- 16 rolling test windows.
- Cumulative selected surfaces: 96.
- 96/96 recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Worst window maximum charged cost remains 0.84 rho.
- Latest window is `fresh_344_359_fixed`.
- Latest window selected surfaces: 9.
- Latest window result: 9/9 recovered, preserved, below rho, zero false
  positives, no missed guarded-oracle surfaces.
- Latest window max charged cost: 0.6350365 rho; mean charged cost:
  0.5316983 rho.
- Latest selected policy mix: `target_fingerprint_hash` 5,
  `global_fingerprint_hash` 2, `low_constant` 1, `low_degree_then_order` 1.

Leave-one-window-out robustness, 280-359 included:

- 17 held-out windows.
- 119/119 selected surfaces recovered, preserved, and below rho.
- Zero quadratic false-positive surfaces.
- Zero gap to the guarded oracle upper bound.
- Zero missed guarded-oracle surfaces.
- No failing held-out windows.
- Worst window maximum charged cost remains 0.84 rho.

Conclusion:

The 344-359 block strengthens the current component line: the full-remainder
below-rho condition repeated on a new transfer block with the same sparse
79-monomial remainder profile, and the frozen public selected-leaf/factor-zero
guard again filtered raw below-rho false positives without losing guarded
portfolio coverage.  This is still not a complete ECDLP index-calculus
algorithm, but it is a better mechanism lead than a one-off quotient/root-scan
artifact.

Non-claims:

- This remains a small-field quotient/FFE component, not a complete ECDLP
  index-calculus algorithm.
- Only 1/15 materialized 344-359 surfaces has full-remainder FFE cost below
  rho.
- One 344-359 surface lacks a preserving Sage factor.
- Direct public quadratic-root selection is false-positive-prone before the
  frozen public guard.
- Hash policies remain in the public selector set and still need a structural
  explanation before becoming a mechanism claim.
