# Next Hypothesis

## Hypothesis

First repair the frontier, then exploit the reinforced `[10,14]` line:

```text
frontier repair:
  materialize direct and rank artifacts for 6288_6295

mechanism line:
  public carrier mode_low_term_support_total5 top_k=16 full support
  posthoc target cluster [1,5], [10,14], [11,15]
  new tail evidence 6298 adds one [10,14] rank gain
```

The `6280` row is still the best coupled priority witness.  The `6298` row is
important because it pushes `[10,14]` to rank_gain_total `11`, but it does not
add a fifth priority-column 15 positive.

## Missing bridge target

The exact missing producer artifacts are:

```text
low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_6288_6295_probe.json
low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_6288_6295_probe.json
```

The scout artifact for `6288_6295` is already visible, so do not replay scout
work just to fill this gap.

## Gap audit command

Run this before treating any later tail as contiguous:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_frontier_gap_audit.py \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --start-min 6280 \
  --end-max <latest_visible_end> \
  --out ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_<latest_visible_end>_probe.json
```

If the output still reports `missing_any_ranges: ["6288_6295"]`, keep
`contiguous_common_end=6287` in summaries even when later evidence is consumed.

## Refresh command path after the bridge appears

Use `<latest_contiguous_end>` only after the gap audit says the common frontier
is contiguous:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --priority-columns 15 \
  --start-min 5984 \
  --end-max <latest_contiguous_end> \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_contiguous_end>_probe.json
```

Then rerun the contrast, family, and priority audits against the same
`<latest_contiguous_end>`.

## Replay priority

1. `6288_6295` direct/rank bridge
   This is the integrity gate for the live frontier.

2. `[10,14]` row replay around `6298`
   Test whether the new `[10,14]` hit is a repeated public carrier event or a
   local tail artifact.

3. Coupled `6280` style rows
   Keep searching for rows that join `[1,5]`, `[10,14]`, and `[11,15]` while
   accepting priority column 15.

4. Priority split against controls
   Stable public tokens are still broad.  The next useful split must separate
   the four priority positives from the 58 selected-priority no-rank controls.

## Promotion evidence

- The missing `6288_6295` direct/rank batch appears and preserves the
  `[10,14]` lead without introducing contradictory rank evidence.
- A fifth `[11,15]` accepted priority row appears outside
  `6117,6228,6256,6280`.
- More top-k16 rows direct-certify the coupled `[1,5]/[10,14]/[11,15]` form
  cluster below rho.
- A second-stage public token split separates priority positives from
  selected-priority controls.

## Demotion evidence

- The missing `6288_6295` bridge lands only on saturated or no-rank controls.
- `[10,14]` stops repeating after `6298`.
- `[11,15]` remains capped at four priority positives.
- Public tokens remain top-k16/full-support only with no reproducible split.
