# Next Hypothesis

## Hypothesis

Repair the one live frontier gap, then test whether `6314` is a real priority
recurrence mechanism rather than a local collision of broad top-k16 support:

```text
frontier repair:
  materialize direct/rank artifacts for 6288_6295

priority mechanism:
  6314 top_k=16 salts 174/175
  forms [2,4], [11,15]
  accepted priority column 15
  rank_gain 1 / unique_gain 3 below rho

parallel family mechanism:
  6320 top_k=16 salts 165/164
  form [10,14]
  rank_gain 1 / unique_gain 1 below rho
```

The older `6280` row remains the strongest coupled witness because it joins
`[1,5]`, `[10,14]`, and `[11,15]` with rank_gain `3`.  The new `6314` row is
the first post-6280 evidence that priority column 15 is recurring.

## Missing bridge target

The exact missing producer artifacts are:

```text
low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_6288_6295_probe.json
low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_6288_6295_probe.json
```

The scout artifact for `6288_6295` is already visible.  Do not spend a replay
slot duplicating scout-only work for that range.

## Gap audit command

Run this before calling any later evidence contiguous:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_frontier_gap_audit.py \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --start-min 6280 \
  --end-max <latest_visible_end> \
  --out ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_<latest_visible_end>_probe.json
```

If `missing_any_ranges` still contains `6288_6295`, keep
`contiguous_common_end=6287` in all summaries.

## Refresh command path

After the bridge appears, refresh with the latest contiguous end:

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

Then rerun the frozen selector, promoted rank audit, contrast miner, family
planner, and priority recurrence audit with the same `<latest_contiguous_end>`.

## Replay priority

1. `6288_6295` direct/rank bridge
   This is still the integrity gate for the frontier.

2. `6314` priority recurrence replay
   Separate the `[2,4]/[10,13]` lower-top-k events from the top-k16
   `[2,4]/[11,15]` accepted priority event.

3. `6320` `[10,14]` replay
   Test the repeated salts `165/164` and top-k16 full-support carrier after
   the `6298` and `6320` repeats.

4. Control split
   Mine a public split between the five priority positives and 63
   selected-priority no-rank controls.

## Promotion evidence

- The `6288_6295` bridge appears without demoting the priority recurrence.
- Another `[11,15]` accepted priority row appears beyond `6314`.
- `[10,14]` repeats beyond `6320` with direct rank gain below rho.
- A public second-stage split isolates priority positives from controls.

## Demotion evidence

- The missing bridge lands only on saturated or no-rank controls.
- `6314` is the last priority recurrence and `[11,15]` stops repeating.
- `[10,14]` stops after `6320`.
- Stable public tokens remain broad top-k16/full-support only.
