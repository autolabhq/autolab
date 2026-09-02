# Next Hypothesis

## Hypothesis

There are now two active public lines:

```text
priority carrier:
  selected_has=13
  positive_count 8 / 8
  no-rank controls 2 / 82
  precision 0.8

priority subcarrier:
  salt_max_mod4=3 AND selected_has=13
  positive_count 4 / 8
  no-rank controls 0 / 82
  transfers 6256,6280,6314,6378

nonpriority accepted-form family:
  support [10,14]
  accepted_missing_rank_gain_count 11
  rank_gain_total 16
  fresh transfers 6402,6405
```

The strongest new evidence after `6399` is the `[10,14]` recurrence.  The
priority subcarrier remains live, but did not receive a new positive in the
`6400..6415` tail.

## Immediate replay target

When more direct/rank tails arrive, score these first:

```text
family queue:   6128,6287,6382,6297,6209,6340,6268,6189,6183,6244,6225,6204
priority queue: 6128,6165,6240,6248,6270,6311,6287,6143,6303,6313,6415,6148
soft queue:     6121,6163,6183,6244,6247,6255,6293,6298,6320,6367,6382,6411
```

For newly arrived direct/rank ranges, prioritize:

```text
1. top_k=16 full-support rows carrying selected_has=13
2. rows whose direct accepted form is [10,14]
3. rows satisfying salt_max_mod4=3 AND selected_has=13
```

`6293` remains scout-only until `6288_6295` direct/rank appears.  The current
gap audit still reports `6288_6295`, `6344_6351`, and `6352_6359` as missing
direct/rank ranges.

## Refresh command path

After the next direct/rank tail appears:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_frontier_gap_audit.py \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --start-min 6280 \
  --end-max <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --priority-columns 15 \
  --start-min 5984 \
  --end-max <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_priority_public_split_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --priority-columns 15 \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --max-rule-size 3 \
  --min-positive-support 2 \
  --rule-limit 50 \
  --out ecdlp_index_calculus_state/low_term_total2_priority_public_split_miner_5984_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_accepted_form_public_feature_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --calibration-start 5984 \
  --calibration-end 6023 \
  --holdout-start 6024 \
  --holdout-end <latest_direct_end> \
  --candidate-limit 200 \
  --out ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_promoted_candidate_rank_audit.py \
  --miner ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_<latest_direct_end>_probe.json \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --candidate-field promoted_holdout_candidates \
  --audit-start 6024 \
  --audit-end <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_accepted_form_contrast_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --promoted-audit ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json \
  --priority-columns 15 \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --open-start 6120 \
  --open-end <latest_direct_end> \
  --top-tokens 40 \
  --out ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_<latest_direct_end>_probe.json
```

Keep running the gap audit first; the contiguous common frontier is still
`6287`.

## Promotion evidence

- Another direct/rank tail row lands in `[10,14]` with rank gain under the
  top-k16 full-support carrier.
- A direct/rank tail row satisfies `salt_max_mod4=3 AND selected_has=13` and is
  an accepted priority-column rank gain.
- The broad `selected_has=13` carrier keeps recall 1.0 while controls stay low.
- The missing `6344_6359` bridge validates either `[10,14]` or the priority
  subcarrier instead of adding controls.

## Demotion evidence

- Future top-k16 full-support rows add no-rank controls faster than `[10,14]`
  rank gains.
- A future no-rank control satisfies `salt_max_mod4=3 AND selected_has=13`.
- The missing `6344_6359` bridge adds controls to the priority subcarrier.
- `[10,14]` stops repeating after `6405`.
