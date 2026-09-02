# Next Hypothesis

## Hypothesis

The accepted priority-column recurrence has a public two-stage structure:

```text
stage 1 carrier:
  selected_has=13
  top_k=16 full-support row family
  captures all six priority positives with only two no-rank controls

stage 2 subcarrier:
  salt_max_mod4=3 AND selected_has=13
  captures 6256,6280,6314 with zero controls
```

This should be tested on the next direct/rank tail, especially the scout-visible
`6344_6351` and `6352_6359` ranges once direct/rank artifacts appear.

## Immediate command path

After the next direct/rank tail appears, refresh the bridge and rerun:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_priority_public_split_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --priority-columns 15 \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --max-rule-size 3 \
  --min-positive-support 2 \
  --rule-limit 50 \
  --out ecdlp_index_calculus_state/low_term_total2_priority_public_split_miner_5984_<latest_direct_end>_probe.json
```

Keep the gap audit in front of the refresh:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_frontier_gap_audit.py \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --start-min 6280 \
  --end-max <latest_visible_end> \
  --out ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_<latest_visible_end>_probe.json
```

## Promotion evidence

- A new priority-positive row appears under `selected_has=13`.
- A new row satisfying `salt_max_mod4=3 AND selected_has=13` is also priority
  positive, keeping the subcarrier at zero controls.
- The posthoc exclusions `salt_gap=2` and `salt_has=172` reject future controls
  without rejecting priority positives.
- The missing `6288_6295` bridge does not introduce controls into the
  `salt_max_mod4=3` subcarrier.

## Demotion evidence

- A future direct/rank row satisfies `salt_max_mod4=3 AND selected_has=13` but
  lands as selected-priority no-rank.
- The broad `selected_has=13` carrier accumulates controls faster than priority
  positives.
- `6342` remains the last `[1,5]/[11,15]` priority repeat.
- The missing `6288_6295` bridge invalidates the apparent subcarrier.
