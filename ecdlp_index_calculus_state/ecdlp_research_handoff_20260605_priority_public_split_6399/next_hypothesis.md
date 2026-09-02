# Next Hypothesis

## Hypothesis

The accepted priority-column recurrence has two public layers:

```text
carrier:
  selected_has=13
  positive_count 8 / 8
  no-rank controls 2 / 75
  precision 0.8

subcarrier:
  salt_max_mod4=3 AND selected_has=13
  positive_count 4 / 8
  no-rank controls 0 / 75
  transfers 6256,6280,6314,6378
```

The new `6394` row is broad-carrier evidence.  It is an accepted
priority-column rank-gain row, but with salts `172/173`, so it does not enter
the `salt_max_mod4=3` subcarrier.

## Immediate replay target

When more direct/rank tails arrive, score top-k16 selected-has-13 rows first,
then prioritize rows with `salt_max_mod4=3`.  The current replay queue heads
from the refreshed family/priority stack are:

```text
priority queue: 6128,6165,6240,6248,6270,6311,6287,6143,6303,6313,6148,6169
family queue:   6128,6287,6382,6297,6209,6340,6268,6189,6183,6244,6225,6204
soft queue:     6121,6163,6183,6202,6244,6247,6255,6293,6298,6320,6367,6382
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
```

Keep running the gap audit first; the contiguous common frontier is still
`6287`.

## Promotion evidence

- Another direct/rank tail row satisfies `salt_max_mod4=3 AND selected_has=13`
  and is an accepted priority-column rank gain.
- The broad `selected_has=13` carrier keeps recall 1.0 while controls stay low.
- The missing `6344_6359` bridge validates the subcarrier instead of adding a
  no-rank control.
- `[10,14]` or `[11,15]` repeats again below rho under the same top-k16
  full-support carrier.

## Demotion evidence

- A future no-rank control satisfies `salt_max_mod4=3 AND selected_has=13`.
- The missing `6344_6359` bridge adds controls to the subcarrier.
- New selected-has-13 rows accumulate no-rank controls faster than accepted
  priority positives.
- `[11,15]` stops repeating after `6394`.
