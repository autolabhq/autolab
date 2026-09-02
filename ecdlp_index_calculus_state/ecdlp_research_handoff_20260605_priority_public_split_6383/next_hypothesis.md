# Next Hypothesis

## Hypothesis

The accepted priority-column recurrence has a validated public subcarrier:

```text
carrier:
  selected_has=13
  positive_count 7 / 7
  no-rank controls 2 / 72

subcarrier:
  salt_max_mod4=3 AND selected_has=13
  positive_count 4 / 7
  no-rank controls 0 / 72
  transfers 6256,6280,6314,6378
```

The new `6378` row is the validation point: it arrived after the `6343` split
was mined and landed inside the subcarrier as an accepted priority-column
rank-gain row.

## Immediate replay target

When more direct/rank tails arrive, score top-k16 selected-has-13 rows first,
then prioritize rows with `salt_max_mod4=3`.  The current replay queue heads
from the family/priority stack are:

```text
priority queue: 6382,6367,6165,6248,6163,6255,6183,6293,6121,6244,6247,6240
family queue:   6128,6287,6382,6297,6209,6340,6268,6189,6183,6244,6165,6225
```

`6293` remains scout-only until `6288_6295` direct/rank appears.

## Refresh command path

After the next direct/rank tail appears:

```bash
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

Keep running the gap audit first; the current missing direct/rank ranges are
`6288_6295`, `6344_6351`, and `6352_6359`.

## Promotion evidence

- Another direct/rank tail row satisfies `salt_max_mod4=3 AND selected_has=13`
  and is an accepted priority-column rank gain.
- The broad `selected_has=13` carrier keeps recall 1.0 while controls stay low.
- The missing `6344_6359` bridge validates the subcarrier rather than adding a
  no-rank control.
- `[1,5]/[11,15]` repeats again below rho.

## Demotion evidence

- A future no-rank control satisfies `salt_max_mod4=3 AND selected_has=13`.
- The missing `6344_6359` bridge adds controls to the subcarrier.
- New selected-has-13 rows accumulate no-rank controls faster than accepted
  priority positives.
- `[11,15]` stops repeating after `6378`.
