# Next Hypothesis

## Hypothesis

The current best nonpriority accepted-form line is:

```text
family: [10,14]
carrier: selected_has=13
positive_count 12 / 12
no-rank controls 2 / 85
fresh tail validation: 6430

subcarrier:
  salt_min_mod4=2 AND selected_has=13
  positive_count 3 / 12
  no-rank controls 0 / 85
  transfers 6164,6405,6430
```

The broad carrier is validated by the `>=6416` replay.  The salt subcarrier is
currently full-fit evidence and needs the next direct/rank tail.

## Immediate replay target

When more direct/rank tails arrive, score these first:

```text
family queue:   6128,6287,6382,6297,6426,6268,6209,6250,6358,6403,6340,6273
priority queue: 6128,6165,6434,6240,6248,6270,6311,6287,6143,6303,6313,6415
soft queue:     6121,6163,6183,6247,6255,6293,6298,6320,6367,6382,6411,6432
```

For newly arrived direct/rank ranges, prioritize:

```text
1. top_k=16 full-support rows carrying selected_has=13
2. rows that satisfy salt_min_mod4=2 AND selected_has=13
3. rows whose direct accepted form is [10,14]
4. rows satisfying the priority subcarrier salt_max_mod4=3 AND selected_has=13
```

The current gap audit reports these missing direct/rank ranges:

```text
6288_6295, 6344_6351, 6352_6359, 6416_6423, 6432_6439
```

Scouts have reached `6448_6455`, but direct/rank stop at `6440_6447`.

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

python3 tasks/ecdlp_index_calculus/low_term_total2_family_public_split_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --family 10,14 \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --replay-start <previous_latest_direct_end_plus_1> \
  --max-rule-size 3 \
  --min-positive-support 2 \
  --rule-limit 50 \
  --out ecdlp_index_calculus_state/low_term_total2_family_public_split_miner_10_14_5984_<latest_direct_end>_replay<previous_latest_direct_end_plus_1>_probe.json
```

Keep running the gap audit first; the contiguous common frontier is still
`6287`.

## Promotion evidence

- Another direct/rank row lands in `[10,14]` under the top-k16 full-support
  carrier.
- A future row satisfies `salt_min_mod4=2 AND selected_has=13` and is a
  `[10,14]` rank gain.
- Missing `6416_6423` or `6432_6439` direct/rank fills with `[10,14]` positives
  instead of no-rank controls.
- The broad selected-has-13 carrier keeps only two controls while positives
  continue to grow.

## Demotion evidence

- Future top-k16 full-support rows add no-rank controls faster than `[10,14]`
  positives.
- A no-rank row satisfies `salt_min_mod4=2 AND selected_has=13`.
- Missing `6416_6423` or `6432_6439` direct/rank adds controls to the
  `[10,14]` carrier.
- `[10,14]` stops repeating after `6430`.
