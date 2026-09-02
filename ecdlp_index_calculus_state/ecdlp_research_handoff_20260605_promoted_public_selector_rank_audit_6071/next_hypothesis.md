# Next Hypothesis

## Hypothesis

Accepted-form rank gain is a two-stage public selection problem:

1. Use selector/top-k/support shape to find compact candidate families.
2. Use row-key salt geometry to separate accepted-missing rank gain from
   saturated accepted-form collapse.

The current compact family is:

```text
mode_low_term_support_total5 | top_k=7 | 0,4,5,6,7,10,11,14,15
```

The first strict second-stage public tokens from expanded calibration are:

```text
salt_gap=6
salt_gap=14
salt_gap=15
```

## Null hypothesis

The salt-gap rules are low-recall artifacts of this target/window and will not
fire on fresh rows.  The broad selector families may require direct replay
labels to become useful, which would make them poor source-side index-calculus
filters.

## Immediate command path

Refresh the direct bridge audit when mounted certificates move past `6071`:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --state-dir /Volumes/Volume/autolab/ecdlp_index_calculus_state \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --priority-columns 15 \
  --start-min 5984 \
  --end-max <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json
```

Then rerun the rank cross-audit over the newly direct-audited holdout:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_promoted_candidate_rank_audit.py \
  --miner ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6079_probe.json \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --audit-start 6024 \
  --audit-end <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json
```

For the next strict forward selector, train through the latest direct-audited
window and apply only to scout-only rows:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_accepted_form_public_feature_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --calibration-start 5984 \
  --calibration-end <latest_direct_end> \
  --holdout-start <latest_direct_end_plus_1> \
  --holdout-end <latest_scout_end> \
  --candidate-limit 80 \
  --out ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_<latest_direct_end>_to_<latest_direct_end_plus_1>_<latest_scout_end>_probe.json
```

## Promotion evidence

- Strict salt-gap or comparable row-key geometry rules fire on a future
  scout-only window and direct replay later confirms accepted-missing rank gain.
- The compact `top_k=7` family keeps capturing rank-gain rows while reducing
  `[3,5]` collapse frequency.
- A source/shared-product charged route reproduces the direct-key evidence
  below rho.
- The resulting accepted-form rows reduce branch-bank deficiency rather than
  only increasing duplicate relation count.

## Demotion evidence

- `salt_gap` rules continue to have zero forward candidates.
- Future direct certificates show the compact family mostly collapses to
  saturated supports.
- `top_k=16` remains dominated by unmatched or outcome-dependent rows.
- No shared-product/source-charged path can reproduce the direct-key hits.
