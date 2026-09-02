# Next Hypothesis

## Hypothesis

The public selector pattern
`mode_low_term_support_total5 | top_k=7 | 0,4,5,6,7,10,11,14,15` predicts a
replayable accepted-form bridge family better than broad selected-support
features.

## Null hypothesis

The `top_k=7` holdout hits are merely posthoc direct-key conveniences.  They
may replay to saturated accepted forms, fail to add missing branch-bank columns,
or add no rank against the existing branch-family bank.

## Immediate command path

Refresh the miner when new support-scout windows appear:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_accepted_form_public_feature_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6023_probe.json \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --calibration-start 5984 \
  --calibration-end 6023 \
  --holdout-start 6024 \
  --holdout-end <latest> \
  --candidate-limit 60 \
  --out ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_<latest>_probe.json
```

Prioritize direct replay/rank-scorer export for these holdout rows:

```text
6028 top_k=7 salts 164/166
6031 top_k=7 salts 162/168
6042 top_k=7 salts 163/172
6043 top_k=7 salts 161/173
```

Use these controls to test false positives:

```text
6030 top_k=7 salts 165/166
6040 top_k=7 salts 161/163
6028/6031/6032/6042 top_k=16 near-full support with column 13
```

## Promotion evidence

- Holdout direct certificates preserve missing columns in accepted forms.
- Rank scorer reports positive gain against the `5480..6007` branch-bank
  baseline.
- Source/shared-product charging remains below rho after replay costs are
  included.
- The selector continues to work on a later untouched support-scout window.

## Demotion evidence

- Replayed rows collapse to `[3,5]` or other saturated accepted forms.
- The rule only succeeds when exact salts/row keys are baked in.
- Broad `top_k=16` or `selected_has_13` predicates dominate fresh candidates.
- No shared-product/source-charged path can reproduce the direct-key evidence.
