# Next Hypothesis

## Direct/Rank Backfill

Prioritize the sharp diagnostic rule:

`selected_has=13 AND salt_adjacent=False AND salt_min_mod4=3`

The first pass should export direct/rank evidence for the four historical gaps:

1. `9728`, salts `163,166`
2. `9755`, salts `175,177`
3. `9790`, salts `167,172`
4. `9715`, salts `171,174`

Then test the same sharp rule on the scout-only frontier rows beyond `9791`:

1. `9814`, salts `163,174`
2. `9839`, salts `163,172`
3. `9803`, salts `163,171`
4. `9842`, salts `167,176`
5. `9840`, salts `163,170`

## Success Criteria

- Minimal positive: at least one sharp-lane row exports rank gain.
- Stronger positive: at least two sharp-lane rows export accepted-missing
  rank gain.
- Frontier-transfer positive: at least one of the scout-only rows past `9791`
  exports rank gain after bridge/direct-rank coverage catches up.

## Rerun Command

After bridge/direct-rank export advances, rerun the manifest with the new
bridge artifact:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_public_lane_validation_manifest.py \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<END>_probe.json \
  --feature-lift ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9791_probe.json \
  --start 9696 \
  --end <END> \
  --limit 100 \
  --out ecdlp_index_calculus_state/low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_9696_<END>_probe.json
```

Then rerun the feature-lift miner if new exported rows are available, so the
sharp `salt_min_mod4=3` rule can be checked against fresh labels rather than
only old comparator evidence.
