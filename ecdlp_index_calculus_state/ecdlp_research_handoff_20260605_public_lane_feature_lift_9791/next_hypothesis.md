# Next Hypothesis

## Immediate Work Order

Ask the direct/rank exporter to prioritize the high-precision diagnostic lane:

`selected_has=13 AND salt_adjacent=False AND salt_min_mod4=3`

Target transfers:

1. `9715`, salts `171,174`
2. `9728`, salts `163,166`
3. `9755`, salts `175,177`
4. `9790`, salts `167,172`

Success condition:

- At least one of the four missing full-family rows exports a rank-gain
  certificate.
- A stronger result is two or more rank-gain exports with accepted-missing
  rank gain, because that would support the sublane rather than one lucky row.

## Expansion Queue

If the high-precision lane does not land, fall back to the broader carrier:

`selected_has=13 AND salt_adjacent=False`

Missing full-family transfers:

`9699, 9700, 9701, 9707, 9713, 9715, 9719, 9728, 9729, 9739, 9743, 9755, 9761, 9767, 9771, 9790`

This broader queue preserved all observed exported rank-gain rows and removed
six exported controls relative to the base `selected_has=13` lane.

## Follow-Up Probe

After direct/rank export advances past `9791`, rerun:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_public_lane_feature_lift_miner.py \
  --workorder ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_<END>_probe.json \
  --base-tokens selected_has=13 \
  --max-size 2 \
  --min-positive-support 3 \
  --min-missing-transfers 3 \
  --limit 40 \
  --out ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_<END>_probe.json
```

Then compare whether `salt_adjacent=False` remains full-recall and whether
`salt_min_mod4=3` keeps zero exported controls.
