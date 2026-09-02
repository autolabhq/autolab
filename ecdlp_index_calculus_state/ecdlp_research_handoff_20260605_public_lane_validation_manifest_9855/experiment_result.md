# Experiment Result

## Inputs

- Feature-lift artifact:
  `ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9791_probe.json`
- Bridge/direct-rank audit:
  `ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9791_probe.json`
- Mounted support scouts:
  `/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json`
- New manifest builder:
  `tasks/ecdlp_index_calculus/low_term_total2_public_lane_validation_manifest.py`
- New manifest:
  `ecdlp_index_calculus_state/low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_9696_9855_probe.json`

## Result

The support-scout frontier is ahead of the bridge/direct-rank frontier:

- latest bridge/direct-rank boundary: `9791`
- latest available and scanned support-scout boundary: `9855`
- raw support-scout rows scanned: `1360`
- materialized public-lane rows: `123`

Status split in the materialized public-lane rows:

- direct certificate exported: `27`
- direct certificate missing: `48`
- support report missing: `48`

The new manifest preserves both row-level validation requests and a compact
unique-transfer queue.  It treats `support_report_missing` as scout-only: the
row exists in mounted support-scout state, but the current bridge/direct-rank
audit does not yet cover it.

## Sharp Lane

Rule:

`selected_has=13 AND salt_adjacent=False AND salt_min_mod4=3`

Sharp-lane validation coverage:

- row variants needing validation: `27`
- full-family transfers needing validation: `9`
- direct-certificate-missing transfers:
  `9715, 9728, 9755, 9790`
- scout-only support-report-missing transfers:
  `9803, 9814, 9839, 9840, 9842`

Recommended unique-transfer order from the manifest:

1. `9728`, salts `163,166`, status `direct_certificate_missing`
2. `9755`, salts `175,177`, status `direct_certificate_missing`
3. `9790`, salts `167,172`, status `direct_certificate_missing`
4. `9715`, salts `171,174`, status `direct_certificate_missing`
5. `9814`, salts `163,174`, status `support_report_missing`
6. `9839`, salts `163,172`, status `support_report_missing`
7. `9803`, salts `163,171`, status `support_report_missing`
8. `9842`, salts `167,176`, status `support_report_missing`
9. `9840`, salts `163,170`, status `support_report_missing`

## Broad Lane

Rule:

`selected_has=13 AND salt_adjacent=False`

Broad-lane validation coverage:

- row variants needing validation: `96`
- full-family transfers needing validation: `32`
- direct-certificate-missing row variants: `48`
- support-report-missing row variants: `48`

Full-family transfer queue:

`9699, 9700, 9701, 9707, 9713, 9715, 9719, 9728, 9729, 9739, 9743, 9755, 9761, 9767, 9771, 9790, 9794, 9799, 9803, 9806, 9814, 9820, 9825, 9827, 9828, 9833, 9837, 9839, 9840, 9842, 9847, 9849`

## Interpretation

This is a validation/backfill artifact, not a new rank-gain certificate.  The
important movement is that the learned public lane now reaches beyond the
direct/rank bridge boundary: the sharp rule has five scout-only transfers past
`9791` that can test whether the earlier zero-control diagnostic lane persists
on new frontier rows.

If the sharp lane exports rank-gain certificates on the scout-only transfers,
the next step is to feed those rows into the FFE/summation-polynomial relation
assembly path as a smaller public hit-stream candidate.
