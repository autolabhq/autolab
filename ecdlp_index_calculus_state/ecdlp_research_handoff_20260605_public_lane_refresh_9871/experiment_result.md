# Experiment Result

## Inputs

- Refreshed bridge:
  `ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9871_probe.json`
- Refreshed selected13 workorder:
  `ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9871_probe.json`
- Refreshed selected13 comparator:
  `ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9871_probe.json`
- Refreshed feature lift:
  `ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9871_probe.json`
- Refreshed validation manifest:
  `ecdlp_index_calculus_state/low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_9696_9871_probe.json`

## Bridge Refresh

The mounted direct-certificate and rank-scorer stream now supports a bridge
audit through `9871`.

Bridge summary:

- direct below-rho certificates: `974`
- rank-gain certificates: `284`
- accepted-missing certificates: `616`
- accepted-missing rank gains: `250`
- rank-gain without accepted missing: `34`

This is still certificate evidence, not a complete faster-than-rho ECDLP
algorithm.

## Selected13 Lane

The selected13 lane improved materially relative to the `9791` checkpoint.

At `9791`:

- exported rank-gain rows: `12`
- full-family exported rank-gain rows: `6`

At `9871`:

- selected13 shared rows: `108`
- exported rows: `38`
- exported rank-gain rows: `24`
- full-family exported rank-gain rows: `12`
- missing full-family transfers: `35`

New selected13 exported rank-gain transfers beyond the old boundary:

`9803, 9820, 9828, 9842, 9849, 9860`

## Feature-Lift Result

The promoted public rule remains:

`selected_has=13 AND salt_adjacent=False AND salt_min_mod4=3`

At `9871` this sharp rule has:

- positive exported rows: `10`
- controls: `0`
- precision: `1.0`
- recall: `0.41666667`
- positive transfers: `9742, 9754, 9776, 9803, 9842`
- missing full-family transfers: `9715, 9728, 9755, 9790, 9814, 9839, 9840`

The earlier sharp-lane scout-only positives were partially validated by the
refreshed bridge: `9803` and `9842` are now exported rank-gain transfers.

## Validation Queue

The current unique-transfer queue should prioritize the remaining sharp-lane
full-family transfers:

1. `9728`, salts `163,166`
2. `9755`, salts `175,177`
3. `9790`, salts `167,172`
4. `9814`, salts `163,174`
5. `9839`, salts `163,172`
6. `9715`, salts `171,174`
7. `9840`, salts `163,170`

The broader queue remains:

`selected_has=13 AND salt_adjacent=False`

with `30` full-family transfers still needing validation through `9871`.

## Interpretation

This is the strongest selected13 public-lane evidence so far.  The same sharp
public salt/support rule kept zero exported controls and gained new positive
frontier transfers after the bridge advanced.  That makes it a better candidate
hit-stream filter for the next FFE/summation-polynomial relation assembly
stage.

The honest boundary is unchanged: this is a public routing/filtering advance
with verifier-backed rank-gain certificates, not yet an end-to-end ECDLP
speedup.
