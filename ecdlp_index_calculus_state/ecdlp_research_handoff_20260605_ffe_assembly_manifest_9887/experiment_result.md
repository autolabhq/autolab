# ECDLP FFE Assembly Manifest Refresh Through 9887

Date: 2026-06-05

## Result

The selected13 public lane was refreshed from the 9871 handoff to the 9887
frontier and materialized into an FFE/summation-polynomial assembly manifest:

- Manifest: `ecdlp_index_calculus_state/low_term_total2_ffe_assembly_manifest_selected13_9696_9887_probe.json`
- Claim status: `FFE_ASSEMBLY_MANIFEST_HAS_VALIDATED_RANK_GAIN_STREAM_AND_BACKFILL_TARGETS`
- Relation rows: 180 selected13 rows across 60 transfers
- Direct status: 63 exported, 117 direct-missing
- Positive transfer union: 14 transfers
- Positive transfers: 9705, 9732, 9742, 9750, 9754, 9776, 9803, 9820, 9828, 9842, 9849, 9860, 9880, 9884

This is a relation assembly artifact, not an ECDLP recovery or Pollard-rho
speedup claim.

## Lane Summary

| Lane | Rows | Positive transfers | Transfer-dedup rank gain | Missing rows | Full-family missing transfers |
| --- | ---: | ---: | ---: | ---: | --- |
| `selected13_all` | 180 | 14 | 18 | 117 | 9696, 9699, 9700, 9701, 9707, 9713, 9715, 9719, 9728, 9729, 9739, 9743, 9755, 9761, 9763, 9767, 9771, 9790, 9795, 9799, 9806, 9814, 9821, 9822, 9825, 9827, 9833, 9837, 9839, 9840, 9847, 9858, 9864, 9867, 9870, 9872, 9873, 9875, 9877 |
| `selected13_nonadjacent` | 147 | 12 | 14 | 99 | 9699, 9700, 9701, 9707, 9713, 9715, 9719, 9728, 9729, 9739, 9743, 9755, 9761, 9767, 9771, 9790, 9799, 9806, 9814, 9825, 9827, 9833, 9837, 9839, 9840, 9847, 9858, 9864, 9867, 9870, 9872, 9873, 9875 |
| `selected13_sharp_minmod3` | 42 | 6 | 7 | 24 | 9715, 9728, 9755, 9790, 9814, 9839, 9840, 9872 |

The sharp lane remains the best FFE-facing assembly target.  It has exact
rank-gain positives on transfers 9742, 9754, 9776, 9803, 9842, and 9884, while
leaving only 8 full-family direct-missing transfers for backfill.

## Refresh Commands

Bridge refresh:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --start-min 5984 --end-max 9887 \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9887_probe.json
```

Selected13 workorder and lane validation:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_workorder.py \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9887_probe.json \
  --families '11,15;10,14;0,5' \
  --tokens selected_has=13 \
  --start 9696 --end 9887 \
  --candidate-limit 1300 \
  --out ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9887_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_comparator_audit.py \
  --workorder ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9887_probe.json \
  --tokens selected_has=13 \
  --out ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9887_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_public_lane_feature_lift_miner.py \
  --workorder ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9887_probe.json \
  --base-tokens selected_has=13 \
  --max-size 2 --min-positive-support 3 --min-missing-transfers 3 --limit 40 \
  --out ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9887_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_public_lane_validation_manifest.py \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9887_probe.json \
  --feature-lift ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9887_probe.json \
  --start 9696 --end 9887 --limit 120 \
  --out ecdlp_index_calculus_state/low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_9696_9887_probe.json
```

FFE assembly manifest:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_ffe_assembly_manifest.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9887_probe.json \
  --comparator-audit ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9887_probe.json \
  --feature-lift ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9887_probe.json \
  --validation-manifest ecdlp_index_calculus_state/low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_9696_9887_probe.json \
  --limit 120 \
  --out ecdlp_index_calculus_state/low_term_total2_ffe_assembly_manifest_selected13_9696_9887_probe.json
```

## Honesty Boundary

- The manifest is a structured assembly queue for FFE/summation-polynomial replay.
- Positive rows inherit direct/rank evidence from the bridge audit.
- Rows with `same_salt_bridge_certificate` or `best_transfer_bridge_certificate`
  must be replayed before being promoted as exact row-level FFE successes.
- Missing rows are direct/rank export targets only.
- No large-field FFE result, target descent, private-key recovery, or proven
  below-rho algorithmic speedup is claimed here.
