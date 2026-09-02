# ECDLP FFE Replay Workorder Through 9935

Date: 2026-06-05

## Result

The selected13 public lane was refreshed from the 9887 assembly manifest to the
9935 direct/rank frontier.  Fresh direct/rank evidence added two selected13
rank-gain transfers, 9914 and 9931, but the sharp promoted lane remains the
same rule:

```text
selected_has=13 AND salt_adjacent=False AND salt_min_mod4=3
```

The new replay workorder is:

```text
ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_replay_workorder_selected13_9696_9935_probe.json
```

Claim status:

```text
FFE_SHARP_LANE_REPLAY_WORKORDER_READY
```

The workorder passed invariant checks: every sharp row has selected13,
non-adjacent public salt pairs, and min salt mod 4 equal to 3; the exact
positive queue matches the lane summary; and the direct/rank backfill queue
matches the full-family missing-transfer set.

## Frontier Delta

Bridge audit through 9935:

- Direct below-rho certificates: 983
- Rank-gain certificates: 288
- Accepted missing-column certificates: 619
- Accepted priority-column certificates: 827

Selected13 comparator through 9935:

- Exported rank-gain transfers: 9705, 9732, 9742, 9750, 9754, 9776, 9803, 9820, 9828, 9842, 9849, 9860, 9880, 9884, 9914, 9931
- Missing full-family transfers: 48 total
- Shared full-family exported rank-gain count: 16

Sharp lane through 9935:

- Row count: 51
- Public first-pass groups: 17
- Exact-positive groups: 6
- Inherited-positive rows: 12
- Direct/rank backfill groups: 11
- Direct status counts: 18 exported, 33 missing
- Median direct ops over rho: 0.67153285

## Replay Queues

Exact-positive replay transfers:

```text
9742, 9754, 9776, 9803, 9842, 9884
```

Direct/rank backfill transfers:

```text
9715, 9728, 9755, 9790, 9814, 9839, 9840, 9872, 9889, 9909, 9913
```

New sharp backfill transfers since the 9887 handoff:

```text
9889, 9909, 9913
```

Lowest-cost sharp backfill candidates:

| Transfer | Direct ops over rho | Salts | Full-family row id |
| ---: | ---: | --- | --- |
| 9728 | 0.6350365 | 163, 166 | `t9728_077c9ce3ade1` |
| 9755 | 0.6350365 | 175, 177 | `t9755_7e4493f8ec34` |
| 9790 | 0.6350365 | 167, 172 | `t9790_ce7877c9f345` |
| 9872 | 0.6350365 | 163, 173 | `t9872_c55f1a9223ae` |
| 9909 | 0.6350365 | 167, 171 | `t9909_a4cd3a970d56` |
| 9913 | 0.6350365 | 163, 171 | `t9913_67bdfbcc3cec` |

## Commands

Bridge refresh:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --start-min 5984 --end-max 9935 \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9935_probe.json
```

Selected13 lane refresh:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_workorder.py \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9935_probe.json \
  --families '11,15;10,14;0,5' \
  --tokens selected_has=13 \
  --start 9696 --end 9935 \
  --candidate-limit 2200 \
  --out ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9935_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_comparator_audit.py \
  --workorder ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9935_probe.json \
  --tokens selected_has=13 \
  --out ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9935_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_public_lane_feature_lift_miner.py \
  --workorder ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9935_probe.json \
  --base-tokens selected_has=13 \
  --max-size 2 --min-positive-support 3 --min-missing-transfers 3 --limit 40 \
  --out ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9935_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_public_lane_validation_manifest.py \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9935_probe.json \
  --feature-lift ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9935_probe.json \
  --start 9696 --end 9935 --limit 160 \
  --out ecdlp_index_calculus_state/low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_9696_9935_probe.json
```

FFE assembly and replay workorder:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_ffe_assembly_manifest.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9935_probe.json \
  --comparator-audit ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9935_probe.json \
  --feature-lift ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9935_probe.json \
  --validation-manifest ecdlp_index_calculus_state/low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_9696_9935_probe.json \
  --limit 160 \
  --out ecdlp_index_calculus_state/low_term_total2_ffe_assembly_manifest_selected13_9696_9935_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_ffe_sharp_lane_replay_workorder.py \
  --assembly-manifest ecdlp_index_calculus_state/low_term_total2_ffe_assembly_manifest_selected13_9696_9935_probe.json \
  --out ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_replay_workorder_selected13_9696_9935_probe.json
```

## Honesty Boundary

- The workorder is a kernel-facing replay specification, not the optimized FFE
  kernel.
- It does not evaluate summation polynomials or compute new rank evidence.
- Inherited-positive rows must become exact row-level certificates before lane
  promotion.
- Backfill rows are direct/rank export targets, not successes.
- No ECDLP recovery or Pollard-rho speedup is claimed.
