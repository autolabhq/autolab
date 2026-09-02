# Next Hypothesis: Promote Selected-13, Keep Salt-Min-Mod4 as Diagnostic

## Hypothesis

`selected_has=13` is the public carrier that selects productive low-term relation surfaces. `salt_min_mod4=2` is not the main selector; it is a stricter subcarrier that appears to steer toward `[1,5]` rank gain without accepted-missing-column gain.

## Immediate Tests

1. Export direct/rank rows for broad-lane missing full-family transfers.
2. Preserve strict-lane tags for `9696`, `9739`, and `9763`.
3. Compare outcomes against exported broad-lane rank-gain transfers:
   - `9705`: `[10,13]`
   - `9732`: `[1,5]`, rank gain without accepted missing
   - `9742`: `[10,14]`
   - `9750`: `[0,5]`, `[1,5]`, `[10,14]`
   - `9754`: `[0,5]`

## Re-run Commands

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --start-min 5984 --end-max 9775 --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9775_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_workorder.py --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9775_probe.json --tokens selected_has=13 --start 9696 --end 9775 --candidate-limit 700 --out ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9775_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_comparator_audit.py --workorder ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9775_probe.json --tokens selected_has=13 --out ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9775_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_public_lane_comparison_rollup.py --lane strict=ecdlp_index_calculus_state/low_term_total2_shared_subcarrier_comparator_audit_9696_9775_probe.json --lane broad_selected13=ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9775_probe.json --out ecdlp_index_calculus_state/low_term_total2_public_lane_comparison_rollup_9696_9775_probe.json
```

