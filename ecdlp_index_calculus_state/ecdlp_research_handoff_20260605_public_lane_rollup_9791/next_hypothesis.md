# Next Hypothesis: Selected-13 Is the Main Public Export Carrier

## Hypothesis

`selected_has=13` is a public signal for a low-term relation surface that produces useful rank-gain certificates across several accepted forms. The stricter `salt_min_mod4=2` condition is best treated as a sublabel that may steer into `[1,5]` rank-gain-without-accepted-missing behavior.

## Immediate Test

Export direct/rank rows for the broad-lane full-family queue, with strict-lane rows annotated:

- Strict: `9696`, `9739`, `9763`
- Broad-only: `9699`, `9700`, `9701`, `9707`, `9713`, `9715`, `9719`, `9728`, `9729`, `9743`, `9755`, `9761`, `9767`, `9771`, `9790`

## Success Criteria

After the next export lands, rerun:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --start-min 5984 --end-max 9791 --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9791_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_workorder.py --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9791_probe.json --tokens selected_has=13 --start 9696 --end 9791 --candidate-limit 900 --out ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9791_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_comparator_audit.py --workorder ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9791_probe.json --tokens selected_has=13 --out ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9791_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_public_lane_comparison_rollup.py --lane strict=ecdlp_index_calculus_state/low_term_total2_shared_subcarrier_comparator_audit_9696_9791_probe.json --lane broad_selected13=ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9791_probe.json --out ecdlp_index_calculus_state/low_term_total2_public_lane_comparison_rollup_9696_9791_probe.json
```

Then measure whether broad-lane exported rank-gain rate stays near `0.5`, and whether strict-lane rows continue to land in `[1,5]` or related rank-gain-without-accepted-missing forms.

