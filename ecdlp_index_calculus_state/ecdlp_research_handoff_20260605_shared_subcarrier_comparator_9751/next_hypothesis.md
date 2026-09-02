# Next Hypothesis: Split the Carrier Into Two Public Lanes

## Hypothesis

Column `13` is the broad public carrier. The `salt_min_mod4=2` refinement marks a narrower lane that can produce rank gain without accepted-missing-column gain.

## Tests

1. Direct/rank export `9696_9703` transfer `9696`.
2. Direct/rank export or materialize the direct-missing transfer `9739`.
3. Compare both against exported strict comparator `9732` and broader accepted-missing hits `9742`, `9750`.

## Expected Outcomes

If the strict lane is real, `9696` or `9739` should produce at least one of:

- rank gain through `[1,5]` or related no-accepted-missing support,
- accepted-missing `[10,14]`, `[0,5]`, or `[11,15]`,
- a diagnostic no-rank/saturated control that refines the salt rule.

If both miss, the carrier should be weakened from a work-order rule to a diagnostic feature and the broad `selected_has=13` full-support lane should become the next queue rule.

## Re-run Commands

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --start-min 5984 --end-max 9751 --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9751_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_workorder.py --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9751_probe.json --start 9696 --end 9751 --candidate-limit 400 --out ecdlp_index_calculus_state/low_term_total2_shared_subcarrier_workorder_9696_9751_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_comparator_audit.py --workorder ecdlp_index_calculus_state/low_term_total2_shared_subcarrier_workorder_9696_9751_probe.json --out ecdlp_index_calculus_state/low_term_total2_shared_subcarrier_comparator_audit_9696_9751_probe.json
```

