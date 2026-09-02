# Next Hypothesis: Shared Carrier Is a Broader Rank-Gain Trigger

## Immediate AutoLab Action

Prioritize direct/rank export for the direct-missing transfer `9696` rows in `9696_9703`.

Highest-priority row:

- `transfer_index = 9696`
- `row_keys = salt170,salt171`
- `selected_term_support = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`
- `tokens = salt_min_mod4=2, selected_has=13`

## Why This Is Now Stronger

The latest completed block produced a direct-exported comparator at transfer `9732` with the same shared public tokens and full clean-family-compatible support. It produced rank gain, but through accepted form `[1,5]` and without accepted-missing-column gain.

That means the public carrier should be tested as a broader low-term rank-gain trigger, not only as a selector for `[11,15]`, `[10,14]`, and `[0,5]`.

## Verification After Export

After `9696_9703` direct/rank export lands, rerun:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --start-min 5984 --end-max 9735 --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9735_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_workorder.py --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9735_probe.json --start 9696 --end 9735 --candidate-limit 300 --out ecdlp_index_calculus_state/low_term_total2_shared_subcarrier_workorder_9696_9735_probe.json
```

Then compare transfer `9696` against the transfer `9732` comparator in the work-order artifact.

