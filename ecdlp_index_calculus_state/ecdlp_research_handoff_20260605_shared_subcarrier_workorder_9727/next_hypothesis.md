# Next Hypothesis: Direct Export Transfer 9696 First

## Immediate Action

Run direct/rank export for the `9696_9703` support-scout rows matching:

- `transfer_index = 9696`
- `salt_min_mod4=2`
- `selected_has=13`

Highest-priority selected support:

`[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`

This row is compatible with all three clean families:

- `[11,15]`
- `[10,14]`
- `[0,5]`

## Why This Is the Right Next Test

The `9720_9727` scout-only block is direct-missing, but it has zero full shared-subcarrier rows. Transfer `9696` is the only currently observed direct-missing tail transfer that matches the shared public rule exactly.

## Success Criteria

After direct/rank export lands for `9696_9703`, rerun:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --start-min 5984 --end-max 9727 --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9727_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_shared_subcarrier_workorder.py --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9727_probe.json --start 9696 --end 9727 --candidate-limit 240 --out ecdlp_index_calculus_state/low_term_total2_shared_subcarrier_workorder_9696_9727_probe.json
```

Then check whether transfer `9696` moved from `direct_certificate_missing` to `direct_certificate_exported`, and whether its accepted form lands in `[11,15]`, `[10,14]`, `[0,5]`, or a mismatch family.

