# Next Hypothesis: Shared Public Carrier Export Queue

## Working Hypothesis

The pair `salt_min_mod4=2 AND selected_has=13` is a public, source-side proxy for a low-term total2 relation-export surface. It appears to feed multiple accepted-form rank-gain families instead of only one family.

## Immediate AutoLab Work Order

Prioritize fresh support-scout rows with:

- `selected_has=13`
- `salt_min_mod4=2`
- selected support compatible with clean families `[11,15]`, `[10,14]`, or `[0,5]`

Treat `[2,4]` and `[11,13]` as diagnostic accepted-form targets, not as direct public-selected queue rules, until the selected-carrier mismatch is explained.

## Verification Targets

1. Run direct/rank export on the next scout-only blocks using the shared subcarrier first.
2. Measure hit rate separately for accepted families `[11,15]`, `[10,14]`, and `[0,5]`.
3. Add a selected-carrier reconciliation audit for `[2,4]` and `[11,13]`:
   - selected support before direct export
   - accepted support after relation reduction
   - rank gain and unique relation gain
4. If the shared carrier persists, derive the FFE/summation-polynomial reason for column `13` plus `salt_min_mod4=2` producing repeated rank-gain accepted forms.

## Exact Commands Used

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_frontier_gap_audit.py --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --start-min 6280 --end-max 9695 --out ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_9695_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' --start-min 5984 --end-max 9695 --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9695_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_accepted_form_contrast_miner.py --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9695_probe.json --audit-start 5984 --audit-end 9695 --open-start 9696 --open-end 9767 --open-candidate-limit 80 --out ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_9695_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_priority_public_split_miner.py --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9695_probe.json --audit-start 5984 --audit-end 9695 --out ecdlp_index_calculus_state/low_term_total2_priority_public_split_miner_5984_9695_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_family_public_split_miner.py --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9695_probe.json --family 10,14 --audit-start 5984 --audit-end 9695 --replay-start 6448 --out ecdlp_index_calculus_state/low_term_total2_family_public_split_miner_10_14_5984_9695_replay6448_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_family_public_split_miner.py --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9695_probe.json --family 11,15 --audit-start 5984 --audit-end 9695 --replay-start 6448 --out ecdlp_index_calculus_state/low_term_total2_family_public_split_miner_11_15_5984_9695_replay6448_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_family_public_split_miner.py --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_9695_probe.json --family 0,5 --audit-start 5984 --audit-end 9695 --replay-start 6448 --out ecdlp_index_calculus_state/low_term_total2_family_public_split_miner_0_5_5984_9695_replay6448_probe.json
```

