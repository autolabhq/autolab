# Next Hypothesis

## Hypothesis

The best next line is exact-row replay plus second-stage splitting for the
coupled `6280` form cluster:

```text
public carrier:
  mode_low_term_support_total5 top_k=16 full support

posthoc target cluster:
  [1,5], [10,14], [11,15]

reason:
  6280 joins all three forms, accepts priority column 15,
  and contributes rank_gain 3 / unique_gain 13 below rho.
```

The older `[10,13]` line remains real, but `[10,14]` has overtaken it by
rank-gain total, and `[11,15]` now has four priority hits.

## Immediate command path

When direct/rank/scout files move past this checkpoint, refresh with
`<latest_direct_end>`:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --priority-columns 15 \
  --start-min 5984 \
  --end-max <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json
```

Then rerun the priority recurrence audit and family planner:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_priority_recurrence_audit.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --promoted-audit ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json \
  --soft-scorer ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_<latest_direct_end>_probe.json \
  --priority-columns 15 \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --open-start 6120 \
  --open-end <latest_direct_end> \
  --top-tokens 40 \
  --open-candidate-limit 24 \
  --out ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_family_replay_planner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --promoted-audit ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json \
  --soft-scorer ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_<latest_direct_end>_probe.json \
  --families '10,13;11,15;10,14;9,11;8,11;1,5' \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --open-start 6120 \
  --open-end <latest_direct_end> \
  --open-candidate-limit 24 \
  --out ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_<latest_direct_end>_probe.json
```

## Priority replay queue

```text
1. 6163 top_k=16 salts 166/172
   Highest priority-recurrence score.

2. 6255 top_k=16 salts 166/168
   High priority queue row near the 6256 positive.

3. 6183 top_k=16 salts 164/165
   High in both priority and family queues.

4. 6121 top_k=16 salts 174/176
   Persistent high soft/priority row.

5. 6244 and 6247 top_k=16
   Fresh post-6239 top-k16 carrier rows.

6. 6202 top_k=16 salts 168/170
   Same transfer as lower-top-k [8,11]/[9,11]/[10,13] positives.

7. 6148 and 6184 top_k=16
   Top heads for [11,15] and [1,5] per-family replay after 6280.
```

## Promotion evidence

- A queued exact top-k16 row direct-certifies with the `[1,5]/[10,14]/[11,15]`
  coupled form cluster.
- A fifth `[11,15]` accepted priority row appears outside the current four-hit
  set.
- A second-stage public token split separates top-k16 priority positives from
  the 56 selected-priority no-rank controls.
- Shared-product/source-charged replay reproduces the `6280` style bridge below
  rho.

## Demotion evidence

- Top priority/family queued rows keep landing on selected-priority no-rank
  controls.
- `[10,14]` and `[11,15]` stop repeating beyond this local window.
- Stable public tokens remain broad support/carrier only.
- No shared-product/source-charged witness appears.
