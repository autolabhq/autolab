# Next Hypothesis

## Hypothesis

The priority-column path is now worth a dedicated exact-row replay pass.  The
public pre-direct carrier is:

```text
selector              mode_low_term_support_total5
top_k                 16
selected support      0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
posthoc form target   [11,15], with side-family [1,5] after 6256
```

The current target is not a stable salt rule.  It is a public carrier recurrence
that needs a second-stage split against top-k16 no-rank controls.

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

Then rerun the priority recurrence audit:

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
```

## Priority replay queue

```text
1. 6163 top_k=16 salts 166/172
   Highest priority-recurrence score; exact row still unmatched.

2. 6255 top_k=16 salts 166/168
   Fresh high priority queue row after the 6256 positive.

3. 6183 top_k=16 salts 164/165
   High priority and family-planner row; shared public full-support carrier.

4. 6121 top_k=16 salts 174/176
   Persistent high soft row in both priority and all-rank queues.

5. 6244 and 6247 top_k=16
   Fresh post-6239 top-k16 public-carrier rows.

6. 6202 top_k=16 salts 168/170
   Same transfer as lower-top-k [8,11]/[9,11]/[10,13] positives.

7. 6143,6169,6250,6208,6165 top_k=16
   Remaining high priority-recurrence exact-row replays.
```

## Promotion evidence

- A queued exact top-k16 row direct-certifies as accepted priority column 15.
- A fourth `[11,15]` priority bridge appears outside the `6117,6228,6256`
  cluster.
- A second-stage public token split separates top-k16 priority positives from
  the 56 selected-priority no-rank controls.
- Shared-product/source-charged replay reproduces the priority bridge below
  rho.

## Demotion evidence

- Top priority-queue rows keep landing on selected-priority no-rank controls.
- `[11,15]` remains at exactly three priority positives.
- The only stable public tokens remain broad support/carrier tokens.
- No shared-product/source-charged witness appears.
