# Next Hypothesis

## Hypothesis

The best next line is to treat the priority-column recurrence as the active
mechanism candidate and split it from controls:

```text
public carrier:
  mode_low_term_support_total5 top_k=16 full support

priority signal:
  column 15 accepted on 6117,6228,6256,6280,6314,6342

fresh replay witness:
  6342 salts 173/162
  forms [1,5], [11,15]
  rank_gain 2 / unique_gain 3 below rho
```

The `6288_6295` direct/rank gap still has to be repaired before calling the
tail contiguous.

## Missing bridge target

The missing producer outputs remain:

```text
low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_6288_6295_probe.json
low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_6288_6295_probe.json
```

I found only consumer/audit scripts for these artifact families in this
checkout.  The direct/rank producer appears to be in the external AutoLab job
stream, while the scout for `6288_6295` is already present.

## Gap audit command

Run this before treating a later tail as contiguous:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_frontier_gap_audit.py \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --start-min 6280 \
  --end-max <latest_visible_end> \
  --out ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_<latest_visible_end>_probe.json
```

If `missing_any_ranges` still contains `6288_6295`, keep
`contiguous_common_end=6287` in summaries.

## Replay priority

1. `6288_6295` direct/rank bridge
   This is the integrity gate for the frontier.

2. `6342` priority recurrence replay
   Compare it to `6256` and `6280`, because all three include `[1,5]` and
   `[11,15]`, while `6342` is a later below-rho repeat.

3. `6314` priority recurrence replay
   Separate the `[2,4]/[10,13]` lower-top-k events from the top-k16
   `[2,4]/[11,15]` accepted priority event.

4. `[10,14]` continuation
   Keep `6298` and `6320` in the queue, but priority recurrence is currently
   the more distinctive line.

5. Public split against controls
   Mine a second-stage public rule for six priority positives versus 66
   selected-priority no-rank controls.

## Promotion evidence

- The `6288_6295` bridge appears without demoting the priority recurrence.
- A seventh `[11,15]` accepted priority row appears beyond `6342`.
- A public token split separates priority positives from controls.
- `[10,14]` repeats beyond `6320` without losing below-rho direct cost.

## Demotion evidence

- The missing bridge lands only on saturated or no-rank controls.
- `6342` is the last priority recurrence and `[11,15]` stops repeating.
- Stable public tokens remain broad top-k16/full-support only.
- New direct/rank tails add controls faster than priority positives.
