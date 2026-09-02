# Experiment Result

## Inputs

- Work order:
  `ecdlp_index_calculus_state/low_term_total2_selected13_workorder_9696_9791_probe.json`
- Comparator:
  `ecdlp_index_calculus_state/low_term_total2_selected13_comparator_audit_9696_9791_probe.json`
- New miner:
  `tasks/ecdlp_index_calculus/low_term_total2_public_lane_feature_lift_miner.py`
- New artifact:
  `ecdlp_index_calculus_state/low_term_total2_public_lane_feature_lift_miner_selected13_9696_9791_probe.json`

## Result

The feature-lift miner deduplicates the work-order rows with the same identity
used by the comparator audit, then mines public token conjunctions inside the
broad `selected_has=13` lane.  Labels are used only on already exported
comparator rows.  Missing rows remain work orders.

Base broad lane after dedupe:

- rows: 60
- exported rows: 24
- exported rank-gain rows: 12
- exported rank-gain rate: 0.5
- missing rows: 36
- missing full-family transfers: 18

Best broad queue carrier:

- rule: `selected_has=13 AND salt_adjacent=False`
- exported positives: 12
- exported controls: 6
- precision: 0.66666667
- recall: 1.0
- missing full-family transfers:
  `9699, 9700, 9701, 9707, 9713, 9715, 9719, 9728, 9729, 9739, 9743, 9755, 9761, 9767, 9771, 9790`

High-precision diagnostic sublane:

- rule: `selected_has=13 AND salt_adjacent=False AND salt_min_mod4=3`
- exported positives: 6
- exported controls: 0
- precision: 1.0
- recall: 0.5
- missing full-family transfers: `9715, 9728, 9755, 9790`
- missing row salt pairs:
  - `9715`: salts `171,174`
  - `9728`: salts `163,166`
  - `9755`: salts `175,177`
  - `9790`: salts `167,172`

Secondary diagnostic:

- rule: `selected_has=13 AND salt_adjacent=False AND salt_max_mod4=0`
- exported positives: 4
- exported controls: 0
- missing full-family transfers: `9700, 9701, 9739, 9790`

## Interpretation

This is a routing improvement, not a completed ECDLP speedup.  The broad
carrier keeps all observed exported rank-gain rows and removes half of the
exported no-rank controls.  The sharp diagnostic sublane has zero exported
controls, but only four missing full-family transfers remain to test.

The result gives the AutoLab loop a smaller public direct/rank backfill target:
first test the four `salt_min_mod4=3` missing transfers, then use the broader
non-adjacent queue if more coverage is needed.

## Honesty Boundary

- The artifact does not create new direct/rank certificates.
- The best rule is selected using exported comparator labels.
- Missing rows still need direct/rank export before they can count as new
  index-calculus progress.
- Latest audited boundary for this handoff remains `9791`.
