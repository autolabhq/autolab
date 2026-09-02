# Experiment Result: priority public split through 6343

## Claim or task

Test whether the accepted priority-column 15 recurrence has a second-stage
public split beyond the broad top-k16 full-support carrier.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / PUBLIC-SPLIT-SUBCARRIER /
NONCONTIGUOUS-FRONTIER-CHECKPOINT.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_priority_public_split_miner.py
ecdlp_index_calculus_state/low_term_total2_priority_public_split_miner_5984_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6343_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_priority_public_split_6343/
```

## Label set

The split miner labels only direct-audited certificates:

```text
priority positives                         6
selected-priority no-rank controls        66
selected-priority other rank-gain rows    23
```

Priority positives:

```text
6117 salts 167/170 form [11,15]                  rank_gain 1 unique_gain 1
6228 salts 167/176 form [11,15]                  rank_gain 1 unique_gain 1
6256 salts 166/167 forms [1,5], [11,15]          rank_gain 2 unique_gain 3
6280 salts 161/175 forms [1,5], [10,14], [11,15] rank_gain 3 unique_gain 13
6314 salts 174/175 forms [2,4], [11,15]          rank_gain 1 unique_gain 3
6342 salts 162/173 forms [1,5], [11,15]          rank_gain 2 unique_gain 3
```

## Public split result

The best full-recall public carrier is simple:

```text
selected_has=13
positive_count 6 / 6
control_count 2 / 66
precision 0.75
controls: 6087,6319
```

Equivalent broad carriers include `top_k=16`, full selected support, and
`selected_has_13_and_15`.  This is not yet a clean rule, but it cuts the
selected-priority no-rank controls from `66` down to `2`.

The best non-exact zero-control subcarrier is:

```text
salt_max_mod4=3 AND selected_has=13
positive_count 3 / 6
control_count 0 / 66
precision 1.0
recall 0.5
jackknife_train_positive_min 2
positive transfers: 6256,6280,6314
```

This is the first useful second-stage public split: it catches the middle
priority recurrence subfamily without using accepted-form support as a feature.
It does not cover the late `6342` row, so it is a subcarrier rather than a full
algorithmic rule.

## Posthoc exclusions

The broad carrier can be made clean on this frozen dataset by excluding public
tokens seen only on the two controls:

```text
carrier: selected_has=13
exclude: salt_gap=2 OR salt_has=172
covered positives: 6117,6228,6256,6280,6314,6342
covered controls before exclusion: 6087,6319
surviving controls after exclusion: none
```

This is explicitly posthoc.  It must be tested on future direct/rank tails
before it becomes a claim.

## Interpretation

The priority recurrence now has a plausible public two-stage filter:

```text
stage 1: selected_has=13 / top-k16 full support carrier
stage 2: salt_max_mod4=3 subcarrier for the 6256,6280,6314 branch
```

The split is not strong enough to claim a finished descent or a large-field
speedup.  It is strong enough to turn the next AutoLab work from generic replay
into a targeted validation question: do new top-k16 selected-has-13 rows keep
landing in accepted priority-column rank gain, and does the `salt_max_mod4=3`
subcarrier continue to have zero controls?

## Frontier boundary

Direct/rank are still missing `6288_6295`, while scout has already moved beyond
the direct/rank tail.  This split audit is therefore a noncontiguous-tail
mechanism audit through `6343`, not a contiguous frontier proof.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The zero-control subcarrier covers only half the positives.
- Posthoc exclusions are not validated until a future direct/rank tail tests
  them.
- Open queued rows are work orders until direct certificate export and rank
  audit match them.
