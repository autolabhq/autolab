# Experiment Result: branch-family rank-frontier rollup through 6007

## Claim or task

Roll up the live branch-family duplicate-hit relation bank and adjacent
direct-source bridge after the mounted AutoLab frontier advanced through
`5992..6007`.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / INDEX-CALCULUS PRECURSOR.

## Inputs

Mounted AutoLab state was treated as the source of truth.  This sandbox did not
write into the mounted state; the rollup artifact and this handoff were written
locally in this worktree.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_branch_family_frontier_rollup.py
ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json

Mounted source artifacts summarized by the rollup:
ecdlp_index_calculus_state/low_term_total2_branch_family_streaming_sweep_min5992_cal5991_pos16_24_forward_through6007_probe.json
ecdlp_index_calculus_state/low_term_total2_branch_family_duplicate_hit_relation_certificates_5992_6007_probe.json
ecdlp_index_calculus_state/low_term_total2_branch_family_rank_novelty_gate_5992_6007_probe.json
ecdlp_index_calculus_state/low_term_total2_branch_family_duplicate_hit_target_eliminated_rank_5480_6007_combined_probe.json
ecdlp_index_calculus_state/low_term_total2_branch_family_duplicate_hit_linear_descent_audit_5480_6007_combined_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_6000_6007_probe.json
ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_6000_6007_probe.json
ecdlp_index_calculus_state/low_term_total2_factor_column_target_scout_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_6000_6007_probe.json
```

## Branch-family result

The latest branch-family slice `5992..6007` is a cheap local recovery but not a
new factor-rank direction:

- Streaming sweep: `4` attempts, `8` contexts, `7` hit contexts, `1` verified
  stop.
- Full-context streaming cost: `0.67153285x` rho per verified stop.
- Exported duplicate-hit certificate: transfer `5992`, branch
  `pos17:1,2,4,4`, rank `2`, public-key verified.
- Certificate-local full-context cost: `0.17518248x` rho.
- Rank-novelty gate: `1` verified rank-2 candidate, `0` rank gain, `0` unique
  factor-relation gain.  The emitted factor support is the existing
  `{1,2,4}` relation.

The combined duplicate-hit branch bank through `6007` now has:

- Local certificates: `22`.
- Unique target-eliminated factor relations: `6`.
- Factor rank: `5` over `16` factor variables.
- Deficiency: `11`.
- Derivable factor variables: `0`.
- Linear-descent audit: `NEGATIVE_RESULT_LOCAL_RECOVERY_NOT_TARGET_DESCENT`.

Unique factor supports and multiplicities are:

```text
[1,2,4]   x4
[1,5,11]  x1
[3,5]     x8
[2,4]     x3
[1,3,5]   x5
[1,2,5]   x1
```

The branch bank currently covers only factor columns
`[1,2,3,4,5,11]`.  The uncovered columns are
`[0,6,7,8,9,10,12,13,14,15]`.

## Direct-source bridge

The latest direct-source bridge is stronger than the latest duplicate-hit
branch relation as a rank-frontier clue:

- Artifact:
  `low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_6000_6007_probe.json`.
- Transfer `6003`, rank `3`, public-key verified.
- Direct cost: `0.75912409x` rho.
- Accepted form supports: `[0,5]` and `[3,5]`.
- The factor-rank candidate scorer reports this direct certificate as
  `MARGINAL_FACTOR_RANK_GAIN`: rank gain `1`, unique factor-relation gain `3`,
  deficiency delta `-1`, against its broader static/direct order-`11779`
  baseline.

The paired target-column scout emits concrete work orders:

- Order `11779`: target unseen/form-only priority column `[15]`.
- Order `9887`: target unseen/form-only priority columns `[0,5,6,7,8]`.

## Rollup classification

The local rollup over `5480..6007` classifies the branch-family slices as:

```text
RANK_FRONTIER_GAIN                    2
LOCAL_RECOVERY_RANK_DEPENDENT         7
LOCAL_RECOVERY_RANK_FRONTIER_UNGRADED 1
NO_DUPLICATE_HIT_SUPPLY               5
BRANCH_HITS_WITHOUT_CERTIFICATES      3
```

## Interpretation

The duplicate-hit branch family remains a clean local relation assembler with
multiple below-rho local recoveries, but its current support family is saturated:
new cheap stops mostly repeat `{1,2,4}`, `{1,3,5}`, or `{3,5}` and do not move
the branch-only rank frontier.

The direct-source bridge points to the next useful objective: move
rank-novelty and missing-column targeting before branch setup.  In particular,
the branch bank has never covered column `0` and still leaves column `15`
uncovered; the direct bridge at `6003` touches column `0` in an accepted form
and the target-column scout explicitly asks for order-`11779` column `15`.

## Non-claims

- This is not a complete ECDLP algorithm or deployed-curve speedup.
- The branch-family bank is not target descent; it has no derivable factor
  variables and deficiency `11`.
- The direct-source bridge rank gain is measured in the broader static/direct
  candidate scorer baseline, not as a completed branch-only descent.
- Below-rho local recovery is only a precursor until source density, row
  generation, factor-rank growth, and fresh-target mapping beat Pollard rho end
  to end.

## Reproduction command

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_branch_family_frontier_rollup.py \
  --state-dir /Volumes/Volume/autolab/ecdlp_index_calculus_state \
  --start-min 5480 \
  --out ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json
```
