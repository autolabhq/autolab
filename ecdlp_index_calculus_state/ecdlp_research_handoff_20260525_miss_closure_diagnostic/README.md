# ECDLP FFE Public Quotient Miss-Closure Diagnostic - 2026-05-25

## Scope

This handoff audits the one surface missed by the current public FFE quotient
route ensemble:

`22050.cf1@11731|22050.cf1@11731:uniform:256:salt164|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:57:22050.cf1@11731`

The probe is diagnostic only. It uses public order policies and public support
checks for candidate selection; preservation labels are written only for audit
and summary.

## Artifacts

- Probe:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_public_quotient_miss_closure_probe.py`
- JSON result:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_quotient_miss_closure_probe.json`
- Baseline route ensemble:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_quotient_route_ensemble_probe.json`

Run command:

```sh
ECDLP_TASK_DIR=/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus \
PYTHONPATH=/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus:/Volumes/Volume/autolab/tasks/ecdlp_index_calculus \
PYTHONPYCACHEPREFIX=/tmp/autolab_pycache \
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_public_quotient_miss_closure_probe.py \
  --out /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_quotient_miss_closure_probe.json
```

## Result

The updated route ensemble is now the strongest clean public statement:

- surface universe: 18
- public route union below rho: 18
- selected preserving: 18
- selected false positives: 0
- remaining miss: 0
- selected route counts:
  - `direct_public_factor`: 9
  - `incremental_factor_index_support`: 9
- mean selected ops/rho: `0.69565937`
- max selected ops/rho: `0.952`

The new miss-closure diagnostic evaluated 28 split rows over 18 unique surfaces.
The best false-positive-free support-only policy is now:

- order policy: `low_factor_index`
- cost model: `incremental_selector_scan_all_hit`
- selected unique surfaces: 9
- selected preserving: 9
- selected false positives: 0
- mean ops/rho: `0.64557989`
- max ops/rho: `0.72992701`
- known route miss selected: yes, preserving, `0.60583942` ops/rho

Hybrid with the previous route ensemble:

- baseline route-ensemble selected surfaces: 17
- incremental support added surfaces: 1
- hybrid selected surfaces: 18 / 18
- hybrid preserving: 18
- hybrid false positives: 0
- added surface:
  `22050.cf1@11731|22050.cf1@11731:uniform:256:salt164|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:57:22050.cf1@11731`

The known miss has two support-accepted candidates in both
`holdout_transfer_57` and `rolling_to_transfer_57`:

- false candidate: `sage_resultant_factor_11`, factor index 11, predicted leaf 90
- preserving candidate: `sage_resultant_factor_2`, factor index 2, predicted leaf 79

Both have full-selector ranked-first all-hit cost `1.02189781` ops/rho. The
public low-factor-index incremental selector instead evaluates factors in
factor-index order and stops at factor 2, charging only 9 selector ops plus the
support/root work. That puts the preserving candidate at `0.60583942` ops/rho
and avoids the later false factor 11.

## Sensitivity Boundary

The earlier sensitivity modes still matter as guardrails:

- `fingerprint_table_all_hit` with `low_factor_index` or `low_predicted_leaf`
  selects the preserving miss at `0.54014599` ops/rho, but introduces one false
  positive.
- `ordered_first_accept_scan_selected_hit` or `ranked_first_selected_hit` with
  `low_factor_index` or `low_predicted_leaf` selects the preserving miss at
  `0.86861314` ops/rho, but also introduces one false positive.
- Under `current_rank`, the same sensitivity modes select the false miss
  candidate `sage_resultant_factor_11` first.

The main false-positive surface for those non-incremental sensitivity modes is:

`67.a1@9803|67.a1@9803:uniform:256:salt202|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:32:67.a1@9803`

Its only support-accepted candidate is `sage_resultant_factor_14`, factor index
14, predicted leaf 8. It is false under the audit labels, costs `1.136` ops/rho
under full-selector ranked-first all-hit, and only falls below rho under the
sensitivity cost models (`0.656` fingerprint-table all-hit or `0.984`
selected-hit). The incremental all-hit selector does not select it below rho.

## Next Work

1. Pre-register the incremental selector on future FFE quotient surfaces before
   claiming a general algorithmic improvement.
2. Stress the cost model: ensure factor-index ordering and early stopping do
   not rely on hidden preservation labels, and ensure unmatched factors are
   charged by public factor-monomial evaluation before the first accepted
   support candidate.
3. Continue searching for public algebraic invariants among accepted singleton
   support candidates, especially as a fallback if future surfaces introduce an
   early false support candidate.
