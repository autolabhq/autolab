# Experiment Result

Artifacts:

- Frozen evaluator:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_frozen_root_policy_evaluator.py`
- Frozen-policy output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_frozen_root_policy_train_le87_test_88_95.json`

Training inputs:

- 72-79 selector:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_all_public_leaf_plus_fresh_72_79.json`
- 72-79 pre-factor gate:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_all_public_leaf_plus_fresh_72_79.json`
- 80-87 selector:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total3_total4_verified_over_rho_80_87.json`
- 80-87 pre-factor gate:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_verified_over_rho_80_87.json`

Holdout inputs:

- 88-95 total3/4 selector:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total3_total4_preregistered_88_95.json`
- 88-95 total3/4 pre-factor gate:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total3_total4_fixed_selector_88_95.json`
- 88-95 total2 selector:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total2_preregistered_88_95.json`
- 88-95 total2 pre-factor gate:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_88_95.json`

Frozen static public policy result:

- eligible policy family: static public root-ordering policies only
- excluded from the frozen claim: learned policies, because the old selector
  outputs trained them inside each evaluated bank
- selected policy from <=87 gated rows:
  `summax_sage_low_constant_target_hash`
- training surfaces: 30
- training target counts: `22050.cf1@11731`: 15,
  `67.a1@9803`: 15
- train preserving: 30/30
- train false positives: 0
- train below rho by scan cost: 28/30
- train below rho by direct-root cost: 30/30
- train mean scan cost: 0.8123601 ops/rho
- train max scan cost: 1.056 ops/rho
- train mean direct-root cost: 0.74341606 ops/rho
- train max direct-root cost: 0.984 ops/rho

88-95 holdout under that frozen policy:

- holdout surfaces: 3
- holdout target counts: `22050.cf1@11731`: 2,
  `67.a1@9803`: 1
- preserving: 3/3
- false positives: 0
- below rho by scan cost: 2/3
- below rho by direct-root cost: 2/3
- mean scan cost: 0.79912409 ops/rho
- max scan cost: 1.12 ops/rho
- mean direct-root cost: 0.74906083 ops/rho
- max direct-root cost: 1.072 ops/rho

The miss:

- target: `67.a1@9803`
- transfer index: 90
- row key: `67.a1@9803:uniform:256:salt206`
- surface:
  `67.a1@9803|67.a1@9803:uniform:256:salt206|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:90:67.a1@9803`
- chosen factor: `sage_resultant_factor_9`
- chosen root: 3394
- chosen factor preserves the selected root pair
- evaluated roots before public zero: 4
- selector eval ops: 48
- root scan ops: 92
- scan cost: 1.12 ops/rho
- direct-root cost: 1.072 ops/rho

Diagnostic, not claim:

- post hoc best static policy on 88-95:
  `global_root_hash`
- post hoc static result: 3/3 preserving, 0 false positives, 3/3 below rho,
  max scan cost 0.928 ops/rho, max direct-root cost 0.88 ops/rho

External learned-prior sensitivity:

The evaluator also rebuilds 88-95 policy rows for learned policies using the
30 <=87 gated surfaces as the external train set. This avoids the old
within-bank learned-policy ambiguity, but the selection rule was not
preregistered before inspecting 88-95.

- train mean-cost all-policy rule selects:
  `learned_global_zero_root_prior`
- train summary for that rule: 27/30 below rho by scan cost, 29/30 by
  direct-root cost, 30/30 preserving, 0 false positives, mean scan cost
  0.67360973 ops/rho
- 88-95 external-train holdout for that rule: 3/3 below rho, 3/3 direct-root
  below rho, 3/3 preserving, 0 false positives, mean scan cost 0.8300146
  ops/rho, max scan cost 0.928 ops/rho
- best measured external-train learned policy on 88-95 by mean scan cost:
  `learned_zero_ensemble_prior`, also 3/3 below rho with mean scan cost
  0.81541606 ops/rho and max scan cost 0.928 ops/rho

Interpretation:

The strict frozen static policy test is a partial negative: the selected
policy keeps the algebraic recovery correct on every 88-95 gated surface, but
the 67.a1@9803 transfer-90 selector overhead pushes both scan and direct-root
accounting over rho.

The useful positive is narrower: externally trained zero-root priors appear to
repair the miss while still using only <=87 training rows. That is a
preregisterable next hypothesis for an unseen window, not a result to count as
already proven by 88-95.
