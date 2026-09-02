# P1436 autoresearch focus report

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

Natural full cells: exact `1/1`, full-rank `0/1`, verified logs `0/1`, below-rho `0/1`.

Stored-but-unusable / oracle-rank-headroom / fixed-rank-headroom cells: `1/1/1`.

Target-policy fixed fraction of oracle rank headroom: `0.5`. Prospective transfer: `positive_rank_transfer`.

Matched hash-policy specificity control: `matched_hash_controls_unavailable`.

Audit-bound promotion allowed: `False`.

Shoup pressure status: `insufficient_comparable_cells` (passes gate: False)
Shoup gate inputs: 0 exact+verified cells, 0 scales, threshold exponent < 0.5 and residual <= 1.0.

## Shoup Pressure

Scale coverage: `0` (required `2`), status: `insufficient_comparable_cells`.
Fitted exponent (log-cost vs log-bits): `None`; max abs residual log: `None`.

Method source: https://x.com/i/web/status/2076737985559822734?s=46
Tweet source captured: https://x.com/i/web/status/2076737985559822734 (text_included=True)
Tweet source summary: Introducing autoresearch with GPT 5.6

We had GPT 5.6 Sol reproduce the key findings from “Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Finetuning”

Compared to GPT-5.5 and even Fable 5, GPT-5.6 stayed more focused on a few, critical experiments and spent less time on peripheral details. It also asked fewer “clarification” questions and independently resolved ambiguities instead of pushing them back to us

@OpenAI pushing the boundaries of the automated research loop with models that don’t have handcuffs
Tweet source summary verbatim? True
Tweet intake mode: public_snapshot_summary

## Next Action
1. `collision_to_rank_routing_ablation`: Report all-edge, cross-shift-only, and within-shift-only rank and augmented-rank matrices.
   Decisive test: Replay all-edge, cross-shift-only, and within-shift-only matrices with exact sources.
   Falsifier: The natural all-edge matrix reaches full RHS-compatible rank.
   Required artifacts: routing_ablation.json, relation_matrices.json

## Focus Queue

1. `collision_to_rank_routing_ablation`: Report all-edge, cross-shift-only, and within-shift-only rank and augmented-rank matrices.
   Hypothesis: Stored collision edges are lost when compiled into independent factor rows.
   Falsifier: The natural all-edge matrix reaches full RHS-compatible rank.
2. `routing_intervention_generalization`: Freeze one routing intervention on development cells and require transfer to every prospective curve.
   Hypothesis: One fixed routing rule recovers natural-route rank on unseen curves.
   Falsifier: The fixed route fails to improve prospective rank or violates exact replay.

## Routing Controls

Matched curves / positive coordinate-specific excess: `0/0`.

P1436 configurations rerun the collector under different frozen routing rules; they do not patch an identical intermediate EC state. Their deltas are matched configuration ablations, not causal self-patching evidence.

## Autoresearch Guidance
Source: https://x.com/i/web/status/2076737985559822734?s=46 (canonical https://x.com/i/web/status/2076737985559822734)
Bounded critical set: True - Keep critical experiments bounded to the smallest unresolved bottlenecks.
Deterministic non-blocking ambiguity handling: True - Record non-blocking ambiguities as deterministic resolutions and keep them diagnostic-only.
Peripheral scope deference: True - Defer peripheral branches until bounded critical queue is exhausted or falsified.
Operator interrupt policy: True - blocking uncertainty requires operator action before any promotion.

## Guidance Compliance
Bounded critical set enforced: True
Peripheral scope deferral enforced: True
Non-blocking ambiguity resolutions recorded: True
Operator interrupt alignment: True
Selected focus candidates fully specified: True

## Major Result Replication
Fully replicated natural cells: 0/1
Per-stage status counts: {'replay_and_exact_residual': {'passed': 1}, 'collision_supply': {'passed': 1}, 'cross_shift_routing': {'passed': 1}, 'source_row_elimination': {'passed': 1}, 'relation_rank': {'blocked': 1}, 'rhs_compatibility': {'blocked': 1}, 'factor_log_verification': {'blocked': 1}, 'target_descent': {'blocked': 1}, 'total_cost': {'blocked': 1}}

## Deferred Experiments

- `routing_specificity_control` (rank 3): outside_current_critical_experiment_budget
- `shoup_pressure_scaling_probe` (rank 4): outside_current_critical_experiment_budget

## Ambiguity Resolutions

- `self_patching_fidelity`: Label every delta a matched configuration ablation and prohibit causal self-patching language. Blocks promotion: `False`.
- `paper_headroom_scope`: Recompute headroom from P1436 rank fractions; never import the paper's numeric range as an ECDLP gate. Blocks promotion: `False`.
- `posthoc_oracle_scope`: Break score ties by configuration name and keep every oracle choice diagnostic-only. Blocks promotion: `False`.
- `synthetic_null_scope`: Use the synthetic stream for occupancy only; assign no rows, logs, or descent credit. Blocks promotion: `False`.
- `fixed_route_availability`: Fall back to the natural route and mark the fixed intervention unavailable. Blocks promotion: `False`.
- `target_descent_absence`: Treat absent descent as untested, never as success. Blocks promotion: `False`.
- `independent_audit_binding`: Withhold promotion. Blocks promotion: `True`.
- `source_claim_status`: Never synthesize a breakthrough claim in post-processing. Blocks promotion: `True`.

## Boundary

Intervention-selected improvements are diagnostic only. Promotion requires the natural preregistered route to pass exact relation supply, full RHS-compatible rank, verified factor logs, separate target descent, the P1436 cost/exponent gate, and an independent audit cryptographically bound to this exact source JSON.
