# P1436 Autoresearch Focus Harness V23 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R73: resultant-valuation grammar

Artifacts:

- `p1553_resultant_valuation_trace_grammar_r73.py`
- `p1553_resultant_valuation_trace_grammar_report_r73.json`
- `p1553_resultant_valuation_trace_grammar_gate_r73.md`
- `frozen_noncp_trace_circuit_grammar.json`
- `restricted_projector_trace_replay.json`
- `rank_two_sparse_convolution_control.json`
- `dyadic_joint_source_and_direct_cost_ledger.json`

Report SHA-256:

`00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915`

For R10's pair histograms `A` and `C`, R73 freezes

```text
R_(A,C)(Z) = product_(u in A,v in C)(Z-u*v).
```

The root valuation at `z` is exactly the requested multiplicative-convolution
coefficient, with occurrence multiplicity:

```text
ord_(Z=z) R_(A,C)
  = sum_u D_12(u) D_34(z/u).
```

All finite controls pass:

```text
blind count                         0
forced-positive count               1
zero-signature mutation count     304
adaptive child counts             1+0 = 1
duplicate-heavy coefficient         4
joint source recovered           true
```

The grammar misses the direct caps in every faithful realization:

```text
expanded parametric resultant state       B^4
specialized coefficient query             B^2
B requested coefficients                  B^3
actual pair/triple signature extension    B^3
```

This rejects only `resultant_valuation_v1`. It is not a lower bound for
quotient-algebra transducers, structured arithmetic circuits, or exact data
structures.

## Harness routing

Artifact:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v23.json`

SHA-256:

`dc6a0073085ab3c0fc03a0a9e5eb1c2aee7eff293f9c379da1902b23e60f1059`

Schema: `ecdlp.p1436_autoresearch_focus_report.v15`.

Nine hash-bound lanes are closed, adding:

```text
resultant_valuation_trace_grammar
```

The sole top frontier is now:

```text
s6_quotient_algebra_trace_transducer
```

It must operate from dyadic unary subproduct trees and pass R73's exact
duplicate-multiplicity coefficient batch, zero signatures, blind and positive
targets, adaptive children, and source replay. It must also evaluate the
actual S6 pair/triple pullback without first constructing a degree-`B^2`
target polynomial, degree-`B^4` parametric resultant, or `B^3` triple deck.

V23 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. All linked alphaXiv
autoresearch guidance checks pass: bounded critical set, deterministic
nonblocking ambiguity handling, peripheral deferral, and fully specified
selected experiments.

## Verification

- Full ECDLP task suite: 94 tests passed.
- Python compilation: four changed modules passed.
- R73 JSON source bindings: passed.
- Eight generated JSON artifacts parsed.
- R73 parent YAML parsed with `breakthrough=false`.
- V23 diagnostic-only and nonpromotion checks: passed.
- `git diff --check`: passed.
