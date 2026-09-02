# Experiment Result

Artifacts:

- Audit script:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_first_fall_linear_factor_audit.py`
- Audit output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_all_public_leaf_plus_fresh_72_79.json`
- Sage-factor source:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_all_public_leaf_plus_fresh_72_79.json`
- Charged public selector source:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_all_public_leaf_plus_fresh_72_79_charged_rollup.json`

Measured facts:

- surfaces audited: 28
- targets: `22050.cf1@11731`, `67.a1@9803`
- Sage factors audited: 526
- factors of shape degree 1 / 3 monomials / root-hyperplane: 526
- root-hyperplane failures: 0
- every surface factor count matches its known hit-root count: yes
- surfaces with at least one preserving factor: 28
- preserving factor count: 39

Public charged selector result joined back to the same surfaces:

- best policy: `summax_sage_low_constant_target_hash`
- surfaces covered: 28
- surfaces below rho: 28
- preserving surfaces: 28
- false positives: 0
- mean charged ops/rho: 0.78788738
- max charged ops/rho: 0.968
- mean evaluated factor count: 9.14285714

Target split:

| Target | Surfaces | Factors | Preserving factors | Mean factors/surface | Mean charged ops/rho | Max charged ops/rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `22050.cf1@11731` | 16 | 272 | 27 | 17.0 | 0.72080292 | 0.87591241 |
| `67.a1@9803` | 12 | 254 | 12 | 21.16666667 | 0.87733333 | 0.968 |

Held-out root-prior sanity check:

| Scope | Top N roots | Preserving root hits | Preserving hit rate | Candidate root hits | Candidate hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| target | 1 | 12/28 | 0.42857143 | 15/28 | 0.53571429 |
| target | 2 | 17/28 | 0.60714286 | 21/28 | 0.75 |
| target | 8 | 19/28 | 0.67857143 | 24/28 | 0.85714286 |
| global | 4 | 17/28 | 0.60714286 | 21/28 | 0.75 |
| global | 8 | 19/28 | 0.67857143 | 24/28 | 0.85714286 |

This says root identity alone has real transfer signal but does not fully solve
selection. A top-8 prior reaches only 19/28 preserving roots even though it
touches some candidate root on 24/28 surfaces. The next selector needs public
features of `r`, target/salt context, or factor fingerprints, not just global
root frequency.

Conclusion: the measured 72-79 bank supports a stronger first-fall statement
than the previous generic FFE quotient phrasing. The resultant factors are
exactly quadratic-root hyperplanes `c + r*b + r^2`; public selection is choosing
which root hyperplane preserves the selected root pairs. This is still not a
completed ECDLP algorithm, because fresh hit-stream generation and root
selection must be validated without relying on an already factored surface bank.
