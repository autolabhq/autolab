# Experiment Result

Artifacts:

- Probe script:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_first_fall_root_hyperplane_selector_probe.py`
- Probe output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_all_public_leaf_plus_fresh_72_79.json`
- First-fall audit input:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_all_public_leaf_plus_fresh_72_79.json`
- Sage-factor source:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_all_public_leaf_plus_fresh_72_79.json`

Measured selector facts:

- surfaces: 28
- targets: `22050.cf1@11731`, `67.a1@9803`
- policies: 16 total, 8 static public orders and 8 held-out zero-root priors
- scan cost model: evaluate root hyperplane candidates until first public zero, then add selected candidate root-scan ops and compare with generic Pollard-rho steps
- direct-root companion: replace selected candidate root-scan ops with `surface_ffe_ops` plus direct validation of the hyperplane root `r`
- best policy: `summax_sage_low_constant_target_hash`
- selected-root-pair surfaces: 27/28
- public-zero-capable surfaces: 28/28
- vacuous preserving surfaces: 1/28
- public-zero recovered: 28/28
- chosen preserving: 28/28
- false positives: 0
- below rho: 28/28
- mean evaluated roots: 9.14285714
- mean selector eval ops: 28.28571429
- mean total ops/rho: 0.78788738
- max total ops/rho: 0.968
- min total ops/rho: 0.5620438
- direct-root below rho: 28/28
- direct-root mean total ops/rho: 0.72040459
- direct-root max total ops/rho: 0.848
- direct-root mean ops saved versus scan: 8.78571429

Diagnostic non-vacuous/public-zero gate:

- gate: `original_selected_root_pair_count > 0` and `public_zero_root_count > 0`
- gate type: ex post diagnostic reporting slice
- gated surfaces: 27/28
- same fixed all-surface best policy on gate: `summax_sage_low_constant_target_hash`
- gated below rho: 27/27
- gated direct-root below rho: 27/27
- gated vacuous preserving surfaces: 0
- gated mean total ops/rho: 0.79625196
- gated max total ops/rho: 0.968
- gated mean direct-root ops/rho: 0.72735118
- gated max direct-root ops/rho: 0.848

Target split for best public policy:

| Target | Surfaces | Below rho | Preserving | Public zero recovered |
| --- | ---: | ---: | ---: | ---: |
| `22050.cf1@11731` | 16 | 16 | 16 | 16 |
| `67.a1@9803` | 12 | 12 | 12 | 12 |

Held-out prior result:

- best held-out policy: `learned_global_zero_root_prior`
- public-zero recovered: 28/28
- chosen preserving: 28/28
- below rho: 26/28
- mean evaluated roots: 2.85714286
- mean selector eval ops: 9.0
- mean total ops/rho: 0.64366215
- max total ops/rho: 1.096
- direct-root below rho: 27/28
- direct-root mean total ops/rho: 0.57617935
- direct-root max total ops/rho: 1.0

Interpretation:

The static public fingerprint order remains the robust certificate on the
measured bank: it recovers a preserving public-zero root on every surface and
stays below rho even with the conservative selected-factor root-scan charge.
One of those preserving labels is vacuous because the surface has no original
selected root pair; the non-vacuous selected-root-pair count is therefore 27/28.
The direct-root companion lowers the best-policy tail from 0.968 to 0.848.  The
held-out root prior is interesting because it is much cheaper on average, but
it still misses the scan rho bound on two `67.a1@9803` transfer-55 surfaces and
reaches exactly rho on one of them under the direct-root companion.

The per-transfer best-policy table in the JSON is an oracle upper bound, not a
validation result: it chooses the best policy after seeing each transfer's test
outcome.  It is useful only as a steering diagnostic.

Conclusion:

This strengthens the candidate algorithmic picture.  The FFE route now has a
measured public root-hyperplane selector with 28/28 below-rho charged wins on
the current factored bank under both conservative scan and direct-root
accounting.  It is still not a claimed ECDLP break: the next proof obligation
is fresh hit-stream generation and root selection without depending on an
already factored calibration bank.
