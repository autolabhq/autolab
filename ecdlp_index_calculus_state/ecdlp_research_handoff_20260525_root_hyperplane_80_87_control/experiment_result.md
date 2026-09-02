# Experiment Result

Artifacts:

- Sage factor output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total3_total4_verified_over_rho_80_87.json`
- First-fall audit:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total3_total4_verified_over_rho_80_87.json`
- Root selector output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total3_total4_verified_over_rho_80_87.json`
- Selector script:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_first_fall_root_hyperplane_selector_probe.py`
- Audit script:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_first_fall_linear_factor_audit.py`

Materialization note:

The first Sage run against local relative defaults produced zero surfaces
because the writable mirror did not contain the bank/config/direct/transfer
inputs.  The successful run used live sources under `/Volumes/Volume/autolab`
and wrote the result into the local worktree.

Sage factor facts:

- signature cases: 8
- verified cases: 8
- materialized surfaces: 6
- surfaces with nontrivial resultant factors: 6/6
- total Sage resultant factor candidates: 133
- surfaces with any preserving factor candidate: 6/6
- preserving root-scan below rho: 6/6
- preserving surface-FFE below rho: 6/6
- preserving full-remainder below rho: 0/6
- min preserving root-scan ops/rho: 0.432
- min preserving surface-FFE ops/rho: 0.44
- min full-remainder FFE ops/rho: 1.928

First-fall audit facts:

- audited surfaces: 6
- factors: 133
- root-hyperplane factors: 133
- factor shape counts: `deg1_mon3_root1 = 133`
- root-hyperplane failures: 0
- all surface factor counts match known hit roots: true
- preserving factors: 80
- mean factors per surface: 22.16666667
- mean preserving factors per surface: 13.33333333

Fixed selector facts:

- best public policy: `low_root_norm`
- selected-root-pair surfaces: 3/6
- public-zero-capable surfaces: 4/6
- vacuous preserving surfaces: 3/6
- public-zero recovered: 4/6
- chosen preserving: 4/6
- conservative scan below rho: 3/6
- conservative scan mean ops/rho: 0.938
- conservative scan max ops/rho: 1.096
- direct-root below rho: 4/6
- direct-root mean ops/rho: 0.852
- direct-root max ops/rho: 0.96

Diagnostic non-vacuous/public-zero gate:

- gate: `original_selected_root_pair_count > 0` and `public_zero_root_count > 0`
- gate type: ex post diagnostic, not yet a fresh preregistered materializer
- gated surfaces: 3/6
- fixed all-surface best policy on gate: `low_root_norm`
- conservative scan below rho on gate: 3/3
- direct-root below rho on gate: 3/3
- chosen preserving on gate: 3/3
- false positives on gate: 0
- vacuous preserving surfaces on gate: 0
- gated scan mean ops/rho: 0.88533333
- gated scan max ops/rho: 0.96
- gated direct-root mean ops/rho: 0.816
- gated direct-root max ops/rho: 0.888

Held-out prior facts:

- best held-out policy: `learned_salt_weighted_zero_prior`
- public-zero recovered: 4/6
- chosen preserving: 4/6
- conservative scan below rho: 2/6
- conservative scan mean ops/rho: 1.046
- conservative scan max ops/rho: 1.288
- direct-root below rho: 3/6
- direct-root mean ops/rho: 0.96
- direct-root max ops/rho: 1.152

Interpretation:

The structural first-fall claim transfers: every factor on the materialized
80-87 diagnostic surfaces is a quadratic root hyperplane and every materialized
surface has nontrivial factors.  This is useful because it says the FFE/Sage
route is exposing a stable algebraic object rather than a one-off artifact of
the 72-79 bank.

The selector claim does not yet transfer as a fresh algorithmic win.  Only four
of six materialized surfaces expose a public zero under the best fixed ordering,
only three have original selected root pairs, and three preserving labels are
vacuous.  The direct-root companion is still valuable: it turns the best fixed
policy from 3/6 below rho under scan charging into 4/6 below rho, with max
direct ops/rho 0.96 on the recovered public-zero surfaces.  But the missing
public-zero surfaces remain missing, so this control is evidence for the
factor-shape mechanism, not for a complete below-rho relation generator.

The new diagnostic gate sharpens the boundary.  Once surfaces without original
selected root pairs and surfaces without any public-zero-capable root are
removed, the same all-surface fixed policy is 3/3 below rho under both scan and
direct-root accounting with zero vacuous rows.  This is not a standalone claim
because the public-zero-capable half of the gate is currently measured after
factorization.  It is the next preregistration target: a fresh materializer must
produce non-vacuous, public-zero-capable 80+ surfaces before selector scoring.

Conclusion:

This run upgrades the root-hyperplane story from "72-79 factored-bank
candidate" to "shape transfers to an 80-87 diagnostic control."  The gated
slice shows the root-ordering cost is already below rho on all currently
eligible non-vacuous surfaces.  The next proof obligation is to generate fresh
80+ surfaces that are non-vacuous and public-zero-capable by construction, then
freeze a root ordering before checking public-zero recovery and direct-root cost
against rho.
