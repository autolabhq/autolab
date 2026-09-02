# Experiment Result

## Fresh 144-151 Run

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_hit_root_policy_preregistration_144_151.json`

Artifacts:

- Stress: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_144_151_probe.json`
- Signature: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_144_151.json`
- Gate: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_144_151.json`
- Sage factors: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_preregistered_144_151.json`
- Factor audit: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_preregistered_144_151.json`
- Root selector controls: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_root_hyperplane_selector_total2_preregistered_144_151.json`
- Primary policy eval: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_hit_root_policy_total2_144_151.json`
- Selected-hit-root control: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_hit_root_policy_total2_144_151.json`

Stress and signature:

- Total2 stress found 5 verified below-rho leaf cases under the best summary
  policy, with best summary cost 0.69343066 ops/rho.
- `target_cap1` also found the best individual stress cost, 0.38686131
  ops/rho.
- Strict signature produced 12 strict positive cases on `22050.cf1@11731`.
- Strict positives used transfer indices 149 and 150.
- Materialized surfaces: 5.
- Pre-factor gate-selected surfaces: 3.

Sage factorization:

- Requested/factored gate-selected surfaces: 3/3.
- Resultant factor candidates: 45.
- All factors were quadratic root hyperplanes.
- Preserving-factor surfaces: 2/3.
- Preserving root-scan below rho: 2/3.
- Preserving direct/surface below rho: 2/3.
- Full remainders remained above rho; best full remainder was 1.13138686
  ops/rho.

Primary policy result:

- `prefactor_unique_leaf_hit_root_first_target_hash`:
  2/3 preserving, 1 false positive, 2/3 scan-below-rho, 2/3
  direct-below-rho.
- The selected-hit-root control had the same 2/3 result.
- Static and learned selector controls also false-positived on the same bad
  surface.

Failure:

- Bad surface:
  `22050.cf1@11731|22050.cf1@11731:uniform:256:salt174|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:150:22050.cf1@11731`
- Pre-factor selected hit roots: `[1976, 3828]`.
- Both roots came from separate selected leaves, so the unique-leaf tie-break
  could not distinguish them.
- Sage factorization found no preserving factor candidate for this surface.
- The policy chose root `3828`, a public-zero false positive, at 0.61313869
  scan ops/rho and 0.60583942 direct ops/rho.

Interpretation:

The all-prefactor-gate 144-151 claim failed. This is not just a root-ordering
miss; the bad surface has no preserving Sage factor. The next gate must reject
multi-hit-root surfaces before policy scoring.

## Single-Hit-Root Gate Diagnostic

Code change:

- `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_prefactor_hit_root_policy_probe.py`

New diagnostic mode:

- `--gate-mode single_prefactor_hit_root`

Aggregate diagnostic artifact:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total2_112_151.json`

Result across total2 112-151:

- Surfaces kept by single-hit-root gate: 13.
- Preserving: 13/13.
- False positives: 0.
- Scan-below-rho: 13/13.
- Direct-below-rho: 13/13.
- Mean evaluated root count: 1.0.
- Mean scan cost: 0.66899495 ops/rho.
- Mean direct cost: 0.58507355 ops/rho.
- Max scan cost: 0.912 ops/rho.
- Max direct cost: 0.816 ops/rho.

Dropped surfaces:

- `67.a1@9803|67.a1@9803:uniform:256:salt208|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:137:67.a1@9803`
  - This was preserving and direct-below-rho but scan-over-rho under the wider
    unique-leaf policy.
- `22050.cf1@11731|22050.cf1@11731:uniform:256:salt174|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:150:22050.cf1@11731`
  - This was the new 144-151 false positive and had no preserving factor.

Interpretation:

The stricter gate trades recall for precision and restores a clean
scan-positive component on the measured total2 sequence. It is not yet a
fresh preregistered promotion because it was selected after seeing 144-151.
