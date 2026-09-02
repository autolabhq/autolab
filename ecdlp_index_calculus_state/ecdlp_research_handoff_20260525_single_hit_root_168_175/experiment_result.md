# Experiment Result

## Fresh 168-175 Run

Preregistration:

- `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_single_hit_root_policy_preregistration_168_175.json`

Artifacts:

- Stress: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_168_175_probe.json`
- Signature: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_168_175.json`
- Gate: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_fixed_selector_168_175.json`
- Sage factors: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_total2_single_hit_root_preregistered_168_175.json`
- Factor audit: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_single_hit_root_preregistered_168_175.json`
- Strict policy eval: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total2_168_175.json`
- Broad unique-leaf control: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_hit_root_policy_total2_168_175.json`
- Broad selected-hit-root control: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_prefactor_hit_root_policy_total2_168_175.json`
- Cross-window bridge: `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_single_hit_root_relation_bridge_152_175.json`

Stress and signature:

- Total2 stress found 5 strict below-rho, public-key-verified cases.
- Targets: `22050.cf1@11731` and `67.a1@9803`.
- Best strict case: `22050.cf1@11731`, transfer 173, salt 165, leaves 65 and
  79, at 0.38686131 ops/rho.
- Best `67.a1@9803` strict case: transfer 174, salts 202 and 203, leaf 7,
  at 0.96 ops/rho.
- Strict signature produced 7 distinct surface IDs.
- Pre-factor gate selected all 7 surfaces.
- Every gate-selected surface had exactly one distinct pre-factor selected hit
  root, so the strict single-hit-root gate and broad all-prefactor gate select
  the same 7 surfaces in this window.

Sage factorization:

- Requested/factored selected surfaces: 7/7.
- Resultant factor candidates: 138.
- Preserving factor surfaces: 7/7.
- All factors were quadratic root hyperplanes.
- Preserving factor root-scan below rho: 7/7.
- Preserving direct/surface below rho: 7/7.
- Best preserving root-scan cost: 0.46715328 ops/rho.
- Full remainders remained above rho; best full remainder was 1.02919708
  ops/rho.

Primary policy result:

- `prefactor_unique_leaf_hit_root_first_target_hash` with
  `single_prefactor_hit_root`: 7/7 preserving, 0 false positives, 7/7
  scan-below-rho, 7/7 direct-below-rho.
- Mean evaluated root count: 1.0.
- Mean selector eval ops: 3.42857143.
- Scan cost: min 0.48905109, mean 0.58884672, max 0.728 ops/rho.
- Direct cost: min 0.46715328, mean 0.51782273, max 0.608 ops/rho.

Controls:

- Broad all-prefactor `prefactor_unique_leaf_hit_root_first_target_hash`:
  7/7 preserving, 0 false positives, 7/7 scan/direct below rho.
- Broad all-prefactor `prefactor_selected_hit_root_first_target_hash`:
  7/7 preserving, 0 false positives, 7/7 scan/direct below rho.
- The broad controls match the strict policy because there were no multi-root
  gate-selected surfaces in 168-175.

Interpretation:

This is the third fresh preregistered positive for the single-hit-root total2
component. Yield is lower than 160-167, but the mechanism is cleaner on cost:
the maximum policy scan ratio drops to 0.728. The repeated pattern across
152-175 is now strong enough that the next important proof obligation is not
another root locator tweak; it is assembling retained hyperplanes into an
end-to-end relation-derived ECDLP path or proving where that assembly fails.

## Cross-Window Bridge Start

The first bridge artifact joins retained single-hit-root hyperplanes from
152-159, 160-167, and 168-175 back to their verifier-backed source signature
cases.

Bridge summary:

- Retained surfaces: 20.
- Retained source cases: 30.
- Targets: `22050.cf1@11731` and `67.a1@9803`.
- False positives among retained surfaces: 0.
- All retained surfaces are scan-below-rho and direct-below-rho.
- Aggregate scan cost: min 0.48905109, mean 0.62122044, max 0.824 ops/rho.
- Aggregate direct cost: min 0.46715328, mean 0.53964379, max 0.72 ops/rho.

The bridge status is intentionally conservative:
`component_join_only_no_scalar_free_ecdlp_derivation_yet`. The next step must
build a verifier-facing relation assembly that proves whether these retained
hyperplanes give enough independent equations to derive the ECDLP secret
without returning a scalar.
