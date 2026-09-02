# Experiment Result

Preregistration:

- `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_preregistration_176_183.json`

Primary artifacts:

- Stress: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_176_183_probe.json`
- Strict signature: `ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_176_183.json`
- Public candidate signature: `ecdlp_index_calculus_state/low_term_total2_candidate_signature_fixed_selector_176_183.json`
- Gate manifest: `ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_candidates_176_183.json`
- Sage factors: `ecdlp_index_calculus_state/ffe_sage_factor_total2_candidates_single_hit_root_176_183.json`
- Factor audit: `ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_candidates_single_hit_root_176_183.json`
- Root policy: `ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total2_candidates_176_183.json`
- Public assembly selector: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_selector_176_183.json`
- Assembly replay: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_replay_176_183.json`

Fresh stress:

- Best fixed policy: `fixed_global_cap3_ow1_hw3_lw0_sw0_cw0_aw0`.
- Strict verified below-rho leaf cases: 10.
- Strict positives are all on `22050.cf1@11731`, transfer 182.
- Best strict ops/rho: 0.70072993.
- Public below-rho candidate cases: 88.
- Best public below-rho unverified candidate: 0.32116788 ops/rho.

FFE/root component:

- Candidate signature materialized 48 surfaces.
- Pre-factor single-hit-root gate selected 3 surfaces.
- Selected surfaces: 2 on `22050.cf1@11731`, 1 on `67.a1@9803`.
- Sage factored 3/3 requested surfaces after rerunning with explicit live state sources.
- All 52 Sage factors were quadratic root hyperplanes.
- Each selected surface had one preserving factor.
- Root-scan below-rho: 3/3.
- Policy result: 3/3 preserving, 0 false positives, 3/3 scan/direct below rho.
- Policy scan ops/rho: min 0.48175182, mean 0.57066667, max 0.712.
- Direct ops/rho: min 0.46715328, mean 0.51410219, max 0.608.

Public assembly replay:

- Public selector retained 3 root surfaces and 7 source assemblies across 3 challenge groups.
- 3/7 selected source assemblies were strict verifier positives, but this was not used for selection.
- Full selected source-case replay verified 3/7 assemblies.
- Retained-root-only replay verified 0/7 assemblies.
- Retained-root-only relation count sum: 6.
- Retained-root-only max rank: 1.
- Same-challenge retained-root groups verified: 0/2.

Interpretation:

The fresh 176-183 test strengthens the FFE component but exposes a precise
assembly boundary.  Public single-hit-root anchors select true below-rho root
hyperplanes on both targets.  However, replaying only the retained root surface
leaves rank at 1, while the full public leaf assembly can reach rank 2 on the
strict positive cases.  The missing ingredient is therefore not another root
locator tweak; it is a public companion-row completion rule that keeps enough
independent rows around a retained root anchor.

Execution note:

The first stress command failed under Apple Python 3.9 because the live helper
uses `zip(..., strict=False)`.  The successful run used `/usr/local/bin/python3`
with `PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache`.  The first Sage subset
run used relative live-module defaults and missed all requested surfaces; the
successful run passed explicit `/Volumes/Volume/autolab/ecdlp_index_calculus_state`
bank/config/direct/transfer sources.
