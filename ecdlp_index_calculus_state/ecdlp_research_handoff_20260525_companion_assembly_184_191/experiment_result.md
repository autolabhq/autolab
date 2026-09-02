# Experiment Result

Preregistration:

- `ecdlp_index_calculus_state/ffe_public_companion_assembly_preregistration_184_191.json`

Primary artifacts:

- Stress: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_184_191_probe.json`
- Strict signature: `ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_184_191.json`
- Candidate signature: `ecdlp_index_calculus_state/low_term_total2_candidate_signature_fixed_selector_184_191.json`
- Gate: `ecdlp_index_calculus_state/ffe_preregistered_gate_manifest_total2_candidates_184_191.json`
- Sage factors: `ecdlp_index_calculus_state/ffe_sage_factor_total2_candidates_single_hit_root_184_191.json`
- Factor audit: `ecdlp_index_calculus_state/ffe_first_fall_linear_factor_audit_total2_candidates_single_hit_root_184_191.json`
- Root policy: `ecdlp_index_calculus_state/ffe_prefactor_unique_leaf_single_hit_root_policy_total2_candidates_184_191.json`
- Anchor selector: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_selector_184_191.json`
- Anchor replay: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_replay_184_191.json`
- Companion selector: `ecdlp_index_calculus_state/ffe_public_companion_assembly_selector_184_191.json`
- Companion replay: `ecdlp_index_calculus_state/ffe_public_companion_assembly_replay_184_191.json`

Fresh stress:

- Strict verified below-rho cases: 1.
- Strict positive target: `22050.cf1@11731`, transfer 189.
- Strict best ops/rho: 0.70072993.
- Public below-rho candidate cases: 98.
- Best public below-rho unverified candidate: 0.32116788 ops/rho.

FFE/root component:

- Candidate signature materialized 59 surfaces.
- Pre-factor single-hit-root gate selected 7 surfaces.
- Selected surfaces: 3 on `22050.cf1@11731`, 4 on `67.a1@9803`.
- Sage factored 7/7 requested surfaces.
- All 147 Sage factors were quadratic root hyperplanes.
- Each selected surface had one preserving factor.
- Root policy result: 7/7 preserving, 0 false positives, 7/7 scan/direct below rho.
- Policy scan ops/rho: min 0.51094891, mean 0.63999166, max 0.76.
- Direct ops/rho: min 0.46715328, mean 0.55449426, max 0.672.

Assembly replay:

- Anchor selector retained 7 anchor surfaces and 10 source cases across 5 challenge groups.
- Companion selector retained 13 surfaces: 7 anchors plus 6 companion surfaces.
- Companion replay verified 1/10 source cases.
- Companion same-challenge groups verified: 1/5.
- Verified group: `22050.cf1@11731`, transfer 189, relation count 2, rank 2.
- Verified case rows: salts 166 and 171, leaf 90 on both rows.
- `67.a1@9803` retained groups remained rank 1 with no public-key derivation.

Anchor-only comparison:

- Anchor-only replay also verified the `22050.cf1@11731` transfer-189 case.
- Difference from 176-183: in this fresh window both independent rows in the
  verified case were themselves retained-root anchors, so the companion rule did
  not need to rescue a dropped non-anchor row.

Interpretation:

The 184-191 run is a fresh public-selector positive for the retained-root
assembly path on `22050.cf1@11731`: root anchors selected before verifier
replay produced a same-challenge rank-2 public-key derivation below rho.  The
same machinery still fails on `67.a1@9803`, where selected anchors and
companions repeatedly produce only one unique relation form.  This narrows the
next problem to public independent-row expansion for the harder target, not
FFE factor shape or root locator correctness.
