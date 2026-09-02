# Experiment Result

Artifacts:

- Probe script:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/low_term_total2_fixed_selector_fresh_stress_probe.py`
- Stress output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_fixed_selector_80_87_probe.json`
- Empty signature output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fixed_selector_80_87.json`

The full retraining run for 80-87 timed out after 1200 seconds before producing
an output artifact. The fixed-selector helper avoided that calibration sweep and
completed bounded 80-87 scans.

Strict primary-selector run:

- 64 row cases scanned.
- 53 source-verified cases.
- 22 row-level verified cases.
- 0 low-term-support total-2 leaf-level verified cases.
- 0 below-rho leaf certificates.

Widened frozen-selector audit:

| Row selector | Row verified | Row below rho | Leaf verified | Leaf below rho | Best leaf ops/rho |
| --- | ---: | ---: | ---: | ---: | ---: |
| `target_cap3_ow0_hw1_lw0_sw0_cw0_aw0` | 22 | 0 | 0 | 0 | null |
| `global_cap3_ow1_hw3_lw0_sw0_cw0_aw0` | 16 | 0 | 2 | 0 | 1.224 |
| `target_cap1_ow1_hw3_lw0_sw0_cw0_aw0` | 1 | 1 | 0 | 0 | null |

The only verifier-backed leaf motif was above rho:

- target: `67.a1@9803`
- transfer index: 82
- top_k: 12
- row salts: 201 and 204
- leaf index: 10 on both rows
- rank: 3
- relation count: 3
- ops/rho: 1.224

The strongest cheap non-certificate was rank deficient:

- target: `22050.cf1@11731`
- transfer index: 80
- top_k: 16
- row salt: 165
- leaf indices: 65 and 79
- ops/rho: 0.35766423
- rank: 1
- relation count: 1
- public key verified: false

The signature aggregator found zero below-rho positive cases for 80-87:

- below_rho_case_count: 0
- target_count: 0
- FFE relation generation targets: none

## Total-3/Total-4 Rescue Follow-Up

Follow-up artifacts:

- Rescue stress output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_80_87_probe.json`
- Generalized total-k signature output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total3_total4_signature_fixed_selector_80_87.json`
- Host-runnable diagnostic FFE signature:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total3_total4_verified_over_rho_diagnostic_signature_80_87.json`
- Generalized signature script:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/low_term_totalk_signature_probe.py`

Leaf selectors tested:

- `mode_cost_low_term_support_total3`
- `mode_cost_low_term_support_total4`
- `mode_low_term_support_total3`
- `mode_low_term_support_total4`

The total-3/total-4 extension improved recall but did not cross the charged
rho boundary:

| Row selector | Row verified | Row below rho | Leaf verified | Leaf below rho | Best leaf ops/rho |
| --- | ---: | ---: | ---: | ---: | ---: |
| `target_cap3_ow0_hw1_lw0_sw0_cw0_aw0` | 22 | 0 | 0 | 0 | null |
| `global_cap3_ow1_hw3_lw0_sw0_cw0_aw0` | 16 | 0 | 8 | 0 | 1.304 |
| `target_cap1_ow1_hw3_lw0_sw0_cw0_aw0` | 1 | 1 | 0 | 0 | null |

The generalized signature aggregator applied the stricter promotion gate
(`below_rho`, `public_key_verified`, rank >= 2, relations >= 2):

- strict positive below-rho cases: 0
- verified-over-rho diagnostics: 8
- below-rho but unverified diagnostics: 50
- FFE relation generation targets: none

Best verified over-rho total-k motif:

- target: `67.a1@9803`
- transfer index: 80
- top_k: 4
- row salts: 204, 206, 205
- leaf index: 3 on all three rows
- rank: 2
- relation count: 2
- ops/rho: 1.304
- diagnostic surface IDs:
  - `67.a1@9803|67.a1@9803:uniform:256:salt204|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:80:67.a1@9803`
  - `67.a1@9803|67.a1@9803:uniform:256:salt205|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:80:67.a1@9803`
  - `67.a1@9803|67.a1@9803:uniform:256:salt206|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:80:67.a1@9803`

Best below-rho unverified total-k motif:

- target: `22050.cf1@11731`
- transfer index: 80
- top_k: 16
- row salt: 165
- leaf indices: 65, 79, 90
- rank: 1
- relation count: 1
- public key verified: false
- ops/rho: 0.3649635

The sandbox does not currently have a `sage` executable on PATH, so the
diagnostic surfaces were not factored here. The diagnostic signature deliberately
uses `positive_source: verified_over_rho`; it is an input for a Sage-capable
host run, not a proof of a below-rho certificate.

Conclusion: 80-87 falsifies the naive "same total-2 public leaf selector keeps
transferring" hypothesis, and the total-3/total-4 rank-rescue extension is also
negative under the same strict promotion gate. The 72-79 FFE result remains
useful, but this window does not add new verifier-backed below-rho surfaces.
