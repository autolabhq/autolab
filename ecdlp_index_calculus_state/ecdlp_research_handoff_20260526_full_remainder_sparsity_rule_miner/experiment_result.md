# Full-Remainder Sparsity Rule Miner Result

Date: 2026-05-26

Status: candidate mechanism clue only. This does not promote a new ECDLP index-calculus algorithm yet.

## Artifacts

- Miner: `tasks/ecdlp_index_calculus/ffe_full_remainder_public_sparsity_miner.py`
- Output: `ecdlp_index_calculus_state/ffe_full_remainder_public_sparsity_miner_328_375.json`
- Inputs:
  - `fresh_328_343`: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_328_343_fixed_probe.json`
  - `fresh_344_359`: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_344_359_fixed_probe.json`
  - `fresh_360_375`: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_360_375_fixed_probe.json`

Command:

```bash
/Users/adamburan/.cache/vllm-venvs/vllm-312/bin/python tasks/ecdlp_index_calculus/ffe_full_remainder_public_sparsity_miner.py --artifact fresh_328_343:ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_328_343_fixed_probe.json --artifact fresh_344_359:ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_344_359_fixed_probe.json --artifact fresh_360_375:ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_fresh_360_375_fixed_probe.json --out ecdlp_index_calculus_state/ffe_full_remainder_public_sparsity_miner_328_375.json
```

## Result

The miner scanned 31 Sage surfaces: 29 had a preserving candidate and 2 were non-preserving. Only 2 surfaces beat Pollard-rho under the full-remainder FFE cost model, and both are the intermittent 79-monomial profile:

| window | transfer | row | candidate | selected leaves | known hit roots | full remainder monomials | ops/rho |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| fresh_328_343 | 342 | `22050.cf1@11731:uniform:256:salt165` | `sage_resultant_factor_0` | 3 | 12 | 79 | 0.96350365 |
| fresh_344_359 | 348 | `22050.cf1@11731:uniform:256:salt165` | `sage_resultant_factor_6` | 1 | 12 | 79 | 0.93430657 |

The best public pre-materialization rules isolate the two in-sample positives with no selected negatives:

| rule | selected | positives | negatives | recall |
| --- | ---: | ---: | ---: | ---: |
| `activate:row_salt_transfer_mod2=165|0` | 2 | 2 | 0 | 1.0 |
| `activate:row_salt_transfer_mod6=165|0` | 2 | 2 | 0 | 1.0 |
| `activate:target_row_salt_transfer_mod6=22050.cf1@11731|165|0` | 2 | 2 | 0 | 1.0 |

This should be treated as high-overfit evidence. It is public and freezeable, but it is based on two positives and abstains in `fresh_360_375` rather than proving a holdout win.

## Boundary

The exact row salt alone is not enough. The same target and `salt165` has preserving over-rho surfaces at transfer 353 and 357:

| transfer | full remainder monomials | ops/rho | known hit roots | selected leaves |
| ---: | ---: | ---: | ---: | ---: |
| 353 | 172 | 1.64233577 | 18 | 1 |
| 357 | 154 | 1.52554745 | 17 | 2 |

At transfer 363, the `salt165` surface is non-preserving. That makes the `fresh_360_375` result a boundary/demotion for broad full-remainder claims.

The closest negative preserving surface in the 31-record corpus is transfer 341, `salt173`, with 106 monomials and 1.16058394 rho. So the full-remainder win appears to need the 79-monomial collapse, not merely a small public factor surface.

## Mechanism Clue

The post-materialization diagnostics are perfectly aligned with the two positives:

- `full_remainder_monomials=79`
- `full_resultant_monomials=91`
- `known_hit_root_count=12`

Those cannot justify a public preselector after the fact, but they point to the likely mechanism: a summation-polynomial/FFE specialization where the quotient remainder collapses only for a narrow row-envelope and transfer residue class.

## Interpretation

The useful result is not "we have a speedup" yet. The useful result is a sharper, freezeable activation hypothesis:

```text
target = 22050.cf1@11731
row_salt = 165
transfer_index mod 6 = 0
```

This hypothesis must now be tested in a pre-registered targeted holdout. It should not be tuned further on the same 328-375 corpus before replay.
