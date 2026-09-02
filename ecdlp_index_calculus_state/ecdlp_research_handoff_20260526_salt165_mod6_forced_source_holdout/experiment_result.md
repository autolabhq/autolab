# ECDLP FFE exact-profile holdout result, 2026-05-26

## What changed

The earlier materialization failures for the `salt165` holdout were not a mathematical replay blocker. They were a source-path problem: the local state did not have the static bank, schedule config, direct witness, and transfer witness files needed to rebuild the exact row profiles. Re-running with the mounted campaign state under `/Volumes/Volume/autolab/ecdlp_index_calculus_state` materialized the profiles cleanly.

The exact-profile Sage wrapper now supports:

- `--profile-from-signature target|transfer_index|row_key`
- mounted `--bank-source`, `--config-source`, `--direct-source`, and `--transfer-source`
- JSON diagnostics even when materialization errors are allowed

The new cross-target branch analogue scanner ranks public selector cases before Sage materialization, joins compact static-bank metadata, excludes already materialized exact profiles, and emits exact-probe groups by target. This gives a bounded way to test whether the transfer-618/salt165 branch recurs elsewhere.

The public sparsity miner now supports `--forbid-atom-regex`, so leakage-prone atom families can be removed from rule generation without deleting the raw diagnostic atoms. This was used to audit whether the below-rho cases still have a public explanation after forbidding transfer identity/residue/distance atoms and then after also forbidding salt identity/residue/distance aliases.

The public sparsity miner also now supports `--label-mode`, so the strong full-remainder label can be mined separately from weaker factor-root-scan and surface-stage labels.

The coverage audit now uses the broader exact-artifact glob `ffe_sage_factor_exact_profiles_22050_salt165_*transfer*.json`, so the same report sees the original `mod6` probes, the `mod16` controls, the remaining forced-source transfer probes, transfer 618 neighbor controls, and adjacent transfer 619/620 controls.

## Evidence summary

Current audit artifact:

- `ecdlp_index_calculus_state/ffe_full_remainder_salt165_mod6_holdout_coverage_audit_376_671.json`

The refreshed audit sees:

- exact attempt artifacts: 18
- materialized exact attempts: 16
- materialized exact profiles: 164
- full-remainder below-rho preserving candidates: 17
- minimum preserving full-remainder ops/rho: 0.81021898
- selected public transfers in coverage: 378, 420, 438, 480, 540, 588, 594, 606, 612, 618, 630, 660

The materialized exact probes show:

| Probe | Profiles | Full-remainder below rho | Min full-remainder ops/rho | Main read |
| --- | ---: | ---: | ---: | --- |
| transfer 378, salt165 | 4 | 0 | 1.13138686 | Materializes, but full remainder is above rho. |
| transfer 420, salt165 | 8 | 6 | 0.99270073 | Real preserving below-rho full-remainder pulse. |
| transfer 420, neighbor controls | 16 | 3 | 0.99270073 | Neighbor salt167 also wins, so salt165 is not unique. |
| transfer 438, salt165 | 16 | 0 | 1.57664234 | Demotes broad salt165/mod6. |
| transfer 480, salt165 | 4 | 0 | 1.35766423 | Materialized negative control. |
| transfer 540, salt165 | 8 | 0 | 1.71532847 | Materialized negative control. |
| transfer 588, salt165 | 8 | 0 | 1.71532847 | Materialized negative control. |
| transfer 594, salt165 | 8 | 0 | 1.20437956 | Materialized negative control. |
| transfer 606, salt165 | 16 | 0 | 1.44525547 | Materialized negative control. |
| transfer 612, salt165 | 8 | 0 | 1.75182482 | Same transfer mod16 as 420, but no full-remainder win. |
| transfer 616, local controls | 12 | 0 | 1.32116788 | Same 616-623 selector window; salts 161, 162, and 176 are negative. |
| transfer 618, salt165 | 8 | 8 | 0.81021898 | Strongest current pocket: 67-monomial full remainders. |
| transfer 618, neighbor controls | 44 | 0 | 1.09489051 | Salts 166, 171, 174, and 177 do not reproduce the win. |
| transfer 619, salt165 | 4 | 0 | 1.44525547 | Adjacent-transfer negative; 154-monomial full remainder. |
| transfer 620, salt165 | 4 | 0 | 2.05109489 | Adjacent-transfer negative; 232-monomial full remainder. |
| transfer 630, salt165 | 4 | 0 | 1.44525547 | Materialized negative control. |
| transfer 660, salt165 | 4 | 0 | 1.57664234 | Same transfer mod16 as 420, but no full-remainder win. |
| transfer 298, salt165 literal branch | 4 | 0 | 2.34306569 | Scanner-selected `row_salt_transfer_mod32=165|10` historical analogue; negative. |
| transfer 362, salt165 literal branch | 4 | 0 | 2.21167883 | Strongest scanner-selected literal branch analogue; negative. |
| target `67.a1@9803`, transfer 359, salt204 | 32 | 0 | 1.512 | Cross-target low-term-span/static-bank analogue; negative for full remainder. |
| transfers 234/237, salts 173/165 | 36 | 0 | 1.57664234 | Requested from the scanner's next mixed-row branch group. The first local-default run failed because mounted source files were not passed; the mounted-source rerun materialized 36/36 profiles. Full remainder is negative, but all 36 are factor-root-scan and surface-stage below rho. |
| transfer 665, salts 161/166/169 | 48 | 0 | 1.86131387 | Fresh public-bounded 664-671 follow-up. The source selector looked below rho, but exact Sage materialization finds no full-remainder win and `verified_case_count=0`; all 48 remain factor-root-scan and surface-stage below rho. |
| transfers 677/673, salts 164/208 | 36 | 0 | 1.32116788 | Fresh public-bounded 672-679 follow-up. The frozen `target_cap1` source selector has verified below-rho stress labels and improves the exact full-remainder floor, but still misses the full-remainder rho threshold; all 36 remain factor-root-scan and surface-stage below rho. |
| transfers 733/730, salts 166/204 | 36 | 0 | 2.272 | Fresh public-bounded 728-735 follow-up. The selector has many verifier-labeled source positives, but the exact full remainder is a strong negative; 34/36 remain factor-root-scan and surface-stage below rho. |

Current miner artifact:

- `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660.json`
- `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_preselector_holdout.json`
- `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_bank_schedule_preselector_holdout.json`
- `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_locality_preselector_holdout.json`
- `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_branch_scanner_controls_preselector_holdout.json`
- `ecdlp_index_calculus_state/ffe_cross_target_branch_analogue_scanner_salt165_transfer618_public_selectors.json`

Miner summary:

- records: 176
- preserving candidates: 173
- positive below-rho records: 17
- positive sources: `t420_salt165` = 6, `t420_neighbors` = 3, `t618_salt165` = 8
- positive 79-profile count: 0
- strongest positives are the transfer 618, salt165, 67-monomial full remainders with known-hit-root count 11
- secondary positives are the transfer 420, salt165/salt167, 92-monomial full remainders with known-hit-root count 13
- closest negative is also a 92-monomial preserving candidate at transfer 420, salt165, selected leaf count 2, ops/rho 1.00729927

The preselector-holdout miner adds public envelope features before Sage full-remainder scoring:

- selected leaf signature and count
- profile policy, leaf selector, and top-k
- source row selector and source ops/rho
- original selected root-pair count
- transfer/salt residues through mod 32
- static-bank row metadata when available
- row schedule metadata when available
- configured reference-distance atoms for transfers 420/618 and salts 165/167

The best all-data pre-materialization rule is:

```text
row_salt_transfer_mod32 = 165|10
```

It selects exactly the 8 transfer-618/salt165 positives in the current exact-profile dataset, with precision 1.0 and recall 0.470588. This is useful as a frozen branch selector, but not a promoted general selector. Leave-one-transfer and leave-one-source holdouts fail to recover the withheld positive pocket:

- train without transfer 618 -> best rule from transfer 420 selects 0/8 transfer-618 positives
- train without transfer 420 -> best rule from transfer 618 selects 0/9 transfer-420 positives
- train without source `t618_salt165` -> best transfer-420 rule selects 0/8 transfer-618 positives
- train without `t420_salt165` or `t420_neighbors` -> best transfer-618 rule selects 0 of those transfer-420 positives

Static-bank metadata was added from:

```text
/Volumes/Volume/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_probe.json
```

For the salt165 row, the bank metadata is `bank_best_filter_mode=low_term_span`, `bank_best_filter_top_k=12`, `bank_best_filter_ops_over_rho=0.46715328`, and `bank_source_window_label=heldout_164`. This explains why the row family is a plausible preselector seed, but it does not separate transfer 618 from other salt165 transfers. The same bank row metadata is present on the salt165 negatives at transfers 378, 438, 480, 540, 588, 594, 606, 612, 619, 620, 630, and 660. Schedule-source metadata did not join for these exact `uniform:256:salt165` rows in the current schedule artifact.

The locality-aware miner adds configured distance atoms from transfers 420 and 618 and salts 165 and 167. These atoms make the transfer-618 branch description explicit: `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0` is also a perfect current selector for the 8 transfer-618 positives. That is a locality diagnosis, not a transferable rule, because leave-one-transfer holdout still selects 0/8 withheld transfer-618 positives.

The cross-target scanner found 60 public bounded selector artifacts with 11,486 selector cases and 32,721 still-unmaterialized row/leaf candidates after excluding 155 exact profiles. Before the latest exact runs it found eight unmaterialized literal `row_salt_transfer_mod32=165|10` profiles; after materializing transfers 298 and 362, that literal queue is exhausted. Both literal branch probes were negative on full remainder. The best target-67 analogue at transfer 359/salt204 also failed the full-remainder criterion: 32/32 materialized, 0 below rho, min full-remainder ops/rho 1.512. It remains only a root-scan/surface-stage lead.

The expanded miner with the scanner-selected controls now has 216 records, 203 preserving candidates, 17 positives, and 13 nonpreserving surfaces. The previous `row_salt_transfer_mod32=165|10` rule is no longer a top perfect rule after adding transfers 298 and 362. The top pre-materialization rules are explicit transfer-618 locality descriptors such as `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`, which confirms that the scanner controls demote the residue rule rather than generalize it.

The transfer-free miner run is:

- `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_branch_scanner_controls_transfer_free_preselector_holdout.json`
- forbidden atoms: `^(transfer_|row_salt_transfer_mod|target_transfer_mod|target_row_salt_transfer_mod)`
- best pre-materialization rule: `row_salt=167 & selected_leaf_index_count=1`
- selected records: 3
- positives: 3
- negatives: 0
- recall: 0.176471

This is only the transfer-420/salt167 neighbor pocket. It does not recover the transfer-618/salt165 positives and fails held-out salt/source/transfer transferability.

The transfer-and-salt-free miner run is:

- `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_branch_scanner_controls_transfer_salt_alias_free_preselector_holdout.json`
- forbidden atoms: `^(transfer_|row_salt|target_row_salt|row_schedule_salt=|target_transfer_)`
- best pre-materialization rule: `profile_top_k=12 & selected_leaf_index_count=2`
- selected records: 11
- positives: 4
- negatives: 7
- precision: 0.363636
- recall: 0.235294

After transfer and salt aliases are removed, no clean public selector remains in the current feature set.

The first transfer-234/237 exact materialization attempt is:

- `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_cross_target_branch_22050_transfer234_237_salt173_165_forced_source.json`
- requested profiles: 36
- missing exact profiles in the selector: 0
- materialized exact profiles: 0
- materialization errors: 36
- dominant error: `candidate spec not found`

That was a command/source-path error: the local mirror does not contain the default relative bank/direct witness sources. The corrected mounted-source exact run is:

- `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_cross_target_branch_22050_transfer234_237_salt173_165_mounted_sources.json`
- requested profiles: 36
- materialized exact profiles: 36
- materialization errors: 0
- full-remainder below-rho preserving candidates: 0
- min preserving full-remainder ops/rho: 1.57664234
- factor-root-scan below-rho preserving candidates: 36
- min preserving factor-root-scan ops/rho: 0.33576642
- surface-stage below-rho preserving candidates: 36
- min preserving surface ops/rho: 0.34306569

An existing public linear-factor orientation audit for the same 232-239 selector window is:

- `ecdlp_index_calculus_state/ffe_public_linear_factor_xmatch_orientation_leaf79_factor1976_all_transfers_22050_232_239.json`
- verified measured below-rho rule count: 0
- min verified measured ops/rho: 1.01459854
- min verified shared-leaf-hit-root ops/rho: 0.98540146

The shared-leaf-hit-root value is an amortized charged model, not a measured direct replay cost. Treat it as a follow-up lead only.

After folding the mounted 234/237 exact run into the full-remainder miner:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_branch_scanner_controls_plus234237_preselector_holdout.json`
- records: 252
- preserving candidates: 239
- full-remainder positives: 17
- best full-remainder preselector: `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`

The new 234/237 profiles are full-remainder negatives and reinforce that transfer-618 remains a local pocket under the strong label.

The factor-root-scan miner is:

- artifact: `ecdlp_index_calculus_state/ffe_factor_root_scan_rule_miner_salt165_mod6_378_660_branch_scanner_controls_plus234237_preselector_holdout.json`
- label: `factor_root_scan_below_rho`
- records: 252
- factor-root-scan positives: 239
- full-remainder positives in the same dataset: 17
- best preselector: `selected_leaf_index_count=1`
- selected records: 193
- selected positives: 193
- selected negatives: 0
- recall: 0.807531

This is a broad weaker-stage selector. It transfers to held-out transfers 234 and 237, but it is not sufficient for a full-remainder ECDLP speedup because the full A/B remainder remains above rho on those profiles.

Fresh 664-671 public-bounded follow-up:

- exact artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer665_salt161_166_169_public_bounded_664_671_mounted_sources.json`
- requested profiles: 48
- materialized exact profiles: 48
- materialization errors: 0
- full-remainder below-rho preserving candidates: 0
- min preserving full-remainder ops/rho: 1.86131387
- factor-root-scan below-rho preserving candidates: 48
- min preserving factor-root-scan ops/rho: 0.33576642
- surface-stage below-rho preserving candidates: 48
- min preserving surface ops/rho: 0.34306569
- verified case count: 0

By row, the best exact full-remainder costs are 2.70072993 for salt161, 1.86131387 for salt166, and 2.70072993 for salt169. This is another factor-stage/surface-stage positive but a strong negative for relation-derived full-remainder recovery.

After adding transfer 665 to the full-remainder miner:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_665_branch_scanner_controls_plus234237_plus665_preselector_holdout.json`
- records: 300
- preserving candidates: 287
- full-remainder positives: 17
- factor-root-scan positives in the same dataset: 287
- surface-stage positives in the same dataset: 287
- full-remainder positive sources remain only `t420_salt165`, `t420_neighbors`, and `t618_salt165`
- best full-remainder preselector remains `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`

The transfer-and-salt-alias-free rerun with transfer 665 added is:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_665_branch_scanner_controls_plus234237_plus665_transfer_salt_alias_free_preselector.json`
- best preselector: `original_selected_root_pair_count=0 & selected_leaf_signature=8,90`
- selected records: 13
- positives: 4
- negatives: 9
- precision: 0.307692
- recall: 0.235294

The factor-root-scan rerun with transfer 665 added is:

- artifact: `ecdlp_index_calculus_state/ffe_factor_root_scan_rule_miner_salt165_mod6_378_665_branch_scanner_controls_plus234237_plus665_preselector.json`
- records: 300
- label positives: 287
- best preselector: `selected_leaf_index_count=1`
- selected records: 234
- selected positives: 234
- selected negatives: 0
- recall: 0.815331

The 665 result reinforces the split between cheap Sage factor/surface triage and actual ECDLP-useful full-remainder collapse.

Fresh 672-679 public-bounded follow-up:

- selector artifact: `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_672_679.json`
- source stress artifact: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_672_679_probe.json`
- exact artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer677_salt164_and_67_transfer673_salt208_public_bounded_672_679_mounted_sources.json`
- selector cases: 216 public-bounded cases, 12 verifier-labeled positives, selected min source ops/rho 0.39416058
- best source case: target `22050.cf1@11731`, transfer 677, salt164, policy `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, selected leaves `65,79,90`, public-key verified, relation count 2, rank 2
- alternate target case: target `67.a1@9803`, transfer 673, salt208, selected source ops/rho 0.424, not verifier-labeled positive
- requested profiles: 36
- materialized exact profiles: 36
- materialization errors: 0
- verified case count: 4
- full-remainder below-rho preserving candidates: 0
- min preserving full-remainder ops/rho: 1.32116788
- factor-root-scan below-rho preserving candidates: 36
- min preserving factor-root-scan ops/rho: 0.33576642
- surface-stage below-rho preserving candidates: 36
- min preserving surface ops/rho: 0.34306569

The best verified `target_cap1` source row does not survive the full-remainder test: the total3 profile `65,79,90` exact-checks at 1.41605839 ops/rho, and the total4 profile `8,65,79,90` exact-checks at 1.43065693 ops/rho. The global minimum 1.32116788 comes from singleton leaf `79` profiles on the same transfer/row, not from the strongest verified source case. This is still a useful improvement over transfer 665, but it remains a controlled negative for a full ECDLP speedup.

After adding 672-679 to the full-remainder miner:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_672679_branch_scanner_controls_plus234237_plus665_plus672679_preselector_holdout.json`
- records: 336
- full-remainder positives: 17
- positive sources remain only `t420_salt165`, `t420_neighbors`, and `t618_salt165`
- best full-remainder preselector remains `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`
- selected records: 8
- selected positives: 8
- selected negatives: 0
- recall: 0.470588

The transfer-and-salt-alias-free rerun with 672-679 added is:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_672679_branch_scanner_controls_plus234237_plus665_plus672679_transfer_salt_alias_free_preselector.json`
- best preselector: `original_selected_root_pair_count=0 & selected_leaf_signature=8,90`
- selected records: 13
- positives: 4
- negatives: 9
- precision: 0.307692
- recall: 0.235294

The factor-root-scan rerun with 672-679 added is:

- artifact: `ecdlp_index_calculus_state/ffe_factor_root_scan_rule_miner_salt165_mod6_378_672679_branch_scanner_controls_plus234237_plus665_plus672679_preselector.json`
- records: 336
- label positives: 323
- best preselector: `selected_leaf_index_count=1`
- selected records: 258
- selected positives: 258
- selected negatives: 0
- recall: 0.798762

The 672-679 result therefore strengthens the weak-stage triage corpus while narrowing the full-remainder interpretation: public source cost, verifier-labeled stress success, and target-cap1 one-row selection are not sufficient for full-remainder collapse.

Fresh 728-735 public-bounded follow-up:

- selector artifact: `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_728_735.json`
- source stress artifact: `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_728_735_probe.json`
- exact artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer733_salt166_and_67_transfer730_salt204_public_bounded_728_735_mounted_sources.json`
- selector cases: 239 public-bounded cases, 38 verifier-labeled positives, selected min source ops/rho 0.39416058
- best source case: target `22050.cf1@11731`, transfer 733, salt166, policy `fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0`, selected leaves `8,56,90`, public-key verified, relation count 2, rank 2
- alternate target case: target `67.a1@9803`, transfer 730, salt204, selected source ops/rho 0.528, relation count 1, rank 1
- requested profiles: 36
- materialized exact profiles: 36
- materialization errors: 0
- verified case count: 5
- full-remainder below-rho preserving candidates: 0
- min preserving full-remainder ops/rho: 2.272
- factor-root-scan below-rho preserving candidates: 34
- min preserving factor-root-scan ops/rho: 0.33576642
- surface-stage below-rho preserving candidates: 34
- min preserving surface ops/rho: 0.34306569

This is a stronger negative than 672-679. The best verified `22050` source case stays above rho under exact Sage full-remainder accounting, and the alternate `67.a1` target is worse: its best preserving full-remainder profile costs 2.272 ops/rho with 232 remainder monomials and 253 resultant monomials. It still contributes useful weak-stage examples, but it argues against promoting verifier-labeled target-cap1 source success as a full-remainder predictor.

After adding 728-735 to the full-remainder miner:

- the original verbose full-holdout artifact failed to write locally with `No space left on device` after the rule computation completed
- compact full-holdout artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_preselector_holdout_compact.json`
- compact output omits repeated embedded `selected_records` samples from rule and holdout blocks while preserving counts, source labels, precision, recall, and the top-level compact record ledger
- no-holdout artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_preselector_no_holdout.json`
- records: 372
- full-remainder positives: 17
- positive sources remain only `t420_salt165`, `t420_neighbors`, and `t618_salt165`
- best full-remainder preselector remains `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`
- selected records: 8
- selected positives: 8
- selected negatives: 0
- recall: 0.470588
- compact holdout entries: 25 transfer-index splits, 15 row-salt splits, 24 source-label splits
- compact holdout result: no top-5 trained rule hits any held-out full-remainder positive under transfer-index, row-salt, or source-label holdout

That last point is the sharper outcome: the in-sample public rule is still a clean locator for the already-known `t618_salt165` pocket, but it should not be promoted as a cross-pocket preselector. The next search should either add a genuinely new public feature that separates full-remainder collapse from the broad factor/root-scan signature, or treat the transfer-420 and transfer-618 pockets as separate branch families.

Enriched pre-materialization feature pass:

- miner upgrade: added selected-leaf shape atoms (`min`, `max`, `span`, `sum`, gap signature, parity signature, sum residues) and static-bank source-provenance atoms (source selector family/hash/seed, source-window kind, artifact stem, guard, guarded-selector count)
- identity filters used for the new rule search: forbid `transfer`, forbid `row_salt|row_schedule_salt`, forbid exact source-window salt labels/starts/offsets, and forbid raw `selected_leaf_signature`/`selected_leaf_has`
- clause-2 result: best rule `original_selected_root_pair_count=0 & selected_leaf_gap_max=82` selects 13 records, 4 positives, 9 negatives, all positives from `t618_salt165`
- clause-3/support-4 result: best rule `bank_best_filter_top_k=12 & original_selected_root_pair_count=0 & profile_top_k=12`
- clause-3 selected records: 16
- clause-3 selected positives: 14
- clause-3 selected negatives: 2
- clause-3 precision: 0.875
- clause-3 recall: 0.823529
- selected positive sources: `t420_salt165` and `t618_salt165`
- selected transfers: 420 and 618
- strict compact holdout artifact: `ecdlp_index_calculus_state/ffe_full_remainder_enriched_public_shape_source_miner_378_728735_transfer_salt_rawleaf_free_clause3_support4_holdout_compact.json`
- leave-one-transfer rediscovery: holding out transfer 420 trains on `t618_salt165` and selects 6/9 held-out transfer-420 positives with 2 negatives; holding out transfer 618 trains on `t420_salt165` and selects 8/8 held-out transfer-618 positives with 0 negatives
- leave-one-source rediscovery: holding out `t420_salt165` selects 6/6 held-out positives with 2 negatives; holding out `t618_salt165` selects 8/8 held-out positives with 0 negatives
- leave-one-row-salt result: still no held-out positives, so this is not yet a salt-general rule
- fixed-rule group audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_fixed_rule_topk12_root0_profile12_group_audit.json`

This is the first stronger second-stage feature in the salt165 branch: the same clause-3 rule is rediscovered across leave-one-transfer and leave-one-source folds while transfer, row-salt, exact source-window salt labels, and raw selected-leaf identity are forbidden. It still selects only row salt 165 in the current corpus and misses the salt167 transfer-420 neighbor positives, so it is a candidate branch feature, not a general full-remainder predictor yet.

A fresh exact-materialization queue was added for the public shell `bank_best_filter_top_k=12 & profile_top_k=12`, with `original_selected_root_pair_count=0` kept as a post-materialization predicate:

- script: `tasks/ecdlp_index_calculus/ffe_full_remainder_candidate_rule_queue.py`
- queue artifact: `ecdlp_index_calculus_state/ffe_full_remainder_topk12_root0_candidate_queue.json`
- queue result triage artifact: `ecdlp_index_calculus_state/ffe_full_remainder_topk12_root0_queue_result_triage.json`
- queue result triage now emits row-diverse recommendations, so 22050/salt165 candidates are visible even when global ranking prefers target67/salt205
- bank top-12 rows: `22050.cf1@11731:uniform:256:salt165`, `67.a1@9803:uniform:256:salt181`, and `67.a1@9803:uniform:256:salt205`
- initial deduped fresh exact-surface candidates: 249
- current deduped fresh exact-surface candidates after excluding the first 136 queue batches: 0
- current candidate rows: none
- current queue exclusion sources: 249 materialized exact surface keys plus 100 previously mined surface keys from `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_transfer_salt_alias_free_preselector.json`
- already-materialized or previously mined source-profile rows skipped by the current queue: 886

The queue-closed public sparsity miner was tightened with threshold atoms for selected-leaf shape fields, while still forbidding transfer identity, row-salt aliases, source-window salt/start/offset aliases, and raw leaf identity:

- miner upgrade: `tasks/ecdlp_index_calculus/ffe_full_remainder_public_sparsity_miner.py` now emits `<=`/`>=` cut atoms for selected-leaf min, max, span, sum, and gap min/max
- strict threshold audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_public_sparsity_miner_leaf_thresholds_rawleaf_strict_no_holdout.json`
- records: 251
- full-remainder positives: 4
- factor-root-scan and surface-stage positives: 245
- best mined pre-materialization rule: `selected_leaf_min<=12:False & source_ops_over_rho<=1.0:False`
- best mined rule result: 9 selected, 1 positive, 8 negatives, precision 0.111111, recall 0.25
- best recall-0.5 family: `original_selected_root_pair_count=1 & selected_leaf_min<=12:False`, selecting 2 positives and 9 negatives
- no mined recall-1 rule appears in the top threshold audit output
- direct structural proxy `original_selected_root_pair_count=1 & selected_leaf_max>=80` catches all four queue positives, but also catches 30 negatives
- attempted `--max-clause-size 3 --min-positive-atom-support 2` rerun did not leave an output artifact, so it is not evidence

This demotes the raw-leaf-free threshold path as a promotion rule. It does, however, give a concrete next work item: add a bounded/manual candidate-rule evaluator or an optimized clause search so suspected structural proxies can be scored without another expensive blind clause-3 sweep.

That bounded evaluator is now implemented in the same miner:

- new CLI: `--evaluate-rule`, with optional `pre:`, `factor:`, or `diagnostic:` stage prefixes
- new CLI: `--skip-mining`, so hand-written rules can be scored without enumerating mined clauses
- manual structural-rule audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_manual_structural_rule_evaluation.json`
- manual audit mode: same 251 queue-closed records, same full-remainder label, same transfer/salt/source-window/raw-leaf identity filters, no mining
- best recall-1 manual proxy: `original_selected_root_pair_count=1 & selected_leaf_max>=80:True`, selecting all 4 positives but also 30 negatives
- best singleton-leaf branch proxy: `original_selected_root_pair_count=1 & selected_leaf_index_count=1 & selected_leaf_max>=80:True`, selecting 2 positives and 9 negatives
- best two-leaf branch proxies: `original_selected_root_pair_count=1 & selected_leaf_gap_max>=80:True` and `original_selected_root_pair_count=1 & selected_leaf_index_count=2 & selected_leaf_max>=80:True & selected_leaf_min<=12:True`, each selecting 2 positives and 19 negatives
- broad public shell plus high leaf remains too weak: `bank_best_filter_top_k=12 & profile_top_k=12 & selected_leaf_max>=80:True` selects all 4 positives but also 91 negatives

The manual evaluator therefore confirms that the obvious high-leaf/single-vs-two-leaf decomposition still does not isolate the 67/78/11 collapse before Sage full-remainder scoring. The next useful feature must distinguish the high-leaf false positives whose post-materialization remainders grow to 106/120/14, 137/153/16, 154/171/17, 191/210/19, and larger bands.

The miner now also supports an explicitly second-stage `pre_factor_stage` evaluation mode. This unions pre-materialization atoms with factor-stage atoms so a broad public selector can be paired with a cheap factor/root-scan gate before full-remainder scoring:

- code path: `--evaluate-rule pre_factor:<rule>`
- factor-stage upgrade: factor-root-scan op count and normalized op/rho cut atoms are now emitted in `factor_stage`
- pre/factor gate audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_pre_factor_gate_evaluation.json`
- same records/label/identity filters as the manual structural audit
- factor-root-scan cost alone is not enough: `factor_root_scan_ops_over_rho<=0.46:True` selects all 4 positives but also 166 negatives
- adding the public/root recovery shell is decisive in this closed corpus: `original_selected_root_pair_count=1 & factor_root_scan_ops_over_rho<=0.46:True` selects exactly the 4 positives and 0 negatives
- adding high-leaf to the same gate gives the same result: 4 positives, 0 negatives
- relaxing the scan threshold to `<=0.5` selects 4 positives and 2 negatives
- exact op-count split: scan ops 62 selects the two singleton positives, and scan ops 63 selects the two two-leaf positives
- grouped fixed-rule check: transfer 242 has 2 selected positives and 0 selected negatives; transfer 294 has 2 selected positives and 0 selected negatives; each positive source-label group has 1 selected positive and 0 selected negatives
- selected-negative group count is 0 across transfer-index, source-label, and source-label-transfer slices for the fixed gate

This is a candidate two-stage gate, not a public preselector and not a proof of speedup. It is nevertheless useful: the closed queue now has a cheap way to distinguish the 67/78/11 full-remainder positives from the high-leaf false positives before doing the full A/B remainder calculation. The grouped fixed-rule check shows both positive transfers and all four positive source buckets are covered without selected negatives, but this is still the same closed corpus. The next validation should try the gate on a widened shell or independent exact-profile batch.

That widened-shell validation was run against the broader 372-record exact-profile corpus from `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_preselector_no_holdout.json`:

- broad pre/factor audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_broader_pre_factor_gate_evaluation.json`
- records: 372
- full-remainder positives: 17, split as 3 `t420_neighbors`, 6 `t420_salt165`, and 8 `t618_salt165`
- factor-root-scan cost alone remains far too broad: `factor_root_scan_ops_over_rho<=0.46:True` selects all 17 positives but also 210 negatives
- the closed-queue gate does not transfer: `original_selected_root_pair_count=1 & factor_root_scan_ops_over_rho<=0.46:True` selects 0 positives and 0 negatives, missing all 17 broader positives
- flipping the root-pair shell is not useful by itself: `original_selected_root_pair_count=0 & factor_root_scan_ops_over_rho<=0.46:True` selects all 17 positives but also 210 negatives
- the older enriched branch remains the useful broad-corpus router: `bank_best_filter_top_k=12 & profile_top_k=12 & factor_root_scan_ops_over_rho<=0.46:True` selects 14 positives and 2 negatives, precision 0.875 and recall 0.823529
- that top-k/profile/factor gate selects the `t420_salt165` and `t618_salt165` positives, but misses the 3 `t420_neighbors` positives because their `bank_best_filter_top_k` is null

This splits the current evidence into at least two branch families rather than one transferable rule: the closed top-k12/root0 queue has a clean `root_pair=1` second-stage gate, while the older 420/618 pockets use the `bank_best_filter_top_k=12 & profile_top_k=12` shell plus a low factor-root-scan cost. A promotable next step needs either a DNF/family evaluator that scores those branches together without using transfer/salt aliases, or a carefully justified promotion of root-scan metadata such as known-hit counts if it is truly available before full A/B remainder scoring.

The miner now has the first version of that family scorer:

- code path: `--evaluate-rule-family NAME=CLAUSE||CLAUSE`, where each clause accepts the same `pre:`, `pre_factor:`, `factor:`, and `diagnostic:` prefixes as `--evaluate-rule`
- broad family audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_broader_pre_factor_gate_family_evaluation.json`
- queue family audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_pre_factor_gate_family_evaluation.json`
- tested family: `pre_factor:original_selected_root_pair_count=1 & factor_root_scan_ops_over_rho<=0.46` OR `pre_factor:bank_best_filter_top_k=12 & profile_top_k=12 & factor_root_scan_ops_over_rho<=0.46`
- on the broader 372-record corpus, the family is identical to the broad branch: 14 positives, 2 negatives, 3 missed positives, because the queue branch selects nothing there
- on the 251-record closed queue, the family is much too broad: 4 positives and 166 negatives, because the broad branch degenerates to the same factor-root-scan-wide selection that was already known to be noisy

Two context-gate probes make the failure sharper:

- broad context probe artifact: `ecdlp_index_calculus_state/ffe_full_remainder_broader_pre_factor_context_gate_probe.json`
- queue context probe artifact: `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_pre_factor_context_gate_probe.json`
- adding `bank_surface_hit_row_count=7`, `bank_source_positive_guarded_selector_count=1`, `bank_source_selector_family=hash20:64:sketch_hit_row_count>=6`, or `bank_source_window_label_kind=heldout` does not change either corpus: it stays 14/2 on the broad corpus and 4/166 on the closed queue
- adding `source_row_selector=target_cap3_ow0_hw1_lw0_sw0_cw0_aw0` or the matching `profile_policy` narrows the broad corpus to 7 positives and 1 negative, but selects 55 closed-queue negatives and no closed-queue positives

So the DNF machinery is ready, but the obvious two-branch OR is not a router. The missing piece is a branch-context predicate that applies the top-k/profile/factor branch to the old 420/618 pockets without firing on the closed top-k12/root0 queue.

The next contrast used factor-stage candidate counts instead of bank provenance as the branch context:

- broad preserving-count family artifact: `ecdlp_index_calculus_state/ffe_full_remainder_broader_pre_factor_preserving_count_family_evaluation.json`
- queue preserving-count family artifact: `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_pre_factor_preserving_count_family_evaluation.json`
- combined preserving-count family artifact: `ecdlp_index_calculus_state/ffe_full_remainder_combined_pre_factor_preserving_count_family_evaluation.json`
- tested family:
  - queue clause: `original_selected_root_pair_count=1 & factor_root_scan_ops_over_rho<=0.46`
  - 618-style clause: `profile_top_k=12 & preserving_candidate_count=11 & factor_root_scan_ops_over_rho<=0.46`
  - 420-style clause: `profile_top_k=12 & preserving_candidate_count=13 & factor_root_scan_ops_over_rho<=0.46`
- on the broader 372-record corpus, this selects all 17 positives and 3 negatives, precision 0.85 and recall 1.0
- on the closed 251-record queue, this selects all 4 positives and 0 negatives, precision 1.0 and recall 1.0
- on the combined 623-record corpus, this selects all 21 positives and 3 negatives, precision 0.875 and recall 1.0
- clause decomposition on the combined corpus: the queue clause is 4/0; the `preserving_candidate_count=11` clause is 8/0 and covers `t618_salt165`; the `preserving_candidate_count=13` clause is 9/3 and covers `t420_salt165` plus `t420_neighbors`

A stricter 420 clause removes those 3 broad false positives:

- combined single-leaf preserving-count artifact: `ecdlp_index_calculus_state/ffe_full_remainder_combined_pre_factor_preserving_count_single_leaf_family_probe.json`
- refined 420 clause: `profile_top_k=12 & preserving_candidate_count=13 & selected_leaf_index_count=1 & factor_root_scan_ops_over_rho<=0.46`
- refined 420 clause score: 9 positives and 0 negatives, covering `t420_salt165` and `t420_neighbors`
- refined DNF score on the combined 623-record corpus: 21 positives and 0 negatives, precision 1.0 and recall 1.0
- source slices are all clean in this combined artifact: `t420_neighbors` 3/0, `t420_salt165` 6/0, `t618_salt165` 8/0, and the four closed-queue source buckets each 1/0

This is the best current family-shaped second-stage classifier, but it is not a public selector and not yet a work-saving algorithm. A timing audit of the exact-profile code shows why: `tasks/ecdlp_index_calculus/ffe_sage_factor_exact_profile_subset_probe.py` builds the hit remainder surface and records `full_remainder_ffe_ops_over_rho` before calling the mounted Sage `evaluate_surface`, and `/Volumes/Volume/autolab/tasks/ecdlp_index_calculus/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_probe.py` computes `candidate_count` and `preserving_candidate_count` inside Sage resultant factorization. Treat the preserving-count family as a post-materialization/factor-stage explanation unless an early-stop/stage-split probe proves those counts can be charged before the expensive full-remainder work we want to avoid.

That timing boundary is now machine-readable in a stage-split charge audit:

- code path: `tasks/ecdlp_index_calculus/ffe_full_remainder_stage_split_charge_audit.py`
- audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_combined_pre_factor_preserving_count_single_leaf_stage_split_charge_audit.json`
- replayed records: 623
- replayed positives: 21
- selected records: 21
- selected positives/negatives: 21/0
- rejected positives/negatives: 0/602
- selected candidate-remainder follow-up cost proxy: 18.89051097 ops/rho summed across selected records, only 0.0149915 of the all-record candidate-remainder sum 1260.08157688
- factor-root-scan on all records plus selected candidate-remainder follow-up: 286.29909447 ops/rho, 0.22720679 of all-record candidate-remainder follow-up
- factor-root-scan on all records plus selected surface-full-remainder follow-up: 286.29909447 ops/rho, 0.21966002 of all-record surface-full-remainder follow-up

The audit explicitly leaves exact surface materialization and Sage factorization uncharged, and flags both as required by the selector. The honest reading is: this family is an excellent post-factor classifier and a plausible work-order for an early-stop implementation, but the speedup claim still depends on measuring or avoiding the materialization/factorization stage.

The first live-source exact queue batch is:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_200_207_live_sources_batch4.json`
- profiles requested/materialized: 4/4
- materialization errors: 0
- target/row: `67.a1@9803`, `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 200 leaf 3, transfer 204 leaf 9, with equivalent source profiles deduped in the queue
- hidden root predicate: `original_selected_root_pair_count=0` for all 4 materialized surfaces
- preserving Sage factor-root-scan positives: 4/4
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- minimum full-remainder ops/rho: 2.272

This is useful but still below the promotion line: the public shell transferred to a new target/row and survived the hidden root-count predicate, and the factor-stage path is below rho, but the full A/B remainder is still above rho.

The second live-source exact queue batch is:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_208_215_live_sources_batch5.json`
- profiles requested/materialized: 5/5
- materialization errors: 0
- target/row: `67.a1@9803`, `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 209 leaves 6 and 6,7; transfer 210 leaves 11 and 7,11; transfer 215 leaf 5
- hidden root predicate: 2/5 have `original_selected_root_pair_count=0`; the transfer-210 and transfer-215 profiles have count 1 and therefore fail the root0 shell after exact materialization
- preserving Sage factor-root-scan positives: 5/5
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- minimum full-remainder ops/rho: 1.512

This improves the target67/salt205 exact negative floor from 2.272 to 1.512, but still does not produce a full-remainder below-rho candidate. It also confirms that the queue is a public shell, not a guarantee: `original_selected_root_pair_count=0` must be checked after materialization.

The third live-source exact queue batch is:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_216_223_live_sources_batch8.json`
- profiles requested/materialized: 8/8
- materialization errors: 0
- target/rows: `67.a1@9803:uniform:256:salt205` and `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: target67 transfer 218 leaves 10 and 2,10; target67 transfer 222 leaves 10 and 1,10; 22050 transfer 220 leaves 90 and 8,90; 22050 transfer 221 leaves 90 and 8,90
- hidden root predicate: 4/8 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 8/8
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- minimum full-remainder ops/rho: 1.75182482

The fourth live-source exact queue batch is:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_264_271_live_sources_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `67.a1@9803`, `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 269 leaf 6, transfer 270 leaf 9
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- minimum full-remainder ops/rho: 2.824

The fifth live-source exact queue batch is:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_272_279_live_sources_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `67.a1@9803`, `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 274 leaf 10, transfer 278 leaf 3
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- minimum full-remainder ops/rho: 2.104

The sixth live-source exact queue batch is:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_256_263_live_sources_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `67.a1@9803`, `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 257 leaf 9, transfer 258 leaf 2
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; transfer 258 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- minimum full-remainder ops/rho: 1.832

The seventh live-source exact queue batch is:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_312_319_live_sources_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `67.a1@9803`, `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 312 leaf 0, transfer 316 leaf 7
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- minimum full-remainder ops/rho: 2.104

The eighth live-source exact queue batch was chosen from the row-diverse triage view:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_288_295_rowdiverse_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 294 leaf 90, transfer 295 leaf 90
- transfer 294 is a fresh full-remainder below-rho exact surface: 67 full-remainder monomials, 78 full-resultant monomials, known-hit-root count 11, and 0.84671533 ops/rho
- transfer 294 selected one root pair and one linear root recovery at the factor stage
- transfer 295 is a paired negative control at 1.57664234 ops/rho
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 1/2
- minimum full-remainder ops/rho: 0.84671533

The ninth live-source exact queue batch checked the next 22050/salt165 row-diverse recommendation:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_384_391_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 389 leaf 90, transfer 391 leaf 90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.86131387

The tenth live-source exact queue batch checked another 22050/salt165 row-diverse follow-up:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_416_423_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 416 leaf 90, transfer 421 leaf 90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.57664234

The eleventh live-source exact queue batch checked the next 22050/salt165 global recommendation after the queue refresh:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_488_495_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 493 leaf 90, transfer 488 leaf 90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.09489051
- transfer 493 is a near miss: 106 full-remainder monomials, 120 full-resultant monomials, known-hit-root count 14, and 1.09489051 ops/rho

The twelfth live-source exact queue batch checked the next 22050/salt165 global recommendation:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_208_215_followup_batch1.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 209 leaf 90
- hidden root predicate: 1/1 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.44525547

The thirteenth live-source exact queue batch checked the next 22050/salt165 global recommendation:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_224_231_followup_batch1.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 226 leaf 90
- hidden root predicate: 1/1 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.09489051
- transfer 226 is another near miss: 106 full-remainder monomials, 120 full-resultant monomials, known-hit-root count 14, and 1.09489051 ops/rho

The fourteenth live-source exact queue batch checked the next 22050/salt165 global recommendation:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_256_263_followup_batch1.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 257 leaf 90
- hidden root predicate: 1/1 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.57664234

The fifteenth live-source exact queue batch checked the next 22050/salt165 global recommendation:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_248_255_followup_batch1.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 249 leaf 90
- hidden root predicate: 0/1 have `original_selected_root_pair_count=0`; the materialized surface has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.52554745
- minimum surface-stage ops/rho: 0.37956204
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.05109489

The sixteenth live-source exact queue batch checked a close transfer-294 locality analogue:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_296_303_followup_batch1.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 299 leaf 90
- hidden root predicate: 1/1 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.44525547

The seventeenth live-source exact queue batch checked the next 22050/salt165 global recommendation:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_312_319_followup_batch1.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 316 leaf 90
- hidden root predicate: 1/1 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.57664234

The eighteenth live-source exact queue batch checked the next 22050/salt165 global recommendation:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_320_327_followup_batch1.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 321 leaf 90
- hidden root predicate: 1/1 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.32116788

The nineteenth live-source exact queue batch checked two 22050/salt165 candidates closer to the transfer-618 window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_584_591_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 585 leaf 90, transfer 591 leaf 90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.34306569

The twentieth live-source exact queue batch checked the next two 22050/salt165 future-window queue recommendations:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_728_735_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 735 leaf 90, transfer 732 leaf 90
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; transfer 732 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.57664234

The twenty-first live-source exact queue batch checked the next 22050/salt165 carry-over candidate from selector 264-271:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_264_271_followup_batch1.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 269 leaf 90
- hidden root predicate: 1/1 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.86131387

The twenty-second live-source exact queue batch checked the next 22050/salt165 global recommendation:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_504_511_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 509 leaf 90, transfer 511 leaf 90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.57664234

The twenty-third live-source exact queue batch checked the next three 22050/salt165 global recommendations:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_592_599_followup_batch3.json`
- profiles requested/materialized: 3/3
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 595 leaf 90, transfer 596 leaf 90, transfer 599 leaf 90
- hidden root predicate: 1/3 have `original_selected_root_pair_count=0`; transfers 595 and 599 each have one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 3/3
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/3
- minimum full-remainder ops/rho: 1.71532847

The twenty-fourth live-source exact queue batch checked the next two 22050/salt165 global recommendations:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_240_247_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 245 leaf 90, transfer 242 leaf 90
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; transfer 242 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 1/2
- minimum full-remainder ops/rho: 0.84671533
- new positive detail: transfer 242 reproduces the 67-remainder-monomial, 78-resultant-monomial, known-hit-root-count-11 shape, but it fails the root0 post-materialization predicate; transfer 245 is the root0 half of the batch and remains above rho at 2.01459854 ops/rho

The twenty-fifth live-source exact queue batch checked the next two 22050/salt165 global recommendations:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_264_271_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 271 leaf 90, transfer 269 leaves 8,90
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; transfer 271 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.34306569
- minimum surface-stage ops/rho: 0.35766423
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.75182482

The twenty-sixth live-source exact queue batch checked the next two 22050/salt165 global recommendations:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_304_311_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 308 leaf 90, transfer 311 leaf 90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.71532847

The twenty-seventh live-source exact queue batch checked the next two 22050/salt165 global recommendations:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_352_359_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 354 leaf 90, transfer 354 leaves 8,90
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; the two-leaf surface has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.32116788
- transfer 354 is a mid-near control: 137 full-remainder monomials, 153 full-resultant monomials, known-hit-root count 16, and still above rho

The twenty-eighth live-source exact queue batch checked the next two 22050/salt165 global recommendations:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_392_399_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 397 leaf 90, transfer 397 leaves 8,90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.44525547

The twenty-ninth live-source exact queue batch checked the next two 22050/salt165 global recommendations:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_400_407_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 405 leaf 90, transfer 405 leaves 8,90
- hidden root predicate: 0/2 have `original_selected_root_pair_count=0`; both surfaces have one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.53284672
- minimum surface-stage ops/rho: 0.37956204
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.21167883
- transfer 405 is a strong negative for full-remainder promotion: 254 full-remainder monomials, 276 full-resultant monomials, known-hit-root count 22, and still above rho despite selected root-pair recovery at the factor stage

The thirtieth live-source exact queue batch checked the next two-leaf 22050/salt165 recommendations from the 416-423 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_416_423_twoleaf_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 421 leaves 8,90, transfer 416 leaves 8,90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.34306569
- minimum surface-stage ops/rho: 0.35766423
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.59124088
- transfer 416 two-leaf profile is a moderate negative at 172/190/18 and 1.59124088 ops/rho; transfer 421 two-leaf profile is worse at 277/300/23 and 2.35766423 ops/rho

The thirty-first live-source exact queue batch checked the next global 22050/salt165 recommendations from the 648-655 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_648_655_followup_batch3.json`
- profiles requested/materialized: 3/3
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 649 leaf 90, transfer 655 leaf 90, transfer 649 leaves 8,90
- hidden root predicate: 2/3 have `original_selected_root_pair_count=0`; transfer 655 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 3/3
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/3
- minimum full-remainder ops/rho: 1.71532847
- transfer 649 leaf-90 and two-leaf profiles share a 191/210/19 full-remainder shape at 1.71532847 and 1.72992701 ops/rho; transfer 655 is worse at 277/300/23 and 2.37956204 ops/rho despite selected root-pair recovery

The thirty-second live-source exact queue batch checked the next global 22050/salt165 recommendations from the 584-591 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_584_591_global_followup_batch3.json`
- profiles requested/materialized: 3/3
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 590 leaf 90, transfer 585 leaves 8,90, transfer 590 leaves 8,90
- hidden root predicate: 2/3 have `original_selected_root_pair_count=0`; transfer 585 two-leaf has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 3/3
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/3
- minimum full-remainder ops/rho: 1.20437956
- transfer 590 leaf-90 and two-leaf profiles share a 121/136/15 full-remainder near-control shape at 1.20437956 and 1.21897810 ops/rho; transfer 585 two-leaf is a strong negative at 352/378/26 and 2.94160584 ops/rho despite selected root-pair recovery

The thirty-third live-source exact queue batch checked the next global 22050/salt165 recommendations from the 328-335 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_328_335_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 334 leaf 90, transfer 334 leaves 8,90
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; the two-leaf surface has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.70072993
- transfer 334 is a decisive negative for full-remainder promotion: both leaf profiles expand to the same 326/351/25 full-remainder shape, with the two-leaf profile slightly worse at 2.75182482 ops/rho despite selected root-pair recovery

The thirty-fourth live-source exact queue batch checked the remaining two-leaf recommendations from the 384-391 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_384_391_twoleaf_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 389 leaves 8,90, transfer 391 leaves 8,90
- hidden root predicate: 0/2 have `original_selected_root_pair_count=0`; both two-leaf surfaces have one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.52554745
- minimum surface-stage ops/rho: 0.39416058
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.91240876
- transfer 391 two-leaf is a 211/231/20 full-remainder negative at 1.91240876 ops/rho; transfer 389 two-leaf is worse at 232/253/21 and 2.06569343 ops/rho

The thirty-fifth live-source exact queue batch checked the next global 22050/salt165 recommendation from the 464-471 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_464_471_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 464 leaf 90, transfer 464 leaves 8,90
- hidden root predicate: 2/2 have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.71532847
- transfer 464 leaf-90 and two-leaf profiles share a 191/210/19 full-remainder negative shape at 1.71532847 and 1.72992701 ops/rho

The thirty-sixth live-source exact queue batch checked the remaining two-leaf recommendations from the 488-495 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_488_495_twoleaf_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 493 leaves 8,90, transfer 488 leaves 8,90
- hidden root predicate: 0/2 have `original_selected_root_pair_count=0`; both two-leaf surfaces have one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.48175182
- minimum surface-stage ops/rho: 0.39416058
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.14598540
- transfer 493 two-leaf preserves the near-control 106/120/14 shape but worsens from the leaf-90 1.09489051 ops/rho to 1.14598540; transfer 488 two-leaf is 154/171/17 at 1.49635036 ops/rho

The thirty-seventh live-source exact queue batch checked the next global 22050/salt165 recommendation from the 512-519 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_512_519_followup_batch2.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 518 leaf 90, transfer 518 leaves 8,90
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; the two-leaf surface has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.71532847
- transfer 518 leaf-90 and two-leaf profiles form another 191/210/19 full-remainder negative band at 1.71532847 and 1.76642336 ops/rho

The thirty-eighth live-source exact queue batch checked the next global 22050/salt165 recommendation from the 304-311 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_304_311_global_followup_batch3.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 308 leaves 8,90, transfer 311 leaves 8,90
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; transfer 308 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.41605839
- minimum surface-stage ops/rho: 0.39416058
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.76642336
- transfer 308 two-leaf repeats the 191/210/19 full-remainder negative band at 1.76642336 ops/rho; transfer 311 two-leaf expands to a 326/351/25 negative band at 2.78832117 ops/rho

The thirty-ninth live-source exact queue batch checked the next global 22050/salt165 recommendation from the 360-367 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_360_367_global_followup_batch4.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 363 leaf 90, transfer 363 leaves 8,56,90
- hidden root predicate: 1/2 have `original_selected_root_pair_count=0`; the three-leaf surface has two original selected root pairs and no preserving Sage factor candidate
- preserving Sage factor-root-scan positives: 1/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.32116788
- transfer 363 leaf-90 is a 137/153/16 full-remainder negative at 1.32116788 ops/rho; the three-leaf 8,56,90 profile materializes at 1.45255474 ops/rho but has no preserving candidate, so it is a factor-stage rejection rather than a root-scan lead

The fortieth live-source exact queue batch checked the next global 22050/salt165 recommendation from the 504-511 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_504_511_global_followup_batch5.json`
- profiles requested/materialized: 3/3
- materialization errors: 0
- verified cases: 3
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 511 leaves 8,90; transfer 511 leaves 8,56,90; transfer 511 leaves 8,54,56,90
- hidden root predicate: all three profiles have one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 3/3
- minimum factor-root-scan ops/rho: 0.54744526
- minimum surface-stage ops/rho: 0.42335766
- full-remainder below-rho positives: 0/3
- minimum full-remainder ops/rho: 1.79562044
- transfer 511 multi-leaf profiles form another 191/210/19 full-remainder negative band at 1.79562044, 1.81021898, and 1.82481752 ops/rho; adding leaves raises the full-remainder cost while keeping the same preserving linear factor

The forty-first live-source exact queue batch checked the next global 22050/salt165 recommendation from the 528-535 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_528_535_global_followup_batch6.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 530 leaf 90, transfer 530 leaves 8,90
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.20437956
- transfer 530 is a near-control 121/136/15 full-remainder band at 1.20437956 and 1.21897810 ops/rho, still above rho and therefore not a transfer-294/618 reproduction

The forty-second live-source exact queue batch checked the next global 22050/salt165 recommendation from the 552-559 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_552_559_global_followup_batch7.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 556 leaf 90, transfer 556 leaves 8,90
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.01459854
- transfer 556 forms a 232/253/21 full-remainder negative band at 2.01459854 and 2.02919708 ops/rho despite root0 and a cheap preserving factor-root scan

The forty-third live-source exact queue batch checked the next global 22050/salt165 recommendation from the 568-575 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_568_575_global_followup_batch8.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 572 leaf 90, transfer 572 leaves 8,90
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 4.18248175
- transfer 572 forms a much larger 529/561/32 full-remainder negative band at 4.18248175 and 4.19708029 ops/rho despite root0 and a cheap preserving factor-root scan

The forty-fourth live-source exact queue batch checked the mixed next-global recommendation from the 584-591 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_584_591_global_followup_batch9.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`, and `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 591 leaves 8,90; target67 transfer 589 leaf 10
- hidden root predicate: target67 transfer 589 has `original_selected_root_pair_count=0`; 22050 transfer 591 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.39416058
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.512
- target67 transfer 589 repeats a 137/153/16 full-remainder negative band at 1.512 ops/rho; 22050 transfer 591 leaves 8,90 forms a 277/300/23 negative band at 2.39416058 ops/rho despite selected root-pair recovery at the factor stage

The forty-fifth live-source exact queue batch checked the next global 22050/salt165 recommendation from the 616-623 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_616_623_global_followup_batch10.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731`, `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 623 leaf 90, transfer 623 leaves 8,90
- hidden root predicate: leaf 90 has `original_selected_root_pair_count=0`; leaves 8,90 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.71532847
- transfer 623 repeats a 191/210/19 full-remainder negative band at 1.71532847 and 1.76642336 ops/rho despite root0 on leaf 90 and selected root-pair recovery in the two-leaf profile

The forty-sixth live-source exact queue batch checked the mixed next-global recommendation from the 632-639 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_632_639_global_followup_batch11.json`
- profiles requested/materialized: 3/3
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`, and `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 635 leaf 90, 22050 transfer 635 leaves 8,90, target67 transfer 637 leaf 2
- hidden root predicate: 22050 leaf 90 and target67 leaf 2 have `original_selected_root_pair_count=0`; 22050 leaves 8,90 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 3/3
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/3
- minimum full-remainder ops/rho: 2.17518248
- 22050 transfer 635 expands to a 254/276/22 full-remainder negative band at 2.17518248 and 2.22627737 ops/rho; target67 transfer 637 expands to a 407/435/28 negative band at 3.672 ops/rho

The forty-seventh live-source exact queue batch checked the mixed next-global recommendation from the 240-247 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_240_247_global_followup_batch12.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`, and `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 242 leaves 8,90; target67 transfer 242 leaf 10
- hidden root predicate: target67 transfer 242 has `original_selected_root_pair_count=0`; 22050 transfer 242 leaves 8,90 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.39416058
- full-remainder below-rho positives: 1/2
- minimum full-remainder ops/rho: 0.86131387
- 22050 transfer 242 leaves 8,90 reproduces the 67/78/11 full-remainder shape at 0.86131387 ops/rho, while the target67 transfer-242 analogue is a 154/171/17 full-remainder negative at 1.648 ops/rho

The forty-eighth live-source exact queue batch checked the next-global 22050/salt165 recommendation from the 624-631 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_624_631_global_followup_batch13.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 629 leaf 90 and transfer 629 leaves 8,90
- hidden root predicate: transfer 629 leaf 90 has `original_selected_root_pair_count=0`; transfer 629 leaves 8,90 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.86131387
- transfer 629 forms a 211/231/20 full-remainder negative band: leaf 90 costs 1.86131387 ops/rho and leaves 8,90 cost 1.91240876 ops/rho, so this extends the weak-stage-positive controls rather than the 67/78/11 collapse pocket

The forty-ninth live-source exact queue batch checked the mixed next-global recommendation from the 648-655 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_648_655_global_followup_batch14.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`, and `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 655 leaves 8,90; target67 transfer 653 leaf 5
- hidden root predicate: target67 transfer 653 has `original_selected_root_pair_count=0`; 22050 transfer 655 leaves 8,90 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.39416058
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.944
- target67 transfer 653 leaf 5 forms a 191/210/19 full-remainder negative at 1.944 ops/rho; 22050 transfer 655 leaves 8,90 repeats the selected-root-pair factor-stage recovery but expands to a 277/300/23 full-remainder negative at 2.39416058 ops/rho

The fiftieth live-source exact queue batch checked the next-global target67/salt205 recommendation from the 544-551 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_544_551_global_followup_batch15.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 548 leaf 10; transfer 545 leaf 6
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.272
- transfer 545 is a 232/253/21 full-remainder negative at 2.272 ops/rho; transfer 548 expands to a 379/406/27 negative at 3.448 ops/rho

The fifty-first live-source exact queue batch checked the mixed next-global recommendation from the 320-327 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_320_327_global_followup_batch16.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`, and `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 321 leaves 8,90; target67 transfer 325 leaf 5
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.34306569
- minimum surface-stage ops/rho: 0.35766423
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.33576642
- 22050 transfer 321 leaves 8,90 is a 137/153/16 full-remainder negative at 1.33576642 ops/rho, close to the previous single-leaf transfer-321 floor but still above rho; target67 transfer 325 leaf 5 expands to a 254/276/22 full-remainder negative at 2.448 ops/rho

The fifty-second live-source exact queue batch checked the mixed next-global recommendation from the 360-367 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_360_367_global_followup_batch17.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 1
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`, and `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 363 leaves 8,54,56,90; target67 transfer 366 leaf 5
- hidden root predicate: target67 transfer 366 has `original_selected_root_pair_count=0`; 22050 transfer 363 leaves 8,54,56,90 has `original_selected_root_pair_count=2` and therefore fails the root0 shell after exact materialization
- preserving Sage factor-root-scan positives: 1/2
- minimum preserving factor-root-scan ops/rho: 0.432
- minimum preserving surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum raw full-remainder ops/rho: 1.46715328
- minimum preserving full-remainder ops/rho: 3.448
- 22050 transfer 363 leaves 8,54,56,90 has raw full-remainder cost 1.46715328 ops/rho but no preserving Sage factor candidate, so it is a root0/preservation failure rather than a lead; target67 transfer 366 leaf 5 is root0 and preserving-factor positive but expands to a 379/406/27 full-remainder negative at 3.448 ops/rho

The fifty-third live-source exact queue batch checked the mixed next-global recommendation from the 248-255 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_248_255_global_followup_batch18.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`, and `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 249 leaves 8,90; target67 transfer 251 leaf 4
- hidden root predicate: target67 transfer 251 has `original_selected_root_pair_count=0`; 22050 transfer 249 leaves 8,90 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.39416058
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.06569343
- 22050 transfer 249 leaves 8,90 expands to a 232/253/21 full-remainder negative at 2.06569343 ops/rho despite selected-root-pair recovery at the factor stage; target67 transfer 251 leaf 4 repeats the 379/406/27 target67 negative band at 3.448 ops/rho

The fifty-fourth live-source exact queue batch checked the next-global target67/salt205 recommendation from the 560-567 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_560_567_global_followup_batch19.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 567 leaf 4; transfer 561 leaf 12
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.024
- target67 transfer 567 leaf 4 expands to a 326/351/25 full-remainder negative at 3.024 ops/rho; target67 transfer 561 leaf 12 expands to a 667/703/36 negative at 5.752 ops/rho, the largest target67 queue negative so far

The fifty-fifth live-source exact queue batch checked the next-global target67/salt205 recommendation from the 296-303 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_296_303_global_followup_batch20.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 297 leaf 1; transfer 300 leaf 12
- hidden root predicate: transfer 297 has `original_selected_root_pair_count=0`; transfer 300 has one selected root pair and one selected linear root recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.848
- target67 transfer 300 leaf 12 has selected-root-pair recovery but remains a 172/190/18 full-remainder negative at 1.848 ops/rho; target67 transfer 297 leaf 1 expands to a 436/464/29 full-remainder negative at 3.904 ops/rho

The fifty-sixth live-source exact queue batch checked the next-global target67/salt205 recommendation from the 456-463 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_456_463_global_followup_batch21.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 463 leaf 1; transfer 460 leaf 9
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.272
- target67 transfer 463 leaf 1 expands to a 232/253/21 full-remainder negative at 2.272 ops/rho; target67 transfer 460 leaf 9 expands to a 466/496/30 negative at 4.144 ops/rho

The fifty-seventh live-source exact queue batch checked the next-global mixed recommendation from the 312-319 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_312_319_global_followup_batch22.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`; `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 316 leaves 8,90; target67 transfer 313 leaf 6
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.34306569
- minimum surface-stage ops/rho: 0.35766423
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.59124088
- 22050 transfer 316 leaves 8,90 expands to a 172/190/18 full-remainder negative at 1.59124088 ops/rho; target67 transfer 313 leaf 6 expands to a 436/465/29 negative at 3.904 ops/rho

The fifty-eighth live-source exact queue batch checked the next-global target67/salt205 recommendation from the 568-575 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_568_575_global_followup_batch23.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 574 leaf 0; transfer 569 leaf 8
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.944
- target67 transfer 569 leaf 8 forms a 191/210/19 full-remainder negative at 1.944 ops/rho; target67 transfer 574 leaf 0 expands to a 352/378/26 negative at 3.232 ops/rho

The fifty-ninth live-source exact queue batch checked the next-global mixed recommendation from the 256-263 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_256_263_global_followup_batch24.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`; `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 257 leaves 8,90; target67 transfer 263 leaf 8
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.34306569
- minimum surface-stage ops/rho: 0.35766423
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.59124088
- 22050 transfer 257 leaves 8,90 expands to a 172/190/18 full-remainder negative at 1.59124088 ops/rho; target67 transfer 263 leaf 8 expands to a 407/435/28 negative at 3.672 ops/rho

The sixtieth live-source exact queue batch checked the next-global mixed recommendation from the 592-599 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_592_599_global_followup_batch25.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/rows: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`; `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 596 leaves 8,90; target67 transfer 592 leaf 7
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.34306569
- minimum surface-stage ops/rho: 0.35766423
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.72992701
- 22050 transfer 596 leaves 8,90 expands to a 191/210/19 full-remainder negative at 1.72992701 ops/rho; target67 transfer 592 leaf 7 expands to a 211/231/20 negative at 2.104 ops/rho

The sixty-first live-source exact queue batch checked the next-global target67/salt205 recommendation from the 672-679 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_672_679_global_followup_batch26.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 674 leaf 4; transfer 673 leaf 10
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.648
- target67 transfer 673 leaf 10 forms a 154/171/17 full-remainder negative at 1.648 ops/rho; target67 transfer 674 leaf 4 expands to a 232/253/21 negative at 2.272 ops/rho

The sixty-second live-source exact queue batch checked the row-diverse 22050/salt165 recommendation from the 728-735 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_728_735_rowdiverse_batch27.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 735 leaves 8,90; transfer 732 leaves 8,90
- hidden root predicate: transfer 735 has `original_selected_root_pair_count=0`; transfer 732 has one selected root pair and one selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.34306569
- minimum surface-stage ops/rho: 0.35766423
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.59124088
- 22050 transfer 735 leaves 8,90 expands to a 172/190/18 full-remainder negative at 1.59124088 ops/rho; 22050 transfer 732 leaves 8,90 has selected-root-pair recovery at the factor stage but remains a 191/210/19 full-remainder negative at 1.76642336 ops/rho

The sixty-third live-source exact queue batch checked the next global 22050/salt165 recommendation from the 208-215 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_208_215_global_followup_batch28.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`
- transfer/leaves: transfer 209 leaves 8,90
- hidden root predicate: `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.34306569
- minimum surface-stage ops/rho: 0.35766423
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.45985401
- 22050 transfer 209 leaves 8,90 is a root0 154/171/17 full-remainder negative at 1.45985401 ops/rho despite preserving factor-root-scan and surface-stage positives

The sixty-fourth live-source exact queue batch checked the next global 22050/salt165 recommendation from the 224-231 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_224_231_global_followup_batch29.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`
- transfer/leaves: transfer 226 leaves 8,90
- hidden root predicate: `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.41605839
- minimum surface-stage ops/rho: 0.43065693
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.18248175
- 22050 transfer 226 leaves 8,90 is a root0 106/120/14 near-control, but it remains full-remainder negative at 1.18248175 ops/rho despite preserving factor-root-scan and surface-stage positives

The sixty-fifth live-source exact queue batch checked the next global 22050/salt165 recommendation from the 264-271 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_264_271_global_followup_batch30.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`
- transfer/leaves: transfer 271 leaves 8,90
- hidden root predicate: `original_selected_root_pair_count=2`
- preserving Sage factor-root-scan positives: 0/1
- minimum factor-root-scan ops/rho: not applicable; no preserving Sage factor candidate
- minimum surface-stage ops/rho: not applicable; no preserving Sage factor candidate
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.80291971
- 22050 transfer 271 leaves 8,90 has nontrivial resultant factors but no preserving Sage factor candidate; the raw A/B remainders have 100 and 91 monomials with resultant monomials 210, so it is a stronger negative than the recent near-controls

The sixty-sixth live-source exact queue batch checked the next global target67/salt205 recommendation from the 296-303 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_296_303_global_followup_batch31.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 303 leaf 0; transfer 296 leaf 5
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.792
- target67 transfer 296 leaf 5 forms a 172/190/18 full-remainder negative at 1.792 ops/rho; target67 transfer 303 leaf 0 expands to a 301/325/24 negative at 2.824 ops/rho

The sixty-seventh live-source exact queue batch checked the next global target67/salt205 recommendation from the 368-375 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_368_375_global_followup_batch32.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 375 leaf 1; transfer 368 leaf 0
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.632
- target67 transfer 368 leaf 0 forms a 277/300/23 full-remainder negative at 2.632 ops/rho; target67 transfer 375 leaf 1 expands to a 301/325/24 negative at 2.824 ops/rho

The sixty-eighth live-source exact queue batch checked the next global target67/salt205 recommendation from the 400-407 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_400_407_global_followup_batch33.json`
- profiles requested/materialized: 3/3
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 402 leaf 6; transfer 400 leaf 10; transfer 407 leaf 9
- hidden root predicate: transfers 402 and 400 have `original_selected_root_pair_count=0`; transfer 407 has selected root-pair and selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 3/3
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/3
- minimum full-remainder ops/rho: 2.272
- target67 transfer 402 leaf 6 forms a 232/253/21 full-remainder negative at 2.272 ops/rho; transfer 400 leaf 10 expands to 379/406/27 at 3.448 ops/rho; transfer 407 leaf 9 has selected-root-pair/linear recovery at the factor stage but remains a 379/406/27 negative at 3.52 ops/rho

The sixty-ninth live-source exact queue batch checked the next global target67/salt205 recommendation from the 232-239 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_232_239_global_followup_batch34.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 239 leaf 7
- hidden root predicate: `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.824
- target67 transfer 239 leaf 7 expands to a 301/325/24 full-remainder negative at 2.824 ops/rho

The seventieth live-source exact queue batch checked the next global target67/salt205 recommendation from the 512-519 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_512_519_global_followup_batch35.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 519 leaf 4; transfer 518 leaf 9
- hidden root predicate: both profiles have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.448
- target67 transfer 518 leaf 9 forms a 254/276/22 full-remainder negative at 2.448 ops/rho; transfer 519 leaf 4 expands to 301/325/24 at 2.824 ops/rho

The seventy-first live-source exact queue batch checked the next global target67/salt205 recommendation from the 376-383 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_376_383_global_followup_batch36.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 377 leaf 5; transfer 378 leaf 4
- hidden root predicate: transfer 378 has `original_selected_root_pair_count=0`; transfer 377 has selected root-pair and selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.792
- target67 transfer 378 leaf 4 forms a 172/190/18 full-remainder negative at 1.792 ops/rho; transfer 377 leaf 5 has selected-root-pair/linear recovery at the factor stage but expands to 352/378/26 at 3.288 ops/rho

The seventy-second live-source exact queue batch checked the next global target67/salt205 recommendation from the 632-639 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_632_639_global_followup_batch37.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 633 leaf 9; transfer 634 leaf 5
- hidden root predicate: both materialized surfaces have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.104
- target67 transfer 633 leaf 9 forms a 211/231/20 full-remainder negative at 2.104 ops/rho; transfer 634 leaf 5 expands to 301/325/24 at 2.824 ops/rho

The seventy-third live-source exact queue batch checked the next global target67/salt205 recommendation from the 360-367 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_360_367_global_followup_batch38.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 361 leaf 0; transfer 365 leaf 7
- hidden root predicate: both materialized surfaces have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.104
- target67 transfer 365 leaf 7 forms a 211/231/20 full-remainder negative at 2.104 ops/rho; transfer 361 leaf 0 forms a 277/300/23 negative at 2.632 ops/rho

The seventy-fourth live-source exact queue batch checked the next global target67/salt205 recommendation from the 672-679 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_672_679_global_followup_batch39.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 676 leaf 8; transfer 678 leaf 2
- hidden root predicate: transfer 678 has `original_selected_root_pair_count=0`; transfer 676 has selected root-pair and selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.488
- target67 transfer 676 leaf 8 has selected-root-pair/linear recovery at the factor stage but expands to a 379/406/27 full-remainder negative at 3.488 ops/rho; transfer 678 leaf 2 is root0 and expands to 407/435/28 at 3.672 ops/rho

The seventy-fifth live-source exact queue batch checked the next global target67/salt205 recommendation from the 224-231 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_224_231_global_followup_batch40.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 225 leaf 8
- hidden root predicate: the materialized surface has `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 3.448
- target67 transfer 225 leaf 8 is root0 and forms a 379/406/27 full-remainder negative at 3.448 ops/rho

The seventy-sixth live-source exact queue batch checked the next global target67/salt205 recommendation from the 232-239 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_232_239_global_followup_batch41.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 233 leaf 4
- hidden root predicate: the materialized surface has `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.104
- target67 transfer 233 leaf 4 is root0 and forms a 211/231/20 full-remainder negative at 2.104 ops/rho

The seventy-seventh live-source exact queue batch checked the row-diverse 22050/salt165 recommendation from the 288-295 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_288_295_rowdiverse_followup_batch42.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 294 leaves 8,90
- hidden root predicate: the materialized surface has one original selected root pair and one selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.45985401
- minimum surface-stage ops/rho: 0.39416058
- full-remainder below-rho positives: 1/1
- minimum full-remainder ops/rho: 0.86131387
- 22050 transfer 294 leaves 8,90 reproduces the 67/78/11 full-remainder collapse as a non-root0 selected-root/linear-recovery surface

The seventy-eighth live-source exact queue batch checked the next global target67/salt205 recommendation from the 248-255 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_248_255_global_followup_batch43.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 253 leaf 0
- hidden root predicate: the materialized surface has `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.104
- target67 transfer 253 leaf 0 is root0 and forms a 211/230/20 full-remainder negative at 2.104 ops/rho

The seventy-ninth live-source exact queue batch checked the next global target67/salt205 recommendation from the 576-583 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_576_583_global_followup_batch44.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 582 leaf 8, transfer 579 leaf 10
- hidden root predicate: both materialized surfaces have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.024
- target67 transfer 582 leaf 8 forms a 326/351/25 negative at 3.024 ops/rho; target67 transfer 579 leaf 10 expands to 497/528/31 at 4.392 ops/rho

The eightieth live-source exact queue batch checked the next global mixed recommendation from the 408-415 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_mixed_408_415_global_followup_batch45.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`, and `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: 22050 transfer 409 leaf 90, target67 transfer 412 leaf 6
- hidden root predicate: 22050 transfer 409 is root0; target67 transfer 412 has one original selected root pair and one selected linear recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.33576642
- minimum surface-stage ops/rho: 0.34306569
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.70072993
- 22050 transfer 409 leaf 90 is a 326/351/25 negative at 2.70072993 ops/rho; target67 transfer 412 leaf 6 is a selected-root/linear-recovery 407/435/28 negative at 3.744 ops/rho

The eighty-first live-source exact queue batch checked the final current row-diverse 22050/salt165 recommendation from the 424-431 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_22050_salt165_424_431_rowdiverse_followup_batch46.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `22050.cf1@11731` on `22050.cf1@11731:uniform:256:salt165`
- transfers/leaves: transfer 424 leaf 90
- hidden root predicate: the materialized surface has one original selected root pair and one selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.54014599
- minimum surface-stage ops/rho: 0.37956204
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.37956204
- 22050 transfer 424 leaf 90 is a selected-root/linear-recovery 277/300/23 negative at 2.37956204 ops/rho

The eighty-second live-source exact queue batch checked the next global target67/salt205 recommendation from the 488-495 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_488_495_global_followup_batch47.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 490 leaf 7 and transfer 492 leaf 7
- hidden root predicate: both materialized surfaces have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.792
- target67 transfer 490 leaf 7 is a root0 172/190/18 negative at 1.792 ops/rho; transfer 492 leaf 7 is a root0 326/351/25 negative at 3.024 ops/rho

The eighty-third live-source exact queue batch checked the next global target67/salt205 recommendation from the 304-311 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_304_311_global_followup_batch48.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 304 leaf 4 and transfer 309 leaf 1
- hidden root predicate: both materialized surfaces have selected root-pair and selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.648
- minimum surface-stage ops/rho: 0.496
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.16
- target67 transfer 309 leaf 1 is a selected-root/linear-recovery 211/231/20 negative at 2.16 ops/rho; transfer 304 leaf 4 is a selected-root/linear-recovery 254/276/22 negative at 2.52 ops/rho

The eighty-fourth live-source exact queue batch checked the next global target67/salt205 recommendation from the 432-439 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_432_439_global_followup_batch49.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 1
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 436 leaf 6 and transfer 438 leaf 12
- hidden root predicate: transfer 436 is root0; transfer 438 has one selected root pair and one selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.648
- target67 transfer 438 leaf 12 is a verifier-backed selected-root/linear-recovery 137/153/16 negative at 1.648 ops/rho; transfer 436 leaf 6 is a root0 407/435/28 negative at 3.672 ops/rho

The eighty-fifth live-source exact queue batch checked the next global target67/salt205 recommendation from the 448-455 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_448_455_global_followup_batch50.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 449 leaf 10 and transfer 454 leaf 10
- hidden root predicate: both materialized surfaces have `original_selected_root_pair_count=0`
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.944
- target67 transfer 454 leaf 10 is a root0 191/210/19 negative at 1.944 ops/rho; transfer 449 leaf 10 is a root0 254/276/22 negative at 2.448 ops/rho

The eighty-sixth live-source exact queue batch checked the next global target67/salt205 recommendation from the 336-343 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_336_343_global_followup_batch51.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 1
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 337 leaf 10 and transfer 339 leaf 10
- hidden root predicate: transfer 337 is root0; transfer 339 has one selected root pair and one selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.672
- target67 transfer 337 leaf 10 is a root0 407/435/28 negative at 3.672 ops/rho; transfer 339 leaf 10 is a verifier-backed selected-root/linear-recovery 436/465/29 negative at 4.072 ops/rho

The eighty-seventh live-source exact queue batch checked the next global target67/salt205 recommendation from the 280-287 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_280_287_global_followup_batch52.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 283 leaf 0 and transfer 283 leaves 0,9
- hidden root predicate: both surfaces are root0 with zero original selected root pairs
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.232
- both transfer-283 surfaces have the 352/378/26 full-remainder band and remain full-remainder negative at 3.232 and 3.248 ops/rho

The eighty-eighth live-source exact queue batch checked the next global target67/salt205 recommendation from the 480-487 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_480_487_global_followup_batch53.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 481 leaf 4 and transfer 482 leaf 9
- hidden root predicate: transfer 481 is root0; transfer 482 has one selected root pair and one selected linear recovery at the factor stage
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.632
- target67 transfer 481 leaf 4 is a root0 277/300/23 negative at 2.632 ops/rho; transfer 482 leaf 9 is a selected-root/linear-recovery 379/405/27 negative at 3.52 ops/rho

The eighty-ninth live-source exact queue batch checked the next global target67/salt205 recommendation from the 272-279 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_272_279_global_followup_batch54.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 273 leaf 12 and transfer 278 leaves 1,3
- hidden root predicate: transfer 273 has one selected root pair and one selected linear recovery at the factor stage; transfer 278 is root0
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.344
- target67 transfer 273 leaf 12 is a selected-root/linear-recovery 232/253/21 negative at 2.344 ops/rho; transfer 278 leaves 1,3 is a root0 277/300/23 negative at 2.648 ops/rho

The ninetieth live-source exact queue batch checked the next global target67/salt205 recommendation from the 320-327 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_320_327_global_followup_batch55.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 326 leaf 7 and transfer 325 leaves 0,5
- hidden root predicate: both surfaces are root0 with zero selected root pairs and zero selected linear recoveries
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.464
- target67 transfer 325 leaves 0,5 is a root0 254/276/22 negative at 2.464 ops/rho; transfer 326 leaf 7 is a root0 379/406/27 negative at 3.448 ops/rho

The ninety-first live-source exact queue batch checked the next global target67/salt205 recommendation from the 360-367 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_360_367_global_followup_batch56.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 360 leaf 9 and transfer 366 leaves 5,7
- hidden root predicate: both surfaces are root0 with zero selected root pairs and zero selected linear recoveries
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.464
- target67 transfer 366 leaves 5,7 is a root0 379/406/27 negative at 3.464 ops/rho; transfer 360 leaf 9 is a root0 436/465/29 negative at 3.904 ops/rho

The ninety-second live-source exact queue batch checked the next global target67/salt205 recommendation from the 256-263 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_256_263_global_followup_batch57.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 257 leaves 0,9 and transfer 258 leaves 2,8
- hidden root predicate: transfer 257 is root0 with zero selected root pairs; transfer 258 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.848
- target67 transfer 258 leaves 2,8 is a selected-root/linear-recovery 172/190/18 negative at 1.848 ops/rho; transfer 257 leaves 0,9 is a root0 277/300/23 negative at 2.648 ops/rho

The ninety-third live-source exact queue batch checked the next global target67/salt205 recommendation from the 312-319 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_312_319_global_followup_batch58.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 312 leaves 0,6 and transfer 313 leaves 0,6
- hidden root predicate: both surfaces are root0 with zero selected root pairs and zero selected linear recoveries
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.12
- target67 transfer 312 leaves 0,6 is a root0 211/231/20 negative at 2.12 ops/rho; transfer 313 leaves 0,6 is a root0 436/465/29 negative at 3.92 ops/rho

The ninety-fourth live-source exact queue batch checked the next global target67/salt205 recommendation from the 392-399 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_392_399_global_followup_batch59.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 395 leaf 6 and transfer 395 leaves 5,6
- hidden root predicate: both surfaces are root0 with zero selected root pairs and zero selected linear recoveries
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 5.184
- target67 transfer 395 leaf 6 is a root0 596/630/34 negative at 5.184 ops/rho; transfer 395 leaves 5,6 repeats the same 596/630/34 band at 5.2 ops/rho

The ninety-fifth live-source exact queue batch checked the next global target67/salt205 recommendation from the 520-527 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_520_527_global_followup_batch60.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 522 leaf 6 and transfer 522 leaves 3,6
- hidden root predicate: both surfaces are root0 with zero selected root pairs and zero selected linear recoveries
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.632
- target67 transfer 522 leaf 6 is a root0 277/300/23 negative at 2.632 ops/rho; transfer 522 leaves 3,6 repeats the same 277/300/23 band at 2.648 ops/rho

The ninety-sixth live-source exact queue batch checked the next global target67/salt205 recommendation from the 632-639 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_632_639_global_followup_batch61.json`
- profiles requested/materialized: 3/3
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 636 leaf 12, transfer 632 leaf 12, and transfer 633 leaves 2,9
- hidden root predicate: transfer 632 leaf 12 and transfer 633 leaves 2,9 are root0; transfer 636 leaf 12 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 3/3
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/3
- minimum full-remainder ops/rho: 1.264
- target67 transfer 632 leaf 12 is a root0 106/120/14 near-control at 1.264 ops/rho; transfer 633 leaves 2,9 is a root0 211/231/20 negative at 2.12 ops/rho; transfer 636 leaf 12 has selected-root/linear recovery but expands to 326/351/25 at 3.08 ops/rho

The ninety-seventh live-source exact queue batch checked the next global target67/salt205 recommendation from the 536-543 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_536_543_global_followup_batch62.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 537 leaf 6 and transfer 537 leaves 0,6
- hidden root predicate: both surfaces are root0 with zero selected root pairs and zero selected linear recoveries
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.648
- target67 transfer 537 leaf 6 is a root0 154/171/17 negative at 1.648 ops/rho; transfer 537 leaves 0,6 repeats the same 154/171/17 band at 1.664 ops/rho

The ninety-eighth live-source exact queue batch checked the next global target67/salt205 recommendation from the 600-607 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_600_607_global_followup_batch63.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 607 leaf 4 and transfer 607 leaves 4,9
- hidden root predicate: neither surface is root0; both have one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.712
- minimum surface-stage ops/rho: 0.512
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.304
- target67 transfer 607 leaf 4 and leaves 4,9 both have selected-root/linear recovery but expand to the 352/377/26 full-remainder negative band at 3.304 and 3.32 ops/rho

The ninety-ninth live-source exact queue batch checked the next global target67/salt205 recommendation from the 232-239 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_232_239_global_followup_batch64.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 239 leaves 4,7 and transfer 233 leaves 1,4
- hidden root predicate: both surfaces are root0; both have zero selected root pairs and zero selected linear root recoveries
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.12
- target67 transfer 233 leaves 1,4 is a root0 211/231/20 negative at 2.12 ops/rho; target67 transfer 239 leaves 4,7 is a root0 301/325/24 negative at 2.84 ops/rho

The one hundredth live-source exact queue batch checked the next global target67/salt205 recommendation from the 464-471 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_464_471_global_followup_batch65.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 470 leaf 10 and transfer 470 leaves 1,10
- hidden root predicate: both surfaces are root0; both have zero selected root pairs and zero selected linear root recoveries
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.824
- target67 transfer 470 leaf 10 and leaves 1,10 both share the 301/325/24 full-remainder negative band at 2.824 and 2.84 ops/rho

The 101st live-source exact queue batch checked the next global target67/salt205 recommendation from the 592-599 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_592_599_global_followup_batch66.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 597 leaf 7 and transfer 592 leaves 7,10
- hidden root predicate: neither surface is root0; both have one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.656
- minimum surface-stage ops/rho: 0.512
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.016
- target67 transfer 597 leaf 7 is a selected-root/linear-recovery 191/210/19 negative at 2.016 ops/rho; transfer 592 leaves 7,10 is a selected-root/linear-recovery 211/231/20 negative at 2.224 ops/rho

The 102nd live-source exact queue batch checked the next global target67/salt205 recommendation from the 344-351 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_344_351_global_followup_batch67.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 350 leaf 3 and transfer 350 leaves 3,9
- hidden root predicate: neither surface is root0; both have one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.64
- minimum surface-stage ops/rho: 0.496
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.0
- target67 transfer 350 leaf 3 and leaves 3,9 are selected-root/linear-recovery 191/210/19 negatives at 2.0 and 2.016 ops/rho

The 103rd live-source exact queue batch checked the next global target67/salt205 recommendation from the 472-479 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_472_479_global_followup_batch68.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 475 leaf 2 and transfer 475 leaves 2,4
- hidden root predicate: transfer 475 leaf 2 is root0; transfer 475 leaves 2,4 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.104
- target67 transfer 475 leaf 2 is a root0 211/231/20 negative at 2.104 ops/rho; transfer 475 leaves 2,4 has selected-root/linear recovery and the same 211/231/20 band at 2.192 ops/rho

The 104th live-source exact queue batch checked the next global target67/salt205 recommendation from the 296-303 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_296_303_global_followup_batch69.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 297 leaves 1,5 and transfer 303 leaves 0,4
- hidden root predicate: transfer 297 leaves 1,5 is root0; transfer 303 leaves 0,4 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.912
- target67 transfer 303 leaves 0,4 has selected-root/linear recovery but remains a 301/325/24 negative at 2.912 ops/rho; transfer 297 leaves 1,5 is root0 and expands to 436/464/29 at 3.92 ops/rho

The 105th live-source exact queue batch checked the next global target67/salt205 recommendation from the 608-615 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_608_615_global_followup_batch70.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 612 leaf 10 and transfer 615 leaf 0
- hidden root predicate: transfer 615 leaf 0 is root0; transfer 612 leaf 10 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.512
- target67 transfer 615 leaf 0 is a root0 137/153/16 near-control at 1.512 ops/rho; transfer 612 leaf 10 has selected-root/linear recovery but expands to 436/465/29 at 3.992 ops/rho

The 106th live-source exact queue batch checked the next global target67/salt205 recommendation from the 632-639 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_632_639_global_followup_batch71.json`
- profiles requested/materialized: 3/3
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 637 leaves 1,2, transfer 632 leaves 3,12, and transfer 634 leaves 0,5
- hidden root predicate: transfer 632 leaves 3,12 is root0; transfers 634 leaves 0,5 and 637 leaves 1,2 each have one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 3/3
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/3
- minimum full-remainder ops/rho: 1.28
- target67 transfer 632 leaves 3,12 is a root0 106/120/14 near-control at 1.28 ops/rho; transfer 634 leaves 0,5 has selected-root/linear recovery but remains 301/325/24 at 2.912 ops/rho; transfer 637 leaves 1,2 has selected-root/linear recovery but expands to 407/435/28 at 3.76 ops/rho

The 107th live-source exact queue batch checked the next global target67/salt205 recommendation from the 248-255 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_248_255_global_followup_batch72.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 251 leaves 4,5 and transfer 253 leaves 0,7
- hidden root predicate: transfer 251 leaves 4,5 is root0; transfer 253 leaves 0,7 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 2.192
- target67 transfer 253 leaves 0,7 has selected-root/linear recovery but remains 211/230/20 at 2.192 ops/rho; transfer 251 leaves 4,5 is root0 but expands to 379/406/27 at 3.464 ops/rho

The 108th live-source exact queue batch checked the next global target67/salt205 recommendation from the 352-359 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_352_359_global_followup_batch73.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 352 leaf 2 and transfer 352 leaves 2,4
- hidden root predicate: transfer 352 leaf 2 is root0; transfer 352 leaves 2,4 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.792
- target67 transfer 352 leaf 2 is a root0 172/190/18 negative at 1.792 ops/rho; transfer 352 leaves 2,4 has selected-root/linear recovery but remains the same 172/190/18 band at 1.848 ops/rho

The 109th live-source exact queue batch checked the next global target67/salt205 recommendation from the 200-207 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_200_207_global_followup_batch74.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 201 leaf 6 and transfer 201 leaves 2,6
- hidden root predicate: transfer 201 leaf 6 is root0; transfer 201 leaves 2,6 has one selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.448
- target67 transfer 201 leaf 6 is a root0 379/406/27 negative at 3.448 ops/rho; transfer 201 leaves 2,6 has selected-root/linear recovery but remains the same 379/406/27 band at 3.536 ops/rho

The 110th live-source exact queue batch checked the next global target67/salt205 recommendation from the 376-383 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_376_383_global_followup_batch75.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 378 leaves 4,8 and transfer 377 leaves 5,10
- hidden root predicate: transfer 378 leaves 4,8 is root0; transfer 377 leaves 5,10 has two original selected root pairs but no preserving Sage factor candidate
- preserving Sage factor-root-scan positives: 1/2
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.808
- target67 transfer 378 leaves 4,8 is a root0 172/190/18 negative at 1.808 ops/rho; transfer 377 leaves 5,10 has only a non-preserving one-root/linear recovery at factor stage and expands to 352/378/26 at 3.408 ops/rho

The 111th live-source exact queue batch checked the next global target67/salt205 recommendation from the 616-623 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_616_623_global_followup_batch76.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 622 leaf 3 and transfer 622 leaves 3,7
- hidden root predicate: both transfer 622 surfaces are root0
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.232
- target67 transfer 622 leaf 3 is a root0 352/378/26 negative at 3.232 ops/rho; transfer 622 leaves 3,7 is also root0 and remains the same 352/378/26 band at 3.36 ops/rho

The 112th live-source exact queue batch checked the shared global and row-diverse target67/salt205 recommendation from the 624-631 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_624_631_global_followup_batch77.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 624 leaf 5 and transfer 624 leaves 3,5
- hidden root predicate: neither transfer 624 surface is root0; leaf 5 has one original selected root pair and leaves 3,5 has two original selected root pairs
- preserving Sage factor-root-scan positives: 1/2
- minimum factor-root-scan ops/rho: 0.704
- minimum surface-stage ops/rho: 0.496
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 3.504
- target67 transfer 624 leaf 5 preserves one selected root/linear recovery but is a 379/406/27 negative at 3.504 ops/rho; transfer 624 leaves 3,5 has only a non-preserving one-root/linear recovery and remains the same 379/406/27 band at 3.576 ops/rho

The 113th live-source exact queue batch checked the next global target67/salt205 recommendation from the 672-679 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_672_679_global_followup_batch78.json`
- profiles requested/materialized: 2/2
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfers/leaves: transfer 673 leaves 3,10 and transfer 676 leaves 4,8
- hidden root predicate: transfer 673 leaves 3,10 is root0; transfer 676 leaves 4,8 has one original selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 2/2
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/2
- minimum full-remainder ops/rho: 1.664
- target67 transfer 673 leaves 3,10 is a root0 154/171/17 negative at 1.664 ops/rho; transfer 676 leaves 4,8 preserves one selected-root/linear recovery but expands to 379/406/27 at 3.504 ops/rho

The 114th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 224-231 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_224_231_global_followup_batch79.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 225 leaves 8,12
- hidden root predicate: transfer 225 leaves 8,12 is non-root0 with one original selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.728
- minimum surface-stage ops/rho: 0.528
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 3.536
- target67 transfer 225 leaves 8,12 preserves one selected-root/linear recovery but expands to a 379/406/27 negative at 3.536 ops/rho

The 115th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 240-247 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_240_247_global_followup_batch80.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 242 leaves 6,10
- hidden root predicate: transfer 242 leaves 6,10 is root0
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.664
- target67 transfer 242 leaves 6,10 is a root0 154/171/17 negative at 1.664 ops/rho

The 116th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 256-263 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_256_263_global_followup_batch81.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 263 leaves 6,8
- hidden root predicate: transfer 263 leaves 6,8 is non-root0 with one original selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.72
- minimum surface-stage ops/rho: 0.512
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 3.744
- target67 transfer 263 leaves 6,8 preserves one selected-root/linear recovery but expands to a 407/435/28 negative at 3.744 ops/rho

The 117th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 264-271 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_264_271_global_followup_batch82.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 269 leaves 1,6
- hidden root predicate: transfer 269 leaves 1,6 is root0
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.552
- minimum surface-stage ops/rho: 0.568
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.952
- target67 transfer 269 leaves 1,6 is a root0 301/325/24 negative at 2.952 ops/rho

The 118th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 336-343 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_336_343_global_followup_batch83.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 1
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 339 leaves 9,10
- hidden root predicate: transfer 339 leaves 9,10 is non-root0 with one original selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.84
- minimum surface-stage ops/rho: 0.624
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 4.088
- target67 transfer 339 leaves 9,10 is verifier-backed and preserves one selected-root/linear recovery but expands to a 436/465/29 negative at 4.088 ops/rho

The 119th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 360-367 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_360_367_global_followup_batch84.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 361 leaves 0,12
- hidden root predicate: transfer 361 leaves 0,12 is non-root0 with one original selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.696
- minimum surface-stage ops/rho: 0.528
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.72
- target67 transfer 361 leaves 0,12 preserves one selected-root/linear recovery but expands to a 277/300/23 negative at 2.72 ops/rho

The 120th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 368-375 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_368_375_global_followup_batch85.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 375 leaves 1,10
- hidden root predicate: transfer 375 leaves 1,10 is root0
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.84
- target67 transfer 375 leaves 1,10 is root0 and weak-stage positive but expands to a 301/325/24 negative at 2.84 ops/rho

The 121st live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 440-447 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_440_447_global_followup_batch86.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 441 leaf 1
- hidden root predicate: transfer 441 leaf 1 is root0
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.448
- target67 transfer 441 leaf 1 is root0 and weak-stage positive but expands to a 254/276/22 negative at 2.448 ops/rho

The 122nd live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 448-455 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_448_455_global_followup_batch87.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 449 leaves 6,10
- hidden root predicate: transfer 449 leaves 6,10 is non-root0 with one original selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.672
- minimum surface-stage ops/rho: 0.512
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.52
- target67 transfer 449 leaves 6,10 preserves one selected-root/linear recovery but expands to a 254/276/22 negative at 2.52 ops/rho

The 123rd live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 456-463 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_456_463_global_followup_batch88.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 458 leaf 10
- hidden root predicate: transfer 458 leaf 10 is root0
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.824
- target67 transfer 458 leaf 10 is root0 and weak-stage positive but expands to a 301/324/24 negative at 2.824 ops/rho

The 124th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 480-487 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_480_487_global_followup_batch89.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 481 leaves 4,6
- hidden root predicate: transfer 481 leaves 4,6 has zero original selected root pairs and both selected leaves miss selected-root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.648
- target67 transfer 481 leaves 4,6 is weak-stage positive but expands to a 277/300/23 negative at 2.648 ops/rho

The 125th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 488-495 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_488_495_global_followup_batch90.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 492 leaves 7,10
- hidden root predicate: transfer 492 leaves 7,10 has zero original selected root pairs and both selected leaves miss selected-root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 3.04
- target67 transfer 492 leaves 7,10 is weak-stage positive but expands to a 326/351/25 negative at 3.04 ops/rho

The 126th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 496-503 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_496_503_global_followup_batch91.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 500 leaf 5
- hidden root predicate: transfer 500 leaf 5 has one original selected root pair and one selected linear root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.648
- minimum surface-stage ops/rho: 0.512
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.864
- target67 transfer 500 leaf 5 preserves selected-root/linear recovery but expands to a 172/190/18 negative at 1.864 ops/rho

The 127th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 504-511 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_504_511_global_followup_batch92.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 505 leaf 1
- hidden root predicate: transfer 505 leaf 1 has zero original selected root pairs and the selected leaf misses selected-root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.432
- minimum surface-stage ops/rho: 0.44
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 1.648
- target67 transfer 505 leaf 1 is weak-stage positive but expands to a 154/171/17 negative at 1.648 ops/rho

The 128th live-source exact queue batch checked the next shared global and row-diverse target67/salt205 recommendation from the 512-519 selector window:

- artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_topk12_root0_queue_target67_salt205_512_519_global_followup_batch93.json`
- profiles requested/materialized: 1/1
- materialization errors: 0
- verified cases: 0
- target/row: `67.a1@9803` on `67.a1@9803:uniform:256:salt205`
- transfer/leaves: transfer 518 leaves 9,12
- hidden root predicate: transfer 518 leaves 9,12 has zero original selected root pairs and both selected leaves miss selected-root recovery
- preserving Sage factor-root-scan positives: 1/1
- minimum factor-root-scan ops/rho: 0.44
- minimum surface-stage ops/rho: 0.456
- full-remainder below-rho positives: 0/1
- minimum full-remainder ops/rho: 2.464
- target67 transfer 518 leaves 9,12 is weak-stage positive but expands to a 254/276/22 negative at 2.464 ops/rho

The final eight live-source exact queue batches consumed the remaining target67/salt205 controls:

| Queue batch | Selector window | Transfer/leaves | Post-materialization predicate | Factor-root scan ops/rho | Full remainder ops/rho | Main read |
| --- | --- | --- | --- | --- | --- | --- |
| 129 | 528-535 | 532 leaf 6 | zero original selected root pairs | 0.432 | 2.632 | 277/300/23 negative |
| 130 | 544-551 | 549 leaf 1 | selected-root recovery | 0.696 | 2.896 | 301/325/24 negative |
| 131 | 560-567 | 561 leaves 6,12 | zero original selected root pairs | 0.44 | 5.768 | 667/703/36 negative |
| 132 | 568-575 | 569 leaves 8,10 | zero original selected root pairs | 0.44 | 1.96 | 191/210/19 negative |
| 133 | 576-583 | 579 leaves 8,10 | zero original selected root pairs | 0.44 | 4.408 | 497/528/31 negative |
| 134 | 632-639 | 636 leaves 3,12 | selected-root recovery | none | 3.152 | no preserving factor candidate |
| 135 | 640-647 | 643 leaf 1 | zero original selected root pairs | 0.432 | 4.648 | 529/561/32 negative |
| 136 | 648-655 | 653 leaves 3,5 | selected-root recovery | 0.632 | 2.0 | 191/210/19 negative |

The final eight materialized 8/8 profiles with zero materialization errors, zero full-remainder below-rho positives, and 7/8 preserving factor-root-scan positives. Batch 134 is the exception: transfer 636 leaves 3,12 has selected-root evidence in the original surface but no preserving Sage factor candidate.

Aggregate queue-batch status after the first 136 exact runs: 251 exact profile records, 249 unique materialized surfaces, 166 unique surfaces with root0, 243/249 unique preserving factor-root-scan positives, 4 full-remainder below-rho positives, and best queue full-remainder cost 0.84671533 ops/rho. This is still a candidate pocket, not an algorithmic speedup by itself; the refreshed triage reports the four queue positives as non-root0 22050/salt165 selected-root surfaces. The current top-k12/root0 queue is exhausted: no 22050/salt165 or target67/salt205 candidates remain, and the triage emits no next recommended batch.

A queue-only public sparsity miner over the first thirty-eight top-k12/root0 batches was added:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_queue38_public_sparsity_miner_transfer_salt_rawleaf_free_holdout_compact.json`
- leakage guards: transfer atoms, row-salt atoms, source-window salt labels/offsets, and raw selected-leaf identity atoms are forbidden
- records/positives: 82 surfaces, 2 full-remainder below-rho positives
- factor/root/surface stage: 82/82 preserving factor-root-scan positives and 82/82 surface-stage positives, so these labels remain too weak for promotion
- best recall-1 pre-materialization clause after the guards: `original_selected_root_pair_count=1 & profile_policy=fixed_global_cap3_ow1_hw3_lw0_sw0_cw0_aw0 & selected_leaf_min=90`
- rule quality: selects 6 surfaces, 2 positives and 4 negatives, precision 0.333333, recall 1.0
- selected negatives under that clause: transfers 595, 599, 655, and 732; the shared leaf/policy/root-pair envelope is therefore not sufficient to predict the 67/78/11 full-remainder collapse

A closed-queue public sparsity miner over all 136 top-k12/root0 batches was added:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_public_sparsity_miner_transfer_salt_rawleaf_free_holdout_compact.json`
- leakage guards: transfer atoms, row-salt atoms, source-window salt labels/offsets, and raw selected-leaf identity atoms are forbidden
- records/positives: 251 records, 4 full-remainder below-rho positives
- factor/root/surface stage: 245/251 preserving factor-root-scan positives and 245/251 surface-stage positives, so these labels remain too weak for promotion
- best guarded pre-materialization clause: `selected_leaf_min=90 & source_ops_over_rho<=1.0:False`
- best-clause quality: selects 9 records, 1 positive and 8 negatives, precision 0.111111, recall 0.25
- best recall-0.5 guarded clause: `original_selected_root_pair_count=1 & selected_leaf_min=90`, selecting 11 records with 2 positives and 9 negatives, precision 0.181818
- transfer holdout: holding out transfer 242 or 294 recovers at most 1/2 held-out positives with the best trained guarded rules
- source-label holdout: the two newer two-leaf positive sources, `22050_salt165_288_295_rowdiverse_followup_batch42` and `mixed_240_247_global_followup_batch12`, are not recovered by the best trained guarded rules
- conclusion: the closed queue still lacks a promotable no-lookahead public feature separating the 67/78/11 positives from the target67 and 22050 negatives

The transfer-and-salt-alias-free rerun with 728-735 added is:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_transfer_salt_alias_free_preselector.json`
- best preselector: `original_selected_root_pair_count=0 & selected_leaf_signature=8,90`
- selected records: 13
- positives: 4
- negatives: 9
- precision: 0.307692
- recall: 0.235294

The factor-root-scan rerun with 728-735 added is:

- artifact: `ecdlp_index_calculus_state/ffe_factor_root_scan_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_preselector.json`
- records: 372
- label positives: 357
- best preselector: `selected_leaf_index_count=1`
- selected records: 284
- selected positives: 284
- selected negatives: 0
- recall: 0.795518

The 728-735 result makes the current split sharper: the factor/root-scan layer is broadly portable, but adding fresh future-window verifier positives still does not add any full-remainder positives beyond the transfer-420 and transfer-618 pockets.

## Interpretation

The strongest current signal is transfer 618, salt165. It is not explained by `salt165` by itself, because most other salt165 transfer probes stay above rho. It is not explained by `transfer_index mod 16 = 10` by itself, because transfer 378 shares that residue and is above rho. It is also not explained by generic neighbor-salt transfer locality, because transfer 618 neighbor controls over salts 166, 171, 174, and 177 all stay above rho.

The primary clue is:

> a transfer-618/salt165 quotient-remainder collapse where Sage finite-field factorization preserves the candidate, the full remainder has 67 monomials, the full resultant has 78 monomials, the known hit-root count is 11, and the full-remainder FFE cost is 0.81021898 of generic rho.

The secondary clue remains transfer 420. Salt167 at transfer 420 also produces preserving below-rho full remainders, so that pocket is not a single-salt accident. In contrast, transfer 618 currently looks salt165-specific inside the tested neighbor set.

The new preselector audit says these should be treated as pocket-specific branches until a cross-pocket public feature is found. The richer public envelope, bank, and locality features improved the branch description but did not provide held-out transferability. Local controls sharpen the transfer-618 boundary: transfer 616 materializes nearby rows from the same selector window, and transfers 619/620 materialize the same salt165 row, but all are decisively above rho.

The new cross-target/literal-branch probes sharpen that boundary further: the literal `row_salt_transfer_mod32=165|10` atom was not merely untested elsewhere, it now has two historical same-target negatives at transfers 298 and 362. Compact static-bank `low_term_span` metadata also does not carry the full-remainder collapse to target `67.a1@9803`.

The forbidden-atom audit sharpens the boundary again: once transfer atoms are removed, the remaining perfect rule only describes the transfer-420/salt167 neighbor pocket. Once transfer and salt aliases are removed together, the best rule has 7 negatives for 4 positives. The current public feature set therefore does not yet explain the full-remainder collapse without local transfer/salt identity.

The 232-239 mixed-row scanner group is now resolved for exact materialization. It is negative for full-remainder collapse and positive only as a weaker factor-root-scan/surface-stage lead. The existing orientation audit still verifies relations but remains above rho under the measured model.

The 664-671 transfer-665 public-bounded follow-up is also resolved. It materializes cleanly and has the same weak factor-stage/surface-stage signature, but no verified relation-derived case and no full-remainder below-rho candidate.

The 672-679 public-bounded follow-up is now resolved as well. It is more promising than transfer 665 at the source/stress layer and improves the exact full-remainder floor to 1.32116788, but the verified `target_cap1` profile remains above rho after full Sage remainder accounting.

The 728-735 public-bounded follow-up is resolved too. It is source-rich and verifier-labeled, but exact materialization raises the full-remainder minimum to 2.272 ops/rho, so it is a stronger negative for the same promotion path. The compact holdout rerun now adds that the mined public rules have zero held-out full-remainder-positive hits across transfer-index, row-salt, and source-label splits.

The row-diverse top-k12/root0 queue found a fresh 22050/salt165 full-remainder pocket at transfer 294. It has the same 67-monomial/78-resultant/known-hit-root-11 shape as transfer 618, and transfer 294 now has two non-root0 selected-root positives at leaf 90 and leaves 8,90, with the two-leaf surface at 0.86131387 ops/rho. Transfer 242 also reproduces that same full-remainder shape as two non-root0 surfaces with selected linear recovery. The root0 branch still is not a residue-only generalization: paired transfer 295 is negative, the earlier scanner-selected transfer 298 control is negative, transfer 245 is the root0 half of the 240-247 batch and is negative, and follow-ups at transfers 389/391 leaf-90 and two-leaf variants, 416/421 leaf-90 and two-leaf variants, 424, 464, 493/488 leaf-90 and two-leaf variants, 518, 209 leaf-90 and two-leaf variants, 226 leaf-90 and two-leaf variants, 257, 249, 299, 316, 321, 334 single- and two-leaf variants, 409, 585 leaf-90 and two-leaf variants, 590 single- and two-leaf variants, 591, 735, 732, 269 single-leaf and two-leaf variants, 271 leaf-90 and two-leaf variants, 509, 511 single-leaf and multi-leaf variants, 530/556/572 single-leaf and two-leaf variants, 595, 596, 599, 308, 311, 354, 397, 405, 649 single- and two-leaf variants, and 655 are also full-remainder negative. Transfers 493 and 226 are the nearest root0-style follow-ups so far at 1.09489051 ops/rho; transfer 493 two-leaf is the next nearest nonroot0 two-leaf follow-up at 1.14598540 ops/rho, and transfer 226 two-leaf is a root0 near-control at 1.18248175 ops/rho, followed by transfers 530 and 590 at 1.20437956 and transfer 354 at 1.32116788. Transfer 209 two-leaf is a root0 154/171/17 negative at 1.45985401 ops/rho, transfer 271 two-leaf has no preserving factor candidate at 1.80291971 ops/rho, transfer 424 is a selected-root/linear-recovery 277/300/23 negative at 2.37956204 ops/rho, transfer 409 is a root0 326/351/25 negative at 2.70072993 ops/rho, and target67 transfers 225/233/239/253/273/278/283/296/303/304/309/325/326/337/339/360/361/365/366/368/375/377/378/400/402/407/412/436/438/449/454/481/482/490/492/518/519/579/582/633/634/676/678 add or repeat 137/153/16, 172/190/18, 191/210/19, 211/230/20, 211/231/20, 232/253/21, 254/276/22, 277/300/23, 301/325/24, 326/351/25, 352/378/26, 379/405/27, 379/406/27, 407/435/28, 436/465/29, and 497/528/31 negatives. Target67 transfer 438 is a verifier-backed selected-root/linear-recovery 137/153/16 negative at 1.648 ops/rho, transfer 490 is a root0 172/190/18 negative at 1.792 ops/rho, transfer 454 is a root0 191/210/19 negative at 1.944 ops/rho, transfer 309 is a selected-root/linear-recovery 211/231/20 negative at 2.16 ops/rho, transfer 273 is a selected-root/linear-recovery 232/253/21 negative at 2.344 ops/rho, transfer 449 is a root0 254/276/22 negative at 2.448 ops/rho, transfer 325 leaves 0,5 is a root0 254/276/22 negative at 2.464 ops/rho, transfer 304 is a selected-root/linear-recovery 254/276/22 negative at 2.52 ops/rho, transfer 481 is a root0 277/300/23 negative at 2.632 ops/rho, transfer 278 is a root0 277/300/23 negative at 2.648 ops/rho, transfer 492 is a root0 326/351/25 negative at 3.024 ops/rho, transfer 283 is a root0 352/378/26 negative at 3.232 ops/rho, transfer 326 is a root0 379/406/27 negative at 3.448 ops/rho, transfer 366 leaves 5,7 is a root0 379/406/27 negative at 3.464 ops/rho, transfer 482 is a selected-root/linear-recovery 379/405/27 negative at 3.52 ops/rho, transfer 337 is a root0 407/435/28 negative at 3.672 ops/rho, transfer 436 is a root0 407/435/28 negative at 3.672 ops/rho, transfer 360 is a root0 436/465/29 negative at 3.904 ops/rho, and transfer 339 is a verifier-backed selected-root/linear-recovery 436/465/29 negative at 4.072 ops/rho. Transfers 556 and 572 are root0 counterexamples to the weak-stage selector, expanding to 2.01459854 and 4.18248175 ops/rho.

The newest target67/salt205 selector-256_263 controls add another two-leaf negative band rather than a collapse: transfer 258 has selected-root/linear recovery but remains a 172/190/18 full-remainder negative at 1.848 ops/rho, while transfer 257 is root0 but expands to 277/300/23 at 2.648 ops/rho. The selector-312_319 controls are also root0 full-remainder negatives: transfer 312 is 211/231/20 at 2.12 ops/rho and transfer 313 is 436/465/29 at 3.92 ops/rho. The selector-392_399 transfer-395 pair is a stronger negative: adding leaf 5 does not change the 596/630/34 full-remainder shape, and both one-leaf and two-leaf variants are above 5 ops/rho. The selector-520_527 transfer-522 pair is a milder repeat: adding leaf 3 does not change the 277/300/23 full-remainder shape, and both variants stay above rho at roughly 2.64 ops/rho.

The newest selector-376_383 repeat resolves the queued transfer 378/377 profiles without adding a collapse. Transfer 378 leaves 4,8 is root0 and keeps the weak factor-stage lead, but the full A/B remainder is 172/190/18 at 1.808 ops/rho. Transfer 377 leaves 5,10 has a non-preserving factor that recovers one selected root/linear relation and misses the other, so the full remainder expands to 352/378/26 at 3.408 ops/rho.

The newest selector-616_623 transfer-622 repeat is also a full-remainder negative. Both the single-leaf and two-leaf variants are root0 and preserve below-rho factor-stage scans, but they share the 352/378/26 full-remainder shape and stay above rho at 3.232 and 3.36 ops/rho.

The newest selector-624_631 transfer-624 repeat consumes the profile that was simultaneously the global and row-diverse target67 recommendation. It is non-root0: the single-leaf profile preserves its selected root/linear recovery but still expands to 379/406/27 at 3.504 ops/rho, while the two-leaf profile only partially recovers one selected root and remains 379/406/27 at 3.576 ops/rho.

The newest selector-672_679 target67 repeat consumes the next global recommendation without adding a full-remainder collapse. Transfer 673 leaves 3,10 is root0 and keeps the weak factor-stage lead, but the full A/B remainder is 154/171/17 at 1.664 ops/rho. Transfer 676 leaves 4,8 has selected-root/linear recovery at the factor stage, but the full remainder expands to 379/406/27 at 3.504 ops/rho.

The newest selector-224_231 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 225 leaves 8,12 has selected-root/linear recovery at the factor stage, but the full remainder expands to 379/406/27 at 3.536 ops/rho.

The newest selector-240_247 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 242 leaves 6,10 is root0 and weak-stage positive, but the full A/B remainder is 154/171/17 at 1.664 ops/rho.

The newest selector-256_263 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 263 leaves 6,8 has selected-root/linear recovery at the factor stage, but the full remainder expands to 407/435/28 at 3.744 ops/rho.

The newest selector-264_271 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 269 leaves 1,6 is root0 and weak-stage positive, but the full A/B remainder is 301/325/24 at 2.952 ops/rho.

The newest selector-336_343 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 339 leaves 9,10 is verifier-backed and has selected-root/linear recovery at the factor stage, but the full remainder expands to 436/465/29 at 4.088 ops/rho.

The newest selector-360_367 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 361 leaves 0,12 has selected-root/linear recovery at the factor stage, but the full remainder expands to 277/300/23 at 2.72 ops/rho.

The newest selector-368_375 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 375 leaves 1,10 is root0 and weak-stage positive, but the full A/B remainder is 301/325/24 at 2.84 ops/rho.

The newest selector-440_447 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 441 leaf 1 is root0 and weak-stage positive, but the full A/B remainder is 254/276/22 at 2.448 ops/rho.

The newest selector-448_455 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 449 leaves 6,10 has selected-root/linear recovery at the factor stage, but the full remainder expands to 254/276/22 at 2.52 ops/rho.

The newest selector-456_463 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 458 leaf 10 is root0 and weak-stage positive, but the full A/B remainder is 301/324/24 at 2.824 ops/rho.

The newest selector-480_487 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 481 leaves 4,6 has zero original selected root pairs and keeps the weak factor-stage lead, but the full A/B remainder is 277/300/23 at 2.648 ops/rho.

The newest selector-488_495 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 492 leaves 7,10 has zero original selected root pairs and keeps the weak factor-stage lead, but the full A/B remainder is 326/351/25 at 3.04 ops/rho.

The newest selector-496_503 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 500 leaf 5 preserves selected-root/linear recovery at the factor stage, but the full A/B remainder is 172/190/18 at 1.864 ops/rho.

The newest selector-504_511 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 505 leaf 1 has zero original selected root pairs and keeps the weak factor-stage lead, but the full A/B remainder is 154/171/17 at 1.648 ops/rho.

The newest selector-512_519 target67 repeat consumes the next shared global and row-diverse recommendation without adding a full-remainder collapse. Transfer 518 leaves 9,12 has zero original selected root pairs and keeps the weak factor-stage lead, but the full A/B remainder is 254/276/22 at 2.464 ops/rho.

The final target67/salt205 queue sweep consumes selectors 528_535, 544_551, 560_567, 568_575, 576_583, 632_639, 640_647, and 648_655 without adding a full-remainder collapse. The closest of these final controls is transfer 569 leaves 8,10 at 1.96 ops/rho, followed by transfer 653 leaves 3,5 at 2.0 ops/rho; transfer 636 leaves 3,12 has no preserving factor candidate.

Root-scan and surface-stage below-rho candidates remain follow-up leads. They are weaker than full-remainder wins because they depend on a factor-stage path and a charged scan over known hit roots. The full-remainder wins at 420 are the cleaner evidence.

## Current status

This is not yet a novel ECDLP index-calculus algorithm. It is a validated candidate component:

1. public selector finds replayable exact profiles,
2. mounted campaign state can materialize the witness surfaces,
3. Sage finite-field factorization finds preserving factors,
4. three transfer-local pockets beat generic rho on full-remainder FFE cost,
5. immediate controls reject broad salt-only, mod6-only, mod16-only, literal `row_salt_transfer_mod32=165|10`, compact static-bank-only, and naive neighbor-salt explanations.
6. transfer/salt-forbidden public rule mining currently rejects a clean nonlocal preselector.
7. factor-root-scan labels have a broad public selector, but this currently stops below the stronger full-remainder threshold.
8. the fresh transfer-665 public-bounded follow-up rejects treating below-rho source-selector cost or root/surface-stage cheapness as sufficient evidence.
9. the fresh 672-679 public-bounded follow-up rejects treating verified `target_cap1` stress success as sufficient evidence, even though it improves the exact negative floor from 1.86131387 to 1.32116788.
10. the fresh 728-735 public-bounded follow-up reinforces that rejection: more verifier-labeled source positives still yield zero full-remainder positives and a worse exact floor of 2.272 ops/rho.
11. the first 136 deduped top-k12/root0 queue batches materialize 251 exact profile records and 249 unique exact surfaces across target67/salt205 and 22050/salt165 controls; the full-remainder below-rho queue positives remain four non-root0 22050/salt165 selected-root surfaces at transfers 242 and 294, all with the 67/78/11 shape. The final target67/salt205 sweep adds eight full-remainder negatives, and the queue now has 0 candidates remaining.
12. a queue-only public sparsity rerun with transfer, row-salt, source-window salt, and raw-leaf identity atoms forbidden finds no promotable simple pre-materialization selector: the best recall-1 clause still selects 4 negatives alongside the two positives.
13. the stricter queue-closed leaf-threshold audit also fails promotion: the best mined threshold rule selects 1 positive and 8 negatives, while the hand-checked high-leaf/root-pair proxy catches 4/4 positives but 30 negatives.
14. the new bounded manual-rule evaluator makes that falsification cheap to reproduce and shows the singleton/two-leaf split does not rescue the selector: the best singleton branch has 2 positives and 9 negatives, while the best two-leaf branch has 2 positives and 19 negatives.
15. an explicitly second-stage pre/factor gate does separate the closed queue: `original_selected_root_pair_count=1 & factor_root_scan_ops_over_rho<=0.46` selects all four 67/78/11 positives and zero negatives, but it uses factor-root-scan metadata and therefore needs held-out validation as a routing gate rather than a pure public selector.

The next milestone is either a frozen pre-materialization rule or a held-out-validated two-stage router that predicts the 67-monomial transfer-618/294 collapse family and the 92-monomial transfer-420 collapse before doing the expensive Sage full-remainder step. The top-k12/root0 queue now gives a disciplined source of fresh exact tests; promotion requires finding which, if any, of those queued factor-stage positives also collapse the full A/B remainder across held-out transfers or source groups.
