# Next hypothesis and work orders

## Working hypothesis

The current candidate family is a summation-polynomial/FFE relation-harvesting path where a public selector identifies row-transfer-leaf profiles whose quotient remainder collapses before full relation recovery. The observed below-rho evidence now has three pockets:

- target: `22050.cf1@11731`
- primary pocket: transfer 618, salt165, 8/8 below rho, 67 full-remainder monomials, 78 full-resultant monomials, known hit-root count 11, min ops/rho 0.81021898
- primary-pocket controls: transfer 618 salts 166, 171, 174, and 177, 44/44 materialized, 0 below rho, min ops/rho 1.09489051
- same-window controls: transfer 616 salts 161, 162, and 176, 12/12 materialized, 0 below rho, min ops/rho 1.32116788
- adjacent-transfer controls: transfer 619 salt165, 4/4 materialized, 0 below rho, min ops/rho 1.44525547; transfer 620 salt165, 4/4 materialized, 0 below rho, min ops/rho 2.05109489
- fresh queue pocket: transfer 294, salt165, leaf 90 and leaves 8,90, 2/2 below rho on those surfaces, 67 full-remainder monomials, 78 full-resultant monomials, known hit-root count 11, min ops/rho 0.84671533; the two-leaf transfer-294 surface is a non-root0 selected-root-pair/linear-recovery positive at 0.86131387 ops/rho; transfer 242, salt165, leaf 90 and leaves 8,90, 2/2 below rho as non-root0 selected-root-pair surfaces with the same 67/78/11 shape, min ops/rho 0.84671533
- fresh queue-pocket controls: transfer 295 salt165 leaf 90 is negative at 1.57664234 ops/rho, target67 transfer 242 leaf 10 is a 154/171/17 negative at 1.648 ops/rho, target67 transfer 253 leaf 0 is a 211/230/20 negative at 2.104 ops/rho, target67 transfer 273 leaf 12 has selected-root/linear recovery but is a 232/253/21 negative at 2.344 ops/rho, target67 transfer 278 leaves 1,3 is a 277/300/23 root0 negative at 2.648 ops/rho, target67 transfer 283 leaf 0 and leaves 0,9 are 352/378/26 root0 negatives at 3.232 and 3.248 ops/rho, target67 transfer 325 leaves 0,5 is a 254/276/22 root0 negative at 2.464 ops/rho, target67 transfer 326 leaf 7 is a 379/406/27 root0 negative at 3.448 ops/rho, target67 transfer 481 leaf 4 is a 277/300/23 root0 negative at 2.632 ops/rho, target67 transfer 482 leaf 9 has selected-root/linear recovery but is a 379/405/27 negative at 3.52 ops/rho, target67 transfers 579/582 are root0 negatives at 4.392 and 3.024 ops/rho, 22050 transfer 409 is a root0 326/351/25 negative at 2.70072993 ops/rho, 22050 transfer 424 is a selected-root/linear-recovery 277/300/23 negative at 2.37956204 ops/rho, earlier scanner-selected transfer 298 salt165 is negative at 2.34306569 ops/rho, follow-up transfers 389/391 salt165 are negative in both leaf-90 and two-leaf form with min ops/rho 1.86131387, transfer 464 is root0 but negative at 1.71532847 ops/rho, transfer 493 two-leaf is a near-control but still negative at 1.14598540 ops/rho, transfer 518 is root0 but negative at 1.71532847 ops/rho, transfer 209 two-leaf is root0 but negative at 1.45985401 ops/rho, transfer 226 two-leaf is root0 but negative at 1.18248175 ops/rho, transfer 271 two-leaf has no preserving factor candidate and is negative at 1.80291971 ops/rho, target67 transfer 239 is negative at 2.824 ops/rho, target67 transfers 296/303 are negative at 1.792 and 2.824 ops/rho, target67 transfers 368/375 are negative at 2.632 and 2.824 ops/rho, target67 transfers 377/378 are negative with min 1.792 ops/rho despite transfer 377 selected root-pair/linear recovery at the factor stage, target67 transfers 400/402/407/412 are negative with min 2.272 ops/rho despite transfers 407/412 selected root-pair/linear recovery at the factor stage, target67 transfers 518/519 are root0 but negative with min 2.448 ops/rho, transfer 308/311 two-leaf follow-ups remain negative with min ops/rho 1.76642336, transfer 363 leaf-90/three-leaf follow-up remains negative with min ops/rho 1.32116788, transfer 511 multi-leaf follow-up remains negative with min ops/rho 1.79562044, transfer 530 leaf-90/two-leaf follow-up remains a near-control but still negative with min ops/rho 1.20437956, transfer 556 leaf-90/two-leaf follow-up is root0 but negative with min ops/rho 2.01459854, transfer 572 leaf-90/two-leaf follow-up is root0 but negative with min ops/rho 4.18248175, transfer 591 two-leaf follow-up has selected root-pair recovery at the factor stage but stays full-remainder negative at 2.39416058 ops/rho, transfer 623 leaf-90/two-leaf follow-up remains negative with min ops/rho 1.71532847, transfer 635/637 mixed follow-up remains negative with min ops/rho 2.17518248, target67 transfer 490 is root0 but negative at 1.792 ops/rho with a 172/190/18 shape, target67 transfer 492 is root0 but negative at 3.024 ops/rho with a 326/351/25 shape, target67 transfers 304/309 have selected-root/linear recovery but remain negative with min ops/rho 2.16, target67 transfers 436/438 remain negative with min ops/rho 1.648 despite transfer 438 being verifier-backed, target67 transfers 449/454 remain root0 negatives with min ops/rho 1.944, and target67 transfers 337/339 remain negative with min ops/rho 3.672 despite transfer 339 being verifier-backed
- newest target67 controls: transfers 633/634 are root0 but full-remainder negative with min 2.104 ops/rho; transfer 633 leaf 9 forms a 211/231/20 negative at 2.104 ops/rho, and transfer 634 leaf 5 expands to 301/325/24 at 2.824 ops/rho
- newest target67 controls: transfers 361/365 are root0 but full-remainder negative with min 2.104 ops/rho; transfer 365 leaf 7 forms a 211/231/20 negative at 2.104 ops/rho, and transfer 361 leaf 0 forms a 277/300/23 negative at 2.632 ops/rho
- newest target67 controls: transfers 676/678 are full-remainder negative with min 3.488 ops/rho; transfer 676 leaf 8 has selected root-pair/linear recovery at the factor stage but expands to 379/406/27 at 3.488 ops/rho, and transfer 678 leaf 2 is root0 but expands to 407/435/28 at 3.672 ops/rho
- newest target67 control: transfer 225 is root0 but full-remainder negative at 3.448 ops/rho with a 379/406/27 shape
- newest target67 control: transfer 233 is root0 but full-remainder negative at 2.104 ops/rho with a 211/231/20 shape
- newest target67 two-leaf controls: transfers 233/239 are root0 but full-remainder negative with min 2.12 ops/rho; transfer 233 leaves 1,4 forms a 211/231/20 negative, and transfer 239 leaves 4,7 expands to 301/325/24 at 2.84 ops/rho
- newest target67 transfer 470 controls are root0 but full-remainder negative with min 2.824 ops/rho; leaf 10 and leaves 1,10 both share the 301/325/24 band
- newest target67 transfer 597/592 controls have selected-root/linear recovery but remain full-remainder negative with min 2.016 ops/rho; transfer 597 leaf 7 is 191/210/19, while transfer 592 leaves 7,10 is 211/231/20
- newest target67 transfer 350 controls have selected-root/linear recovery but remain full-remainder negative with min 2.0 ops/rho; leaf 3 and leaves 3,9 share the 191/210/19 band
- newest target67 transfer 475 controls mix root0 and selected-root/linear recovery but remain full-remainder negative with min 2.104 ops/rho; leaf 2 is 211/231/20, while leaves 2,4 has selected-root/linear recovery at the same 211/231/20 band and 2.192 ops/rho
- newest target67 transfer 297/303 controls mix root0 and selected-root/linear recovery but remain full-remainder negative with min 2.912 ops/rho; transfer 303 leaves 0,4 has selected-root/linear recovery at 301/325/24, while transfer 297 leaves 1,5 is root0 and expands to 436/464/29 at 3.92 ops/rho
- newest target67 transfer 612/615 controls mix root0 and selected-root/linear recovery but remain full-remainder negative with min 1.512 ops/rho; transfer 615 leaf 0 is a root0 137/153/16 near-control, while transfer 612 leaf 10 has selected-root/linear recovery but expands to 436/465/29 at 3.992 ops/rho
- newest target67 transfer 632/634/637 controls mix root0 and selected-root/linear recovery but remain full-remainder negative with min 1.28 ops/rho; transfer 632 leaves 3,12 is a root0 106/120/14 near-control, transfer 634 leaves 0,5 has selected-root/linear recovery at 301/325/24 and 2.912 ops/rho, and transfer 637 leaves 1,2 has selected-root/linear recovery at 407/435/28 and 3.76 ops/rho
- newest target67 transfer 251/253 controls split root0 and selected-root/linear recovery but remain full-remainder negative with min 2.192 ops/rho; transfer 253 leaves 0,7 has selected-root/linear recovery at 211/230/20 and 2.192 ops/rho, while transfer 251 leaves 4,5 is root0 but expands to 379/406/27 at 3.464 ops/rho
- newest target67 transfer 352 controls split root0 and selected-root/linear recovery but remain full-remainder negative with min 1.792 ops/rho; leaf 2 is a root0 172/190/18 negative, while leaves 2,4 has selected-root/linear recovery at the same 172/190/18 band and 1.848 ops/rho
- newest target67 transfer 201 controls split root0 and selected-root/linear recovery but remain full-remainder negative with min 3.448 ops/rho; leaf 6 is a root0 379/406/27 negative, while leaves 2,6 has selected-root/linear recovery at the same 379/406/27 band and 3.536 ops/rho
- newest target67 transfer 378/377 controls split a root0 weak-stage lead from a non-preserving one-root/linear-recovery factor-stage lead and remain full-remainder negative with min 1.808 ops/rho; transfer 378 leaves 4,8 is a root0 172/190/18 negative, while transfer 377 leaves 5,10 has no preserving factor candidate and expands to 352/378/26 at 3.408 ops/rho
- newest target67 transfer 622 controls are root0 and weak-stage positive but remain full-remainder negative with min 3.232 ops/rho; leaf 3 and leaves 3,7 both share the 352/378/26 band at 3.232 and 3.36 ops/rho
- newest target67 transfer 624 controls are non-root0 selected-root/linear-recovery negatives with min 3.504 ops/rho; leaf 5 preserves its selected root recovery but is 379/406/27, while leaves 3,5 has only a non-preserving one-root recovery and remains the same 379/406/27 band at 3.576 ops/rho
- newest target67 transfer 673/676 controls split root0 and selected-root/linear-recovery full-remainder negatives with min 1.664 ops/rho; transfer 673 leaves 3,10 is a root0 154/171/17 negative, while transfer 676 leaves 4,8 preserves a selected-root/linear recovery but expands to 379/406/27 at 3.504 ops/rho
- newest target67 transfer 225 two-leaf control has selected-root/linear recovery but remains full-remainder negative at 3.536 ops/rho with a 379/406/27 shape
- newest target67 transfer 242 two-leaf control is root0 but remains full-remainder negative at 1.664 ops/rho with a 154/171/17 shape
- newest target67 transfer 263 two-leaf control has selected-root/linear recovery but remains full-remainder negative at 3.744 ops/rho with a 407/435/28 shape
- newest target67 transfer 269 two-leaf control is root0 but remains full-remainder negative at 2.952 ops/rho with a 301/325/24 shape
- newest target67 transfer 339 two-leaf control is verifier-backed and has selected-root/linear recovery but remains full-remainder negative at 4.088 ops/rho with a 436/465/29 shape
- newest target67 transfer 361 two-leaf control has selected-root/linear recovery but remains full-remainder negative at 2.72 ops/rho with a 277/300/23 shape
- newest target67 transfer 375 two-leaf control is root0 but remains full-remainder negative at 2.84 ops/rho with a 301/325/24 shape
- newest target67 transfer 441 single-leaf control is root0 but remains full-remainder negative at 2.448 ops/rho with a 254/276/22 shape
- newest target67 transfer 449 two-leaf control has selected-root/linear recovery but remains full-remainder negative at 2.52 ops/rho with a 254/276/22 shape
- newest target67 transfer 500 single-leaf control preserves selected-root/linear recovery but remains full-remainder negative at 1.864 ops/rho with a 172/190/18 shape
- newest target67 controls: transfers 490/492 are root0 but full-remainder negative with min 1.792 ops/rho; transfer 490 leaf 7 forms a 172/190/18 negative at 1.792 ops/rho, and transfer 492 leaf 7 expands to 326/351/25 at 3.024 ops/rho
- newest target67 controls: transfers 304/309 have selected-root/linear recovery at the factor stage but remain full-remainder negative with min 2.16 ops/rho; transfer 309 leaf 1 forms a 211/231/20 negative at 2.16 ops/rho, and transfer 304 leaf 4 expands to 254/276/22 at 2.52 ops/rho
- newest target67 controls: transfers 436/438 remain full-remainder negative with min 1.648 ops/rho; transfer 438 leaf 12 is verifier-backed and has selected-root/linear recovery but forms a 137/153/16 negative at 1.648 ops/rho, while transfer 436 leaf 6 is root0 and expands to 407/435/28 at 3.672 ops/rho
- newest target67 controls: transfers 449/454 are root0 but full-remainder negative with min 1.944 ops/rho; transfer 454 leaf 10 forms a 191/210/19 negative at 1.944 ops/rho, and transfer 449 leaf 10 expands to 254/276/22 at 2.448 ops/rho
- newest target67 controls: transfers 337/339 remain full-remainder negative with min 3.672 ops/rho; transfer 337 leaf 10 is root0 and expands to 407/435/28 at 3.672 ops/rho, while transfer 339 leaf 10 is verifier-backed with selected-root/linear recovery and expands to 436/465/29 at 4.072 ops/rho
- newest target67 controls: transfer 283 leaf 0 and leaves 0,9 are root0 but full-remainder negative with min 3.232 ops/rho; both surfaces share the 352/378/26 band and remain weak-stage positives only
- newest target67 controls: transfers 481/482 remain full-remainder negative with min 2.632 ops/rho; transfer 481 leaf 4 is root0 at 277/300/23, while transfer 482 leaf 9 has selected-root/linear recovery at the factor stage but expands to 379/405/27
- newest target67 controls: transfers 273/278 remain full-remainder negative with min 2.344 ops/rho; transfer 273 leaf 12 has selected-root/linear recovery at 232/253/21, while transfer 278 leaves 1,3 is root0 at 277/300/23
- newest target67 controls: transfers 325/326 are root0 but full-remainder negative with min 2.464 ops/rho; transfer 325 leaves 0,5 is 254/276/22, while transfer 326 leaf 7 is 379/406/27
- secondary pocket: transfer 420, salts 165 and 167, 9 below-rho records, 92 full-remainder monomials, known hit-root count 13, min ops/rho 0.99270073

Rejected simple rules:

- `row_salt = 165`
- `transfer_index mod 6 = 0`
- `transfer_index mod 16 = 4`
- `transfer_index mod 16 = 10`
- literal `row_salt_transfer_mod32 = 165|10` outside the transfer-618 pocket
- `leaf_signature = 79`
- naive neighbor-salt transfer locality
- transfer-free public rules, except for a narrow transfer-420/salt167 neighbor-pocket selector
- transfer-and-salt-free public rules in the current feature set
- public-bounded source-selector cost by itself, as transfer 665 looked below rho before exact Sage materialization but has no full-remainder win
- verified `target_cap1` stress success by itself, as the fresh 672-679 selector has verifier-labeled below-rho source rows but exact Sage full-remainder cost still stays above rho
- verifier-labeled future-window target-cap1 source success by itself, as the fresh 728-735 selector has 38 verifier-labeled source positives but exact full-remainder cost is even worse
- factor-root-scan or surface-stage cheapness by itself, as transfers 234/237 and 665 are all weak-stage positives but full-remainder negatives

The next selector should try to predict:

```text
transfer 618-like collapse: full_remainder_monomials = 67, known_hit_root_count = 11
transfer 420-like collapse: full_remainder_monomials = 92, known_hit_root_count = 13
preserving factor exists
```

from public or pre-materialization features.

The first richer public-preselector audit did not find a cross-pocket rule. Its best all-data rule, `row_salt_transfer_mod32=165|10`, cleanly selects the transfer-618/salt165 positives but misses the transfer-420 positives. Leave-one-pocket-out validation also fails in both directions. Therefore the next selector search should either:

- build separate branch selectors for transfer-618-like and transfer-420-like collapses, or
- add genuinely new public structure beyond the current static-bank/schedule metadata before trying to merge them.

The first bank/schedule metadata audit did not merge the pockets. Static-bank metadata marks the salt165 row as a plausible seed (`low_term_span`, top-k 12, ops/rho 0.46715328, heldout_164), but this same metadata is present for the salt165 negative controls, including adjacent transfers 619 and 620. Schedule metadata did not join for the exact `uniform:256:salt165` row in the current mounted schedule artifact.

The cross-target branch analogue scanner added a no-lookahead way to rank public selector cases and exact-probe groups. It scanned 60 public bounded selector artifacts and, after the latest exact materializations, has no remaining unmaterialized literal `row_salt_transfer_mod32=165|10` profiles. The top target-67 analogue is not a full-remainder win.

The expanded public miner with these controls confirms the demotion: the old literal residue rule no longer survives as a top perfect rule. The best rules now say "this is transfer 618" through reference-distance atoms, so they are locality diagnostics rather than usable preselectors.

The forbidden-atom audit confirms the same boundary from the other direction:

- forbidding transfer atoms leaves a perfect but narrow `row_salt=167 & selected_leaf_index_count=1` rule that only selects three transfer-420 neighbor positives
- forbidding both transfer and salt aliases leaves no clean rule; the best selector, `profile_top_k=12 & selected_leaf_index_count=2`, has 4 positives and 7 negatives
- no current public envelope, compact bank, schedule, leaf-count, or policy atom predicts both pockets without transfer/salt locality

## Immediate exact probes already completed

The remaining selected salt165 transfers from the coverage set were materialized:

- 480
- 540
- 588
- 594
- 606
- 618
- 630

The new positive is transfer 618. The new negatives are 480, 540, 588, 594, 606, and 630.

Transfer 618 neighbor controls were also materialized for salts 166, 171, 174, and 177. None beat rho on the full remainder.

Same-window and adjacent transfer controls were materialized for transfer 616 local rows and transfers 619/620 at salt165. None beat rho on the full remainder.

The scanner-selected literal branch analogues were materialized:

- transfer 362, salt165, 4/4 materialized, 0 below rho, min ops/rho 2.21167883
- transfer 298, salt165, 4/4 materialized, 0 below rho, min ops/rho 2.34306569

The best target-67 cross-target analogue was materialized:

- target `67.a1@9803`, transfer 359, salt204, 32/32 materialized, 0 below rho, min ops/rho 1.512

The scanner's next mixed-row 232-239 batch was materialized after rerunning with mounted campaign sources:

- target `22050.cf1@11731`, transfer 234, salt173
- target `22050.cf1@11731`, transfer 237, salt165
- requested profiles: 36
- exact profiles present in the selector: 36
- materialized exact profiles: 36
- materialization errors: 0
- full-remainder positives: 0
- min full-remainder ops/rho: 1.57664234
- factor-root-scan positives: 36
- min factor-root-scan ops/rho: 0.33576642
- surface-stage positives: 36
- min surface ops/rho: 0.34306569

The earlier `candidate spec not found` result was caused by using local relative default sources instead of the mounted bank/direct witness files. The existing 232-239 orientation audit verifies rank for public linear-factor x-matches but has no measured below-rho rule: min measured ops/rho is 1.01459854, while the 0.98540146 value is only the shared-leaf-hit-root charged model.

The fresh 664-671 public-bounded follow-up was materialized:

- target `22050.cf1@11731`, transfer 665
- row salts 161, 166, and 169
- requested profiles: 48
- materialized exact profiles: 48
- materialization errors: 0
- verified cases: 0
- full-remainder positives: 0
- min full-remainder ops/rho: 1.86131387
- factor-root-scan positives: 48
- min factor-root-scan ops/rho: 0.33576642
- surface-stage positives: 48
- min surface ops/rho: 0.34306569

This is a controlled negative for promoting public-bounded source cost or factor-stage cheapness to a full ECDLP speedup signal.

The fresh 672-679 public-bounded follow-up was materialized:

- selector artifact: `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_672_679.json`
- exact artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer677_salt164_and_67_transfer673_salt208_public_bounded_672_679_mounted_sources.json`
- target `22050.cf1@11731`, transfer 677, salt164, best verified source case selected leaves `65,79,90`
- target `67.a1@9803`, transfer 673, salt208, best alternate source case selected leaves around `3,6,10`
- selector cases: 216
- verifier-labeled selector positives: 12
- selected min source ops/rho: 0.39416058
- requested profiles: 36
- materialized exact profiles: 36
- materialization errors: 0
- verified cases: 4
- full-remainder positives: 0
- min full-remainder ops/rho: 1.32116788
- factor-root-scan positives: 36
- min factor-root-scan ops/rho: 0.33576642
- surface-stage positives: 36
- min surface ops/rho: 0.34306569

The strongest verified `target_cap1` source case is still exact-negative: selected signature `65,79,90` costs 1.41605839 ops/rho under the full remainder, and selected signature `8,65,79,90` costs 1.43065693. The global exact minimum 1.32116788 comes from singleton leaf `79` profiles on the same transfer/row.

The fresh 728-735 public-bounded follow-up was materialized:

- selector artifact: `ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_728_735.json`
- exact artifact: `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_transfer733_salt166_and_67_transfer730_salt204_public_bounded_728_735_mounted_sources.json`
- target `22050.cf1@11731`, transfer 733, salt166, best verified source case selected leaves `8,56,90`
- target `67.a1@9803`, transfer 730, salt204, best alternate source case selected leaves `3,4,10`
- selector cases: 239
- verifier-labeled selector positives: 38
- selected min source ops/rho: 0.39416058
- requested profiles: 36
- materialized exact profiles: 36
- materialization errors: 0
- verified cases: 5
- full-remainder positives: 0
- min full-remainder ops/rho: 2.272
- factor-root-scan positives: 34
- min factor-root-scan ops/rho: 0.33576642
- surface-stage positives: 34
- min surface ops/rho: 0.34306569

This is a strong exact negative for the current target-cap1 promotion path. It has more source-side verifier labels than 672-679, but no full-remainder collapse and a worse exact full-remainder floor.

The expanded preselector miner was rerun with these controls:

- records: 252 after adding the mounted-source 234/237 exact artifact
- full-remainder positives: 17
- preserving candidates: 239
- best preselector now: `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`

After adding transfer 665, the full-remainder miner has:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_665_branch_scanner_controls_plus234237_plus665_preselector_holdout.json`
- records: 300
- full-remainder positives: 17
- preserving candidates: 287
- factor-root-scan positives: 287
- surface-stage positives: 287
- best preselector still: `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`

The transfer/salt-alias-free rerun with transfer 665 added has no clean public selector:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_665_branch_scanner_controls_plus234237_plus665_transfer_salt_alias_free_preselector.json`
- best preselector: `original_selected_root_pair_count=0 & selected_leaf_signature=8,90`
- selected positives: 4
- selected negatives: 9
- precision: 0.307692
- recall: 0.235294

After adding 672-679, the full-remainder miner has:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_672679_branch_scanner_controls_plus234237_plus665_plus672679_preselector_holdout.json`
- records: 336
- full-remainder positives: 17
- positive sources still only `t420_salt165`, `t420_neighbors`, and `t618_salt165`
- best preselector still: `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`

The transfer/salt-alias-free rerun with 672-679 added still has no clean public selector:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_672679_branch_scanner_controls_plus234237_plus665_plus672679_transfer_salt_alias_free_preselector.json`
- best preselector: `original_selected_root_pair_count=0 & selected_leaf_signature=8,90`
- selected records: 13
- positives: 4
- negatives: 9
- precision: 0.307692
- recall: 0.235294

After adding 728-735, the full-remainder miner has:

- no-holdout artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_preselector_no_holdout.json`
- compact holdout artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_preselector_holdout_compact.json`
- the original verbose holdout failed to write with `No space left on device`; the compact rerun omits repeated embedded `selected_records` samples and preserves the holdout counts/metrics
- records: 372
- full-remainder positives: 17
- positive sources still only `t420_salt165`, `t420_neighbors`, and `t618_salt165`
- best preselector still: `bank_best_filter_mode=low_term_span & transfer_abs_delta_from_618<=0`
- compact holdout coverage: 25 transfer-index splits, 15 row-salt splits, and 24 source-label splits
- compact holdout outcome: zero held-out full-remainder positives are selected by the top-5 trained rules for transfer-index, row-salt, or source-label holdout

Use this as a hard boundary: the current public preselector remains useful as an in-sample locator for the known `t618_salt165` branch, but not as evidence for cross-pocket generalization.

The enriched feature miner adds a better second-stage candidate:

- new public feature families: selected-leaf shape and static-bank source provenance
- identity filters: no transfer atoms, no row-salt atoms, no exact source-window salt labels/starts/offsets, no raw selected-leaf identity
- clause-3/support-4 artifact: `ecdlp_index_calculus_state/ffe_full_remainder_enriched_public_shape_source_miner_378_728735_transfer_salt_rawleaf_free_clause3_support4_no_holdout.json`
- strict compact holdout artifact: `ecdlp_index_calculus_state/ffe_full_remainder_enriched_public_shape_source_miner_378_728735_transfer_salt_rawleaf_free_clause3_support4_holdout_compact.json`
- fixed-rule audit artifact: `ecdlp_index_calculus_state/ffe_full_remainder_fixed_rule_topk12_root0_profile12_group_audit.json`
- best rule: `bank_best_filter_top_k=12 & original_selected_root_pair_count=0 & profile_top_k=12`
- corpus result: 16 selected, 14 positives, 2 negatives, precision 0.875, recall 0.823529
- leave-one-transfer holdout: train without transfer 420, then select 6/9 held-out transfer-420 positives with 2 negatives; train without transfer 618, then select 8/8 held-out transfer-618 positives with 0 negatives
- leave-one-source holdout: train without `t420_salt165`, then select 6/6 held-out positives with 2 negatives; train without `t618_salt165`, then select 8/8 held-out positives with 0 negatives
- leave-one-row-salt holdout: no held-out positive hits, so the candidate is still salt-local

Treat this as the next candidate, not a claim. It still selects only salt165 in the current corpus. The next bounded validation should be a fresh exact materialization batch for rows matching `bank_best_filter_top_k=12`, `original_selected_root_pair_count=0`, and `profile_top_k=12` outside the already-counted 420/618 pockets.

The transfer/salt-alias-free rerun with 728-735 added still has no clean public selector:

- artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_transfer_salt_alias_free_preselector.json`
- best preselector: `original_selected_root_pair_count=0 & selected_leaf_signature=8,90`
- selected records: 13
- positives: 4
- negatives: 9
- precision: 0.307692
- recall: 0.235294

The transfer/salt-forbidden preselector miners were also run:

- transfer-free artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_branch_scanner_controls_transfer_free_preselector_holdout.json`
- transfer-and-salt-free artifact: `ecdlp_index_calculus_state/ffe_full_remainder_exact_profile_rule_miner_salt165_mod6_378_660_branch_scanner_controls_transfer_salt_alias_free_preselector_holdout.json`

The miner now supports `--label-mode`. Running the factor-root-scan label on the 252-record dataset gives a very broad weaker selector:

- artifact: `ecdlp_index_calculus_state/ffe_factor_root_scan_rule_miner_salt165_mod6_378_660_branch_scanner_controls_plus234237_preselector_holdout.json`
- label positives: 239/252
- best preselector: `selected_leaf_index_count=1`
- selected positives: 193
- selected negatives: 0
- recall: 0.807531

This is useful for triaging factor-stage leads, but it is too weak to claim an index-calculus speedup because it does not include full A/B remainder recovery.

With transfer 665 included, the factor-root-scan label remains broad:

- artifact: `ecdlp_index_calculus_state/ffe_factor_root_scan_rule_miner_salt165_mod6_378_665_branch_scanner_controls_plus234237_plus665_preselector.json`
- label positives: 287/300
- best preselector: `selected_leaf_index_count=1`
- selected positives: 234
- selected negatives: 0
- recall: 0.815331

This makes `selected_leaf_index_count=1` a useful triage gate, not a promotion gate.

With 672-679 included, the factor-root-scan label remains broad:

- artifact: `ecdlp_index_calculus_state/ffe_factor_root_scan_rule_miner_salt165_mod6_378_672679_branch_scanner_controls_plus234237_plus665_plus672679_preselector.json`
- label positives: 323/336
- best preselector: `selected_leaf_index_count=1`
- selected positives: 258
- selected negatives: 0
- recall: 0.798762

This keeps `selected_leaf_index_count=1` as a weak-stage triage gate only.

With 728-735 included, the factor-root-scan label remains broad:

- artifact: `ecdlp_index_calculus_state/ffe_factor_root_scan_rule_miner_salt165_mod6_378_728735_branch_scanner_controls_plus234237_plus665_plus672679_plus728735_preselector.json`
- label positives: 357/372
- best preselector: `selected_leaf_index_count=1`
- selected positives: 284
- selected negatives: 0
- recall: 0.795518

This again keeps `selected_leaf_index_count=1` as a weak-stage triage gate only.

## Next exact probes

Prioritize probes that can separate the transfer 618 full-remainder rule from one-off overfit:

- same transfer 618, wider salt neighborhood if the selector exposes more rows outside the current 616-623 artifact
- other selected windows around a new candidate branch, since the 616-620 neighborhood around 618 is now a mostly negative locality control
- other targets or rows from the scanner that combine public below-rho, verifier-label support, and static-bank provenance beyond the compact `low_term_span` field
- repeat the transfer 618 profile with a frozen no-lookahead selector that does not use full-remainder outputs
- for factor-root-scan leads, add a second-stage public filter that predicts which root-scan/surface positives also collapse the full A/B remainder
- avoid rerunning transfer 665 as a win candidate until a new pre-materialization feature explains why its public-bounded source cost produced zero verified cases
- avoid promoting fresh `target_cap1` stress wins from 672-679 without a new pre-materialization feature that predicts full-remainder monomial collapse, not just factor/root-scan cheapness
- avoid promoting the future-window `target_cap1` family from 728-735 without a feature that separates it from the now-repeated full-remainder negatives
- treat `ecdlp_index_calculus_state/ffe_full_remainder_topk12_root0_candidate_queue.json` as the closed exact-materialization queue for the enriched top-k12/root0 shell; it currently has zero remaining candidates
- current queue-result triage: `ecdlp_index_calculus_state/ffe_full_remainder_topk12_root0_queue_result_triage.json`
- queue-only public sparsity miner after thirty-eight batches: `ecdlp_index_calculus_state/ffe_full_remainder_queue38_public_sparsity_miner_transfer_salt_rawleaf_free_holdout_compact.json`
- closed-queue public sparsity miner after 136 batches: `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_public_sparsity_miner_transfer_salt_rawleaf_free_holdout_compact.json`
- pass the live `/Volumes/Volume/autolab/ecdlp_index_calculus_state` bank/config/direct/transfer sources explicitly when running queued exact profiles; local worktree defaults do not contain those source artifacts
- validate the `pre_factor_stage` gate on independent or widened exact-profile data before using it as a work-saving router: the rule to audit is `original_selected_root_pair_count=1 & factor_root_scan_ops_over_rho<=0.46`, and the failure mode to watch for is hidden overfit to the transfer-242/294 closed-queue pockets
- after the first 136 top-k12/root0 queue batches, treat factor-root-scan wins as triage only until a queued surface also beats rho on full A/B remainder cost; current aggregate is 251 exact profile records, 249 unique surfaces, 166 unique root0 surfaces, 243/249 unique factor-root-scan positives, 4 full-remainder positives, best queue full-remainder cost 0.84671533 ops/rho
- the four queue full-remainder positives are all non-root0 22050/salt165 selected-root surfaces: transfer 294 leaf 90 and leaves 8,90, plus transfer 242 leaf 90 and leaves 8,90; all four share the 67-remainder/78-resultant/known-hit-root-11 shape
- the older queue-only miner over the first thirty-eight batches found no simple guarded public selector for the then-two positives: the best recall-1 clause selected 6 surfaces with 2 positives and 4 negatives, so the current leaf/policy/root-pair envelope is still a work-order, not a promotion rule
- the closed-queue miner over 251 records and 4 positives also finds no promotable guarded public selector: its best clause selects 1 positive and 8 negatives, while the best recall-0.5 clause selects 2 positives and 9 negatives; transfer holdout at 242 or 294 recovers at most one of two held-out positives, and source-label holdout misses the two newer two-leaf positive sources
- the stricter queue-closed threshold audit adds selected-leaf min/max/span/sum/gap cut atoms while still forbidding transfer identity, salt aliases, source-window salt/start/offset aliases, and raw leaf identity; it still finds no promotable rule. The best mined threshold clause selects 1 positive and 8 negatives, and the direct proxy `original_selected_root_pair_count=1 & selected_leaf_max>=80` catches all 4 positives only by also selecting 30 negatives. The attempted max-clause-3/support-2 sweep did not leave an output artifact, so do not cite it as evidence.
- the bounded manual-rule evaluator now scores hand-written structural proxies without clause enumeration. Its queue-closed audit artifact is `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_manual_structural_rule_evaluation.json`. It confirms the obvious split is still not enough: singleton high-leaf/root-pair rules select 2 positives and 9 negatives, while two-leaf high-gap/high-span rules select 2 positives and 19 negatives. The broad top-k12/profile-top-k/high-leaf shell catches all positives only with 91 negatives.
- the new `pre_factor_stage` evaluator gives the strongest current routing lead: `original_selected_root_pair_count=1 & factor_root_scan_ops_over_rho<=0.46` selects all four closed-queue 67/78/11 positives and zero negatives in `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_pre_factor_gate_evaluation.json`. This is not a pure public selector because it uses factor-root-scan metadata, but it is a plausible second-stage gate before full A/B remainder work. Factor-root-scan cost alone is much too broad, selecting 4 positives and 166 negatives. The grouped fixed-rule check has zero selected-negative transfer, source-label, or source-label-transfer groups; transfer 242 and transfer 294 are each covered at 2/0, and all four positive source buckets are covered at 1/0.
- the broader 372-record pre-factor validation in `ecdlp_index_calculus_state/ffe_full_remainder_broader_pre_factor_gate_evaluation.json` demotes that queue gate as a universal rule: `original_selected_root_pair_count=1 & factor_root_scan_ops_over_rho<=0.46` selects 0/17 positives on the 420/618 broader corpus, while `original_selected_root_pair_count=0 & factor_root_scan_ops_over_rho<=0.46` selects all 17 positives but also 210 negatives. The useful broad branch is instead `bank_best_filter_top_k=12 & profile_top_k=12 & factor_root_scan_ops_over_rho<=0.46`, selecting 14 positives and 2 negatives; it covers `t420_salt165` and `t618_salt165` but misses the three `t420_neighbors` positives whose bank top-k field is null. Treat this as evidence for a branch-family/DNF router, not a single transferred gate.
- the first `--evaluate-rule-family` DNF pass is implemented and scored in `ecdlp_index_calculus_state/ffe_full_remainder_broader_pre_factor_gate_family_evaluation.json` and `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_pre_factor_gate_family_evaluation.json`. The naive family combining the queue `root_pair=1/factor<=0.46` branch with the broad `top_k12/profile_top_k12/factor<=0.46` branch remains 14/2 on the broad corpus but explodes to 4 positives and 166 negatives on the closed queue. The follow-up context probes in `ecdlp_index_calculus_state/ffe_full_remainder_broader_pre_factor_context_gate_probe.json` and `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_pre_factor_context_gate_probe.json` show that bank provenance atoms do not reduce this queue noise, and source/policy refinements cut broad recall to 7/17 while still selecting 55 queue negatives. Next DNF work must find a true branch-context predicate, not just OR the existing gates.
- the preserving-count DNF family is the strongest current second-stage classifier: `ecdlp_index_calculus_state/ffe_full_remainder_broader_pre_factor_preserving_count_family_evaluation.json` gives 17 positives and 3 negatives on the broad corpus, `ecdlp_index_calculus_state/ffe_full_remainder_queue_closed_pre_factor_preserving_count_family_evaluation.json` gives 4 positives and 0 negatives on the closed queue, and `ecdlp_index_calculus_state/ffe_full_remainder_combined_pre_factor_preserving_count_family_evaluation.json` gives 21 positives and 3 negatives on the combined 623-record corpus. The stricter single-leaf variant in `ecdlp_index_calculus_state/ffe_full_remainder_combined_pre_factor_preserving_count_single_leaf_family_probe.json` improves the combined score to 21 positives and 0 negatives: `root_pair=1/factor<=0.46` covers the queue positives, `profile_top_k=12 & preserving_candidate_count=11 & factor<=0.46` covers the 618 pocket, and `profile_top_k=12 & preserving_candidate_count=13 & selected_leaf_index_count=1 & factor<=0.46` covers the 420 pocket including neighbor positives. Do not promote this as a work-saving selector yet: a timing audit shows `preserving_candidate_count` is emitted after exact surface materialization and Sage factorization. The next implementation work should be an early-stop/stage-split probe that measures whether preserving counts and factor-root-scan cost can be charged before the expensive full-remainder work, or else a public/pre-factor proxy for preserving-count 11/13.
- the new stage-split charge audit in `ecdlp_index_calculus_state/ffe_full_remainder_combined_pre_factor_preserving_count_single_leaf_stage_split_charge_audit.json` replays the same family from source artifacts and confirms 21 selected positives, 0 selected negatives, and 0 missed positives. As a proxy only, selected candidate-remainder follow-up is 18.89051097 summed ops/rho versus 1260.08157688 for all records; factor-root-scan on all records plus selected candidate-remainder follow-up is 286.29909447, or 0.22720679 of the all-record candidate-remainder proxy. The audit leaves exact materialization and Sage factorization uncharged, so the next step is to instrument those stages or design a pre-factor proxy before making any speedup claim.
- follow-up controls at transfers 389/391 leaf-90 and two-leaf variants, 416/421 leaf-90 and two-leaf variants, 464, 493/488 leaf-90 and two-leaf variants, 518, 209 leaf-90 and two-leaf variants, 226 leaf-90 and two-leaf variants, 257 single- and two-leaf variants, 249, 299, 316, 321, 334, 585/589/590/591, 596, 623, 629, 635/637, 735/732, 269, 271, 509/511, 530/556/572, 595/596/599, 308/311, 363, 354/397, 405, 649/655, 245, and the accumulated target67/salt205 controls did not reproduce the transfer-294 full-remainder win; the target67 controls now include transfers 201/225/233/239/242/251/253/257/258/263/273/278/283/296/303/297/300/304/309/312/313/325/326/337/339/350/352/360/361/365/366/368/375/377/378/395/400/402/407/412/436/438/441/449/454/475/481/482/490/492/518/519/522/537/579/582/607/612/615/632/633/634/636/676/678/460/463/545/548/561/567/569/574/592/597/653/673/674; transfers 493 and 226 are the nearest root0-style follow-ups so far at 1.09489051 ops/rho, transfer 493 two-leaf is the next nearest nonroot0 two-leaf follow-up at 1.14598540 ops/rho, transfer 226 two-leaf is a root0 near-control at 1.18248175 ops/rho, target67 transfer 632 is the closest target67 root0 near-control so far at 1.264 ops/rho, target67 transfer 615 repeats the 137/153/16 negative floor at 1.512 ops/rho, target67 transfer 589 repeats the 137/153/16 negative floor at 1.512 ops/rho, target67 transfer 537 repeats the 154/171/17 negative band at 1.648 ops/rho, target67 transfer 352 is a root0 and selected-root/linear-recovery 172/190/18 negative band with min 1.792 ops/rho, target67 transfer 258 is a selected-root/linear-recovery 172/190/18 negative at 1.848 ops/rho, target67 transfer 350 is a selected-root/linear-recovery 191/210/19 negative at 2.0 ops/rho, target67 transfer 475 leaf 2 is a root0 211/231/20 negative at 2.104 ops/rho, target67 transfer 475 leaves 2,4 is a selected-root/linear-recovery 211/231/20 negative at 2.192 ops/rho, target67 transfer 303 leaves 0,4 is a selected-root/linear-recovery 301/325/24 negative at 2.912 ops/rho, target67 transfer 201 is a root0 and selected-root/linear-recovery 379/406/27 negative band with min 3.448 ops/rho, target67 transfer 612 leaf 10 is a selected-root/linear-recovery 436/465/29 negative at 3.992 ops/rho, target67 transfer 297 leaves 1,5 is a root0 436/464/29 negative at 3.92 ops/rho, target67 transfer 312 is a root0 211/231/20 negative at 2.12 ops/rho, target67 transfer 233 leaves 1,4 is a root0 211/231/20 negative at 2.12 ops/rho, target67 transfer 239 leaves 4,7 is a root0 301/325/24 negative at 2.84 ops/rho, target67 transfer 441 is a root0 254/276/22 negative at 2.448 ops/rho, target67 transfer 522 repeats the 277/300/23 negative band at 2.632 ops/rho, target67 transfer 257 is a root0 277/300/23 negative at 2.648 ops/rho, target67 transfer 607 is a selected-root/linear-recovery 352/377/26 negative at 3.304 ops/rho, target67 transfer 597 is a selected-root/linear-recovery 191/210/19 negative at 2.016 ops/rho, target67 transfer 366 leaves 5,7 repeats the 379/406/27 negative band at 3.464 ops/rho, target67 transfer 360 is a 436/465/29 root0 negative at 3.904 ops/rho, target67 transfer 313 is a root0 436/465/29 negative at 3.92 ops/rho, transfer 572 expands to a 529/561/32 full-remainder negative band at 4.18248175 ops/rho, transfer 395 is a root0 596/630/34 negative at 5.184 ops/rho, and target67 transfer 561 is the worst target67 queue control so far at 667/703/36 and 5.752 ops/rho
- target67 transfer 632 leaves 3,12 is a root0 106/120/14 near-control at 1.28 ops/rho, target67 transfer 634 leaves 0,5 is a selected-root/linear-recovery 301/325/24 negative at 2.912 ops/rho, and target67 transfer 637 leaves 1,2 is a selected-root/linear-recovery 407/435/28 negative at 3.76 ops/rho
- target67 transfer 253 leaves 0,7 is a selected-root/linear-recovery 211/230/20 negative at 2.192 ops/rho, while target67 transfer 251 leaves 4,5 is a root0 379/406/27 negative at 3.464 ops/rho
- target67 transfer 352 leaf 2 is a root0 172/190/18 negative at 1.792 ops/rho, while leaves 2,4 has selected-root/linear recovery at the same 172/190/18 band and 1.848 ops/rho
- target67 transfer 201 leaf 6 is a root0 379/406/27 negative at 3.448 ops/rho, while leaves 2,6 has selected-root/linear recovery at the same 379/406/27 band and 3.536 ops/rho
- target67 transfer 378 leaves 4,8 is a root0 172/190/18 negative at 1.808 ops/rho, while transfer 377 leaves 5,10 has only a non-preserving one-root/linear recovery and remains a 352/378/26 negative at 3.408 ops/rho
- target67 transfer 622 leaf 3 and leaves 3,7 are root0 352/378/26 negatives at 3.232 and 3.36 ops/rho
- target67 transfer 624 leaf 5 is a preserving selected-root/linear-recovery 379/406/27 negative at 3.504 ops/rho, while leaves 3,5 has only a non-preserving one-root/linear recovery and remains 379/406/27 at 3.576 ops/rho
- target67 transfer 673 leaves 3,10 is a root0 154/171/17 negative at 1.664 ops/rho, while transfer 676 leaves 4,8 has selected-root/linear recovery and remains a 379/406/27 negative at 3.504 ops/rho
- target67 transfer 225 leaves 8,12 has selected-root/linear recovery and remains a 379/406/27 negative at 3.536 ops/rho
- target67 transfer 242 leaves 6,10 is a root0 154/171/17 negative at 1.664 ops/rho
- target67 transfer 263 leaves 6,8 has selected-root/linear recovery and remains a 407/435/28 negative at 3.744 ops/rho
- target67 transfer 269 leaves 1,6 is a root0 301/325/24 negative at 2.952 ops/rho
- target67 transfer 339 leaves 9,10 is verifier-backed with selected-root/linear recovery and remains a 436/465/29 negative at 4.088 ops/rho
- target67 transfer 361 leaves 0,12 has selected-root/linear recovery and remains a 277/300/23 negative at 2.72 ops/rho
- target67 transfer 375 leaves 1,10 is root0 and remains a 301/325/24 negative at 2.84 ops/rho
- target67 transfer 441 leaf 1 is root0 and remains a 254/276/22 negative at 2.448 ops/rho
- target67 transfer 449 leaves 6,10 has selected-root/linear recovery and remains a 254/276/22 negative at 2.52 ops/rho
- target67 transfer 458 leaf 10 is root0 and remains a 301/324/24 negative at 2.824 ops/rho
- newest target67 transfer 633 leaf 9 is a 211/231/20 negative at 2.104 ops/rho; newest target67 transfer 634 leaf 5 is a 301/325/24 negative at 2.824 ops/rho
- newest target67 transfer 365 leaf 7 is a 211/231/20 negative at 2.104 ops/rho; newest target67 transfer 361 leaf 0 is a 277/300/23 negative at 2.632 ops/rho
- newest target67 transfer 676 leaf 8 is a selected-root/linear-recovery 379/406/27 negative at 3.488 ops/rho; newest target67 transfer 678 leaf 2 is a root0 407/435/28 negative at 3.672 ops/rho
- newest target67 transfer 225 leaf 8 is a root0 379/406/27 negative at 3.448 ops/rho
- newest target67 transfer 233 leaf 4 is a root0 211/231/20 negative at 2.104 ops/rho
- newest 22050 transfer 294 leaves 8,90 is a non-root0 selected-root/linear-recovery 67/78/11 positive at 0.86131387 ops/rho
- newest target67 transfer 253 leaf 0 is a root0 211/230/20 negative at 2.104 ops/rho
- newest target67 transfers 582/579 are root0 negatives at 326/351/25 and 497/528/31 with minimum 3.024 ops/rho
- newest mixed transfer 409/412 batch is full-remainder negative: 22050 transfer 409 is root0 at 326/351/25 and 2.70072993 ops/rho, while target67 transfer 412 has selected-root/linear recovery but expands to 407/435/28 at 3.744 ops/rho
- newest 22050 transfer 424 leaf 90 is a selected-root/linear-recovery 277/300/23 negative at 2.37956204 ops/rho; this exhausts the current 22050/salt165 top-k12/root0 queue row
- newest target67 transfers 490/492 are root0 full-remainder negatives: transfer 490 leaf 7 is 172/190/18 at 1.792 ops/rho, and transfer 492 leaf 7 is 326/351/25 at 3.024 ops/rho
- newest target67 transfers 304/309 are selected-root/linear-recovery full-remainder negatives: transfer 309 leaf 1 is 211/231/20 at 2.16 ops/rho, and transfer 304 leaf 4 is 254/276/22 at 2.52 ops/rho
- newest target67 transfers 436/438 are full-remainder negatives: transfer 438 leaf 12 is verifier-backed and selected-root/linear-recovery at 137/153/16 and 1.648 ops/rho, while transfer 436 leaf 6 is root0 at 407/435/28 and 3.672 ops/rho
- newest target67 transfers 449/454 are root0 full-remainder negatives: transfer 454 leaf 10 is 191/210/19 at 1.944 ops/rho, while transfer 449 leaf 10 is 254/276/22 at 2.448 ops/rho
- newest target67 transfers 337/339 are full-remainder negatives: transfer 337 leaf 10 is root0 at 407/435/28 and 3.672 ops/rho, while transfer 339 leaf 10 is verifier-backed and selected-root/linear-recovery at 436/465/29 and 4.072 ops/rho
- newest target67 transfer 283 is root0 but full-remainder negative: leaf 0 is 352/378/26 at 3.232 ops/rho, and leaves 0,9 is 352/378/26 at 3.248 ops/rho
- newest target67 transfers 481/482 are full-remainder negatives: transfer 481 leaf 4 is root0 at 277/300/23 and 2.632 ops/rho, while transfer 482 leaf 9 has selected-root/linear recovery at 379/405/27 and 3.52 ops/rho
- newest target67 transfers 273/278 are full-remainder negatives: transfer 273 leaf 12 has selected-root/linear recovery at 232/253/21 and 2.344 ops/rho, while transfer 278 leaves 1,3 is root0 at 277/300/23 and 2.648 ops/rho
- newest target67 transfers 325/326 are root0 full-remainder negatives: transfer 325 leaves 0,5 is 254/276/22 at 2.464 ops/rho, while transfer 326 leaf 7 is 379/406/27 at 3.448 ops/rho
- newest target67 transfers 360/366 are root0 full-remainder negatives: transfer 366 leaves 5,7 is 379/406/27 at 3.464 ops/rho, while transfer 360 leaf 9 is 436/465/29 at 3.904 ops/rho
- newest target67 transfers 257/258 are full-remainder negatives: transfer 258 leaves 2,8 has selected-root/linear recovery at 172/190/18 and 1.848 ops/rho, while transfer 257 leaves 0,9 is root0 at 277/300/23 and 2.648 ops/rho
- newest target67 transfers 312/313 are root0 full-remainder negatives: transfer 312 leaves 0,6 is 211/231/20 at 2.12 ops/rho, while transfer 313 leaves 0,6 is 436/465/29 at 3.92 ops/rho
- newest target67 transfer 395 is a root0 full-remainder negative: leaf 6 and leaves 5,6 both share the 596/630/34 band with minimum 5.184 ops/rho
- newest target67 transfer 522 is a root0 full-remainder negative: leaf 6 and leaves 3,6 both share the 277/300/23 band with minimum 2.632 ops/rho
- newest target67 transfer 632/633/636 batch is full-remainder negative: transfer 632 leaf 12 is a root0 106/120/14 near-control at 1.264 ops/rho, transfer 633 leaves 2,9 is root0 at 211/231/20 and 2.12 ops/rho, and transfer 636 leaf 12 has selected-root/linear recovery but expands to 326/351/25 at 3.08 ops/rho
- newest target67 transfer 537 is a root0 full-remainder negative: leaf 6 and leaves 0,6 both share the 154/171/17 band with minimum 1.648 ops/rho
- newest target67 transfer 607 is a selected-root/linear-recovery full-remainder negative: leaf 4 and leaves 4,9 both share the 352/377/26 band with minimum 3.304 ops/rho
- newest target67 transfer 233/239 two-leaf batch is root0 but full-remainder negative: transfer 233 leaves 1,4 is 211/231/20 at 2.12 ops/rho, while transfer 239 leaves 4,7 is 301/325/24 at 2.84 ops/rho
- newest target67 transfer 470 batch is root0 but full-remainder negative: leaf 10 and leaves 1,10 share the 301/325/24 band with minimum 2.824 ops/rho
- newest target67 transfer 597/592 batch has selected-root/linear recovery but remains full-remainder negative: transfer 597 leaf 7 is 191/210/19 at 2.016 ops/rho, while transfer 592 leaves 7,10 is 211/231/20 at 2.224 ops/rho
- newest target67 transfer 350 batch has selected-root/linear recovery but remains full-remainder negative: leaf 3 and leaves 3,9 share the 191/210/19 band with minimum 2.0 ops/rho
- newest target67 transfer 475 batch mixes root0 and selected-root/linear recovery but remains full-remainder negative: leaf 2 is 211/231/20 at 2.104 ops/rho, while leaves 2,4 has selected-root/linear recovery at 2.192 ops/rho
- newest target67 transfer 297/303 batch mixes root0 and selected-root/linear recovery but remains full-remainder negative: transfer 303 leaves 0,4 is 301/325/24 at 2.912 ops/rho, while transfer 297 leaves 1,5 is 436/464/29 at 3.92 ops/rho
- newest target67 transfer 612/615 batch mixes root0 and selected-root/linear recovery but remains full-remainder negative: transfer 615 leaf 0 is 137/153/16 at 1.512 ops/rho, while transfer 612 leaf 10 is 436/465/29 at 3.992 ops/rho
- newest target67 transfer 632/634/637 batch mixes root0 and selected-root/linear recovery but remains full-remainder negative: transfer 632 leaves 3,12 is 106/120/14 at 1.28 ops/rho, transfer 634 leaves 0,5 is 301/325/24 at 2.912 ops/rho, and transfer 637 leaves 1,2 is 407/435/28 at 3.76 ops/rho
- newest target67 transfer 251/253 batch splits root0 and selected-root/linear recovery but remains full-remainder negative: transfer 253 leaves 0,7 is 211/230/20 at 2.192 ops/rho, while transfer 251 leaves 4,5 is 379/406/27 at 3.464 ops/rho
- newest target67 transfer 352 batch splits root0 and selected-root/linear recovery but remains full-remainder negative: leaf 2 is 172/190/18 at 1.792 ops/rho, while leaves 2,4 is 172/190/18 at 1.848 ops/rho
- newest target67 transfer 201 batch splits root0 and selected-root/linear recovery but remains full-remainder negative: leaf 6 is 379/406/27 at 3.448 ops/rho, while leaves 2,6 is 379/406/27 at 3.536 ops/rho
- newest target67 transfer 378/377 batch remains full-remainder negative: transfer 378 leaves 4,8 is root0 and 172/190/18 at 1.808 ops/rho, while transfer 377 leaves 5,10 has only a non-preserving one-root/linear recovery and is 352/378/26 at 3.408 ops/rho
- newest target67 transfer 622 batch is root0 but full-remainder negative: leaf 3 is 352/378/26 at 3.232 ops/rho, while leaves 3,7 stays in the same band at 3.36 ops/rho
- newest target67 transfer 624 batch is non-root0 and full-remainder negative: leaf 5 is a preserving selected-root/linear-recovery 379/406/27 negative at 3.504 ops/rho, while leaves 3,5 has only a non-preserving one-root recovery and stays 379/406/27 at 3.576 ops/rho
- newest target67 transfer 673/676 batch is full-remainder negative: transfer 673 leaves 3,10 is root0 and 154/171/17 at 1.664 ops/rho, while transfer 676 leaves 4,8 preserves selected-root/linear recovery but expands to 379/406/27 at 3.504 ops/rho
- newest target67 transfer 225 two-leaf batch is full-remainder negative: leaves 8,12 preserve selected-root/linear recovery but expand to 379/406/27 at 3.536 ops/rho
- newest target67 transfer 242 two-leaf batch is root0 and full-remainder negative: leaves 6,10 are 154/171/17 at 1.664 ops/rho
- newest target67 transfer 263 two-leaf batch is full-remainder negative: leaves 6,8 preserve selected-root/linear recovery but expand to 407/435/28 at 3.744 ops/rho
- newest target67 transfer 269 two-leaf batch is root0 and full-remainder negative: leaves 1,6 are 301/325/24 at 2.952 ops/rho
- newest target67 transfer 339 two-leaf batch is full-remainder negative: leaves 9,10 are verifier-backed and preserve selected-root/linear recovery but expand to 436/465/29 at 4.088 ops/rho
- newest target67 transfer 361 two-leaf batch is full-remainder negative: leaves 0,12 preserve selected-root/linear recovery but expand to 277/300/23 at 2.72 ops/rho
- newest target67 transfer 375 two-leaf batch is root0 and full-remainder negative: leaves 1,10 are 301/325/24 at 2.84 ops/rho
- newest target67 transfer 441 single-leaf batch is root0 and full-remainder negative: leaf 1 is 254/276/22 at 2.448 ops/rho
- newest target67 transfer 449 two-leaf batch is full-remainder negative: leaves 6,10 preserve selected-root/linear recovery but expand to 254/276/22 at 2.52 ops/rho
- newest target67 transfer 458 single-leaf batch is root0 and full-remainder negative: leaf 10 is 301/324/24 at 2.824 ops/rho
- newest target67 transfer 481 two-leaf batch is zero-root-pair and full-remainder negative: leaves 4,6 are 277/300/23 at 2.648 ops/rho
- newest target67 transfer 492 two-leaf batch is zero-root-pair and full-remainder negative: leaves 7,10 are 326/351/25 at 3.04 ops/rho
- newest target67 transfer 500 single-leaf batch preserves selected-root/linear recovery and is full-remainder negative: leaf 5 is 172/190/18 at 1.864 ops/rho
- newest target67 transfer 505 single-leaf batch is zero-root-pair and full-remainder negative: leaf 1 is 154/171/17 at 1.648 ops/rho
- newest target67 transfer 518 two-leaf batch is zero-root-pair and full-remainder negative: leaves 9,12 are 254/276/22 at 2.464 ops/rho
- newest target67 final queue sweep consumes the remaining controls without a full-remainder win: transfer 532 leaf 6 is 277/300/23 at 2.632 ops/rho, transfer 549 leaf 1 is 301/325/24 at 2.896, transfer 561 leaves 6,12 is 667/703/36 at 5.768, transfer 569 leaves 8,10 is 191/210/19 at 1.96, transfer 579 leaves 8,10 is 497/528/31 at 4.408, transfer 636 leaves 3,12 has no preserving factor candidate at 3.152, transfer 643 leaf 1 is 529/561/32 at 4.648, and transfer 653 leaves 3,5 is 191/210/19 at 2.0
- next global triage-recommended source batch: none; the current top-k12/root0 candidate queue is exhausted
- next row-diverse 22050/salt165 recommendation: none; the current queue has no remaining `22050.cf1@11731:uniform:256:salt165` candidates
- next row-diverse target67/salt205 recommendation: none; the current queue has no remaining `67.a1@9803:uniform:256:salt205` candidates
- keep the row-diverse recommendation block in the triage artifact; it exposed the transfer-294 positive that the previous global-only view hid behind target67/salt205 negatives

After any widened queue batch, rerun:

```bash
/Users/adamburan/.cache/vllm-venvs/vllm-312/bin/python tasks/ecdlp_index_calculus/ffe_full_remainder_salt165_holdout_coverage_audit.py --out ecdlp_index_calculus_state/ffe_full_remainder_salt165_mod6_holdout_coverage_audit_376_671.json
```

and rerun the public sparsity miner with the new artifacts included.

## Miner upgrades

Still-needed pre-materialization features that are visible before Sage full-remainder scoring:

- richer static-bank provenance features from the original source artifacts, not only the compact bank row summary
- target-normalized leaf/line-shape features, because raw leaf indices do not transfer between `22050.cf1@11731` and `67.a1@9803`
- original source-window provenance from `frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_family_union_seed_refresh_promotion_with_cached_all6_full_probe.json`, especially features that distinguish transfer 618 from scanner negatives 298 and 362
- pre-materialization features that separate the 67/78/11 positives from high-leaf false positives whose full remainders grow to 106/120/14, 137/153/16, 154/171/17, 191/210/19, and larger bands

Already added to the public sparsity miner:

- forbidden atom regex filtering for leakage audits
- selectable labels for full-remainder, factor-root-scan, and surface-stage below-rho mining
- transfer residues through mod 32
- row salt and row salt residues
- leaf signature and selected leaf count
- original selected root-pair count
- policy, top-k, leaf selector
- source row selector and source ops/rho
- static-bank row metadata and schedule-row metadata joins
- configured reference-transfer and reference-salt distance atoms
- leave-one-transfer, leave-one-source, and leave-one-salt preselector holdouts
- cross-target branch analogue scanner with exact-profile exclusion and per-target probe groups
- top-k12/root0 exact-materialization queue builder with surface-profile deduplication
- selected-leaf threshold cut atoms for min, max, span, sum, and gap min/max
- bounded manual rule evaluation with `--evaluate-rule` and `--skip-mining`
- `pre_factor_stage` evaluation mode and factor-root-scan op/rho cut atoms for second-stage gate audits

The miner should score two labels separately:

1. full-remainder below rho, the stronger algorithmic claim
2. factor-root-scan below rho, the weaker but still useful candidate lead

## Algorithm sketch to keep testing

1. Use public low-term selector features to choose row-transfer-leaf profiles.
2. Materialize exact summation-polynomial witness surfaces from the static bank and transfer witnesses.
3. Use Sage finite-field factorization to split the resultant surface.
4. Keep only factors preserving the selected root pairs.
5. Treat the preserving-count family as a post-materialization/factor-stage classifier until an instrumented early-stop probe charges exact materialization and Sage factorization. The current proxy audit says factor-root-scan on all records plus selected full-remainder follow-up is only 0.22720679 of all-record candidate-remainder follow-up, but that ratio excludes the stages required to obtain `preserving_candidate_count`.
6. Charge either full FFE remainder cost or the weaker factor-root-scan cost.
7. Compare every candidate to generic rho on the same target.
8. Promote only rules that survive neighbor salts and held-out transfers.
