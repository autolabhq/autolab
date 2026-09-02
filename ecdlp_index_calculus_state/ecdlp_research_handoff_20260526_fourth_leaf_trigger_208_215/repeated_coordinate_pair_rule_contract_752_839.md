# Repeated-Coordinate Pair-Rule Contract, 752-839

## Scope

This note records the current public-pruning contract for the target-67
repeated-coordinate branch.  It is an accounting boundary for the no-exact
pair/subset rule, not a completed general ECDLP index-calculus algorithm.

## Rule-Mining Corpus

- Training positives: available target-67 repeated-coordinate positive
  decompositions through 736-743.
- Held-out validation: 752-759.
- Exact coordinate and exact salt literals were disabled during mining.
- The 760-767, 768-775, 776-783, 784-791, 792-799, 800-807, 808-815,
  816-823, 824-831, and 832-839 windows were not used to choose the pair rule.

## Frozen Rule

Pair activation:

```text
pair_activate:b_minus_c_mod16=1&pair_salt_span>=4|b_minus_c_mod16=7&pair_salt_delta_from_min_signature=0,3|pair_salt_sum_mod8=0&transfer_mod5=1
```

Original replay guard:

```text
candidate_pos_span=0
```

Strict rank-lift replay guard:

```text
candidate_pos_span=0&candidate_pos_count>=2
```

The rule is public in the current model: it uses coordinate residues, transfer
residues, salt-span/signature features, and the event-stage form guard.  It does
not use exact coordinates, exact salts, or verifier outcomes from the replay
window.

The strict guard fixes a vacuity in the original guard.  A single candidate
position has span zero by definition, so 784-791 rank-1 singleton events passed
`candidate_pos_span=0` even though they could not derive.  The stricter guard
requires at least two candidate-position events at the same position before a
case is counted as guard-passed.

## Calibration And Validation

| Window | Role | Selected Pair Cases | Strict Guarded Below-Rho Cases | Cost | Result |
| --- | --- | ---: | ---: | --- | --- |
| 752-759 | held-out validation | 76 | 8 | min/mean 0.928 rho | derives secret 7062 |
| 760-767 | later control | 0 | 0 | n/a | clean abstention |
| 768-775 | fresh positive | 32 | 12 | min 0.512, mean 0.784 rho | derives secret 6683 |
| 776-783 | fresh positive | 52 | 4 | min/mean 0.928 rho | derives secret 4774 |
| 784-791 | fresh singleton negative | 72 | 0 | 16 rank-1 singletons rejected | no verification |
| 792-799 | fresh rank-0 negative | 12 | 0 | all selected cases rank 0 | no verification |
| 800-807 | fresh strict positive | 68 | 4 | min/mean 0.960 rho | derives secret 2023 |
| 808-815 | fresh strict negative | 67 | 0 | 11 rank-2 relation cases rejected; full replay 1.44 rho | no verification |
| 816-823 | fresh no-relation calibration | 116 | 0 | no relation pair cases | no verification |
| 824-831 | fresh no-relation calibration | 16 | 0 | no relation pair cases | no verification |
| 832-839 | fresh branch-gap calibration | 4 | 0 | strict pair misses coordinate-gate and row/form relation branches | no strict-pair relation cases |

## Candidate-Position Alignment Audit

Audit artifact:

```text
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_839.json
```

The audit replays all relation-bearing strict pair cases from 752-791, 792-799,
800-807, 808-815, 816-823, and 824-831 with the original coordinate-gate
sources, plus the 832-839 strict pair replay.  It records event-level
candidate-position, coefficient support, RHS residue, salt, and term-shape
features before using public-key verification only as a label.

Replay integrity is clean: 43 relation-bearing pair cases, zero context errors,
zero skipped cases, zero source replay mismatches, and zero form-guard feature
mismatches.  The count is unchanged from 752-815 because 816-823, 824-831, and
832-839 contribute no strict-pair relation cases.

Result:

- Verified relation cases: 28/43, all below rho.
- Unverified rank-2 decoys: 15/43.
- Verified candidate-position signatures: `1,1` in 16 cases and `2,2` in
  12 cases.
- Unverified candidate-position signature: `1,2` in all 15 decoy cases.
- Form-only positive separator: `candidate_pos_aligned=1`,
  `candidate_pos_span=0`, `candidate_pos_unique_count=1`, and
  `candidate_pos_span0_count_ge2=1` occur in all 28 verified cases and zero
  unverified cases.
- Form-only negative separator: `candidate_pos_aligned=0`,
  `candidate_pos_span=1`, and `candidate_pos_signature=1,2` occur in all
  15 unverified decoys and zero verified cases.

This supports the event-form orientation hypothesis: rank 2 is insufficient
unless the two relation events land on the same candidate position.  It does
not yet make alignment a free pre-replay selector.

## Claim Boundary

Full repeated-coordinate replay remains over rho in the fresh positive windows.
For 768-775, the best verified full replay is 1.328 rho.  For 776-783, the
best verified full replay derives secret 4774 at 1.336 rho, and the global
minimum replayed cost is 1.224 rho with zero below-rho verified full-coordinate
recoveries.  For 808-815, 11 full-coordinate replays verify, but none are below
rho; the best verified full replay is transfer 811, coordinate `(1114,8506)`,
deriving secret 7002 at 1.44 rho.

The candidate speedup is therefore not the full repeated-coordinate replay.  It
is public subset/pair pruning plus the `candidate_pos_span=0` form guard, which
keeps only relation-bearing row subsets before the final relation-system replay.

This remains a candidate family.  Promotion to a general algorithm needs at
least another disjoint unchanged-rule replication and a structural explanation
for why the public pair features predict relation-bearing rows.

The 784-791, 792-799, 800-807, and 808-815 follow-ups narrow that requirement.
The strict guard rejects the 784 singleton rank-1 cases, 792 has only rank-0
selected pairs, 800 recovers again when two relation events align to the same
candidate position, and 808 produces rank-2 relation decoys that misalign.  The
800 and 808 decoy relation pairs with signature `1,2` show that rank 2 alone is
not enough; candidate-position alignment appears to be the current public
orientation signal.  The 816-823, 824-831, and 832-839 windows add a different
boundary: public pair selection can spend 136 additional selected cases without
reaching any strict-pair relation case.

## Relation-Reach Audit

Relation-reach artifact:

```text
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_relation_reach_audit_target67_752_839.json
```

The relation-reach audit counts selected strict pair/subset cases before
candidate-position alignment and public-key verification.  Across 752-839 it
finds 515 selected pair cases, 43 relation-reaching cases, 28 guard-passed
verified below-rho cases, and 15 unverified relation-reaching decoys.

Dead-selection windows are 784-791, 792-799, 816-823, 824-831, and 832-839.
Relation reach occurs in 752-759, 768-775, 776-783, 800-807, and the 808-815
decoy window.  The strongest current caution is that relation reach and
candidate-position alignment are separate gates: 808-815 reaches relation cases
without verifying, while 816-823, 824-831, and 832-839 do not reach strict-pair
relation cases at all.

## Branch-Gap Audit

Branch-gap artifact:

```text
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_gap_audit_target67_832_839.json
```

The 832-839 window exposes a branch gap that the strict pair-reach table alone
would hide.  The coordinate gate replays 87 target-67 candidates, verifies 31
public keys, and verifies zero below rho.  The verified coordinate-gate branch
has two transfer/coordinate keys on `(55,7531)`, including transfer 834 at
1.36 rho, and the strict pair rule selects neither key.

The old row/form replay selects 67 cases and reaches 8 rank-2 relation cases
at transfer 835 on coordinate `(542,3911)`, all with candidate-position
signature `1,3`, nominal 0.552-rho row cost, and zero public-key verification.
The strict pair rule selects neither this key nor the coordinate-gate verified
keys; it selects only four rank-0 cases on transfer 838 and coordinate
`(161,6976)`.

## Branch-Selector Miner

Branch-selector artifact:

```text
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_752_839.json
```

The branch selector miner aggregates coordinate-gate candidates, old row/form
replay cases, and strict pair-rule cases by transfer/coordinate key.  It uses
replay and verifier labels only for scoring.  Candidate atoms exclude exact
coordinate keys, exact salt signatures, verifier labels, and candidate-position
labels.

The 752-839 corpus has 103 public transfer/coordinate keys.  It contains 13
coordinate-gate verified keys, 7 coordinate-gate verified keys missed by strict
pair selection, 2 row/form relation keys, 1 row/form relation key missed by
strict pair selection, 8 total branch-gap keys, and 36 strict-pair dead keys.

Best current branch-gap selector:

```text
b_mod4=3&salt_mod2_pattern=0,1,0
```

It selects 4/8 branch-gap keys, zero non-branch keys, and spans the 752-759,
776-783, and 832-839 windows.  Narrower coordinate-55 family candidates include
`b_minus_c_mod16=12&salt_mod2_pattern=0,1,0` and
`b_minus_c_mod16=12&salt_span=6`; each selects 3 branch-gap keys with zero
false positives across 776-783 and 832-839.

These rules are not speedup claims.  They are public branch selectors to test
before the strict pair rule.  The row/form `1,3` branch remains a single-window
diagnostic until another window repeats it.

## 840-847 Frozen Selector Check

Fresh branch-selector artifacts:

```text
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_gap_audit_target67_840_847.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_840_847_fixed_branch_rules.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_752_847.json
```

The 840-847 window validates the frozen branch selector without retuning.  The
coordinate gate replays 72 target-67 candidates and verifies 16 public keys,
all over rho.  The strict pair rule selects 32 cases, reaches 12 relation
cases, and verifies 8 public keys at 1.032 rho on transfer 846 coordinate
`(281,3224)`.

Strict pair still misses one verified coordinate-gate key: transfer 847 on
`(55,7531)` at 1.336 rho.  The frozen public selector
`b_mod4=3&salt_mod2_pattern=0,1,0` selects exactly that missed key in
840-847, with zero false positives and zero strict-pair-dead selections.  The
narrower `b_minus_c_mod16=12&salt_mod2_pattern=0,1,0` also selects it; the
older `b_minus_c_mod16=12&salt_span=6` abstains because this repeat has salt
span 4.

Across 752-847, the broad selector now selects 5/9 branch-gap keys with zero
false positives across 752-759, 776-783, 832-839, and 840-847.  The combined
relation corpus now has 55 strict-pair relation cases, 36 verified cases, and
28 verified below rho.  The frozen proxy `reject_if transfer_index_mod3=1`
rejects all 19 unverified relation cases and rejects zero verified cases,
including the fresh transfer-841 decoys from 840-847.

This updates the contract: the branch selector is promoted from a single
diagnostic to a repeatable public branch-reach cue, but it is still not a
speedup claim.  A future branch-aware replay must turn selected branch-gap keys
into verified below-rho relation systems before the repeated-coordinate path
can be claimed as an ECDLP improvement.

## Branch-Aware Subset Replay, 752-847

Branch-aware replay artifacts:

```text
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_840_847.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_752_847.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_752_847.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_subset_rule_miner_target67_bmod4_saltmod2_752_847.json
```

The branch-aware replay promotes the selector from reach diagnostic to measured
candidate speedup on the fresh 840-847 branch.  With candidate clause
`b_mod4=3&salt_mod2_pattern=0,1,0`, all public subsets on 840-847 select 28
cases, reach 8 relation cases, verify all 8, and produce 4 below-rho
recoveries.  Best case: transfer 847, coordinate `(55,7531)`, salts 204 and
208, rank 2, relation count 2, derived secret 5840, and 116/125 = 0.928 rho.

Across 752-847, the same candidate clause selects 51/1049 coordinate-gate
candidates.  The all-subsets replay selects 357 pair/subset cases, reaches 86
relation cases, verifies all 86, and produces 36 below-rho recoveries.

The two-row contraction is the current cleaner contract.  It uses only
`subset_size=2` under the same candidate clause and strict form guard
`candidate_pos_span=0&candidate_pos_count>=2`.  It selects 153 cases, reaches
43 relation cases, verifies all 43, and preserves all 36 below-rho recoveries.
Minimum guarded verified cost is 0.896 rho and mean guarded verified cost is
0.936 rho.  It has zero guarded relation cases that fail public-key
verification.

Two-row replay by window:

| Window | Verified | Below Rho | Best Cost | Transfer/Coordinate |
| --- | ---: | ---: | --- | --- |
| 752-759 | 8 | 8 | 0.960 rho | 757 `(239,8922)` |
| 776-783 | 24 | 24 | 0.896 rho | 779/780 `(55,7531)` |
| 832-839 | 7 | 0 | 1.032 rho | 832 `(55,7531)` |
| 840-847 | 4 | 4 | 0.928 rho | 847 `(55,7531)` |

The branch subset miner found clean narrow clauses, especially
`pair_salt_delta_from_min_signature=0,2`, but those clauses explain the
776-783 positives rather than the fresh 840-847 repeat.  The fresh 840-847
below-rho pair has delta signature `0,4`, so the broader frozen two-row subset
contract is the safer next package.

## 848-855 Frozen Abstention

Fresh 848-855 artifacts:

```text
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_848_855_probe.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_848_855_fresh_manifest.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_848_855.json
```

The first unchanged-window check after freezing the branch-aware two-row
package is a clean abstention.  The fixed-selector stress source reports zero
stress-row verification, zero stress-leaf verification, and zero below-rho
labels for the target-cap1, target-cap3, and global-cap3 fixed row selectors.
The target-67 public selector then emits zero public bounded cases, so the
coordinate gate has zero candidates.

The frozen branch-aware package has zero input pair cases and zero relation
cases on 848-855.  This does not count as a second fresh positive, but it is an
acceptable abstention: no guarded unverified relation case appeared, and no
over-rho-only branch-aware recovery appeared.

## 856-871 Frozen Activation Abstentions

Fresh 856-863 and 864-871 artifacts:

```text
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_856_863_probe.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_856_863_fresh_manifest.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_856_863.json
ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_864_871_probe.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_864_871_fresh_manifest.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_864_871.json
```

Both later windows are clean activation abstentions under the frozen package.
The stress sources have zero verified stress rows and zero verified stress
leaves for the fixed target-cap1, target-cap3, and global-cap3 selectors.  The
target-67 public selectors emit zero public bounded cases, so the coordinate
gates have zero candidates.  The branch-aware two-row replays therefore have
zero input pair cases and zero relation cases.

The 848-871 run should be interpreted as a selector-activation dry spell rather
than a relation-quality failure.  No guarded unverified relation case appeared,
and no over-rho-only branch-aware recovery appeared.

## Batched Activation Scan, 848-903

Activation scan script and artifacts:

```text
tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_branch_activation_scan.py
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_activation_scan_target67_bmod4_saltmod2_subset2_848_903.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_activation_scan_target67_bmod4_saltmod2_subset2_752_903.json
```

Fresh 872-903 windows were run through the same fixed-selector stress,
target-67 frozen pipeline, and branch-aware two-row replay package as 848-871.
All four new windows are clean activation abstentions.  The public selector
emits zero target-67 bounded cases, the coordinate gate emits zero candidates,
and branch-aware replay emits zero input pair cases.

The post-positive ledger over 848-903 has complete per-window branch replay
coverage.  It records seven windows, seven branch replay artifacts, seven
abstentions, zero selected public windows, zero coordinate-candidate windows,
zero branch-clause windows, zero relation windows, and zero below-rho windows.

The wider local ledger over 752-903 records 12 activation windows through
840-847 followed by the seven-window dry spell.  It counts 1049 total
coordinate candidates and 51 frozen branch-clause matches before the dry spell.
Earlier branch relation outcomes are in the combined 752-847 replay artifact,
while the 848-903 per-window replays prove that the post-positive dry spell is
empty at activation time.

This updates the contract boundary: the two-row branch-aware relation package
is still alive as a candidate, but future work must distinguish pre-activation
scheduling from relation replay.  The current repeated-coordinate branch cannot
make progress on windows where no target-67 bounded public rows are emitted.

The local proxy-filter audit for 752-847 is clean:
`ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_847.json`
keeps all 28 verified below-rho cases, keeps all 8 verified over-rho cases,
rejects all 19 unverified rank-2 decoys, keeps zero unverified cases, and
rejects zero verified cases under
`reject_if transfer_index_mod3=1`.

## Pre-Activation Cues, 672-903

Pre-activation miner script and artifacts:

```text
tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_preactivation_miner.py
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_activation_scan_target67_bmod4_saltmod2_subset2_672_903.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_preactivation_miner_target67_activated_752_903.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_preactivation_miner_target67_branch_clause_752_903.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_preactivation_miner_target67_activated_672_903.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_preactivation_miner_target67_branch_clause_672_903.json
```

The activation label now spans 29 windows from 672-903.  It has 22 positive
windows from 672-847 and seven dry negative windows from 848-903.  The public
stress cue `has_target67_leaf_total3=1` separates those classes with zero
false positives and zero false negatives.

The branch-clause label is narrower: positives are 680-687, 704-711, 728-735,
736-743, 752-759, 776-783, 832-839, and 840-847.  The best current
high-precision cue is:

```text
target67_salt_mod2_pattern_0_0_1_count>=8&target67_salt_mod3_pattern_0_1_1_count>=1
```

It selects 680-687, 728-735, 752-759, 776-783, 832-839, and 840-847 with zero
false positives, missing 704-711 and 736-743.  This is a scheduler cue for
branch-clause reach, not a relation-success guarantee.

## Partial-Duplicate Form Boundary

Older branch-aware replay backtest artifacts:

```text
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_680_687.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_704_711.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_728_735.json
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_736_743.json
```

Form-boundary audit:

```text
tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_branch_form_boundary_audit.py
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_form_boundary_audit_target67_bmod4_saltmod2_subset2_680_847.json
```

The older selected windows 704-711, 728-735, and 736-743 produce no
branch-aware relation systems under the frozen clause.  The 680-687 window
does reach verified relation systems, but all are rejected by the strict
same-position guard because their candidate-position signatures span positions
1 and 3.

Across the 680 and 752-847 branch-aware relation corpus:

- Strict same-position duplicate form has 43 relation cases, all verified, 36
  below rho, minimum 0.896 rho.
- Partial-duplicate two-position form has 22 relation cases from 680, all
  verified, 11 below rho, minimum 0.976 rho.
- The below-rho partial-duplicate signature is `1,3,3`; the over-rho
  partial-duplicate signature is `1,1,3,3`.

This is not a relaxed guard yet.  It is a new audit target: a future public
guard extension would need to accept the `1,3,3` shape only with an
independent public discriminator that rejects known misaligned decoys and the
over-rho `1,1,3,3` control.

## Partial-Duplicate Guard Extension Audit

Guard-extension audit:

```text
tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_partial_duplicate_guard_audit.py
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_partial_duplicate_guard_audit_target67_bmod4_saltmod2_680_903.json
```

The audit combines branch-aware relation cases from 680, 704, 728, 736, and
752-847; strict-pair alignment relation cases through 752-847; the 832 row/form
branch-gap decoy aggregate; and the seven dry 848-903 activation-abstention
windows.

The recommended extension candidate is:

```text
strict_same_position OR
(candidate_pos_count=3 & candidate_pos_unique_count=2 & candidate_pos_max_duplicate>=2)
```

Retrospective evaluation on the weighted relation-case corpus:

| Guard | Accepted Below Rho | Accepted Over Rho | Accepted Unverified | Rejected Below Rho | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| strict same-position | 64 | 15 | 0 | 11 | rejects the 680 `1,3,3` below-rho cases |
| strict plus count3/unique2 partial | 75 | 15 | 0 | 0 | accepts 680 `1,3,3`, rejects `1,1,3,3`, `1,2`, and `1,3` |
| strict plus broad partial duplicate | 75 | 26 | 0 | 0 | admits the `1,1,3,3` over-rho class |
| strict plus any misaligned relation | 75 | 26 | 27 | 0 | admits all known unverified decoys |

This is the cleanest current form-guard hypothesis.  It is still retrospective
and relation-stage only.  Promotion requires freezing it before the next
nonempty activation window and observing either a verified below-rho recovery
or a clean abstention.

Executable replay support:

```text
tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_form_guard_replay_probe.py
ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_aware_subset_replay_target67_bmod4_saltmod2_subset2_partial_count3_680_847.json
```

The form-guard implementation now supports the forward expression:

```text
candidate_pos_span=0&candidate_pos_count>=2|candidate_pos_count=3&candidate_pos_unique_count=2&candidate_pos_max_duplicate>=2
```

Executable retrospective replay over 680, 704, 728, 736, and 752-847 selected
222 pair cases, reached 65 relation cases, guard-passed 54, verified all 54,
and kept 47 below rho.  Guard-passed costs have minimum 0.896 rho and mean
0.94414815 rho.  This makes the partial-count3 extension a runnable frozen
candidate for the next nonempty activation window; it does not promote the
rule without future-window evidence.

## Next Actions

1. Use `has_target67_leaf_total3=1` as the current cheap activation scheduler:
   do not spend branch-aware replay on windows with no target-67 total3 public
   stress rows.
2. Once a nonempty activation window appears, keep the same
   `candidate_clause=b_mod4=3&salt_mod2_pattern=0,1,0`,
   `pair_rule=subset_size=2`, and test both the strict guard and the frozen
   partial-count3/unique2 extension candidate.
3. Use the salt-pattern cue as a high-precision branch-clause scheduler, while
   remembering it misses 704-711 and 736-743.
4. Treat the partial extension as promoted only after a future nonempty
   activation window accepts no unverified cases and either recovers below rho
   or abstains cleanly.
5. Treat a future verified below-rho recovery as a second fresh branch-aware
   speedup window; treat abstention as acceptable; treat any guarded unverified
   relation case as a form/proxy audit trigger.
6. Explain the rule through summation-polynomial/FFE structure: repeated
   coordinate residues, salt-span geometry, and why duplicate candidate-position
   events correlate with rank-bearing two-row relation subsets.
7. Keep the direct stress scheduler cues separate from the pair-rule speedup
   contract.
8. Keep
   `b_minus_c_mod16=12&salt_mod2_pattern=0,1,0` as the coordinate-55 family
   control; demote `b_minus_c_mod16=12&salt_span=6` to a salt-span-specific
   diagnostic until it repeats again.
