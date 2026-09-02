# Next Hypothesis

Latest update after 752-759:

The repeated-coordinate branch has a new, sharper candidate but not a promoted
algorithmic claim.  Fresh 752-759 direct stress is negative below rho, and the
frozen coordinate gate verifies 24 repeated-coordinate recoveries with zero
below-rho verified cases.  Decomposition of transfer 753 coordinate
`(429,3910)` shows that the algebraic relation system itself is cheap if the
dead public row is pruned: salts 205 and 208 derive secret 7062 at
116/125 = 0.928 rho, while carrying salt 206 raises the full replay to
167/125 = 1.336 rho.

The immediate next hypothesis is public row pruning for repeated-coordinate
families, not more fused-kernel optimization.  The post-hoc rule
`coordinate_key=429,3910 & leaf_min=5 & salt_mod3=1 & policy_family=global_cap3`
selects 20 cases on 752-759; 8 verify, all 8 are below rho at 0.928, and all
8 pass `candidate_pos_span=0`.  Because this predicate was chosen after seeing
the 752-759 decomposition, the required promotion test is to freeze either this
rule or a no-exact-coordinate residue variant and run it on a later fresh
window.  A later-window abstention is acceptable evidence; a later-window
guarded below-rho verification would be the first honest public-pruning
promotion for this repeated-coordinate branch.

Latest update after 760-767:

The exact 752-759 row-pruning predicate failed its first later-window
validation.  The coordinate `(429,3910)` recurs, but the salt neighborhood
changes from the useful 752-759 pair `{205,208}` to `{204,205,206}` on
transfer 764.  The frozen `salt_mod3=1` rule selects only salt 205, produces
zero relation cases, and verifies nothing.  The next repeated-coordinate
experiment should stop treating a static salt residue as the selector.  The
better hypothesis is a public salt-pair predictor: predict two relation-bearing
rows from pair features such as salt gaps, row cost, candidate position
signature, and coordinate/leaf residues, then replay that pair under the same
`candidate_pos_span=0` guard.  The promotion criterion is unchanged: a frozen
pair rule must recover on a later fresh window below rho, or abstain without
false-positive verification.

Latest update after no-exact pair mining:

The first no-exact-coordinate pair rule is better than the exact salt-residue
rule.  Training on positive decompositions through 736-743 and validating on
752-759 gives
`pair_activate:b_minus_c_mod16=1&pair_salt_span>=4|b_minus_c_mod16=7&pair_salt_delta_from_min_signature=0,3|pair_salt_sum_mod8=0&transfer_mod5=1`.
It hits the 752-759 held-out positive pair with zero validation false
positives, replays 8 guarded below-rho cases at 0.928 rho, and then abstains
on 760-767 with zero selected pair cases.  The next clean promotion test is a
fresh 768-775 replay of this exact pair rule.  A guarded below-rho verification
would be the first no-exact pair-rule promotion; an abstention would still
support the rule as a conservative scheduler; any verified over-rho or
unguarded relation case should demote it back to diagnostic.

Latest update after 768-775:

The no-exact pair rule achieved its first fresh later-window positive.  It
selected 32 pair/subset cases on 768-775 and produced 12 guarded verified
below-rho recoveries, best 64/125 = 0.512 rho on coordinate `(161,6976)`,
transfer 771, salt 208, deriving secret 6683.  The full coordinate replay for
the same family remains over rho at 1.328, so the claimed mechanism is now
public subset pruning plus form guard, not full repeated-coordinate replay.
The next experiment should be a replication on 776-783 with this exact rule
unchanged.  In parallel, write a compact contract note that separates the
rule-mining corpus, frozen features, selected subset cost, form-guard cost,
and secret-derivation check.  Do not merge this with the direct target-cap1
stress signal from 768-775; that is a separate scheduler cue.

Latest update after 776-783:

The no-exact pair rule replicated on a second fresh later window.  Direct
776-783 stress remains below-rho negative, and full repeated-coordinate replay
again remains over rho, but the unchanged pair/subset rule selected 52 cases
and produced 4 guarded verified below-rho recoveries.  The best and mean
guarded cost is 116/125 = 0.928 rho on transfer 779, coordinate `(542,3911)`,
salts 203 and 206, deriving secret 4774 with `candidate_pos_span=0`.

This is now a serious candidate scheduler for a summation-polynomial/FFE
index-calculus branch, but still not a general algorithmic claim.  The current
honest claim is: a public no-exact-coordinate pair rule, mined on older
positive decompositions, can prune repeated-coordinate rows into below-rho
relation systems on multiple later windows.  The next clean promotion test is
to run 784-791 unchanged.  In parallel, derive why the public rule predicts
relation-bearing pairs in FFE terms: shared coordinate residues, salt-span
geometry, and the zero form guard need an algebraic explanation rather than
only a mined predicate.

Latest update after 784-791:

The unchanged rule did not replicate a third positive.  Direct stress was
again below-rho negative and the frozen repeated-coordinate gate verified zero
public keys.  The pair rule selected 72 public pair/subset cases, but produced
zero relation pair cases, zero guarded verified cases, and zero guarded
below-rho cases; 16 selected cases passed `candidate_pos_span=0`, but they
stopped at rank 1 and did not verify.

This should not be treated as a fatal false positive, because no over-rho
verified recovery or unguarded relation case appeared.  It should also not be
called a clean abstention.  The next hypothesis is now sharper: the no-exact
pair rule is a relation-bearing scheduler only when the selected pair reaches
rank 2, and 784-791 supplies dead rank-1 guard-passing examples for a structural
audit.  Before changing the rule, run one more unchanged window or compare the
784 rank-1 cases against the 768/776 rank-2 positives to isolate an algebraic
FFE feature that predicts rank lift rather than just a cheap guarded pair.

Latest update after strict rank-lift guard and 792-799:

The 784 rank-1 boundary has a simple event-stage explanation: the old
`candidate_pos_span=0` guard passed singleton relation events vacuously.  A
strict guard, `candidate_pos_span=0&candidate_pos_count>=2`, preserves all 24
known guarded below-rho recoveries across 752-759, 768-775, and 776-783, while
rejecting all 16 singleton rank-1 784-791 dead cases.  This is a better
contract boundary, not a new row selector.

The next fresh 792-799 window is a clean repeated-coordinate negative under the
strict guard: 12 selected pair/subset cases, all rank 0, zero relation cases,
and zero verified recoveries.  It also has a separate target-cap1 direct stress
positive on `22050.cf1@11731`, best 0.43065693 rho, which should stay in the
scheduler-cue lane rather than being merged into the repeated-coordinate
pair-rule claim.  The next useful experiment is now either an unchanged
800-807 strict-guard replay, or an algebraic audit of why the strict
candidate-position double event appears in the 752/768/776 positives but not
the 784/792 negatives.

Latest update after 800-807:

The strict rank-lift guard replicated on the next fresh window.  Full
repeated-coordinate replay on 800-807 verified 8 target-67 public keys but
stayed over rho, best 1.368.  The unchanged no-exact pair rule plus
`candidate_pos_span=0&candidate_pos_count>=2` selected 68 pair/subset cases and
produced 4 guarded verified below-rho recoveries, all on transfer 806,
coordinate `(161,6976)`, salts 201 and 207, deriving secret 2023 at
120/125 = 0.96 rho.

This is the fourth positive window for the pair-pruning branch and the first
positive after the strict rank-lift guard was frozen.  The strongest structural
cue is now candidate-position alignment: the 800-807 decoy relation pairs on
coordinate `(542,3911)` have two relation events but signature `1,2`, rank 2,
and no public-key verification; the successful cases have signature `2,2`.
The next experiment should stop merely counting relation events and audit why
same-position duplicate events correspond to key-consistent summation-polynomial
forms.  A follow-up 808-815 strict replay is useful, but the higher-value work
is to connect the candidate-position alignment to FFE/residual geometry.

Latest update after candidate-position alignment audit:

The replay-backed audit across 752-807 confirms candidate-position alignment as
the current clean event-form separator.  It rehydrated 32 relation-bearing
strict pair cases with zero context errors and zero source-replay mismatches:
28 verified below rho and 4 unverified rank-2 decoys.  Every verified case has
`candidate_pos_aligned=1`, `candidate_pos_span=0`, and
`candidate_pos_span0_count_ge2=1`; every unverified decoy has signature `1,2`,
span 1, and `candidate_pos_aligned=0`.

This does not make the guard a free selector, because the candidate-position
feature is still observed after relation-event scanning.  The next hypothesis
should therefore be sharper: derive a public pre-event proxy for same-position
alignment from FFE/summation-polynomial residual structure, or run 808-815 with
the unchanged strict event-stage contract to measure whether the separator
continues before changing the rule.

Latest update after 568-583:

The rank-first target-67 target-cap1 family branch should now be demoted from
standalone scheduler to candidate generator.  It cleanly selected 528, 537, and
559, but 568-575 exposed a verified nonzero-triple miss and 576-583 produced a
strict leaf-zero false positive.  A broad no-leaf-zero split captures the 569
nonzero-triple replay but also selects two fresh 576-583 false positives.  The
next hypothesis is therefore not another leaf tuple.  It is a two-stage
orientation gate: target-cap1 low-term rank/relation shape proposes rows;
axis-root line lift proposes public lines; x-match orientation features must
then reject the 580/582 false positives while preserving the 528/537/559/569
line-gated successes.

Latest update after 584-591:

The first frozen orientation rule is useful but not sufficient.  Mining over
528/537/559/569 successes and 580/582 hard negatives gives the public rule
`activate:top_k<=12&xmatch_count>=8`, selecting 4/4 measured-below-rho
successes with zero hard-negative false positives.  The next fresh window,
584-591, has no target-67 target-cap1 candidate, so the intended branch
abstains.  A target-cap3 diagnostic on transfer 586 does activate the frozen
rule and derives secret 9387 through two repeated public lines, but best
measured replay is still 1.336 rho.  Therefore the rule is a line/orientation
gate, not a speedup gate.  It needs a public policy/cost guard before promotion:
either restrict it to target-cap1 candidate rows or add a public source/replay
cost ceiling that rejects the 586 over-rho target-cap3 rows.

Latest update after 592-599:

The v2 gate is now `activate:row_xmatch_count_max>=8&top_k<=12`.  It is mined
from the 528/537/559/569 below-rho successes, 580/582 hard negatives, and the
586 over-rho target-cap3 line recoveries.  It keeps 4/4 measured-below-rho
successes and rejects all seven non-speedup rows in the table.  On the fresh
592-599 window, target-cap1 is strong overall but target-67 target-cap1 rows
are rank-0/verifier-negative, so the family generator abstains.  A target-cap3
diagnostic on transfer 595 exposes repeated line `2716*b+c+4800`, but the v2
gate blocks it before replay.  The pipeline is now better factored: public
rank/relation generation filters target-cap1 rank-0 holes, and v2 orientation
filters target-cap3 over-rho line recoveries.

Keep the target-67 line-family activation rule frozen, but shift the next
experiment from row-signature discovery to rank-aware public orientation
prediction or strict repeated-line amortization.  The line-stage audit through
transfer 511 shows that fresh target-67 rows can still expose below-rho
preserving lines, but the new transfer-503 line does not replay, the post-hoc
top-16 branch fails the immediate holdout, and the simple
`where:candidate_eq_scheduled` orientation rule stalls at rank 1.  A
rank-aware retrospective miner selects `all` x-matches as the current honest
orientation baseline, so the main next blocker is public activation of
line-present success cases or repeated-line amortization.  Re-mining
line-success activation on cases through transfer 463 reproduces the same
public rule and validates it on the later transfer-473/503 line-present
boundary.  A simple public line-feature dictionary audit then fails to predict
any later preserving line, so the next line predictor has to use more of the
FFE/summation-polynomial residual structure than leaf signatures and x-match
counts.  A post-factorization factor-cloud audit also fails to isolate the
preserving factor by order or coefficient-size heuristics, so the residual
feature needs to be algebraic, not just a ranking over the final factor list.
The first algebraic residual test is now positive: c-axis roots of the
resultant plus selected-leaf line lifts recover every known target-67
preserving line before full bivariate factorization.  The proxy cost ledger
keeps that lead alive but narrows the claim: the lift alone is below rho on all
four replay-success cases, while per-row lift+replay is not; strict `n=2`
reuse is the current break-even target.
Fresh 512-519 testing adds a sharper boundary: the frozen target-cap1 branch
does not reach the window at all, but an over-rho target-cap3 diagnostic on
transfer 514 reuses the old line `945*b + c + 952`; axis-root lifting emits it
as a singleton on the two selected-root-positive rows, and public replay
derives the secret at 1.272 rho.
Fresh 520-527 testing refines that again: target-cap1 does reach target 67
with four below-rho public candidates, but all are rank-0 and verifier-negative.
The cheapest target-cap1 miss still has a singleton axis-root line
`1966*b + c + 2774`, but replay verifies nothing.  In the same window, an
over-rho target-cap3 case reuses known line `3394*b + c + 711`, derives secret
9642, and again costs 1.272 rho.
Fresh 528-543 testing changes the boundary from branch reach to branch
taxonomy.  Target-cap1 now recovers target 67 below rho in two consecutive
fresh windows: transfer 528 verifies from the public source at 0.712 rho and
line-gated replay through `8741*b + c + 499` derives the secret at 0.544 rho;
transfer 537 verifies from the public source at 0.56 rho and line-gated replay
through `8142*b + c + 4278` derives the secret at 0.448 rho.  The old scheduler
misses both, and the exact post-hoc v4 leaf signature from 528 fails on
536-543.  The stable signal is broader target-cap1 plus low-term total3/total4
family reach with rank/relation positivity, not a fixed leaf tuple.
Fresh 544-567 testing gives the first family-level scheduler candidate.  The
frozen public target-67 target-cap1 family rule calibrates cleanly on 528-543,
abstains on the verifier-negative 544-551 target-cap1 hole, selects a fresh
transfer-559 case on 552-559, and abstains again on 560-567 when target-cap1
positives move to `22050` rather than target 67.  The rank-first per-transfer
refinement is the cleaner branch: 2/2 verified on calibration, 0/0 in the 544
and 560 controls, and 1/1 verified on 552-559.  Exact profiling of transfer 559
does not find a preserving degree-1 factor, but axis-root line lift emits three
candidate lines and line `3379*b + c + 6949` replays below rho.  The
`where:candidate_pos=2` orientation now has one later target-cap1 validation at
0.464 rho instead of being only a transfer-528 post-hoc diagnostic.

Why:

- The frozen activation rule is still
  `activate:top_k=7&xmatch_count>=6|xmatch_count>=8`.
- The 464-479 validation remains the clean positive: frozen branch selection
  chose target-67 transfer 473, exact FFE found `9481*b + c + 5654`, and replay
  derived secret 7675 at 0.448 rho.
- The 480-495 validation did not reach activation.  The unchanged scheduler
  selected three old 22050 branch rows, all verifier-negative, and selected no
  target-67 rows.
- There were public target-67 near-misses in 488-495, especially transfer 494,
  salt 208, top-k 16, leaves `[3,9,17]` at 0.424 rho, but they are
  verifier-negative.  Exact factorization of those rows would not be a clean
  activation-rule validation.
- The target-67 line-stage audit finds 7 preserving degree-1 lines, all with
  below-rho root-scan costs, and 4 successful public replays.  But the minimum
  additive `line confirmation + replay` cost is 1.216 rho, so exact line
  confirmation cannot simply be bolted onto the replay stage as a per-row cost.
- The same audit gives a concrete amortization target: `line confirmation / n
  + replay` breaks below rho at `n = 2` for the replay-success rows.  When row
  selection is also charged, only one observed row has a finite break-even and
  it needs `n = 33`, so row-selection cost must remain a separately audited
  stage.
- The repeated-line amortization probe finds five line families and two repeated
  line-present families: `1119*b + 1*c + 7180 mod 9803` and
  `3394*b + 1*c + 711 mod 9803`.  Each has one replay-success row and one
  replay-failure row, so strict success-only `n = 2` amortization is not yet
  observed.
- Counting failed line-present rows makes both repeated families look
  amortized-below-rho, but that ledger is diagnostic only.  It is not an ECDLP
  recovery because the failed rows do not derive the secret.
- Fixed-line future checks for lines `1119` and `3394` over 480-495 did not
  activate at all under the frozen rule, so current line reuse evidence should
  not be treated as generalizing to future rows.
- The 496-503 future window found a fresh target-67 target-cap1 below-rho row:
  transfer 503, salt 204, top-k 16, leaves `[7,10,13]` and `[6,7,10,13]`.
  Exact Sage factorization found the shared preserving line
  `8022*b + c + 5592 mod 9803`, with root-scan charges 0.928 and 0.936 rho.
- Replay through line `8022` is negative.  The frozen activation rule activates
  zero cases; diagnostic `activation-rule all` activates the profiles but still
  verifies zero orientation rules.  Its factor-zero leaf emits only four
  x-matches, so this is an orientation failure, not an ECDLP recovery.
- The top-16 target-67 branch learned from transfer 503 was frozen and tested
  on 504-511.  It abstained with zero selected cases, so it must remain a
  post-hoc diagnostic branch.
- The old non-target-67 scheduler did validate on 504-511 with three verified
  below-rho `22050` selections, best 0.39416058 rho.  That is useful scheduler
  evidence, but it does not close the target-67 line-prediction gap.
- A public x-match orientation miner trained on the four known target-67
  replay successes froze `where:candidate_eq_scheduled`.  Replay of that fixed
  rule on all nine line-present 328-511 cases activates every case but verifies
  zero public keys, including zero of the original four successes.
- The feature probe
  `ecdlp_index_calculus_state/ffe_target67_orientation_feature_probe_328_511.json`
  explains the miss: every replay-success row has two valid x-matches total,
  while `candidate_eq_scheduled` keeps one valid relation and reaches only
  rank 1.  The next orientation rule must preserve a public pair/rank
  structure, not a single good-looking relation.
- The rank-aware rule miner
  `ecdlp_index_calculus_state/ffe_public_xmatch_orientation_rank_rule_miner_target67_328_511.json`
  chooses `all` as the best already-replayed public rule: 4/4 replay successes,
  0/5 line-present false positives, and verified costs at 0.512, 0.544, 0.544,
  and 0.560 rho.  This does not solve line prediction, but it gives a clean
  frozen replay baseline for the next holdout.
- The line-success activation miner
  `ecdlp_index_calculus_state/ffe_target67_line_success_activation_rule_miner_328_511.json`
  trains on transfers <=463 and validates on later line-present cases.  Its
  best rule is still `activate:top_k=7&xmatch_count>=6|xmatch_count>=8`; it
  selects all three train successes with zero train false positives and then
  selects transfer 473 while rejecting both transfer-503 line-8022 failures.
- The public line prediction audit
  `ecdlp_index_calculus_state/ffe_target67_public_line_prediction_audit_328_511.json`
  tests exact signatures, top-k, x-match counts, scheduled-trial counts, and
  leaf subsets as dictionaries from public features to preserving lines.  Across
  cutoffs 431, 463, and 495 it has zero unique later line matches, including
  zero replay-success matches.
- The pair `{5,6}` is only a set-valued diagnostic: it carries both
  `3394*b + c + 711` and `1119*b + c + 7180`.  Exact `[0,1,2,3]` reuse is also
  not stable, moving from line `945` to `8741` to `9481` across later cases.
- The residual factor-cloud audit
  `ecdlp_index_calculus_state/ffe_target67_residual_factor_cloud_audit_328_511.json`
  checks factor order and coefficient-size heuristics after exact
  factorization.  The best heuristic, `min_constant`, hits only 2/9
  line-present cases and 1/4 replay-success cases; preserving indices span
  `[0,1,2,7,8,20,20,22,22]`.
- The axis-root line-lift audit
  `ecdlp_index_calculus_state/ffe_target67_axis_root_line_lift_audit_328_511.json`
  rematerializes all 14 known target-67 surfaces and uses `R(0,c)=0` roots
  plus selected leaf monic points to lift candidate linear factors before
  bivariate factorization.  It recovers all 9 preserving lines and all 4
  replay-success lines; replay-success candidate sets are all singleton.
- The axis-root lift cost audit
  `ecdlp_index_calculus_state/ffe_target67_axis_root_lift_cost_audit_328_511.json`
  charges one operation per c-axis root plus one selected-leaf
  line-substitution test per candidate.  Under this proxy, the four
  replay-success lift costs are 0.672, 0.864, 0.88, and 0.92 rho.  No
  replay-success case is below rho after adding replay per row, but all four
  are below rho with `n=2` strict reuse; the break-even reuse counts are all 2.
- The 512-519 fresh window is a branch-reach negative for the frozen
  target-cap1 package.  The public bounded source has 196 cases and 112
  target-67 cases, but zero target-cap1 target-67 candidates; the frozen
  multi-branch scheduler abstains with zero selected cases.
- The same 512-519 window gives a useful over-rho target-cap3 diagnostic:
  exact profiles for transfer 514, rows salt208/salt203/salt207, selected leaf
  1, all have below-rho degree-1 preserving factor root scans.  Axis-root lift
  over those exact profiles emits candidate-line counts `[0,1,1]`; the two
  selected-root-positive rows are both singleton line `945*b + c + 952`.
- Public replay through fresh line 945 derives secret 7260 on transfer 514
  under `candidate_pos_min`, but it costs 1.272 rho measured and remains a
  speedup negative.  This should be counted as line-family reuse evidence, not
  a below-rho recovery.
- The 520-527 fresh window shows target-cap1 branch reach without relation
  rank: four below-rho target-cap1 target-67 public candidates appear on
  transfer 520/salt208/top-k 16, but all are rank 0 and verifier-negative.
  The frozen scheduler still abstains.
- Exact profiling the cheapest rank-0 target-cap1 candidate finds a below-rho
  degree-1 factor and singleton axis-root line `1966*b + c + 2774`, but it
  covers only one of the selected leaves and public replay verifies zero
  rules.  This is a rank/leaf-coherence negative.
- Exact profiling the 520-527 verifier-positive target-cap3 case at transfer
  527 finds below-rho degree-1 factors and singleton axis-root line
  `3394*b + c + 711` on two selected-root-positive rows.  Public replay derives
  secret 9642 at 1.272 rho, so line 3394 now has another fresh over-rho
  line-family reuse point.
- The 528-535 fresh window has a target-cap1 public source recovery on
  transfer 528, salt208, top-k 7, leaves `[0,1,2,4]`, at 0.712 rho with rank 4
  and four relations.  Line 8741 replay under the frozen activation derives
  secret 574 at 0.544 rho; the lower 0.448 rho `candidate_pos=2` orientation is
  post-hoc until it validates later.
- The 536-543 fresh window replicates target-cap1 recovery on transfer 537,
  salt202, top-k 12, leaves `[0,6,10]`, at 0.56 rho with rank 2 and two
  relations.  Axis-root exact profiling finds singleton line
  `8142*b + c + 4278`, and frozen line replay derives secret 5938 at 0.448 rho
  under `candidate_pos_min`.
- The old branch scheduler selects zero cases in both 528-535 and 536-543, and
  the post-hoc 528 v4 exact signature selects zero cases on 536-543.  The
  scheduler needs a family-level branch over target-cap1/low-term/rank
  features instead of another exact leaf signature.
- Axis-root lift plus replay remains a separate charged ledger: transfer 528
  has three axis-root candidate lines at 1.12 rho with no
  selected-root-positive preserving-factor surface, while transfer 537 has a
  singleton line at 0.8 rho.  Direct public source, line-gated replay, and
  line-lift confirmation must be reported separately.
- The new script
  `tasks/ecdlp_index_calculus/ffe_public_target67_family_branch_predictor.py`
  freezes a family-level target-67 target-cap1 selector over public target,
  row policy, low-term total3/total4 leaf modes, below-rho public cost, rank
  and relation count, leaf-zero support, and a compact-total4-or-wide-span
  rule.  The rank-first per-transfer option keeps the highest-rank/relation
  public candidate per transfer.
- On 528-543, the unfurled family selector selects 10/10 verified below-rho
  cases.  The rank-first refinement selects 2/2 verified below-rho cases.
- On 544-551, target-cap1 emits 8 cheap target-67 candidates but verifies zero
  labels; the frozen family and rank-first refinement both abstain with zero
  false positives.  A broad diagnostic selector selects 8/8 false positives,
  confirming that the family gate is doing useful work.
- On 552-559, the unfurled family selector selects 12 cases with 2 verified
  labels and 10 false positives.  The rank-first refinement selects exactly the
  transfer-559 salt204 top-k 4 total4 case, verifies below rho at public source
  cost 0.712, and has zero false positives.
- Exact profiling transfer 559 finds no preserving Sage degree-1 factor and
  full remainder cost 2.6 rho.  Axis-root line lift emits three candidate lines
  at 0.84 rho: `945*b + c + 952`, `3379*b + c + 6949`, and
  `4136*b + c + 261`.
- Frozen activation replay through line 3379 derives secret 2952: the
  `where:candidate_pos=2` rule costs 0.464 rho, while `all` and
  `term_shape:2+1+1` cost 0.56 rho.  Lines 945 and 4136 do not activate.
- On 560-567, target-cap1 positives exist but not for target 67; the rank-first
  target-67 family branch abstains with zero selected cases.  This is a target
  filter control, not a falsification of the branch.
- On 568-575, the strict rank-first branch abstains, but this is a recall miss:
  transfer 569, salt206, top-k 12, leaves `[8,9,10]` verifies at public source
  cost 0.648 rho.  Axis-root line lift emits two lines at 0.64 rho, and public
  gate replay through `1296*b + c + 3303` derives secret 2272 at 0.576 rho.
- A broad no-leaf-zero rank-dedup diagnostic captures the 569 nonzero-triple
  case and remains clean retrospectively through 528-575: 4 selected, 4
  verified below rho, zero false positives.  This is diagnostic only because
  it was widened after observing the 568-575 miss.
- On 576-583, the strict branch selects transfer 582, salt206, top-k 12,
  leaves `[0,5,8,9]` at 0.592 rho with rank 2 and relation count 2, but the
  public key does not verify.  The broad branch selects two false positives:
  transfer 582 leaves `[0,8,9]` at 0.584 rho and transfer 580 leaves
  `[6,7,8,16]` at 0.632 rho.
- Exact and axis-root diagnostics on the 576-583 false positives show why raw
  FFE/rank features are insufficient.  The transfer-580 false positive has a
  preserving degree-1 Sage factor with below-rho root-scan proxy cost, yet the
  relation system does not derive the target.  All five candidate false-positive
  lines fail x-match orientation replay with zero verified rules.
- The orientation feature miner
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_miner_528_583.json`
  freezes `activate:top_k<=12&xmatch_count>=8` under measured-below-rho labels:
  4 true positives, 0 false positives, 0 false negatives on the 528-583
  training/hard-negative table.
- The 584-591 forward target-cap1 scheduler check is an abstention: target-cap1
  verifies below rho on `22050`, but both strict and broad target-67
  target-cap1 generators select zero cases.
- The same 584-591 window has a target-67 target-cap3 over-rho diagnostic:
  transfer 586, top-k 12, three leaf-9 rows, source cost 1.496 rho.  Exact
  profiling finds preserving below-rho root-scan factors on all three rows,
  and axis-root lift emits repeated public lines `1427*b+c+7108` and
  `7515*b+c+142`.
- Replaying lines 1427 and 7515 under the frozen activation rule derives secret
  9387, but both remain over rho: best measured replay is 1.336 rho, best
  shared-leaf plus hit-root charged replay is 1.304 rho.  This validates
  orientation/repeated-line structure, not a speedup.
- The v2 feature miner
  `ecdlp_index_calculus_state/ffe_target67_axis_orientation_feature_rule_miner_v2_528_591.json`
  adds the 586 over-rho lines as hard negatives and chooses
  `activate:row_xmatch_count_max>=8&top_k<=12`; planned policy and source-cost
  guards also score cleanly on the same table.
- The 592-599 fresh window is another target-cap1-positive but target-67
  scheduler-negative control.  Target-cap1 has 16 verified below-rho leaf hits
  overall, but target-67 target-cap1 candidates are rank-0/verifier-negative,
  and both strict and broad family generators abstain.
- Transfer 595 target-cap3 is a repeated-line diagnostic: exact profiles have
  preserving below-rho root-scan factors, axis-root lift emits repeated line
  `2716*b+c+4800` on two rows, and the v2 gate blocks activation with zero
  verified replay rules.

Latest update after 600-607:

- The desired target-67 target-cap1 forward win appeared in 600-607.  The
  strict rank-dedup family generator selected transfer 607, salt206, top-k 4,
  total4 leaves `[0,1,2,3]` with rank 2, relation count 2, and public source
  cost 0.568 rho.
- Sage exact profiling found a preserving degree-1 factor on that exact
  row/leaf profile.  The full remainder is over rho at 2.456, but the
  preserving factor surface and root-scan proxies are below rho at 0.624 and
  0.76.
- Axis-root lift produced a singleton candidate line
  `161*b+c+6315` at 0.84 rho.
- Replaying that line under the frozen v2 activation gate
  `row_xmatch_count_max>=8&top_k<=12` derived secret 1752 below rho:
  `candidate_pos_min` costs 0.448 rho and `all` costs 0.544 rho.
- The v2 feature audit updated through 600-607 keeps the same frozen rule as
  best, with 5 true positives, 0 false positives, and 0 false negatives over
  13 line-present records including the 595 target-cap3 block case.

Latest update after 608-615:

- The strict target-67 target-cap1 family generator selected another
  rank/relation-positive below-rho row: transfer 612, salt206, top-k 4,
  total4 leaves `[0,1,2,3]`, public source cost 0.576 rho, rank 2, relation
  count 2.  Strict selected precision through 528-615 is now 5/6.
- This window is not line-backed.  Sage exact profiling on the strict total4
  row found no preserving factor and full remainder cost 3.864 rho.
- The broad total3 sibling `[1,2,3]` is also verifier-positive at 0.568 rho,
  but it likewise has no preserving factor and full remainder cost 3.848 rho.
- Axis-root lift emits repeated old lines `945*b+c+952` and
  `3379*b+c+6949` for both profiles, but selected-root-positive surface count
  is zero.  Frozen v2 replay blocks all four line/profile combinations with
  activated case count 0.
- The v2 feature audit updated through 608-615 remains clean: 17 records,
  5 true positives, 0 false positives, 0 false negatives.  The rule is still
  a useful orientation/line-stage gate, but 608-615 shows the generator can
  have relation-layer wins without a preserving FFE line.

Latest update after 616-623:

- The frozen target67 strict and broad family predictors both abstain on
  616-623.  The rolling target67 audit now covers 12 windows; strict remains
  5/6 verified-below-rho when selected, and broad remains 6/8.
- The same frozen target-cap1 row selector finds a fresh cross-target
  `22050.cf1@11731` row: transfer 618, salt166, top-k 7, total3 leaves
  `[8,56,90]`, public source cost 0.43065693 rho, rank 2, relation count 2,
  and public-key verified.
- The older frozen multi-branch scheduler also reactivates on this window:
  it selects the `22050_v1_85690` total3 system and the `22050_v1_8345690`
  total4 sibling at transfer 618/salt166, both verifier-backed below rho.
  This makes the 22050 branch public-selection evidence, not only post-hoc
  exact-line discovery.
- Sage exact profiling finds a preserving degree-1 factor on the 22050 total3
  row with root-scan cost 0.45255474 rho and surface FFE cost 0.47445255 rho.
  The total4 sibling `[8,34,56,90]` repeats the same mechanism at root-scan
  cost 0.45985401 rho.
- Axis-root lift emits two public lines through leaf 8:
  `4745*b+c+3236` and `6202*b+c+10586`.  Both line-gated replays derive
  secret 4620 at measured replay cost 0.41605839 rho.
- Ungated public orientation with `candidate_pos_min` derives the same secret
  at measured oriented cost 0.37226277 rho.  The frozen target67 v2 gate does
  not activate because the row has `row_xmatch_count_max=6`, so this is a
  revived 22050 line-gate branch, not an extension of the target67 v2 branch.

Latest update after 624-631:

- The frozen target-cap1 row selector remains best on the next holdout:
  6 verified below-rho leaf hits, 3 verified below-rho row hits, and best
  leaf cost 0.47445255 rho.
- The old 22050 public scheduler selects `22050_v1_19348990` at transfer 625,
  salt167, top-k 4, leaves `[19,34,89,90]`.  The source row is verified below
  rho at 0.47445255 rho, rank 3, relation count 3.
- Sage exact profiling on that 22050 row finds no preserving factor, but
  axis-root emits three public lines.  One line, `5230*b+c+7939`, isolates
  leaf 89 and derives secret 7198 below rho: line-gated replay costs
  0.37956204 rho and public `candidate_pos_min` orientation costs
  0.35036496 rho.
- The strict target67 family still abstains.  The broad low-term diagnostic
  selects transfer 628, salt206, top-k 16, total3 leaves `[2,3,6]`, source
  cost 0.528 rho, rank 2, relation count 2.
- That broad target67 total3 row is line-backed: Sage finds a preserving
  degree-1 factor, axis-root emits the singleton selected-root-positive line
  `161*b+c+6315`, line-gated replay derives secret 955 at 0.512 rho, and
  public `candidate_pos_min` orientation derives it at 0.448 rho.
- The frozen target67 v2 gate does not activate the top-k 16 broad row, so
  this is a broad-branch diagnostic success rather than an extension of the
  strict v2 claim.  Strict target67 precision through 528-631 stays 5/6;
  broad rises to 7 verified-below-rho selections out of 9.

Recommended next test:

1. Validate 632-639 with two frozen ledgers:
   the strict target67 v2 gate unchanged, and a separate broad target67
   top-k-16 diagnostic ledger.  Do not merge the broad success into v2 unless
   a future rule is frozen before seeing verifier labels.
2. For target67 broad, test whether public features
   `top_k=16`, `leaf_selector=mode_cost_low_term_support_total3`,
   `salt=206`, and singleton axis line reproduce below-rho replay on a fresh
   later window.  Treat this as a preregistered diagnostic, not yet a
   generalized branch.
3. For 22050, test whether the scheduler-selected `22050_v1_19348990` branch
   plus public axis-line orientation can reproduce a below-rho replay on
   632-639.  Track preserving-factor-backed and axis-line-only successes
   separately.
4. Mine a public pre-exact trigger for the 616-631 22050 branches, but treat it
   as a hypothesis until it validates on a later window.  Candidate features to
   test first are branch name, top-k, row salt residue, line-gate leaf collapse
   to one leaf, and x-match count with `candidate_pos_min`.
5. Keep the 569 nonzero-triple branch as a diagnostic split, not a claim, until
   a future window validates the no-leaf-zero extension without adding new
   false positives.
6. Report four ledgers separately: public source cost, line-gated replay cost,
   axis-root line-lift proxy cost, and additive line-lift plus replay cost.
   Current single-row line-gated replays can beat rho, but fully charged line
   prediction still needs either a validated orientation+cost gate or strict
   repeated-line amortization.
7. Continue scanning future target-67 windows for strict repeated-line
   amortization: at least two below-rho replay-success rows under unchanged
   line and orientation rules.

Success criterion:

A fresh target-67 row satisfies the frozen strict v2 gate, a fresh broad
target67 row validates a preregistered top-k-16 diagnostic rule, or a fresh
22050-like target-cap1 row satisfies a preregistered public 22050 line-gate
trigger; in every case, public replay must derive the secret below rho without
verifier labels.  A stronger success is two or more replay-success rows under
unchanged line/orientation rules, allowing strict success-only amortization of
line confirmation.

Failure criterion:

The public line stage is either over rho, absent on selected rows, repeated only
across replay failures, fails orientation replay, selects the 580/582 style
false positives, or works only after adding new post-hoc row/leaf signatures.
In that case keep the result as a scheduler/line-stage negative, not as
evidence for a generalized index-calculus algorithm.

Latest update after 632-639:

- The frozen target-cap1 row selector is now very strong on the next holdout:
  24 verified below-rho leaf hits, 5 verified below-rho row hits, and best
  leaf cost 0.39416058 rho.
- The old 22050 scheduler selects three transfer-636/salt173 systems.  All are
  verified below rho with rank 3 and relation count 3, but Sage finds no
  preserving factor on any of them.  Axis-root still emits the repeated public
  line `6110*b+c+4058`, and that line derives secret 850 at 0.37956204 rho on
  all three scheduled profiles.
- Strict target67 no longer abstains: it selects transfer 639 total4
  `[0,2,4,6]` and transfer 634 top-k 16 total3 `[0,5,15]`, both verified
  below rho.  Broad selects the transfer 639 total3 sibling `[0,2,4]` plus the
  same transfer 634 top-k 16 row.
- All three selected target67 profiles have preserving degree-1 factors.  The
  transfer 639 total3/total4 sibling line is `8098*b+c+5337` and derives
  secret 9577 at 0.48 rho.  The transfer 634 top-k 16 line `562*b+c+2148`
  derives secret 650 at 0.512 rho by gate replay and 0.448 rho with public
  `candidate_pos_min` orientation.
- The frozen target67 v2 activation gate still activates zero cases in
  632-639, including the top-k 7 transfer 639 rows.  Rolling precision through
  528-639 is now 7/8 for strict rank-dedup and 9/11 for broad.

Recommended next test:

1. Validate 640-647 with the same three ledgers: strict target67 rank-dedup,
   broad target67 low-term diagnostic, and 22050 scheduler-plus-axis-line.
   Keep the frozen v2 activation rule unchanged and count it separately from
   ungated line-gated recovery.
2. Add a pre-exact public trigger miner for the two repeated line families:
   target67 transfer-639 style `8098*b+c+5337` and 22050 `6110*b+c+4058`.
   Candidate public features are branch name, top-k, low-term total, singleton
   axis line, line-gate collapse to leaf 90 or leaf 4, x-match count, and
   `candidate_pos_min` success.
3. For target67, test whether a public activation weaker than
   `row_xmatch_count_max>=8&top_k<=12` can be frozen on historical windows
   without reintroducing the 580/582 false positives.  Promote only after a
   later-window validation, not from the 632-639 successes.
4. For 22050, keep preserving-factor-backed and axis-line-only successes
   separate.  The repeated `6110` line is operationally useful, but without a
   preserving-factor certificate it should remain a revived scheduler branch
   rather than target67-style FFE factor evidence.
5. Continue reporting four ledgers separately: public source cost, exact
   factor/root-scan cost, axis-root line-lift proxy cost, and measured
   line-gated/oriented replay cost.

Success criterion:

A fresh 640-647 target67 row satisfies the unchanged v2 activation gate, or a
new pre-exact trigger frozen before 640-647 selects a target67 or 22050 line
that derives below rho without verifier labels.  A stronger success is two or
more replay-success rows under the unchanged trigger, allowing strict
success-only amortization of line confirmation.

Failure criterion:

The next window either produces no line-gated recoveries, only post-hoc line
choices, v2 activation remains empty with no validated replacement trigger, or
the proposed trigger selects false positives.  In that case keep 632-639 as
strong staged line-gate evidence but not a promoted algorithmic claim.

Latest update after 640-647:

- The next holdout is a clean abstention/negative for the frozen ledgers:
  target-cap1 has 0 verified rows and 0 verified leaves, while target-cap3 and
  global-cap3 have verified rows but no verified total3/total4 leaf cases.
- The public bounded source has 136 cases and zero verifier-positive labels.
  The best 22050 public-cost row is below rho at 0.94160584 but rank 0, and
  the best target67 row is over rho at 1.224 and rank 0.
- The 22050 scheduler, strict target67, and broad target67 ledgers all
  abstain with zero selected cases.  Rolling precision is unchanged:
  strict remains 7/8 and broad remains 9/11 when selected.
- The repeated 22050 line `6110*b+c+4058` was tested on the 640-647
  global-cap3 decoy.  It is nonempty and below rho-costed, but it does not
  verify under gate replay or orientation.
- The target67 line-stage table was refreshed through the 632-639 positives:
  42 surfaces, 29 preserving lines, 16 replay successes, 9 replay failures,
  and 13 no-line surfaces.  No additive line+replay charge beats rho without
  amortization.
- A patched line-success activation miner now supports three-clause rules.
  Trained through transfer 631, the training-best zero-FP rule catches 2/3
  held-out 632-639 successes.  The validation-favorable zero-FP rule
  `activate:top_k=16&xmatch_count>=6|top_k=4&scheduled1_count>=4|top_k=7&xmatch_count>=4`
  catches all three held-out 632-639 target67 line successes, but it is a
  post-line-gate activation rule, not a public degree-1 line predictor.

Recommended next test:

1. Validate 648-655 with the same frozen stress, public bounded selector, and
   three ledgers.  If the ledgers abstain again, record it as another reach
   failure rather than forcing exact/axis follow-up.
2. When a target67 exact line is present, replay the validation-favorable
   activation rule above before inspecting verifier labels.  Keep it separate
   from the older v2 gate and from the training-best rule.
3. Add a true pre-exact line-prediction step only after one of the public
   features can propose the line family before Sage factorization.  Current
   trigger mining chooses when to trust a line-gated replay, not how to find
   the line.
4. For 22050, require either target-cap1 branch reach or a new public feature
   that distinguishes the 632-639 verified `6110` cases from the 640-647
   global-cap3 decoy.

Success criterion:

A fresh 648-655 window either validates the unchanged target67 v2 gate, or has
a line-present target67 row where the validation-favorable activation rule
derives below rho with zero false positives, or has a 22050 target-cap1
`6110` branch that verifies again.  A stronger success still requires public
line prediction or repeated-line amortization sufficient to charge line
confirmation below rho.

Failure criterion:

The next window abstains or only produces global-cap3/rank-0 decoys; the
target67 activation rule admits a false positive; or line successes require
post-hoc line choices not fixed before the validation window.  In that case,
keep the candidate family alive but do not promote it.

Latest update after 648-655:

- The frozen 22050 scheduler, strict target67 branch, and broad target67 branch
  all abstained on 648-655 with zero selected cases.
- The public bounded source has 200 cases and 8 verifier-positive labels.  All
  verifier positives are target67 transfer 653 over-rho rows; the best 22050
  public-cost case is below rho at 0.3649635 but is not public-key verified.
- Rolling strict target67 precision through 528-655 remains 7 verified-below-rho
  selections out of 8 selected cases.  Broad remains 9/11 with 2 false
  positives.
- Exact Sage follow-up on the target67 transfer653 target-cap3 verifier
  positives found 6 preserving factor surfaces.  Preserving root-scan costs are
  below rho, from 0.432 to 0.688, but no full-remainder FFE cost beats rho.
- Axis-root lift collapses 4 of those surfaces to the same singleton public
  line, `6185*b+c+2919` over GF(9803).  The line selects the same leaf-3 monic
  coefficients `(b,c)=(239,8922)` on salts 205, 208, and 207.
- Public line-gated replay derives the correct secret 3588, but remains over
  rho: factor-gate replay bottoms out at 1.304 rho and x-match orientation at
  1.272 measured / 1.24 shared-leaf-hit-root.  The validation-favorable
  post-line activation rule activates the case, but does not make it below rho.
- The exact line slope is not needed to reproduce this 648-655 relation:
  canonical public coordinate gates `b=239` and `c=8922` both select the same
  leaf-3 rows and derive the same secret 3588 with the same over-rho costs.
  This turns the transfer653 lead from "predict this exact Sage line" into
  "predict and amortize this repeated monic coordinate point."
- The line-stage split audit reports 6 preserving lines, 4 replay successes,
  and no line+replay below-rho additive charge.  Minimum line+replay is 1.896
  rho, so this is a line-prediction/amortization lead rather than an
  end-to-end speedup.

Recommended next test:

1. Validate 656-663 with the same frozen stress, public bounded selector, and
   three ledgers, preserving the current abstention/success accounting.
2. Add a pre-exact repeated-coordinate probe for the transfer653 pattern:
   predict whether a row will expose the shared leaf-3 monic point `(239,8922)`
   from public row/salt/selector features before Sage factorization.  Treat
   `6185*b+c+2919` as one exact-factor witness through that point, not as the
   only usable gate.
3. Run the same exact/axis split for any future target67 over-rho verifier
   positives, but charge them as staged evidence unless public line prediction
   or repeated-line amortization brings replay below rho.
4. Keep the validation-favorable activation rule separate from the line
   predictor.  It answers whether to trust a line-gated replay after x-match
   metadata exists; it does not find the line.

Success criterion:

A future window either validates a frozen branch below rho, or a pre-exact
public rule predicts a repeated monic coordinate gate before Sage factorization
and the resulting gated/oriented replay derives the secret below rho after
honest amortized coordinate-confirmation accounting.

Failure criterion:

The next window only repeats over-rho coordinate-gated recoveries, the
coordinate predictor needs verifier labels or exact-factor leakage, or the
activation rule starts admitting false positives.  In that case keep target67
transfer653 as a strong mechanism lead but not an algorithmic claim.

Latest update after 664-671:

- The frozen 22050 scheduler, strict target67 branch, and broad target67 branch
  all abstained on 664-671 with zero selected cases.
- The public bounded source has 128 cases and 4 verifier-positive labels.  The
  best 22050 case is a 0.94160584 rho public-cost decoy with rank 0, and the
  best target67 case is 1.224 rho with rank 0.
- Rolling strict target67 precision through 528-671 remains 7 verified-below-rho
  selections out of 8 selected cases.  Broad remains 9/11 with 2 false
  positives.
- The pre-exact repeated-coordinate miner replays 64 target67 candidates on
  664-671 and finds 0 verifier-positive coordinate replays, matching the
  negative-label status of the target67 bounded source.
- A public repeated-coordinate activation rule was mined on 632-639 plus
  648-655 positives while using 640-647 and 656-663 as negative validation:
  `activate:b_minus_c_mod16=0&source_ops_millirhos>=1312|c_mod16=10&source_ops_millirhos>=1368`
- The rule selects 29 of 43 training positives with 0 false positives
  (precision 1.0, recall 0.6744186), but all selected verified replays remain
  over rho: best 1.304 rho, mean 1.36303448 rho.
- The same rule abstains on 640-647, 656-663, and fresh 664-671, so it is a
  cleaner pre-exact filter rather than a speedup claim.  Allowing exact
  coordinate features during mining produced the same top rule, which is a good
  sign that the current rule is not only memorizing `(239,8922)`.
- A repeated-coordinate amortization audit now groups exact `(b,c)`, `b=`, and
  `c=` gates and dedupes policy/selector/top-k aliases down to unique
  `(gate, transfer, derived secret)` recoveries.
- Under the frozen activation rule, the audit sees 87 exact/axis gate records,
  12 unique verified gate recoveries, 0 direct below-rho replays, and exactly
  one two-recovery family up to axis views: `(1114,8506)`, `b=1114`, and
  `c=8506` on transfers 632 and 636.
- That family has maximum replay cost 1.368 rho.  With two unique recoveries,
  the measured replay charge would need at least a 0.5380117 reusable fraction
  to cross rho.  The additive source-plus-replay proxy is not feasible with a
  <=1.0 reusable fraction.
- The all-coordinate sanity audit does not reveal a stronger hidden family:
  unactivated repeats such as `(161,6976)` recur often but have only one unique
  verified recovery.
- A row-cost decomposition reruns exact-coordinate recoveries with full row scan
  details.  All five sampled verified coordinate recoveries have below-rho
  verifier-informed subsets:
  `(1114,8506)` at 0.928 and 0.96 rho, `(239,8922)` at 0.896 rho,
  `(1219,8787)` at 0.512 rho, and `(161,6976)` at 0.48 rho.
- In every sampled recovery, zero-event rows have `selected_hit_roots=0` and
  relation-bearing rows have `selected_hit_roots=1`.  This is the cleanest
  diagnostic row-pruning surface so far, but it is not free under the current
  preassociation cost model.

Recommended next test:

1. Freeze the coordinate activation rule above and run it on the next fresh
   window before any new feature mining.  Promotion requires a selected
   repeated-coordinate gate that verifies below rho directly or has a documented
   amortization count that brings confirmation plus replay below rho.
2. Mine a public root-positive row predictor for repeated coordinates.  The
   label to predict is `selected_hit_roots > 0`, not final public-key
   verification.  Candidate public features should include coordinate residue,
   transfer residue, salt residue/pattern, leaf index, policy family, top-k,
   and row pre-schedule metadata available before `filtered_leaf_gcd_association`
   is paid.
3. If no public predictor separates root-positive rows, implement a staged cost
   audit that charges the cheapest possible root-hit test separately from full
   relation replay.  Do not count `selected_hit_roots` as a free selector unless
   the cheaper test is actually implemented or measured.
4. Search for a cheaper confirmation path on the coordinate surface before
   widening the activation rule.  Useful probes are axis-only `b=`/`c=` replay
   reuse, low-degree summation-polynomial residue classes, and FFE row/leaf
   profiles that reuse the same coordinate under different salts.
5. Keep exact Sage lines as mechanism witnesses, but make the algorithmic
   candidate "predict repeated monic coordinate gate, then amortize or cheaply
   confirm replay."  Do not promote "inspect resultant factors, then choose a
   line."

Success criterion:

A fresh future window has a repeated-coordinate gate selected by a rule frozen
before that window, derives the secret, and either beats rho directly or has a
credible reuse/amortization count that brings coordinate confirmation plus
replay below rho without verifier labels or exact-factor leakage.

Failure criterion:

The coordinate activation rule selects rank-0 decoys like the 656-663
`(239,8922)` case or the 664-671 target67 rank-0 leaf-3 case, needs
exact-factor information to choose the gate, or keeps all verified coordinate
replays over rho without a defensible amortization or root-positive row-pruning
model.  In that case preserve the miner as a diagnostic and continue searching
for a stronger public selector surface.

Latest update after public row-rule replay over 632-671:

- A public root-positive row miner was added for repeated-coordinate cases.  It
  trains on `selected_hit_roots > 0`, but candidate rules use only public row,
  salt, transfer, selector, policy, top-k, and coordinate residue features.
- On the frozen activated coordinate surface, the best row rule is
  `row_activate:b_minus_c_mod16=0&salt_mod4=3|b_mod16=10&salt_delta_to_max=1|salt_delta_from_min<=1&salt_mod2=0`.
  It selects 47/87 activated rows with precision 1.0 and recall 0.81034483
  against root-positive rows.  Replaying only those rows gives 27/27 verified
  below-rho cases.
- The unique activated-surface wins are now measured, not just
  verifier-informed subset diagnostics: `(1219,8787)` transfer 634 at 0.512
  rho from one row, `(1114,8506)` transfer 636 at 0.928 rho from two rows, and
  `(1114,8506)` transfer 632 at 0.960 rho from two rows.
- This is not yet an independent speedup claim.  The coordinate activation rule
  abstains on validation windows 640-647, 656-663, and 664-671, so the
  activated row rule has no held-out activated-coordinate rows.
- The broader no-exact-coordinate row rule
  `row_activate:b_mod5=4&salt=203|leaf_min_mod8=4&salt_delta_from_min=1|salt_mod4=3&transfer_mod16=8`
  validates root-positive precision on held-out rows: validation precision 1.0
  and recall 0.275.  But replay selects only one relation per validation case,
  so it verifies no held-out secrets.
- The row-pruning blocker has therefore narrowed again: public features can
  detect some root-positive rows, but a fresh speedup needs either a frozen
  coordinate activation that reaches new relation-complete cases or a public
  row rule that predicts two independent relation rows on the same coordinate.

Recommended next test:

1. Freeze both row rules above.  Do not re-mine on the next window before
   replaying them.
2. Run the repeated-coordinate gate miner and row-rule replay on the next fresh
   target67 window.  Success requires a selected repeated coordinate with
   public-key verification below rho, not merely a root-positive one-row hit.
3. Add a rank-completion row-pair miner over public row features.  The target
   label should be "two selected rows give rank 2 / unique form count 2" rather
   than just `selected_hit_roots > 0`.
4. If validation keeps producing one-row root hits only, implement a cheap
   root-hit confirmation timer and charge it separately from full relation
   replay.  That would decide whether one-row hits can be treated as an
   amortized prefilter or are simply attractive but too weak diagnostics.

Success criterion:

A preregistered coordinate rule plus preregistered row rule verifies a fresh
target67 secret below Pollard-rho cost, or a rank-completion rule selects two
public rows that derive the secret below rho without verifier labels or
exact-factor leakage.

Failure criterion:

Fresh validation repeats the 656-671 pattern: row rules identify root-positive
rows but only one relation per coordinate, or the below-rho wins remain confined
to the 632-639 activated training surface.  In that case keep this as the best
mechanism lead so far, but keep looking for a public rank-completion surface.

Latest update after row-pair/rank-completion mining:

- The validation split is not relation-empty.  Across 640-647, 656-663, and
  664-671 there are 20 relation-complete coordinate cases and 40 one-relation
  cases under the row-scan label.
- A one-clause pass mostly rediscovers the frozen coordinate activation:
  `row_activate:candidate_activation_selected=1` selects train completes and
  abstains on validation.  The validation-reaching cost bucket
  `row_activate:source_ops_millirhos=1448` reaches two validation complete
  cases, but replay verifies zero public keys.
- A broader public row-pair rule,
  `row_activate:leaf_index>=10&source_ops_millirhos=1344|leaf_min_mod8=4&salt_min=206|source_ops_millirhos=1336&transfer_mod3=0`,
  selects 16/20 validation relation-complete cases with zero validation
  incomplete cases under the row label.  Direct replay is still negative:
  0 verified validation keys, 0 below-rho cases, and selected costs at
  1.336-1.368 rho.
- The important held-out negatives are now sharper:
  `(281,3224)` transfer 661 in 656-663 and `(1145,8648)` transfer 666 in
  664-671 both reach relation count 2, rank 2, and two unique forms, but fail
  public-key verification.
- Therefore the current blocker is not just finding root-positive rows or two
  relation events.  It is selecting key-consistent relation forms/orientations
  before replay cost is paid.

Recommended next test:

1. Build a form-orientation miner over selected relation event metadata from
   the row-rule replay outputs.  The target label should be public-key
   verification, with hard negatives `(281,3224)` transfer 661 and
   `(1145,8648)` transfer 666.
2. Keep the successful activated row rule as the positive control: it proves
   public row pruning can beat rho when selected forms are key-consistent.
3. Search for public invariants of verified versus rank-only relation systems:
   coefficient parity/residue, duplicated form signatures, selected leaf order,
   salt order, and whether one row contributes two independent forms.
4. If no public orientation invariant separates these cases, instrument a
   cheaper form-consistency test and charge it separately.  The algorithmic
   candidate should become "coordinate gate -> public row prune -> cheap
   form-consistency prune -> replay", not another raw row-rule DNF.

Success criterion:

A frozen form-orientation rule preserves the activated below-rho positives and
rejects the held-out rank-only negatives, then verifies a fresh coordinate case
below rho or under a documented amortized confirmation ledger.

Failure criterion:

Form-orientation features either need verifier labels/exact-factor leakage, or
they cannot distinguish verified 632-639 systems from rank-2 validation decoys.
In that case the repeated-coordinate row-pruning path remains a strong
mechanism but not yet an algorithm.

Latest update after public form-orientation audit:

- A form-orientation audit now records relation-form features from three
  row-rule surfaces before using public-key verification as the label:
  activated row pruning, all-coordinate root-positive pruning, and
  pair-completion pruning.
- The cleanest separator is `candidate_pos_span=0`.  Among 70
  relation-complete systems it selects 50/50 verified systems and rejects all
  20 unverified rank-only systems.
- The guard preserves all 39 below-rho verified systems in the audit.  It also
  preserves 11 key-consistent pair-completion systems that remain over rho,
  which is useful evidence that the guard is about form consistency rather than
  merely below-rho cost.
- The two sharp held-out rank-only negatives have nonzero candidate-position
  spans:
  `(281,3224)` transfer 661 has `candidate_pos_signature=1,2`, and
  `(1145,8648)` transfer 666 has `candidate_pos_signature=1,3`.
- This is the first public form-level guard that separates key-consistent
  relation systems from rank-only decoys in the repeated-coordinate branch.
  The caveat is important: `candidate_pos_span` is only known after relation
  event scanning, so it is an orientation/form-consistency guard, not a free
  row selector.
- No 672-679 repeated-coordinate gate artifact was present, so fresh promotion
  has to wait for the next coordinate-positive window or for generating the
  next gate artifact from fresh selector output.

Recommended next test:

1. Freeze the row-prune plus orientation candidate:
   `candidate_pos_span=0` after either the activated row rule or the broader
   all-coordinate root-positive row rule.
2. Generate or wait for the next fresh repeated-coordinate gate artifact after
   664-671, then replay the frozen row rule and audit `candidate_pos_span=0`
   without re-mining.
3. If the guard validates, split the cost ledger into row scan, relation-event
   extraction, form-orientation rejection, and final derivation.  The goal is
   to determine whether candidate-position alignment can be tested cheaper
   than full replay.
4. If the guard fails, inspect whether failure is from candidate-position
   mismatch, source coordinate activation, or public-key derivation despite
   span zero.  These are different failure modes and should not be collapsed.

Success criterion:

On a fresh coordinate-positive window, a frozen public row rule plus
`candidate_pos_span=0` preserves a verifier-backed below-rho recovery or
rejects all rank-only decoys while selecting no verifier-negative span-zero
relation systems.

Failure criterion:

Fresh span-zero systems become verifier-negative, or the only verified systems
require paying the same full relation scan cost without any reusable or
amortized confirmation path.  In that case candidate-position alignment remains
a diagnostic orientation feature rather than an index-calculus step.

Latest update after frozen form-guard replay wrapper:

- `candidate_pos_span=0` is now implemented as a reusable replay wrapper, not
  only as an audit observation.
- Control replay over 632-671 with the activated, all-root, and
  pair-completion row rules gives:
  70 relation-complete cases, 50 guard-passed cases, 50 guard-passed verified
  cases, 39 guard-passed below-rho cases, and 0 rejected-but-verified cases.
- By surface, the guard preserves all known below-rho row-pruned wins:
  activated has 27/27 guard-passed verified below rho, all-root has 12/12
  guard-passed verified below rho, and pair-completion has 11 guard-passed
  verified over-rho controls.
- The guard also rejects 20 relation-complete rank-only systems without losing
  a verified system in this control run.
- No 672-679 public-bounded selector or repeated-coordinate gate artifact is
  present in the current worktree or mounted state.  The next promotion test
  therefore starts by generating or waiting for that fresh coordinate-positive
  artifact.

Recommended next command path:

1. When a fresh public selector exists, run
   `ffe_public_repeated_coordinate_gate_miner.py` with the frozen selector
   source, `--target 67.a1@9803`, `--include-axis-replay`, and `--replay-top`
   set high enough to replay all repeated-coordinate candidates.
2. Then run
   `ffe_public_repeated_coordinate_form_guard_replay_probe.py` with the frozen
   activated/all-root row rules and `--form-guard candidate_pos_span=0`.
3. Count success only if a guard-passed fresh case verifies below rho, or if a
   guard-passed verified case has an explicit amortized confirmation ledger
   below rho.

Latest update after frozen pipeline runner control:

- The frozen selector -> repeated-coordinate gate -> form-guard replay path is
  now packaged as
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_frozen_pipeline_runner.py`.
- A 664-671 control run produced a manifest at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_664_671_control_manifest.json`.
- The control selected 128 public cases, replayed 64 target-67
  repeated-coordinate gate candidates, and found no target-67 verified or
  below-rho candidate.
- The frozen guard saw 4 relation-complete activated-rule cases in the target-67
  control and rejected all 4 under `candidate_pos_span=0`, with 0
  rejected-but-verified cases.
- This is the right negative-control behavior: the harness reproduces the full
  path without promoting rank-only relation events into claims.

Recommended next command path:

Run the pipeline runner unchanged on the next fresh coordinate-positive stress
source:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache \
ECDLP_TASK_DIR=/Volumes/Volume/autolab/tasks/ecdlp_index_calculus \
/usr/local/bin/python3 \
  tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_frozen_pipeline_runner.py \
  --stress-source <fresh_stress_source_after_671>.json \
  --window-name <fresh_window_name> \
  --target 67.a1@9803 \
  --out-prefix ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_<fresh_window_name> \
  --max-ops-over-rho 1.5 \
  --gate-max-candidates 256 \
  --gate-replay-top 256 \
  --form-guard candidate_pos_span=0
```

Prefer 672-679 if that stress source appears.  Treat a manifest as a promotion
candidate only if the frozen guard passes a fresh verifier-backed below-rho case
or if it yields a verified case with a separately charged amortized
confirmation ledger below rho.

Latest update after 672-679 fresh promotion test:

- The 672-679 fresh stress source is now generated and checked.
- The frozen public selector family still has a live below-rho signal, but it
  lands on `22050.cf1@11731`, not target 67: best fresh case is transfer 677,
  salt 164, leaves `[65,79,90]`, rank 2, relation count 2, verified at
  0.39416058 rho.
- Target 67 has a cheap public rank-0 miss at transfer 673/salt208/leaves
  `[3,6,10]` and a repeated-coordinate verifier-positive recovery at transfer
  677 through coordinate `(161,6976)`, but the full coordinate replay costs
  1.392 rho.
- The older frozen activated/all-root row rules plus `candidate_pos_span=0`
  select no fresh target-67 relation-complete cases, so that package should be
  demoted from promotion candidate to calibration/control.
- The useful new candidate is coordinate-specific row pruning:
  `row_activate:coordinate_key=161,6976&leaf_min=2&salt_mod5=3` plus
  `candidate_pos_span=0`.
- In a combined 632-679 diagnostic, that rule gives 16/16 guarded verified
  below-rho duplicate cases, min 0.48 rho and mean 0.704 rho.  The fresh
  transfer-677 case derives secret 9544 at 0.928 rho after pruning the dead
  row.

Recommended next command path:

1. Generate the next post-679 stress source with the same frozen total3/total4
   selector parameters, using transfer indices 680-687 if available.
2. Run the repeated-coordinate pipeline for `target=67.a1@9803`.
3. Replay the new candidate rule unchanged:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache \
ECDLP_TASK_DIR=/Volumes/Volume/autolab/tasks/ecdlp_index_calculus \
/usr/local/bin/python3 \
  tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_form_guard_replay_probe.py \
  --window 'w680_687|<target67_coordinate_gate_artifact>' \
  --row-rule 'coord161_saltmod5_3|row_activate:coordinate_key=161,6976&leaf_min=2&salt_mod5=3' \
  --form-guard candidate_pos_span=0 \
  --out ecdlp_index_calculus_state/ffe_public_repeated_coordinate_form_guard_replay_target67_coord161_saltmod5_3_680_687.json
```

Count it as a promotion only if this exact rule selects a fresh verified
below-rho target-67 recovery.  If it misses, inspect whether the miss is from
coordinate absence, salt-mod row pruning, or candidate-position/form mismatch.

Latest update after 680-695 repeated-coordinate holdouts:

- The coord161 candidate rule did not validate as a portable selector.  On
  680-687 it had zero input cases because coordinate `(161,6976)` was absent;
  on 688-695 it selected 4 cases but no relation-complete systems.
- The deeper signal is still real: every later verifier-positive target-67
  repeated-coordinate family checked so far has a below-rho relation-row subset
  after decomposition.
  - 672-679: `(161,6976)`, salts 203/208, 0.928 rho.
  - 680-687: `(757,1452)`, salts 202/209, 0.96 rho.
  - 680-687: `(55,7531)`, salt 208, 0.568 rho, but rejected by span-zero form
    guard.
  - 688-695: `(117,6119)`, salts 205/206, 0.896 rho.
  - 688-695: `(861,9226)`, salts 206/208, 0.96 rho.
- The first no-exact row predictor was overfit.  It looked strong when scored
  on 680-687, but on the true 688-695 holdout its row-label precision fell to
  0.18181818 and recall to 0.02, and form-guard replay found no
  relation-complete case.
- The current best hypothesis is no longer "reuse coordinate 161"; it is
  "predict relation-bearing two-row subsets before verifier events."  Good
  candidates for the next miner are public features of the coordinate-gate
  candidate plus row positions: salt pair shape, salt deltas, candidate
  position signature, full-coordinate relation count, and low-term leaf
  signature.  Avoid counting any rule as public if it uses per-row
  `selected_hit_roots`, derived secret, or public-key verification labels.
- The 688-695 stress source also keeps the non-target-67 branch alive:
  target-cap1 found a fresh `22050.cf1@11731` verified below-rho source case at
  0.43065693 rho.  That is separate from the target-67 repeated-coordinate
  blocker and should be reported separately.

Recommended next command path:

1. Generate 696-703 with the same frozen total3/total4 stress parameters.
2. Run the target-67 frozen repeated-coordinate pipeline unchanged.
3. Replay the frozen diagnostics unchanged: coord161, the 680-derived no-exact
   salt/cost rule, and the latest exact-coordinate recipes only as controls.
4. Add a new public subset-pair miner that treats the decomposed below-rho
   subsets as labels but only emits pre-verifier public predicates over row
   metadata.  Validate any emitted rule on 696-703 or later, not on the window
   used to mine it.

Promotion criterion:

A target-67 coordinate gate selects a fresh verifier-backed recovery where the
public subset rule chooses the relation-bearing rows, `candidate_pos_span=0`
passes, and the charged replay is below rho.

Failure criterion:

Fresh windows continue to show below-rho verifier-informed subsets but all
public subset predictors either miss, false-positive, or require the same full
row scan as the over-rho full-coordinate replay.  In that case the
repeated-coordinate path remains a diagnostic algebraic compression, not yet a
novel index-calculus algorithm.

Latest update after 696-719 subset-pair tests:

- The new subset-pair miner and replay wrapper are now implemented:
  `ffe_public_repeated_coordinate_subset_pair_rule_miner.py` and
  `ffe_public_repeated_coordinate_subset_pair_rule_replay_probe.py`.
- The first no-exact pair rule, trained on 672-679 and 680-687 and scored on
  688-695, is:
  `pair_activate:b_minus_c_mod16=3&pair_salt_delta_from_min_signature=2,4|pair_salt_index_span>=1&pair_salt_sum_mod5=1`.
- This rule has mixed but important forward behavior:
  - It reproduces 688-695 guarded below-rho pair replays.
  - It misses 696-703, where target67 has no verifier-positive full
    coordinate replay.
  - It misses 704-711, where the positive pair is salts 203/205.
  - It validates forward on 712-719: coordinate `(757,1452)`, salts 202/209,
    `candidate_pos_span=0`, derived secret 702 at 0.952 rho.
- A widened no-exact candidate chosen after seeing 704 catches both 704 and
  712, but it is post-704 and needs another unseen window before promotion.
- The pair-label miner should not use "best subset equality" as the only
  success metric.  On 712-719, the decomposition's cheapest subset for
  `(757,1452)` is singleton salt209 at 0.544 rho, while the public pair rule
  selects salts 202/209 and still verifies below rho at 0.952.  The replay
  wrapper is therefore the authoritative promotion test, not label equality.

Recommended next command path:

1. Generate 720-727 with the same frozen stress parameters.
2. Run the target-67 frozen repeated-coordinate pipeline.
3. Replay both no-exact pair rules:
   - the older 672/680 rule, which already has a clean 712 forward win;
   - the widened post-704 rule, which needs its first post-712 validation.
4. If target67 has verifier-positive full coordinate replays, decompose every
   verified coordinate and compare:
   - full replay cost,
   - cheapest verifier-informed subset,
   - public pair-rule replay cost,
   - whether `candidate_pos_span=0` passes.

Promotion criterion:

Treat the branch as a serious algorithm candidate if a no-exact pair rule
gets at least one more post-712 target67 guarded verified below-rho replay, and
the replay uses only public pair features plus charged row scans and the frozen
form guard.

Failure criterion:

If the next coordinate-positive windows keep producing below-rho decomposed
subsets but neither pair rule selects a below-rho verifier-backed replay, the
pair rule is a promising but brittle selector and the next step should mine
features over the FFE/summation-polynomial residuals rather than salt-pair
congruences alone.

Latest update after 720-727:

- The 720-727 target67 window is coordinate-positive but not
  speedup-positive.  The frozen pipeline found coordinate `(161,6976)` on
  transfer 721, deriving secret 1767 at 1.44 rho, but the best
  verifier-informed subset is salts 206/207 at 1.032 rho.
- The older 672/680 pair rule and the widened post-704 candidate both selected
  20 public pairs and found zero relation-bearing guarded pairs.  This does
  not count as a second forward validation, but it is also not a clean false
  positive against a below-rho subset because no below-rho subset exists in
  the decomposition.
- The old coord161 salt-mod row rule did not activate on this recurrence, so
  coordinate reuse alone is not enough; the salts and relation rows moved.
- The next run should keep both frozen pair rules unchanged and generate
  728-735.  Record three outcomes separately:
  full-coordinate abstention, full-coordinate over-rho-only compression, and
  full-coordinate below-rho subset compression.  Only the third is a fair
  promotion test for the pair rules.
- If another coordinate lands just above rho like 720-727, mine public
  residual features for a small cost shave rather than another salt-congruence
  rule: low-term total support, duplicate form count, selected root count, and
  FFE factor/remainder shape for the relation rows are the likely places to
  recover the missing few operations.

Updated promotion criterion:

Promote the repeated-coordinate branch only after a post-712 fresh window has
a full target67 coordinate replay, a verifier-informed below-rho subset, and a
no-exact public pair rule that selects a guarded verified below-rho replay.
The 720-727 window fails the middle condition, so it should be logged as a
near-threshold compression boundary, not as pair-rule validation.

Latest update after 728-735:

- The next fresh window repeats the same boundary more sharply.  Target67 again
  has verified full coordinate replays for `(161,6976)`, now on transfer 732,
  deriving secret 3693 at 1.432 rho.
- Decomposition improves the replay to salts 207/208 at 128 ops versus 125
  rho steps, or 1.024 rho.  This is only 3 operations above the generic-rho
  line, but still not a speedup.
- Both frozen no-exact pair rules miss: the older 672/680 rule selected 60
  public pairs with zero relation pairs, and the widened post-704 rule
  selected 80 public pairs with zero relation pairs.  The old coord161
  salt-mod row rule selected 19 cases and also found zero relation cases.
- A new near-threshold audit artifact,
  `ffe_public_repeated_coordinate_near_threshold_audit_target67_672_728.json`,
  compares the decomposed target67 recoveries.  It finds 6 verified below-rho
  subsets and 2 over-rho near misses.  The two near misses are exactly the
  repeated `(161,6976)` cases from 720 and 728.
- Both near misses use selected leaf 2, term shape `2+1+1`, and factor support
  `1|9|13`; only candidate-position/scheduled-trial details move.  That makes
  a residual FFE/summation-polynomial cost shave more plausible than another
  salt-congruence-only rule.

Recommended next command path:

1. Keep 736-743 as the next fresh-window classifier, but do not make pair-rule
   promotion the only question.
2. First build a public residual audit for the `(161,6976)` near-threshold
   family:
   - exact public row/leaf profiles for leaf 2 on salts 206/207/208,
   - factor/root or slice-quadratic costs for the shared `2+1+1` /
     `1|9|13` event shape,
   - an explicit ledger for the 3-operation gap on 728-735.
3. Promote only if the residual rule is public before relation labels, derives
   the same target67 secret, and brings 728-735 below 125 charged operations.

Updated failure criterion:

If the residual audit cannot make the leaf-2 `2+1+1` / `1|9|13` choice public,
or its public charge remains above rho, then `(161,6976)` should be treated as
a repeated near miss.  In that case the campaign should continue with 736-743
classification and reserve pair-rule promotion for windows that already have a
below-rho verifier-informed subset.

Latest update after residual-share audit and 736-743:

- The residual-share audit makes the 720/728 near-miss cost model explicit:
  public-before-event sharing is too weak, one shared hit-event pass would
  move 728 below rho, and 720 needs one shared hit-event pass plus duplicate
  leaf/root sharing.  This is a concrete FFE/summation-polynomial target, but
  it is still diagnostic until made public and materialized.
- The next classifier window, 736-743, is stronger than the near misses.
  Target67 again hits `(161,6976)`, now on transfer 740, and full replay
  derives secret 4924 at 1.344 rho.  Decomposition exposes a real below-rho
  subset: salts 203/209, leaf 2, 112 ops / 125 rho = 0.896.
- The original 672/680 pair rule, the widened post-704 rule, and the old
  coord161 row rule all miss this fair test, finding zero relation pairs or
  rows.
- A post-736 no-exact candidate,
  `pair_activate:b_minus_c_mod16=1&pair_salt_index_signature=0,2`, replays the
  736 pair below rho at 0.896 with `candidate_pos_span=0`.  This candidate is
  not promoted yet because it was surfaced after seeing the 736 validation
  score; it is now the rule to freeze for the next unseen window.

Recommended next command path:

1. Generate fresh 744-751 with the same frozen stress parameters.
2. Run the target67 frozen repeated-coordinate pipeline.
3. Replay three pair-rule families unchanged:
   - the older 672/680 rule,
   - the widened post-704 rule,
   - the new post-736 candidate
     `pair_activate:b_minus_c_mod16=1&pair_salt_index_signature=0,2`.
4. If `(161,6976)` recurs as a near miss rather than a below-rho subset, run
   the residual-share audit and check whether the same public coordinate/leaf
   group only needs a shared hit-event pass.
5. Promote only if the post-736 candidate validates on a truly unseen window,
   or if the residual event-share computation is implemented as a public
   replay and beats rho without relation-label pruning.

Latest update after 744-751:

- The truly unseen 744-751 holdout did not validate the post-736 pair rule.
  The target67 coordinate gate found a different verified coordinate,
  `(1114,8506)` on transfer 744, deriving secret 6227 at 1.44 rho.  The best
  verifier-informed subset used salts 206/209 and leaf 12, but stayed just
  above rho at 129 ops / 125 rho = 1.032.
- The original 672/680 pair rule, the widened post-704 rule, and the
  post-736 candidate all found zero relation pairs on 744-751.
- The near-threshold audit is now 10 verified decompositions with 7 below-rho
  subsets and 3 over-rho near misses.  The new miss shares the `2+1+1` event
  shape but not the old `(161,6976)` coordinate or leaf-2 signature.
- The residual-share audit still reports 0 public-before-event shared models
  below rho.  On the new 744 miss, one shared hit-event pass only ties rho;
  crossing below rho also needs duplicate leaf/root sharing.

Recommended next command path:

1. Do not promote `pair_activate:b_minus_c_mod16=1&pair_salt_index_signature=0,2`.
   It is a 736-local explanation until a later unseen window revives it.
2. Add a public event-share materialization probe for the near-threshold
   records, starting with the common `2+1+1` term-shape family and comparing
   leaf/root/event-pass charges before relation labels are inspected.
3. Freeze a diagnostic-form predictor before the next fresh window.  The
   candidate features to test are public coordinate key, selected leaf
   signature, low-term total support, duplicate form count, and preassociation
   FFE factor/remainder shape.
4. Run the next fresh window only after the event-share predictor is fixed.
   Promotion requires an end-to-end public replay below 125 charged operations,
   not merely a verifier-informed decomposition below rho.

Latest update after public hit-event share probe:

- The new public hit-event share probe is a better next branch than more
  salt-pair mining.  `all_hit_event_rows`, `all_hit_root_rows`, and
  `largest_leaf_hit_event_group` select public-key-verified replays in all 10
  decompositions from 672 through 744.
- Five of those replays are already below rho.  With duplicate leaf/root plus
  one shared hit-event pass modeled, 8 of 10 are below rho.
- The three recent near misses are now concrete shared-work targets:
  720 goes from 129 ops to modeled 123, 728 goes from 128 to modeled 120, and
  744 goes from 129 to modeled 123 when leaf/root/event-pass sharing is
  charged.

Recommended next command path:

1. Implement the shared-pass kernel behind the public hit-event selector,
   starting with a narrow replay-only accounting patch for
   `all_hit_event_rows`.
2. The promoted metric should be: selected rows are public before relation
   labels, replay derives the target secret, and charged ops are below 125
   after only materialized shared work is subtracted.
3. Keep the older pair rules as controls, not as the primary path.

Latest update after charged replay accounting:

- The hit-event share probe now supports `--charge-policy` and emits a
  `charged_replay` block instead of only listing detached shared-charge models.
- With public rule `all_hit_event_rows` and charge policy
  `share_leaf_hit_root_and_one_event_pass`, the 672-744 audit has 10/10
  verified selected-row replays and 8/10 charged below rho.
- The three fresh near misses that matter most are all charged below rho:
  720 at 123 ops, 728 at 120 ops, and 744 at 123 ops.
- The charged artifact remains explicitly accounting-only.  The algorithmic
  gap is no longer row selection; it is implementing the shared
  leaf/root/event-pass computation as a real FFE/summation-polynomial replay
  primitive.

Recommended next command path:

1. In the replay scanner, expose the public hit-event table before relation
   derivation: row key, leaf signature, selected hit roots, selected hit event
   count, and the low-term support shape needed to batch duplicate work.
2. Replace `accounting_only_not_algebraic_kernel` with a materialized shared
   pass for identical leaf signatures, then rerun
   `ffe_public_repeated_coordinate_hit_event_charged_replay_target67_672_744.json`.
3. Keep promotion frozen to the same `all_hit_event_rows` rule and the same
   672-744 decomposition set.  A real promotion needs the charged-below-rho
   counts to survive when the savings come from code paths that actually share
   computation, not from post-hoc subtraction.

Latest update after trace materialization:

- The replay wrapper now stores the public trace needed for a shared pass:
  `selected_hit_root_values` and `hit_event_summaries` for every x-match.
- All 10 repeated-coordinate decompositions from 672 through 744 were rerun
  with that richer trace, then the `all_hit_event_rows` charged replay was
  rerun unchanged.
- The materialization audit now reports 8 charged-below-rho replays, 7 of
  which require nonzero shared savings.  All 7 shared-savings wins have the
  hit-root IDs and all-xmatch traces needed to audit the charged pass.
- The previous blocker, missing public scan trace, is closed.  The remaining
  blocker is implementation-level: the replay still subtracts shared counters
  rather than executing one fused shared FFE/summation-polynomial pass.

Recommended next command path:

1. Add a fused shared-pass executor for identical selected leaf signatures:
   one leaf association, one hit-root stream, then per-row relation acceptance
   labels only after the public x-match trace is built.
2. Recompute charged ops from the executor's actual work counters.  The audit
   should stop relying on `shared_*_ops` fields as inputs.
3. Promotion criterion stays fixed: `all_hit_event_rows`, same 672-744 corpus,
   7/7 shared-savings wins below rho, and no missing public trace fields.

Latest update after trace-fused executor audit:

- The trace-fused audit now recomputes shared work from the public materialized
  trace itself: selected leaf signatures, selected hit-root IDs, and every
  x-match summary keyed by `(leaf_signature, scout_pos, original_trial)`.
- This corrects the charged-counter ledger downward.  The previous accounting
  had 8 charged-below-rho replays and 7 shared-savings wins; strict
  trace-derived identities leave 6 below-rho replays and 5 shared-savings wins.
- The important fresh 744 recovery survives: `coord1114_744` remains below rho
  at 123/125 = 0.984 after trace-derived sharing.
- The older 720 and 728 near misses should become negative controls.  They no
  longer cross rho under strict x-match identity sharing: 720 is 127/125 and
  728 is 126/125.
- The claim boundary is now cleaner: row selection is public, trace material is
  complete, and the best fresh target has trace-derived sub-rho work counters;
  the remaining non-claim is a true fused FFE/summation-polynomial kernel.

Recommended next command path:

1. Implement the fused executor against the 744 pattern first:
   selected leaf signature `12`, salts 206/209, hit roots `[2716,5973]`, and
   duplicate x-match events grouped by scout/original-trial identity.
2. Keep 720 and 728 in the same regression run as negative controls.  A correct
   executor should not accidentally reproduce their optimistic charged-counter
   costs unless it exposes a stronger public sharing identity than the current
   trace key.
3. Promote only after the executable path, not the audit model, preserves the
   744 recovery below rho and reports actual work counters without consuming
   `charged_ops` or `shared_*_ops` as inputs.

Latest update after fused public worklist audit:

- The fused worklist audit now emits the concrete public kernel plan: shared
  leaf work, hit-root unions, unique first-pass x-match event keys, and
  row-specific second-pass relation emission.
- All 10 records are worklist-ready with complete traces, no duplicate-event
  public-field conflicts, and no missing accepted-event relation summaries.
- The sharper result is that only one below-rho record actually needs
  first-pass event reuse: the fresh `coord1114_744` recovery.  It saves four
  reusable first-pass events and stays at 123/125 = 0.984 rho.
- The 720 and 728 controls are now cleaner negatives: both are worklist-ready,
  both have zero reusable first-pass event keys, and both remain above rho.
- This narrows the next implementation from a broad shared-pass rewrite to one
  concrete kernel target: consume the 744 worklist directly and make the replay
  scanner report 123 actual work units without reading charged counters.

Recommended next command path:

1. Add a replay-scanner mode that takes the fused worklist for one
   leaf-signature group and executes the public first pass once per event key.
2. Start with the 744 worklist because it is the only current below-rho case
   whose margin depends on event reuse:
   `(12,17,469)`, `(12,30,469)`, `(12,174,469)`, `(12,210,469)`.
3. Keep 720/728 wired into the same command as regression controls: the actual
   scanner should report no first-pass savings for them unless the code exposes
   a stronger public identity than the current worklist.

Latest update after fused worklist relation-form replay:

- The fused worklist is now consumed by a replay-layer probe rather than only
  audited.  The consumer reconstructs compact verifier linear forms from the
  accepted second-pass relation summaries, derives the secret, and recomputes
  fused work counters from the worklist itself.
- The first run caught a real modulus trap: target suffix `@9803` is the field
  prime, while the relation modulus is base order 9887.  The consumer now
  sources `base_order` from `frontier_targets.json`.
- All 10 worklists consume successfully and derive the same secrets as the
  upstream public-key-verified replays.
- The branch remains sharply focused on 744: `coord1114_744` consumes and
  derives secret 6227 at 123/125 = 0.984 rho with 4 first-pass saved ops.
- The 720/728 controls also derive their secrets but remain above rho, so the
  worklist consumer did not accidentally revive the earlier optimistic charged
  interpretation.

Recommended next command path:

1. Move from replay-layer consumption to scanner-layer consumption: use the
   744 worklist to drive candidate equality and relation-form emission, rather
   than reading accepted relation summaries from the artifact.
2. Preserve the same accounting boundary in the scanner output: base order from
   `frontier_targets.json`, actual fused first-pass count 8, second-pass
   instance count 12, and total 123 work units.
3. Keep the same three-record regression check as the first gate:
   `coord1114_744` must stay below rho and derive 6227; `coord161_720` and
   `coord161_728` must derive but stay above rho unless a new public identity
   is explicitly exposed.

Latest update after fused worklist scanner-layer replay:

- Scanner-layer consumption is now implemented in
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_worklist_scanner_probe.py`.
- The probe rematerializes row contexts, schedules only the public worklist
  event keys, re-runs candidate x-match checks, and emits relation forms
  through the verifier predicate during the run.
- The three-record gate passed:
  `coord1114_744` derives 6227 at 123/125 = 0.984 rho, while `coord161_720`
  and `coord161_728` derive their secrets but stay above rho at 127/125 and
  126/125.
- The full 10-record corpus also passed:
  10/10 consumed, 10/10 derived, 10/10 fused-counter matches, 6/10 below rho,
  and only 1 below-rho event-reuse target.
- All scanner worklist observations are clean: zero missing instances, zero
  unexpected instances, and zero acceptance mismatches.

Recommended next command path:

1. Treat `coord1114_744` as the first low-level kernel target: leaf signature
   `12`, salts 206/209, hit roots `[2716,5973]`, 8 unique first-pass event
   keys, 12 second-pass row instances, and target secret 6227.
2. Replace the scanner wrapper with an actual fused FFE/summation-polynomial
   pass that computes the public first-pass event keys once and performs
   row-specific relation emission as the second pass.
3. Keep the same proof gate: the optimized kernel must reproduce the scanner
   artifact's 123/125 cost for 744 and keep 720/728 above rho unless it exposes
   a new public identity that is recorded in the artifact.

Latest update after fused candidate-point reuse audit:

- Candidate-point reuse is now audited in
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_candidate_point_probe.py`.
- The probe groups row instances by public event key and verifies that duplicate
  groups share the same elliptic-curve candidate point before charging one
  planned first-pass operation.
- The three-record gate passed:
  `coord1114_744` has zero candidate-point conflicts, derives 6227, and stays
  at 123/125 = 0.984 rho; `coord161_720` and `coord161_728` still derive but
  stay above rho with zero candidate-point reuse.
- The accepted duplicate 744 event `(12,174,469)` reuses representative point
  `[9251,9216]` for both row instances and still emits both row-specific
  relation forms.
- The full 10-record corpus passed with 10/10 verified reuse boundaries, zero
  candidate-point conflict records, 10/10 secret matches, 10/10 fused-counter
  matches, 6/10 below rho, and only one below-rho event-reuse target.

Recommended next command path:

1. Implement the 744 fused kernel as the same public plan but with the audit
   recomputation removed from the hot path: compute the representative
   first-pass candidate point once per event key, then feed the row-specific
   relation predicate.
2. Keep the candidate-point probe as the regression oracle.  The optimized
   kernel must reproduce zero conflicts, secret 6227, rank 3, 8 first-pass
   candidate-point groups, 12 second-pass instances, and 123/125 work units.
3. Use 720 and 728 as negative controls in the same run; they should still
   derive their secrets but report no reusable candidate-point groups and stay
   above rho.

Latest update after fused hot-path executor:

- The hot-path candidate-point executor now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_executor_probe.py`.
- It computes one representative candidate point per public event key and uses
  that point for every row-specific relation predicate in the group.  The
  candidate-point audit artifact is only a post-run oracle.
- The three-record gate passed:
  `coord1114_744` derives 6227 at 123/125 = 0.984 rho with 8 first-pass groups,
  4 reused groups, and 12 second-pass row instances.  `coord161_720` and
  `coord161_728` still derive but stay above rho with zero reused groups.
- The full 10-record corpus passed:
  10/10 hot-path verified, 10/10 secret matches, 10/10 fused-counter matches,
  10/10 audit-oracle matches, 6/10 below rho, and only one below-rho
  event-reuse target.
- This closes the audit-only candidate-point gap.  The remaining implementation
  gap is lower-level: replace the Python row-context/verifier executor with an
  optimized FFE/summation-polynomial kernel that preserves this exact worklist
  and counter contract.

Recommended next command path:

1. Specialize the low-level kernel interface for the 744 contract:
   leaf signature `12`, hit-root union `[2716,5973]`, first-pass event keys
   `(12,17,218)`, `(12,17,469)`, `(12,30,218)`, `(12,30,469)`,
   `(12,174,218)`, `(12,174,469)`, `(12,210,218)`, `(12,210,469)`.
2. Emit a kernel trace with exactly 8 first-pass candidate-point outputs and 12
   row-specific relation checks, then compare it to
   `ffe_public_repeated_coordinate_fused_kernel_executor_target67_744_controls.json`.
3. Keep 720/728 in the regression harness as no-reuse controls.

Latest update after fused kernel contract:

- The portable contract emitter/verifier now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_contract_probe.py`.
- The 744 contract freezes the low-level target:
  8 first-pass candidate-point outputs, 12 second-pass row checks, 3 accepted
  compact relation summaries, rank 3, secret 6227, and 123/125 work units.
- The contract independently derives over base order 9887 and recomputes work
  counters without using the executor's counters as evidence.
- The full 10-record corpus passed:
  10/10 contracts verified, 10/10 secret matches, 10/10 fused-counter matches,
  6/10 below rho, and only one below-rho event-reuse target.

Recommended next command path:

1. Implement a lower-level 744-only FFE/summation-polynomial kernel trace that
   emits the same contract fields, starting with the eight public event keys
   and representative candidate points in
   `ffe_public_repeated_coordinate_fused_kernel_contract_target67_744_controls.json`.
2. Compare the low-level trace to the portable contract, not to the older
   scanner/executor artifacts.
3. Keep the negative-control contract rows for 720 and 728 wired into the same
   verifier so any accidental extra sharing is caught.

Latest update after fused kernel ABI/curve invariant probe:

- The ABI guard now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_abi_probe.py`.
- It consumes the portable contract and verifier curve records, then checks
  first-pass candidate points for finite field range, on-curve status, and
  base-order subgroup membership.  It also checks that second-pass row checks
  consume only first-pass event-key fanout and that accepted summaries match
  the row-specific relation ABI.
- The 744/720/728 control artifact passed:
  3/3 ABI-curve verified, 32 candidate-point curve/subgroup checks, 36
  second-pass row checks, 7 accepted relations, and the exact 744 reuse anchor
  verified.
- The full 10-record artifact passed:
  10/10 ABI-curve verified, 77 candidate-point checks, 81 second-pass checks,
  22 accepted relation summaries, 6 below rho, and only one below-rho
  event-reuse target.

Recommended next command path:

1. Build a 744-only low-level kernel trace emitter whose first output is the
   eight event-key candidate points from
   `ffe_public_repeated_coordinate_fused_kernel_abi_target67_744_controls.json`.
2. Treat `ffe_public_repeated_coordinate_fused_kernel_abi_probe.py` as the
   promotion gate: the lower-level trace must produce the same candidate-point
   groups and second-pass check stream before any broader claims.
3. Keep the 720/728 no-reuse controls in the same ABI gate so below-rho reuse
   cannot be introduced by an accidental row merge.

Latest update after fused kernel affine field-operation trace:

- The affine trace emitter now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_affine_trace_probe.py`.
- It rematerializes the public worklist context and emits first-pass candidate
  point additions as explicit generalized-Weierstrass field operations:
  operands, slope numerator/denominator, denominator inverse, slope,
  intercept, and result.
- The 744/720/728 control artifact passed:
  3/3 affine traces verified, 32 first-pass groups matched the ABI candidate
  points, 36 row-instance affine adds matched the verifier formula, and all 4
  reused groups were the intended `coord1114_744` groups.
- The critical 744 reused event is now field-traced:
  event `["12", 174, 469]` computes
  `[4979, 3914] + [7537, 8016] -> [9251, 9216]`, and both row instances reuse
  that candidate point.
- The full 10-record artifact passed:
  10/10 affine traces verified, 77 first-pass groups matched ABI points, 81
  row-instance affine adds matched verifier addition, 6 below rho, and only one
  below-rho event-reuse target.

Recommended next command path:

1. Promote `ffe_public_repeated_coordinate_fused_kernel_affine_trace_probe.py`
   into a 744-only native/FFE trace contract: start by emitting the same
   affine fields for the eight 744 event keys without calling
   `verifier.add_points`.
2. Then wire that native/FFE trace into
   `ffe_public_repeated_coordinate_fused_kernel_abi_probe.py` so the same ABI
   and second-pass row-check gates stay unchanged.
3. Keep 720/728 as negative controls at both the affine-trace and ABI layers.

Latest update after fused kernel finite-field replay:

- The verifier-independent field replay now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_field_replay_probe.py`.
- It consumes the affine trace and replays the first-pass candidate-point
  arithmetic with plain modular instructions only.  It imports no verifier
  helpers and emits a native-kernel-shaped register stream.
- The 744/720/728 control artifact passed:
  3/3 field replays verified, 32 representative first-pass instruction streams
  matched ABI outputs, and 36 row-instance replays were consistent.  The 32
  representatives used 544 field ops: 32 inversions, 160 multiplications,
  64 additions, 256 subtractions, and 32 negations.
- The critical 744 reused event remains exact:
  `["12", 174, 469]` replays to `[9251, 9216]` from the pure field register
  prefix `4102`, `2558`, `6741`, `7122`, `927`.
- The full 10-record artifact passed:
  10/10 field replays verified, 77 representative first-pass instruction
  streams matched ABI outputs, 81 row-instance replays were consistent, 6
  below rho, and only one below-rho event-reuse target.  The 77
  representatives used 1309 field ops.

Recommended next command path:

1. Implement a small native/FFE candidate-point runner for the eight
   `coord1114_744` event keys that emits the same register names and values as
   `ffe_public_repeated_coordinate_fused_kernel_field_replay_target67_744_controls.json`.
2. Compare the native/FFE runner against the field replay artifact first, then
   pass the unchanged affine, ABI, contract, and second-pass relation gates.
3. Keep 720/728 controls in the same field-replay comparison so any accidental
   sharing beyond the known 744 reuse anchor is caught immediately.

Latest update after fused kernel native C reference:

- The native reference generator/checker now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_native_reference_probe.py`.
- It consumes the field replay, generates standalone C99 source, compiles it,
  and runs the resulting binary.  The C binary replays modular
  add/sub/mul/neg/inv instructions and checks every register value plus the
  candidate-point output.
- The 744/720/728 control artifact passed:
  C compiled and executed, 3/3 records verified, 32 native first-pass cases
  verified, 544 native field instructions verified, 0 failures, and the same 4
  `coord1114_744` reused groups.
- The full 10-record artifact passed:
  C compiled and executed, 10/10 records verified, 77 native first-pass cases
  verified, 1309 native field instructions verified, 0 failures, 6 below rho,
  and only one below-rho event-reuse target.

Recommended next command path:

1. Replace the generated C reference cases for `coord1114_744` with a compact
   native/FFE kernel function that computes the eight event-key outputs from
   operand arrays rather than static instruction records.
2. Compare that function's register stream to
   `ffe_public_repeated_coordinate_fused_kernel_native_reference_target67_744_controls.json`,
   then keep the same ABI/contract second-pass gates unchanged.
3. Keep the generated C reference as a regression oracle while optimizing the
   first-pass candidate-point kernel.

Latest update after compact native affine-add kernel:

- The compact native kernel generator/checker now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_native_kernel_probe.py`.
- It consumes the affine trace, generates C operand cases, and verifies one
  reusable `fused_candidate_point_kernel` function.  The kernel computes the
  slope registers and candidate point from operands instead of replaying static
  instruction records.
- The 744/720/728 control artifact passed:
  C compiled and executed, 3/3 records verified, 32 operand-driven native
  first-pass cases verified, 32/32 register streams matched, 32/32 candidate
  points matched, and the operation counts matched the native reference.
- The full 10-record artifact passed:
  C compiled and executed, 10/10 records verified, 77 operand-driven native
  first-pass cases verified, 77/77 register streams matched, 77/77 candidate
  points matched, 6 below rho, and only one below-rho event-reuse target.

Recommended next command path:

1. Specialize the compact kernel for `coord1114_744` by grouping the repeated
   right operand `[7537, 8016]` and denominator inverse `6741` across its eight
   event-key cases, while preserving the exact register outputs.
2. Compare the specialized kernel against
   `ffe_public_repeated_coordinate_fused_kernel_native_kernel_target67_744_controls.json`.
3. Only after the specialized 744 kernel passes, feed its outputs back through
   the unchanged ABI/contract and second-pass relation gates.

Latest update after shared-denominator native specialization:

- The shared-denominator specialization now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_shared_denominator_probe.py`.
- It groups native candidate-point cases by field, curve, left x-coordinate,
  and right point, then prepares `right_x - left_x` and its inverse once per
  group before computing each case from the varying left y-coordinate.
- The 744/720/728 control artifact passed:
  C compiled and executed, 3/3 records verified, 2 shared groups covered 32
  cases, 32/32 register streams matched, 32/32 candidate points matched, and
  operation count dropped from 544 to 484.  The `coord1114_744` group is the
  intended right point `[7537, 8016]`, denominator 2558, inverse 6741.
- The full 10-record artifact passed:
  C compiled and executed, 10/10 records verified, 6 shared groups covered 77
  cases, 77/77 register streams matched, 77/77 candidate points matched, and
  operation count dropped from 1309 to 1167.  The savings are exactly 71
  inversions and 71 subtractions.

Recommended next command path:

1. Turn the shared-denominator groups into an explicit batch/vector kernel:
   one denominator/inverse preparation per group, then a tight loop over the
   group left-y values.
2. Start with `coord1114_744`'s 8-case group and compare against
   `ffe_public_repeated_coordinate_fused_kernel_shared_denominator_target67_744_controls.json`.
3. Keep the compact native kernel, field replay, ABI, and contract artifacts as
   the regression ladder for each specialization.

Latest update after grouped-left-y batch lanes:

- The grouped-left-y batch kernel now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_batch_lane_probe.py`.
- It keeps the shared denominator/inverse per group, then computes only one
  candidate point per unique left-y lane and fans that output out to every
  event case in the lane.
- The 744/720/728 control artifact passed:
  C compiled and executed, 3/3 records verified, 2 shared groups collapsed 32
  event cases into 4 batch lanes, 4/4 lane outputs matched, 32/32 event cases
  matched, and field ops dropped from 484 to 64.
- The 744 event-reuse target now has only two actual candidate computations:
  left y 3914 -> `[9251, 9216]` and left y 5888 -> `[2705, 9753]`.
- The full 10-record artifact passed:
  C compiled and executed, 10/10 records verified, 6 shared groups collapsed
  77 event cases into 12 batch lanes, 12/12 lane outputs matched, 77/77 event
  cases matched, and field ops dropped from 1167 to 192.

Recommended next command path:

1. Promote the batch-lane kernel outputs back into the portable ABI contract:
   the ABI should charge first-pass candidate work by lane count, while still
   preserving the 77 event-case fanout checks.
2. Recompute below-rho work counters using lane count instead of event-key
   first-pass count, then compare against the existing 123/125 `coord1114_744`
   boundary.
3. Keep the old event-key contract as a conservative fallback while the
   lane-amortized contract is audited.

Latest update after batch-lane portable contract:

- The lane-amortized contract verifier now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_batch_lane_contract_probe.py`.
- It keeps the original portable contract's second-pass event checks and
  relation summaries, but charges first-pass candidate-point work by validated
  native batch-lane count.
- The 744/720/728 control artifact passed:
  3/3 contracts verified and derived, all 3 below rho under lane accounting,
  total work dropped from 376 to 350, and `coord1114_744` moved from
  123/125 = 0.984 rho to 117/125 = 0.936 rho.
- The full 10-record artifact passed:
  10/10 contracts verified, 10/10 derived, 8/10 below rho, total work dropped
  from 1232 to 1175, and mean below-rho lane cost is 0.85 rho.

Recommended next command path:

1. Add a batch-lane ABI guard that checks the lane outputs themselves against
   the curve/subgroup constraints, while preserving event-case second-pass
   checks.
2. Compare lane-amortized and event-key contracts side by side in the handoff
   so the conservative and optimized cost claims stay distinct.
3. Then push the lane-amortized contract into the next target window and see
   whether the 8/10 below-rho result survives beyond 672-744.

Latest update after batch-lane ABI/curve guard:

- The batch-lane ABI guard now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_batch_lane_abi_probe.py`.
- It reconstructs lanes from the affine trace, cross-checks the native
  batch-lane artifact, verifies each source-local lane on the curve and in the
  base-order subgroup, and confirms the lane-amortized contract still preserves
  second-pass fanout and relation-derived secret recovery.
- The 744/720/728 control artifact passed:
  3/3 records verified, 4 global native lanes covered 32 event cases, 6
  source-local contract lanes preserved 36 second-pass checks, and all 3
  records stayed below rho with mean cost 0.93333333 rho.
- The full 10-record artifact passed:
  10/10 records verified, 12 global native lanes covered 77 event cases, 20
  source-local contract lanes preserved 81 second-pass checks, 10/10 relation
  replays derived the recorded secret, and 8/10 records stayed below rho with
  mean below-rho lane cost 0.85.

Recommended next command path:

1. Run the lane-amortized contract and ABI guard on the next target-67 transfer
   window after 744, using the event-key contract as a conservative baseline.
2. Add a side-by-side contract comparison artifact that reports event-key
   fused ops, lane-amortized ops, and ABI status per source so below-rho claims
   cannot mix accounting models.
3. If the next window preserves the below-rho rate, move the batch-lane shape
   from generated C into a fixed FFE/summation-polynomial kernel interface.

Latest update after contract comparison:

- The side-by-side comparison probe now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_fused_kernel_contract_comparison_probe.py`.
- It consumes the event-key contract, the lane-amortized contract, and the
  batch-lane ABI guard, then emits explicit conservative and optimized claim
  permissions per source.
- The 744/720/728 control artifact passed:
  3/3 comparisons verified, conservative event-key below-rho count was 1/3,
  ABI-guarded batch-lane below-rho count was 3/3, and the lane-only promotions
  were exactly `coord161_720` and `coord161_728`.
- The full 10-record artifact passed:
  10/10 comparisons verified, conservative event-key below-rho count was 6/10,
  ABI-guarded batch-lane below-rho count was 8/10, total work dropped from
  1232 to 1175, and the only lane-only promotions remained `coord161_720` and
  `coord161_728`.

Recommended next command path:

1. Find or materialize a truly fresh post-744 repeated-coordinate recovery
   rather than replaying the transfer-744 case inside the 744-751 window.
2. Run the full ladder on that fresh source: event-key contract, batch-lane
   native kernel, lane-amortized contract, batch-lane ABI guard, and contract
   comparison.
3. Promote the generated C lane shape into a fixed FFE/summation-polynomial
   kernel interface only if the fresh comparison preserves ABI-guarded
   below-rho lane claims without relying on the old 744 recovery.

Latest update after 808-815 strict pair-rule replication:

- Fresh 808-815 stress now lives at
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_808_815_probe.json`.
- The frozen repeated-coordinate pipeline artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_808_815_fresh_manifest.json`
  and its sibling public-selector, coordinate-gate, and guard-replay files.
- The strict unchanged no-exact pair-rule replay now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_808_815.json`.
- The combined alignment audit now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_815.json`.
- Result: 808-815 is a strict-guard negative.  The coordinate gate verifies 11
  full repeated-coordinate cases, but none below rho; the best full replay is
  transfer 811, coordinate `(1114,8506)`, secret 7002 at 1.44 rho.  The pair
  rule finds 11 rank-2 relation cases at nominal 0.96 rho, but all have
  candidate-position signature `1,2`, span 1, misalignment, zero guard-passed
  cases, and zero verified recoveries.
- Across 752-815, the alignment audit has 43 relation-bearing strict pair cases:
  28 verified below rho and 15 unverified rank-2 decoys.  The positive
  form-only atoms `candidate_pos_aligned=1`, `candidate_pos_span=0`,
  `candidate_pos_unique_count=1`, and `candidate_pos_span0_count_ge2=1` still
  occur in all 28 verified cases and zero unverified cases.

Recommended next command path:

1. Mine a cheaper pre-event proxy for candidate-position alignment from the
   coordinate-gate rows: start with coefficient support signatures, RHS residues,
   salt deltas, and the FFE/summation-polynomial residuals that distinguish
   `1,1`/`2,2` from the 808-815 `1,2` decoys.
2. Keep 808-815 as a negative control when training that proxy: it should reject
   transfer 811 coordinate `(1114,8506)` despite the rank-2 nominal 0.96-rho
   relation cases.
3. Only then run 816-823 with the unchanged pair rule plus the new pre-event
   proxy; a plain unchanged-rule replay alone is now a calibration check, not a
   promotion step.

Latest update after pre-event proxy mining:

- The proxy miner now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_pre_event_proxy_miner.py`.
- The 752-815 proxy artifact now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_815.json`.
- It mines public-pair and form-residual surfaces while excluding
  candidate-position labels, exact coordinates, and exact salt signatures.
- Best public negative-control guard:
  `transfer_index_mod3=1`.
- Current corpus result: the guard rejects all 15 unverified rank-2 decoys from
  transfers 802 and 811, rejects zero of the 28 verified below-rho cases, and
  passes leave-one-window checks on the two negative windows:
  - hold out 800-807: rejects 4/4 unverified, 0 verified;
  - hold out 808-815: rejects 11/11 unverified, 0 verified.
- Positive public atoms are not yet a full selector.  `pair_salt_max_mod5=3` and
  `transfer_index_mod3=0` each cover 20 verified cases with zero unverified
  cases, but miss the later 776-783 and 800-807 positives.

Recommended next command path:

1. Add a small proxy-aware replay wrapper that applies the frozen
   `reject_if transfer_index_mod3=1` gate before counting pair-rule-selected
   relation cases.
2. Run that wrapper on 752-815 first to confirm it preserves the 28 verified
   below-rho cases and rejects the 15 known decoys without changing replay
   semantics.
3. Then run 816-823 with the unchanged pair rule, strict rank-lift guard, and
   frozen proxy gate.  If it rejects another `1,2` decoy without killing a
   positive, start looking for a summation-polynomial reason for the mod-3
   residue; if it kills a positive, demote it to an overfit calendar artifact.

Latest update after proxy guard filter audit:

- The proxy filter audit now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_proxy_guard_filter_audit.py`.
- The 752-815 filter artifact now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_815.json`.
- Applying `reject_if transfer_index_mod3=1` to the current relation-case corpus
  keeps all 28 verified below-rho cases, rejects all 15 unverified rank-2
  decoys, keeps zero unverified cases, and rejects zero verified cases.

Recommended next command path:

1. Materialize 816-823 stress and frozen repeated-coordinate pipeline artifacts.
2. Replay the unchanged pair rule with the strict guard on 816-823.
3. Run the proxy filter audit with `reject_if transfer_index_mod3=1` on the
   updated audit corpus.  The mod-3 proxy is promotable only if it avoids
   false-rejecting any verified below-rho relation case in the fresh window.

Latest update after 816-823 calibration:

- Fresh 816-823 stress now lives at
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_816_823_probe.json`.
- The frozen repeated-coordinate pipeline artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_816_823_fresh_manifest.json`
  and its sibling public-selector, coordinate-gate, and guard-replay files.
- The strict unchanged no-exact pair-rule replay now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_816_823.json`.
- The combined 752-823 alignment/proxy artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_823.json`,
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_823.json`, and
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_823.json`.
- Result: 816-823 is a no-relation calibration window.  The coordinate gate
  selected 116 target-67 candidates with zero verified public keys.  The strict
  pair rule selected 116 pair/subset cases but produced zero relation pair
  cases, zero guard-passed cases, and zero verified recoveries.
- The 752-823 relation-case audit remains 28 verified below rho and 15
  unverified rank-2 decoys, because 816-823 contributed no relation cases.  The
  frozen proxy `reject_if transfer_index_mod3=1` still keeps all 28 verified
  cases and rejects all 15 decoys, but 816-823 did not create a new proxy
  accept/reject challenge.

Recommended next command path:

1. Run 824-831 with the same stress, frozen pipeline, strict pair replay, and
   proxy audit.  The next useful evidence is a window that reaches relation
   cases again.
2. If 824-831 is another no-relation window, add a pair-selection cost gate that
   predicts relation-case reach before candidate-position alignment; the current
   bottleneck would then be dead public-pair selection, not false acceptance.
3. If 824-831 has relation cases, freeze `reject_if transfer_index_mod3=1` before
   inspection and test whether it rejects `1,2` decoys without rejecting aligned
   verified systems.

Latest update after 824-831 relation-reach calibration:

- Fresh 824-831 stress now lives at
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_824_831_probe.json`.
- The frozen repeated-coordinate pipeline artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_824_831_fresh_manifest.json`
  and its sibling public-selector, coordinate-gate, and guard-replay files.
- The strict unchanged no-exact pair-rule replay now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_824_831.json`.
- The combined 752-831 alignment/proxy/reach artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_831.json`,
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_831.json`,
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_831.json`, and
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_relation_reach_audit_target67_752_831.json`.
- Result: 824-831 is another no-relation calibration window.  The target-67
  coordinate gate replayed 12 candidates with zero verified public keys, and
  the strict pair rule selected 16 pair/subset cases but reached zero relation
  cases.
- The 752-831 candidate-position and proxy audits are relation-case unchanged:
  43 relation cases, 28 verified below rho, 15 unverified rank-2 decoys, and
  `reject_if transfer_index_mod3=1` still keeps all 28 verified cases while
  rejecting all 15 decoys.
- The new relation-reach audit changes the next bottleneck: across 752-831
  there are 511 selected strict pair/subset cases but only 43 relation-reaching
  cases.  The dead-selection windows are 784-791, 792-799, 816-823, and
  824-831; relation-reach windows are 752-759, 768-775, 776-783, 800-807, and
  the 808-815 decoy reach.

Recommended next command path:

1. Freeze a public relation-reach/dead-selection diagnostic that excludes exact
   coordinates, exact salts, verifier labels, and candidate-position labels.
   Start from the missed-only atoms in the 752-831 relation-reach audit, but
   treat them as diagnostics until replayed.
2. Run the unchanged stress, frozen pipeline, strict pair replay, relation-reach
   audit, and proxy audit on 832-839.  The first acceptance condition is reaching
   relation cases cheaply; candidate-position/proxy checks only become decisive
   after relation cases exist.
3. If 832-839 is also dead selection, demote the current no-exact pair rule to a
   relation-reach subproblem and mine a new FFE/summation-polynomial predictor
   for relation-case reach before spending more work on downstream alignment.

Latest update after 832-839 branch-gap calibration:

- Fresh 832-839 stress now lives at
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_832_839_probe.json`.
- The frozen repeated-coordinate pipeline artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_832_839_fresh_manifest.json`
  and its sibling public-selector, coordinate-gate, and guard-replay files.
- The strict unchanged no-exact pair-rule replay now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_832_839.json`.
- The combined 752-839 alignment/proxy/reach artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_839.json`,
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_839.json`,
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_839.json`, and
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_relation_reach_audit_target67_752_839.json`.
- A new branch-gap artifact now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_gap_audit_target67_832_839.json`.
- Result: 832-839 is not just another dead strict-pair window.  The coordinate
  gate verifies 31 target-67 public keys over rho, with two unique
  transfer/coordinate keys on `(55,7531)`, but the strict pair rule selects none
  of them.  The old row/form replay reaches 8 rank-2 relation cases at
  transfer 835 on `(542,3911)`, all with candidate-position signature `1,3`,
  but none verify and none are selected by the strict pair rule.
- The strict pair rule itself selects only 4 cases, all transfer 838 on
  `(161,6976)`, and reaches zero relation cases.  Across 752-839 the strict
  pair corpus is now 515 selected cases with only 43 relation-reaching cases.
- The proxy guard `reject_if transfer_index_mod3=1` remains clean on existing
  strict-pair relation cases, but 832-839 is outside its current usefulness
  because the strict pair rule produced no relation case.

Recommended next command path:

1. Mine a public branch selector over coordinate-gate candidates and old
   row/form relation cases, not only over strict pair-selected cases.  Use
   832-839 as a hard branch-gap diagnostic: a useful selector should expose the
   over-rho `(55,7531)` coordinate-gate branch and explain why the strict pair
   rule instead picked rank-0 `(161,6976)`.
2. Keep the candidate-position guard frozen as a downstream verifier of relation
   form.  The 832-839 row/form relation cases have signature `1,3`, so they are
   a new misalignment control distinct from the older `1,2` decoys.
3. Run 840-847 only after the branch selector is explicit.  Repeating the same
   strict pair replay without a branch-gap selector is likely to keep measuring
   selector reach failure rather than improving the FFE/summation-polynomial
   algorithm.

Latest update after 752-839 branch selector mining:

- The branch selector miner now lives at
  `tasks/ecdlp_index_calculus/ffe_public_repeated_coordinate_branch_selector_miner.py`.
- The 752-839 selector artifact now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_752_839.json`.
- It aggregates coordinate-gate candidates, old row/form replay cases, and
  strict pair-rule cases by transfer/coordinate key.  Candidate atoms exclude
  exact coordinate keys, exact salt signatures, verifier labels, and
  candidate-position labels.
- The mined corpus has 103 public keys across 11 windows: 13 coordinate-gate
  verified keys, 7 coordinate-gate verified keys missed by strict pair
  selection, 2 row/form relation keys, 1 row/form relation key missed by strict
  pair selection, 8 branch-gap keys, and 36 strict-pair dead keys.
- Best current branch-gap selector:
  `b_mod4=3&salt_mod2_pattern=0,1,0`.  It selects 4/8 branch-gap keys, 0 false
  positives, and spans 752-759, 776-783, and 832-839.
- Narrower coordinate-55 candidates:
  `b_minus_c_mod16=12&salt_mod2_pattern=0,1,0` and
  `b_minus_c_mod16=12&salt_span=6`; each selects 3 branch-gap keys with zero
  false positives across 776-783 and 832-839.
- The 832-839 row/form `1,3` relation branch has only one public key, so its
  best-looking separators remain single-window diagnostics.

Recommended next command path:

1. Run 840-847 with the same stress and frozen repeated-coordinate pipeline.
2. Before interpreting strict-pair output, evaluate the frozen branch selector
   `b_mod4=3&salt_mod2_pattern=0,1,0` on the 840-847 aggregate branch corpus.
3. Promotion criteria for the selector are modest but concrete: it should select
   a coordinate-gate verified or row/form relation key without selecting only a
   strict-pair-dead key.  If it selects nothing, treat it as an abstention; if it
   selects a dead strict-pair key, demote it to a calibration artifact.

Latest update after 840-847 branch-selector validation:

- Fresh 840-847 stress now lives at
  `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_840_847_probe.json`.
- The frozen repeated-coordinate pipeline artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_frozen_pipeline_target67_840_847_fresh_manifest.json`
  and its sibling public-selector, coordinate-gate, and guard-replay files.
- The strict unchanged no-exact pair-rule replay now lives at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_subset_pair_rule_replay_target67_pairminer_positive_decomp_no_exact_span0_count2_840_847.json`.
- New branch-selector artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_840_847_fixed_branch_rules.json` and
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_branch_selector_miner_target67_752_847.json`.
- The combined 752-847 alignment/proxy/reach artifacts now live at
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_candidate_position_alignment_audit_target67_752_847.json`,
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_pre_event_proxy_miner_target67_752_847.json`,
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_proxy_guard_filter_audit_target67_752_847.json`, and
  `ecdlp_index_calculus_state/ffe_public_repeated_coordinate_relation_reach_audit_target67_752_847.json`.
- Result: 840-847 is a useful mixed window.  The coordinate gate verified 16
  target-67 public keys, all over rho; the old row/form path reached no
  relation cases; the strict pair rule reached 12 relation cases and verified 8
  public keys at 1.032 rho on transfer 846 `(281,3224)`.
- The frozen branch selector
  `b_mod4=3&salt_mod2_pattern=0,1,0` selected the one coordinate-gate verified
  key still missed by strict pair, transfer 847 `(55,7531)`, with zero false
  positives.  Across 752-847 it now covers 5/9 branch-gap keys with zero false
  positives across four windows.
- The proxy guard remains clean on the expanded relation corpus: 55 relation
  cases total, 36 verified, 28 verified below rho, 19 unverified.  The frozen
  `reject_if transfer_index_mod3=1` rejects all 19 unverified cases and rejects
  zero verified cases.

Recommended next command path:

1. Promote the branch selector into an explicit branch-aware replay wrapper:
   first route `b_mod4=3&salt_mod2_pattern=0,1,0` coordinate-gate misses, then
   run a relation-producing subset search under the frozen downstream
   candidate-position/proxy guards.
2. Keep the narrower
   `b_minus_c_mod16=12&salt_mod2_pattern=0,1,0` as the coordinate-55 family
   control.  It validated fresh on 840-847; `b_minus_c_mod16=12&salt_span=6`
   should be demoted to a salt-span-specific diagnostic because it abstained on
   the fresh span-4 repeat.
3. Do not claim speedup yet.  The current repeatable progress is public branch
   reach plus clean relation-case rejection; the next acceptance condition is a
   branch-aware replay that produces verified below-rho relation systems on a
   future fresh window.

Latest update after branch-aware subset replay:

The branch selector has crossed the next acceptance bar on the current fresh
window.  Routing the frozen public candidate clause
`b_mod4=3&salt_mod2_pattern=0,1,0` into subset replay turns the missed 840-847
coordinate-gate key at transfer 847 `(55,7531)` from a 1.336-rho full replay
into a verified two-row recovery at 0.928 rho, deriving secret 5840.

The cheaper two-row package is stronger than the exploratory all-subsets
package.  Across 752-847 it selects 153 cases, reaches 43 relation systems,
verifies all 43, and preserves all 36 below-rho recoveries from the broader
all-subsets replay.  Window breakdown: 752-759 has 8/8 verified below rho,
776-783 has 24/24 verified below rho, 832-839 has 7 verified over rho and no
below-rho recoveries, and 840-847 has 4/4 verified below rho.

The next hypothesis should now freeze the candidate branch as:

```text
candidate_clause: b_mod4=3&salt_mod2_pattern=0,1,0
pair_rule: subset_size=2
form_guard: candidate_pos_span=0&candidate_pos_count>=2
```

Run the package unchanged on 848-855.  A verified below-rho recovery would be a
second fresh branch-aware speedup window; a clean abstention is acceptable; a
guarded unverified relation case should trigger a form/proxy audit; and an
over-rho-only verified result should be treated as branch reach without
speedup.  In parallel, explain why the repeated-coordinate salt pattern and
two-row subset expose rank-bearing summation-polynomial/FFE relations.  The
miner found narrow clean clauses such as
`pair_salt_delta_from_min_signature=0,2`, but 840-847 needs the broader
two-row rule because its below-rho pair has delta signature `0,4`.

Latest update after 848-855:

The frozen 848-855 check is a clean abstention.  The fixed-selector stress
source has zero verified stress-row or stress-leaf labels for target-cap1,
target-cap3, and global-cap3; the target-67 public selector emits zero public
bounded cases; the coordinate gate has zero candidates; and the frozen
branch-aware two-row replay has zero input pair cases.

This does not promote the algorithm, but it also does not hurt the package: no
guarded unverified relation case appeared, and there is no over-rho-only
branch-aware recovery to explain away.  The next useful frozen test is 856-863
with the same package.  In parallel, work on the algebraic explanation: the
current evidence says the speedup branch lives where the public residue/salt
pattern exposes a two-row repeated-coordinate relation with aligned candidate
positions, not where the public selector emits no target-67 bounded rows at
all.

Latest update after 856-871:

The next two frozen windows, 856-863 and 864-871, are also clean activation
abstentions.  In both windows the fixed-selector stress source has zero
verified stress-row or stress-leaf labels for target-cap1, target-cap3, and
global-cap3.  The target-67 public selector emits zero public bounded cases,
so the coordinate gate has zero candidates and the branch-aware two-row replay
has zero input pair cases.

This changes the next hypothesis from "run one more adjacent window" to
"separate activation from relation quality."  The branch-aware relation package
has one fresh positive on 840-847 and three clean post-positive abstentions on
848-871.  The missing piece is now a public pre-activation cue for when the
target-67 repeated-coordinate branch exists at all.  A useful next experiment
is a batched frozen activation scan over later windows, recording only the
public-selector count, coordinate-gate candidate count, and branch-clause
match count until another nonempty target-67 branch appears.

The current two-row relation evidence is still internally clean.  Across
752-847, verified below-rho cases have candidate-position signature `1,1` and
salt-delta signatures `0,1`, `0,2`, `0,3`, or `0,4`; the 832-839 verified
over-rho controls also have `1,1`, so candidate-position alignment is a
verification-form cue, not by itself a below-rho cue.  No guarded relation case
in the two-row replay fails public-key verification.

Latest update after batched activation scan through 903:

The post-positive scan now covers 848-903 with per-window frozen branch replay
artifacts.  All seven windows are activation abstentions: zero selected
target-67 public cases, zero coordinate-gate candidates, zero branch-clause
matches, and zero branch-replay input pairs.  This is stronger evidence that
the immediate blocker is not the two-row relation form; the branch never gets
presented to it.

The next hypothesis should therefore test activation, not replay.  Mine or
derive a public pre-activation cue from stress-level row features before the
target-67 public selector emits bounded repeated-coordinate cases.  The cue
should predict windows like 752-847, where coordinate candidates exist and the
branch clause can match, while rejecting the 848-903 dry spell.  Until a
nonempty activation window appears, running the same branch replay only proves
abstention again.

The algebraic angle to pursue is whether the FFE/summation-polynomial residual
structure has a cheap pre-signal for repeated-coordinate availability: repeated
monic-coordinate residues, salt parity pattern `0,1,0`, and two-row
candidate-position alignment appear after activation, but the current pipeline
lacks a public way to predict activation before materializing target-67
bounded rows.

Latest update after pre-activation mining and 680 form-boundary backtest:

The first pre-activation cue is simple but useful.  Over 672-903,
`has_target67_leaf_total3=1` separates all 22 target-67 activation windows
from the seven dry 848-903 windows with zero false positives and zero false
negatives.  For the narrower branch-clause target, the public salt-pattern cue
`target67_salt_mod2_pattern_0_0_1_count>=8&target67_salt_mod3_pattern_0_1_1_count>=1`
selects 6/8 branch-clause windows with zero false positives, missing 704-711
and 736-743.  This should be treated as a high-precision scheduler cue, not a
complete relation-success model.

The older-window branch-aware backtest adds a sharper form boundary.  The
frozen branch clause reaches relation cases on 680-687, but the strict
`candidate_pos_span=0&candidate_pos_count>=2` guard rejects them because their
candidate-position signatures are `1,3,3` and `1,1,3,3`.  These are not dead:
the `1,3,3` class has 11 verified below-rho cases at 0.976 rho, while
`1,1,3,3` has 11 verified over-rho cases at 1.032 rho.  The form-boundary
audit across 680 and 752-847 now records 43 strict same-position relation
cases, all verified with 36 below rho, plus 22 partial-duplicate relation
cases from 680, all verified with 11 below rho.

The next hypothesis is therefore a partial-duplicate guard audit, not an
immediate guard relaxation.  Candidate guard shape: allow a duplicate
candidate position with one additional public position only if an independent
public feature rejects the historical misaligned decoys and the over-rho
`1,1,3,3` class.  Test this against the 752-847 strict positives, the 680
partial-duplicate positives, the 808/832 decoys, and the 848-903 dry spell
before promoting it.  In parallel, keep the pre-activation cue as a cheap
window scheduler: run relation replay only after target-67 total3 public rows
are present, then apply the frozen branch clause.

Latest update after partial-duplicate guard audit:

The partial-duplicate guard audit is retrospectively clean.  The candidate
extension is:

```text
strict_same_position OR
(candidate_pos_count=3 & candidate_pos_unique_count=2 & candidate_pos_max_duplicate>=2)
```

On the combined weighted relation-case corpus from branch-aware replay,
strict-pair alignment, 832 row/form decoys, and 848-903 dry windows, this
extension accepts 75 verified below-rho cases and 15 verified over-rho strict
cases, accepts zero unverified cases, rejects all 27 known unverified decoys,
and rejects the 11 verified over-rho `1,1,3,3` controls.  It recovers exactly
the 11 below-rho 680 `1,3,3` cases that the strict guard rejected.

The next experiment should freeze this as an extension candidate, not a new
claim.  The forward package is now:

```text
activation_scheduler: has_target67_leaf_total3=1
candidate_clause: b_mod4=3&salt_mod2_pattern=0,1,0
pair_rule: subset_size=2
form_guard: strict_same_position OR partial_count3_unique2
```

Run that package only on the next nonempty activation window.  A below-rho
verification would be meaningful promotion evidence; an abstention remains
acceptable; any accepted unverified case immediately demotes the partial
extension back to diagnostic.  Do not spend more effort replaying 848-903
unless a different activation scheduler finds target-67 public rows there.

Latest update after executable partial guard replay:

The partial-count3 guard is now executable in the replay tooling, not just an
audit filter.  The exact forward guard string is:

```text
candidate_pos_span=0&candidate_pos_count>=2|candidate_pos_count=3&candidate_pos_unique_count=2&candidate_pos_max_duplicate>=2
```

Retrospective execution over 680, 704, 728, 736, and 752-847 selected 222 pair
cases, reached 65 relation cases, guard-passed 54, verified all 54, and kept
47 below rho with minimum 0.896 rho and mean 0.94414815 rho.  That reproduces
the partial-duplicate audit through the same branch-aware replay path future
windows will use.

The next action remains forward activation, not more retrospective replay.  On
the next nonempty 904+ target-67 window, run the fixed-selector stress source,
the frozen target-67 pipeline, and the branch-aware subset replay with:

```text
candidate_clause: b_mod4=3&salt_mod2_pattern=0,1,0
pair_rule: branch_two_row_subsets|pair_activate:subset_size=2
form_guard: candidate_pos_span=0&candidate_pos_count>=2|candidate_pos_count=3&candidate_pos_unique_count=2&candidate_pos_max_duplicate>=2
max_subset_size: 3
```

No 904+ target-67 activation artifact was available in the current worktree.
Treat a future clean abstention as useful scheduler evidence, a verified
below-rho relation as promotion evidence, and any guard-passed unverified
relation as an immediate demotion trigger for the partial-count3 extension.
