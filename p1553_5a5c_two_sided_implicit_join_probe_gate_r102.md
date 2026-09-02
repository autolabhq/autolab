# P1553 5A/5C Two-Sided Implicit Join Gate R102

Date: 2026-07-29

Status: `EXACT_OVER_CAP_JOIN__DENSITY_ONE_FILTER_CONSERVATION`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can the two passing R101 local side oracles be composed into an exact fresh
`5A+5C` source inside `B^(9/4)` setup/state and `B^(5/4)` fresh
work/workspace?

R102 freezes all atom-count splits, one canonical exact source convention,
and disjoint source partitions before outcomes.

## Exact canonical join

The best setup-eligible direct splits are `4A+1C` versus `1A+4C` and their
complementary orientation:

```text
stored canonical prefixes = B^(11/5)
fresh canonical suffixes   = B^(14/5).
```

The stored state fits `B^(9/4)`, but the fresh query misses `B^(5/4)`.
Boundary indices make the split canonical, so every unordered `5A+5C`
source is counted exactly once.

Across eight actual family/offset instances, the canonical all-target
histograms equal direct enumeration. Every first source replays, all sampled
fresh queries return exact integer counts and one source, and blind, identity,
and repeated-atom targets are exact. A cyclic duplicate-atom control
preserves joint multiplicity.

## Split optimization

R102 exhausts all 34 nontrivial atom-count splits. Fifteen fit the setup cap.
The largest setup-eligible state exponent is `11/5`; therefore `14/5` is the
smallest direct complementary query exponent.

## Filter conservation

The ordered five-factor source universe and subgroup order both have exponent
`B^5`. Bounded permutations and pairings add only constant representation
slack.

Reducing the `B^(14/5)` query to `B^(5/4)` by an additional independent or
disjoint filter requires pruning exponent

```text
14/5 - 5/4 = 31/20.
```

At density one, such a filter retains only `B^(-31/20)` true
representations. Constant success therefore needs `B^(31/20)` classes or
repetitions, restoring

```text
5/4 + 31/20 = 14/5.
```

Exact scalar-blind source partitions modulo `2,4,8,16` conserve every source
incidence and recover all positive targets only through their class union.

This closes extra thinning filters only. It is not a lower bound on a
target-forced algebraic identity satisfied by every true join.

## Admission

Passed obligations: `24/34`

Exact over-cap baseline admitted: `true`

Full lane admitted: `false`

Missing gates include a target-forced algebraic filter, projective and proper
subsum branches, known-RHS rank, factor logs, identical descent, a
generic-prime family algorithm, and a complete Shoup comparison.

## Exactly one next action

Construct or refute one target-forced algebraic join filter for `L+R=T`.
Freeze a public rational invariant and its Semaev/FFE relation before
outcomes. Every true source must survive without polynomial thinning; compact
state must fit `B^(9/4)` and a fresh exact count/coupled-source query must fit
`B^(5/4)`, with false positives and every exceptional branch charged.
