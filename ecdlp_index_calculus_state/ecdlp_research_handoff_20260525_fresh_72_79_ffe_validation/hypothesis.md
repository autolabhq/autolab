# Fresh 72-79 FFE Validation Hypothesis

## Hypothesis

The public low-term total-2 leaf family and target-scoped row selector that
held through transfer windows 48-71 should continue to produce verifier-backed
below-rho signatures on a disjoint transfer window 72-79. If those signatures
are not just replay artifacts, true finite-field resultant factorization should
again expose low-degree preserving factors, and a public factor-order policy
should select them below Pollard-rho after charging selector evaluations plus
root scan.

## Baseline

Before this run, the expanded all-public-leaf bank had:

- 33 below-rho signature cases
- 22 Sage-factorized FFE surfaces
- 22/22 preserving Sage factor root-scan paths below rho
- 22/22 charged public selector paths below rho
- best combined policy: `summax_sage_low_constant_target_hash`
- max charged ops/rho: 0.968

The remaining boundary was independent fresh-row validation beyond the
artifact-mined 22-surface bank.

## Changed Condition

Train row selectors on the previous calibration window:

```text
0-3,16-71
```

Then stress fixed public compact-leaf selector
`mode_cost_low_term_support_total2` on fresh transfer indices:

```text
72,73,74,75,76,77,78,79
```

The run used the existing public row policies:

```text
balanced,target_balanced,target_recall_first,target_rho_first
```

No Sage factor labels, preservation labels, or below-rho labels are used for
row or leaf selection.
