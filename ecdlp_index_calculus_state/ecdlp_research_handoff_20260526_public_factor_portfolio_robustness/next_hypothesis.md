# Next Hypothesis

Keep the guarded public-factor portfolio unchanged for the next materialized
low-term total-2 FFE window:

```text
guard: selected_leaf_count_eq1_and_factor_zero_eq1
policy eligibility: include hash policies
min_surfaces: 2
min_preserve_ratio: 1.0
min_recovered_ratio: 1.0
max_false_positive_rate: 0.0
max_train_worst_ratio: unset
held-out selection: cheapest_guarded_public_zero
```

The next useful experiment is not another LOO audit.  It is a new materialized
window after `fresh_264_279_fixed`, then the same guarded portfolio run with
the frozen public settings above.

Promotion evidence:

- A new window recovers every guarded public-zero selected surface.
- Every recovered selected surface preserves selected root pairs.
- Zero quadratic false positives.
- Every selected surface remains below rho under the charged
  public-factor-quadratic-root cost.

Demotion evidence:

- Any guard-passed false positive.
- A new below-rho oracle surface that the public portfolio misses.
- Another hash-only rescue that suggests the factor order is acting as a
  target-specific hash lookup rather than a structural quotient signal.

The structural question to pursue in parallel is why hash-ordered public
factors rescue `67.a1@9803` transfer 268.  If the hash factor order is just a
stable public encoding of low-degree quotient shape, it should have measurable
features in the emitted factor metadata.  If not, keep it as an empirical
public selector but do not elevate it into a mechanism claim.
