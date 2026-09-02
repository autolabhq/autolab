# Next Hypothesis

Keep the frozen guarded portfolio unchanged and test the next transfer block
after 312-327.

Frozen settings:

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

The 312-327 result changes the emphasis: the raw public-factor audit can now
produce false positives, so future promotion should be judged at the guarded
portfolio layer and should report how many raw rows were filtered.

Promotion evidence for the next block:

- The latest-window guarded portfolio is recovered, preserving, below rho, and
  false-positive-free.
- The latest-window guarded portfolio has zero gap to the guarded oracle.
- LOO remains false-positive-free and zero-gap after adding the new block.
- Raw false positives, if present, are rejected by a public guard rather than
  by label inspection.

Demotion evidence:

- Any guard-passed false positive.
- Any below-rho guarded-oracle selected surface missed by the public portfolio.
- A latest-window portfolio max charged cost above rho.
- A collapse in selected-surface count that suggests the guard is filtering
  away most fresh evidence.

Structural audit to run next:

Study the 312-327 raw false-positive row and the selected guarded replacements.
Compare factor metadata for:

```text
false-positive surface:
  target=22050.cf1@11731
  transfer=324
  row_key=22050.cf1@11731:uniform:256:salt167
  policy=global_fingerprint_hash
  selected_leaf_count=3

guarded latest-window selected policies:
  low_constant
  low_degree_then_order
  target_fingerprint_hash
```

The goal is to turn the guard from an empirical filter into a structural
explanation: why does `selected_leaf_count=1 and factor_zero_leaf_count=1`
separate preserving factors from below-rho false positives?
