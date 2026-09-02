# Next Hypothesis

Keep the frozen guarded portfolio unchanged and test whether the
full-remainder below-rho profile repeats for transfer block 360-375.

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

Promotion evidence for the next block:

- Latest-window guarded portfolio is recovered, preserving, below rho, and
  false-positive-free.
- Latest-window guarded portfolio has zero gap to the guarded oracle.
- LOO remains false-positive-free and zero-gap after adding the new block.
- At least one full-remainder FFE below-rho candidate repeats the sparse
  profile: 3-monomial factor surface, 79 full-remainder monomials, about 12
  known hit roots.

Demotion evidence:

- Any guard-passed false positive.
- Any below-rho guarded-oracle selected surface missed by the public portfolio.
- A latest-window portfolio max charged cost above rho.
- A collapse in selected-surface count that suggests the guard is filtering
  away most fresh evidence.
- Full-remainder wins that require label-selected factors or disappear when
  the public guard is enforced.

Structural work order:

Compare the two full-remainder wins:

```text
328-343:
  target=22050.cf1@11731
  transfer=342
  row_key=22050.cf1@11731:uniform:256:salt165
  selected_leaf_count=3
  full_remainder_monomials=79
  full_remainder_ffe_ops_over_rho=0.96350365

344-359:
  target=22050.cf1@11731
  transfer=348
  row_key=22050.cf1@11731:uniform:256:salt165
  selected_leaf_count=1
  full_remainder_monomials=79
  full_remainder_ffe_ops_over_rho=0.93430657
```

The row key and 79-monomial remainder profile are stable across both wins,
while the transfer index and selected leaf count change.  That suggests the
next mechanism search should focus on row-envelope/salt-specific quotient
sparsity rather than treating selected-leaf count as the primary cause.
