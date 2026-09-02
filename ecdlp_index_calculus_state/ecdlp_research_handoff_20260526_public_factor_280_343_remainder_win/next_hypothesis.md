# Next Hypothesis

Keep the frozen guarded portfolio unchanged, but split the next work into two
tracks: another transfer-window extension and a structural hunt for more
full-remainder FFE wins.

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
- Any new full-remainder FFE below-rho candidate preserves the selected root
  pairs without using a label-selected factor choice.

Demotion evidence:

- Any guard-passed false positive.
- Any below-rho guarded-oracle selected surface missed by the public portfolio.
- A latest-window portfolio max charged cost above rho.
- A collapse in selected-surface count that suggests the guard is filtering
  away most fresh evidence.
- Full-remainder wins that vanish when the selected leaf count is forced to 1
  or when the factor-order policy is frozen from prior windows.

Structural work order:

Study the full-remainder witness and compare it to the other preserving
328-343 surfaces:

```text
full-remainder witness:
  target=22050.cf1@11731
  transfer=342
  row_key=22050.cf1@11731:uniform:256:salt165
  candidate=sage_resultant_factor_0
  selected_leaf_count=3
  full_remainder_monomials=79
  full_remainder_ffe_ops_over_rho=0.96350365

near misses:
  target=22050.cf1@11731, salts 173 and 175, full remainder 1.16-1.26 rho
  target=67.a1@9803, salts 202/204/205, full remainder 2.31-3.71 rho
```

The next useful mechanism question is why the `22050.cf1@11731` transfer-342
surface has a small enough full remainder while nearby preserving factors do
not.  Candidate explanatory variables are selected leaf count, resultant
factor index, factor monomial count, hit-root count, and target-specific
quotient sparsity.
