# Next Hypothesis

Keep the frozen guarded portfolio unchanged, but stop treating
full-remainder below-rho as a generic per-block expectation.  The next
experiment should target the row-envelope sparsity condition directly.

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

Promotion evidence for the next block or mechanism audit:

- Latest-window guarded portfolio remains recovered, preserving, below rho,
  and false-positive-free.
- LOO remains false-positive-free and zero-gap after adding the new block.
- A public rule predicts the 79-monomial full-remainder profile before using
  preservation or below-rho labels.
- The predicted row-envelope/salt profile repeats beyond
  `22050.cf1@11731:uniform:256:salt165`.

Demotion evidence:

- Any guard-passed false positive.
- Any below-rho guarded-oracle selected surface missed by the public portfolio.
- A latest-window portfolio max charged cost above rho.
- Further latest-window selected-surface collapse.
- Full-remainder wins only appearing after label-selected factor choices.

Structural work order:

Compare the full-remainder-positive and boundary rows:

```text
positive profile:
  target=22050.cf1@11731
  row_key=22050.cf1@11731:uniform:256:salt165
  transfers=342,348
  full_remainder_monomials=79
  full_remainder_ffe_ops_over_rho=0.96350365,0.93430657

boundary profile:
  target=67.a1@9803
  row_key=67.a1@9803:uniform:256:salt207
  transfer=364
  full_remainder_monomials=92
  full_remainder_ffe_ops_over_rho=1.208

false-positive boundary:
  target=22050.cf1@11731
  row_key=22050.cf1@11731:uniform:256:salt165
  transfer=363
  false candidate=sage_resultant_factor_4
  selected_leaf_count=3
```

The next useful probe should mine public row-envelope features that predict
full-remainder monomial count and known-hit-root count.  Treat selected-leaf
count as a secondary variable: it changed across the two full-remainder wins,
while row key `salt165` and the 79-monomial profile remained stable.
