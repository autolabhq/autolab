# Next Hypothesis

Keep the frozen guarded portfolio unchanged and test the next transfer block
after 296-311.

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

The next block should be judged by the portfolio result, not by the best
single-policy quadratic-root audit.  The 296-311 window showed that a single
policy can have over-rho rows while the frozen guarded portfolio still matches
the guarded oracle.

Promotion evidence:

- The new window adds guard-passed selected surfaces.
- The rolling latest-window portfolio remains recovered, preserving, below rho,
  false-positive-free, and zero-gap to the guarded oracle.
- LOO with the new window remains false-positive-free and zero-gap.

Demotion evidence:

- Any guard-passed false positive.
- Any below-rho guarded-oracle selected surface missed by the public portfolio.
- A latest-window portfolio max charged cost above rho.
- A collapse in selected-surface count that suggests the guard is filtering
  away most fresh evidence.

Structural audit to run in parallel:

Measure whether `target_fingerprint_hash` selections on the latest 67.a1@9803
and 22050.cf1@11731 surfaces can be explained by public factor metadata:
factor total degree, monomial count, zero leaf index, coefficient support,
surface transfer index, and selected row salt.  A structural replacement for
hash ordering would be the cleanest route toward a real mechanism claim.
