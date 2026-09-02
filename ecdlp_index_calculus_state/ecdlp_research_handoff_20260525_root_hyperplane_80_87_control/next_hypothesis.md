# Next Hypothesis

The next useful experiment should not chase more vacuous preserving labels.
It should materialize a fresh 80+ bank with two gates enforced before selector
scoring:

- `original_selected_root_pair_count > 0`
- `public_zero_root_count > 0`

If those gates can be met without looking at the eventual selector ordering, a
frozen public root policy can be evaluated honestly on the remaining surfaces.
The primary comparison should report both conservative scan cost and direct-root
companion cost against Pollard-rho.

The current diagnostic control says the gate is worth pursuing: on the three
80-87 surfaces that satisfy both conditions after factorization, the fixed
`low_root_norm` policy is 3/3 below rho with max scan ops/rho 0.96 and max
direct-root ops/rho 0.888.  The next run must make this gate public or
preregistered before selector evaluation.

Concrete next work orders:

1. Build a materializer for total3/total4 or total2 leaves that records
   non-vacuous selected root-pair counts before selector evaluation.
2. Reuse the 72-79 policy family, but freeze selection on old transfers before
   scoring the 80+ bank.
3. Report three counts separately: materialized surfaces, public-zero-capable
   surfaces, and non-vacuous selected-root-pair surfaces.
4. Treat full-remainder FFE cost as a blocker unless a separate trick drives it
   below rho; the 80-87 control still has min full-remainder ops/rho 1.928.
