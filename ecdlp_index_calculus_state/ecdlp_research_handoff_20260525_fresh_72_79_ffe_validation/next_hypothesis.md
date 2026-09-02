# Next Hypothesis

## Immediate Next Test

Run the same public row/leaf selector and FFE factorization pipeline on a second
fresh transfer window:

```text
80,81,82,83,84,85,86,87
```

Use calibration through 79 and keep the leaf selector frozen to
`mode_cost_low_term_support_total2`.

Success criteria:

- at least one verifier-backed below-rho signature case on the fresh window
- all fresh FFE surfaces Sage-factorized with no missing requested surfaces
- all selected public factor paths preserving selected root pairs
- charged selector-plus-root-scan ops/rho below 1.0 for every selected surface
- no false-positive source rows

## Follow-Up Algebra Test

Investigate the transfer-76 `22050.cf1@11731:uniform:256:salt168` full-remainder
below-rho case. The first check should distinguish a useful structural
remainder win from a vacuous root-empty preservation case.

Suggested audit:

- compare every preserving factor on that surface, not only the minimum
  remainder candidate
- require `selected_surface_zero_leaves > 0` or nonempty selected root recovery
  before counting a full-remainder win as relation-bearing
- test whether the same low-remainder structure appears on salt162 transfer 76
  or only on the root-empty salt168 surface
- if it repeats, search for a public coefficient identity or affine
  transformation that explains the lower remainder cost

## Boundary

Do not claim a finished ECDLP index-calculus algorithm until the route produces
either relation-derived recovery on fresh targets with honest amortized cost or
a reproducible structural reason why the public FFE factor path scales beyond
these frontier windows.
