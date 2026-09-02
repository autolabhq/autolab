# Hypothesis

The 72-79 FFE quotient route may not need generic high-degree resultant
factorization. If the Sage factors in the measured low-term-support bank all
fall to linear root hyperplanes of the form

```text
c + r*b + r^2
```

then the observed public-factor success is really a first-fall root-selection
problem for the monic leaf quadratic `x^2 + b*x + c`, not a broad multivariate
factorization problem.

This would give the next algorithmic target a sharper shape:

- generate low-term-support leaves;
- use public features to select a small set of candidate roots `r`;
- test the root hyperplane `c + r*b + r^2 = 0`;
- charge selector work plus root tests against generic Pollard rho.

The audit should only use already factored Sage-backed surfaces, and should not
claim fresh generalization beyond the measured surface bank.

