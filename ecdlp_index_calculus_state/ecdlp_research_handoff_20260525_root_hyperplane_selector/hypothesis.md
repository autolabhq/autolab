# Hypothesis

The measured FFE first-fall route is not behaving like a generic high-degree
resultant search.  After Sage factorization, every public factor in the
72-79 plus public-leaf bank is a root hyperplane

```text
c + r*b + r^2
```

for the monic leaf quadratic `x^2 + b*x + c`.

This suggests a sharper index-calculus subroutine:

1. Build the FFE quotient/resultant surface from summation-polynomial leaves.
2. Factor only until first-fall linear hyperplanes in `(b,c)` appear.
3. Rank candidate roots `r` using public factor fingerprints or held-out public-zero priors.
4. Evaluate `c + r*b + r^2` on selected leaf coefficients until the first public zero.
5. Treat the hit root `r` as the recovered leaf root and charge selector work plus root recovery against generic Pollard-rho.

If this holds on fresh streams, the algorithmic object is a public
root-hyperplane selector rather than a black-box factor selector.
