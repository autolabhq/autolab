# Next Hypothesis

Freeze a single root-selection rule using only windows through 87, then test it
on the fresh 88-95 manifest-selected surfaces and on the next unseen window.
The first acceptable result should preserve the full preregistered order:

1. generate fixed row/leaf signatures;
2. write strict signature candidates;
3. write the pre-factor gate manifest;
4. factor only gate-selected surfaces;
5. apply the frozen root selector;
6. compare scan and direct-root costs against Pollard-rho.

Immediate work orders:

- Build a combined total2 plus total3/4 frozen-policy evaluator that trains or
  selects policy only from <=87 artifacts, then scores 88-95 without choosing
  the best policy after factorization.
- Extend the same manifest-first path to 96-111 or to unseen target labels.
- Track gate precision and recall against post-factor preserving factors, but
  do not use those fields to change the gate on the same fresh bank.
- Separate three accounting lines in every report: full-remainder FFE cost,
  conservative root-scan cost, and direct-root cost.
- Promote only results that produce verifier-backed relation rank and public
  key verification, not merely low polynomial-operation estimates.

Stop condition for the current line:

If a frozen <=87 policy misses the 88-95 preserving root hyperplanes or loses
below-rho cost on the next unseen window, keep the pre-factor gate but treat the
post-factor selector family as diagnostic rather than algorithmic.
