# Next Hypothesis

The next selector should keep the public total3/total4 FFE gate and add a
public proxy for relation-form independence.

Recommended test:

1. Use the public total3/total4 stress selector as the candidate generator.
2. Keep the pre-factor gate and prefactor unique-leaf single-hit-root policy.
3. Add a deterministic public diversity term before replay:
   target/transfer, row salt, row schedule key, root-hyperplane root,
   selected leaf signature, and top-k.
4. Select small same-challenge bundles that maximize those public diversity
   features instead of only minimizing ops/rho.
5. Replay the bundles and report unique forms, duplicate forms, rank, derived
   secret, and public-key verification.

Success criterion:

Promote if a fresh window repeats the `67.a1@9803` style rank-2 public-key
derivation using only public total3/total4 candidate selection and public FFE
root-policy fields.

Failure criterion:

If root-diverse bundles still produce rank-1 or non-derived rank-2 systems,
the next branch should learn a public relation-form independence predictor from
event summaries, not add more low-term leaves.
