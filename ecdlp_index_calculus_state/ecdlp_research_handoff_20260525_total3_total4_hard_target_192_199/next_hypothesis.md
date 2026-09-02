# Next Hypothesis

Promote the 192-199 hard-target result only after removing the remaining
signature-row dependency.

Recommended preregistered 200-207 test:

1. Freeze the public row selector family from the 192-199 success.
2. Use only total3/total4 public leaf selectors.
3. Build an assembly selector that chooses target, transfer, row salt, top-k,
   and leaf selector from public row/leaf scores plus the FFE pre-factor gate.
4. For each selected FFE surface, keep only leaves chosen by that public
   selector, not by verifier-backed signature labels.
5. Sage-factor only pre-factor gated surfaces.
6. Apply the prefactor unique-leaf single-hit-root root policy.
7. Replay same-challenge groups and compare against Pollard-rho.

Success criterion:

Promote if a fresh `67.a1@9803` or new hard-target group reaches public-key
verified rank 2 below the rho proxy with no signature-provided row/leaf
selection.

Failure criterion:

If the standalone selector loses the transfer-196 style relation, preserve the
192-199 row/leaf trace and train a public scorer for the specific total3/total4
leaf signatures `[0, 2, 3]` and `[0, 1, 2, 3]`.
