# Hypothesis

The 192-199 hard-target positive still passed through a strict signature file
whose membership was defined with verifier labels.  The next hypothesis is
that the same retained-root result can be recovered from raw stress rows using
only public row/leaf fields:

1. Read total3/total4 raw fixed-selector stress output.
2. Select cases by public cost, row selector, leaf selector, top-k, and
   row/leaf keys.
3. Forbid public-key verification, relation count, rank, source verification,
   and row-selector verification in the selector.
4. Feed the public-selected cases into the existing pre-factor FFE gate,
   Sage factorization subset, prefactor unique-leaf root policy, and relation
   replay harness.

The follow-up 200-207 test additionally allows bounded over-rho public rows
to check whether the FFE root route can rescue candidates whose raw relation
scan is not below rho.
