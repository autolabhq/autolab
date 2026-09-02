# Next Hypothesis

The next useful experiment is a root-hyperplane harvester that avoids full Sage
factorization during selection:

1. Use the already measured root distributions from the audit to build public
   root priors per target/salt band. Do not rely on root frequency alone:
   held-out top-8 root priors hit preserving roots on only 19/28 surfaces.
2. For a fresh or held-out surface, test a bounded set of candidate `r` values
   through the equation `c + r*b + r^2 = 0` on selected leaves.
3. Verify whether the resulting root pairs derive the target public key.
4. Charge selector evaluations plus root tests against generic rho.

The success gate should match the recent strict policy:

- public-key verified;
- selected root pairs preserved;
- total selector plus root-test ops below generic rho;
- no false positives on held-out transfer windows;
- no use of the fresh window's verifier outcomes to choose the root policy.

Recommended next work order:

- Implement `ffe_first_fall_root_hyperplane_selector_probe.py`.
- Train root priors on the existing 48-79 factored surfaces.
- Add public root features beyond frequency, such as target-specific root hash,
  salt offset, low/high residue buckets, and selected leaf-index context.
- Hold out by transfer index and target.
- Then run the fixed root-hyperplane selector on the diagnostic 80-87
  verified-over-rho surfaces, especially `67.a1@9803` transfer 80 with salts
  204, 205, and 206, leaf 3.

If the bounded root-hyperplane selector fails on 80-87, switch to first-fall
relation harvesting over multiple surfaces rather than further widening leaf
caps.
