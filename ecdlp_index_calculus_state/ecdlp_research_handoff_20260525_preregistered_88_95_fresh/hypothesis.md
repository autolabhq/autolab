# Hypothesis

A preregistered pre-factor FFE gate can be applied to fresh strict below-rho
low-term-support signatures before Sage factorization, and the gate-selected
surfaces should preserve the same root-hyperplane structure that was observed
in the 72-79 and 80-87 controls.

The gate is intentionally narrow:

- materialize the full public FFE/summation-polynomial surface;
- require at least one selected root pair on the full surface;
- require the selected public leaf to have a known hit root before
  factorization;
- do not use Sage factorization, root policy output, public-zero recovery, or
  below-rho cost when deciding whether a surface enters the factorization set.

If this holds on fresh 88-95 strict signatures, the next proof obligation is to
freeze root selection from earlier windows and test it on unseen windows without
choosing a best selector after seeing the fresh factors.
