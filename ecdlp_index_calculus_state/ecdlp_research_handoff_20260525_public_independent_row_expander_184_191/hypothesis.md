# Hypothesis

The 184-191 companion replay proved that the public retained-root path can
derive one fresh same-challenge key on `22050.cf1@11731`, but `67.a1@9803`
remained rank 1.  The next test asks whether this is only a row-diversity
problem:

1. Start from the public single-hit-root anchor selector.
2. Keep the same 184-191 candidate signature and challenge groups.
3. For each anchor challenge, add public below-rho cases from the same
   target/transfer group.
4. Rank additions by new row salts, row keys, leaf signatures, surface IDs,
   then lower public ops/rho.
5. Retain all row/leaf surfaces in the selected cases and replay through the
   verifier.

Selection still forbids public-key verification, relation count, rank,
preserving labels, false-positive labels, and below-rho labels as scoring
features.
