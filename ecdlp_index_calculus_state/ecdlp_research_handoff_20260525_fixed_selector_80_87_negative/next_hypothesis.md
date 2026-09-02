# Next Hypothesis

The fresh 80-87 miss suggests the useful signal is not just a stable
low-term-support total-2 leaf rule. The next test should treat total-2 as a
seed surface and let the FFE/summation-polynomial layer search for the missing
rank or cheaper public factor route.

Recommended next probes:

1. Materialize an FFE surface for the over-rho `67.a1@9803` transfer 80 motif
   using salts 204, 205, and 206 with shared leaf 3. Score whether a public
   factor selector can reduce the charged root route below rho without false
   positives. This is now the best verifier-backed total-k diagnostic. Use
   `low_term_total3_total4_verified_over_rho_diagnostic_signature_80_87.json`
   as the signature source on a Sage-capable host.
2. Materialize an FFE surface for the over-rho `67.a1@9803` transfer 82 motif
   using salts 201 and 204 with shared leaf 10. Score whether a public factor
   selector can reduce the charged root route below rho without false positives.
3. Do not keep widening low-term-support caps on 80-87 as the next move. Total-3
   and total-4 improved verified recall but still produced zero strict
   below-rho certificates; the failure mode is now charged cost and rank/public
   verification, not leaf cap alone.
4. Add a transfer-window holdout audit for selector salt bands. The 48-79
   positives cluster around previous salt anchors; 80-87 needs a public offset
   or anchor penalty validated without using 80-87 verifier outcomes.
5. Keep the FFE scoring standard from the 72-79 validation: a candidate only
   counts when public selector evaluation plus root scan is below generic rho
   and the selected root pairs are preserved in Sage-backed factors.

Stop condition for direct leaf-count extension on this branch has now fired:
total-3/total-4 also failed to produce a verifier-backed below-rho case on
80-87. Shift the campaign away from direct leaf-count extension and toward
first-fall relation harvesting across the already factored 72-79 surfaces plus
targeted FFE scoring of the verified over-rho 80-87 motifs.

Sage note: this sandbox does not currently expose `sage` on PATH, so the next
FFE factorization needs to run either on the live host with Sage available or
after installing/exposing Sage to this worktree.
