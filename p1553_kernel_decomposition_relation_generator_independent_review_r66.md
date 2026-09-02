# Independent review: R66 kernel-decomposition relation generator

Date: 2026-07-20

Verdict: `PASS` after provenance, relation-coefficient, and literal-cost
corrections.

The reviewer independently replayed the three-dimensional multiplication
kernel, all 1,158,780 first-right candidates, the completion histograms, every
divisor-disjointness filter, and equality with the unique R65 primitive line.
The exact normalized-section relation

```text
65*s1 + 112*s2 + 16*s3 = 0 mod 193
```

was reproduced. All eleven loaded P1553 source dependencies match the hashes
persisted in the report.

The initial review required three corrections. The generator is now scoped to
precomputed scalar-labelled toy catalogs rather than called unconditionally
public; the normalized three-term coefficients are stored; and all live
transitive sources are pinned. A final review then corrected the literal table
builder cost to `Theta(N log N)` group operations from `N` independent
double-and-add multiplications plus `O(N log N)` sorting. The sub-rho-blocking
conclusion and all kernel counts were unchanged.

Reviewed corrected hashes:

```text
script  8cd9ff9530616bb44d621814b9db3e8ec2f9c5db67b10ea5436d51d45067d693
report  753c16d1f15e444517ac7bc503fd9f888d8219e724e1dced6c384ed6b80a4081
gate    7032c45a1f49249951bcad021ff2ac6b7510be84b102e3ad7348fce586a3bf58
```

No file was edited by the reviewer.
