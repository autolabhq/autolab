# Independent review: R61 product-hypersurface locator

Date: 2026-07-20

Verdict: `PASS` after one cost-accounting correction.

The reviewer independently reconstructed the rank-six multiplication map,
confirmed kernel dimension three, reproduced the degree ranks
`6,21,56,126,252,461`, and verified the unique degree-six equation. An
independent triangular 28-point unisolvent grid gave full rank on both ternary
factor spaces and zero on all 784 pullback evaluations, confirming that the
sextic pullback is identically zero. The R60 restriction and factorization

```text
179*t*(t+160)*(t+164)*(t^2+27*t+151)
```

were reproduced, including roots `0,29,33` and the product point at infinity.

The initial review found that the no-field-scan wording omitted the current
toy preprocessing: two `C(N,3)` catalog enumerations, subgroup calibration,
and replay. R61 was corrected to scope the scan-free claim to online use after
cached preprocessing and to record the current `Theta(N^3)` setup as uncharged
and unamortized. The reviewer then replayed the corrected artifacts and found
no remaining issue.

Reviewed corrected hashes:

```text
script  b8331ae9534fde907df8cffda074d2443f219324174fee6f49756cd40351bbfb
report  f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64
gate    a737e50f7b587ccae70edf988ca9687f4706778db549b3ca58432463017fa688
```

No file was edited by the reviewer.
