# P1436 autoresearch harness V119 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V119 binds R170 as the 106th closed frontier lane and routes the first
experiment to `s62_slp_streaming_output_sensitive_target_norm_mod_u` at
priority 302.

## R170 result

In `A=F_p[X]/(U)`, let

```text
c_j(P) = U(x(T_j-P)),   C = product_j c_j.
```

R170 proves and replays that the corrected constant term of R169's scalar
resolvent equals `C` componentwise by R167 Weil reciprocity. Consequently,
the lambda-zero Fitting support is exactly `gcd(U,C)`: it is the existing
aggregate target norm up to explicit units, not a new constructor.

Across six controls, every target-factor interpolant, aggregate product,
corrected Fitting value, and candidate gcd is exact. The candidate factors
contain exactly the 140 R167 roots. All target factors and all aggregate
elements have degree `n-1` and 100% coefficient density. This finite density
receives no circuit lower-bound credit.

## Cost boundary

```text
compact target-divisor SLP state:       B^(5/4)
N represented quotient-ring factors:    B^(7/2)
standard fraction-free product/PRS:      B^(7/2)
represented aggregate element:           B^(9/4)
swapped Fitting matrix:                   B^(9/2)
expected candidate factor:               B^(3/4)
signed verification:                     B^2
rho proxy:                               B^(5/2)
```

The output element fits below rho, but the known represented constructors do
not. The surviving primitive is an SLP-streaming target norm that emits the
aggregate element or gcd in less than `B^(5/2)` without `N` dense quotient-
ring elements, an `nN` coefficient body, or an `n^2` matrix.

Prokofev and Zabrodin's complex sigma-function elliptic Cauchy identities are
bound as primary literature. Their full-matrix factorization does not provide
the required finite-field, candidate-safe streaming algorithm.

R170 passes 16 of 24 obligations. Lane admission, rho improvement, Shoup
improvement, and breakthrough flags remain false.

## V119 routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v106`.
- Bound and closed frontier lanes: 106.
- First focus: `s62_slp_streaming_output_sensitive_target_norm_mod_u`.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R170 focused tests: 16 passed in 25.54 seconds.
- R170 plus harness tests: 152 passed, 6 subtests passed in 26.09 seconds.
- Full ECDLP suite: 1,083 passed, 10 subtests passed in 269.24 seconds.
- R170 clean replay: all six JSON outputs byte-identical.
- V119 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R170: 95 receipts, 1,836 recursive path/hash bindings, zero
  mismatches, missing paths or rounds, and duplicate rounds.

## R170 hashes

- Producer: `71f94407c05de7c0a4bcc77643c63e5f601b1c609639cbdba874a72c56f0091d`
- Report: `a354427601094c8346253ac38bc0fd53f878e67d71962846561088ee6495162c`
- Frozen interface: `34735b27f7b9d9a018a2c158cccaab4c9d61942144636d72b4cdcdc95d9d9f97`
- Cost ledger: `97cf12e1011c10148588da6febba2edbcfaa445f12fcb4df12ca81dc11278113`
- Replay: `84a36d4ab086664e4a2382e6b057f45b741dd43c38c7acd19026c0ff8f51d3cd`
- Controls: `1c00d68dcc15217ce1d3b7192e0992ccb9a1443450df3d100cb1d1beb2fd409f`
- Fitting ledger: `fa7b5427696e244c257fa95168e3b7f5646c9bdff5d10ff25fd0b8193d28bea1`
- Tests: `51f82e0b8e762591669eec9aaa240e80c44b34ec1e7e56e80ca8ee83e955d2f3`
- Gate: `564ad5ffd47317768209244b1432f13cc2a0d1245a3d582e59c9bb0174375e02`
- Parent: `a6bcb432f330f492669dbb597361d3cc9c7ceff468292a3b00bebe231965df91`

## V119 hashes

- Harness: `f41db6ee0aa3587e6f1e57d2860ece385d927bd3b71263a74d2147dbee871ff2`
- Harness tests: `e1065a52fe00a9edd60529e5debad67ad1635ade30071f14ba3425fa019e79a5`
- Focus report: `8caa8e838124fcb874608e1db493dcdf993470d916e729df23d1c9a06ffce377`
- Note: `c3a83a886c2ab42befa7fe232abebcc3f3c22dfbe0f395f05c4103c6e68f62ad`
- Evidence inventory: `0182df3c92c552e0382cd5216777938b12c855841102cc8a96942f79c9b86319`
- Replay plan: `55eb15fb446098a3116a8a2ce50a56db2549c97a8dc0e9eedf70be0eed8ebe34`
- Elliptic Cauchy paper: `43e94a255b6999505893f6ac1d4c70facee3500dda758b30465cd793dc0fab74`

## Claim boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact equivalence, dense finite
controls, and verifier passes receive no asymptotic attack credit.

## Next action

Open the generalized Miller line SLP and seek a merge law whose target norm can
be updated directly modulo `U` with additive `n+N` state. Any merge that
expands one dense quotient-ring element per target closes this route at
`B^(7/2)` rather than advancing it.
