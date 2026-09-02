# P1436 autoresearch harness V125 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V125 binds R176 as the 112th closed frontier lane and routes the first
experiment to `s68_factored_trilinear_elliptic_resultant` at priority 314.

## R176 Result

Let the R167 compact target witness be

```text
h = F_num / F_den,
div(h) = sum_T [T] - sum_R [R],
```

with auxiliary poles `R` chosen away from all selected pair sums. Then

```text
h(P+Q) = 0  if and only if  P+Q is a retained target.
```

This is a signed condition. It handles `P=Q` through the elliptic group law,
without differentiating the interpolated `V` side table or invoking a separate
tangent approximation.

Across six controls, all 8,922 selected pair sums avoid the auxiliary poles.
Exactly 241 values vanish: 5 diagonal and 236 off-diagonal. Their 140 left
roots are exactly the R175 and R167 candidate roots.

## Pontryagin Product

For a selected subset `A` of size `m` and full selected divisor `D` of degree
`n`, define

```text
A * D = sum_{P in A, Q in D} [P+Q].
```

This effective Pontryagin cycle has degree `mn`, and

```text
H(A) = h(A * D) = product_{P in A, Q in D} h(P+Q).
```

`H(A)=0` exactly when `A` contains an R175 candidate. The principal function
therefore replays all 358 R175 balanced-tree query paths and zero patterns.

Every queried cycle also verifies

```text
sum(A * D) = n sum(A) + m sum(D).
```

## Completed Reciprocity

When the pair-cycle sum is nonzero, R176 appends its negative `C` and constructs
a generalized Miller function with divisor

```text
A*D + [C] - (mn+1)[O].
```

When the sum is zero, it uses `A*D-mn[O]`. All completion values and
auxiliary-pole products are units in the controls.

Weil reciprocity gives

```text
h(div f) = f(div h),
```

and therefore, for a finite completion,

```text
h(A*D)
  = f(targets) / f(auxiliary poles) * h(O)^(mn+1) / h(C).
```

Candidate nodes are the zero specialization and are never inverted. Every one
of the 358 completed-cycle identities is exact:

```text
literal disjoint-support nonzero identities:  42
candidate-specialized zero identities:       316
```

## Cost Boundary

Ordinary reciprocity removes the explicit target-factor loop only after a
degree-`mn` pair-sum cycle or principal function has been represented. The
finite queried nodes use:

```text
pair-cycle degree volume:                    56,468
generalized Miller merges:                   56,468
charged Miller probe evaluations:           868,648
maximum deterministic shuffle retry index:        1
```

These are finite correctness counts and receive no asymptotic credit.

At campaign scale:

```text
compact target principal witness h:             B^(5/4)
compact selected divisor state:                 B^(9/4)
root Pontryagin cycle degree n^2:                B^(9/2)
balanced queried pair-cycle volume:             B^(9/2)
represented completed Miller function:          B^(9/2)
direct h-SLP evaluation on pair cycle n^2 N:     B^(23/4)
fast represented evaluation after pair listing: B^(9/2)
conditional factored trilinear resultant:        B^(9/4)
R163 label/backpointer postprocessing:           B^2
rho proxy:                                       B^(5/2)
```

Thus ordinary represented Weil reciprocity improves on direct triple
expansion but remains above rho. This closes only represented pair-cycle,
principal-function, and generalized Miller routes. It is not an arithmetic-
circuit lower bound.

The remaining primitive is a factored trilinear elliptic resultant that
returns `h(A*D)` directly from

```text
(U_A,V_A), (U_D,V_D), h
```

with one softly `O(n+N)` reusable setup and softly `O(m+N)` work per balanced
node, without constructing `mn` pair sums or an equivalent `n^2` kernel.

R176 passes 16 of 23 obligations. It admits principal signed incidence,
completed-cycle construction, exact R175 tree replay, and represented Weil
reciprocity only. Factored-resultant admission, lane admission, rho
improvement, Shoup improvement, and breakthrough flags remain false.

## V125 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v112`.
- Bound and closed frontier lanes: 112.
- First focus: `s68_factored_trilinear_elliptic_resultant`.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R176 focused tests: 18 passed in 29.548 seconds.
- Harness tests: 142 passed in 1.152 seconds.
- Full ECDLP suite: 1,183 passed in 564.172 seconds.
- R176 clean replay: all six JSON outputs byte-identical.
- V125 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R176: 101 receipts, 1,982 recursive path/hash bindings,
  zero mismatches, missing paths or rounds, and duplicate rounds.

## R176 Hashes

- Producer: `6d487f22177941a03c5533c0ee185456a4ea831b31df389e679f03783afe73ef`
- Report: `6d00706a225c0e1e2bb087a20f2e20ca327acfa25067f63241bb4e7302521632`
- Frozen interface: `743d55aeb444a5b05d566b0178e3bb1d9582cef8d6d8d3127e5bb8e76d12c874`
- Cost ledger: `a76a53b4e0c0f59815617f329e36b61675b17e8fa050f173a361e688526fb8f5`
- Replay: `387040fa5d26ef5798f1db553632010b556e9b3c1ffce031e14336350b60d9a4`
- Controls: `1fd6f58d2ae0f0fe16407af76bb16b22e740398b692595c3f58642a344e514ae`
- Resultant ledger: `967f07d990aa8296f3256daacd5632b8ad8c8410e5e1e94cefc351901bd1a5d2`
- Tests: `dab2a234da7837967b7d7dc0483a257f30bde2a5e5c3f655cc747b591ad50f61`
- Gate: `a64140ac0bd7a0b05cc55edf69a4334c105992e799a1be0534f7b0cf78604a56`
- Parent: `c9bd7f7782376a4b147efca899fd1ff4ab072edc029798f5e3d9d9e59a11bd3f`

## V125 Hashes

- Harness: `0db936b5e848a01cd6027da7dbcbf52742489b6d880c7b3e8e62f077a8ccda60`
- Harness tests: `c8411306eede707768d3944e0faf02e38e51396ea95277212ebd54996b73c38d`
- Focus report: `411199e7dd34ecc56b2769e28dca06768e4e49ede8e1de3ba926e23296fa7871`
- Note: `d67bb56db51be9947342fd39a358be7c26a158528547d943467ac1a5c8efdd2b`
- Evidence inventory: `f7e90a592d5a3d612f42d4fb0cc808963d97aaa79390362451a01b4653ddcf9c`
- Replay plan: `56b21870285ec5b96573b2ccef61247399fd0070b8200a0d6d002e2b63871dd1`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact signed identities, finite
controls, generalized Miller evaluations, verifier passes, and represented-
degree negatives receive no unconditional attack or circuit-lower-bound
credit.

## Next Action

Construct or refute a factored trilinear elliptic-resultant arithmetic DAG on
compact `A,D,h` inputs. Reject explicit `mn` pair sums, a degree-`mn` principal
function or Miller program, `n^2` tensor or displacement state, target-
dependent per-node preprocessing, candidate inversions, and unit-cost
resultant, norm, root, count, marginal, rank, source, DLP, or generic locator
oracles.
