# P1436 Autoresearch Focus Harness V22 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R71: S4 centered carry

Artifacts:

- `p1553_s4_centered_carry_rank_probe_r71.py`
- `p1553_s4_centered_carry_rank_probe_report_r71.json`
- `p1553_s4_centered_carry_rank_probe_gate_r71.md`

Report SHA-256:

`6ee62d61f6c567a5a65947b6b47026182bd961565dde3c1aee540cf3cbe5b72f`

R71 evaluates the exact target-specific

```text
S4 = Res_z(S3(x1,x2,z),S3(x3,xR,z))
```

on scalar-blind legal x-class decks. The closed resultant formula matches a
fraction-free Bareiss Sylvester determinant, and all 4,096 full-grid tuples
match exhaustive signed group-relation replay.

Across four small prime-field curves, two target types, every prefix
`B=3,...,8`, and all three modes:

```text
centered remainder mode rank = B
centered carry mode rank     = B
```

At `B=8`, the raw lift has exact mode rank five while both post-centering
tensors have rank eight. This rejects only the frozen canonical `S4`, `k=1`
carry representation.

## R72: S6 centered-carry minors

Artifacts:

- `p1553_s6_centered_carry_rank_minor_probe_r72.py`
- `p1553_s6_centered_carry_rank_minor_probe_report_r72.json`
- `p1553_s6_centered_carry_rank_minor_probe_gate_r72.md`

Report SHA-256:

`7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43`

R72 tests the actual five-label predicate

```text
S6 = Res_z(
       S4(x1,x2,x3,z),
       S4(x4,x5,xR,z)
     )
```

on secp256k1, P-256, P-384, and P-521. Five public SHA-256 x-coordinate decks
of size 18 are scalar-blind. Blind and forced-positive target controls produce
1,944 exact size-three predicate replays with zero Semaev/group mismatches.

Two exact integer lifts are tested:

1. least-nonnegative curve coefficients; and
2. centered curve coefficients.

The decisive rank result is uniform across four curves, two targets, five
modes, two lifts, and two auxiliary primes:

```text
raw S6 mode rank           17
centered remainder rank    18
centered carry rank        18
full carry profiles       160 / 160
```

Degree 16 in each `S6` variable gives the raw rank upper bound 17; matching
modular minors prove exact raw rational rank 17. Every named carry has a
certified rational CP-rank lower bound of 18. The arity-six low-carry-rank
exception is therefore absent from both natural resultant lifts.

This is finite and lift-specific. It does not lower-bound the Fermat projector,
another integer lift, or a non-CP trace contraction.

## Harness routing

Artifact:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v22.json`

SHA-256:

`9ab05fee4c799346f6612abf9e4cdd18832c9b59b166adc8218e32652d8a3d34`

Schema: `ecdlp.p1436_autoresearch_focus_report.v14`.

Eight hash-bound lanes are closed:

1. `constructive_closure_collision`;
2. `factor_line_direct_root_equivalence`;
3. `idea340_public_chart`;
4. `multiplicative_x_s3_closure`;
5. `presurface_full_charge`;
6. `s4_centered_carry_rank`;
7. `s6_centered_carry_rank_minor`; and
8. `slice_quadratic_public_source`.

The sole top frontier is now:

```text
s6_noncp_balanced_trace_contraction
```

Its required object is R9/R10's exact restricted projector-trace contraction:
compute full-box and adaptive-child root counts, plus one occurrence-labelled
source, without a centered carry, `B^3` triple table, `B^5` quotient, dense
character table, or root-presupposing projector factors.

The circuit grammar must pass the rank-two sparse multiplicative-convolution
control, retain zero signatures and rectangle identity, and expose direct
setup, state, online, workspace, and source-recovery costs.

Promotion remains withheld.

## Verification

- Full ECDLP task suite: 90 tests passed.
- Python compilation: passed.
- R71 and R72 exact JSON nonclaim checks: passed.
- V22 diagnostic-only and nonpromotion checks: passed.
- R71/R72 parent YAML parsing: passed.
- `git diff --check`: passed.
