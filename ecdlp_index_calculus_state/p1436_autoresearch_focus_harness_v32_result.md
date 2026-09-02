# P1436 Autoresearch Focus Harness V32 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R82: Cartesian-sum compact divisor

R82 leaves one-dimensional coordinate cosets and freezes the scalar-blind
addition-pushforward factor base

```text
|A|=B^(2/5), |C|=B^(3/5), F={A_i+C_j}, |F|=B.
```

This produces the first passing compact-divisor S4 compiler in the current
lane:

```text
3F=3A+3C,
retained endpoint state            B^(9/5+o(1)),
fresh endpoint source query        B^(6/5+o(1)),
setup cap                          B^(9/4+o(1)),
fresh-query cap                    B^(5/4+o(1)).
```

All eight frozen instances are injective and every sampled local source
replays through group addition without scalar labels. Public rectangle
identities have rank `B-u-v+1`; verifier-only known-RHS rows reach the
remaining `u+v-1` meaningful directions and together span all nominal
columns.

The local compression does not survive the full relation:

```text
5F=5A+5C,
best explicit equality join        B^(13/5) work,
best explicit join state           B^(12/5),
generic collision baseline         B^(5/2)=N^(1/2).
```

Thus the full known-RHS source and identical descent return to rho scale.
Wagner filtering receives no credit because no addition-compatible proper
quotient exists in the prime-order group, and no subcap summation-polynomial
or FFE source solver is supplied.

Report SHA-256:

`ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832`

Gate SHA-256:

`7c34e1d905c95a756689d4ec0ea92c6bd47808bcb3407d858ce08cccf75fd55e`

Parent receipt SHA-256:

`7e1f819fef24d082cc0d361975d19d09e20594d88960656b2b45251eee1512e3`

## Harness routing

V32 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v32.json`

SHA-256:

`71104db7bd7f07d5499549f7d7fce49ffa592aeceff1e2d9224017f3a26a2aaf`

Schema: `ecdlp.p1436_autoresearch_focus_report.v24`.

Eighteen hash-bound lanes are closed. The top frontier is:

```text
s6_addition_compatible_5a5c_field_filtration
```

The next experiment must freeze one field-coordinate filtration for the
colored `5A+5C` source. Every partial constraint must compose under elliptic
addition, charge false positives, retain an exact jointly coupled source,
avoid verifier logs, fit `B^(9/4)` setup and `B^(5/4)` fresh work, and pass
the same construction on matched random decks.

V32 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              6ac9182209c74c37eb64521f434c05d3739160b7bdeb4985168ea694328d6c66
harness tests        082ca36aa00beb73470d8c47c233b0c5740e4817ab1cf91a384dd8f87f70ab43
V32 focus note       9a40c133a650f7229d1429b603bb8e3f9fba367b9fb75fc8514a8ac6bbe03705
V32 FFE inventory    fc89a8d7c934585850e24cea1f548a87c2d4c19d6cf377e2dd2e0ccf92585282
V32 FFE replay       26191ca7ee4b78e8d3c451fad34fb047732d1d1ff09ef363b551a01ea2a81ff2
```

## Verification

- R82 targeted tests: 5 passed.
- Focus-harness tests: 48 passed.
- Full ECDLP task suite: 136 tests passed.
- Seventeen R69-R82 and harness Python modules compiled.
- Seven R76-R82 parent YAML receipts parsed; all 73 declared input/artifact
  hashes and all seven archived-gate bindings matched.
- All seven parent receipts have `breakthrough=false`.
- A clean R82 rerun reproduced all six generated JSON artifacts byte for
  byte.
- V32 has 18 closed lanes, `promotion_allowed=false`, and the expected top
  field-filtration action.
- `git diff --check` and explicit trailing-whitespace checks passed after
  final freeze.
