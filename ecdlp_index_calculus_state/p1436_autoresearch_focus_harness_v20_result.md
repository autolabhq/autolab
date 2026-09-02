# P1436 Autoresearch Focus Harness V20 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## Constructive closure-collision gate R69

Artifacts:

- `p1553_constructive_closure_collision_gate_r69.py`
- `p1553_constructive_closure_collision_gate_report_r69.json`
- `p1553_constructive_closure_collision_gate_r69.md`

Report SHA-256:

`401b8ae56607f4962c9ae10ed99889d5c27ac865d0e27b3359f7b163a04dd451`

R69 proves the exact rank identity that a constructive step introducing `f`
fresh atom variables and at most `t` independent equations changes nullity by
at least `f-t`. A line, summation, or factored-fiber step with one fresh
residual and one row cannot reduce unresolved-log nullity. Only a residual
already represented independently, or reached by another path, can add
rank-reducing information.

The scalar-blind order-103 replay freezes 12 public seeds and evaluates all 66
seed pairs:

- fresh residual rows: 48;
- rank of fresh rows: 48;
- nullity after fresh rows: 12;
- closure collisions: 18;
- independent collision rows: 11;
- final atom count/rank/nullity: `60/59/1`.

The public identity and generator orient the one-dimensional nullspace. All 60
factor logs verify. A caller-supplied target absent from precomputation has 18
decompositions, all yielding the unique verified log `53 mod 103`.

The positive toy result is above rho before omitted costs:

- rho baseline: 13 group operations;
- source pair proposals: 66, or `5.0769` rho;
- online target lookups: 60, or `4.6154` rho.

Under the uniform residual model, obtaining `Theta(B)` independent collisions
requires `Omega(sqrt(B*N))` proposals. This is a model boundary, not an
unrestricted lower bound.

## Multiplicative-x S3 screen R70

Artifacts:

- `p1553_multiplicative_x_s3_closure_screen_r70.py`
- `p1553_multiplicative_x_s3_closure_screen_report_r70.json`
- `p1553_multiplicative_x_s3_closure_screen_gate_r70.md`

Report SHA-256:

`9e58c6178eb18b7535c59e117ad942465d6c7853890291dec2c7c0abdb4ffd89`

R70 freezes size-eight scalar-blind bases from public multiplicative subgroups
of `F_p^*` on four distinct prime-field curves. It compares each base with 32
matched SHA-256 bases and independently verifies every candidate relation with
the third Semaev polynomial.

Candidate independent collision ranks are:

```text
p193/q103: 3, control mean/max 3.03125/5
p257/q281: 1, control mean/max 1.34375/3
p337/q163: 4, control mean/max 1.56250/4
p449/q463: 0, control mean/max 1.03125/3
```

No family exceeds its control maximum, and the signal does not transfer. All
112 candidate `S3` checks pass exactly. Each complete 28-pair probe exceeds
its toy rho baseline, and no sub-pair locator or target descent is supplied.
The raw multiplicative-x prefix candidate is therefore closed.

## Harness routing

Artifact:

`p1436_autoresearch_focus_report_seed1432001_exact_v20.json`

SHA-256:

`ba7c7dff21fcd71121227d3c621207db91b044c1129e6041063fad579235c6e4`

Schema: `ecdlp.p1436_autoresearch_focus_report.v12`.

Six hash-bound lanes are closed:

1. `constructive_closure_collision`;
2. `factor_line_direct_root_equivalence`;
3. `idea340_public_chart`;
4. `multiplicative_x_s3_closure`;
5. `presurface_full_charge`; and
6. `slice_quadratic_public_source`.

The single top frontier remains `structured_closure_collision_locator`, now
with the explicit restriction that FFT-compatible coordinates receive no
credit without a prospective density theorem, independent transferred rank,
an exact sub-pair locator, and fully charged descent below rho.

Promotion remains withheld.

## Verification

- Full ECDLP task suite: 82 tests passed.
- Python compilation: passed.
- Exact JSON nonclaim checks: passed.
- `git diff --check`: passed.
