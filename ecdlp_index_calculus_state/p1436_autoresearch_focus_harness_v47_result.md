# P1436 Autoresearch Focus Harness V47 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R97: factored transposed projector trace

R97 freezes the standard pointwise transposed grammar for

```text
delta_0(h_i) = 1-h_i^(p-1)
delta_0'(h_i) = h_i^(p-2) in F_p.
```

On blind nonzero vectors over `F_101` with dimensions `4, 8, 16, 32`, the
pointwise projector Jacobian has rank `D`. The pullbacks of all `2D-1`
complete dyadic interval masks also have rank `D`.

The finite source controls are exact. A unique zero gives projector count one
and an exact dyadic source; a blind vector gives count zero and bottom; two
zero occurrences give count two.

The product gradient localizes a unique zero with one nonzero coordinate, but
becomes the zero vector when two zero occurrences are present. It therefore
does not provide multiplicity-complete source semantics.

For `D=Theta(B^(12/5))`, pointwise Fermat evaluation, a source-valued balanced
product tree, a full reverse adjoint, and complete linearized dyadic state all
retain the `B^(12/5)` exponent. They miss the `B^(9/4)` setup/state and
`B^(5/4)` fresh-work/workspace caps.

This is a scoped negative. The finite full-rank result is not an asymptotic
lower bound on a nonlinear tensor-tower trace whose states are built directly
from compact A/C divisor circuits.

```text
R97 producer  f9015647b49a246680258ce259dfa49a80a237172162e34f3e4390b87b726972
R97 report    84217a15910b0ed15b6b69c3ec8875f7f2b913a17e4c70fc89cc905c25f492f6
R97 gate      ef91d8a15dc7a218c21355b5b2f2777db948978514ebed26d6884ce929c09182
R97 parent    8255339cba2516b4bce22f4c2247cf4c2a4650b343d0069e6b693da2d3f63703
R97 tests     6d99038725043186a208ea63f56437d153cd9722ac81a62906f3afc8174e88b0
```

R96 details and receipts are recorded in
`ecdlp_index_calculus_state/p1436_autoresearch_focus_harness_v46_result.md`
with SHA-256
`bc00f3ad710ff575fdaf886af172869916f21e1d5a92dc69e7b4f497e68979a9`.

## Harness routing

V47 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v47.json`

SHA-256:

`efdbb8c6b3b757cca364728182c3451d1803c8116352b9c0af870426dcab2db1`

Schema: `ecdlp.p1436_autoresearch_focus_report.v39`.

Thirty-three provided hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_nonlinear_tensor_tower_trace
```

The next experiment must freeze a tensor algebra and target transition whose
node states arise directly from compact A/C divisor circuits. It receives no
credit if it emits source leaves, quotient bases, moment vectors, linearized
dyadic adjoints, or an equivalent `B^(12/5)` body.

V47 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              edea81cc55f9c84b7eadf6f18dd17dfd824ddfdce5e23a28dc80715c0b232ef8
harness tests        a7018748e417dbad7f0ba4120fdcb1d3b60b058c5db428c422ed305f5ab58ae0
V47 focus note       2ef021316b70c13a112bcd9eed970f16495aa709fd8fe5b8ae6ab60cec0f9955
V47 FFE inventory    feda2991d60696bdb5f88ca4cce9aeab2c2ed2b25d3832f901b406aed4859df6
V47 FFE replay       d9ec2181b5de81c2001b1f69de5a6dbace023d027915d551404b07ac33d8da9f
```

## Verification

- R97 targeted tests: 8 passed.
- Harness tests: 63 passed.
- Full ECDLP task suite: 239 tests passed.
- Twenty-three R76-R97 and harness Python modules compiled.
- Twenty-two R76-R97 parent YAML receipts parsed.
- All 308 declared input/artifact hashes matched.
- No R97 parent or nested JSON artifact sets `breakthrough=true` or
  `shoup_bound_improvement=true`.
- A clean R97 rerun reproduced all six generated JSON artifacts byte-for-byte.
- V47 has 33 closed lanes, `promotion_allowed=false`, and the expected
  nonlinear tensor-tower trace action.
