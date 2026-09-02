# P1436 Autoresearch Focus Harness V26 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R76: exact subset-incidence count

For each nonempty subset `S` of the at-most-four signed Kummer endpoints,
R76 constructs occurrence histograms `h_P(S)` for three-deck prefixes and
`h_Q(S)` for target-plus-two-deck suffixes. The exact tuple count is

```text
sum_(S nonempty) (-1)^(|S|+1) h_P(S) h_Q(S).
```

The identity preserves duplicate factor occurrences and corrects pairs that
share more than one endpoint. A synthetic control has four intersecting tuple
pairs: naive singleton counting returns six, while Mobius inversion returns
four.

On secp256k1, P-256, P-384, and P-521 at `B=6,10,14,18`, all 32 target
instances match direct deduplicated endpoint incidence. Every blind target
has count zero, every forced target has count one and a verified source, and
every dyadic parent count equals the sum of its children.

At `B=18`, all `87,480 = 15*B^3` prefix subset contributions and all
`4,860 = 15*B^2` suffix contributions are distinct. The construction avoids
the literal `B^5` grid but costs:

```text
prefix work/state        B^3
fresh-target work        B^2
```

Both exceed the direct `B^(9/4)` setup/state and `B^(5/4)` online caps.

Report SHA-256:

`de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93`

Gate SHA-256:

`14761973b0063d680b4a7dc5a8821dfdd3e9d1a631be3f6880f1ee2786bb6900`

Parent receipt SHA-256:

`7b5136f376f0c8f7fbd381fbb4c913bc97c2bbac69def5465d2bfb4fe03e8561`

## Harness routing

V26 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v26.json`

SHA-256:

`311f91f3dd66e935fb6c668363f9fed7f6a387c8b75723469279622b3f2be28a`

Schema: `ecdlp.p1436_autoresearch_focus_report.v18`.

Twelve hash-bound lanes are closed. The top frontier is:

```text
s6_target_translated_subset_frequency_oracle
```

It must compress target-independent prefix advice below `B^(9/4+o(1))` and
answer a fresh translated target below `B^(5/4+o(1))`, without enumerating
either `B^3` prefix triples or `B^2` suffix pairs. It must retain R76 exact
counts, duplicate/multiple-root correction, blind zero, one source, and every
dyadic child, with the existing Query2P1 gates bound.

V26 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. All alphaXiv-derived
autoresearch guidance checks pass.

## Verification

- Full ECDLP task suite: 106 tests passed.
- Four changed Python modules compiled.
- R76 parent YAML parsed.
- All seven locally declared R76 input/artifact paths and hashes passed.
- The archived R2 gate hash is bound through the present R31 registry.
- R76 and V26 both retain `breakthrough=false` and nonpromotion status.
- `git diff --check` passed after final freeze.
