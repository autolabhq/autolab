# P1553 distinguished branch-value mass gate R24

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 complete-image and
  support-co-design lane; no new idea ID.
- Evidence: exact deterministic counting theorem; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_NEGATIVE`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: a prospective small alphabet of planted zeros, poles,
  or other distinguished `psi` values touches too little of the complete
  endpoint/fifth source rectangle to create asymptotically useful global
  compression. This is not an unrestricted support theorem, Shoup-bound
  improvement, or ECDLP breakthrough.

R23 proves that an independently sampled fifth deck is almost injective. The
first adversarial escape is to choose the fifth deck and `psi` together so
one translated branch repeatedly lands in a planted fiber. R22's training
data visibly used this mechanism: its norm-zero keys carried a disproportionate
share of the observed collision excess.

R24 closes the asymptotic small-alphabet version by a direct source-mass
count. A degree-`d` Kummer map has at most `d*r` preimages above `r`
distinguished values. For each fifth class and each such preimage, at most two
endpoint Kummer classes put that preimage on one of the two translated
branches. Thus the entire distinguished stratum has at most `2*L*d*r`
geometric sources, independent of how the endpoint deck was chosen.

In the campaign rectangle `S=B^2`, `L=B`, and `d<=B`, every `r=o(B)`
alphabet covers only `o(SL)` sources. A zero bit, pole bit, one planted full
fiber, or any other constant-value tag therefore cannot by itself produce
the `Theta(d)` complete-image compression required by R13.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R13 erased-image geometric-resolution gate | `78ff09226852f140a8c33ce5ba98106ed62007b7126673d3f3af2c49d08a2cc5` |
| R21 quadratic-intermediate-field gate | `ad8d9e83b919ea92c98eefcb657831f01378b6760d49b2aaea0a892d6ed52be1` |
| R22 list-only toy gate | `ba1661d183f4e8b7291a1a34b48bdab2d01485da45c9753bbfb16e82b52208ab` |
| R22 exact search report | `40ef33016a6e12b802ee7c40183c853e4d180d89b7993abf85f1d6e6c48a5af8` |
| R23 random fifth-deck correlation gate | `de9a0a496fc6696ed6e976351476c56262484124c8c7d7cc2cdbddd16a6cc184` |
| R23 parent report | `12494c453b4c9c3fa45b72c4f1de76b11bbbeb8a2bf6a6dd14d6dede6f6a4b25` |
| R23 bundle hash list | `29bfdbeca9cd14a579f134ca2b5926e3fb4858f13455e1d8dfc066cbcb1008da` |
| R23 staging receipt | `5239f8c910e0d83b26eea5aa80ca6ef63bc58639b938c4b070cc32f365d47bde` |

No theorem in R24 depends on an empirical R22 count.

## Frozen model

Let `G=<P>` have odd prime order `N` different from the field characteristic,
and put

```text
X=(G minus {O})/{plus or minus 1}.
```

Fix arbitrary distinct Kummer supports

```text
U subset X, |U|=S,
Q subset X, |Q|=L,
```

and a separable rational Kummer map

```text
psi:P^1->P^1, degree d,
f=psi composed with x:E->P^1.
```

For `u=[A]` and `q=[Q]`, the complete translated branch divisor is

```text
F_u(q)=[f(A+Q)]+[f(A-Q)].
```

All values are projective. In particular, `infinity` may be included in a
distinguished value set and denominator poles are counted, not discarded.

Let `C subset P^1` be any prospectively fixed set of `r` distinguished output
values. Define its prime-subgroup Kummer preimage

```text
Z_C={z in X union {[O]}:f(z) in C}.
```

Because `psi` has degree `d`, including multiplicity and infinity fibers,

```text
|Z_C|<=d*r.                                          (1)
```

## Theorem 1: exact distinguished-branch mass bound

Define the geometric source stratum

```text
H_C={(u,q) in U x Q:
     at least one value in F_u(q) belongs to C}.
```

For a fixed fifth Kummer class `q=[Q]` and one preimage class `z=[Z]`, the
condition that `z` be one of the two branch classes leaves at most

```text
u=[Z+Q] or u=[Z-Q].
```

These expressions include all sign choices after quotienting by `plus or
minus 1`. They also cover `Z=O`, tangent, return, and infinity cases in the
complete projective map. Therefore, by (1),

```text
|H_C|<=2*L*|Z_C|<=2*L*d*r.                         (2)
```

The bound is deterministic and does not assume random endpoints, random
fifth classes, generic coefficients, or a common-factor model. It permits
the supports to be adversarially co-designed with `psi`.

Dividing by the full geometric source count `n=S*L` gives

```text
|H_C|/n<=2*d*r/S.                                  (3)
```

Thus a distinguished branch mechanism can cover a fixed positive fraction
`alpha` of the source rectangle only if

```text
r>=alpha*S/(2*d).                                  (4)
```

For `S=Theta(B^2)` and `d<=B`, constant source mass requires

```text
r=Omega(B).
```

Every `r=o(B)` distinguished alphabet covers `o(n)` sources.

## Theorem 2: a small special stratum cannot drive global compression

Suppose a proposed router's only certified nontrivial key identifications
involve sources in `H_C`: every source outside `H_C` remains a singleton and
shares no key with `H_C`. Even granting that all of `H_C` collapses to one
key, its attained complete image obeys

```text
K>=n-|H_C|+1
 >=n-2*L*d*r+1.                                    (5)
```

To certify `K<=c*n/d` for a fixed constant `c` and growing `d`, equations
(2) and (5) require

```text
r >= (S/(2*d))*(1-c/d-o(1))=Omega(S/d).           (6)
```

The premise is deliberately scoped to the claimed mechanism. R24 does not
assert that sources outside `H_C` cannot collide for another reason; R23 and
R21 address two such reasons under their own hypotheses. It says that a
zero, pole, or small-value bit cannot take credit for compression of sources
it never touches.

## Consequences for natural planted-fiber proposals

### One zero or pole fiber

For `C={0}` or `C={infinity}`, equation (2) gives

```text
|H_C|<=2*L*d.
```

At `S=B^2`, `L=B`, `d<=B`, this is at most `O(B^2)` of the `B^3` sources.
Even perfect collapse of that stratum saves only an `O(1/B)` fraction of the
global image.

### Constant-many planted fibers

Any fixed number of values, including separately tagged zero, pole, tangent,
or chosen endpoint fibers, has the same asymptotically vanishing mass. Adding
constant-many projective charts changes only the absolute constant.

### A growing value alphabet

The first mass-admissible escape has

```text
r=Theta(S/d)=Theta(B)
```

and a preimage support `Z_C` of possible size `Theta(S)`. Passing the mass
gate is not compression: the constructor must still prove that the complete
unordered branch pairs collide, represent their counts and sources, update
under a fresh target, and answer R10. A list of `B` distinguished values is
inside the setup cap but is no longer a one-bit or constant-channel
mechanism.

### Relation to R22

R22 deliberately planted one zero fiber and observed nine repeated norm-zero
keys carrying 34 of 160 toy sources. That small experiment used only eight
endpoint classes rather than the asymptotic `S=B^2` rectangle, so equation
(3) is not offered as a tight numerical prediction for it. R24 explains the
scaling boundary: one planted fiber cannot retain constant global mass once
the full endpoint deck grows quadratically.

## Controls and limits

1. The theorem counts distinct geometric endpoint/fifth classes. Heavy
   occurrence weights can enlarge a repeated stratum but do not create new
   independent factor-base columns; weighted rank must be charged separately.
2. `C` may depend prospectively on `psi`; no randomness assumption is used.
3. Pole and infinity values are included. Deleting them invalidates the
   complete-key interface.
4. A growing `r=Omega(B)` alphabet survives this mass theorem.
5. High additive energy or approximate translation stability of `Z_C` is not
   classified here; those mechanisms overlap the existing BSG, Plunnecke,
   and almost-periodic support lanes and need their source and target costs.
6. The theorem does not lower-bound arbitrary circuits or prove the absence
   of collisions outside `H_C`.
7. No fresh target, R10 index, relation density, independent rank,
   factor-base logs, or scalar-blind descent is supplied.
8. No unrestricted lower bound, Shoup improvement, or breakthrough is
   claimed.

## Scoped disposition

```text
constant distinguished-value alphabet: closed as global compressor
r=o(S/d) distinguished alphabet: closed as global compressor
r=Omega(S/d) prospective alphabet: survives mass gate only
complete-pair collision identity on that alphabet: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Analyze the first mass-admissible case `r=Theta(S/d)`: require a prospective
value set `C`, its complete preimage `Z_C`, and a fifth deck `Q` for which
`Theta(SL)` sources have both translated branches in `Z_C` and the resulting
complete keys meet the R23 collision budget `Omega(SLd)`. Prove all-chart
containment, exact source replay, and a fresh-target/R10 interface before any
experiment; reject at the first exposed scalar progression, adaptive list
fit, hidden common factor, repeated-column rank loss, or `SL` table.
