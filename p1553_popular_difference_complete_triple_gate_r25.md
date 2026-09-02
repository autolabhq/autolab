# P1553 popular-difference complete-triple gate R25

## Classification

- Owner: existing P1553/IDEA-195 complete-image lane, routed to the rejected
  IDEA-340 BSG public-chart owner for additive-energy extraction; no new idea
  ID.
- Evidence: exact deterministic reduction and cost/interface audit; no run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: the first R24-admissible support co-design necessarily
  creates a large popular-difference and additive-energy certificate, but that
  certificate neither constructs the support below the setup cap nor
  compresses the complete value triples. No Shoup-bound improvement or ECDLP
  breakthrough is supplied.

R24 forces any distinguished output alphabet carrying constant source mass to
have `r=Omega(S/d)`. At the balanced point

```text
S=Theta(B^2), L=Theta(B), d=Theta(B), r=Theta(B),
```

its full Kummer preimage can have `Theta(S)` points. R25 asks what dense
two-branch containment would mean before any collision or target claim.

The answer is exact. If a positive fraction of all endpoint/fifth sources has
both translated branches in the preimage set `Z`, then a positive fraction of
the fifth shifts are popular differences of `Z`, each with `Omega(S)`
representations. Consequently `Z` has additive energy `Omega(L*S^2)`.

This is a useful routing theorem, not the missing algorithm. A scalar
arithmetic progression passes the popular-difference control, but exposes the
hidden DLP coordinate. General BSG extraction is already IDEA-340 and requires
the dense relation graph or an unsupplied public chart. More importantly,
three `r`-valued complete coordinates still permit `r^3=Theta(B^3)` keys;
dense containment alone does not deliver the required `Theta(B^2)` image.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R23 random fifth-deck correlation gate | `de9a0a496fc6696ed6e976351476c56262484124c8c7d7cc2cdbddd16a6cc184` |
| R23 bundle hash list | `29bfdbeca9cd14a579f134ca2b5926e3fb4858f13455e1d8dfc066cbcb1008da` |
| R24 distinguished branch-value gate | `dd9b27fa5114e2121463612f5afaf09db12de30a0a0c4397f1dc37aafac1e99f` |
| R24 parent report | `ad05866c6a61717fbbc9e25ab41adb06d37bc836b7f211a82bb240bd6f57dcfb` |
| R24 bundle hash list | `070027b38243a269f55604ac6b054ee05fca49da5358d89b3de0da8418dcabbc` |
| R24 staging receipt | `60cc93025f82f69aeaa6286f3a30df75a4dc07f3d39b8dd4358ea0be18bbdd26` |

## Frozen signed model

Let `G=<P>` have odd prime order `N` different from the field
characteristic. Let `U,Q subset G` be signed representative supports with

```text
|U|=S, |Q|=L.
```

Constant sign-closure factors do not change any exponent. Let

```text
f=psi composed with x:E->P^1,
C subset P^1, |C|=r,
Z={R in G:f(R) in C}.
```

The evenness of `f` makes `Z` sign-closed. A degree-`d` Kummer map gives

```text
|Z|<=2*d*r+1,                                      (1)
```

where the harmless extra point allows the identity/infinity class. Work with
the complete projective values throughout.

For one fifth shift `q`, define the exact dense two-branch containment count

```text
T_q=#{A in U:A+q in Z and A-q in Z},
T=sum_(q in Q) T_q.                                (2)
```

Let

```text
r_(Z-Z)(t)=#{(x,y) in Z^2:x-y=t}
```

be the ordered difference multiplicity and

```text
E_plus(Z)=sum_t r_(Z-Z)(t)^2
```

its additive energy.

## Theorem 1: both-branch containment forces popular differences

For every source counted by `T_q`, put

```text
x=A+q,
y=A-q.
```

Then `x,y in Z`, `x-y=2q`, and the map `A->(x,y)` is injective. Hence

```text
T_q<=r_(Z-Z)(2q),
T<=sum_(q in Q) r_(Z-Z)(2q).                       (3)
```

Suppose for some fixed `0<alpha<=1` that

```text
T>=alpha*S*L.                                      (4)
```

Since every `T_q<=S`, at least

```text
alpha*L/(2-alpha)                                  (5)
```

distinct fifth shifts satisfy

```text
r_(Z-Z)(2q)>=alpha*S/2.                            (6)
```

Indeed, if `g` shifts pass (6), then

```text
T<=g*S+(L-g)*alpha*S/2,
```

and (4) rearranges to (5).

Therefore

```text
E_plus(Z)
 >= alpha^3*L*S^2/(4*(2-alpha))
 =Omega_alpha(L*S^2).                              (7)
```

At `S=B^2` and `L=B`, dense two-branch containment requires

```text
E_plus(Z)=Omega(B^5).                              (8)
```

If `|Z|=Theta(S)`, this is normalized energy

```text
E_plus(Z)=Omega(|Z|^3/(S/L)),
```

so the BSG parameter is at best `K=O(S/L)=O(B)`. This routes the structural
extraction question directly to IDEA-340. It does not provide a constant-
density small-doubling subset or a public scalar-free chart for free.

## Theorem 2: containment is not complete-key compression

If additionally `U subset Z`, every contained source has the three projective
values

```text
(f(A), unordered pair {f(A+q),f(A-q)}) in C x Sym^2(C).
```

The ambient tagged value alphabet has size at most

```text
r*r*(r+1)/2=Theta(r^3).                            (9)
```

For `r=Theta(B)`, this is `Theta(B^3)`, the same exponent as the source
rectangle. Equation (9) is only an upper bound, not a lower bound, but it
shows why containment supplies no `B^2` image theorem: a further exact
correlation among the three values is mandatory.

R23 makes that requirement quantitative. To obtain

```text
K_complete=O(S*L/d)=O(B^2),
```

the constructor must certify

```text
C_complete=Omega(S*L*d)=Omega(B^4)                 (10)
```

ordered nontrivial collisions of the full endpoint-labelled unordered branch
key. The popular-difference count (3) says only that both branch points lie in
`Z`; it neither equates their `f` values across two sources nor supplies the
collision mass (10).

## Positive scalar-progression control

Let, in the hidden scalar coordinate,

```text
Z={R+iP:0<=i<S}
```

without wrap, and take `Q` from `L` short shifts. Then
`r_(Z-Z)(2q)=Theta(S)` and (4)-(8) pass. This is the correct positive control
for the reduction.

It is not an ECDLP factor-base construction:

1. If the progression origin and step labels are public, the factor-base log
   differences are already known and all unknown logs share one scalar
   orientation.
2. If that orientation is hidden, constructing and indexing the progression
   in the scalar coordinate is the DLP chart problem.
3. The control does not provide a degree-`d` rational `psi` and a size-`r`
   value set `C` with `Z=f^(-1)(C)`.
4. It also does not prove the complete triple collision budget (10).

Thus a scalar progression validates the additive combinatorics while failing
the cryptanalytic interface. It must not be reported as evidence for a
generic-prime support co-design.

## Construction and literature boundary

Explicitly constructing all popular differences from `Z` by pair enumeration
costs `|Z|^2=Theta(B^4)`, above the setup cap. A passing rule must derive
`C,Z,Q` prospectively from the degree-`d` algebraic description without
accessing that graph.

The existing elliptic sum-product and bilinear exponential-sum estimates do
not close this exact regime. They do not prove that a degree-`B` rational
pullback set of size `B^2` lacks `B` popular prime-subgroup differences, and
their square-root field errors are too coarse for the `B=N^(1/5)` fifth deck.
This is a literature boundary, not an assertion that the desired set exists.

## Primary-source boundary

- Ahmadi and Shparlinski,
  [On the Sum-Product Problem on Elliptic Curves](https://arxiv.org/abs/0806.0640),
  gives a two-set sum-product dichotomy for elliptic x-coordinates. It does
  not bound the popular differences of one rational pullback set required by
  (6).
- Ahmadi and Shparlinski,
  [Exponential Sums over Points of Elliptic Curves](https://arxiv.org/abs/1302.4210),
  gives bilinear exponential-sum estimates. The resulting square-root field
  terms do not certify nonexistence at the frozen `B^2/B`, degree-`B` scale.
- Balog and Szemeredi,
  [A statistical theorem of set addition](https://doi.org/10.1007/BF01212974),
  and the quantitative BSG route bound in IDEA-340 convert energy to
  structure only after the relation graph and extraction losses are charged.
- Shoup,
  [Lower Bounds for Discrete Logarithms and Related Problems](https://www.shoup.net/papers/dlbounds1.pdf),
  remains the generic-group baseline. R25 does not extend or beat it.

## Deduplication

- R23 owns the exact complete collision budget and random-deck theorem.
- R24 owns the `2*L*d*r` one-distinguished-branch mass gate.
- IDEA-340 owns BSG extraction, public Freiman charts, and the hidden scalar
  orientation obstruction.
- IDEA-351 owns approximate almost-period quotients and exact singleton-support
  correction.
- R25 adds only the exact implication from dense complete two-branch
  containment to `L` popular differences, energy `Omega(L*S^2)`, and the
  separate complete-triple collision obligation.

## Controls and limits

1. Equation (7) is deterministic and exact up to the displayed constants.
2. High additive energy is necessary for the frozen containment mechanism,
   not sufficient for complete-key compression or ECDLP.
3. BSG extraction is not executed and no relation graph is supplied.
4. Generalized progressions or other high-energy sets outside the scalar
   interval control remain possible but still need a public DLP-free chart.
5. Existing incomplete character-sum bounds are not promoted beyond their
   proved parameter ranges.
6. No output-sensitive image constructor, fresh target, R10 index, relation
   density, independent rank, factor logs, or scalar-blind descent is supplied.
7. No unrestricted lower bound, Shoup improvement, or breakthrough is
   claimed.

## Scoped disposition

```text
r=Theta(B) mass-admissible alphabet: survives R24
dense both-branch containment: implies popular differences and B^5 energy
public extraction below setup cap: absent
complete B^4 collision budget: absent
B^2 complete image: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Prove or refute one algebraic popular-difference constructor: a separable
degree-`Theta(B)` `psi`, a prospective `C` of size `Theta(B)`, and a public
DLP-free rule producing `Z=f^(-1)(C)` of size `Theta(B^2)` plus `Theta(B)`
popular shifts, while the complete triple image has `O(B^2)` keys and exact
sources. Require construction below `B^(9/4)`, a fresh-target/R10 interface,
and rank/log/descent receipts; reject scalar orientation, pair enumeration,
adaptive fit, hidden common factors, or containment without (10).
