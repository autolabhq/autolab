# P1553 high-degree Kummer provenance gate R12

## Classification

- Owner: existing P1553/P1515/IDEA-001/IDEA-057/IDEA-195
  list-specific endpoint-router frontier; no P1554.
- Evidence: coordinator theorem derivation and independent theorem review;
  no run.
- Status: `REVISE_SCOPED_THEOREM`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, factor-log solve, blind descent,
  generic lower-bound violation, or ECDLP breakthrough.

R11 proved that a rational Kummer map of degree `d_psi<B/16` cannot merge
distinct source classes across a full `B`-point Cartesian branch deck, and
that a selective branch-oblivious merge needs `d_psi=Omega(B^(1/2))`.
R12 charges the next object that an implementation would actually retain:
the branch-state headers and source provenance for a pair-endpoint support
crossed with the fifth list.

The result is a state/construction accounting split, not an unrestricted lower
bound. A fifth-label
indexed, `psi`-respecting explicit table has a new
`d_psi=Omega(B^(3/4))` state floor on dense
pair support. Erasing the fifth label enlarges a branch fiber to
`O(d_psi^2)` and leaves an implicit algebraic image/inversion route open.
Storing explicit preimages instead restores the full `S*B` incidence payload.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R11 Cartesian Kummer rigidity gate | `6c79b486bfa4cfd14674033a438db3d91ddf7bbe4d2c4aaf309f1a0706f0df4e` |
| P1553 R11 parent report | `5f65da4f50e301067258f14f4b8e6dec91a0f096ac04e2418e4957bd649604b2` |
| P1553 R11 independent red team | `dec795cf4abcb14fd33b4e264dbe13eca168f504fc5557dfb6421bd04d99c14b` |
| P1553 R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| P1551 finite-domain selector circuit gate | `5f1bd9c12ca700074c9cd327f6539bc880ec60b27431dc5f34e23b0a12f6c68f` |
| IDEA-001 exact spectral rank/density gate | `e572713a3910ef6a3e31ac360123aa8b5135c75d4210cbfb579e6831e9746fca` |
| IDEA-057 prime-order composable-bucket theorem | `524a59c1728bcbea804ac4be42ace5a965b68a6332e85d941829b89e04fc4225` |
| P1515 R1-R11 independent audit | `7e7609716f87b1b4df5ffc77406a912ad0303cc309ec1b84be42ebcc0d09539e` |

The process focus is also bound to the existing P1436 autoresearch enrichment
contract and the alphaXiv post
`https://x.com/askalphaxiv/status/2076737985559822734`: use a few decisive
tests, resolve nonblocking ambiguity locally, and defer peripheral branches.
That is workflow guidance only, not mathematical evidence.

## Frozen occurrence interface

Let `G=<P>` have odd prime order `N=p^(1+o(1))` and put

```text
B=N^(1/5).
```

The first two color decks have `B` occurrence labels each. Their ordered pair
occurrences form

```text
O_12=I_1 x I_2,        |O_12|=B^2.
```

Map every pair occurrence to its Kummer endpoint class

```text
u=x(P_(1,a)+P_(2,b)).
```

Let `U` be the set of distinct attained endpoint classes and write

```text
S=|U|,                 B <= S <= B^2
```

when the stated lower bound is separately certified. R12 never substitutes
the `B^2` occurrence count for `S`. A source table may retain all `B^2` pair
occurrence backpointers, which fits the setup/state cap. Cross-color copies
remain distinct until a returned row is completely verified and its columns
are aggregated.

If `kappa` is the maximum number of pair occurrences in one endpoint class,
then `S>=B^2/kappa`. Hence every support bound below also has the occurrence
form obtained by replacing `S` with `B^2/kappa`. No dense-support conclusion
is drawn unless `S=Theta(B^2)` or `kappa=B^(o(1))` is separately certified.

Let the fifth deck `I_5` contain `B` distinct Kummer classes with occurrence
labels. For a representative `A` of `u`, a representative `Q` of `q`, and a
nonconstant rational map

```text
psi:P^1->P^1,          deg(psi)=d,
f=psi composed with x, deg(f)=2d,
```

define the complete projective unordered branch divisor

```text
D_(u,q)=[f(A+Q)]+[f(A-Q)].
```

It is independent of the signs of `A` and `Q`. Multiplicity, poles, identity,
infinity, tangent, and vertical cases are retained. A trace-only label is not
an admitted state.

The full setup/state cap is `B^(9/4+o(1))`. A fresh target, every dyadic child
query, and target workspace are capped by `B^(5/4+o(1))`.

## Theorem 1: fifth-label indexed header floor

Freeze an explicit header grammar whose key retains the fifth occurrence
label, the endpoint's `psi` label, and the complete divisor:

```text
key(u,q)=(q,psi(u),D_(u,q)).
```

For fixed `q` and fixed key, all endpoint coordinates lie in one fiber of
`psi:P^1->P^1`. A degree-`d` map has at most `d` distinct geometric points in
every projective fiber, including the fiber over infinity and regardless of
ramification or inseparability. The divisor component can only split that
fiber. Therefore
the number `K_indexed` of nonempty headers obeys

```text
K_indexed >= B*ceil(S/d) >= S*B/d.
```

More generally, if `S=B^(sigma+o(1))` and `d=B^(delta+o(1))`, the header
exponent is at least

```text
sigma+1-delta.
```

Under dense pair support `S=Theta(B^2)`, fitting only the headers beneath
`B^(9/4+o(1))` requires

```text
d=Omega(B^(3/4-o(1))).
```

This is a header floor, not a construction lower bound. A bucket payload that
lists every admitted endpoint for every fifth label contains exactly `S*B`
endpoint/fifth incidences before compression of that payload. Alternatively,
an implementation may retain only headers and invert the defining degree-`d`
equations when a header is selected. R12 does not prove that such an inversion
answers the missing target query.

## Theorem 2: fifth-label erasure and provenance accounting

If the key drops `q` but retains `(psi(u),D_(u,q))`, fix

```text
D=[alpha]+[beta].
```

For signed representatives put

```text
X=A+Q,       Y=A-Q.
```

The transformation is bijective on the odd-order subgroup. For either
ordering of `alpha,beta`, there are at most `(2d)^2` choices for `(X,Y)`.
Thus a fifth-label-erased complete-key fiber contains at most

```text
8d^2
```

signed `(A,Q)` pairs, and no more Kummer-class pairs. Consequently

```text
K_erased >= S*B/(8d^2).
```

For dense support this state count alone would require only
`d=Omega(B^(3/8-o(1)))`, which is weaker than R11's independent
`d=Omega(B^(1/2))` floor for retaining a positive fraction of genuinely
compressed branch incidences.

An exact source-faithful implementation must therefore choose one of four
charged routes:

1. Retain `q` in each header. Then the `S*B/d` indexed-header floor applies.
2. Erase `q` but store explicit `(u,q)` preimages. The total uncompressed
   provenance payload is `S*B`, equal to `B^3` at dense support.
3. Evaluate all `S*B` incidences once, but retain only an exact count and one
   witness per nonempty header. Persistent provenance can then scale with the
   header count, while the construction pass still costs `S*B` work.
4. Erase `q` and recover a preimage algebraically or through an implicit
   circuit. The `8d^2` fiber bound applies, but construction of the image,
   membership in the frozen source decks, target querying, dyadic restriction,
   and one-source inversion remain unproved.

The fourth route is the sole R12 survivor under both local caps. Treating its
algebraic fiber size as an algorithm would confuse bounded output multiplicity
with a constructor. The third route is source-faithful but fails the setup-work
cap at dense support; R12 does not misclassify that work as persistent state.

## Literal-construction charge

Evaluating `D_(u,q)` independently on every attained endpoint and fifth entry
touches `S*B` input incidences. At dense support this is `B^3`, above both the
setup/state and fresh-target caps. Sorting, hashing, or deduplicating after
those evaluations does not refund their construction cost.

R12 does not prove that every constructor must enumerate those incidences.
A valid escape is an oracle-free algebraic image algorithm that constructs
the nonempty divisor headers or an equivalent query index directly from the
two unary factor decks, retains exact integer multiplicities and source
membership, and supports all dyadic restrictions. No such algorithm is
supplied here.

## Isogeny and ECFFT control

The strongest standard source of exact cheap branch transport does close on
the prime-order campaign subgroup. Let `phi:E->E'` be a separable isogeny of
degree `d<N`. Then

```text
ker(phi) intersect G = {O}.
```

Indeed, the intersection is a subgroup of the prime-order group `G`; if it
were nontrivial it would contain all `N` points of `G`, while `ker(phi)` has
only `d<N` geometric points. Thus `phi` is injective on `G`. Its induced
Kummer label is also injective: equality of two target x-coordinates implies
`phi(A')=+-phi(A)`, hence `A'=+-A` after applying the kernel-intersection
identity. Every ECFFT, Lattes, or quotient-isogeny map of degree through
`O(B)` therefore supplies exact branch composition but no source compression
on the ordinary prime-order subgroup.

There is a sharp composite-torsion positive control. On an auxiliary curve
with independent order-`B` points `P,Q`, take

```text
F_1=<P>, F_2=<Q>,
H=<(B/d)P>,             d=B^(3/4),
phi:E->E/H.
```

For compatible power parameters, `H` has order `d`. The pair grid has
`Theta(B^2)` Kummer endpoint classes and `Theta(B^2/d)=Theta(B^(5/4))`
quotient Kummer labels; the hidden constants account for sign identification.
Crossing them with a `B`-point fifth deck produces `Theta(B^(9/4))` indexed
label/fifth states, and

```text
phi(A+-R)=phi(A)+-phi(R)
```

gives exact constant-branch transport. A kernel-coset representative and the
pair occurrence table define the inversion search, but one representative is
not by itself an exact source. The control must enumerate the `d=B^(3/4)`
kernel-coset candidates, test membership in the frozen pair support, follow an
occurrence backpointer, and verify the original branch. This fits the
`B^(5/4)` online cap. Its charged source index contains `Theta(B^2)` endpoint
membership/backpointer entries, `Theta(B^(5/4))` quotient representatives,
and `Theta(B^(3/4))` kernel data, all within setup/state. The control should
pass the R12 transport
and provenance exponents, then fail the mandatory prime-order gate because
all compression is the nontrivial intersection `H` with the source subgroup.
It is a harness control, not evidence for the target ECDLP family.

## Degree-linear structured collapse control

The `d=Theta(B)` boundary genuinely permits dramatic finite-deck collapse.
Choose five disjoint scalar intervals of known multiples of `P` so that:

```text
F_i={aP:a in A_i},       1<=i<=5,
```

all fivefold endpoints, pair endpoints, and their plus/minus fifth-list
branches are nonidentity,
and the union `Omega` of their distinct affine x-coordinates has size `O(B)`.
For sufficiently large `N/B`, define

```text
psi_Omega(Z)=product_(omega in Omega)(Z-omega).
```

This degree-`O(B)` polynomial maps every frozen endpoint and branch coordinate
to zero, so every complete branch divisor is `2[0]`. Its coefficients can be
constructed from the public grid within the setup cap. This is a positive
compression control and prevents any stronger finite-list no-collapse claim.

It is not a useful ECDLP route. The scalar logs of these factor points are
already known, the fivefold attainable endpoint support is only `O(B)`, and a
uniform known-log relation target succeeds with probability at most `O(B/N)`.
Even granting one independent row per success, collecting `B` rows takes
`Omega(N)` target attempts. Adding endpoint or scalar corrections restores
source discrimination and removes the claimed collapse.

The control separates two requirements that interpolation alone conflates:
finite-grid compression and cryptographically useful target coverage.

## Literature screen

Bukh and Tsimerman,
[Sum-product estimates for rational functions](https://arxiv.org/abs/1002.2554),
prove expansion results and characterize exceptional fixed-degree rational
forms under hypotheses on finite-field sets. Their estimates do not provide a
source-faithful list-specific map of degree growing from `B^(1/2)` to `B`, and
they do not construct or invert the branch-state image required here.

Ahmadi and Shparlinski,
[On the Sum-Product Problem on Elliptic Curves](https://arxiv.org/abs/0806.0640),
show that one of two coordinate sets associated with additive and
multiplicative scalar operations is large. That alternative is not an image
bound for `psi(x(A+B))` on an arbitrary list-specific factor base and carries
no occurrence-source replay.

The earlier Kummer and ECFFT comparisons remain Chalcraft and Fryers,
[Kummer structures](https://arxiv.org/abs/0806.0409), and Ben-Sasson et al.,
[Elliptic Curve Fast Fourier Transform Part I](https://arxiv.org/abs/2107.08473).
ECFFT obtains fast polynomial operations from smooth elliptic-curve subgroups
and isogeny-based divide and conquer. Its exact quotient mechanism is covered
by the prime-order injectivity gate above; it does not supply the growing-degree
non-isogeny implicit image constructor left by R12.

Parameter counting from the `O(d)` coefficients of a rational function is not
used as a theorem. Finite interpolation conditions can be dependent, and the
structured collapse above is an explicit warning against that shortcut.

## Rank-two and query controls

R10's exact rank-two control reduces the all-nonzero count to sparse
multiplicative convolution of pair histograms. Any general transporter grammar
must express that control with exact integer multiplicities, zero strata,
queried-rectangle identity, and one occurrence backpointer. The header floors
above neither implement that convolution nor prove it impossible.

Likewise, a compact target-independent image does not yet answer

```text
sum_(a_3,a_4,a_5) h_I(V_R(a_3,a_4,a_5))
```

for a fresh `R`. A surviving construction must expose the target action on the
image, support every adaptive child after a zero or positive count, and return
one verified signed occurrence source. A supplied image, target-trained map,
or precomputed target orbit is charged as advice.

## Complete campaign accounting

For a hypothetical backend with setup `B^s`, state `B^s_m`, fresh-target work
`B^kappa`, fresh-target workspace `B^w`, and reciprocal accepted-density loss
`B^d_loss`, the favorable R10 bookkeeping remains

```text
lambda=max(s,1+d_loss+kappa,2)/5,
mu=max(s_m,w,2)/5.
```

Promotion still requires exact signed relation density, independent row rank,
repeated-column aggregation, factor-log completion, identical scalar-blind
target descent, field-to-bit conversion, and complete projective verification.
The structured collapse control has `d_loss>=4-o(1)` and therefore fails before
those later stages.

## Route disposition

| Route | R12 decision |
|---|---|
| `d<B/16` full Cartesian merger | closed by R11 |
| General useful selective merger with `d=o(B^(1/2))` | closed by R11 incidence bound |
| Fifth-label indexed `psi`-respecting explicit headers, dense pair support | state cap forces `d=Omega(B^(3/4))` |
| Indexed headers plus explicit endpoint payloads | payload has `S*B` incidences |
| Fifth-label erased divisor headers | fiber at most `8d^2`; state bound alone does not close route |
| Fifth-label erased explicit provenance | `S*B`, hence `B^3` at dense support |
| Full incidence pass with one retained witness per header | cap-sized state may remain, but dense construction work is `S*B=B^3` |
| Literal evaluation of every endpoint/fifth incidence | `S*B` construction work |
| Degree-`d<N` separable isogeny or ECFFT quotient on `G` | exact cheap branching but injective on prime-order Kummer classes |
| Composite-torsion quotient-isogeny control at `d=B^(3/4)` | hits `B^(9/4)` indexed state with exact branching; fails prime-order gate |
| Degree-linear arithmetic-progression collapse | exact positive control; small support and density failure |
| Fixed-degree rational-function expansion theorems | hypotheses do not cover growing list-specific degree and source replay |
| Implicit algebraic image plus exact inversion | open in `d=Omega(B^(1/2))` through `O(B)` |
| Arbitrary nonlinear arithmetic circuits | open |

## Deduplication

- P1553/P1551 own exact endpoint counts, Query2P1, dyadic replay, signs, and
  occurrence-labelled source output.
- IDEA-001 owns endpoint-support rank and the one-witness density tradeoff.
- IDEA-057/P1523 own global Kummer labels, prime-order noncollapse, and ECFFT
  auxiliary-map screens.
- P1515 owns nonlinear implicit target batches and list-specific routers.
- IDEA-195 owns the non-Cartesian `S_3` intertwiner residual.

R12 adds a scoped explicit-state/provenance theorem under those owners. It
creates no idea ID, experiment, contract, or research-status promotion.

## Exactly one next action

Derive or refute one oracle-free implicit algebraic image-and-inversion
algorithm for the fifth-label-erased complete map
`(u,q)->(psi(u),[f(A+Q)]+[f(A-Q)])` on a frozen dense pair support, with
`d=Omega(B^(1/2))` through `O(B)`. It must construct the image without
enumerating `S*B` incidences, express the R10 rank-two sparse-convolution
control, answer every full-box and adaptive-child fresh-target query within
`B^(5/4+o(1))` work/workspace, retain total setup/state within
`B^(9/4+o(1))`, invert one accepted state to exact occurrence labels, and
charge density, independent rank, factor logs, and identical blind descent.
A negative closes only this frozen divisor-image grammar; a positive image or
count remains model-bound until the complete ECDLP path is proved.
