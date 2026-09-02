# P1553 M6 sub-output implicit C5 character/pairing gate R120

## Claim boundary

R120 closes base-field algebraic-group characters, self-pairings, and
uncharged pairing transfers as realizations of the R119 implicit `C5`
interface. It constructs an exact reduced-Tate character on all eight R82
fixtures and proves that an independent nondegenerate pairing re-encodes the
five-sum predicate as a five-product predicate. It does not construct the
remaining multiplicative membership/source circuit and does not cover
pairing-unfriendly inputs. It supplies no relation rank, factor logs,
identical target descent, Pollard-rho improvement, Shoup improvement, or
ECDLP breakthrough.

Classification:

```text
BASE_FIELD_ALGEBRAIC_CHARACTERS_TRIVIAL__SELF_WEIL_PAIRING_TRIVIAL_ON_CYCLIC_G__INDEPENDENT_TORSION_PAIRING_GIVES_INJECTIVE_C5_PRODUCT_ENCODING_OVER_FPK__PAIRING_COST_SCALES_WITH_EMBEDDING_DEGREE_AND_R82_CONTROLS_ARE_SUPERSINGULAR_K2_EXCEPTIONS__FOUR_PRIME_ORDER_CURVES_REALIZE_MAXIMAL_K_QMINUS1_FINITE_CONTROLS__SMALL_K_PAIRING_ONLY_REENCODES_EXACT_MULTIPLICATIVE_FIVE_SUM__SUBOUTPUT_MULTIPLICATIVE_C5_CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Character boundary

An algebraic-group morphism

```text
chi: E -> G_m
```

is a nowhere-zero global regular function on the proper geometrically
connected curve `E`. Every such function is constant, and a constant group
morphism is the identity character. Thus no nonconstant base-field
algebraic character linearizes the group law.

The Weil pairing does not repair this using only the original cyclic
subgroup `G=<P>`:

```text
e_q(aP,bP) = e_q(P,P)^(ab) = 1
```

by bilinearity and alternation.

If an independent torsion point `R in E[q]` satisfies `e_q(P,R) != 1`, then

```text
chi_R(X) = e_q(X,R)
```

is a nontrivial homomorphism from the prime-order group `G` to `mu_q` and is
therefore injective. Consequently,

```text
X1+X2+X3+X4+X5 = T
```

holds if and only if

```text
chi_R(X1) chi_R(X2) chi_R(X3) chi_R(X4) chi_R(X5) = chi_R(T).   (1)
```

The same five occurrence backpointers replay in both representations.
Equation (1) is an exact re-encoding, not a membership algorithm.

## Extension charge

Let

```text
k = ord_q(p),
```

the least positive integer for which `q | p^k-1`. Then `mu_q` lies in
`F_(p^k)`. An independent `q`-torsion point may require this or a further
extension, so `k` is a lower-bound extension degree for the pairing values,
not an upper bound on every torsion construction.

Miller evaluation uses `O(log q)` operations in the extension field.
A coordinate representation of `F_(p^k)` has at least `k` base-field
coordinates, so one target character has base-field cost at least linear in
`k`, up to the field-arithmetic model. Writing `k=B^kappa`, the R118 outer
branch costs at least

```text
B^(5/4+kappa+o(1)).
```

The frozen `B^(5/4+o(1))` batch cap therefore permits this route only when
`k=B^(o(1))`. R120 neither assumes nor proves a uniformly subpolynomial
embedding degree for generic prime-field inputs.

## Exact pairing controls

Every R82 curve has

```text
p = 6q-1,
p mod q = -1,
k = 2.
```

They are supersingular `j=0` curves, so they are deliberately
pairing-friendly fixtures. Over `F_(p^2)=F_p[u]/(u^2+3)`, R120 uses

```text
zeta = (-1+u)/2,
psi(x,y) = (zeta x,y)
```

and evaluates the reduced Tate character against `psi(P)` with an explicit
Miller loop and final exponentiation.

For all four R82 families and both offsets:

- the distortion point is on-curve and has `q`-torsion;
- the generator image is a nontrivial `q`-th root of unity;
- every canonical five-source product equals the pairing image of its
  projective endpoint;
- every product map is injective on the finite canonical source set;
- every projective source replays exactly;
- the inherited absent target has an image outside the five-product support;
- no candidate scalar label is consumed.

These eight controls receive no asymptotic credit.

Four separate exact prime-order curve controls show the opposite finite
extreme:

| field | curve | order `q` | `ord_q(p)` |
| ---: | --- | ---: | ---: |
| `F_7` | `y^2=x^3+x+6` | 11 | 10 |
| `F_7` | `y^2=x^3+3` | 13 | 12 |
| `F_11` | `y^2=x^3+2x+4` | 17 | 16 |
| `F_13` | `y^2=x^3+2` | 19 | 18 |

Each has embedding degree `q-1`, the maximum possible. These finite examples
do not prove an asymptotic family; they prevent the `k=2` fixture behavior
from being silently generalized.

## Surviving small-k route

Even when `k=B^(o(1))`, the pairing only changes additive five-sum membership
into multiplicative five-product membership in `mu_q`.

- materializing the random-deck five-product support still costs
  `B^(15/4+o(1))`;
- the current bound `k=6` index still costs `B^(33/8+o(1))` state at
  polylogarithmic query;
- converting pairing images to scalar exponents invokes a finite-field
  discrete logarithm and receives no unit-cost oracle credit;
- no arithmetic-circuit, cell-probe, or RAM lower bound excludes a
  sub-output multiplicative index.

The preserved interface is therefore:

```text
input regime: k=B^(o(1)),
setup: B^(9/4+o(1)) state or less,
query: polylogarithmic field work,
input: pairing images of a scalar-blind C deck and arbitrary target,
output: exact empty or five occurrence backpointers.
```

Pairing-unfriendly inputs remain unsolved even if this restricted circuit
exists.

## Semantic deduplication

`ECDLP-IDEA-008 / P1542` already studies distortion lifts, pairing inversion,
and return geometry. R120 does not register a new idea ID. Its distinct
scope is forward pairing-character evaluation for the exact `C5`
membership/source interface. Pairing inversion and a map back from the torus
receive no new claim.

## Admission

Twelve of nineteen obligations pass:

- thirteen immutable source bindings;
- inherited R119 interface and nonclaim boundary;
- base-field algebraic-character triviality;
- self-Weil-pairing triviality on `G`;
- independent-torsion pairing-character theorem;
- embedding-degree and extension-field charge;
- eight exact R82 pairing controls;
- exact `k=2` and nontrivial-character checks;
- exact five-product endpoint identities;
- exact projective source and empty-image replay;
- four maximal-embedding-degree prime-order curve controls;
- semantic deduplication and preservation of the small-`k` residual.

The multiplicative membership circuit, source recovery, known-RHS relation
rank, factor logs, identical target descent, Shoup improvement, and
breakthrough remain open.

Disposition:

```text
ADMIT_EXACT_SMALL_K_PAIRING_CHARACTER_AND_C5_PRODUCT_REPLAY_ONLY__REJECT_BASE_FIELD_CHARACTER_SELF_PAIRING_UNCHARGED_EMBEDDING_DEGREE_EXPLICIT_PRODUCT_SUPPORT_CURRENT_K6_AND_UNIT_FIELD_DLP_ROUTES__PRESERVE_SMALL_K_SUBOUTPUT_MULTIPLICATIVE_C5_CIRCUIT_AND_PAIRING_UNFRIENDLY_INPUT_GAP__NO_LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

On the surviving `k=B^(o(1))` branch, construct or refute an exact sub-output
multiplicative five-product membership/source circuit over `mu_q` with
`B^(9/4+o(1))` state and polylogarithmic query. It must work directly on
field elements without taking their discrete logarithms, avoid the
`B^(15/4)` product support and `B^(33/8)` current index, provide exact empty
certification and five backpointers, and charge extension construction,
independent torsion, every pairing operation, all FFE branches, relation use,
and identical descent.

## Primary sources

- Andreas Enge, *Bilinear pairings on elliptic curves*,
  <https://arxiv.org/abs/1301.5520>.
- Victor Miller, *The Weil Pairing, and Its Efficient Calculation*,
  <https://crypto.stanford.edu/miller/>.
- Menezes, Okamoto, and Vanstone, *Reducing elliptic curve logarithms to
  logarithms in a finite field*, <https://doi.org/10.1145/103418.103434>.
