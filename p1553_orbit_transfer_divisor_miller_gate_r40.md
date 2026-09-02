# P1553 orbit-transfer divisor and Miller/net gate R40

## Classification

- Owner: existing P1553/IDEA-195 selected-path and implicit-target frontier;
  no new idea ID.
- Evidence: exact divisor theorem, exhaustive toy dynamic program, and scoped
  standard Miller/net/isogeny screen; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_POSITIVE_COBOUNDARY_NEGATIVE_STANDARD_MILLER_GATE`.
- Labels: `theorem-only`, `toy`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: an alternative rational locator with the same subgroup
  zero path can be made into a multiplicative translation coboundary. This is
  a positive algebraic construction. Its transfer divisor, however, has at
  least `B^2` explicit support and degree `Omega(B^4)` for a `B^2`-point path.
  Standard generalized Miller evaluation therefore costs `Omega(B^2)` work,
  and division-polynomial or isogeny-kernel compression has the wrong zero
  geometry in a prime-order subgroup. A new partial-orbit product identity is
  not ruled out. No Shoup-bound improvement or ECDLP breakthrough follows.

R39 showed that the natural R38 pencil locator is not a translation
coboundary because its zeros and poles occupy different `q`-orbits. R40 asks
the stronger question: can the poles be redesigned on the subgroup orbit so
that the same selected zero path is a coboundary, then evaluated by Miller or
elliptic-net identities without expanding its large divisor?

The first answer is yes. The second answer is no for the standard explicit
divisor, single Miller-function, division-polynomial, and proper-kernel norm
grammars. The exact remaining exception is a new compressed product over a
proper consecutive segment of a prime-order orbit.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R36 source/target-support gate | `103670908f33baee0915372434899a3a710861f501e19c91599f4221fe3f2aa5` |
| R38 target-walk support/work gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R39 implicit norm/coboundary gate | `2901305c2ac83fe51d2f23957389cb2dfb6d58b3c4bc0d2e439ad4ef3f6085f6` |
| R39 bundle hash list | `44d37773e2a4948efa5247111b8d160b8d4b2dd9fbd9df1b33c4cbb18e2c4b70` |
| R39 staging receipt | `51b9e6a732a1ffe9055a4d555517b7f5117341552d5ce8bf1be1ab37091922f8` |
| P1540 translated-pole annihilator gate | `d9a4040230022c24f7011932ef7cd9b5bcea51236a80c042bb498d2012428437` |
| P1540 independent audit | `8032be2d3a645ac64c046783191cc9c634715518eb18e4702acf66e077223d45` |

## Frozen orbit-transfer interface

Let `q` have odd prime order `N` and let

```text
R_i=R_0+[i]q,       i in Z/NZ.                   (1)
```

The start `R_0` may have unknown scalar. Its point coordinates and the public
shift `q` suffice to enumerate any requested interval. Fix

```text
I={0,1,...,L-1},    L=B^2,                       (2)
Z_I={R_i:i in I}.                                (3)
```

Seek a principal divisor

```text
div(h)=sum_i d_i [R_i]                           (4)
```

supported on the subgroup orbit such that

```text
F=tau_q(h)/h                                     (5)
```

has a simple zero at every point of `Z_I`, no other subgroup zero, and all
subgroup poles outside `Z_I`. Since

```text
ord_(R_i)(F)=d_(i+1)-d_i,                        (6)
```

the zero conditions become

```text
d_(i+1)-d_i=1,       i in I,
d_(i+1)-d_i<=0,      i notin I.                  (7)
```

On an elliptic curve, (4) is principal exactly when

```text
sum_i d_i=0,
sum_i d_i R_i=O.                                 (8)
```

The first condition removes `R_0` from the second. Since `q` has order `N`,
(8) reduces to

```text
sum_i i*d_i=0 mod N.                             (9)
```

No discrete logarithm of `R_0` is used.

## Theorem 1: a selected path admits an exact coboundary locator

Let a pole multiset `P` of cardinality `L` outside `I` define

```text
a_i=1                    for i in I,
a_i=-multiplicity_P(i)   for i notin I.          (10)
```

If the cumulative integers `d_(i+1)=d_i+a_i` can be shifted to satisfy (8),
then Abel's theorem gives a rational function `h` with divisor (4). Its ratio
(5) has divisor orders (10), so

```text
F(R)=0 on the subgroup iff R in Z_I.             (11)
```

Thus moving the reference poles onto the subgroup orbit repairs the exact
orbit-mass obstruction from R39. The alternative locator keeps the same zero
set used by the R36 endpoint marks. It need not lie in the degree-`B` pencil
subalgebra, and that compatibility is not credited for free.

The construction is scalar-blind in `R_0`: all divisor points are obtained by
public additions `R_0+[i]q`, and (8)-(9) are independent of its unknown scalar.

## Theorem 2: every simple interval transfer has large degree and support

Equation (7) makes

```text
d_0,d_1,...,d_L
```

strictly increasing integers. At most one can be zero, hence

```text
|support(div(h))|>=L.                            (12)
```

Their range is at least `L`. Shifting an integer ramp cannot make both its
positive and negative mass smaller than a constant multiple of `L^2`. Since a
principal divisor has equal positive and negative degree,

```text
deg(h)=Omega(L^2).                               (13)
```

At the selected-path scale,

```text
support(div(h))>=B^2,
deg(h)=Omega(B^4).                               (14)
```

The degree statement alone is not a circuit lower bound. The support statement
does close the standard generalized Miller grammar that takes the explicit
positive and negative divisor supports as input. A sequential generalized
Miller construction merges at least `L` distinct support points; a balanced
tree reduces depth but retains `Omega(L)=Omega(B^2)` line-quotient work per
evaluation. This is above R36's density-adjusted `B^(1/2)` fresh-target gate.

The divisor can be built once in setup, but evaluating its stored Miller
straight-line program at a fresh target still executes its `Omega(B^2)` gates.
Expanding the function instead exposes its `Omega(B^4)` divisor degree.

This is a lower bound only for explicit-support generalized Miller evaluation.
A new arithmetic identity whose input is the interval endpoints and length,
not its `L` support points, lies outside the theorem.

## Theorem 3: exact degree-21 optimum for the R38 toy

The deterministic script
`p1553_orbit_transfer_divisor_selfcheck_r40.py` reindexes the R38 selected
points by `q=[38]P`. Since `38^(-1)=19 mod 103`, their `q`-coordinates are

```text
(19,20,21,22,23,24,25,26,27),                   (15)
```

which rotate to `I={0,...,8}`.

For simple interval zeros, let the nine subgroup pole positions be a multiset
`p_1<=...<=p_9` outside `I`. The cumulative divisor has `d_0=-k` exactly when

```text
sum_j p_j=36+103k.                               (16)
```

The Picard condition (9) is equivalent to the second-moment congruence

```text
sum_j p_j(p_j+1)
  =sum_(i=0)^8 i(i+1) mod 103.                   (17)
```

The script scans positions `9,...,102` and dynamically retains the minimum
positive divisor degree for every tuple

```text
(poles_used, exact_first_moment, second_moment_mod_103). (18)
```

This exhausts every allowed pole multiset, including repetitions. The exact
minimum degrees for `k=1,...,8` are

```text
(47,29,24,21,21,24,29,47).                      (19)
```

The lexicographically first optimum has `k=4`, pole divisor

```text
4[R_9]+[R_15]+2[R_98]+[R_99]+[R_102],           (20)
```

and a principal transfer divisor of positive and negative degree 21. It has
19 distinct support points, maximum absolute order 5, total degree zero, and
Picard moment `-927=-9*103`. Its consecutive differences verify nine simple
zeros exactly on `I` and the pole multiplicities (20).

A generic sequential Miller construction of the positive and negative
degree-21 divisors uses at most

```text
(21-1)+(21-1)=40                                 (21)
```

line-quotient merges. This is a constructive positive toy control, not an
ECDLP algorithm. The dynamic-program optimum is scoped to simple selected
zeros, no extra subgroup zeros, and both `F` and `h` supported on the subgroup
orbit.

## Theorem 4: standard short elliptic identities have the wrong support

The familiar high-degree/short-program exceptions do not match (2):

1. A single Miller function has divisor
   `m*(Q)-([m]Q)-(m-1)*(O)`, supported on at most three points rather than a
   length-`L` interval.
2. The nonzero zeros of a division polynomial `psi_m` are `E[m]`. For
   `1<m<N`, the prime-order subgroup has trivial intersection with `E[m]`
   because `gcd(m,N)=1`; taking `m` divisible by `N` has degree at least
   quadratic in `N`.
3. An isogeny-kernel product compresses a subgroup norm. The prime-order group
   `G` has no proper nontrivial subgroup, so a proper interval of length
   `L<N` is not such a kernel. The only nontrivial subgroup norm uses all `N`
   points.
4. A standard rank-two net term locates one fixed linear relation among its
   input points. Multiplying the terms for `L` consecutive translates restores
   the `L` explicit leaves. The quartic net recurrence evaluates individual
   indexed terms; it is not an identity for their partial consecutive product.

These are support-shape screens for named standard identities, not a theorem
against every elliptic net or arithmetic circuit. An as-yet-unknown partial
orbit multiplication formula could be mechanism-new.

## Theorem 5: one fast coboundary evaluation would still leave an outer norm

Even granting a hypothetical polylogarithmic evaluation of (5), the fresh
target predicate remains

```text
A_V(R)=product_(v in V) F(R-v),       |V|=B^2.   (22)
```

Testing all factors costs `B^2`. If `V` is represented by translation paths,
updating (22) through explicit entering/leaving factors is exactly the R38
schedule and pays `B^3` expected work per scalar-blind hit after base-mask
density. Therefore the complete survivor needs two nested compressions:

```text
inner: evaluate the partial-orbit coboundary locator F,
outer: locate and return one vanishing translated factor of A_V. (23)
```

A field value `A_V(R)=0` is not enough. The operation must return `v`, the
selected endpoint `z=R-v`, and the R36 source mark, or support exact subset
restriction for logarithmic unranking. No current Miller/net identity supplies
the outer marked factor locator.

## Cost and ECDLP gate

The surviving object must satisfy all of:

```text
asymptotic degree-B low-boundary pencil family,
target-independent setup <=B^(9/4+o(1)),
average density-adjusted fresh query <B^(1/2-o(1)),
all initialization and negative masks charged,
marked (v,z) output and complete source replay,
R10 queried coefficients, rank, factor logs, blind descent. (24)
```

The repaired coboundary supplies none of the asymptotic pencil, outer locator,
R10, rank, logs, or descent obligations. It is an exact positive algebraic
control that sharpens the missing operation.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_POSITIVE_COBOUNDARY_NEGATIVE_STANDARD_MILLER_GATE
SUBGROUP_SUPPORTED_SELECTED_PATH_COBOUNDARY_EXISTS
TRANSFER_SUPPORT_AT_LEAST_B2
TRANSFER_DEGREE_OMEGA_B4
EXPLICIT_SUPPORT_GENERALIZED_MILLER_WORK_OMEGA_B2
TOY_SIMPLE_ZERO_OPTIMUM_DEGREE21_SUPPORT19
TOY_STANDARD_SEQUENTIAL_MILLER_UPPER_BOUND40
SINGLE_MILLER_DIVISION_POLYNOMIAL_AND_ISOGENY_KERNEL_SUPPORTS_MISMATCH
TWO_NESTED_PARTIAL_ORBIT_COMPRESSIONS_REQUIRED
NEW_ELLIPTIC_SHIFTED_FACTORIAL_AND_MARKED_OUTER_LOCATOR_OPEN
NO_DEGREE9_PENCIL_TARGET_R10_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: build or refute a partial-orbit product circuit with
input `(R_0,q,L)` and work sublinear in `L`, not an explicit divisor list.
Derive its addition law from elliptic sigma/net identities, test exact doubling
and concatenation on the R40 degree-21 toy, and require a second-level marked
factor locator for (22). Reject any construction that only evaluates one net
term, a full subgroup norm, or an unmarked product bit; bind a genuine survivor
to a degree-nine pencil lift and the R10 source interface.
