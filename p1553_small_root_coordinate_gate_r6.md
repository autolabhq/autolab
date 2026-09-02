# P1553 coordinate-small-root compiler gate R6

## Classification

- Owner: existing P1553/P1551/P1516/IDEA-049 frontier; no P1554.
- Evidence: coordinator theorem screen pending independent red team; no run.
- Status: `REVISE_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, rank theorem, factor-log solve,
  scalar-blind descent, Shoup-bound improvement, or ECDLP breakthrough.

R5 closed the standard elliptic-net product realization but preserved arbitrary
representation-sensitive aggregate compilers. R6 screens one coordinate-specific
candidate: use the two pair occurrence indexes as small integer variables, lift
complete elliptic addition to the integers, and apply multivariate small-root
recovery to the target-fresh `2+2+1` query.

The Coppersmith backend is not the proposed mechanism. The load-bearing operation
would be a public, operation-preserving bounded integral transducer with an exact
source inverse. That operation is already owned by IDEA-049. R6 asks whether the
P1553 pair preprocessing supplies the missing bounded variables. The first
pair-label realization is scoped negative. A stronger five-original-label
correction survives as an explicit modular predicate whose root locator is
unconstructed.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R4 target-label common-factor gate | `8cf15364c2da6830255216f3766a5d016b847d4bd012df92fd86c462ee6a9bc1` |
| staged P1553 R5 elliptic-net gate | `aa980ac35fb2a9e1af9a08310c5b62fc024a779d09d766311dd70cbcb9e1192d` |
| IDEA-049 hypothesis | `1a58addced8c2bb14590eb9a74252707409d01e7acb6fbab206dc12af974b1e0` |
| IDEA-049 integral-lift derivation | `cbbe0f83426067fefde0a1f440afafad3d856ad49540469219df78da5a716b4d` |
| P1551 selector gate | `5f1bd9c12ca700074c9cd327f6539bc880ec60b27431dc5f34e23b0a12f6c68f` |
| IDEA-165 pair-quotient theorem | `18cebc9c209c6ba0d705e43da7f921885e60d3436b201375e306e14f4ae0bdb2` |
| P1553 R3 indexing gate | `b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e` |
| focused-loop documentation | `8ba64525354917ab0cf9995955e87d11d86bbf42b147b28b36ae7a43b0c2347e` |
| focus selector | `ba6848118998482b997177dcef9f65672641a0a3f59a7d0d8ed7a1c409805ed5` |
| focus selector tests | `edea657a488d0ff4f09315e9a5d3095ad4085b7239769dc74802583b13a81b42` |

The last three inputs record the alphaXiv-inspired attention discipline now used
by the harness. They are workflow evidence, not evidence for an ECDLP claim.

## Frozen interface

Let `G=<P>` have prime order `N=p^(1+o(1))` and put `B=N^(1/5)`. The
target-independent pair trees retain

```text
D_12={(a_1,a_2,u=a_1+a_2)},      |D_12|=Theta(B^2),
D_34={(a_3,a_4,v=a_3+a_4)},      |D_34|=Theta(B^2).
```

A fresh target `R` and a fifth occurrence `a` satisfy

```text
u+v+a=R.
```

The exact query must work under every canonical dyadic restriction, including
all negative children used to recover one signed x-class-labelled source.
The caps, including replay, are

```text
preprocessing, advice, retained state    B^(9/4+o(1)),
online time and temporary workspace      B^(5/4+o(1)).
```

## Candidate transducer

The favorable integer variables are occurrence labels

```text
0<=i<Theta(B^2),
0<=k<Theta(B^2),
0<=j<Theta(B).
```

One would like polynomial or rational maps

```text
i -> u_i,       k -> v_k,       j -> a_j
```

followed by an integral lift of `u_i+v_k+a_j=R`. A small-root backend would
then recover `(i,k,j)` and the pair-tree backpointers would return all five
factor-base occurrences.

This is a real source interface, not merely a relation certificate. It is also
where the pair-label version fails in the named standard representations.

## Index-box gate

The natural source box has volume

```text
Theta(B^2)*Theta(B^2)*Theta(B)=Theta(B^5)=Theta(N).
```

Thus replacing point coordinates by occurrence labels does not by itself put
the unknown tuple in a sub-modulus box. Multivariate small-root methods require
an explicit polynomial system, root bounds, a monomial set, and a determinant
inequality. The equality `B^5=N` supplies none of those conditions and leaves no
entropy saving in the raw three-index box.

This is not a general multivariate Coppersmith lower bound. It is a failure of
the claimed argument that pair labels alone are sufficiently small.

## Selector gate

For an arbitrary pair deck, the maps `i -> (x(u_i),y(u_i))` and
`k -> (x(v_k),y(v_k))` are public tables of `Theta(B^2)` field elements. There
are three standard exact ways to expose them:

1. one-hot selection, with `Theta(B^2)` channels;
2. dense interpolation in the occurrence label, with degree
   `Theta(B^2)` and `Theta(B^2)` supplied coefficients; or
3. an endpoint-coefficient or lookup oracle.

The first two require `B^(2+o(1))` represented traffic for one fresh target,
before lattice construction or source replay. This exceeds the
`B^(5/4+o(1))` online cap, and `B` relation targets cost `B^(3+o(1))`.
The third is P1551's missing selector operation, not a construction.

A random-access table can verify a proposed index cheaply. It does not turn an
unknown table index into a polynomial small root. Calling table access from
inside the polynomial system hides the same source lookup.

This gate is scoped to arbitrary or dense pair-deck coordinate maps. It does
not close the stronger correction below.

## Five-label correction

Return to five coloured decks of public nonidentity x-classes `{P,-P}` in the
prime-order subgroup. Give each selected dyadic x-class subdeck local labels

```text
0,1,...,s_i-1,          s_i<=B,
```

Each class stores a canonical representative `P_C`, its x-coordinate, global
factor-base column, and occurrence backpointers. Interpolate the public
x-coordinate of the representative and bind the legal node domain:

```text
D_i(z)=product_(a=0)^(s_i-1)(z-a),
X_i(z) in F_p[z],                         degree < s_i.
```

Because the labels are distinct modulo `p`, these maps exist and their
coefficients can be built with standard product/remainder trees. Across one
canonical dyadic tree the sum of node degrees is `B log B`; all five trees fit
`B^(1+o(1))` work and state.

Define

```text
D_i(z_i)=0 mod p,                         i=1,...,5,
F_R(z_1,...,z_5)
  =S_6(X_1(z_1),...,X_5(z_5),x(R)) mod p.
```

The arity is fixed. Evaluating the five dense univariate maps and then the
fixed `S_6` circuit costs `B^(1+o(1))`, below the online cap. Since `S_6` has
degree `16` in each argument,

```text
deg_(z_i) F_R <= 16(s_i-1).
```

A zero means that there are signs
`epsilon_1,...,epsilon_5,epsilon_R` for which

```text
sum_i epsilon_i P_i(z_i) = epsilon_R R.
```

The sign is part of the returned relation coefficient, not a second deck
label. Every dyadic restriction is on x-classes and is therefore sign closed.
If `epsilon_R=-1`, negate every sign to normalize the target coefficient to
`+1`, in the convention

```text
R + sum_i sigma_i P_(C_i)=O.
```

For one returned label tuple, enumerate the `2^5` source-sign patterns and
verify the equation by complete projective addition.
Semaev's defining biconditional then makes `F_R=0` exact for signed relation
existence on the restricted x-class dictionaries; there is no wrong-sign
output family to enumerate.

The identity is excluded from factor decks. A zero target is handled by a
separate direct branch. For asymptotically odd prime `N`, the subgroup has no
nonidentity two-torsion. Repeated x-classes across colours and intermediate
identity, vertical, or tangent additions do not create false roots: `S_6` is
the denominator-free existential relation, and the returned branch is accepted
only after complete projective verification.

If one global x-class appears in several colours, aggregate its signed
occurrences into the same relation column:

```text
c_C=sum_(i:C_i=C) sigma_i.
```

These are exact rows, but cancellation may make them low support or useless
for rank. Pairwise class-disjoint coloured decks remain the clean density
control. For a known-log target `R=[r]P`, the row equation is

```text
r + sum_C c_C log_P(P_C)=0 mod N.
```

For blind descent with `R=Q+[t]P`, the identical source inverse gives

```text
x=-t-sum_C c_C log_P(P_C) mod N.
```

Known-log collection skips or resamples `R=O`; in blind descent that event
immediately yields `x=-t`. An API that insists on a relation query at `O`
must separately charge the corresponding `S_5` branch.

The label maps are replay compatible. At each dyadic restriction use the
already built local coordinate maps, ask for exact no-root or one root, verify
the signed source, and retain a positive child. Only the map-selection and
candidate-evaluation overhead over `O(log B)` levels is proved
`B^(1+o(1))`; root, exact-miss, output, and negative-child costs remain part of
the missing locator. Thus the earlier `B^2` pair-selector objection does not
reject this five-label selector.

This is a genuine refinement of the IDEA-049 label-selector subgate: the
hidden variables are public integer labels bounded by `B`, while the exact
signed modular predicate and source inverse are constructed before a source is
known. It is not yet a complete bounded integral transducer or an algorithm,
because constructing and evaluating the predicate is not the same as deciding
whether it has a bounded root and returning one.

## Coordinate-and-carry gate

Keeping point coordinates instead of occurrence labels restores variables in
`F_p`. The natural complete affine lift introduces slopes and integral
quotients. For a secant branch,

```text
lambda*(x_2-x_1)-(y_2-y_1)=k_s*p,
lambda^2-x_1-x_2-x_3=k_x*p,
lambda*(x_1-x_3)-y_1-y_3=k_y*p.
```

IDEA-049 proves that a complete target-independent affine chart over the whole
ordinary prime-field group needs `Omega(p)` centered slope height; the natural
carry variables also reach modulus scale. The pair preprocessing does not
alter those identities or prove that every required pair, target, dyadic
restriction, and negative replay branch lies in a smaller root region.

Eliminating slopes and quotients gives the denominator-free addition relation
and, after eliminating signs, the usual Semaev `S3` control. Combining those
relations with the factor-deck selectors returns the R3/R4 resultant and
membership-quotient routes. It does not construct `z_R` or the exact decision.

## Small-root boundary

Coppersmith and lattice reduction begin after the polynomial system and bounds
exist. The five-label correction supplies an exact signed modular predicate,
label bounds, and source inverse, but not the root theorem. Its initial box has
volume

```text
product_i s_i <= B^5=N.
```

There is therefore no raw box-volume saving over the modulus. At full dyadic
nodes, the rectangular degree box for the expanded `S_6` specialization has

```text
product_i (16(s_i-1)+1)=Theta(B^5)=Theta(N)
```

monomial positions. A full dense total-degree embedding has the same
`Theta(B^5)` order. Thus a full rectangular coefficient vector for `F_R`, or
a `k=1` shifted vector, is already an over-budget represented object before
LLL, determinant inequalities, or algebraic-independence assumptions.
Target-symbolic
precomputation needs `B^5` retained coefficient state; materializing the fresh
target specialization needs `B^5` online traffic. Higher powers and shifts only
enlarge this representation.

The predicate nevertheless has a short separated circuit: five univariate
maps followed by a fixed `S_6` circuit. The dense monomial count is not a lower
bound against selected sparse shifts, tensorized, circuit-native, or nonlattice
root locators. R6
supplies no theorem that a lattice can retain the separation, no compressed
normed integer basis, and no determinant inequality proving exact bounded-root
decision plus one witness inside `B^(5/4)` work and workspace. In particular,
the standard
rigorous univariate modular small-root theorem does not supply this
multivariate result; known multivariate modular extensions retain heuristic
root-recovery steps.

Consequently no admitted route supplies all of the following:

- a complete signed projective or saturated integral atlas;
- a tensorized basis that avoids expanded `B^5` coefficient traffic;
- bounds placing every positive and negative restriction root in the declared
  box;
- a monomial family and determinant inequality proving recovery;
- a zero-error inverse to exact occurrence labels; and
- construction, failed-lattice, replay, output, and bit costs inside the caps.

The pair-index route fails at its input transducer. The five-label correction
passes that input gate and fails at the theorem-level root locator. A toy
small root or a valid relation would not establish the determinant bound,
complete negative behavior, or campaign cost.

## Complete-path accounting

Conditional on an operation outside the screened representations, the earlier
favorable path remains

```text
pair-index setup               B^2,
B known-log relation targets   B^(9/4),
sparse factor-log algebra      B^2,
one masked descent             B^(5/4),
lambda                         0.45,
mu                             0.40.
```

The coordinate-small-root route does not supply the online root constructor,
constant verified relation density, `Theta(B)` independent rows, factor-log
completion, identical scalar-blind descent, or bit complexity. Its pair-label
dense-selector realization pays `B^2` per target. Its stronger five-label
  realization evaluates a supplied target predicate in `B^(1+o(1))`, but a
  full dense monomial-coefficient root lattice begins with `Theta(B^5)`
  ambient positions and no compressed exact decision-and-witness algorithm is
  proved.

## Deduplication

- IDEA-049 owns complete integral lifts, bounded roots, carry transducers, and
  Coppersmith source extraction.
- P1551 owns endpoint-coefficient and finite-domain selector access.
- P1516 owns the two pair indexes and the missing target-local router.
- P1553 owns Query2P1, `z_R`, restrictions, and complete source replay.
- IDEA-165 rules out treating a bounded-degree pair quotient as the missing
  target-local selector.

The pair-label proposal changes the downstream decoder without constructing a
new information channel. The five-label correction is an explicit IDEA-049
transducer refinement, not a new owner. It therefore receives no new idea ID,
P1554, contract, fixture, or run.

## Disposition

```text
REVISE_SCOPED_REDUCTION__PAIR_OCCURRENCE_INDEX_BOX_HAS_VOLUME_B5_EQ_N__PAIR_INDEX_TO_ENDPOINT_ACCESS_IS_SUPPLIED_B2_SELECTOR_OR_DENSE_DEGREE_B2_INTERPOLATION__FIVE_ORIGINAL_LABEL_CORRECTION_CONSTRUCTS_DEGREE_B_COORDINATE_MAPS_AND_COMPLETE_MODULAR_PREDICATE_EVALUATOR_IN_B1__DYADIC_LOCAL_MAPS_FIT_SETUP_AND_REPLAY_EVALUATION__FIVE_LABEL_ROOT_BOX_STILL_B5_EQ_N__STANDARD_EXPANDED_SMALL_ROOT_LATTICES_CAN_EXPOSE_B5_COEFFICIENTS__NO_TENSORIZED_DETERMINANT_INEQUALITY_OR_COMPLETE_ROOT_LOCATOR__AFFINE_COORDINATE_LIFT_RETAINS_FULL_FIELD_VARIABLES_OR_MODULUS_SIZE_CARRIES__SPECIAL_TENSOR_OR_NONLINEAR_TRANSDUCER_NOT_RULED_OUT__IDEA049_P1551_P1516_P1553_MERGE__RANK_LOGS_DESCENT_UNSUPPLIED__NO_P1554__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Derive or refute one tensorized multivariate small-root construction that acts
directly on the five-label Lagrange-coordinate `S_6` circuit,
proves a determinant/root-region theorem without `B^5` coefficient expansion,
returns one exact restricted signed source or an exact no-root certificate
inside the frozen total caps; otherwise preserve Query2P1 unchanged.

## Primary controls

- Coppersmith, *Finding a small root of a bivariate integer equation*,
  <https://doi.org/10.1007/3-540-68339-9_14>.
- Howgrave-Graham, *Finding small roots of univariate modular equations
  revisited*, <https://doi.org/10.1007/BFb0054862>.
- Jacobson, Koblitz, Silverman, Stein, and Teske, *Analysis of the Xedni
  calculus attack*, <https://pages.cpsc.ucalgary.ca/~jacobs/PDF/xedni.pdf>.
- Bosma and Lenstra, *Complete systems of two addition laws for elliptic
  curves*, <https://doi.org/10.1006/jnth.1995.1088>.
- Semaev, *Summation polynomials and the discrete logarithm problem*,
  <https://eprint.iacr.org/2004/031.pdf>.
- Coron, *Finding Small Roots of Bivariate Integer Polynomial Equations: a
  Direct Approach*, <https://www.iacr.org/archive/crypto2007/46220372/46220372.pdf>.
