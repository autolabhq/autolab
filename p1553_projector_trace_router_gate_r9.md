# P1553 projector-trace router gate R9

## Classification

- Owner: existing P1553/P1534/P1536/IDEA-089/IDEA-198/IDEA-250/IDEA-266
  exact-query and source-router frontier; no P1554.
- Evidence: coordinator theorem screen and independent red team; no run.
- Status: `REVISE_SCOPED_THEOREM`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, elliptic trace-contraction
  algorithm, factor-log solve, blind descent, Shoup-bound improvement, or
  ECDLP breakthrough.

R8 requested a branch-complete representation of the Fermat projector or an
equivalent unit-or-zero-divisor interface. R9 proves that one scalar is already
branch complete for the frozen R6 decks: the algebra trace of the restricted
projector is the exact integer root count, not merely the count modulo `p`.
Dyadic self-reduction then returns one signed occurrence-labelled source.

This sharpens the semantic interface but does not construct the trace inside
the caps. It also removes projector flattening rank as a useful universal
obstruction: the projector is low rank precisely when the useful target fiber
is sparse, but its source-labelled low-rank factors are the answer being
sought.

## Bound inputs

| Input | SHA-256 |
|---|---|
| P1553 R8 integer-valued quotient gate | `78c856187bb43adcd97d0f02f2259c7299874ab93a03954dacb4dd1b8b007ed9` |
| P1553 R8 parent report | `110f0893b7a32d84114ef85c4d8c2baad00c05b86af21d51de1b64162bc7369a` |
| P1553 R8 independent red team | `c98229e8a179c4f70d9f38aca91e8243e070ec79ad3150dfc40c5618cf46c983` |
| P1534 quotient-kernel independent audit | `6a2c96f41552f91ab6d6ddc4801d6e4f958cf5845f6f81676de7f4db89653c53` |
| P1536 Frobenius-projector/norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |
| IDEA-089 residue-idempotent splitter hypothesis | `084ffa5cd92f6f50e96d6fd1f63ca1abda975c87c81c7574e99db9e2bf8c80ce` |
| IDEA-198 carry-state source-unranking hypothesis | `834211bd8e26e3d421df0749653a1bc55b786aab56a698edb9cd71013a0c058a` |
| IDEA-250 Frobenius-splitting source-strata hypothesis | `9dc90c1f073c2af434fbc4b8d3074f688ea75c3393283d2f66a6fe65ebfe487d` |
| IDEA-266 dynamic-evaluation source-tree hypothesis | `a9529076339b09b881d4504de45c132219352d4e0edc282cc0d2d955577ea1b1` |

## Frozen interface

Let `N=p^(1+o(1))`, `B=N^(1/5)`, and let `I_i` be any canonical dyadic
subdeck of the R6 nonidentity x-class occurrence labels. Put

```text
Gamma_I = I_1 x ... x I_5,
A_I = F_p^(Gamma_I),
Z_(R,I) = {a in Gamma_I : F_R(a)=0},
z_(R,I) = |Z_(R,I)|.
```

The query must give an exact empty answer or one source, preserve x-class
sign closure and occurrence backpointers, enumerate the constant sign family,
verify with complete projective addition, and aggregate repeated global
columns only after verification. The caps remain

```text
target-independent setup and state    B^(9/4+o(1)),
fresh-target time and workspace        B^(5/4+o(1)).
```

## Exact root-count bound

Assume each colored deck contains each global x-class at most once; repeated
global x-classes across distinct colors remain separate occurrence labels.
After target-sign normalization, fixing four labels and their `2^4` sign
choices determines at most one legal fifth x-class. Therefore

```text
z_(R,I) <= 16 min_j product_(i!=j) |I_i|.
```

The frozen `32 B^4` bound is conservative. If within-color multiplicities are
admitted, the bound must be multiplied by their maximum multiplicity and the
threshold rechecked. Infinity or an illegal fifth point contributes no legal
source. Since `B=p^(1/5+o(1))`, the campaign must and asymptotically can check

```text
32 B^4 < p.
```

Below that explicit finite-size threshold, the field trace would determine
the count only modulo `p`; R9 makes no exact-integer claim there.

## Projector-trace theorem

In the reduced split quotient define

```text
G_(R,I)   = F_R^(p-2),
chi_(R,I) = 1-F_R^(p-1).
```

Coordinatewise Fermat gives

```text
F_R G_(R,I) = 1-chi_(R,I),
F_R chi_(R,I) = 0,
chi_(R,I)^2 = chi_(R,I).
```

Multiplication by `chi_(R,I)` is diagonal in CRT coordinates, with one on a
root and zero elsewhere. Hence the algebra trace is

```text
C_(R,I) = Tr_(A_I/F_p)(chi_(R,I)) = z_(R,I) mod p.
```

The root-count bound makes the canonical representative of `C_(R,I)` the
exact integer `z_(R,I)`. Therefore

```text
C_(R,I)=0  iff  the restricted fiber is empty.
```

The trace counts occurrence-labelled x-tuples satisfying the Semaev
biconditional, not sign patterns. Every counted tuple has at least one valid
sign pattern, recovered by constant enumeration and complete verification at
the singleton leaf.

Query the full count once. At each split query one child; retain it if positive,
otherwise retain its complement, whose positivity follows from disjoint count
additivity. This uses at most

```text
1 + sum_i ceil(log_2 |I_i|) <= 1 + 5 ceil(log_2 B)
```

count calls, after which one occurrence tuple remains. Enumerate at most `2^5`
source signs, apply the frozen exceptional handling, verify the group equation,
and then aggregate repeated global columns. An initial zero count is an exact
miss.

Thus exact restricted trace counting implies exact Query2P1 with one source up
to `B^(o(1))` replay overhead. The converse is not proved: an empty-or-source
router does not recover exact multiplicity. The projector, quotient kernel,
zero-divisor split, norm-zero test, and restricted source tree remain semantic
presentations of the same existence-and-source frontier. R9 introduces no new
idea owner.

## Conditional separated backend

Let a supplied target-fresh projector have an exact CP representation

```text
chi_R = sum_(alpha=1)^A w_alpha(R)
          tensor_(i=1)^5 v_(alpha,i,R).
```

The indicator of a dyadic rectangle is a rank-one CRT mask. Precompute all
dyadic interval sums of every supplied factor. Then every restricted count is

```text
C_(R,I) = sum_alpha w_alpha(R)
             product_i sum_(a in I_i) v_(alpha,i,R)(a).
```

Writing all target-fresh factors and their interval sums costs
`A B^(1+o(1))`; each self-reduction path then costs `A B^(o(1))`. The direct
fresh-target cap therefore requires

```text
A <= B^(1/4+o(1))
```

for this supplied-factor implementation. If the factors are target-independent
and only the weights specialize, their construction belongs to setup and must
still satisfy the state cap. In that case setup/state costs
`A B^(1+o(1))` and fresh-target replay costs `A B^(o(1))`; both caps allow
`A<=B^(5/4+o(1))`. This remains conditional on supplying that target-symbolic
factorization. R7's separate naive CP-Gram lattice route retains its stronger
`A<=B^(1/8+o(1))` condition. Neither bound applies to an unrelated exact
contraction that proves the direct caps.

This is a complete conditional router, but constructing its source-free
factors is the original problem.

## Frobenius and rank corrections

Because `A_I` is a product of copies of `F_p`, absolute Frobenius is the
identity:

```text
h^p=h  for every h in A_I.
```

The exponent `p-1` is a Fermat zero test, not a nontrivial Frobenius orbit
recurrence. Writing `F_R^(p-1)=F_R G_R` moves the missing work into pointwise
inversion and exact zero-divisor handling. IDEA-250 already owns the fact that
Frobenius is point-blind on the split finite-etale source locus.

In the CRT delta basis,

```text
chi_(R,I) = sum_(a in Z_(R,I))
              e_(a_1) tensor ... tensor e_(a_5).
```

Consequently

```text
rank_CP(chi_(R,I)) <= z_(R,I),
rank_CP(F_R^(p-1)) <= z_(R,I)+1,
```

and every flattening has the same upper bound. Empty and singleton fibers have
projector rank zero and one. A universal `A>=B` projector-rank obstruction is
therefore false on the sparse fibers most useful to the campaign. A
source-faithful delta decomposition already contains the source coordinates;
low output rank is post-support structure, not a constructor.

Carry rank is also not a modular invariant. For

```text
K_1=(tilde(F)-res_p(F))/p,
```

replacing `tilde(F)` by `tilde(F)+pL` leaves the canonical modulo-`p`
remainder fixed and gives

```text
K'_1=K_1+L.
```

Higher-power centered remainders and carries are also lift- and
schedule-specific; the displayed affine formula is asserted only for `k=1`.
Carry rank can reject only a frozen lift and arithmetic schedule. It cannot
reject an unrestricted finite-field trace circuit. R8's Shi control remains
valid for its named least-nonnegative lift, but it cannot become an elliptic
projector lower bound.

## Constructor audit

Let `r_R` be the number of nonzero separated summands after target
specialization. The raw formal multinomial expansion of `F_R^(p-1)` has

```text
binomial(p+r_R-2,r_R-1)
```

composition-indexed summands, and every multinomial scalar is nonzero modulo
`p`. For fixed `r_R>=2`, this is

```text
p^(r_R-1+o(1)) = B^(5(r_R-1)+o(1)).
```

Products may coincide before quotient reduction and may cancel afterward, so
this proves only that literal uncompressed expansion is over cap, not a rank
or circuit lower bound. The `r_R=1` boundary is excluded from that conclusion.
The addition chain has only `O(log p)` arithmetic gates, but no exact cap-sized
recompression or trace contraction follows.

The named standard exact routes remain:

| Route | First charged source-faithful object | Disposition |
|---|---:|---|
| Explicit CRT projector/inverse/trace | `B^5` values | exact, over cap |
| Generic quotient power projection | `B^(5+o(1))` quotient work | exact, over cap |
| Norm or first jet | `B^5` product/quotient traffic | exact, over cap |
| Balanced pair versus triple | `B^2` pair dictionary plus `B^3` target-fresh triple search | exact standard route, over cap |
| Sequential inverse/resultant | up to `B^4` partial outer coefficients, then full quotient data | empty branch or over cap |
| Supplied CP projector | `A B` target-fresh factors | exact conditional router |

For the balanced route, write

```text
F_R(a_1,...,a_5)
  = sum_(rho=1)^r U_rho(a_1,a_2) V_(rho,R)(a_3,a_4,a_5).
```

The separated formula for `V` itself has only factor storage. The standard
over-cap step is materializing or searching its `B^3` source-labelled triple
evaluations to perform the nonlinear zero match against the `B^2` pair
dictionary. No theorem here says every implicit orthogonality, trace, or
batch-contraction algorithm must pay that cost.

## Campaign consequence

A hypothetical passing trace router with setup/state `B^s,B^s_m` and
query/workspace `B^kappa,B^w` would have the conditional exponents

```text
lambda = max(s,1+kappa,2)/5,
mu     = max(s_m,w,2)/5.
```

The dyadic replay is only `B^(o(1))`. At the direct caps this gives
`lambda,mu<=0.45`; `mu=0.40` requires persistent state at most `B^2`.

The standard `B^3` triple search gives `B^4=N^0.8` relation collection and
`B^3=N^0.6` descent. No current route proves constant verified relation
density, `Theta(B)` independent signed rows, factor-log completion, or the
identical blind-masked target query.

## Disposition

```text
REVISE_SCOPED_THEOREM__ACTUAL_RESTRICTED_S6_ROOT_COUNT_AT_MOST_32_B4_LT_P__PROJECTOR_TRACE_IS_EXACT_INTEGER_EXISTENCE_COUNT__FROBENIUS_IS_IDENTITY_ON_SPLIT_RATIONAL_QUOTIENT__FERMAT_PROJECTOR_AND_PSEUDOINVERSE_IDENTITIES_EXACT_BUT_NOT_CONSTRUCTIVE__MULTIROOT_DYADIC_ROUTER_REDUCES_TO_EXACT_TRACE_CONTRACTION__SUPPLIED_CP_PROJECTOR_GIVES_BRANCH_COMPLETE_ROUTER_WITH_DIRECT_A_LE_B1_OVER_4__PROJECTOR_AND_FINAL_FERMAT_POWER_HAVE_RANK_AT_MOST_Z_AND_Z_PLUS_1__SPARSE_FIBERS_DEFEAT_PROJECTOR_FLATTENING_AS_AN_OBSTRUCTION__CARRY_RANK_NOT_LIFT_INVARIANT__SHI_CONTROL_REMAINS_LIFT_SCOPED__STANDARD_TRACE_B5_OR_BALANCED_TRIPLE_B3__CAP_SIZED_TRACE_EVALUATOR_UNSUPPLIED__P1534_P1536_IDEA089_IDEA198_IDEA250_IDEA266_P1553_MERGE__NO_NEW_IDEA__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Derive or refute one exact balanced `2|3` contraction computing only
`C_(R,I)` on the full box and every queried dyadic child, without materializing
or source-supplying the `B^3` triple table, within `B^(9/4+o(1))` setup/state
and `B^(5/4+o(1))` fresh-target time/workspace. Charge the initial count, one
child per level including zero children, target specialization, exact field
and bit complexity, occurrence backpointers, leaf sign verification,
repeated-column aggregation, relation density and rank, factor logs, and
identical blind-masked descent. A negative result closes only the explicit
contraction representation it proves.
