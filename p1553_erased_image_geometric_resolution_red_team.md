# Independent P1553 erased-image geometric-resolution R13 red team

Reviewer: `019f7d35-2ed5-7e73-be3b-d04013bd8d40`
Record: coordinator transcription of the independent theorem-only response
Evidence: read-only review; no run

## Terminal verdict

```text
ACCEPT__TRANSPOSED_FIXED_ENDPOINT_BRANCH_ENERGY_GIVES_K_AT_LEAST_S_SQUARED_L_SQUARED_OVER_D_TIMES_S_L_PLUS_32_D_BINOMIAL_L_2__DENSE_D_AT_MOST_B_GIVES_K_OMEGA_B_CUBED_OVER_D__EXPLICIT_IMAGE_STATE_FORCES_D_AT_LEAST_B_THREE_QUARTERS__O_K_IMAGE_MULTIPLICITY_AND_JOINT_SOURCE_REPRESENTATION_ACCEPTED__STANDARD_GRAPH_FIRST_ELIMINATION_EXPOSES_N_EQUALS_S_L__COMPLETENESS_ADAPTIVE_CHILD_FRESH_TARGET_AND_RANK_TWO_REMAIN_OPEN__P1510_OUTPUT_SENSITIVE_EXCEPTION_PRESERVED__NO_NEW_IDEA__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

The first reviewed R13 draft received `REVISE`. The producer then bound the
R11 gate, parent report, and red-team receipt directly; attributed the
transposed rigidity theorem to P1515/IDEA-057; and specified that the pair and
fifth selectors are one jointly chosen source section. The reviewer matched
the revised hashes and returned `ACCEPT`.

## Accepted transposed energy bound

For each endpoint `u`, let `k_u` be the number of distinct complete branch
divisors attained over the `L` fifth labels and let `C_u` count unordered
fifth-label collisions at that endpoint. Symmetry gives

```text
D_(u,q)=D_(q,u).
```

R11's pole-inclusive finite-Cartesian theorem therefore bounds, for each
distinct pair `q,r`, the number of endpoints satisfying
`D_(u,q)=D_(u,r)` by `16d`. Hence

```text
sum_u C_u <= 16d*binomial(L,2).
```

Cauchy-Schwarz within each endpoint and across all endpoints gives

```text
sum_u k_u >=
  S^2*L^2/(S*L+32d*binomial(L,2)).
```

A global key `(psi(u),D)` is represented by at most `d` endpoints because a
degree-`d` projective fiber has at most `d` distinct points. Therefore

```text
K >= S^2*L^2 /
     (d*(S*L+32d*binomial(L,2))).
```

At certified dense support `S=Theta(B^2)`, `L=Theta(B)`, and `d<=B`, this is
`K=Omega(B^3/d)`. Any explicit representation with one field word per image
state fits below `B^(9/4+o(1))` only for
`d=Omega(B^(3/4-o(1)))`.

The review explicitly rejected a broader cross-endpoint claim. R11 does not
bound all pairs `(u,v)` with `psi(u)=psi(v)` and
`D_(u,q)=D_(v,r)` by `O(d^2)`; common components of that fiber product remain
unclassified. The accepted proof needs only fixed-endpoint fifth collisions.

## Accepted representation and graph-first split

For an injective scalar coordinate on the finite image, the squarefree image
polynomial, geometric and occurrence multiplicity interpolants, full-key
coordinates, and one source section use `O(K)` coefficients. The selectors

```text
(J_12(t_y),J_5(t_y))
```

must jointly identify one common preimage of `y`. Independent marginal
selectors are insufficient.

The full graph algebra has dimension `n=S*L`. Its norm or characteristic
polynomial has degree `n`, with each image root repeated by its geometric
fiber multiplicity. Explicit multiplication matrices, graph geometric
resolutions, and pushforward/Fitting modules retain this source dimension.
At dense support they expose `B^3` represented traffic. Radicalizing only
after that construction does not refund the charge.

This is a theorem about the frozen graph-first represented grammar. P1510's
audited output-sensitive marked-resultant compiler remains a valid control
against extending it to all resultants, image algorithms, or circuits.

## Accepted completeness and replay boundary

One witness per represented root proves only that the root has a source.
Witnesses plus total geometric and occurrence masses do not prove that every
actual source image is represented. Standard containment verification returns
to the `S*L` source algebra; a compressed exact proof remains open.

One global source section also cannot answer a child that excludes its chosen
source while retaining another source in the same image fiber. Exact child
weights and child source selection require separate packages, a range index,
or an implicit fiber query within the online cap.

The compact image package supplies neither a fresh-target action nor R10's
exact sparse multiplicative-convolution control. It does not establish
relation density, independent signed row rank, factor logs, blind descent, or
an end-to-end rho/Shoup improvement.

## Exactly one next action

Audit one explicit output-sensitive compiler in the surviving dense-support
range `B^(3/4-o(1))<=d<=B`. Require the radical image, exact occurrence
weights, jointly coupled source section, compressed containment proof, every
adaptive child, fresh-target action, and R10 rank-two control inside the
campaign caps before evaluating density, rank, factor logs, and blind descent.
