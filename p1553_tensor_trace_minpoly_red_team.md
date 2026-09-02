# Independent P1553 tensor-trace minimal-polynomial R14 red team

Reviewer: `019f7d58-7430-7a43-b0ed-0d8e7d85c64a`
Record: coordinator transcription of the independent theorem-only response
Evidence: read-only review; no run

## Terminal verdict

```text
ACCEPT__SCOPED_NEGATIVE__COMPACT_BRANCH_MAP_AND_TRACE_HANKEL_DECODER_EXACT__STANDARD_MOMENT_CONSTRUCTORS_EXPOSE_OVER_CAP_OBJECTS__COMPACT_MOMENT_CONSTRUCTOR_CONTAINMENT_AND_JOINT_SOURCE_QUERY_REMAIN_OPEN__NO_NEW_IDEA__NO_RUN__NO_SHOUP_OR_BREAKTHROUGH_CLAIM
```

The reviewer matched the frozen producer hashes and returned `ACCEPT` without
correction.

## Accepted compact branch compiler

On a short Weierstrass chart, the two coordinates `x(A+Q),x(A-Q)` are the
roots of the explicit quadratic

```text
H_(u,q)(X)
  =(u-q)^2*X^2
   -2*((u+q)*(u*q+c_4)+2*c_6)*X
   +(u*q-c_4)^2-4*c_6*(u+q).
```

The resultant of `H` with `Q_psi(X)*Y-P_psi(X)` is quadratic in `Y` and
compiles the complete local branch divisor. Constant-many saturated charts
cover poles and exceptional cases. The resulting map has bidegree `O(d)` and
an `O(d^2)` dense coefficient description. Together with the degree-`S` and
degree-`L` support polynomials, the map input has `O(S+L+d^2)=B^(2+o(1))`
words and fits setup.

## Accepted trace-Hankel decoder

For a scalar coordinate separating the `K` complete image keys, put

```text
s_j=Tr_A(a^j)=sum_y m_y*t_y^j.
```

The geometric weights obey `1<=m_y<=8d^2<p`. The `K by K` Hankel matrix
factors as

```text
H_K=V*diag(m_y)*V^T,
```

so its determinant is the product of all `m_y` and the squared Vandermonde in
the distinct `t_y`. It is nonzero even if one image value is zero. Given a
certified `K`, the first `2K` moments therefore recover the exact squarefree
image minimal polynomial through a fast recurrence or Pade solve.

Occurrence-weighted moments recover every positive integer occurrence mass,
and constant-many key-coordinate moments recover the complete projective key
coordinates after division by `m_y`. The downstream decoder uses `O(K)` state
and `K^(1+o(1))` field work when moments, roots, and the certified stopping
bound are supplied.

The finite prefix alone is not completeness evidence. A valid constructor
must prove the image-size bound, prove `P(a)=0`, or supply an equivalent
compressed containment receipt.

## Accepted source boundary

Source-marker traces recover sums or averages over each image fiber. Pair and
fifth marker sums do not necessarily name one common preimage. Recovering a
complete source polynomial for every fiber has total degree `S*L`; a compact
joint section exists as a representation but is not constructed by the
moment decoder.

Natural direct inversion exposes an `O(d^2)` preimage join or `O(dL)` scan,
both above the online cap in the surviving degree range. These are standard
route costs, not lower bounds for every source selector.

## Accepted constructor ledger

The independent reconstruction accepted the following scoped charges:

| Route | First charged object |
|---|---:|
| Split values | `S*L=B^3` |
| Fifth-first dense resultant | `dL^2=B^(11/4..3)` coefficients |
| Endpoint-first dense resultant | `dS^2=B^(19/4..5)` coefficients |
| Full norm/resultant | degree `S*L` before radicalization |
| Sparse resultant | multidegrees `(dL,dS,SL)` |
| Standard Canny-Emiris region | `Theta((S+d)(L+d))=Theta(SL)` |
| Quotient/Krylov/power projection | source vector dimension `S*L` |
| Bihomogeneous multiplication map | source/mixed-volume scale |
| Dense repeated powers | eventual `S by L` remainder rectangle |
| Standard containment | `S*L` remainder or split values |

These are failures only for the named represented routes. Compact trace-only
circuits, truncated factorizations, and output-sensitive compilers remain
outside the theorem.

## Accepted positive controls

For multiplicatively separable maps, power sums factor into one unary sum per
deck. For additively separable maps, factorial-scaled generating functions
turn the binomial moment formula into one polynomial convolution because
`2K<p`. Under a certified image bound, both give output-sensitive image and
multiplicity decoders. An `S`-side hash table and `L`-side scan return an exact
count and one joint source in `O(L)` online work, with zero strata handled by
unary counts and backpointers.

P1510 and the composite-torsion quotient are further positive controls. They
show that family-specific factorization or uniform fibers can bypass ambient
expansion. Neither supplies a generic-prime compiler for the actual complete
Kummer branch map.

## Retained campaign omissions

R10's rank-two query asks for a few shifted values of a sparse multiplicative
convolution whose complete output can have `B^4` support. Constructing the
whole image is the wrong interface; known exact evaluation remains `B^3`
direct work or `p-1` dense modes.

No accepted route supplies adaptive child range counting, a fresh-target
action, relation density, independent signed row rank, factor logs, or
scalar-blind descent. No new idea, run, Shoup claim, or breakthrough is
authorized.

## Exactly one next action

Write one compact complete-chart tensor-trace constructor for the actual
Kummer coordinate. Require ordinary, occurrence-weighted, and complete-key
moments, a compressed `P(a)=0` proof, one jointly coupled source, and exact
adaptive-child, fresh-target, and R10 rank-two queries without an `S*L`
vector, `dL^2` dense resultant, `d^2` online join, or unproved stopping rule.
