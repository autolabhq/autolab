# P1553 M6 aggregate-union factor/label-recovery gate R163

Date: 2026-08-01

## Scope

R163 reduces the R162 nonlinear target batch to one aggregate left-endpoint
union factor and proves that target labels and C3+C3 source backpointers are
affordable postprocessing once that factor is available.

This is an exact reduction and finite correctness gate. The finite producer
enumerates endpoints to construct the union factor and receives no attack
credit.

## Same-target projector

In the split algebra `A=F_p[X]/U`, let `a_j` and `b_j` be R161's regular
x-membership and signed-y-membership residuals for target `j`. Define

```text
e_j = (1-a_j^(p-1))(1-b_j^(p-1))
m_j = 1-e_j.
```

At every C3 root of `U`, `e_j=1` exactly when both residuals vanish for the
same target. Therefore

```text
H = product_j m_j
G = gcd(U,H)
```

has exactly the left C3 endpoints participating in at least one target
decomposition. Roots where `u_j-X=0` are checked directly and unioned with
the regular factor.

It is not valid to use

```text
gcd(U, product_j a_j, product_j b_j),
```

because the two products may vanish from different targets. Coupling x and y
before the target product is required.

## Output degree

If `K` positive targets each have one positive-C6 coefficient source and the
signed C3 x-map is injective, each target contributes at most
`binomial(6,3)=20` left roots. Hence

```text
deg(G) <= 20K = B^(3/4+o(1)).
```

Factoring `G` is correspondingly output-sized once it has been constructed.

## Target labels and backpointers

Store the persistent point-to-C3-source dictionary. For each root `P` of
`G`, scan all `N=B^(5/4)` public targets, compute `Q=T_j-P`, and hash `Q` in
the dictionary. Every hit returns:

- the target label;
- the left and right C3 backpointers;
- the positive-C6 coefficient source;
- a direct public point-identity verification.

With `B^(3/4)` union roots, this costs

```text
B^(3/4) B^(5/4) = B^2,
```

below Pollard rho's `B^(5/2)` by exponent `B^(1/2)`. Thus target labeling is
not part of the remaining hard primitive.

## Deduplication

R88 already proves logarithmic localization given a correct translated-
resultant zero oracle. R163 instead aggregates the complete R159 target batch
and gives direct output-sensitive label/backpointer recovery. R161 supplies
the per-target signed membership residuals; R162 supplies the batch linear
layer and global below-rho cost window. R163 does not claim a fast aggregate
product constructor.

## Finite controls

Six public controls over three curves and two seeds verify:

- the exact aggregate union factor;
- the degree-at-most-`20K` bound;
- all expected target labels and positive-C6 sources;
- every recovered public point identity;
- empty-target rejection;
- inherited denominator-exception semantics.

A separate positive exceptional control sets `Q=-2P` and `T=-P`, so
`P+Q=T` while `x(T)=x(P)`. It verifies one genuine exceptional match and the
opposite regular orientation.

Finite endpoint and target scans receive no asymptotic attack credit and use
no candidate DLP, root, count, marginal, rank, or source oracle.

## Admission

Twenty of twenty-seven obligations pass. Admit the same-target projector,
aggregate-union biconditional, degree bound, exceptional semantics, and
`B^2` label/backpointer reduction.

Do not admit a below-rho aggregate-union constructor, deterministic
hash-to-curve transfer, an unconditional generic-prime algorithm, a
Pollard-rho or Shoup improvement, or an ECDLP breakthrough.

Disposition:

```text
ADMIT_SAME_TARGET_PROJECTOR_AND_AGGREGATE_UNION_SEMANTICS__UNION_DEGREE_B3O4__TARGET_LABEL_AND_SOURCE_BACKPOINTER_POSTPROCESS_B2__AGGREGATE_UNION_CONSTRUCTOR_OPEN__NO_RHO__NO_SHOUP_IMPROVEMENT__NO_BREAKTHROUGH
```

## Next action

Construct `G` itself below `B^(5/2)`, ideally in `B^(9/4+o(1))`, without
forming every target's `U(phi_j)` and `V(phi_j)`. The constructor may return
an unlabeled degree-`O(B^(3/4))` factor, but must couple signed x/y membership
per target before aggregation and handle denominator exceptions.
