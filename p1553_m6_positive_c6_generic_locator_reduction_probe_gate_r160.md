# P1553 M6 positive-C6 generic-locator reduction gate R160

Date: 2026-08-01

## Scope

R160 determines whether the missing R159 positive-C6 source locator can be
an encoding-invariant generic-group algorithm. It binds the admitted R159
coverage/rank theorem, the earlier R116 C3+C3 interface, the R148 direct-batch
cost audit, and Shoup's classical generic-group lower bound.

This is a reduction and finite correctness gate. It does not construct a
coordinate-aware locator or a generic-prime ECDLP algorithm.

## Semantic deduplication

R160 does not claim the C3+C3 split or the standard scan/table costs as new:

- R116 already proves the exact C3+C3 source interface and charges standard
  indexing, translated-divisor, resultant, group-algebra, and Fourier routes.
- R148 already charges a length-`B^(9/4)` C3/occurrence list over a
  `B^(5/4)` target batch: scan work `B^(7/2)` and pair-table state
  `B^(9/2)`.

The R160 delta is that R159 now supplies zero all-column coverage and a
full-rank theorem. A locator at the requested caps would therefore complete a
generic DLP reduction, rather than merely answer an isolated pair-sum query.

## Generic challenge embedding

Let the prime-order challenge be

```text
Q = [x]G in a group of order q=B^5.
```

For every factor-base column, sample public `a_i in F_q` and nonzero
`b_i in F_q`, and form

```text
C_i = [a_i]G + [b_i]Q.
```

For fixed `x`, the scalar `a_i+b_i x` is uniform because `a_i` is uniform.
Thus the `C_i` are iid uniform group elements. Rejecting zero and
equal-up-to-sign encodings gives the conditioned ideal factor-base law used
by R159. This construction uses only generic scalar multiplication, group
addition, equality, and public coefficient sampling.

## Relation solve and DLP extraction

For each R159 target

```text
tG + [s_j]C_j,
```

a returned positive-C6 source `v` gives the public equation

```text
(v-s_j e_j) dot ell = t,
```

where `ell_i=log_G(C_i)`. R159 proves all-column coverage and full rank with
probability `1-o(1)` in the iid model. Solving the resulting system gives
every `ell_i`, after which

```text
x = (ell_i-a_i)/b_i mod q.
```

Public scalar multiplication verifies the candidate. The separate target
descent is not needed for this contradiction, although the finite controls
also replay it successfully.

## Cost contradiction

With `q=B^5`, the relevant costs are

```text
factor-base dimension       B^(3/4) = q^(3/20)
C3/setup cap                B^(9/4) = q^(9/20)
R159 target batch           B^(5/4) = q^(1/4)
dense d-by-d solve          B^(9/4) = q^(9/20)
reduced generic DLP total   B^(9/4+o(1)) = q^(9/20+o(1))
generic lower bound         q^(1/2)
```

Therefore an encoding-invariant classical generic-group locator satisfying
the R159 success interface at `B^(9/4)` setup and `B^(5/4)` batch work would
contradict Shoup's generic lower bound by exponent `q^(1/20)=B^(1/4)`.

Primary source:

```text
Victor Shoup, Lower Bounds for Discrete Logarithms and Related Problems,
EUROCRYPT 1997.
https://www.shoup.net/papers/dlbounds1.pdf
local sha256 89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3
```

This is not a general static-data-structure lower bound, an arithmetic-circuit
lower bound, or a lower bound on algorithms that inspect elliptic coordinates
and perform finite-field operations unavailable in the generic model.

## Finite controls

The producer runs six public prime-order controls: three curve families and
two fixed seeds. Each control embeds a verifier DLP challenge using public
`(a_i,b_i)` coefficients and uses an explicit C3 scan as the source locator.

All six controls:

- cover every factor-base column;
- produce exact public relation identities;
- attain full relation rank;
- recover and publicly verify every factor log;
- recover the embedded DLP from `(ell_i-a_i)/b_i`;
- publicly verify the recovered DLP;
- verify identical positive-C6 target descent;
- consume no candidate DLP, root, count, marginal, rank, or source oracle.

The finite locator scans the `B^(9/4)` C3 list for every target, costing
`B^(7/2)` over the asymptotic batch. It receives no attack credit.

## Admission

Seventeen of twenty-four obligations pass. Admit:

- the challenge embedding into iid generic factor points;
- the R159 relation-system-to-DLP reduction;
- the `q^(9/20)` total exponent calculation;
- exclusion of an encoding-invariant locator at the requested caps under
  Shoup's classical generic-group model;
- all six finite correctness controls.

Do not admit:

- a coordinate-specific S7, resultant, or FFE source locator;
- deterministic hash-to-curve pseudorandomness;
- an unconditional total attack cost;
- a generic-prime coordinate-family algorithm;
- a Pollard-rho or Shoup improvement;
- an ECDLP breakthrough.

Disposition:

```text
ADMIT_R159_TO_GENERIC_DLP_REDUCTION__EXCLUDE_ENCODING_INVARIANT_LOCATOR_AT_CAPS_UNDER_SHOUP__PRESERVE_COORDINATE_SUMMATION_POLYNOMIAL_RESULTANT_FFE_ESCAPE__NO_LOCATOR__NO_RHO__NO_SHOUP_IMPROVEMENT__NO_BREAKTHROUGH
```

## Next action

Construct a coordinate-specific source-returning positive-C6 locator from a
compact factor-base x-polynomial and the elliptic `S7` summation relation, or
an equivalent resultant/FFE circuit. It must:

- use at most `B^(9/4+o(1))` setup and `B^(5/4+o(1))` complete-batch work;
- expose the coordinate operation that prevents generic-group simulation;
- avoid materializing the `B^(9/2)` C3+C3 body;
- return positive-C6 source backpointers for relation collection and descent;
- replay the admitted R159 factor-log solve and identical descent.
