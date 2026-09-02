# P1553 aligned-run histogram matrix-multiplication gate R32

## Classification

- Owner: existing P1553/IDEA-195 aligned-run colored-histogram residual; no
  new idea ID.
- Evidence: exact fine-grained reduction, primary-source algorithm boundary,
  and deterministic R31 compiler self-check; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_HARDNESS_REDUCTION`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: the generic aligned-run histogram problem left by
  R31 contains dense Boolean matrix multiplication even when the represented
  input, run count, alphabet, and attained histogram all have the favorable
  `O(B^2)` size. A generic `B^(9/4+o(1))` exact histogram compiler would
  therefore imply square matrix multiplication exponent at most `2.25`,
  improving the current published upper bound below `2.371339`. This is not a
  conditional impossibility theorem: `omega` may equal two, and the actual
  elliptic run family has translation consistency absent from the reduction.
  It rejects treating the residual as routine group-by, sparse convolution,
  or run decompression. No value compiler, target interface, or ECDLP
  breakthrough is supplied.

R31 gives an `O(B^2)` representation of all complete selected source domains
as aligned runs. The remaining operation is to aggregate the color triple at
every represented position, with exact integer counts and one source for each
attained key. The total expanded length is `B^3`, while the claimed complete
image has only `O(B^2)` keys.

Output sparsity alone does not make this a standard sparse polynomial
product. The three colors are arbitrary pointwise labels read from aligned
strings; they are not independent additive exponents. R32 makes the generic
difficulty exact by embedding one matrix-product entry in one designated
histogram key.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R15 translation-correlation gate | `ea007be0b237b91127e6a42501873f914ad31d6d4fd4393c3ae6b85005893b9b` |
| R31 near-period run compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R31 parent report | `00da003569e4e99e2236904fc26e1ec971a089a566fe4f1616825bd464c04483` |
| R31 bundle hash list | `af7d63c555fd8c63f87876aa77dc1c90224fa8765000223f982da7c1a99ca87e` |
| R31 staging receipt | `39b3c3824527bf739630a09386d63486e47e817f3907a9784d497b156a5f7012` |
| R31 deterministic self-check script | `7882e47d2f328250f4faf12c5bd31cc6132f457daa3c4925cc2663ca1e07170a` |
| R31 deterministic self-check report | `d1733f58385df12ca65b5fd493d1b61a887c74bcebf750a2dab5b9de86e1aacf` |

## Generic aligned-run histogram interface

Fix an alphabet `Lambda` of size `Theta(c)`. The represented input contains:

1. `O(c)` strings of length `c`, with total stored labels `O(c^2)`;
2. `O(c^2)` run records, each choosing three strings, three starting offsets,
   and a common length at most `c`; and
3. the promise that the exact histogram of ordered or one-coordinate-plus-
   unordered-pair color keys has at most `O(c^2)` nonzero entries.

For a run record `rho`, its expanded positions produce

```text
K_(rho,t)=
  (lambda_0(rho,t),
   unordered pair {lambda_+(rho,t),lambda_-(rho,t)}).
```

The output contains every attained key, its exact nonnegative integer count,
and one `(rho,t)` source. The expanded source count may be `Theta(c^3)`.

R31 instances form a special subclass: runs arise from translations of one
subset of a prime cyclic elliptic group and reuse one path decomposition.
R32 first studies the generic interface. Any transplanted database or string
algorithm that ignores the elliptic consistency must handle this class.

## Theorem 1: exact Boolean matrix multiplication reduction

Let

```text
A,B in {0,1}^(c by c).
```

Build `c` row strings `R_i`, `c` column strings `S_k`, and one constant
string `U`, all of length `c`. Use disjoint alphabet symbols

```text
alpha_i for 1<=i<=c,
beta_k for 1<=k<=c,
alpha_0,beta_0,gamma.
```

Define at position `j`:

```text
R_i[j]=alpha_i if A[i,j]=1, else alpha_0,
S_k[j]=beta_k  if B[j,k]=1, else beta_0,
U[j]=gamma.                                          (1)
```

For every pair `(i,k)`, add one length-`c` run aligning

```text
(R_i,S_k,U)
```

from offset zero. There are exactly `c^2` run records and
`2c^2+c=O(c^2)` stored labels.

For the designated complete key

```text
Y_(i,k)=(alpha_i, unordered pair {beta_k,gamma}),
```

its exact histogram count is

```text
count(Y_(i,k))
 =#{j:A[i,j]=1 and B[j,k]=1}
 =(A*B)[i,k] over the integers.                       (2)
```

All other positions use only the key families

```text
(alpha_0,{beta_k,gamma}),
(alpha_i,{beta_0,gamma}),
(alpha_0,{beta_0,gamma}).                            (3)
```

Therefore the complete attained histogram has at most

```text
c^2+2c+1=O(c^2) keys.                                (4)
```

The reduction is deterministic, exact, count-preserving, and source-
preserving. Reading the `c^2` designated counts returns the full integer
matrix product. It retains every favorable size promise of the R31 residual:

```text
stored labels O(c^2), run records c^2,
total expanded length c^3, output keys O(c^2).        (5)
```

## Corollary: the generic campaign cap improves `omega`

Suppose a generic exact aligned-run histogram algorithm used

```text
c^(9/4+o(1))
```

time on every input satisfying (5). Applying it to Theorem 1 and reading the
output would multiply two `c by c` Boolean matrices over the integers in the
same exponent. Hence

```text
omega<=9/4=2.25.                                     (6)
```

The current primary-source bound is

```text
omega<2.371339
```

from Alman, Duan, Vassilevska Williams, Xu, Xu, and Zhou,
[More Asymmetry Yields Faster Matrix Multiplication](https://arxiv.org/abs/2404.16349).
Their paper gives an upper bound, not a lower bound. Equation (6) therefore
marks an independent algorithmic breakthrough consequence; it does not prove
that the requested histogram algorithm cannot exist.

## Standard-transplant audit

### Sparse nonnegative convolution

The near-linear sparse-convolution results of Bringmann, Fischer, and Nakos,
[Sparse Nonnegative Convolution Is Equivalent to Dense Nonnegative
Convolution](https://arxiv.org/abs/2105.05984), and Jin and Xu,
[Shaving Logs via Large Sieve Inequality](https://arxiv.org/abs/2403.20326),
assume additive integer indices and a convolution whose output exponent is a
sum of independent input exponents. In (1)-(2), one output key reads two
matrix-dependent colors at the same hidden position `j`. The run offset is an
integer, but the color key is a pointwise join, not an additive convolution.
Encoding the key as an integer does not make its histogram a product of two
independent sparse polynomials.

### Factorized databases and FAQ

Factorized aggregation and the InsideOut/FAQ framework can avoid some large
intermediate joins when the represented factors and variable ordering expose
the needed distributivity. See Bakibayev et al.,
[Aggregation and Ordering in Factorised Databases](https://arxiv.org/abs/1307.0441),
and Abo Khamis, Ngo, and Rudra,
[FAQ: Questions Asked Frequently](https://arxiv.org/abs/1504.04044).
Theorem 1 is already a factorized run input with small grouped output. A
generic theorem giving the campaign exponent from those size promises alone
would also give (6). No cited result supplies that consequence.

### Compressed strings

R31 compresses source domains, not their color strings. Run-length or grammar
indexes accelerate matching when the strings themselves are compressible.
The row and column strings in (1) are arbitrary matrix rows and columns. A
routine that silently assumes repeated color runs or a small grammar does not
cover the residual.

These comparisons are interface screens, not claims that the cited
algorithms fail in their native models.

## Elliptic-family escape

Theorem 1 does not prove that its arbitrary run alignment is realizable by
R31's elliptic translations. In the actual family:

1. every signed shift acts by one group translation;
2. source and translated paths come from one common subset `Z`;
3. run mappings for different shifts commute where all points remain in
   `Z`;
4. colors are fibers of one separable degree-`c` map `g`; and
5. R28-R30 force selected singularity and residual-divisor identities.

A positive algorithm may exploit these constraints and need not solve generic
matrix multiplication. It must state which identity excludes the matrix
encoding and how that identity supports exact histogram counts and sources.

The exact next object is therefore narrower than generic group-by:

```text
an elliptic-translation-consistent aligned-run histogram for the labels of
one degree-c pencil, using the R28 residual-divisor saturation and R31 path
maps to beat generic matrix-multiplication width.
```

## Self-check receipt

The deterministic R31 script uses the additive cyclic group of order `1009`,
a 64-point support split into four paths, and eight fifth shifts. It compares
every compiled one-branch and complete domain with brute force.

```text
base paths                         4
complete run records              28
complete represented sources      229
maximum complete runs per shift   4
theorem bound per shift            12
all reconstructions exact          true
```

This checks the run compiler implementation only. It does not instantiate an
elliptic degree-`c` pencil, the histogram, a target, or ECDLP.

## Controls and boundaries

1. The reduction computes integer matrix products; field characteristic and
   modular wrap do not enter.
2. The alphabet uses `2c+3=Theta(c)` symbols, a harmless constant-factor
   enlargement of the campaign alphabet.
3. The favorable `O(c^2)` output promise is preserved exactly.
4. Equation (6) is a consequence of a hypothetical generic algorithm, not a
   lower bound under a matrix-multiplication conjecture.
5. Actual elliptic translation consistency is admitted as the surviving
   escape and must be used explicitly.
6. The self-check is `toy` theorem verification, not performance or
   cryptanalytic evidence.
7. No exact elliptic histogram, adaptive restriction package, fresh target,
   R10 index, density, rank, factor logs, blind descent, Shoup improvement,
   or breakthrough is supplied.

## Deduplication

- R14-R15 own generic moment, tensor, and translation-correlation routes.
- R31 owns the positive near-period run compiler and aligned-run residual.
- R32 adds the output-sparse matrix-multiplication reduction, standard
  database/convolution transplant screen, and binds the deterministic R31
  implementation check.

## Scoped disposition

```text
generic aligned-run histogram input: O(B^2)
generic aligned-run histogram output: O(B^2)
generic expanded source length: B^3
generic B^2.25 exact compiler consequence: omega<=2.25
current published omega upper bound: below 2.371339
generic standard transplant at campaign cap: absent
actual elliptic translation-consistent escape: open
exact elliptic complete histogram: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Exploit or refute the actual elliptic-family escape. Use the commuting R31
path maps plus the R28 residual-divisor identities to derive an exact
histogram algorithm below `B^(9/4+o(1))`, or prove that one frozen
translation-consistent grammar still embeds the matrix product. Require
complete keys, counts, one source, adaptive restrictions, fresh target, R10,
rank, logs, and descent; reject generic group-by claims that would imply
`omega<=2.25` without supplying that algorithmic advance.
