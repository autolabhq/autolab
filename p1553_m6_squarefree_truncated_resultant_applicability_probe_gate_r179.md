# P1553 M6 squarefree truncated-resultant applicability gate R179

Date: 2026-08-01

## Scope

R179 tests whether the fast x-adic truncated-resultant algorithm of Moroz and
Schost directly constructs the R178 signed aggregate norm modulo the arbitrary
squarefree selected-divisor polynomial `U`.

The published interface is a genuine fast local algorithm, but its modulus is
`x^k`. The campaign requires a residue modulo squarefree `U`, and the two direct
adaptations both cost softly `n^2`. A factored arbitrary-squarefree dynamic-
evaluation operator remains open. This is not a resultant lower bound, an ECDLP
algorithm, or a Pollard-rho or Shoup improvement.

## Interface mismatch

Moroz-Schost computes

```text
Res_y(P,Q) mod x^k
```

in softly `O(dk)` field operations when the bivariate input degree is at most
`d`, under the paper's characteristic conditions. The quotient is local and
nonreduced.

R178 instead requires

```text
C_h mod U,  U squarefree, deg U = n.
```

For every root `a` of `U`, the polynomials `0` and `(X-a)^n` agree modulo
`(X-a)^n` but differ modulo `U`. Thus one order-`n` local expansion cannot be
substituted for the required arbitrary-modulus residue.

All six controls verify this alias witness exactly. They also verify
`gcd(U,U')=1` and reconstruct every R178 aggregate from its `n` order-one CRT
residues.

## Direct costs

There are two literal applications:

```text
one local expansion:     d >= n, k = n, softly O(n^2)
n squarefree CRT calls:  d >= n, k = 1 each, softly O(n^2)
```

Across the six controls, both degree-precision ledgers total `8,922`, exactly
the represented pair count. The 202 order-one residues reconstruct the 202
aggregate slots exactly. These finite equalities validate accounting only and
receive no asymptotic credit.

At campaign scale:

```text
n:                                      B^(9/4)
N target factors:                       B^(5/4)
desired factored total:                 B^(9/4)
one order-n x-adic application:         B^(9/2)
n order-one CRT applications:           B^(9/2)
represented target dual-Chow body:      B^(5/2)
standard nN target grid:                 B^(7/2)
rho proxy:                               B^(5/2)
```

Expanding the `N` target linear factors also exposes `Theta(N^2)` represented
dual-Chow coefficients, reaching rho before the outer norm is constructed.

## Admission

Admit the x-adic interface scope, squarefree/local alias witness, exact CRT
replay, and the two standard cost specializations.

Do not infer a lower bound against factored D5/dynamic evaluation, transposed
modular composition, custom arithmetic circuits, or output-sensitive
constructors. No generic-prime transfer, factor logs, target descent, complete
attack, or Shoup improvement is supplied.

Disposition:

```text
ADMIT_MOROZ_SCHOST_XADIC_INTERFACE_SCOPE__SIX_R178_SIGNED_NORM_CONTROLS_REPLAYED__SQUAREFREE_U_IS_N_ORDER_ONE_CRT_COMPONENTS__LOCAL_POWER_ALIAS_WITNESSES_EXACT__ONE_ORDER_N_EXPANSION_AND_N_ORDER_ONE_EXPANSIONS_BOTH_STANDARD_N2__REPRESENTED_TARGET_CHOW_AT_RHO__DIRECT_TRUNCATED_RESULTANT_ROUTE_CLOSED__FACTORED_SQUAREFREE_DYNAMIC_EVALUATION_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute one factored arbitrary-squarefree dynamic-evaluation
operator. It must keep `U,V` and all `N` target factors factored, split `U` only
by charged gcds at actual nonunits, share elimination work across every CRT
component, and emit `C_h mod U` or `G_1` in softly `O(n+N)` total work. Reject
one order-`n` x-adic surrogate, `n` local resultants, `N` quotient-ring elements,
`N^2` coefficient expansion, `nN` or `n^2` grids, candidate inversions, and
unit-cost resultant, norm, multipoint, root, count, marginal, rank, or source
oracles.
