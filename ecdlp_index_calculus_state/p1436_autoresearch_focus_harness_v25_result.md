# P1436 Autoresearch Focus Harness V25 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R73: exact resultant valuation

R73 proves that the R10 rank-two multiplicative-convolution coefficient is
the root valuation of

```text
product_(u in D12,v in D34)(Z-u*v).
```

It preserves duplicate occurrence multiplicity, zero strata, dyadic
containment, and one source. Its faithful representations cost:

```text
expanded state              B^4
specialized coefficient     B^2
B coefficients              B^3
actual triple extension     B^3
```

Report SHA-256:

`00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915`

## R74: residual decision diagram

R74 uses the exact squarefree S4 endpoint support as an algebraic residual
key. On secp256k1, P-256, P-384, and P-521, every `B^3` triple key is distinct
for `B=6,10,14,18`.

The outcome-aware Boolean diagram has one state on blind targets and two on
forced targets, but only after the unique positive incidence is known. It is
root-presupposing compression and receives no algorithmic credit.

Report SHA-256:

`1558482f504bd5e05b112464ee7c6734dd30ee518789735ac7abc8c305a5f740`

## R75: iterated norm support

R75 tests the factored-norm alternative:

```text
product_(a3 in I3) S4(X1,X2,x(a3),Z).
```

Every S4 factor has all 125 monomials in its degree-four cube. Every first
norm product fills the complete `(4B+1)^3` coefficient cube on all four
curves for `B=2,3,4,6,8`.

At `B=8` this is 35,937 coefficients. For `B>4`, Z-degree `4B` is below the
degree-`B^2` suffix modulus, so remainder reduction cannot remove any term
before the `B^3` coefficient body appears.

Report SHA-256:

`c5f41fbb7325fe3f4e85fe6084fbaed6ab2df9d9101bd031cd1c589a95a3ba8c`

## Harness routing

V25 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v25.json`

SHA-256:

`c2ba092cd59e340f16cef9dfa7199c0d463166aee0a08a2e45f4153f5d7fbd4a`

Schema: `ecdlp.p1436_autoresearch_focus_report.v17`.

Eleven hash-bound lanes are closed. The top frontier is:

```text
s6_transposed_norm_scalar_functional
```

The required object computes only the suffix gcd or exact zero certificate
from the factored unary S4 norm. It may not represent the `(4B+1)^3`
coefficient cube, a `B^3` value vector, or a degree-`B^2` specialized target
polynomial. It must still pass R73 duplicate multiplicity, zero-signature,
dyadic-child, and joint-source controls inside the direct caps.

V25 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. All alphaXiv-derived
autoresearch guidance checks pass.

## Verification

- Full ECDLP task suite: 102 tests passed.
- Eight changed Python modules compiled.
- R73, R74, and R75 parent YAML receipts parsed.
- All 25 parent input/artifact hash bindings passed.
- All three parent receipts have `breakthrough=false`.
- V25 diagnostic-only and nonpromotion checks passed.
- `git diff --check` passed.
