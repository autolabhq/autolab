# P1553 M6 marked-Fitting/signed-norm deduplication gate R178

Date: 2026-08-01

## Scope

R178 asks whether the R177 restricted-kernel characteristic polynomial creates
a distinct below-rho ECDLP primitive. It does not. Its first multiplicity layer
is exactly the R174 signed aggregate-norm candidate factor, while higher layers
record repeated incidences without adding candidate roots.

The filtration and six finite controls are exact. This is mechanism-level
deduplication, not an output-sensitive constructor, an arithmetic-circuit
lower bound, an ECDLP algorithm, or a Pollard-rho or Shoup improvement.

## Signed Row Norm

For each selected point `P`, define

```text
C_h(P) = product_(Q in D) h(P+Q),
```

where `h` is the R176 principal signed-incidence witness and all auxiliary
denominators are units on the selected pair grid. Then

```text
C_h(P)=0  if and only if  there exists Q in D with h(P+Q)=0.
```

Interpolating these values on the squarefree selected divisor gives one
aggregate element `C_h` modulo `U`. Its candidate factor is

```text
G_1 = gcd(U,C_h).
```

All six controls verify that this factor is identical, including its canonical
coefficient hash, to both the R174 signed dual-Chow factor and the R177 gcd.

## Fitting Filtration

Let

```text
mu(P) = #{Q in D : h(P+Q)=0}
```

and define the squarefree threshold factors

```text
G_r(A) = product_(mu(P) >= r) (A-x(P)).
```

The R177 marker is

```text
L(A) = det(AI-X_1 | ker K)
     = product_P (A-x(P))^mu(P)
     = product_(r>=1) G_r(A).
```

In particular,

```text
gcd(U,L) = G_1 = gcd(U,C_h).
```

The controls have global threshold degrees

```text
deg G_1: 140
deg G_2:  71
deg G_3:  23
deg G_4:   7
sum:      241
```

The sum is the R177 kernel dimension and marker degree. The first layer contains
all 140 distinct candidate roots. The remaining 101 degree units encode only
incidence multiplicities and do not alter R163 label/backpointer recovery or
signed candidate verification.

## Exact Controls

Across three curves and two seeds:

```text
selected pair evaluations:                 8,922
signed row norms:                            202
aggregate quotient-ring output slots:        202
nonzero aggregate coefficients:              202
distinct candidate roots:                    140
incidence multiplicity sum:                  241
maximum incidence multiplicity:                4
```

Every aggregate interpolation is exact and fully dense. Every row-norm zero
set equals the R177 candidate set. Every threshold product equals the R177
marker. These finite computations receive no asymptotic credit.

## Algorithmic Deduplication

For the ECDLP contract, only distinct candidate roots matter. A full-marker
constructor emits `G_1` after one gcd with `U`; directly emitting `C_h mod U`
or `G_1` is sufficient. Reading compact `U,V` already costs `Theta(n)`, so the
desired full-marker and signed-aggregate interfaces have the same softly
`O(n+N)` campaign envelope.

R177 therefore does not open a separate asymptotic lane. It gives a useful
multiplicity-refined certificate for the same candidate factor isolated by
R174. The unresolved algorithmic object is still the nonlocal signed elliptic
translate product or fused dual-Chow outer norm.

## Cost Boundary

```text
selected divisor input n:                       B^(9/4)
target witness input N:                         B^(5/4)
distinct candidate factor:                      B^(3/4)
marked multiplicity output:                     B^(3/4)
represented aggregate output:                   B^(9/4)
explicit signed target grid nN:                  B^(7/2)
R177 pair algebra n^2:                          B^(9/2)
R177 explicit marker interpolation:              B^6
conditional nonlocal signed norm total:          B^(9/4)
R163 label/backpointer postprocessing:             B^2
rho proxy:                                       B^(5/2)
```

The represented aggregate output fits below rho, but no constructor for it is
supplied. R171 proves that standard node-local Miller streaming telescopes back
to the same `N` target leaves at `nN` work. R174's represented target grids and
R177's pair/Fitting constructions are also above rho. These are standard-route
costs, not general lower bounds.

## Admission

Admit the signed row-norm support identity, exact aggregate interpolation,
Fitting threshold filtration, equality of `G_1` with the R174 and R177
candidate factors, all six controls, and mechanism-level lane deduplication.

Do not admit a nonlocal signed translate-product constructor, an arithmetic-
circuit lower bound, deterministic hash-to-curve transfer, a generic-prime
coordinate-family algorithm, a complete ECDLP attack, or a Pollard-rho or
Shoup improvement.

Disposition:

```text
ADMIT_MARKED_FITTING_FILTRATION_L_EQUALS_PRODUCT_G_R__G1_IS_GCD_U_SIGNED_ROW_NORM__SIX_EXACT_CONTROLS__THRESHOLD_DEGREES_140_71_23_7__G1_IDENTICAL_TO_R174_AND_R177_CANDIDATE_FACTOR__HIGHER_MULTIPLICITIES_ADD_NO_ECDLP_ROOTS__MARKED_FITTING_NOT_DISTINCT_ALGORITHMIC_LANE__UNIFIED_NONLOCAL_SIGNED_TRANSLATE_PRODUCT_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Return to the unified nonlocal signed elliptic translate-product primitive.
From compact `U,V` and the target divisor or principal witness, emit the
aggregate signed norm modulo `U` or `G_1=gcd(U,C_h)` in softly `O(n+N)` work.
Reject `nN` target grids, `N` dense quotient-ring elements, `n^2` pair or
Fitting state, the full marked determinant body, candidate inversions, and
unit-cost norm, resultant, multipoint, root, count, marginal, rank, source, or
generic locator oracles. Higher multiplicity layers are optional diagnostics
and receive no ECDLP attack credit.
