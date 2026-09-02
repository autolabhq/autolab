# P1553 M6 sparse-projector Prony locator gate R183

Date: 2026-08-02

## Scope

R183 narrows the R182 output-sensitive resultant opening. It proves an exact
scalar output interface: target-gcd multiplicities become weights on source
x-coordinates, and `2c` weighted moments recover the degree-`c` candidate
locator by Berlekamp-Massey/Prony. It does not construct those moments directly
from the compact line-product SLP, prove a circuit lower bound, construct a
generic-prime ECDLP algorithm, or improve Pollard rho or Shoup.

## Nested Projector Identity

Let the squarefree source divisor be `A=F[X]/U`, and partition the deduplicated
elliptic targets into squarefree x-injective charts

```text
B_r = F[u]/W_r.
```

For a source point `P`, R182 supplies the target-chart remainder `R_P`. Define

```text
m_P = sum_r Tr_Br(1 - M_RP^(p-1)).
```

On every split chart component, Fermat's identity makes the summand one when
`R_P` vanishes and zero otherwise. Therefore

```text
m_P = sum_r deg gcd(W_r, R_P).
```

The combined target degree is below the characteristic in every admitted
instance. Thus the integer multiplicity does not vanish modulo `p`, and

```text
m_P != 0  iff  P is a candidate source component.
```

This handles repeated incidences without assuming all candidate weights equal
one. Public duplicate removal and the two-chart opposite-point treatment are
inherited from R182.

## Sparse Locator

Let `c` be the number of candidate source x-coordinates and form

```text
s_k = sum_P m_P x(P)^k,  0 <= k < 2c.
```

Distinct source roots and nonzero weights give a rank-`c` weighted Vandermonde
factorization of the Hankel matrix. Berlekamp-Massey recovers order `c`; after
coefficient reversal, its recurrence polynomial is exactly

```text
G_1(X) = product over candidate P of (X - x(P)).
```

The degree need not be known in advance. This proves the output and recovery
stage once the moments exist; it does not supply a compact constructor for the
moments.

## Exact Controls

R183 replays all twelve R182 controls and adds a strided source-divisor family
at seed `18307`. The new family uses a degree-seven source divisor, five unique
targets in two charts, three candidates, four noncandidates, and maximum target
multiplicity two. Its candidate support is checked independently by elliptic
group addition.

```text
control count:                                      13
R182 replay / new divisor controls:             12 / 1
selected source degree sum:                         411
unique target degree sum:                            84
candidate / noncandidate source components:    256 / 155
target-gcd multiplicity sum:                        414
weighted moment count:                              512
represented source-target value slots:            2,997
distinct target-value spectrum degree sum:        2,839
full-spectrum source rows / source rows:       298 / 411
```

Every Fermat-projector trace equals the corresponding gcd degree, every gcd
degree equals the direct group-law multiplicity, every nonzero multiplicity
selects exactly the candidate support, every weighted Hankel rank and
Berlekamp-Massey order equals `c`, and all thirteen locators equal the exact
candidate factors. No candidate oracle receives credit.

An exact collision over `F_101` swaps target-value rows
`[0,3,7]` and `[4,5,6]` between source roots one and two. All twelve unweighted
global target power sums remain identical, while the candidate locator changes
from `X-1` to `X-2`. A source-blind scalar target trace therefore cannot recover
source labels.

## Cost Boundary

```text
n selected source degree:                     B^(9/4)
N target degree:                              B^(5/4)
c candidate degree:                           B^(3/4)
2c moment output:                             B^(3/4)
Prony/Berlekamp-Massey recovery:               B^(3/4)
candidate-target verification:                    B^2
desired compact setup:                        B^(9/4)
represented tensor projector:                 B^(7/2)
A-linear N-term target Krylov output:          B^(7/2)
rho proxy:                                     B^(5/2)
```

The desired moment interface and its downstream work are below rho, but no
algorithm realizes the interface from compact inputs at that cost. Explicitly
representing `R`, the Fermat projector, or an A-valued target Krylov sequence
retains the `nN=B^(7/2)` body. Fast modular composition and power projection do
not remove that represented dimension by themselves.

## Prior-Lane Deduplication

- R97 tested pointwise Fermat projectors and reverse adjoints; it closed the
  represented projector lane. R183 adds R182 target-gcd multiplicity weights
  and sparse source moments, but inherits the missing compact nonlinear trace.
- R163 established that a degree-`c` locator is an acceptable output and that
  its postprocessing can be charged at candidate scale. R183 supplies an exact
  weighted-moment recovery of that locator, not its compact upstream producer.
- R168 tested denominator-aware elliptic Cauchy traces and candidate poles.
  R183 instead detects target-chart zeros through gcd multiplicities; neither
  gate supplies the required SLP-direct nested trace.
- Wiedemann-style scalar minimal sequences explain the recurrence recovery;
  they do not preserve source labels after a source-blind projection. The exact
  row-swap collision records this distinction.

## Admission

Admit the nested Fermat-projector trace identity, multiplicity
non-cancellation under total target degree below `p`, all thirteen exact
controls, the source-blind collision, and candidate-locator recovery from `2c`
weighted moments. Nineteen of twenty-eight obligations pass.

Close source-blind global target traces as label-incomplete. Charge represented
tensor projectors and A-linear target Krylov output at `B^(7/2)`.

Keep the lane closed because no SLP-direct nested projector-moment constructor
is supplied. There is no factor logarithm stage, identical fresh-target
descent, complete attack, deterministic generic-prime transfer, rho
improvement, Shoup improvement, or breakthrough.

Disposition:

```text
ADMIT_NESTED_FERMAT_PROJECTOR_TRACE_EQUALS_TARGET_GCD_DEGREE__NONZERO_MULTIPLICITY_SUPPORT_EXACT_BECAUSE_N_LT_P__TWELVE_R182_REPLAYS_AND_ONE_NEW_STRIDED_DIVISOR_FAMILY__WEIGHTED_2C_SOURCE_MOMENTS_RECOVER_G1_BY_PRONY_BERLEKAMP_MASSEY__SOURCE_BLIND_GLOBAL_KRYLOV_TRACE_LABEL_COLLISION_EXACT__OUTPUT_AND_RECOVERY_B3O4__CANDIDATE_VERIFICATION_B2__REPRESENTED_PROJECTOR_AND_A_LINEAR_KRYLOV_B7O2__SLP_DIRECT_NESTED_MOMENT_CONSTRUCTOR_OPEN__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute one SLP-direct nested projector-moment operator. From
compact `U,V`, target charts, and the R181 signed line-product SLP, emit the
`2c` scalars

```text
Tr_A(X^k Tr_B(1-M_R^(p-1)))
```

in softly `O(n+N+c)` work without representing `R`, the Fermat projector, an
A-valued target Krylov sequence, or any `nN` tensor body. Test seed `18308` and
a second strided divisor family; reject source-blind traces and candidate
oracles.
