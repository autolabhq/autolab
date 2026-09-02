# P1553 torus C5 explicit-split global-rebalance gate R122

## Claim boundary

R122 proves a scoped exponent lower bound for every fixed-arity relation
route that stores an explicit `C_s` occurrence/output table and enumerates a
complementary `C_r` query after the R118 one-C branch. It does not lower-bound
nonoccurrence arithmetic circuits, arbitrary data structures, or
actual-filtered-deck compression with a new support theorem.

No inside-cap locator, known-RHS rank, factor logs, identical descent,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is claimed.

Classification:

```text
ONE_C_BRANCH_WITH_STORED_CS_AND_ENUMERATED_CR_HAS_RELATION_COLLECTION_L_DELTA_MALPHA_RPLUS2BETA__SUPPLY_AND_SETUP_FORCE_L_AT_LEAST_11O4_PLUS_BETA_ABOVE_RHO_FOR_EVERY_ARITY__R115_M6_R121_C2C3_VERTEX_COSTS_B11O4_FRESH_AND_B7O2_COLLECTION__IID_SETUP_SIDE_SUPPORT_REMAINS_OCCURRENCE_SCALE_WHP__FILTERED_COLLISION_COMPRESSION_AND_NONOCCURRENCE_TORUS_CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Frozen model

For a relation

```text
mF = mA + mC
```

write

```text
|A| = B^(alpha+o(1)),
|C| = B^(beta+o(1)),
0 <= alpha <= beta,
q = B^(5+o(1)).
```

The R115 density retry and meaningful factor-log rank exponents are

```text
delta = max(0, 5-m(alpha+beta)),
rank  = beta.
```

After enumerating `A_m` and one distinguished C atom, answer the remaining
`C_(m-1)` query by storing `C_s` and enumerating `C_r`, where

```text
s+r=m-1.
```

The charged exponents are:

```text
stored state       = s*beta,
query enumeration  = r*beta,
one branch attempt = m*alpha+(r+1)*beta,
relation collection
                   = L
                   = delta+m*alpha+(r+2)*beta.
```

The final `beta` in `L` collects the required meaningful independent rows.

## All-arity theorem

Substitute `r=m-1-s`:

```text
L = (delta+m(alpha+beta)) + (1-s)*beta.
```

By the definition of the retry exponent,

```text
delta+m(alpha+beta) >= 5.
```

The setup condition is

```text
s*beta <= 9/4.
```

Therefore every setup-eligible explicit split satisfies

```text
L >= 5+beta-s*beta
  >= 11/4+beta
  > 5/2.
```

The strict gap above Pollard rho is at least

```text
1/4+beta.
```

This holds for every fixed relation arity and every polynomially nonempty
meaningful C deck (`beta>0`). Increasing arity cannot absorb the R121 query
cost inside this grammar.

## R115/R121 vertex

At

```text
m=6, alpha=1/12, beta=3/4, s=3, r=2,
```

the exact costs are

```text
C3 stored state              B^(9/4),
C2 query                     B^(3/2),
density-adjusted fresh work B^(11/4),
relation collection          B^(7/2).
```

Thus the exact finite C2|C3 source algorithm from R121 is not rescued by
the R115 global balance.

## Random-deck support

For an iid uniform deck in a prime cyclic group and fixed source arity `s`,
two distinct canonical multiplicity vectors collide with probability
exactly `1/q` once `q>s`. With

```text
M_s=binom(n+s-1,s),
```

the expected colliding-pair count is `binom(M_s,2)/q`, and

```text
Pr[|sC| < (1-epsilon)M_s]
  <= (M_s-1)/(2*epsilon*q).
```

Every setup-eligible stored side has `s*beta<=9/4<5`, so its iid endpoint
support remains `B^(s*beta+o(1))` with high probability. Replacing occurrence
storage by a distinct-endpoint dictionary therefore does not improve the
exponent in that model.

This theorem is not transferred to every filtered deck. A claimed
collision-compressed filtered table must supply its own support,
multiplicity, source, and relation-rank receipts.

## Controls

- Exhaustive `q=11,n=3` decks verify the exact `1/q` canonical-source pair
  collision theorem for source arities one through four.
- Three selected rational regimes, including the R115/R121 vertex, satisfy
  the symbolic identity and setup lower bound exactly.
- A denominator-48 grid checks 253,368 regimes for arities 3 through 20.
- Of those, 97,453 are setup eligible and none violates the theorem.
- The finite grid is a regression control and receives no proof credit.

## Admission

Eleven of eighteen obligations pass. The nonoccurrence target-specialized
torus circuit, known-RHS rank, factor logs, identical descent, Shoup
improvement, and breakthrough obligations remain false.

Disposition:

```text
REJECT_GLOBAL_ARITY_REBALANCE_WITH_EXPLICIT_CS_TABLE_AND_CR_ENUMERATION__ADMIT_EXACT_L_GE_11O4_PLUS_BETA_THEOREM_AND_IID_SETUP_SUPPORT_ONLY__PRESERVE_FILTERED_COLLISION_COMPRESSION_AND_TARGET_SPECIALIZED_NONOCCURRENCE_TORUS_C5_CIRCUIT__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one target-specialized nonoccurrence torus C5
membership/source circuit outside stored `C_s` endpoint tables and enumerated
`C_r` complements. It may inject the target into the exact R121 form before
endpoint expansion, or use filtered-deck compression only with an actual
support/source theorem. Require `B^(9/4+o(1))` state, polylogarithmic
arbitrary-target work, exact empty certification, five projective
backpointers, no field DLP, and complete pairing-to-descent cost receipts.
