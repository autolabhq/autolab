# P1553 elliptic-net aggregate gate R5

## Classification

- Owner: existing P1553/P1513/P1551/P1516 frontier; no P1554.
- Evidence: theorem-only producer reconstruction plus independent red team; no
  run.
- Status: `REVISE_SCOPED_NEGATIVE`.
- Labels: `theorem-only`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, rank theorem, factor-log solve,
  scalar-blind descent, Shoup-bound improvement, or ECDLP breakthrough.

R4 left one representation-sensitive exception: an oracle-free,
gauge-invariant elliptic-net or division-polynomial constructor for
`r_R mod g_I` or exact dyadic zero/unit decisions inside total
`B^(5/4+o(1))). R5 tests the standard componentwise realization of that
exception. It does not construct the required aggregate state.

## Review chain

| Receipt | SHA-256 |
|---|---|
| coordinator producer report transcription | `16d5b3499b9255e1f70db7641cfdd82154a952401c5502c1ab3bf7c0b8432a82` |
| coordinator theorem reconstruction | `695de47e3debcf8f316071a31c34feb71326e0de02dc086b2f9bb92926b8b420` |
| independent red-team transcript YAML | `20f75bf1916bf3aa622c1ef7559668ece0d7ef90531ff0fb3abb23cc3bd25da6` |
| independent red-team transcript notes | `384344ab599dd1fe8c1c412d044eb8b5f40f19019300e0a29d0a8a77e27ab228` |

The original producer agent returned read-only output with two defects that
were never installed: it used the raw `(j,1)` interface at normalized
exceptional indices and emitted one non-ASCII YAML key plus one mistyped
Markdown hash. The coordinator transcriptions repair the syntax and expose the
index correction for review. The independent red team reconstructs the
mathematics and narrows the product-closure claim.

## Frozen interface

Let the two source-labelled pair decks have

```text
n=Theta(B^2)
```

occurrences each, and freeze the fifth deck to

```text
A_j=A_0+[j]T,                 0<=j<B.
```

For a fresh target `R` and labelled pair endpoints `u,v`, put

```text
q_(u,v)=A_0+u+v-R.
```

The relation condition is

```text
u+v+A_j=R  iff  [j]T+q_(u,v)=O.
```

The pair trees, ordered labels, complete addition branches, dyadic ancestors,
multiplicities, and backpointers cost `B^(2+o(1))` preprocessing and retained
state. No target-dependent coefficient, seed, cache, pointer, or orbit is
preprocessed. Total fresh-target work and workspace, including every replay
call, remain capped at `B^(5/4+o(1))`.

## Corrected net zero interface

Stange normalizes

```text
Psi_(0,1)=Psi_(1,1)=1.
```

Therefore the raw term `Psi_(j,1)(T,q)` cannot report relation zeros for
`j=0,1`. Shift the second point instead:

```text
Q_(u,v)=q_(u,v)-2T,
W_(u,v)(j)=Psi_(j+2,1)(T,Q_(u,v)).
```

Then

```text
(j+2)T+Q_(u,v)=jT+q_(u,v),
```

so on Stange's nondegenerate rank-two chart

```text
W_(u,v)(j)=0  iff  u+v+A_j=R.
```

This is an exact component zero statement, not an aggregate algorithm.
The standard rank-two net chart assumes nonzero points with no equal or inverse
pair. Components with `Q=O`, `Q=+/-T`, identity, infinity, tangent,
vertical, repeated endpoints, or nonreduced intersection require the complete
projective group-law masks from R4. Unsaturated denominator clearing is not
allowed.

## Gauge boundary

An equivalent component net has

```text
W'_q(z)=c_q*f_q(z)*W_q(z),
```

where the quadratic gauge factor is nonzero. Consequently:

- individual relation zeros are invariant;
- aggregate zero support and the fifth-label gcd support are invariant;
- interval zero/nonzero decisions are invariant;
- raw nonzero aggregate values and raw interval unit values are not invariant.

A value-sensitive constructor must rigidify and charge its normalization,
including branches where an affine normalizer vanishes. R4's `z_R` remains a
valid gauge-safe support output.

## Mixed-seed gate

There are

```text
|D_12 x D_34|=Theta(B^4)
```

generically distinct target-dependent points `Q_(u,v)`. A bounded initial net
block for one component contains nontrivial values depending on its own
`Q_(u,v)`; the normalized values `Psi_ei=Psi_(ei+ej)=1` do not supply that
mixed state.

The standard choices are:

| Representation | Fresh-target work | Workspace |
|---|---:|---:|
| all component mixed seeds | `B^(4+o(1))` | `B^(4+o(1))` |
| one exact fixed-label resultant surrogate `r_R(t_j)` | `B^(2+o(1))` | `B^(2+o(1))` |
| all `B` fixed-label surrogates | `B^(3+o(1))` | `B^(2+o(1))` peak |

The raw product of net terms is not proved equal to R4's key-difference
resultant; it may carry gauge units. R5 therefore grants only the strongest
safe surrogate with the same zero support, `r_R(t_j)`. Even one such mixed
seed exceeds the `B^(5/4)` cap. Calling it a recurrence seed, oracle,
target-trained value, or precomputed orbit does not construct it.

The earlier `B^(5/2)` orbit line remains only an optimistic
supplied-recurrence envelope, not an elliptic-net algorithm.

## Standard product-closure gate

For each pair-pair component, group the three quartic net-recurrence summands as

```text
A_q+B_q+C_q=0.
```

The desired diagonal products are

```text
A=product_q A_q,
B=product_q B_q,
C=product_q C_q.
```

The component identities do not imply `A+B+C=0`. Multiplying them introduces
all mixed choices of `A_q,B_q,C_q`, not a closed three-term recurrence on the
diagonal products.

This rejects only the standard componentwise product construction. R5 does not
prove generic mixed-monomial independence, an arithmetic- or Boolean-circuit
lower bound, a data-structure lower bound, or impossibility of a new compact
aggregate recurrence with independently constructed state.

## Restriction and no-relation gate

For a fifth interval `I`, a formal product

```text
U_R(I)=product_(j in I) r_R(t_j)
```

is zero exactly when `I` contains an extendible fifth label. Its raw nonzero
value is not a net-gauge invariant, but its zero decision is.

On a no-relation target every fixed-label resultant and every interval product
is a unit. No zero divisor enables an early split. Exact source recovery still
requires every positive and negative dyadic child call plus final complete-law
singleton verification. Repeated endpoints change row multiplicity, not the
squarefree fifth-label support.

## Complete path

Conditional on the still-missing subset-stable constructor, the earlier
accounting remains

```text
pair-index setup               B^2,
B known-log relation targets   B^(9/4),
sparse factor-log algebra      B^2,
one masked descent             B^(5/4),
lambda                         0.45,
mu                             0.40.
```

The constructor, constant verified relation and target density,
`Theta(B)` independent rows, collision and deck-rebuild accounting,
factor-log completion, identical scalar-blind descent, failed replay, and bit
costs remain unproved.

## Deduplication

- P1540 owns direct net evaluation, quadratic gauge, translated-coordinate,
  QRT/Lax, EDS, Fourier, and index-location controls.
- P1513/IDEA-121 owns translated pair products, nonlinear common-factor
  location, and source replay.
- P1551 owns represented endpoint-coefficient access.
- P1516 owns the two pair indexes and missing target-local router.
- P1553 R4 owns exact `z_R` semantics and the represented resultant route
  ledger.

The fifth-orbit net formulation is a semantic merge of these operations. It
creates no mechanism owner, idea ID, P1554, contract, or run.

## Provenance adjudication

The independent reviewer compared historical receipt hashes with later shared
files and called the old bindings stale. That objection is rejected.
The R4 producer and reviewer hashes were verified against their read-scope
inputs at intake; later coordinator edits correctly change mutable ledger,
focus, and queue bytes without retroactively invalidating immutable receipts.

The effective P1540-enriched NET task card is staged locally because this
session cannot write the external checkout. Its staged canonical hash is not
claimed to be the current authoritative queue hash. This is an installation
limitation, not mathematical evidence.

## Disposition

```text
REVISE_SCOPED_NEGATIVE__RAW_J0_J1_NET_INDEX_REPAIRED_BY_JPLUS2_QMINUS2T__EXACT_COMPONENT_ZERO_ONLY_ON_NONDEGENERATE_CHART__COMPLETE_PROJECTIVE_STRATA_REQUIRED__GCD_SUPPORT_GAUGE_INVARIANT__RAW_UNIT_VALUES_GAUGE_DEPENDENT__NORMALIZATION_DOES_NOT_SUPPLY_MIXED_SEEDS__COMPONENT_STATE_B4__CHARITABLE_ONE_LABEL_RESULTANT_SEED_B2__ALL_LABELS_B3__STANDARD_COMPONENTWISE_NET_RECURRENCES_DO_NOT_CLOSE_ON_DIAGONAL_PRODUCTS__NO_GENERIC_MONOMIAL_OR_CIRCUIT_LOWER_BOUND__NO_RELATION_ALL_UNIT_AND_REPLAY_UNPAID__P1540_P1513_P1551_P1516_MERGE__RANK_LOGS_DESCENT_UNSUPPLIED__NO_P1554__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Rerank outside standard elliptic-net, division-polynomial, EDS, QRT/Lax, and
supplied-recurrence families. Under existing P1553/P1513/P1551/P1516
ownership, admit only one theorem-only mechanism-distinct compiler that
constructs both the target-fresh mixed aggregate state and a
restriction-stable exact zero decoder inside total `B^(5/4+o(1))), with
complete charts, positive and negative replay, and exact source recovery;
otherwise preserve the specialized Query2P1 exception unchanged.

## Primary controls

- Stange, *Elliptic Nets and Elliptic Curves*,
  <https://arxiv.org/abs/0710.1316>.
- Lauter and Stange, *The elliptic curve discrete logarithm problem and
  equivalent hard problems for elliptic divisibility sequences*,
  <https://arxiv.org/abs/0803.0728>.
- Bostan, Gaudry, and Schost, *Linear Recurrences with Polynomial
  Coefficients*, <https://doi.org/10.1137/S0097539704443793>.

These sources supply individual net recurrences, normalization, quadratic
scale equivalence, zero apparition, EDS index-location controls, and
supplied-recurrence evaluation. They do not supply the mixed aggregate
compiler required by P1553.
