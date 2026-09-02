# Independent Red Team: TASK-20260719-P1553-NET-RT-R1

Read-only theorem audit. No experiments, runs, contracts, P1554 allocation, Shoup claim, or breakthrough claim.

## Terminal Scoped Verdict

```text
REVISE_SCOPED_NEGATIVE__RAW_INDEX_CORRECTED__EXACT_ZERO_SUPPORT_ONLY_ON_COMPLETE_NONDEGENERATE_CHART__GAUGE_SUPPORT_INVARIANT_RAW_UNITS_NOT_INVARIANT__STANDARD_MIXED_SEEDS_B4_OR_FIXED_LABEL_B2__B5OVER2_ONLY_SUPPLIED_RECURRENCE_ENVELOPE__DIAGONAL_PRODUCT_NONCLOSURE_SCOPED_TO_STANDARD_COMPONENTWISE_RECURRENCES__NO_RELATION_ALL_UNIT__RANK_LOGS_DESCENT_UNSUPPLIED__P1513_P1540_P1551_P1516_MERGE__NO_GENERAL_LOWER_BOUND__NO_SHOUP_CLAIM__NO_P1554__NO_RUN__NO_BREAKTHROUGH
```

## Producer Defects

1. The raw interface `W(j,1)=Psi_(j,1)(T,q)` fails at `j=0,1`. Stange normalizes `Psi_(0,1)` and `Psi_(1,1)` to 1, so these values cannot certify zeros.

2. The exact correction is:
   `q=A_0+u+v-R`,
   `Q=q-2T`,
   `W(j+2,1)=Psi_(j+2,1)(T,Q)`.
   Since `(j+2)T+Q=jT+q`, the corrected zero condition is exact on the nondegenerate chart.

3. The producer's ambient projection statement is wrong if it calls
   `E_K x Spec(A_I) -> Spec(A_I)` finite. It is proper/projective. Only the restriction of the intersection scheme to the finite divisor is finite.

4. The raw product of net terms is not proved equal to the R4 key-difference resultant. The charitable replacement is only the same zero-support surrogate `r_R(t_j)`, not equality of raw values.

5. The `B^(5/2)` line is only a supplied-recurrence envelope. It is not an elliptic-net construction and exceeds the `B^(5/4)` online cap.

6. Complete chart handling is mandatory for `Q=O`, `Q=+/-T`, tangent, vertical, infinity, repeated, and nonreduced strata. Denominator clearing without saturation can create false components.

7. The producer receipts contain stale SHA bindings:
   - producer ledger hash `21cb...` versus actual `62da...`;
   - producer current-plan hash `422e...` versus actual `d752...`;
   - producer dispatch hash `a66d...` versus actual `623c...`;
   - producer canonical task hash `097e...` versus actual ZR-P1 task hash `a915...`;
   - red-team receipt canonical task hash `7d12...` versus actual ZR-RT-R1 task hash `ee60...`;
   - red-team dispatch hash `bc21...` versus actual `623c...`.

8. The checked producer YAML, red-team YAML, and parent YAML all parse successfully. No malformed or non-ASCII YAML key was found.

## Mathematical Reconstruction

Let `n=Theta(B^2)` be the size of each pair deck. There are
`Theta(B^4)` labelled pair-pair components. For a fresh target:

```text
q_(u,v)=A_0+u+v-R
Q_(u,v)=q_(u,v)-2T
W_(u,v)(j+2,1)=Psi_(j+2,1)(T,Q_(u,v)).
```

The corrected term tests:

```text
[j]T+q_(u,v)=O
```

and therefore the relation:

```text
u+v+A_j=R.
```

This is a proved support statement only after complete chart handling.

The exceptional cases must be handled as follows:

- `Q=O`: direct identity handling; relation condition `(j+2)T=O`.
- `Q=+T`: direct complete-law handling; relation condition `(j+3)T=O`.
- `Q=-T`: direct complete-law handling; relation condition `(j+1)T=O`.
- Tangent: use the tangent branch only under its nonzero denominator mask.
- Vertical: use the inverse/equal-x branch and retain output `O`.
- Infinity: use projective identity branches.
- Repeated endpoints: retain labels and multiplicity; `z_R` records support once.
- Nonreduced intersections: local length changes, but Fitting support does not.

## Gauge Boundary

Under independent nonzero quadratic rescaling,

```text
W'_q(z)=c_q*f_q(z)*W_q(z).
```

Thus zeros, zero support, gcd support, and interval zero/nonzero decisions are invariant. Raw nonzero values and raw dyadic unit products are not invariant. A value-sensitive algorithm must construct and charge a normalization.

## Mixed Seeds And Costs

The scalar fifth orbit preserves only a heuristic coverage argument. With four generic decks, `Theta(B^4)` labelled tuples and `B` shifts give constant expected coverage under a random-endpoint model. Correlations, relation density, and independent row rank remain unproved.

The standard costs are:

| Object | Cost |
|---|---:|
| Individual mixed pair-pair seeds | `B^(4+o(1))` |
| One fixed-label `r_R(t_j)` surrogate | `B^(2+o(1))` |
| All `B` fixed-label resultants | `B^(3+o(1))` |
| Explicit translated endpoints | `B^(3+o(1))` |
| Optimistic supplied recurrence envelope | `B^(5/2+o(1))` |
| Required online cap | `B^(5/4+o(1))` |

The one-fixed-label surrogate is the strongest charitable grant. It already exceeds the online cap and does not return all fifth labels or source witnesses.

## Recurrence Closure

For each component, a standard quartic net recurrence has the form:

```text
A_q+B_q+C_q=0.
```

For diagonal products,

```text
A=product_q A_q
B=product_q B_q
C=product_q C_q.
```

The component equations do not imply `A+B+C=0`. Multiplication expands into mixed choices over `{A,B,C}^q`. This proves non-closure for the standard componentwise diagonal-product construction.

It does not prove monomial independence, an arithmetic-circuit lower bound, a Boolean-circuit lower bound, or impossibility of an unknown compact aggregate recurrence.

## No-Relation And Replay

On a no-relation target, every component resultant is a unit. Therefore every `F_R(j)` and every dyadic interval product is a unit. No early zero divisor or dynamic split is available. Positive and negative child tests must both be charged, followed by complete-law singleton verification.

Independent gauges preserve the zero decision but alter every raw unit value.

## Conditional ECDLP Path

The proposed path remains conditional:

```text
setup                         B^2
B known-log relation targets  B^(9/4)
sparse factor-log algebra     B^2
one masked descent            B^(5/4)
lambda                        0.45
mu                            0.40
```

Missing claims are the target-fresh constructor, relation and target success density, independent rank, factor-log completion, scalar verification, identical scalar-blind descent, collision and deck-rebuild accounting, failed replay accounting, and bit costs.

## Operation-Level Deduplication

- P1540 owns net evaluation, quadratic gauge, translated-coordinate, QRT, EDS, Fourier, and index-location controls.
- P1513 owns translated products, common-factor location, and source replay.
- P1551 owns finite-domain selector and represented endpoint-coefficient access.
- P1516 owns the two pair indexes and missing target-local router.

The proposed net aggregate is therefore a semantic merge of existing operations. It does not justify a new owner or P1554.

## Exactly One Next Action

Under existing P1553/P1513/P1551/P1516 ownership, require one theorem-only oracle-free construction of `r_R mod g_I` or exact dyadic unit products within total `B^(5/4+o(1))`, including complete charts, target freshness, and positive and negative replay. Otherwise preserve this scoped exception unchanged.

## Source Binding

All hashes below are SHA-256 over the read bytes.

| Source | SHA-256 |
|---|---|
| `AGENTS.md` | `4b9810aaa2c96a9e8d7db097d6abfc8cbeb24038df3a09e98f0beb4c23a6d362` |
| `agents/red-team.md` | `7ae9372d518fba2b9868eccf1d99102cde1ac6dae2d7bb593971d264314893f5` |
| `p1553_target_label_common_factor_gate_r4.md` | `8cf15364c2da6830255216f3766a5d016b847d4bd012df92fd86c462ee6a9bc1` |
| `p1540_elliptic_net_translated_pole_annihilator_gate.md` | `d9a4040230022c24f7011932ef7cd9b5bcea51236a80c042bb498d2012428437` |
| `p1540_r1_independent_audit.md` | `8032be2d3a645ac64c046783191cc9c634715518eb18e4702acf66e077223d45` |
| `zr_report.yaml` | `2501e4fa56f425186313472e0da57cd8c5fae66d3cea22f0464ca4d82fc62adf` |
| `zr_theorem_gate.md` | `41a93d0f0d9c1c0b1a1adb0debf39b8a84425f0d4f7147ea5eb12c89719c2b08` |
| `red_team_report.yaml` | `0f1041d6a8aa2859e8e087e88462044f7cab76da08c8e78f0597f857838fc8a3` |
| `zr_red_team.md` | `e7b9150064adc393850c410a092f26b3ebd1fadbf7a9b714ff0f9b70b5d0216c` |
| `translated_product_common_norm_v3_audit_v2.md` | `407e3c7da6345f156f7c6bcaa75749e16b6184735d32be4b6e4aca69427763d5` |
| `ku_circuit_reduction_v2.md` | `6fcca1d12e911f6eb2142ac96b6d0a83b6ac20db11efd06bc24c0abb7c99dc48` |
| `pair_sum_quotient_theorem.md` | `18cebc9c209c6ba0d705e43da7f921885e60d3436b201375e306e14f4ae0bdb2` |
| `FINDING-PF-IC-001.md` | `62da10b47e9c0a9dd289f7ef4890df0d23098cc94ab9be52ab748f41411254b4` |
| `current_plan.json` | `d7527467e6f88fed7f86dbd81feef865b4832980ef94579833aec74dfcc033b6` |
| `current_plan.md` | `5d5d360a0e99e98f90d062fd5908d0690397ee060161f2700422291c7ae2563d` |
| `dispatch_queue.json` | `623ce5e6644e7831f6781bdc7240a0ba1cb1480dd14037c39ea590cee84eb7b2` |
| `p1553_net_parent_report.yaml` | `16d5b3499b9255e1f70db7641cfdd82154a952401c5502c1ab3bf7c0b8432a82` |
| `p1553_net_parent_theorem.md` | `695de47e3debcf8f316071a31c34feb71326e0de02dc086b2f9bb92926b8b420` |
