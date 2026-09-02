# Experiment Result: source-support diversity scout for motif-miss windows

## Claim or task

Test whether the current known-column pressure "no structural form" windows are
true absences, or whether adjacent public source selectors/top-k choices expose
known-column relation motifs before the direct pressure screen sees them.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND.

## Inputs

Mounted AutoLab state was read as the source of truth.  The mounted
`state_update.py report` path is not writable from this sandbox, so all new
artifacts below were written locally in this worktree.

No-structural pressure windows tested:

```text
3064..3071
3080..3087
3136..3143
3168..3175
3200..3207
3208..3215
```

The scout compares each pressure miss against the mounted
`low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*`
artifacts and ranks replay candidates by selected-support diversity, target
motif coverage, priority column visibility, and existing direct/shared
verification metadata.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_source_support_diversity_scout.py
tasks/ecdlp_index_calculus/low_term_total2_direct_certificate_support_audit.py
ecdlp_index_calculus_state/low_term_total2_source_support_diversity_scout_22050_col15_no_structural_3064_3215_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_support_diversity_3064_3071_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_support_diversity_3080_3087_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_support_diversity_3136_3143_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_support_diversity_3168_3175_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_support_diversity_3200_3207_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_support_diversity_3208_3215_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_certificate_support_audit_22050_support_diversity_3064_3215_probe.json
ecdlp_index_calculus_state/low_term_total2_known_column_pressure_direct_screen_22050_col15_selector_expanded_3080_3087_support5_relaxed_probe.json
ecdlp_index_calculus_state/low_term_total2_known_column_pressure_direct_screen_22050_col15_selector_expanded_3080_3087_support3_relaxed_probe.json
```

## Source-support scout result

The six pressure windows are all still pressure-screen misses, but the adjacent
selector-expanded support scouts are not empty:

- Pressure no-structural windows: `6/6`.
- Support-scout cases scanned: `280`.
- Outside-degenerate selected-support cases: `270/280`.
- Selected-support target-motif candidates: `220/280`.
- Selected-support candidates with direct public-key verification metadata:
  `28`.
- Selected-support candidates with shared-product verification metadata: `9`.
- Priority-column candidates: `196`.

This is only pre-relation evidence: a selected support containing `[8,11]`,
`[9,11]`, `[10,13]`, `[10,14]`, or `[11,13]` is not counted as target descent
unless the accepted relation form also contains useful support.

## Direct certificate replay

The replay manifest had `24` cases across the six windows.  Mounted direct
certificate export was run with local outputs:

- `3064..3071`: `0/4` direct certificates passed.
- `3080..3087`: `3/4` direct certificates passed.
- `3136..3143`: `0/4` direct certificates passed.
- `3168..3175`: `4/4` direct certificates passed.
- `3200..3207`: `0/4` direct certificates passed.
- `3208..3215`: `0/4` direct certificates passed.

The form-support audit over all `24` exported certificates reports:

- Direct-below-rho replay ledger: `24/24`.
- Known-only relation-support certificates: `2`.
- Target-motif relation-support certificates: `2`.
- Priority-column relation-support certificates: `6`.
- Avoid-support certificates: `4`.
- Unique accepted form supports:
  `[11,15]`, `[0,5]`, `[3,5]`, `[10,14]`, `[1,5]`, `[0,2]`, `[2,4]`.

The main positive is `3080..3087`: both `mode_low_term_support_total5` and
`mode_low_term_support_total3` at transfer `3086`, top-k `16`, export accepted
relation support `[10,14]`.  This directly hits one of the known-column target
motifs in a window previously classified as no-structural under the older
pressure family.

## Known-factor screen replay

The existing mounted known-column pressure screen was repointed at the
selector-expanded `3080..3087` source/support artifacts with relaxed
selected-support filters.

For `mode_low_term_support_total5`:

- Selected reports: `4`.
- Structural known-support cases: `1`.
- Public known-factor recoveries: `1`.
- Recovery: transfer `3086`, support `[10,14]`, derived secret `11718`,
  public-key verified.
- Full selected direct cost: `2.98540147x` rho.
- First public hit cost under current relaxed order: `1.51824818x` rho.

For `mode_low_term_support_total3`:

- Selected reports: `4`.
- Structural known-support cases: `1`.
- Public known-factor recoveries: `1`.
- Recovery: transfer `3086`, support `[10,14]`, derived secret `11718`,
  public-key verified.
- Full selected direct cost: `2.89051096x` rho.
- First public hit cost under current relaxed order: `1.45255475x` rho.

## Interpretation

This is a constructive improvement over the stale "no structural form" label:
an adjacent public source family does expose a verified known-factor descent
form in `3080..3087`.  It is not yet below rho as a public stop rule because
the current relaxed order scans transfer `3080` before the useful `3086` row.

The result narrows the next gate.  The source-family expansion can create the
right relation support; now the algorithm needs a public source-order or
RHS/target-compatibility pre-screen that moves `[10,14]`-style rows before
unknown-support rows like `[0,5]` and `[3,5]`, then reruns the same known-factor
screen under source-generation/shared-product charges.

## Non-claims

- This is not a complete ECDLP algorithm or deployed-curve speedup.
- The successful relation is direct-source, not yet shared-product/source
charged.
- The current stop order is above rho on `3080..3087`.
- Selected-leaf motif visibility often does not survive into accepted relation
forms.
- Salt or transfer ordering should be treated as a diagnostic unless frozen
before validation.
