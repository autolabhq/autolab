# Next Hypothesis

## Hypothesis

A public pre-setup objective that targets missing factor columns can move the
frontier more efficiently than continuing the current duplicate-hit branch band
unchanged.

The branch-family bank is saturated on supports using columns
`[1,2,3,4,5,11]`, while the latest direct-source bridge shows accepted forms on
`[0,5]` and `[3,5]` with a rank-gain score under the broader factor-rank
candidate scorer.  The next scout should target uncovered or form-only columns
before paying full branch setup, especially order-`11779` column `15` and the
uncovered branch-bank columns `[0,6,7,8,9,10,12,13,14,15]`.

## Null hypothesis

The direct-source rank gain is not transferable into a public branch/source
schedule.  It may be a direct-only certificate that does not improve
source-charged relation supply, does not map fresh targets into the bank, or
requires outcome labels to identify.

## Immediate command path

Refresh the durable frontier after each new source slice:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_branch_family_frontier_rollup.py \
  --state-dir /Volumes/Volume/autolab/ecdlp_index_calculus_state \
  --start-min 5480 \
  --out ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_<end>_probe.json
```

Then run the next scout against fresh selector-expanded source artifacts with
this objective:

```text
Prefer candidates whose accepted relation forms touch uncovered branch-bank
columns or the target-column work-order columns before branch setup.
Reject or delay rows whose target-eliminated supports are already in the
duplicate-hit span: [1,2,4], [1,5,11], [3,5], [2,4], [1,3,5], [1,2,5].
```

## Promotion evidence

- A frozen public rule selects a fresh branch/source candidate with marginal
  factor-rank gain before full branch setup.
- The selected relation touches an uncovered branch-bank column or column `15`
  under order `11779`.
- Public-key verification succeeds and the full accounting remains below rho.
- A fresh public target maps into the accumulated factor bank, not only the
  original local target.

## Demotion evidence

- New branch-family hits keep repeating the six existing factor supports.
- Missing-column selection only works after reading verifier outcomes.
- Direct-source rank-gain candidates cannot be source-charged or converted into
  reusable factor-rank progress.
- Fresh-target mapping remains negative after the direct bridge is added.
