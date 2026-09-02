# Next Hypothesis

The next useful experiment is a public source-order/RHS-compatibility screen
for the selector-expanded `mode_low_term_support_total5` and
`mode_low_term_support_total3` branches.

## Hypothesis

The `3080..3087` recovery shows that the expanded source family can emit an
accepted relation form on known support `[10,14]`.  A public order or
pre-screen can place such rows before unknown-support rows without reading
known-factor verification labels.

## Null hypothesis

The useful `3086` row is only identifiable by replay labels or brittle
salt/transfer residue correlations.  Any public order that finds it early on
calibration either misses future motif rows or admits unknown-support rows with
above-rho scan cost.

## Immediate command path

Replay the expanded known-factor screen on a fresh validation slice only after
freezing a public ordering rule:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_known_column_pressure_direct_screen.py \
  --source <selector-expanded-source.json> \
  --support <selector-expanded-support-scout.json> \
  --training-audit ecdlp_index_calculus_state/low_term_total2_certificate_bank_linear_descent_audit_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_cumulative_2400_2488_probe.json \
  --selector mode_low_term_support_total3 \
  --forbid-selected-columns '' \
  --max-selected-support-size 16 \
  --min-pressure-score -100 \
  --out <validation-known-factor-screen.json>
```

## Promotion evidence

- A frozen public order stops before rho on at least one fresh selector-expanded
  window.
- The stopped relation form has support fully inside known columns or adds a
  priority factor-rank column.
- Public-key verification succeeds after known-factor substitution.
- Unknown-support rows such as `[0,5]`, `[3,5]`, `[2,4]`, and `[1,5]` are
  rejected or delayed without using verifier outcome labels.

## Demotion evidence

- The ordering rule is just an exact transfer/salt lookup.
- The rule moves the useful row earlier on `3080..3087` but fails on fresh
  windows.
- Direct-source evidence cannot be converted into source-charged or
  shared-product accounting below rho.

## Structural target

Focus on relation-form support, not selected-leaf support.  The accepted forms
from the support-diversity replay show that selected leaves can contain all
target motifs while the final forms collapse to unrelated supports.  The next
screen should predict or construct accepted form motifs, especially `[10,14]`,
before direct verification.
