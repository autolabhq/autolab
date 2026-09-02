# Signature Inspection: 250-257

Artifacts:

- `ecdlp_index_calculus_state/future_witness_signature_inspect_baseline_250_257.json`
- `ecdlp_index_calculus_state/future_witness_signature_inspect_fresh_a_250_257.json`

Both runs keep the same row window. The fresh run freezes row, scout, and filter
schedules to baseline and changes only the shared challenge seed to
`ecdlp-frontier-signed-dual-sieve-v1-fresh-transfer-a`.

Summary:

| run | positives | all-target positives | best verifier ops/rho | targets with events | targets with same-signature pairs | same-signature pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 1 | 0.64 | 2 | 2 | 6 |
| fresh-transfer-a challenge-only | 0 | 0 | null | 1 | 1 | 28 |

Baseline public signature shape:

- `22050.cf1@11731` produced 7 events and 4 same-signature pairs. Its repeated
  signatures were mostly `2+2` term shapes, including
  `[56, [11, 11, 13, 13], "2+2"]` and
  `[89, [13, 13, 14, 14], "2+2"]`.
- `67.a1@9803` produced 6 events and 2 same-signature pairs. Its repeated
  signatures used `2+1+1` term shapes, including
  `[3, [1, 7, 7, 12], "2+1+1"]`.

Fresh-transfer-a challenge-only shape:

- `22050.cf1@11731` produced 8 events and 28 same-signature pairs, all flowing
  through the repeated public signature `[56, [10, 10, 13, 13], "2+2"]`.
- `67.a1@9803` produced no events.

Interpretation:

The fresh seed can increase same-target repetition while destroying
cross-target alignment. A successful next selector needs to distinguish
"dense one-target repetition" from "two-target algebraic compatibility" before
verifier-backed pair derivation.
