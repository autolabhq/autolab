# P1436 Autoresearch Focus Harness V18 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## Public pre-surface full-charge preflight

Artifact:
`p1436_presurface_full_charge_preflight.json`

SHA-256:
`0956ee00f4dac2cd1ed6847f266b06c5f5a1b8d6e080334ff3439941953c337b`

Classification: `PRESURFACE_PUBLIC_COMPONENT_ONLY`.

Three of eight obligations pass:

1. the proposal generator is label-sequestered and has complete observed recall;
2. all 356 public proposals are materialized by the Sage backfill; and
3. the materialized factor guard has complete strict recall with no false positives.

Five obligations fail:

1. only two target families are represented;
2. the best aggregate full-recall evaluated-factor proxy is `1.00205556` times rho;
3. the corresponding actual selector ledger is `1.86975209` times rho;
4. the proxy omits full Sage factorization cost; and
5. no fresh independent fixed-sum rank delta is recorded.

The older 232-375 corpus remains a narrow proxy near miss: its evaluated-factor
proxy is exactly two operations above the aggregate rho denominator, while its
selector ledger is 13,298 operations above that denominator. It is not a
speedup.

## Factor-line equivalence and source-ledger deduplication

Artifact:
`p1436_factor_line_direct_root_equivalence_probe.json`

SHA-256:
`0ecd23ae4b4088ad2ad179c3db78fe51b81f20b64b8801c9451b57b8d235ceb3`

Across 17 retained Sage windows and 103 surfaces, all 2,097 retained resultant
factors have the normalized form

`c + r*b + r^2`.

For all 2,097 factors, testing a zero at a monic leaf point `(b,c)` is exactly
equivalent to testing whether `r` is a root of `x^2 + b*x + c`.

The direct public quadratic replay preserves all existing selected pairs on all
356 proposal surfaces. It exactly matches 341 surfaces; the remaining 15 expose
30 extra polynomial-root matches.

Full verification accepts 20 relation forms from those extra matches. Considered
alone, they have rank 12 across the two targets. After deduplication against the
complete 114-form source ledger, however:

- fresh relation forms: 0;
- rank before: 21;
- rank after: 21; and
- fresh rank delta: 0.

One historical window, `fresh_904_911_expanded`, references a QR/Sage artifact
that is no longer present on the mounted volume. The 356-proposal direct replay
is reconstructed independently from the authoritative row sources; the
factor-line identity count is explicitly limited to the 17 retained windows.

## Harness routing

Artifact:
`p1436_autoresearch_focus_report_seed1432001_exact_v18.json`

SHA-256:
`f9345beb0639014b15817f27cb941059d41a541e9f6ade22711f0d6b8f0e42b0`

Schema: `ecdlp.p1436_autoresearch_focus_report.v10`.

The factor-selector lane is hash-bound and closed at 2 of 6 source obligations.
The single top action is now `scalar_blind_fixed_sum_source_generator`: any new
candidate must act before selected-leaf materialization, emit previously unseen
verified relation rows, add exact modular rank after full-ledger deduplication,
and beat direct pair-complement and rho costs on at least four target families.

Promotion remains withheld.

## Verification

- Full ECDLP task suite: 75 tests passed.
- Sage replay: 356 proposals reconstructed and verified.
- Python compilation: passed.
- `git diff --check`: passed.
