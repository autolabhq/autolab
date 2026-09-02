# Challenge Seed Stability Matrix

Scope: row, scout, and filter schedules fixed to
`ecdlp-frontier-signed-dual-sieve-v1`; only the shared challenge seed changes.
Each run uses one 8-salt window, public same-signature scoring before pair
verification, and verifier labels only after the public window score.

| challenge seed | window | positive windows | all-target positives | best verifier ops/rho | targets with events | targets with same-signature pairs | same-signature pair count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 228-235 | 1 | 1 | 0.78832117 | 2 | 2 | 2 |
| baseline | 250-257 | 1 | 1 | 0.64 | 2 | 2 | 6 |
| baseline | 282-289 | 1 | 1 | 0.624 | 2 | 2 | 7 |
| fresh-transfer-a | 228-235 | 0 | 0 | null | 1 | 1 | 28 |
| fresh-transfer-a | 250-257 | 0 | 0 | null | 1 | 1 | 28 |
| fresh-transfer-a | 282-289 | 0 | 0 | null | 1 | 1 | 28 |
| fresh-transfer-b | 228-235 | 0 | 0 | null | 0 | 0 | 0 |
| fresh-transfer-b | 250-257 | 0 | 0 | null | 0 | 0 | 0 |
| fresh-transfer-b | 282-289 | 0 | 0 | null | 0 | 0 | 0 |
| fresh-transfer-c | 228-235 | 0 | 0 | null | 1 | 1 | 56 |
| fresh-transfer-c | 250-257 | 0 | 0 | null | 1 | 1 | 56 |
| fresh-transfer-c | 282-289 | 0 | 0 | null | 1 | 1 | 56 |

For fresh-transfer-a and fresh-transfer-c, the surviving public events were on
`22050.cf1@11731`; `67.a1@9803` had no event stream. Fresh-transfer-b produced
no public events on either target.

Interpretation:

The original same-signature selector is not merely overfit to a salt window;
it is coupled to the shared challenge seed. Fresh challenge seeds can preserve
a dense one-target public signature family, but the cross-target alignment
needed for compact two-equation witnesses disappears.

Next useful probe:

Compare the original two-target signatures against the one-target fresh
signatures by degree partition, leaf support, signed coefficient pattern, and
selected leaf rank. The target is a public normal form that predicts whether a
signature family can survive both targets before verifier-backed derivation.
