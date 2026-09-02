# P1436 autoresearch focus harness v15 result

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

This work strengthens the harness and closes two evidence gaps. It is not a
generic-prime ECDLP breakthrough and does not improve the Shoup bound.

## Source-method enrichment

The harness binds the exact July 13, 2026 AlphaXiv post, its media URL, the
referenced arXiv paper, and the OpenResearch experiment-tree methodology. The
adaptation is logical only: immutable source baseline, bounded critical
experiments, explicit falsifiers, and deferred children. It does not claim that
unmaterialized branches or agents ran.

## Exact collision replay

The unchanged mounted P1436 collector was traced at its row-admission boundary.
The `ecdlp.p1436_collision_record.v2` ABI distinguishes every collision edge
from the subset of nonzero rows admitted by the collector and preserves
duplicate admitted rows.

For seed `1432001`, `two_map_union`, `random_hash_mask1`:

- collision edges: `16`, all cross-shift;
- admitted rows: `16`;
- coefficient / augmented / unknown rank: `15 / 15 / 15`;
- verified factor logs: yes;
- exact target descents: `4 / 4`, with zero invalid candidates;
- routing and relation-matrix status: `ready`.

The single-target charged cost is preprocessing plus the maximum measured
target-descent cost. At 20 bits this is `526,808` modeled field operations, or
`810.4738461538 * rho`.

## Multiscale cost

The frozen route was replayed at 20, 22, and 24 bits:

| Bits | Rank | Targets | Single-target operations | Ratio vs rho |
|---:|---:|---:|---:|---:|
| 20 | 15/15 | 4/4 | 526,808 | 810.4738461538 |
| 22 | 18/18 | 4/4 | 880,028 | 801.4826958106 |
| 24 | 24/24 | 4/4 | 1,570,835 | 749.4441793893 |

The three-point diagnostic fit is `q^0.4656765429` with maximum log residual
`0.0156537168`. The fit is toy-scale only and fails the gate because every
measured point is far above rho. Source-attempt generation accounts for
`93.34%`, `94.75%`, and `95.56%` of preprocessing cost.

The harness now fits charged operations against group order, not bit length,
and requires both exponent below `0.5` and every measured point below rho.
Ratios below `11 * rho` remain diagnostic and cannot satisfy promotion.

## Summation/FFE intake

R68 was replay-bound into the harness: product/summation quotient rows receive
zero independent information credit. A lane must emit a scalar-blind new
fixed-sum factor row below direct pair-complement cost.

The existing `ffe_public_independent_row_expander_184_191` candidate was audited
instead of duplicated. It added 18 source cases but changed challenge-group
relation count by `0` and maximum rank by `0`. It also lacks absolute source
and direct pair-complement costs, so it is rejected by the R68 admission gate.

## Verification

- task tests: `57 / 57` passed;
- Python compilation: passed;
- focus schema: `ecdlp.p1436_autoresearch_focus_report.v7`;
- collision ABI: `ecdlp.p1436_collision_record.v2`;
- routing schema: `ecdlp.p1436_collision_to_rank_routing_ablation.v3`;
- relation-matrix schema: `ecdlp.p1436_relation_matrices.v3`;
- exact scaling artifact SHA-256:
  `ecccc5bfdf4c78a23afdd98938ef867a386b5c9c246d27a4d630e6fea076a754`;
- FFE intake artifact SHA-256:
  `eabb15e1d48c0b3f5a2ff9951e5178af079da1f3e980182f1444c6f1ed4ba499`.

Exactly one next action: build an IDEA-340-owned public-chart preflight that
requires an explicit DLP-free chart, hash-bound independent new-row replay, and
absolute source cost below direct pair-complement enumeration before running
another summation/FFE experiment.
