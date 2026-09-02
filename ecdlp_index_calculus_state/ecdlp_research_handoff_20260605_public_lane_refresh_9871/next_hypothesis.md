# Next Hypothesis

## Immediate Validation

Keep the promoted public rule fixed:

`selected_has=13 AND salt_adjacent=False AND salt_min_mod4=3`

Prioritize the remaining sharp-lane full-family transfers:

1. `9728`, salts `163,166`
2. `9755`, salts `175,177`
3. `9790`, salts `167,172`
4. `9814`, salts `163,174`
5. `9839`, salts `163,172`
6. `9715`, salts `171,174`
7. `9840`, salts `163,170`

## Success Criteria

- Minimal: one additional sharp-lane transfer exports rank gain.
- Strong: two or more additional sharp-lane transfers export accepted-missing
  rank gain.
- Best next signal: a new rank-gain transfer past `9871` that matches the same
  sharp rule with no new exported controls.

## FFE/Summation-Polynomial Follow-Up

If the sharp lane continues to validate, route its positive rows into the FFE
assembly path as a public hit-stream candidate.  The candidate stream should
include:

- row keys and salts from the selected13 workorder
- selected term support
- accepted form support from the bridge certificate
- rank-gain and unique-factor gain from the rank scorer

The first assembly test should compare:

1. the full selected13 lane,
2. the broad non-adjacent lane, and
3. the sharp `salt_min_mod4=3` lane.

The required claim boundary for that stage is relation assembly efficiency only
unless it produces relation-derived ECDLP recovery below the Pollard-rho
baseline.
