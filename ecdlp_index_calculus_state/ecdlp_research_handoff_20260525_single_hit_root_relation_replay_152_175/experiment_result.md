# Experiment Result

Command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 tasks/ecdlp_index_calculus/ffe_single_hit_root_relation_replay_probe.py --out ecdlp_index_calculus_state/ffe_single_hit_root_relation_replay_152_175.json
```

Primary artifact:

- `ecdlp_index_calculus_state/ffe_single_hit_root_relation_replay_152_175.json`

Result:

- 30/30 retained-only source cases publicly re-derived from replayed relation events.
- 0 retained-only replay mismatches against the signature cases.
- 0 materialization/context errors.
- 10/10 same-challenge groups publicly verified.
- Retained-only relation count sum: 60.
- Challenge-group unique relation count sum: 23.
- Challenge-group max rank: 3.
- Retained-only ops/rho: min 0.37956204, mean 0.77921168, max 0.96.

Target breakdown:

- `22050.cf1@11731`: 11/11 retained-only cases verified, relation count sum 22, ops/rho mean 0.59057731, max 0.72992701.
- `67.a1@9803`: 19/19 retained-only cases verified, relation count sum 38, ops/rho mean 0.88842105, max 0.96.

Verifier interpretation:

The retained single-hit-root surfaces now have case-level and same-challenge
group-level relation replay evidence.  The replay reconstructs the selected
row/leaf sets from the original signature JSON, reruns the live relation
scanner, deduplicates relation forms, and calls the public derivation path.

Important correction:

The first replay run used an overly coarse scan cache keyed only by row and
leaf set.  That could reuse events across distinct transfer challenge seeds.
The final script keys the cache by `(challenge_seed, row_key, leaves)` and the
validated artifact above comes from that corrected run.
