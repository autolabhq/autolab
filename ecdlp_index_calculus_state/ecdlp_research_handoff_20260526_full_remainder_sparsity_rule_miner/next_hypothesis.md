# Next Hypothesis

Freeze the public selector before the next replay:

```text
target = 22050.cf1@11731
row_salt = 165
transfer_index mod 6 = 0
```

Rationale: in the 328-375 Sage surface corpus, this selects exactly the two full-remainder below-rho positives and no negatives. The selector is public before full remainder materialization, but its support is tiny and overfit risk is high.

## Holdout

Run a targeted holdout that forces or harvests the selected row envelope for future transfer indices satisfying the frozen residue, for example 378, 384, 390, 396, and 402 if the harvester can expose those surfaces. Do not alter the selector after seeing the holdout rows.

Record for every selected surface:

- whether a preserving Sage factor candidate exists
- full resultant monomials
- full remainder monomials
- known hit root count
- full-remainder FFE ops/rho
- whether the selected candidate preserves all selected root pairs

Primary pass signal:

```text
at least two preserving selected holdout surfaces, with full_remainder_monomials <= 92
and full_remainder_ffe_ops_over_rho < 1.0, without a preserving false-positive gap
```

Primary fail signal:

```text
the selected holdout surfaces are mostly non-preserving, or preserving selected
surfaces stay above rho with full_remainder_monomials >= 106
```

## Controls

Use controls to separate residue structure from row-specific luck:

- same target and row salt with transfer residues that were negative in-sample: 3 and 5 modulo 6
- same target and transfer residue 0 modulo 6, but neighboring salts such as 164, 167, 173, and 174
- `67.a1@9803` rows with the same transfer residue class, if those surfaces are available

## Mechanism Work

The strongest post-materialization signature is:

```text
full_resultant_monomials = 91
full_remainder_monomials = 79
known_hit_root_count = 12
```

A useful next miner should try to predict this collapse before paying for full remainder materialization. Candidate predictors:

- factorization-level `candidate_count=12`
- public row envelope plus transfer residue
- selected leaf index count and original selected root-pair count
- low-degree factor surface shape before full quotient evaluation

If these predictors survive holdout, the algorithmic direction becomes: generate summation-polynomial surfaces, factor to cheap FFE candidates, and materialize full remainders only inside the frozen row-envelope/residue activation cell.

## Claim Boundary

No speedup claim should be made from the current evidence alone. The current result is a preselector hypothesis and a mechanism clue. A promoted result needs a frozen holdout that beats Pollard-rho after preservation checks and full-remainder FFE charging.
