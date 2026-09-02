# Experiment Result

Added and ran:

`tasks/ecdlp_index_calculus/ffe_public_quotient_route_ensemble_probe.py`

Primary artifact:

`ecdlp_index_calculus_state/ffe_public_quotient_route_ensemble_probe.json`

Inputs:

- `ecdlp_index_calculus_state/ffe_support_unique_diag.json`
- `ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_predictor_probe.json`

The diagnostic support rerun had to be launched from `/Volumes/Volume/autolab`
because several source paths are relative to the live AutoLab state. The
resulting support artifact includes the newer miss diagnostics.

Summary:

- Surface universe: 18 FFE quotient surfaces
- Direct public factor route below rho: 9 unique surfaces
- Cross-validated fingerprint leaf-locator route below rho: 8 unique surfaces
- Public route union below rho: 17 unique surfaces
- Selected preserving routes: 17
- Selected false positives: 0
- Mean selected ops/rho: 0.81043195
- Best selected ops/rho: 0.50364964
- Worst selected ops/rho: 0.99270073

By target:

- `22050.cf1@11731`: 9/10 surfaces selected below rho, 0 false positives
- `67.a1@9803`: 8/8 surfaces selected below rho, 0 false positives

The single missed surface is:

`22050.cf1@11731|22050.cf1@11731:uniform:256:salt164|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:57:22050.cf1@11731`

Its direct public factor route preserves the selected root pairs, but costs
`ops/rho = 1.91240876`. The fingerprint support route sees a preserving
candidate at `ops/rho = 1.06569343`, still just above rho.

Interpretation:

This is a stronger public selector result than either component route alone.
The route decision is made by public estimated cost before auditing
`preserves_selected_root_pairs` and `same_selected_root_pairs`.

This is still not a full ECDLP index-calculus break. It covers pre-existing
FFE quotient surfaces from the low-term total-2 branch and still relies on the
surface-generation stack. The next required validation is a pre-registered
future-surface run: train/calibrate the fingerprint leaf-locator on earlier
transfers, freeze the direct-factor route, and test the public route ensemble
on newly generated transfer surfaces.
