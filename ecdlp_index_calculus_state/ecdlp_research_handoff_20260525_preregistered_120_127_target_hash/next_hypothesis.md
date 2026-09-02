# Next Hypothesis

The next fresh window should isolate the total2 first-fall route instead of
letting total3/4 gate-selected surfaces decide the promotion.

Concrete proposal for 128-135:

- Preregister a total2-only root-hyperplane policy test.
- Primary policy: `target_root_hash`.
- Declared total2 control: `summax_low_root_norm_target_hash`, because it was
  posthoc best on the 120-127 total2 selected surfaces.
- Keep total3/4 as a separate negative-control channel unless a new public
  pre-factor gate can reject non-preserving surfaces like the 120-127 salt174
  failure before Sage factorization.

Reason:

- On 120-127, total2 produced two gate-selected 67.a1 surfaces.
- Both total2 surfaces had preserving Sage factors and root-scan below rho.
- `target_root_hash` preserved both total2 surfaces with zero false positives
  and scan costs 0.672 and 0.808 ops/rho.
- The all-channel claim failed only because total3/4 selected a surface with no
  preserving factor candidate.

Parallel diagnostic work:

- Add a public pre-factor channel discriminator for total3/4 surfaces. The
  immediate negative example is:
  `22050.cf1@11731|22050.cf1@11731:uniform:256:salt174|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:127:22050.cf1@11731`.
- Look for pre-factor fields that distinguish total2 preserving surfaces from
  total3/4 non-preserving surfaces without using Sage factor outcomes.
- Keep rejecting full-remainder claims; all 120-127 full remainders remained
  above rho.

This makes the next falsifiable claim sharper: total2-only public root-hash
ordering may now be a stable first-fall component, while total3/4 needs a
separate gate rather than promotion by association.
