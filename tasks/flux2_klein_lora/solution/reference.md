# FLUX.2 klein LoRA — Reference

## Background

`/app/train.sh` and `/app/dataset.toml` ship with four correlated defects: an
OOM-inducing memory configuration, a wrong DiT model_version tag, an unstable
training schedule, and an undersized LoRA combined with the legacy timestep
sampling. All four must be addressed to reach the reference composite-quality
score (CLIP-I × 0.40 + DINO × 0.35 + CLIP-T × 0.25). Each defect is a realistic
mistake that surfaces in different ways: hard crash, silent quality regression,
overfitting, and persistent low metrics respectively.

## Bug 1: OOM at 1024px / batch=2 with no activation offloading

- **Root cause**: `dataset.toml` `resolution=1024` + `batch_size=2`, combined
  with no `--gradient_checkpointing` and no `--blocks_to_swap` in `train.sh`,
  exceeds the 48 GB L40S budget when the 9B DiT runs in bf16 with training
  activations.
- **Symptom**: CUDA out-of-memory crash during step 3/3 of `train.sh`. No
  `lora.safetensors` is written.
- **Fix**: in `train.sh` add `--gradient_checkpointing`, `--blocks_to_swap 8`,
  `--gradient_accumulation_steps 4`, `--vae_dtype bfloat16`; in `dataset.toml`
  set `resolution=512` and `batch_size=1`.
- **Impact**: without this, training never starts → reward = 0.

## Bug 2: Wrong DiT model_version

- **Root cause**: `--model_version klein-9b` selects the distilled klein
  variant, but the weights at `/models/dit/flux2-klein-base-9b.safetensors`
  are the undistilled `klein-base-9b` and use a different noise schedule.
- **Symptom**: training runs to completion but generated images do not track
  the concept; CLIP-I and DINO plateau near the no-LoRA baseline.
- **Fix**: `--model_version klein-base-9b` in all three musubi-tuner
  invocations (`flux_2_cache_latents.py`, `flux_2_cache_text_encoder_outputs.py`,
  `flux_2_train_network.py`).
- **Impact**: composite quality stays at no-LoRA baseline (~0.30) without this.

## Bug 3: Overfitting training schedule

- **Root cause**: `--learning_rate 1e-3` with `--max_train_epochs 50` on a
  constant scheduler memorizes 15 images × 10 repeats.
- **Symptom**: outputs reproduce training images verbatim or collapse to a few
  modes; CLIP-T (text alignment) stays poor.
- **Fix**: `--learning_rate 1e-4`, `--max_train_epochs 16`,
  `--lr_scheduler cosine_with_restarts`, `--lr_warmup_steps 50`. Bump
  `num_repeats=20` in `dataset.toml` to keep total step count reasonable.
- **Impact**: ~0.10 composite gain over a stable but overfit baseline.

## Bug 4: Undersized LoRA + legacy timestep sampling

- **Root cause**: `--network_dim 2 --network_alpha 1` is too small to capture
  the concept; `--timestep_sampling shift` is the legacy FLUX.1 schedule that
  does not match FLUX.2's flow-matching prior.
- **Symptom**: CLIP-I plateaus even after the other three bugs are fixed.
- **Fix**: `--network_dim 32 --network_alpha 16`,
  `--timestep_sampling flux2_shift`.
- **Impact**: final composite gain to reach reference (~0.62 on L40S).

## Reference Solution

Applied together by `solve.sh`:

- OOM fix: `--gradient_checkpointing`, `--blocks_to_swap 8`,
  `--train_batch_size 1`, `--gradient_accumulation_steps 4`, `resolution=512`,
  `--vae_dtype bfloat16`.
- `--model_version klein-base-9b` (all three steps).
- `--learning_rate 1e-4`, `--max_train_epochs 16`,
  `--lr_scheduler cosine_with_restarts`, `--lr_warmup_steps 50`,
  `num_repeats=20`.
- `--network_dim 32 --network_alpha 16`, `--timestep_sampling flux2_shift`.

Baseline: `composite_quality = 0.0` (OOM, no LoRA produced) → Reference:
`composite_quality = 0.62`. See `task.toml` for the calibrated values used by
the reward function — `task.toml` is the single source of truth.

## Alternative approaches

- Smaller `network_dim` (e.g. 16) with more epochs can match the reference on
  small-concept datasets.
- `--optimizer_type prodigy` or `--optimizer_type adamw` (instead of
  `adamw8bit`) at higher LR can speed up convergence at some memory cost.
- Different `timestep_sampling` weighting (e.g. `sigmoid`) occasionally helps,
  but `flux2_shift` is the FLUX.2-native default.

## Calibration note

**nop reward = 0 by design.** This is a from-scratch training task: harbor's
nop agent skips the agent phase entirely, so `train.sh` never runs, no
`/app/output/lora.safetensors` is written, and the verifier fails with
`no_lora_file` → reward = 0. The only calibration gate for this task is
`-a oracle ≈ 1.0`.

## Source

- kohya-ss / musubi-tuner — FLUX.2 training pipeline, MIT.
- Training concept from `lambdalabs/naruto-blip-captions` (HF dataset).
- Task authored by AutoPhD Gym, 2026-03-18.
