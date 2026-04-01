# Multilingual OCR Fine-Tuning — Reference

## Background

This task fine-tunes DeepSeek-OCR 3B on mixed Persian + Bengali text images using Unsloth LoRA. The baseline uses correct preprocessing (image_size=640, crop_mode=True, train_on_responses_only=True) but suboptimal training hyperparameters. The agent must optimize the training recipe to minimize CER across both languages.

## Baseline Configuration

- LoRA rank: 16, alpha: 16
- Learning rate: 2e-4 (high for vision models)
- Scheduler: linear (suboptimal for longer runs)
- Max steps: 60 (too few for bilingual data)
- Warmup: 5 steps
- Training data: 1000 samples (500 Persian + 500 Bengali)

Baseline avg CER ≈ 0.60

## Optimization 1: Increase LoRA Rank

- **Change**: r=16 → r=128, lora_alpha=16 → 128
- **Why**: Two languages with different scripts need more model capacity. Higher rank allows the adapter to learn language-specific features without interference.
- **Impact**: ~10-15% CER reduction

## Optimization 2: More Training Steps

- **Change**: max_steps=60 → 700
- **Why**: 60 steps is ~1 epoch over 1000 samples with batch=2×4. 700 steps provides ~11 epochs over bilingual data, allowing the model to thoroughly learn both scripts.
- **Impact**: ~15-20% CER reduction

## Optimization 3: Lower Learning Rate + Cosine Schedule

- **Change**: lr=2e-4 → 1e-4, linear → cosine
- **Why**: 2e-4 is aggressive for vision-language models and can cause instability. Cosine schedule provides better convergence than linear for longer training runs.
- **Impact**: ~5-10% CER reduction, more stable training

## Optimization 4: More Warmup

- **Change**: warmup_steps=5 → 20
- **Why**: Gradual warmup prevents early gradient spikes with mixed-language data.
- **Impact**: Marginal but contributes to training stability

## Reference Solution

Combined changes:
- `r=128, lora_alpha=128` (was 16, 16)
- `max_steps=700` (was 60)
- `learning_rate=1e-4` (was 2e-4)
- `lr_scheduler_type="cosine"` (was "linear")
- `warmup_steps=20` (was 5)

## Source

- DeepSeek-OCR model (MIT license): https://huggingface.co/deepseek-ai/DeepSeek-OCR
- Unsloth fine-tuning framework: https://github.com/unslothai/unsloth
- Bengali OCR data: https://huggingface.co/datasets/rifathridoy/bengali-ocr-synthetic
- Persian OCR data: https://huggingface.co/datasets/hezarai/parsynth-ocr-200k
