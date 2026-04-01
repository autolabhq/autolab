# Reference: Multi-Source GRPO Visual Math Reasoning

## Background

This task trains Qwen2.5-VL-7B with GRPO on visual math problems. The baseline is
**correct** (no bugs) but **suboptimal**: it only uses Geometry3K data (~2400 examples)
with moderate hyperparameters. Performance gains come from multi-source data strategy,
hyperparameter tuning, and image resolution optimization.

## Optimization 1: Multi-Source Training Data

**Baseline**: Only `/data/geometry3k/` (~2400 examples)

**Reference**: Geometry3K + MathVision + ChartQA with upsampling (geometry3k 2x, mathvision 2x, chartqa 1x)

- **MathVision** (~2000 examples): Competition-level visual math covers algebraic,
  geometric, and graph-based problems.
- **ChartQA** (~1500 examples): Chart/graph understanding. Included with lower weight
  to maintain VQA retention without diluting math signal.

**Impact**: More diverse RL signal, broader coverage of MathVista problem types.

## Optimization 2: Learning Rate

**Baseline**: `2e-6`

**Reference**: `3e-6`

**Impact**: Faster convergence within the training budget.

## Optimization 3: Number of Generations

**Baseline**: `num_generations=3`

**Reference**: `num_generations=4`

**Impact**: Better advantage estimation via larger group comparison.

## Optimization 4: LoRA Configuration

**Baseline**: `r=12, lora_alpha=12`

**Reference**: `r=16, lora_alpha=32` (alpha/r ratio = 2.0)

**Impact**: Higher rank + increased alpha scaling for stronger task adaptation.

## Optimization 5: Training Duration

**Baseline**: `0.4 epochs, max_steps=100`

**Reference**: `1.0 epochs, max_steps=300`

**Impact**: 3x more training steps with the expanded multi-source dataset.

## Optimization 6: Image Resolution

**Baseline**: `max_pixels=768*768`, `min_pixels=192*192`

**Reference**: `max_pixels=896*896`, `min_pixels=224*224`

**Impact**: Higher resolution preserves mathematical diagrams and geometric figures.

## Optimization 7: Gradient Accumulation

**Baseline**: `gradient_accumulation_steps=1`

**Reference**: `gradient_accumulation_steps=2`

**Impact**: Larger effective batch size for more stable RL gradients.

## Results

| Config | MathVista Accuracy | Reward |
|--------|-------------------|--------|
| Baseline (Geometry3K only) | ~0.20 | ~0.36 |
| Reference (all optimizations) | ~0.56 | ~1.00 |

## Source

- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- Unsloth GRPO: https://github.com/unslothai/unsloth
- MathVista: https://mathvista.github.io/
