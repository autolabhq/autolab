# Compute-Optimal Language Model Pretraining — Reference

## Background

This task asks the agent to pretrain a language model from scratch on WikiText-103 using LitGPT, minimizing test-set perplexity within a 6-hour compute budget on a single H100 GPU. The baseline provides a functional but deliberately suboptimal configuration (tiny model, float32, no LR schedule). The challenge is finding the compute-optimal balance between model size, training efficiency, and hyperparameters — a practical application of scaling laws.

There are no bugs to fix. This is a pure optimization task.

## Optimization 1: Model Architecture

- **Baseline**: pythia-14m style (n_embd=128, n_layer=6, n_head=4, GptNeoxMLP, LayerNorm, partial RoPE, bias=True) — ~14M params
- **Reference**: Llama-style (n_embd=768, n_layer=16, n_head=12, LLaMAMLP/SwiGLU, RMSNorm, full RoPE, GQA n_query_groups=4, no bias) — ~179M params
- **Impact**: Dramatically more model capacity. Llama-style architecture is more parameter-efficient than Pythia-style due to SwiGLU, RMSNorm, and full RoPE.

## Optimization 2: Training Efficiency

- **Baseline**: float32, no torch.compile
- **Reference**: bfloat16 + torch.compile(mode="max-autotune")
- **Impact**: ~2-3x throughput improvement. Same wall-clock time processes 2-3x more tokens, enabling more training for the larger model.

## Optimization 3: Hyperparameters

- **Baseline**: constant LR 1e-3, no warmup, no weight decay, betas=(0.9, 0.999), seq_length=256, batch_size=16
- **Reference**: cosine LR 3e-4 with 500-step warmup, weight_decay=0.1, betas=(0.9, 0.95), seq_length=1024, batch_size=32, gradient_accumulation_steps=2
- **Impact**: Cosine schedule with warmup gives much better convergence. seq_length=1024 matches eval and captures long-range dependencies. Larger effective batch (32×2=64) improves gradient estimates.

## Optimization 4: Sequence Length Alignment

- **Baseline**: trains with seq_length=256, but verifier evaluates at seq_length=1024 (block_size=512 < 1024 causes fallback)
- **Reference**: block_size=1024, seq_length=1024 — matches evaluation exactly
- **Impact**: Eliminates train/eval distribution mismatch. With baseline block_size=512, verifier can only evaluate at 512 tokens, hurting perplexity.

## Reference Solution

**Summary of all changes:**
- Architecture: 14M pythia → 179M Llama-style (RMSNorm, SwiGLU, GQA, full RoPE)
- Precision: float32 → bfloat16 + torch.compile
- LR schedule: constant 1e-3 → cosine 3e-4 with 500-step warmup
- Regularization: no weight decay → 0.1 weight decay, betas (0.9, 0.95)
- Data: seq_length 256 → 1024, batch 16 → 32×2 grad accum
- Training: 25,000 steps (~2.5h on H100)

## Source

- Chinchilla scaling laws: Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)
- LitGPT: https://github.com/Lightning-AI/litgpt (Apache-2.0)
- Reference solution derived from agent run, verified
