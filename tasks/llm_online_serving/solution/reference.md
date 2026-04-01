# Optimize Online LLM Serving in SimpleLLM — Reference

## Background

SimpleLLM is a ~560-line inference engine for gpt-oss-20b (21B MoE, 3.6B active params per token, MXFP4 quantized). It implements continuous batching with CUDA graph acceleration for decode steps. The baseline has two realistic performance bottlenecks common in production LLM serving systems.

## Bug 1: Lazy CUDA Graph Capture

- **Root cause**: `_ensure_graph_captured()` lazily captures one CUDA graph per batch size on first use. During online serving, batch size varies continuously (1→2→...→64→63→...), triggering ~64 separate capture events during inference.
- **Symptom**: Latency spikes of ~50-200ms each time a new batch size is first encountered. Throughput oscillates and is unstable during the first pass through all batch sizes. Total overhead: several seconds across a full serving run.
- **Fix**: Pre-capture CUDA graphs for all batch sizes 1 through `max_num_seqs` during engine initialization, after the warmup pass.
- **Impact**: Eliminates all lazy-capture latency spikes. Mean completion time reduced ~15-20%. Throughput becomes stable from the first request.

## Bug 2: Conservative Prefill Chunk Budget

- **Root cause**: `MAX_PREFILL_TOKENS = 4096` in `_run_prefill()` limits how many prompt tokens are processed per prefill chunk. On H100 with MXFP4, the GPU can handle larger prefill batches without memory pressure.
- **Symptom**: With short-to-medium prompts (256-2048 tokens), the 4096 budget means only 2-4 prompts are prefilled per chunk. This creates many prefill/decode alternations, each with Python overhead and GPU pipeline bubbles.
- **Fix**: Increase `MAX_PREFILL_TOKENS` to 8192, allowing more prompts to be batched in each prefill pass.
- **Impact**: Reduces number of prefill/decode alternations by ~50%. Throughput increases ~8-12%.

## Reference Solution

1. Pre-capture CUDA graphs for batch sizes 1–64 at engine startup (after warmup)
2. Increase MAX_PREFILL_TOKENS from 4096 to 8192

Combined effect: ~46% throughput improvement, ~44% completion time reduction (measured on H100 with 96 requests at Poisson arrival rate 6 req/s).

## Source

- SimpleLLM repo: https://github.com/naklecha/simple-llm
- CUDA graph lazy vs eager capture is a known tradeoff in vLLM/SGLang
- Prefill chunk sizing discussed in Sarathi-Serve (OSDI 2024)
