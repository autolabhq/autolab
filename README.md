# Autolab

AutoLab is a benchmark for evaluating AI agents on frontier research tasks. It presents 23 open-ended challenges spanning low-level systems optimization and GPU-intensive model training, each with a real codebase, a compute budget, and a goal to optimize. Instead of measuring one-shot correctness, AutoLab tests what matters in practice: the ability to diagnose bottlenecks, formulate hypotheses, run experiments, and iteratively improve under realistic constraints.

## Environment Setup

```bash
uv pip install harbor
bash harbor_patch.sh  # enable gpu and extended waiting
```

> ⚠️ Some tasks require GPU access. You need either local Docker environment with GPU support or a cloud sandbox such as [Modal](https://modal.com/).

## Overview

AutoLab consists of 23 scored tasks spanning two categories:

## Task Lists (Initial Release)

### Model Development (5 tasks)

- **scaling_law** — Train a GPT model from scratch on WikiText-103 to minimize perplexity within a fixed compute budget. (Python/LitGPT)
- **grpo_multisource** — Fine-tune Qwen2.5-VL-7B with GRPO on multi-source visual math data to maximize MathVista accuracy. (Python/Unsloth)
- **data_select_ifeval** — Select the best 5,000 samples from a 50k pool to maximize instruction-following accuracy after LoRA fine-tuning. (Python/LlamaFactory)
- **multilingual_ocr** — Fine-tune DeepSeek-OCR-3B with LoRA to minimize character error rate on Persian and Bengali text. (Python/Unsloth)
- **llm_online_serving** — Optimize LLM online serving throughput and latency. (Python)

### System Optimization (18 tasks)

- **aes128_ctr** — Encrypt 256 MiB using AES-128-CTR as fast as possible. (C)
- **sha256_throughput** — Hash a 512 MiB buffer with SHA-256 as fast as possible. (C)
- **gaussian_blur** — Apply a 17×17 Gaussian kernel to a 4K image (×5 passes) as fast as possible. (C)
- **flash_attention** — Compute scaled dot-product attention (n=4096, d=64) with cache-efficient tiling. (C)
- **hash_join** — Inner equi-join two tables (20k × 5M rows) as fast as possible. (C)
- **radix_sort** — Sort 50M random uint32s as fast as possible. (C)
- **vliw_scheduler** — Pack a sequential instruction stream into VLIW bundles to minimize cycle count. (C)
- **bvh_raytracer** — Build a BVH and accelerate ray-triangle intersection for a 638×638 scene. (C++)
- **bm25_search_go** — Execute BM25 ranking queries on a synthetic corpus as fast as possible. (Go)
- **concurrent_kv_wal** — Optimize a WAL-backed key-value store with 4 concurrent goroutines. (Go)
- **fft_rust** — Compute DFT of a 32768-point real signal as fast as possible. (Rust)
- **regex_engine** — Compile regex patterns and search 100k haystacks as fast as possible. (Rust)
- **sstable_compaction_rs** — Optimize LSM-style SSTable compaction merging sorted runs. (Rust)
- **discover_sorting** — Find a 16-input sorting network with the fewest comparators. (Python)
- **fredkin_sort_network** — Build a 4-input stable sorting network using reversible gates with minimal gate count. (Python)
- **stack_machine_golf** — Compute a 256-element dot product minimizing executed instruction count on a stack machine. (Assembly)
- **toy_isa_opt** — Compute a 512-element dot product minimizing simulated cycle count on a toy ISA. (Assembly)
- **smallest_game_player** — Train the smallest neural network (by parameter count) that achieves ≥95% accuracy on perfect-play Connect-3. (Python/PyTorch)

## Task Structure

Each task gives the agent:

- A working but unoptimized program (not a blank slate but a realistic starting point)
- A fixed compute budget and timeout (1–12 hours depending on task complexity)
- A target metric (throughput, latency, perplexity, accuracy, parameter count, etc.)
- A sandbox (with a H100 or L40S GPU if it requires model training or inference)

The agent must diagnose bottlenecks, propose interventions, execute them, measure results, and iterate, exactly as a human researcher would!

Each task is a self-contained environment with Harbor-compatible format:

```
tasks/<task_name>/
├── task.toml              # Metadata, scoring, resource limits
├── instruction.md         # Agent-facing problem description
├── environment/
│   ├── Dockerfile         # Reproducible container (pinned deps, offline data)
│   └── <editable files>   # The code the agent must optimize (e.g. solve.c, train.py)
├── solution/              # Private — invisible to the agent
│   ├── solve.sh           # Reference solution script
│   └── reference.md       # Full explanation of each optimization opportunity
└── tests/
    ├── test.sh            # Runs benchmark, computes reward, writes reward.json
    └── <verifier scripts> # Correctness checks
```

A typical task instruction:

```markdown
# Multi-Source Visual Math Reasoning via GRPO

Fine-tune Qwen2.5-VL-7B with GRPO to maximize accuracy on MathVista visual math problems.

## Setup

| Item | Path / Value |
|------|-------------|
| Training script | `/app/train.py` (editable) |
| Reward functions | `/app/rewards.py` (editable) |
| Training entrypoint | `bash /app/train.sh` |
| Local eval | `python3 /app/evaluate_local.py` |
| Base model | `/models/Qwen2.5-VL-7B-Instruct-bnb-4bit` (4-bit quantized) |

## Training Data Sources

| Dataset | Path | Size | Content |
|---------|------|------|---------|
| Geometry3K | `/data/geometry3k/` | ~2400 | Geometric reasoning with diagrams |
| MathVision | `/data/mathvision/` | ~2000 | Competition-level visual math |
| ChartQA | `/data/chartqa/` | ~1500 | Chart/graph understanding |

All datasets use a unified format: `question`, `answer`, `image`.

## Your Goal

Maximize `mathvista_accuracy` on 100 held-out MathVista problems. A retention gate applies: if general VQA accuracy drops more than 10% relative to the base model, the score is zero.

## Evaluation
python3 /app/evaluate_local.py   # quick check (20 MathVista + 10 VQA)

Reward = `your_score / reference_score`. Matching the reference scores 1.0.

## Rules

- Edit only `/app/train.py` and `/app/rewards.py`
- Do NOT modify `/orig/`, or `/models/`
- LoRA adapter must be saved to `/app/output/`
- No external network access
- Single L40S GPU (48GB)
- Time budget: 8 hours
```

## Scoring

`task.toml` defines a baseline and a human-written reference solution. The two task categories use different reward formulas:

**System Optimization tasks:** Log-scaled speedup, capped at 1.0:

```
speedup = baseline / agent_score
reward  = min(1.0, 0.5 × log(speedup) / log(ref_speedup))
```

**Machine Learning tasks:** Simple multiplier, uncapped:

```
reward  = agent_score / reference   (higher-is-better)
reward  = reference / agent_score   (lower-is-better)
```

## Usage

### Single Task

```bash
harbor run -p [task folder] -a terminus-2 -m [Your Model Name]
```

### All Tasks

```bash
harbor run -p ./tasks -a terminus-2 -m [Your Model Name]
```

## Citation

If you use AutoLab in your research, please cite us:

```bibtex
@misc{autolab-2026,
  title   = {AutoLab: Can Models Begin to Participate
             in the Loops That Drive Scientific
             and Engineering Progress?},
  author  = {AutoLab Team},
  year    = {2026},
  url     = {https://github.com/autolabhq/autolab}
}
```
