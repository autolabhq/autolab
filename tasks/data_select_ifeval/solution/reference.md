# Data Selection for Instruction Following — Reference

## Background

This task tests the agent's ability to select high-quality training data from a diverse 50k-sample pool to maximize IFEval (instruction-following) performance. The pool contains 19 sources ranging from instruction-following-specific data to math, code, multilingual, and safety data. The training pipeline is fixed — only the data selection algorithm changes. This is a realistic scenario: curating training data is a key skill in post-training.

## Bug 1: Random Data Selection

- **Root cause**: The baseline `select_data.py` randomly samples 5,000 from the 50k pool. Most pool sources (math, code, multilingual) are irrelevant to instruction following.
- **Symptom**: The fine-tuned model's IFEval performance barely changes or degrades from the base model, since random selection includes too much off-task data that dilutes instruction-following capability.
- **Fix**: Source-aware selection that prioritizes IF-relevant sources: `tulu_v3.9_persona_ifeval_turn_1` (explicit format constraints), `coconot_sft_v2.0`, `no_robots_sft`, `wildchat_gpt-4_sft_v2.0` (high-quality instruction data). Filter for English, quality, and reasonable length.
- **Impact**: IFEval prompt_strict improves from ~0.38 (random) to ~0.42 (source-aware selection).

## Reference Solution

**Summary of all changes:**
- Replace random selection with source-priority-based algorithm
- Allocate budgets: persona_ifeval (2500), coconot (1000), no_robots (500), wildchat (500), flan (300), oasst1 (200)
- Apply quality filters: English check, minimum message length, user+assistant turn required
- Fill remaining budget from IF-relevant sources

## Source

- Data pool: allenai/tulu-3-sft-mixture (ODC-BY-1.0)
- IFEval benchmark: "Instruction-Following Evaluation for Large Language Models" (arXiv:2311.07911)
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
