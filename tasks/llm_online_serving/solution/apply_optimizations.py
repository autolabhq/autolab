"""Reference solution: pre-capture CUDA graphs + increase prefill chunk size.

Optimization 1 — Pre-capture CUDA graphs at startup
  The baseline lazily captures one CUDA graph per batch size on first use.
  Each capture blocks inference for ~50-200ms, causing latency spikes.
  Pre-capturing all common batch sizes during init eliminates these spikes
  and ensures the fast path is always available from the first decode step.

Optimization 2 — Increase MAX_PREFILL_TOKENS
  The default 4096 is conservative.  Increasing to 8192 lets more prompts
  be prefilled per chunk, reducing the number of prefill/decode alternations
  and improving throughput.
"""

path = "/app/llm.py"
with open(path) as f:
    code = f.read()

# --- Optimization 1: Pre-capture CUDA graphs at startup ---
old_init = '''        self.model.clear_all_slots()
        tqdm.write'''

new_init = '''        self.model.clear_all_slots()

        # Pre-capture CUDA graphs for all common batch sizes to eliminate
        # lazy-capture latency spikes during inference.
        for bs in range(1, self.max_num_seqs + 1):
            self._ensure_graph_captured(bs)

        tqdm.write'''

code = code.replace(old_init, new_init, 1)

# --- Optimization 2: Increase prefill chunk size ---
code = code.replace(
    "MAX_PREFILL_TOKENS = 4096",
    "MAX_PREFILL_TOKENS = 8192"
)

with open(path, "w") as f:
    f.write(code)

# Verify
with open(path) as f:
    content = f.read()
assert "Pre-capture CUDA graphs" in content, "Opt 1 not applied"
assert "MAX_PREFILL_TOKENS = 8192" in content, "Opt 2 not applied"

print("Applied 2 optimizations: pre-capture graphs + larger prefill chunks")
