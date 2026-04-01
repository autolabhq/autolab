#!/bin/bash
set -e

# =============================================================================
# Verifier: Evaluate checkpoint on WikiText-103 test set
# Scoring: multiplier (direction=lower → reward = reference / score)
# =============================================================================

CHECKPOINT_PATH="/app/output/checkpoint.pt"
LOG_DIR="/logs/verifier"
mkdir -p "$LOG_DIR"

python3 - <<'VERIFIER_SCRIPT'
import math
import json
import sys
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial

from litgpt import GPT, Config


class TokenDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.n_chunks = len(data) // (seq_length + 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * (self.seq_length + 1)
        return self.data[start : start + self.seq_length + 1]


def main():
    checkpoint_path = "/app/output/checkpoint.pt"
    log_dir = "/logs/verifier"

    BASELINE_PPL = 95.0
    REFERENCE_PPL = 23.0
    EVAL_SEQ_LENGTH = 1024
    EVAL_BATCH_SIZE = 8

    # --- Check checkpoint exists ---
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        result = {
            "reward": 0.0,
            "error": "Checkpoint not found",
        }
        with open(os.path.join(log_dir, "reward.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(log_dir, "reward.txt"), "w") as f:
            f.write("0.0\n")
        return

    # --- Load checkpoint ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]

    print(f"Model config: n_embd={model_config.get('n_embd')}, "
          f"n_layer={model_config.get('n_layer')}, "
          f"n_head={model_config.get('n_head')}, "
          f"block_size={model_config.get('block_size')}")
    print(f"Training step: {ckpt.get('step', '?')}")
    print(f"Saved val PPL: {ckpt.get('val_ppl', '?')}")
    print(f"Parameters: {ckpt.get('n_params', '?')}")

    # --- Check block_size ---
    block_size = model_config.get("block_size", 512)
    if block_size < EVAL_SEQ_LENGTH:
        print(f"WARNING: block_size ({block_size}) < eval_seq_length ({EVAL_SEQ_LENGTH})")
        print(f"Evaluating with seq_length={block_size} instead")
        EVAL_SEQ_LENGTH = block_size

    # --- Reconstruct model ---
    config = Config(**model_config)
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # --- Load test data ---
    print(f"\nEvaluating on test set with seq_length={EVAL_SEQ_LENGTH}")
    test_data = torch.load("/data/wikitext103/test.pt", weights_only=True)
    test_dataset = TokenDataset(test_data, EVAL_SEQ_LENGTH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )

    # --- Evaluate ---
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    print(f"\nTest loss: {avg_loss:.4f}")
    print(f"Test perplexity: {ppl:.2f}")
    print(f"Tokens evaluated: {total_tokens:,}")

    # --- Compute reward (multiplier: direction=lower → reference / score) ---
    reward = REFERENCE_PPL / ppl if ppl > 0 else 0.0

    print(f"\nScoring:")
    print(f"  Baseline PPL:  {BASELINE_PPL}")
    print(f"  Reference PPL: {REFERENCE_PPL}")
    print(f"  Achieved PPL:  {ppl:.2f}")
    print(f"  Reward:        {reward:.4f}")

    # --- Save results ---
    result = {
        "reward": reward,
        "perplexity": ppl,
        "test_loss": avg_loss,
        "tokens_evaluated": total_tokens,
        "eval_seq_length": EVAL_SEQ_LENGTH,
        "n_params": n_params,
        "model_config": {
            "n_embd": model_config.get("n_embd"),
            "n_layer": model_config.get("n_layer"),
            "n_head": model_config.get("n_head"),
            "block_size": model_config.get("block_size"),
            "norm_class_name": model_config.get("norm_class_name"),
            "mlp_class_name": model_config.get("mlp_class_name"),
        },
        "training_step": ckpt.get("step"),
        "baseline_ppl": BASELINE_PPL,
        "reference_ppl": REFERENCE_PPL,
    }
    with open(os.path.join(log_dir, "reward.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(log_dir, "reward.txt"), "w") as f:
        f.write(f"{reward}\n")

    print(f"\nResults saved to {log_dir}/")


if __name__ == "__main__":
    main()
VERIFIER_SCRIPT

echo ""
echo "=== Verifier Complete ==="
cat /logs/verifier/reward.txt
