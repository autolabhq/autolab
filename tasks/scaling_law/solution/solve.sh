#!/bin/bash
set -euo pipefail

# Apply optimized training configuration
cat > /app/train.py << 'EOF'
#!/usr/bin/env python3
import math
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from litgpt import GPT, Config

MODEL_CONFIG = dict(
    block_size=1024,
    n_layer=16,
    n_embd=768,
    n_head=12,
    vocab_size=50257,
    padding_multiple=512,
    intermediate_size=2048,
    norm_class_name="RMSNorm",
    mlp_class_name="LLaMAMLP",
    rope_base=10000,
    rotary_percentage=1.0,
    bias=False,
    parallel_residual=False,
    n_query_groups=4,
)

TRAIN_CONFIG = dict(
    seq_length=1024,
    batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=500,
    lr_scheduler="cosine",
    weight_decay=0.1,
    grad_clip=1.0,
    max_steps=25000,
    eval_interval=500,
    dtype="bfloat16",
    betas=(0.9, 0.95),
    compile_model=True,
    max_train_hours=12.0,
)


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


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr, scheduler):
    if scheduler == "constant":
        return max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    if scheduler == "cosine":
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * coeff
    return max_lr


@torch.no_grad()
def evaluate(model, val_loader, device, dtype_ctx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in val_loader:
        batch = batch.to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        with dtype_ctx:
            logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    model.train()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    train_data = torch.load("/data/wikitext103/train.pt", weights_only=True)
    val_data = torch.load("/data/wikitext103/validation.pt", weights_only=True)

    train_dataset = TokenDataset(train_data, TRAIN_CONFIG["seq_length"])
    val_dataset = TokenDataset(val_data, TRAIN_CONFIG["seq_length"])

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            num_workers=4, pin_memory=True)

    config = Config(**MODEL_CONFIG)
    model = GPT(config)
    model = model.to(device)
    if TRAIN_CONFIG["compile_model"]:
        model = torch.compile(model, mode="max-autotune")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"],
                                  betas=TRAIN_CONFIG["betas"], weight_decay=TRAIN_CONFIG["weight_decay"])

    os.makedirs("/app/output", exist_ok=True)
    global_step = 0
    micro_step = 0
    grad_accum = TRAIN_CONFIG["gradient_accumulation_steps"]
    tokens_per_step = TRAIN_CONFIG["batch_size"] * TRAIN_CONFIG["seq_length"] * grad_accum
    min_lr = TRAIN_CONFIG["learning_rate"] * 0.1
    start_time = time.time()
    best_val_ppl = float("inf")
    done = False

    while not done:
        for batch in train_loader:
            if (time.time() - start_time) / 3600 >= TRAIN_CONFIG["max_train_hours"]:
                done = True
                break
            if global_step >= TRAIN_CONFIG["max_steps"]:
                done = True
                break

            batch = batch.to(device)
            with dtype_ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       batch[:, 1:].reshape(-1)) / grad_accum
            loss.backward()
            micro_step += 1
            if micro_step % grad_accum != 0:
                continue

            lr = get_lr(global_step, TRAIN_CONFIG["warmup_steps"], TRAIN_CONFIG["max_steps"],
                        TRAIN_CONFIG["learning_rate"], min_lr, TRAIN_CONFIG["lr_scheduler"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Step {global_step:>6d} | Loss {loss.item()*grad_accum:.4f} | "
                      f"LR {lr:.2e} | {elapsed/3600:.2f}h")

            if global_step > 0 and global_step % TRAIN_CONFIG["eval_interval"] == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device, dtype_ctx)
                print(f"  >> Eval: ppl={val_ppl:.2f} (best={best_val_ppl:.2f})")
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_config": MODEL_CONFIG,
                        "train_config": {k: list(v) if isinstance(v, tuple) else v
                                         for k, v in TRAIN_CONFIG.items()},
                        "step": global_step, "val_ppl": val_ppl, "n_params": n_params,
                    }, "/app/output/checkpoint.pt")
                if math.isnan(val_loss):
                    done = True
                    break
            global_step += 1

    val_loss, val_ppl = evaluate(model, val_loader, device, dtype_ctx)
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        torch.save({"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG,
                     "train_config": {k: list(v) if isinstance(v, tuple) else v for k, v in TRAIN_CONFIG.items()},
                     "step": global_step, "val_ppl": val_ppl, "n_params": n_params},
                    "/app/output/checkpoint.pt")

    json.dump({"best_val_ppl": best_val_ppl, "total_steps": global_step, "n_params": n_params},
              open("/app/output/metrics.json", "w"), indent=2)
    print(f"Done. Best PPL: {best_val_ppl:.2f}")


if __name__ == "__main__":
    main()
EOF

echo "Running training..."
bash /app/train.sh
