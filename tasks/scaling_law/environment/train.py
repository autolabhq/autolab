#!/usr/bin/env python3
"""WikiText-103 language model pretraining using LitGPT."""

import math
import json
import os
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from litgpt import GPT, Config

# Model configuration
MODEL_CONFIG = dict(
    block_size=512,
    n_layer=6,
    n_embd=128,
    n_head=4,
    vocab_size=50257,
    padding_multiple=512,
    intermediate_size=512,
    norm_class_name="LayerNorm",
    mlp_class_name="GptNeoxMLP",
    rope_base=10000,
    rotary_percentage=0.25,
    bias=True,
    parallel_residual=True,
)

# Training configuration
TRAIN_CONFIG = dict(
    seq_length=256,
    batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=0,
    lr_scheduler="constant",
    weight_decay=0.0,
    grad_clip=1.0,
    max_steps=5000,
    eval_interval=200,
    dtype="float32",
    betas=(0.9, 0.999),
    compile_model=False,
    max_train_hours=12.0,
)


class TokenDataset(Dataset):

    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        # +1 for the target shift
        self.n_chunks = len(data) // (seq_length + 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * (self.seq_length + 1)
        chunk = self.data[start : start + self.seq_length + 1]
        return chunk


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr, scheduler):
    if scheduler == "constant":
        return max_lr
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    # Cosine decay
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
    print(f"Device: {device}")

    # --- Precision setup ---
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[TRAIN_CONFIG["dtype"]]
    if TRAIN_CONFIG["dtype"] == "float32":
        dtype_ctx = nullcontext()
    else:
        dtype_ctx = torch.autocast(device_type="cuda", dtype=dtype)

    use_scaler = TRAIN_CONFIG["dtype"] == "float16"
    scaler = torch.amp.GradScaler() if use_scaler else None

    # --- Load data ---
    print("Loading data...")
    train_data = torch.load("/data/wikitext103/train.pt", weights_only=True)
    val_data = torch.load("/data/wikitext103/validation.pt", weights_only=True)
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    seq_length = TRAIN_CONFIG["seq_length"]
    train_dataset = TokenDataset(train_data, seq_length)
    val_dataset = TokenDataset(val_data, seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --- Model ---
    config = Config(**MODEL_CONFIG)
    model = GPT(config)
    model.apply(model._init_weights)
    model = model.to(device)

    if TRAIN_CONFIG["compile_model"]:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Architecture: n_embd={config.n_embd}, n_layer={config.n_layer}, "
          f"n_head={config.n_head}, block_size={config.block_size}")
    print(f"  norm={config.norm_class_name}, mlp={config.mlp_class_name}, "
          f"rope_pct={config.rotary_percentage}, bias={config.bias}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        betas=TRAIN_CONFIG["betas"],
    )

    # --- Training loop ---
    os.makedirs("/app/output", exist_ok=True)
    best_val_ppl = float("inf")
    start_time = time.time()
    max_seconds = TRAIN_CONFIG["max_train_hours"] * 3600
    global_step = 0
    micro_step = 0
    grad_accum = TRAIN_CONFIG["gradient_accumulation_steps"]
    min_lr = TRAIN_CONFIG["learning_rate"] * 0.1

    tokens_per_step = (
        TRAIN_CONFIG["batch_size"] * grad_accum * seq_length
    )
    print(f"\nTraining config:")
    print(f"  seq_length={seq_length}, batch={TRAIN_CONFIG['batch_size']}, "
          f"grad_accum={grad_accum}")
    print(f"  effective tokens/step: {tokens_per_step:,}")
    print(f"  dtype={TRAIN_CONFIG['dtype']}, compile={TRAIN_CONFIG['compile_model']}")
    print(f"  lr={TRAIN_CONFIG['learning_rate']}, scheduler={TRAIN_CONFIG['lr_scheduler']}, "
          f"warmup={TRAIN_CONFIG['warmup_steps']}")
    print(f"  max_steps={TRAIN_CONFIG['max_steps']}, max_hours={TRAIN_CONFIG['max_train_hours']}")
    print()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    done = False

    while not done:
        epoch += 1
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                print(f"Time limit reached ({TRAIN_CONFIG['max_train_hours']}h)")
                done = True
                break
            if global_step >= TRAIN_CONFIG["max_steps"]:
                print(f"Max steps reached ({TRAIN_CONFIG['max_steps']})")
                done = True
                break

            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            with dtype_ctx:
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                loss = loss / grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            # --- Gradient step ---
            # Update learning rate
            lr = get_lr(
                global_step,
                TRAIN_CONFIG["warmup_steps"],
                TRAIN_CONFIG["max_steps"],
                TRAIN_CONFIG["learning_rate"],
                min_lr,
                TRAIN_CONFIG["lr_scheduler"],
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Clip gradients
            if TRAIN_CONFIG["grad_clip"] > 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), TRAIN_CONFIG["grad_clip"]
                )

            # Step
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # --- Logging ---
            if global_step % 50 == 0:
                train_ppl = math.exp(loss.item() * grad_accum)
                elapsed = time.time() - start_time
                toks_sec = (
                    (global_step + 1) * tokens_per_step / elapsed
                    if elapsed > 0
                    else 0
                )
                print(
                    f"Step {global_step:>6d} | Loss {loss.item()*grad_accum:.4f} | "
                    f"PPL {train_ppl:.1f} | LR {lr:.2e} | "
                    f"{elapsed/3600:.2f}h | {toks_sec:.0f} tok/s"
                )

            # --- Evaluation ---
            if global_step > 0 and global_step % TRAIN_CONFIG["eval_interval"] == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device, dtype_ctx)
                print(
                    f"  >> Eval: loss={val_loss:.4f} ppl={val_ppl:.2f} "
                    f"(best={best_val_ppl:.2f})"
                )

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "model_config": MODEL_CONFIG,
                        "train_config": {
                            k: list(v) if isinstance(v, tuple) else v
                            for k, v in TRAIN_CONFIG.items()
                        },
                        "step": global_step,
                        "val_ppl": val_ppl,
                        "n_params": n_params,
                    }
                    torch.save(checkpoint, "/app/output/checkpoint.pt")
                    print(f"  >> Saved checkpoint (ppl={val_ppl:.2f})")

                if math.isnan(val_loss) or math.isinf(val_loss):
                    print("Training diverged! Stopping.")
                    done = True
                    break

            global_step += 1

    # --- Final evaluation ---
    val_loss, val_ppl = evaluate(model, val_loader, device, dtype_ctx)
    print(f"\nFinal: loss={val_loss:.4f} ppl={val_ppl:.2f} (best={best_val_ppl:.2f})")

    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": MODEL_CONFIG,
            "train_config": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in TRAIN_CONFIG.items()
            },
            "step": global_step,
            "val_ppl": val_ppl,
            "n_params": n_params,
        }
        torch.save(checkpoint, "/app/output/checkpoint.pt")

    # --- Save metrics ---
    total_time = time.time() - start_time
    metrics = {
        "final_val_loss": val_loss,
        "final_val_ppl": val_ppl,
        "best_val_ppl": best_val_ppl,
        "total_steps": global_step,
        "training_time_seconds": total_time,
        "n_params": n_params,
    }
    with open("/app/output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"\nDone. Time: {total_time/3600:.2f}h | Steps: {global_step} | "
        f"Best PPL: {best_val_ppl:.2f} | Params: {n_params:,}"
    )


if __name__ == "__main__":
    main()
