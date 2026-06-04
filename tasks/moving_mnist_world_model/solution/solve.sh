#!/bin/bash
set -euo pipefail

# Reference solution for moving_mnist_world_model.
# - Deeper ConvLSTM (3 layers, 64 hidden ch) with teacher forcing on context
# - Scheduled sampling during future rollout
# - MSE + 0.1 * L1 loss
# - bfloat16 autocast + cosine LR schedule with warmup
# - 15k steps, batch 32

cat > /app/model.py << 'PYEOF'
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel, padding=pad)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, h, w, device, dtype):
        zeros = torch.zeros(batch, self.hidden_ch, h, w, device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTMWorldModel(nn.Module):
    def __init__(self, hidden_ch=64, n_layers=3, kernel=3):
        super().__init__()
        self.cells = nn.ModuleList([
            ConvLSTMCell(1 if i == 0 else hidden_ch, hidden_ch, kernel)
            for i in range(n_layers)
        ])
        self.head = nn.Conv2d(hidden_ch, 1, 1)

    def _step(self, x, states):
        new_states = []
        h = x
        for cell, st in zip(self.cells, states):
            h, c = cell(h, st)
            new_states.append((h, c))
        return h, new_states

    def step_predict(self, x, states):
        """One-step prediction of next frame as a clamped residual."""
        h, states = self._step(x, states)
        delta = self.head(h)
        return torch.clamp(x + delta, 0.0, 1.0), states

    def forward(self, context, n_steps, target=None, tf_prob=0.0):
        """If target given and tf_prob > 0, do scheduled sampling on the future frames."""
        B, T, _, H, W = context.shape
        device, dtype = context.device, context.dtype
        states = [cell.init_state(B, H, W, device, dtype) for cell in self.cells]

        # Ingest context (predictions thrown away)
        for t in range(T):
            _, states = self._step(context[:, t], states)

        preds = []
        prev = context[:, -1]
        for t in range(n_steps):
            pred, states = self.step_predict(prev, states)
            preds.append(pred)
            if target is not None and tf_prob > 0.0 and torch.rand(()) < tf_prob:
                prev = target[:, t]
            else:
                prev = pred
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def rollout(self, context, n_steps):
        self.eval()
        return self.forward(context, n_steps)


def build_model(config):
    return ConvLSTMWorldModel(
        hidden_ch=config.get("hidden_ch", 64),
        n_layers=config.get("n_layers", 3),
        kernel=config.get("kernel", 3),
    )
PYEOF


cat > /app/train.py << 'PYEOF'
#!/usr/bin/env python3
import json, math, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import build_model


MODEL_CONFIG = dict(hidden_ch=64, n_layers=3, kernel=3)

TRAIN_CONFIG = dict(
    context_len=10,
    predict_len=10,
    batch_size=32,
    learning_rate=5e-4,
    min_lr=5e-5,
    weight_decay=1e-5,
    warmup_steps=500,
    max_steps=15000,
    eval_interval=500,
    grad_clip=1.0,
    l1_weight=0.1,
    tf_start=1.0,
    tf_end=0.0,
    tf_anneal_steps=10000,
    dtype="bfloat16",
    max_train_hours=4.0,
    seed=0,
)


class ClipDataset(Dataset):
    def __init__(self, data, ctx, pred):
        self.data = data; self.ctx = ctx; self.pred = pred
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, i):
        clip = self.data[i].float() / 255.0
        return clip[:self.ctx], clip[self.ctx:self.ctx + self.pred]


def get_lr(step):
    if step < TRAIN_CONFIG["warmup_steps"]:
        return TRAIN_CONFIG["learning_rate"] * (step + 1) / (TRAIN_CONFIG["warmup_steps"] + 1)
    progress = (step - TRAIN_CONFIG["warmup_steps"]) / max(
        1, TRAIN_CONFIG["max_steps"] - TRAIN_CONFIG["warmup_steps"])
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return TRAIN_CONFIG["min_lr"] + (TRAIN_CONFIG["learning_rate"] - TRAIN_CONFIG["min_lr"]) * coeff


def get_tf_prob(step):
    t = min(step / max(1, TRAIN_CONFIG["tf_anneal_steps"]), 1.0)
    return TRAIN_CONFIG["tf_start"] + (TRAIN_CONFIG["tf_end"] - TRAIN_CONFIG["tf_start"]) * t


@torch.no_grad()
def evaluate(model, loader, device, pred_len):
    model.eval()
    tot = 0.0; n = 0
    for ctx, tgt in loader:
        ctx = ctx.to(device); tgt = tgt.to(device)
        pred = model.forward(ctx, pred_len)
        tot += F.mse_loss(pred, tgt, reduction="sum").item(); n += tgt.numel()
    model.train()
    mse = tot / n
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    return mse, psnr


def main():
    torch.manual_seed(TRAIN_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    train_data = torch.load("/data/moving_mnist/train.pt", weights_only=True)
    val_data = torch.load("/data/moving_mnist/val.pt", weights_only=True)
    tr = ClipDataset(train_data, TRAIN_CONFIG["context_len"], TRAIN_CONFIG["predict_len"])
    va = ClipDataset(val_data, TRAIN_CONFIG["context_len"], TRAIN_CONFIG["predict_len"])

    train_loader = DataLoader(tr, batch_size=TRAIN_CONFIG["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=TRAIN_CONFIG["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(MODEL_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params  config={MODEL_CONFIG}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=TRAIN_CONFIG["learning_rate"],
                                  weight_decay=TRAIN_CONFIG["weight_decay"])

    os.makedirs("/app/output", exist_ok=True)
    start = time.time()
    max_sec = TRAIN_CONFIG["max_train_hours"] * 3600
    best = -float("inf"); step = 0; done = False
    model.train()

    while not done:
        for ctx, tgt in train_loader:
            if time.time() - start > max_sec or step >= TRAIN_CONFIG["max_steps"]:
                done = True; break

            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tf_prob = get_tf_prob(step)

            with dtype_ctx:
                pred = model(ctx, TRAIN_CONFIG["predict_len"],
                             target=tgt, tf_prob=tf_prob)
                mse = F.mse_loss(pred, tgt)
                l1 = F.l1_loss(pred, tgt)
                loss = mse + TRAIN_CONFIG["l1_weight"] * l1

            lr = get_lr(step)
            for pg in optimizer.param_groups: pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            optimizer.step()

            if step % 100 == 0:
                m = mse.item()
                psnr = 10.0 * math.log10(1.0 / max(m, 1e-12))
                print(f"Step {step:>5d} | loss {loss.item():.5f} | mse {m:.5f} | "
                      f"PSNR {psnr:5.2f} dB | tf {tf_prob:.2f} | lr {lr:.2e} | "
                      f"{(time.time()-start)/60:.1f} min")

            if step > 0 and step % TRAIN_CONFIG["eval_interval"] == 0:
                vm, vp = evaluate(model, val_loader, device, TRAIN_CONFIG["predict_len"])
                print(f"  >> val MSE {vm:.5f} | val PSNR {vp:.2f} dB (best {best:.2f} dB)")
                if vp > best:
                    best = vp
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_config": MODEL_CONFIG,
                        "train_config": TRAIN_CONFIG,
                        "step": step, "val_psnr": vp, "val_mse": vm,
                        "n_params": n_params,
                    }, "/app/output/checkpoint.pt")
            step += 1

    vm, vp = evaluate(model, val_loader, device, TRAIN_CONFIG["predict_len"])
    if vp > best:
        best = vp
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": MODEL_CONFIG,
            "train_config": TRAIN_CONFIG,
            "step": step, "val_psnr": vp, "val_mse": vm,
            "n_params": n_params,
        }, "/app/output/checkpoint.pt")

    json.dump({"best_val_psnr": best, "final_val_psnr": vp, "final_val_mse": vm,
               "total_steps": step, "n_params": n_params},
              open("/app/output/metrics.json", "w"), indent=2)
    print(f"Done. Best val PSNR: {best:.2f} dB | Params: {n_params:,}")


if __name__ == "__main__":
    main()
PYEOF

echo "Running reference training..."
bash /app/train.sh
