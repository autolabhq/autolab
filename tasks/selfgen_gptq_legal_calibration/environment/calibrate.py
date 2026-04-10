from __future__ import annotations

"""Calibration token builders and sensitivity collection helpers."""

import torch


def _collect_column_sensitivity(model, token_batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Cheap Hessian proxy: per-input-column second moment from linear layer inputs.
    Returns a dict keyed by parameter name (e.g. blocks.0.1.weight).
    """
    stats: dict[str, torch.Tensor] = {}
    hooks = []

    name_to_module = dict(model.named_modules())

    for name, module in name_to_module.items():
        if isinstance(module, torch.nn.Linear):
            pkey = name + ".weight"

            def _mk_hook(key):
                def _hook(_m, inp, _out):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    col_var = (x * x).mean(dim=0).cpu()
                    if key not in stats:
                        stats[key] = col_var
                    else:
                        stats[key] += col_var

                return _hook

            hooks.append(module.register_forward_hook(_mk_hook(pkey)))

    with torch.inference_mode():
        _ = model.forward_logits(token_batch)

    for h in hooks:
        h.remove()
    for k in list(stats.keys()):
        stats[k] = stats[k].clamp_min(1e-8)
    return stats


def random_calibration_tokens(
    vocab_size: int,
    num_seqs: int,
    seq_len: int,
    seed: int = 0,
) -> torch.Tensor:
    """Sample synthetic calibration tokens from RNG."""

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randint(0, vocab_size, (num_seqs, seq_len), generator=g, dtype=torch.int64)


def autoregressive_selfgen_tokens(
    model,
    vocab_size: int,
    num_seqs: int,
    seq_len: int,
    temperature: float = 0.8,
    seed: int = 0,
) -> torch.Tensor:
    """Autoregressively self-generate calibration tokens from the model."""

    # Baseline implementation is intentionally simple; participants can optimize quality.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    device = next(model.parameters()).device
    x = torch.randint(0, vocab_size, (num_seqs, 1), generator=g, dtype=torch.int64)
    x = x.to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(seq_len - 1):
            logits = model.forward_logits(x)[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
    return x.cpu()


def build_calibration_pack(
    model,
    strategy: str,
    vocab_size: int,
    num_seqs: int,
    seq_len: int,
    seed: int,
    temperature: float = 0.8,
) -> dict[str, object]:
    """Build calibration payload consumed by quantization pipeline."""

    if strategy not in {"random", "selfgen_ar"}:
        raise ValueError(f"unsupported calibration strategy: {strategy}")

    if strategy == "random":
        tokens = random_calibration_tokens(vocab_size, num_seqs, seq_len, seed=seed)
    else:
        tokens = autoregressive_selfgen_tokens(
            model,
            vocab_size=vocab_size,
            num_seqs=num_seqs,
            seq_len=seq_len,
            temperature=temperature,
            seed=seed,
        )

    dev = next(model.parameters()).device
    sens = _collect_column_sensitivity(model, tokens.to(dev))
    return {
        "tokens": tokens,
        "col_sensitivity": sens,
        "strategy": strategy,
        "num_tokens": int(tokens.numel()),
    }
