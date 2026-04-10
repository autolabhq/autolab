from __future__ import annotations

"""Quantization and artifact-size utilities for the benchmark."""

import copy
import io
import lzma
from typing import Dict, Tuple

import torch


def _quantize_per_row_int6(w: torch.Tensor, clip_range: int = 31) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row int6 quantization with one scale per output channel."""

    w32 = w.float()
    row_clip = w32.abs().amax(dim=1).clamp_min(1.0 / max(clip_range, 1))
    scale = (row_clip / float(clip_range)).to(torch.float16)
    q = torch.clamp(torch.round(w32 / scale[:, None]), -clip_range, clip_range).to(torch.int8)
    return q, scale


def _quantize_per_tensor_int6(w: torch.Tensor, clip_range: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor int6 quantization (weaker baseline path)."""

    w32 = w.float()
    amax = w32.abs().max().clamp_min(1.0 / max(clip_range, 1))
    scale = torch.tensor(float(amax.item()) / float(clip_range), dtype=torch.float16)
    q = torch.clamp(torch.round(w32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def quantize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    method: str = "gptq_lite",
    calib_stats: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    """Quantize state dict with baseline or hessian-proxy method."""

    if method not in {"gptq_lite", "hessian_proxy"}:
        raise ValueError(f"unsupported quantization method: {method}")

    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        t = v.detach().cpu().contiguous()
        if t.ndim == 2 and t.is_floating_point() and t.numel() > 4096:
            if method == "gptq_lite":
                # Deliberately weaker baseline path: per-tensor and lower effective range.
                q, s = _quantize_per_tensor_int6(t, clip_range=7)
                out[k + ".meta"] = torch.tensor([2], dtype=torch.int32)
            else:
                # Better path: per-row scales and optional column sensitivity ordering.
                if calib_stats is not None and k in calib_stats and calib_stats[k].ndim == 1:
                    sens = calib_stats[k].float()
                    perm = torch.argsort(sens, descending=True)
                    inv_perm = torch.argsort(perm)
                    tp = t[:, perm]
                    q, s = _quantize_per_row_int6(tp, clip_range=31)
                    q = q[:, inv_perm]
                else:
                    q, s = _quantize_per_row_int6(t, clip_range=31)
                out[k + ".meta"] = torch.tensor([1], dtype=torch.int32)
            out[k + ".q"] = q
            out[k + ".scale"] = s
        else:
            out[k] = t.to(torch.float16) if t.is_floating_point() else t
    return out


def dequantize_state_dict(
    quantized: Dict[str, torch.Tensor],
    template_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Dequantize a quantized state dict back to template dtypes."""

    out: Dict[str, torch.Tensor] = {}
    for k, v in template_sd.items():
        if k + ".meta" in quantized:
            q = quantized[k + ".q"].float()
            s = quantized[k + ".scale"].float()
            if s.ndim == 0:
                out[k] = (q * s).to(v.dtype)
            else:
                out[k] = (q * s[:, None]).to(v.dtype)
        else:
            t = quantized[k]
            if t.is_floating_point():
                out[k] = t.to(v.dtype)
            else:
                out[k] = t
    return out


def artifact_size_bytes(quantized_sd: Dict[str, torch.Tensor], code_bytes: int) -> int:
    """Compute compressed artifact bytes including editable code size."""

    buf = io.BytesIO()
    torch.save(quantized_sd, buf)
    blob = lzma.compress(buf.getvalue(), preset=6)
    return len(blob) + code_bytes


def apply_selective_prune(
    quantized_sd: Dict[str, torch.Tensor],
    target_total_bytes: int,
    code_bytes: int,
) -> Dict[str, torch.Tensor]:
    """Prune low-impact +/-1 entries until byte cap is met."""

    # Baseline implementation: prune a fraction of +/-1 entries by smallest impact proxy.
    candidate = copy.deepcopy(quantized_sd)
    size_now = artifact_size_bytes(candidate, code_bytes)
    if size_now <= target_total_bytes:
        return candidate

    one_positions: list[tuple[str, int, float]] = []
    for k in list(candidate.keys()):
        if not k.endswith(".q"):
            continue
        scale_key = k[:-2] + ".scale"
        if scale_key not in candidate:
            continue
        q = candidate[k]
        s = candidate[scale_key]
        m = (q.abs() == 1)
        if not bool(m.any()):
            continue
        flat = torch.arange(q.numel()).reshape(q.shape)[m]
        if s.ndim == 0:
            err = float((s.float() ** 2).item())
            for idx in flat.tolist():
                one_positions.append((k, idx, err))
        else:
            rows = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[m]
            errs = (s.float()[rows] ** 2)
            for idx, e in zip(flat.tolist(), errs.tolist()):
                one_positions.append((k, idx, float(e)))

    if not one_positions:
        return candidate

    one_positions.sort(key=lambda x: x[2])
    lo, hi = 0, len(one_positions)
    best = candidate
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = copy.deepcopy(candidate)
        for i in range(mid):
            key, pos, _ = one_positions[i]
            trial[key].view(-1)[pos] = 0
        trial_size = artifact_size_bytes(trial, code_bytes)
        if trial_size <= target_total_bytes:
            best = trial
            hi = mid - 1
        else:
            lo = mid + 1
    return best
