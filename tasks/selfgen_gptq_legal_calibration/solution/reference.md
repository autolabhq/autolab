# Self-Generated Legal Calibration for GPTQ — Reference

## Background

This task targets a practical post-training quantization (PTQ) trade-off: improve quantized model quality without violating data-access rules. Calibration is intentionally constrained to legal token sources (random or model self-generation), while quality is measured on a held-out validation stream.

## Baseline Approach

The baseline path in this task combines:

1. random calibration tokens
2. weaker `gptq_lite` quantization configuration
3. no advanced calibration-quality adaptation

This gives a stable but non-optimal `val_bpb` anchor for comparison.

## Possible Optimization Directions

1. **Autoregressive self-generated calibration**
   - Replace random calibration with model-generated sequences
   - Tune temperature and sequence shape for better activation coverage
2. **Stronger Hessian-proxy quantization**
   - Move from weaker per-tensor quantization to sensitivity-aware reconstruction
   - Use collected column statistics to improve retained information
3. **Size-aware selective pruning**
   - Remove low-impact quantized entries only when needed for byte-cap compliance
   - Preserve legality and correctness while improving compression efficiency

## Reproducible Anchors

Run under `environment/`:

```bash
python3 benchmark.py --mode baseline --seeds "42,314,999" --enforce-legal 1
python3 benchmark.py --mode eval --seeds "42,314,999" --enforce-legal 1
```

Anchors used in `task.toml`:

- baseline mean `val_bpb`: `8.2402`
- reference target `val_bpb`: `8.2360`

## Minimal Ablation Framing

1. `random + gptq_lite (per-tensor)`
2. `selfgen_ar + gptq_lite (per-tensor)`
3. `selfgen_ar + hessian_proxy (per-row + column sensitivity)`
4. `selfgen_ar + hessian_proxy + selective_pruning`

Strong submissions should converge toward (3)/(4).

## Source

- Frantar et al. (2022), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- Nagel et al. (2021), *A White Paper on Neural Network Quantization*
- Standard PTQ practice on calibration data quality vs compression trade-offs
