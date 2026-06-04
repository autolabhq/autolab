# Moving MNIST World Model — Reference

## Background

The agent is asked to train a video world model from scratch on Moving MNIST
(20-frame, 64×64 grayscale clips) that takes 10 context frames and predicts
the next 10 autoregressively. The metric is **PSNR** over the 10-step rollout;
reward is `agent_psnr / reference_psnr`.

The baseline is a 1-layer ConvLSTM with 16 hidden channels, MSE loss, constant
LR 1e-3, and only 2000 training steps. It works end-to-end but plateaus around
~14 dB — enough to see blurry digits staying roughly in place, with rapidly
growing rollout drift.

There are no bugs. This is a pure modelling + training optimisation task.

## Optimization 1: Model capacity

- **Baseline**: 1-layer ConvLSTM, 16 hidden channels (~10k params).
- **Reference**: 3-layer ConvLSTM, 64 hidden channels (~850k params).
- **Why**: The baseline lacks the capacity to model two independent bouncing
  digits. Depth adds temporal receptive field; width gives the model enough
  channels to represent position + velocity of each digit in its hidden state.

## Optimization 2: Exposure bias / scheduled sampling

- **Baseline**: Always feeds the model's own prediction back in at train time.
  This matches eval but gives no stable learning signal early on — the model
  sees its own noise rather than clean targets, and drifts.
- **Reference**: Anneals teacher-forcing probability from 1.0 → 0.0 linearly
  over the first 10k steps. Early training uses ground-truth previous frames;
  late training matches the eval distribution.
- **Why**: Eliminates the cold-start problem while converging to the eval
  regime. Worth ~2–3 dB.

## Optimization 3: Loss function

- **Baseline**: Pure MSE.
- **Reference**: `MSE + 0.1 * L1`. L1 pulls the optimum toward the median
  pixel value, which reduces the "grey blur" failure mode MSE encourages when
  the model is uncertain about digit position.
- **Why**: PSNR is MSE-based, so L1 doesn't directly optimise the metric, but
  it regularises the model toward sharper predictions which in turn reduces
  MSE in practice.

## Optimization 4: Optimiser + schedule

- **Baseline**: Adam, constant LR 1e-3, no weight decay, no warmup, 2000 steps.
- **Reference**: AdamW (wd=1e-5), cosine LR 5e-4 → 5e-5 with 500 warmup steps,
  15000 steps total.
- **Why**: Warmup prevents early divergence of the larger model. Cosine decay
  squeezes out the last ~1 dB. The 7.5× longer training is affordable because
  of the precision / throughput changes below.

## Optimization 5: Mixed precision

- **Baseline**: float32.
- **Reference**: `torch.autocast(bfloat16)`.
- **Why**: ~2× throughput on H100/L40S with no stability issues on this task,
  which is what makes 15k steps fit in the 4-hour budget.

## Reference summary

| Knob | Baseline | Reference |
|------|----------|-----------|
| Layers | 1 | 3 |
| Hidden ch | 16 | 64 |
| Params | ~10k | ~850k |
| Precision | float32 | bfloat16 |
| Loss | MSE | MSE + 0.1·L1 |
| LR schedule | constant 1e-3 | cosine 5e-4→5e-5 w/ 500 warmup |
| Weight decay | 0 | 1e-5 |
| Scheduled sampling | none | 1.0 → 0.0 over 10k steps |
| Steps | 2000 | 15000 |
| Batch size | 16 | 32 |
| Expected test PSNR | ~14 dB | ~15.5 dB |

## Sources

- ConvLSTM: Shi et al., *Convolutional LSTM Network: A Machine Learning
  Approach for Precipitation Nowcasting* (NeurIPS 2015).
- Scheduled sampling: Bengio et al., *Scheduled Sampling for Sequence
  Prediction with Recurrent Neural Networks* (NeurIPS 2015).
- Moving MNIST: Srivastava et al., *Unsupervised Learning of Video
  Representations using LSTMs* (ICML 2015).
