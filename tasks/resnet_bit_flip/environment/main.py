#!/usr/bin/env python3
"""
main.py — Harness for the ResNet bit-flip attack puzzle.

Loads the frozen CIFAR-10 classifier, passes a fresh model + a sample batch of
labelled test images to solve.find_bit_flips(), validates the returned flip list,
applies flips via uint32 XOR to a fresh state dict, reloads into the model, and
evaluates top-1 accuracy on the full CIFAR-10 test set.

Interface expected in /app/solve.py:
    def find_bit_flips(model, images, labels) -> list[tuple[str, int, int]]:
        ...
where each tuple is (param_name, flat_index, bit_index) with bit_index in [0, 31].

DO NOT MODIFY THIS FILE.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from torchvision import datasets, transforms

from model import MiniResNet

MODEL_PATH = '/app/model.pth'
DATA_DIR = '/data/cifar10'
LOG_DIR = '/logs/verifier'

MAX_FLIPS = 10000
GATE_ACC = 0.12            # strict less-than; below this the attack is accepted
MAX_ABS_WEIGHT = 10.0      # after flips, every float32 weight must have |w| <= this
SAMPLE_SEED = 777
SAMPLE_SIZE = 1000         # images+labels given to find_bit_flips

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_frozen_model():
    model = MiniResNet()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model, sd


def load_testset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=transform)


def get_sample(testset):
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.permutation(len(testset))[:SAMPLE_SIZE]
    imgs = torch.stack([testset[int(i)][0] for i in idx])
    labels = torch.tensor([testset[int(i)][1] for i in idx], dtype=torch.long)
    return imgs, labels


def evaluate_accuracy(model, testset, batch_size=512):
    loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total


def normalize_and_validate(flips, state_dict, allowed_names):
    """Return a deduped, validated list of (name, idx, bit). Raises on bad entries.

    allowed_names: set of state_dict keys that may be targeted. Restricted to
    learnable parameters (model.named_parameters()) — BatchNorm running stats
    and other buffers are NOT flippable.
    """
    if not isinstance(flips, (list, tuple)):
        raise ValueError(f"find_bit_flips must return a list, got {type(flips).__name__}")
    if len(flips) > MAX_FLIPS:
        raise ValueError(f"too many flips: {len(flips)} > {MAX_FLIPS}")

    seen = set()
    out = []
    shapes = {k: v.numel() for k, v in state_dict.items()}
    dtypes = {k: v.dtype for k, v in state_dict.items()}

    for i, item in enumerate(flips):
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            raise ValueError(f"flip[{i}] must be a 3-tuple, got {item!r}")
        name, idx, bit = item
        if not isinstance(name, str):
            raise ValueError(f"flip[{i}] name must be str, got {type(name).__name__}")
        if name not in state_dict:
            raise ValueError(f"flip[{i}] unknown name: {name!r}")
        if name not in allowed_names:
            raise ValueError(f"flip[{i}] {name!r} is not a learnable parameter "
                             f"(buffers like BN running stats are not flippable)")
        if dtypes[name] != torch.float32:
            raise ValueError(f"flip[{i}] param {name} is {dtypes[name]}, only float32 supported")
        try:
            idx_i = int(idx)
            bit_i = int(bit)
        except (TypeError, ValueError):
            raise ValueError(f"flip[{i}] idx/bit must be ints, got {idx!r}, {bit!r}")
        if not (0 <= idx_i < shapes[name]):
            raise ValueError(f"flip[{i}] idx {idx_i} out of range for {name} "
                             f"(numel={shapes[name]})")
        if not (0 <= bit_i < 32):
            raise ValueError(f"flip[{i}] bit {bit_i} out of range [0, 31]")
        key = (name, idx_i, bit_i)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def apply_flips(state_dict, flips):
    """Mutate state_dict in place by XOR-ing the requested bits in float32 weights."""
    for name, idx, bit in flips:
        tensor = state_dict[name]
        flat = tensor.view(-1)
        np_view = flat.numpy().view(np.uint32)   # shared memory with tensor
        np_view[idx] ^= np.uint32(1) << np.uint32(bit)


def fmt(x):
    return f"{x:.6f}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--benchmark"
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data and the frozen model once
    testset = load_testset()
    sample_imgs, sample_labels = get_sample(testset)

    # Give the agent a fresh model loaded from disk — they may mutate it
    agent_model, frozen_sd = load_frozen_model()

    # Import the agent's solve module
    spec = importlib.util.spec_from_file_location("solve", "/app/solve.py")
    solve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solve)

    allowed_names = {n for n, _ in agent_model.named_parameters()}

    t0 = time.perf_counter()
    flips = solve.find_bit_flips(agent_model, sample_imgs, sample_labels)
    t1 = time.perf_counter()

    try:
        flips = normalize_and_validate(flips, frozen_sd, allowed_names)
    except ValueError as e:
        print(f"result=fail reason=invalid_flips msg={e}")
        sys.exit(1)

    n_flips = len(flips)
    if n_flips == 0:
        print(f"result=fail reason=no_flips bits_flipped=0")
        sys.exit(1)

    # Apply flips to a fresh copy of the frozen weights
    eval_sd = {k: v.clone() for k, v in frozen_sd.items()}
    apply_flips(eval_sd, flips)

    eval_model = MiniResNet()
    eval_model.load_state_dict(eval_sd)
    eval_model.eval()

    # Sanity: the flipped weights must not be NaN/Inf — if they are, attack is invalid
    if not all(torch.isfinite(p).all() for p in eval_model.parameters()):
        print(f"result=fail reason=non_finite_weights bits_flipped={n_flips}")
        sys.exit(1)

    max_abs = max(float(p.abs().max().item()) for p in eval_model.parameters()
                  if p.dtype == torch.float32)
    if max_abs > MAX_ABS_WEIGHT:
        print(f"result=fail reason=weight_magnitude_exceeded "
              f"bits_flipped={n_flips} max_abs_weight={max_abs:.3e} cap={MAX_ABS_WEIGHT}")
        sys.exit(1)

    accuracy = evaluate_accuracy(eval_model, testset)
    gate_passed = accuracy < GATE_ACC
    t2 = time.perf_counter()

    print(f"result={'ok' if gate_passed else 'fail'} "
          f"accuracy={fmt(accuracy)} bits_flipped={n_flips} "
          f"gate={'pass' if gate_passed else 'fail'} "
          f"solve_time={fmt(t1 - t0)} eval_time={fmt(t2 - t1)}")

    if mode == "--verify":
        sys.exit(0 if gate_passed else 1)


if __name__ == "__main__":
    main()
