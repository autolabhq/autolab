################################################################################
#
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import random


@dataclass
class Workload:
    flops: int
    memory_bytes: int


@dataclass
class TestCase:
    get_init_inputs: Callable
    get_inputs: Callable
    workload: Optional[Workload] = None


# Add DeepGEMM to path if available
try:
    from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8, align, ceil_div
except ImportError:
    # Fallback: simple FP8 quantization
    def ceil_div(x, y):
        return (x + y - 1) // y

    def align(x, y):
        return ceil_div(x, y) * y

    def per_token_cast_to_fp8(x, use_ue8m0=False):
        m, n = x.shape
        padded_n = align(n, 128)
        x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
        x_padded[:, :n] = x
        x_view = x_padded.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        sf = x_amax / 448.0
        return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous(), sf

    def per_block_cast_to_fp8(x, use_ue8m0=False):
        m, n = x.shape
        x_padded = torch.zeros((align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        sf = x_amax / 448.0
        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(x_view.size(0), x_view.size(2))


def make_get_init_inputs(m, num_groups, n, k, expected_m_per_group):

    def get_init_inputs():
        return [m, num_groups, n, k, expected_m_per_group]

    return get_init_inputs


def make_get_inputs(m, num_groups, n, k, expected_m_per_group, seed):

    def get_inputs():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

        # Generate actual_m values for each group
        actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
        aligned_ms = [align(actual_m, 128) for actual_m in actual_ms]
        total_m = sum(aligned_ms)

        # Generate input tensors in BF16
        a_bf16 = torch.randn((total_m, k), dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn((num_groups, n, k), dtype=torch.bfloat16, device="cuda")

        # Quantize to FP8
        a_fp8_data, a_fp8_sf = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)
        b_fp8_data = torch.empty((num_groups, n, k), dtype=torch.float8_e4m3fn, device="cuda")
        b_fp8_sf = torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), dtype=torch.float32, device="cuda")
        for i in range(num_groups):
            b_fp8_data[i], b_fp8_sf[i] = per_block_cast_to_fp8(b_bf16[i], use_ue8m0=False)

        # Generate m_indices
        m_indices = torch.empty(total_m, dtype=torch.int32, device="cuda")
        start = 0
        for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
            actual_end = start + actual_m
            aligned_end = start + aligned_m
            m_indices[start:actual_end] = i
            m_indices[actual_end:aligned_end] = -1
            start = aligned_end

        return [(a_fp8_data, a_fp8_sf), (b_fp8_data, b_fp8_sf), m_indices]

    return get_inputs


test_cases = []
configs = [
    # (num_groups, expected_m_per_group, n, k)
    (8, 4096, 4096, 4096),
    (8, 6144, 4096, 4096),
    (8, 5120, 4096, 4096),
    (4, 8192, 4096, 4096),
    (8, 7168, 4096, 4096),
]
for num_groups, expected_m_per_group, n, k in configs:
    actual_ms = [int(expected_m_per_group * 1.0) for _ in range(num_groups)]
    aligned_ms = [align(actual_m, 128) for actual_m in actual_ms]
    total_m = sum(aligned_ms)
    flops = 2 * num_groups * expected_m_per_group * n * k
    memory_bytes = total_m * k * 1 + num_groups * n * k * 1 + total_m * n * 2
    test_cases.append(
        TestCase(get_init_inputs=make_get_init_inputs(total_m, num_groups, n, k, expected_m_per_group),
                 get_inputs=make_get_inputs(total_m, num_groups, n, k, expected_m_per_group,
                                            0), workload=Workload(flops=flops, memory_bytes=memory_bytes)))
