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
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    M-grouped FP8 GEMM NT contiguous implemented by PyTorch.
    """

    def __init__(self, m, num_groups, n, k, expected_m_per_group):
        super(Model, self).__init__()
        self.m = m
        self.num_groups = num_groups
        self.n = n
        self.k = k
        self.expected_m_per_group = expected_m_per_group

    def forward(self, a, b, m_indices):
        """
        Performs M-grouped FP8 GEMM NT with contiguous layout.

        Args:
            a: tuple of (torch.Tensor of shape (m, k), torch.Tensor of shape (m, ceil_div(k, 128))), 
               dtype=(torch.float8_e4m3fn, torch.float32)
            b: tuple of (torch.Tensor of shape (num_groups, n, k), torch.Tensor of shape (num_groups, ceil_div(n, 128), ceil_div(k, 128))),
               dtype=(torch.float8_e4m3fn, torch.float32)
            m_indices: torch.Tensor of shape (m,), dtype=torch.int32

        Returns:
            d: torch.Tensor of shape (m, n), dtype=torch.bfloat16
        """
        a_data, a_sf = a
        b_data, b_sf = b
        m, k = a_data.shape
        num_groups, n, k_b = b_data.shape
        assert k == k_b
        assert m_indices.shape[0] == m

        # Dequantize a to float32 (per-token scaling)
        # a_sf shape: [m, ceil_div(k, 128)]
        # Each row of a_sf corresponds to a 128-element block in k dimension
        # k_blocks = a_sf.shape[1]
        block_size_k = 128  # Default block size from DeepGEMM

        # Expand a_sf to match k dimension: [m, k]
        # For each row i, expand a_sf[i, :] to [k] by repeating each block_size_k times
        a_sf_expanded = a_sf.unsqueeze(-1)  # [m, k_blocks, 1]
        a_sf_expanded = a_sf_expanded.repeat_interleave(block_size_k, dim=-1)  # [m, k_blocks, block_size_k]
        a_sf_expanded = a_sf_expanded.view(m, -1)  # [m, k_blocks * block_size_k]
        # Handle case where k is not exactly divisible by block_size_k
        if a_sf_expanded.shape[1] > k:
            a_sf_expanded = a_sf_expanded[:, :k]
        elif a_sf_expanded.shape[1] < k:
            # Pad with last value if needed
            padding = a_sf_expanded[:, -1:].repeat(1, k - a_sf_expanded.shape[1])
            a_sf_expanded = torch.cat([a_sf_expanded, padding], dim=1)
        a_fp32 = a_data.float() * a_sf_expanded

        d = torch.zeros((m, n), dtype=torch.bfloat16, device=a_data.device)

        for g in range(num_groups):
            group_mask = (m_indices == g)
            group_indices = torch.where(group_mask)[0]
            if len(group_indices) > 0:
                group_a = a_fp32[group_indices]  # [group_size, k]
                # Dequantize b for this group (per-block scaling)
                b_group_data = b_data[g]  # [n, k]
                b_group_sf = b_sf[g]  # [ceil_div(n, 128), ceil_div(k, 128)]

                # Expand b_sf to match n and k dimensions (per-block scaling)
                # n_blocks = b_group_sf.shape[0]
                # k_blocks_b = b_group_sf.shape[1]
                block_size_n = 128  # Default block size
                block_size_k_b = 128  # Default block size

                # Expand along n dimension first
                b_sf_expanded = b_group_sf.unsqueeze(-1)  # [n_blocks, k_blocks, 1]
                b_sf_expanded = b_sf_expanded.repeat_interleave(block_size_n,
                                                                dim=0)  # [n_blocks * block_size_n, k_blocks, 1]
                if b_sf_expanded.shape[0] > n:
                    b_sf_expanded = b_sf_expanded[:n]
                elif b_sf_expanded.shape[0] < n:
                    padding = b_sf_expanded[-1:].repeat(n - b_sf_expanded.shape[0], 1, 1)
                    b_sf_expanded = torch.cat([b_sf_expanded, padding], dim=0)

                # Expand along k dimension
                b_sf_expanded = b_sf_expanded.unsqueeze(-1)  # [n, k_blocks, 1, 1]
                b_sf_expanded = b_sf_expanded.repeat_interleave(block_size_k_b,
                                                                dim=2)  # [n, k_blocks, block_size_k_b, 1]
                b_sf_expanded = b_sf_expanded.view(n, -1)  # [n, k_blocks * block_size_k_b]
                if b_sf_expanded.shape[1] > k:
                    b_sf_expanded = b_sf_expanded[:, :k]
                elif b_sf_expanded.shape[1] < k:
                    padding = b_sf_expanded[:, -1:].repeat(1, k - b_sf_expanded.shape[1])
                    b_sf_expanded = torch.cat([b_sf_expanded, padding], dim=1)

                b_fp32 = b_group_data.float() * b_sf_expanded
                # Compute GEMM
                d[group_indices] = torch.matmul(group_a, b_fp32.t()).to(torch.bfloat16)

        # Zero out padding rows
        padding_mask = (m_indices == -1)
        d[padding_mask, :] = 0

        return d
