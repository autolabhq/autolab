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
import deep_gemm


class Model(nn.Module):
    """
    M-grouped FP8 GEMM NT contiguous implemented by DeepGEMM.
    
    SeedKernelBench-Hint:
    The path to DeepGEMM source code is in env variable DEEP_GEMM_PATH.
    The main source code of DeepGEMM m_grouped_fp8_gemm_nt_contiguous is in:
        - ${DEEP_GEMM_PATH}/csrc/apis/gemm.hpp#L147
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

        d = torch.zeros((m, n), dtype=torch.bfloat16, device=a_data.device)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices)

        # Zero out padding rows (DeepGEMM may not zero them automatically)
        padding_mask = (m_indices == -1)
        d[padding_mask, :] = 0

        return d
