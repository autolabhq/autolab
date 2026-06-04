#pragma once

#include <cuda_runtime.h>

/*
 * ICP correspondence step on GPU.
 *
 * Inputs (all device pointers):
 *   src        : N * 3 floats (row-major: x0,y0,z0, x1,y1,z1, ...)
 *   tgt        : M * 3 floats (row-major)
 *   N          : number of source points
 *   M          : number of target points
 *   d_max      : maximum correspondence distance; pairs with squared
 *                distance > d_max*d_max are rejected.
 *   kd_nodes   : flat KD-tree over the target point cloud (built once
 *                during task setup; agent must not rebuild it). See
 *                kdtree.h for the node layout.
 *   n_nodes    : number of KD-tree nodes.
 *
 * Outputs (all device pointers, written by the call):
 *   H_out      : 9 floats. Row-major 3x3 cross-covariance matrix
 *                  H = sum_i (p_i - p_centroid) (q_i - q_centroid)^T
 *                where q_i = NN(p_i) restricted to valid pairs.
 *                fp64 should be used for the running sum; the final
 *                value is cast to fp32.
 *   err_out    : 1 float. Sum of squared distances over valid pairs.
 *   count_out  : 1 int. Number of valid correspondences.
 *
 * The function may launch multiple kernels and use the supplied stream
 * for ordering. It MUST NOT cudaMalloc / cudaFree / cudaMemcpy host
 * data inside the call; any persistent scratch space must be cached in
 * file-scope static variables (allocated lazily on first call).
 */
void icp_correspondence(const float* src,
                        const float* tgt,
                        int N, int M,
                        float d_max,
                        const void* kd_nodes,
                        int n_nodes,
                        float* H_out,
                        float* err_out,
                        int* count_out,
                        cudaStream_t stream);
