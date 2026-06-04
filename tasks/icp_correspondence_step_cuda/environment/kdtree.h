#pragma once

#include <cstdint>

/*
 * Flat KD-tree node over a 3D point cloud.
 *
 * The tree is stored as an array of KDNode, with the root at index 0.
 * It is a balanced left-balanced tree built by recursive median split
 * along the axis of largest extent at each node.
 *
 * For internal nodes:
 *   axis     : 0, 1, 2  -- splitting axis (x, y, z).
 *   point_idx: index into the original target array of the node's pivot
 *              point (which lies on the splitting plane).
 *   left     : index of left subtree root, or -1 if absent.
 *   right    : index of right subtree root, or -1 if absent.
 *   bbox_min : axis-aligned bounding box of all points in this subtree.
 *   bbox_max : axis-aligned bounding box of all points in this subtree.
 *
 * For leaf nodes:
 *   axis = 3, left = -1, right = -1; point_idx is the only target index
 *   represented by the node, and bbox_min == bbox_max == that point.
 *
 * The tree is built once on the host before timing begins and uploaded
 * to device memory; the agent must NOT rebuild it inside
 * icp_correspondence().
 */
struct KDNode {
    float bbox_min[3];
    float bbox_max[3];
    int   point_idx;
    int   left;
    int   right;
    int   axis;        /* 0=x, 1=y, 2=z, 3=leaf */
};
