/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef BH_HPP_
#define BH_HPP_

#include <cmath>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/fixed_capacity_vector.hpp>
#include <cosmictiger/cuda/device_vector.hpp>
#include <cosmictiger/math/fixed.hpp>
#include <cosmictiger/math/range.hpp>

struct BhTreeNode {
    Array<float, kDimensions> pos;
    Array<int, kDimensions> children;
    Pair<int> parts;
    float mass;
    float radius;
    float meanSquaredRadius;
    BhTreeNode() {
        children[kLeftChild] = children[kRightChild] = -1;
        mass = 0;
        radius = 0;
        meanSquaredRadius = 0;
    }
};

// Behroozi, Wechsler & Wu (2013), https://arxiv.org/html/1110.4372v3,
// Appendix B, following Eq. (B3): potential-error target.
inline constexpr float kBhMaximumRelativePotentialError = 0.04f;

namespace detail {
// ROCKSTAR Appendix B: Eq. (B1) bounds the monopole error; Eq. (B3) inverts it.
// Here source.radius is r_max and source.meanSquaredRadius is sigma_x^2 (B2).
// Our sink-bucket extension uses its nearest possible point. Our additional
// compact-support guard keeps the Newtonian bound outside softened sources.
CUDA_EXPORT inline bool bhAcceptMonopole(float distanceSquared, BhTreeNode const& source,
    float sinkRadius, float supportRadius, float relativeError) {
    float const halfRadius = 0.5f * source.radius;
    float const errorDistance =
        halfRadius + sqrtf(halfRadius * halfRadius + source.meanSquaredRadius / relativeError);
    float const supportDistance = source.radius + supportRadius;
    float const minimumDistance =
        sinkRadius + (errorDistance > supportDistance ? errorDistance : supportDistance);
    return distanceSquared > minimumDistance * minimumDistance;
}
}    // namespace detail

struct BhSource {
    Array<float, kDimensions> x;
    float m;
};

DeviceVector<float> bhEvaluatePotential(
    DeviceVector<Array<float, kDimensions>>& x, bool gpu = false);
DeviceVector<float> bhEvaluatePoints(DeviceVector<Array<float, kDimensions>>& y,
    DeviceVector<Array<float, kDimensions>>& x, bool gpu = false);
DeviceVector<float> bhEvaluatePotentialGpu(DeviceVector<BhTreeNode> const& treeNodes,
    DeviceVector<Array<float, kDimensions>> const& x, DeviceVector<int> const sinkBuckets,
    float relativeError, float hsoft, float gm, int maximumBlockCount = 0);
DeviceVector<float> bhEvaluatePotentialPointsGpu(DeviceVector<BhTreeNode> const& treeNodes,
    DeviceVector<Array<float, kDimensions>> const& x,
    DeviceVector<Array<float, kDimensions>> const& y, float relativeError, float hsoft, float gm);

#endif /* BH_HPP_ */
