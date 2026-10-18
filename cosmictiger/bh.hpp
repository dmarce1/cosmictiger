/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef BH_HPP_
#define BH_HPP_

#include <cmath>

#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fixed_capacity_vector.hpp>
#include <cosmictiger/range.hpp>

struct BhTreeNode {
    Array<float, NDIM> pos;
    Array<int, NDIM> children;
    Pair<int> parts;
    float mass;
    float radius;
    float meanSquaredRadius;
    BhTreeNode() {
        children[LEFT] = children[RIGHT] = -1;
        mass = 0;
        radius = 0;
        meanSquaredRadius = 0;
    }
};

inline constexpr float kBhMaximumRelativePotentialError = 0.04f;

namespace detail {
// ROCKSTAR Appendix B, Eq. B3, applied to the nearest point in the sink
// bucket. Its Newtonian error bound only applies outside every source's
// compact softening support, so overlapping supports must be opened.
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
    Array<float, NDIM> x;
    float m;
};

DeviceVector<float> bhEvaluatePotential(DeviceVector<Array<float, NDIM>>& x, bool gpu = false);
DeviceVector<float> bhEvaluatePoints(
    DeviceVector<Array<float, NDIM>>& y, DeviceVector<Array<float, NDIM>>& x, bool gpu = false);
DeviceVector<float> bhEvaluatePotentialGpu(DeviceVector<BhTreeNode> const& treeNodes,
    DeviceVector<Array<float, NDIM>> const& x, DeviceVector<int> const sinkBuckets,
    float relativeError, float hsoft, float gm, int maximumBlockCount = 0);
DeviceVector<float> bhEvaluatePotentialPointsGpu(DeviceVector<BhTreeNode> const& treeNodes,
    DeviceVector<Array<float, NDIM>> const& x, DeviceVector<Array<float, NDIM>> const& y,
    float relativeError, float hsoft, float gm);

#endif /* BH_HPP_ */
