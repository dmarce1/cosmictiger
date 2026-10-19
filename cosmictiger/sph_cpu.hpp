/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <vector>

#include <cosmictiger/simd_math.hpp>
#include <cosmictiger/sph_kernel.hpp>
#include <cosmictiger/vector3.hpp>

// Each invocation owns one target leaf. Sources remain immutable for the entire
// phase, including when another leaf updates the corresponding local sink.
SphRunReturn sphRunCpuNode(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);
SphRunReturn sphCpuPrehydro1(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);
SphRunReturn sphCpuPrehydro2(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);
SphRunReturn sphCpuHydro(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);
SphRunReturn sphCpuRungs(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);

#ifndef __CUDACC__
namespace detail {

inline std::vector<int> sphCpuNeighbors(SphRunData const& data, SphTreeNode const& targetNode,
    bool symmetric, bool activeOnly = false, int minimumRung = 0) {
    std::vector<int> neighbors;
    for (int nodeIndex = targetNode.neighborRange.first;
        nodeIndex < targetNode.neighborRange.second; nodeIndex++) {
        auto const& neighborNode = data.treeNodes[data.neighborNodeIndices[nodeIndex]];
        for (int sourceIndex = neighborNode.particleRange.first;
            sourceIndex < neighborNode.particleRange.second; sourceIndex++) {
            if (activeOnly && data.sourceRungs[sourceIndex] < minimumRung) {
                continue;
            }
            Array<Fixed32, NDIM> position;
            position[XDIM] = data.sourcePositionX[sourceIndex];
            position[YDIM] = data.sourcePositionY[sourceIndex];
            position[ZDIM] = data.sourcePositionZ[sourceIndex];
            if (targetNode.outerBox.contains(position) ||
                (symmetric &&
                    sphSupportIntersects(
                        targetNode.innerBox, position, data.sourceSupportRadius[sourceIndex]))) {
                neighbors.push_back(sourceIndex);
            }
        }
    }
    return neighbors;
}

inline SimdFloat sphCpuLoad(
    float const* values, std::vector<int> const& neighbors, size_t first, float padding) {
    return SimdFloat([&](auto lane) {
        size_t const index = first + size_t(lane);
        return index < neighbors.size() ? values[neighbors[index]] : padding;
    });
}

inline SimdFloat::mask_type sphCpuValidLanes(std::vector<int> const& neighbors, size_t first) {
    return SimdFloat([&](auto lane) {
        return first + size_t(lane) < neighbors.size() ? 1.0f : 0.0f;
    }) > SimdFloat(0.0f);
}

inline Vector3<SimdFloat> sphCpuLoad3(float const* x, float const* y, float const* z,
    std::vector<int> const& neighbors, size_t first, float padding) {
    return Vector3<SimdFloat>(sphCpuLoad(x, neighbors, first, padding),
        sphCpuLoad(y, neighbors, first, padding), sphCpuLoad(z, neighbors, first, padding));
}

// Sink minus source positions, r_a - r_b, for one pack of neighbors.
inline Vector3<SimdFloat> sphCpuSeparations(
    SphRunData const& data, int particleIndex, std::vector<int> const& neighbors, size_t first) {
    Fixed32 const* positions[NDIM] = {
        data.sourcePositionX, data.sourcePositionY, data.sourcePositionZ};
    Vector3<SimdFloat> separation;
    for (int dim = 0; dim < NDIM; dim++) {
        int const sinkPosition = static_cast<int32_t>(positions[dim][particleIndex].raw());
        SimdInt const sourcePosition([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ?
                static_cast<int32_t>(positions[dim][neighbors[index]].raw()) :
                sinkPosition;
        });
        separation[dim] =
            simdCast<SimdFloat>(simdPositionDifference(SimdInt(sinkPosition), sourcePosition)) *
            SimdFloat(kFixed2float);
    }
    return separation;
}

}    // namespace detail
#endif
