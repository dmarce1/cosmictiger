/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <vector>

#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/run_data.hpp>

// Each invocation owns one target leaf. Sources remain immutable for the entire
// phase, including when another leaf updates the corresponding local sink.
SphRunReturn sphRunCpuNode(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);
SphRunReturn sphCpuSmoothingLengths(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);
SphRunReturn sphCpuDensityAndGradients(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);
SphRunReturn sphCpuHydro(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);
SphRunReturn sphCpuRungs(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex);

#ifndef __CUDACC__
namespace detail {

// paddedX_b → neighbor position as the record of the box tests.
inline std::vector<int> sphCpuNeighbors(SphRunData const& data, SphTreeNode const& targetNode,
    bool symmetric, sph_flops::Counter& flops, bool activeOnly = false, int minimumRung = 0) {
    std::vector<int> neighbors;
    for (int nodeIndex = targetNode.neighborRange.first;
        nodeIndex < targetNode.neighborRange.second; nodeIndex++) {
        auto const& neighborNode = data.treeNodes[data.neighborNodeIndices[nodeIndex]];
        for (int sourceIndex = neighborNode.particleRange.first;
            sourceIndex < neighborNode.particleRange.second; sourceIndex++) {
            if (activeOnly && data.sourceRungs[sourceIndex] < minimumRung) {
                continue;
            }
            Vector3<Position> const paddedX_b = data.sourcePaddedPosition(sourceIndex);
            if (targetNode.outerBox.contains(paddedX_b, &flops.doublePrecision) ||
                (symmetric &&
                    sphSupportIntersects(targetNode.innerBox, paddedX_b,
                        data.sourceSmoothingLength[sourceIndex], &flops))) {
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

// A pack of neighbor vectors from the separate component arrays, over the active axes.
inline sph_geometry::SpatialVector<SimdFloat> sphCpuLoad3(float const* x, float const* y,
    float const* z, std::vector<int> const& neighbors, size_t first, float padding) {
    float const* const components[kDimensions] = {x, y, z};
    sph_geometry::SpatialVector<SimdFloat> result;
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        result[dim] = sphCpuLoad(components[dim], neighbors, first, padding);
    }
    return result;
}

// Separations r𞁑𞁓 = x𞁑 - x𞁓 for one pack of neighbors, over the active axes.
// x𞁑, x𞁓 → sink, neighbor coordinate along one axis; r𞁑𞁓 → separation (return value).
inline sph_geometry::SpatialVector<SimdFloat> sphCpuSeparations(
    SphRunData const& data, int particleIndex, std::vector<int> const& neighbors, size_t first) {
    Position const* positions[kDimensions] = {
        data.sourcePositionX, data.sourcePositionY, data.sourcePositionZ};
    sph_geometry::SpatialVector<SimdFloat> r𞁑𞁓;
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        SignedPositionInteger const x𞁑 =
            static_cast<SignedPositionInteger>(positions[dim][particleIndex].raw());
        SimdPosition const x𞁓([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < neighbors.size() ?
                static_cast<SignedPositionInteger>(positions[dim][neighbors[index]].raw()) :
                x𞁑;
        });
        r𞁑𞁓[dim] = simdCast<SimdFloat>(simdPositionDifference(SimdPosition(x𞁑), x𞁓)) *
            SimdFloat(kPositionToFloat);
    }
    return r𞁑𞁓;
}

// The component-wise additive reduction of a vector of SIMD partial sums.
inline sph_geometry::SpatialVector<float> sphCpuSum(
    sph_geometry::SpatialVector<SimdFloat> const& pack) {
    sph_geometry::SpatialVector<float> sum;
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        sum[dim] = simdSum(pack[dim]);
    }
    return sum;
}

// One lane of a vector pack.
inline sph_geometry::SpatialVector<float> sphCpuLane(
    sph_geometry::SpatialVector<SimdFloat> const& pack, size_t lane) {
    sph_geometry::SpatialVector<float> value;
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        value[dim] = float(pack[dim][lane]);
    }
    return value;
}

}    // namespace detail
#endif
