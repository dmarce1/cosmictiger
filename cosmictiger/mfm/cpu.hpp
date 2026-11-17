/*
 CosmicTiger - A cosmological N-Body code
 Typed CPU MFM preparation entry points and precision-matched SIMD gathers.
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <vector>

#include <cosmictiger/mfm/kernel.hpp>
#include <cosmictiger/mfm/run_data.hpp>

namespace mfm {

template <typename Real, int dimensions>
PrehydroResult<Real> smoothingLengthsCpu(PrehydroParameters<Real> const& parameters,
    RunData<Real, dimensions> const& data, int targetNodeIndex);

template <typename Real, int dimensions>
PrehydroResult<Real> densityAndGeometryCpu(PrehydroParameters<Real> const& parameters,
    RunData<Real, dimensions> const& data, int targetNodeIndex);

#ifndef __CUDACC__
namespace detail {

    inline constexpr int kCpuLaneCount = 8;

    template <typename Real>
    using Pack = SimdVector<Real, kCpuLaneCount>;

    template <typename Real, int dimensions>
    inline std::vector<int> neighbors(
        RunData<Real, dimensions> const& data, SphTreeNode const& targetNode, bool symmetric) {
        std::vector<int> result;
        for (int nodeIndex = targetNode.neighborRange.first;
            nodeIndex < targetNode.neighborRange.second; ++nodeIndex) {
            auto const& node = data.treeNodes[data.neighborNodeIndices[nodeIndex]];
            for (int source = node.particleRange.first; source < node.particleRange.second;
                ++source) {
                auto const position = data.sourcePaddedPosition(source);
                if (targetNode.outerBox.contains(position) ||
                    (symmetric &&
                        supportIntersects(
                            targetNode.innerBox, position, data.sourceSmoothingLength[source]))) {
                    result.push_back(source);
                }
            }
        }
        return result;
    }

    template <typename Real>
    inline Pack<Real> load(
        Real const* values, std::vector<int> const& sources, size_t first, Real padding) {
        return Pack<Real>([&](auto lane) {
            size_t const index = first + size_t(lane);
            return index < sources.size() ? values[sources[index]] : padding;
        });
    }

    template <typename Real>
    inline typename Pack<Real>::mask_type validLanes(
        std::vector<int> const& sources, size_t first) {
        return Pack<Real>([&](auto lane) {
            return first + size_t(lane) < sources.size() ? Real(1) : Real(0);
        }) > Pack<Real>(Real(0));
    }

    // r𞁑𞁓=x𞁑−x𞁓, with modular integer subtraction before Real conversion.
    // Symbol directory: x𞁑, x𞁓 → sink and packed source coordinates; r𞁑𞁓 → separation pack.
    template <typename Real, int dimensions>
    inline math::Vector<Pack<Real>, dimensions> separations(RunData<Real, dimensions> const& data,
        int particleIndex, std::vector<int> const& sources, size_t first) {
        using PositionPack = SimdVector<SignedPositionInteger, kCpuLaneCount>;
        Position const* const components[kDimensions] = {
            data.sourcePositionX, data.sourcePositionY, data.sourcePositionZ};
        math::Vector<Pack<Real>, dimensions> r𞁑𞁓;
        for (int dim = 0; dim < dimensions; ++dim) {
            SignedPositionInteger const x𞁑 =
                static_cast<SignedPositionInteger>(components[dim][particleIndex].raw());
            PositionPack const x𞁓([&](auto lane) {
                size_t const index = first + size_t(lane);
                return index < sources.size() ?
                    static_cast<SignedPositionInteger>(components[dim][sources[index]].raw()) :
                    x𞁑;
            });
            r𞁑𞁓[dim] = simdCast<Pack<Real>>(simdPositionDifference(PositionPack(x𞁑), x𞁓)) *
                Pack<Real>(Real(kPositionToFloat));
        }
        return r𞁑𞁓;
    }

}    // namespace detail
#endif

}    // namespace mfm
