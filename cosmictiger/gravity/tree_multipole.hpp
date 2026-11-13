/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <algorithm>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree_source_moments.hpp>
#include <cosmictiger/math/vector3.hpp>

// Order-templated multipole construction for treeCreate. treeCreate stays
// order-agnostic and calls one of these per node through
// the build order. Both write the leading FmmOrder<order>::kMultipoleCoefficients_
// entries of `multi` and return the flop count.

template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, SourceMultipoleBuffer const& left,
    SourceMultipoleBuffer const& right, Vector3<Position> const& leftCenter,
    Vector3<Position> const& rightCenter, Vector3<Position> const& center);
template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<Position> const& center, FmmSourceView const& sources);
#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
// The definitions pull in the generated kernels, so only the per-order instance
// translation units see them; tree.cpp compiles against the declarations above.

#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/math/simd_math.hpp>

/// Translate both child moments with native fixed-position displacements and retain the
/// parent sum in double precision.
template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, SourceMultipoleBuffer const& left,
    SourceMultipoleBuffer const& right, Vector3<Position> const& leftCenter,
    Vector3<Position> const& rightCenter, Vector3<Position> const& center) {
    namespace cartesian = cosmictiger::gravity::cartesian;
    constexpr int kCount = FmmOrder<order>::kMultipoleCoefficients_;
    cartesian::Multipole<SimdDouble> Mi{}, Mo{};
    Vector3<SimdDouble> displacement{};
    for (int i = 0; i < kCount; ++i) {
        simdSetLane(Mi[i], kLeftChild, left[i]);
        simdSetLane(Mi[i], kRightChild, right[i]);
    }
    for (int dim = 0; dim < kDimensions; ++dim) {
        simdSetLane(
            displacement[dim], kLeftChild, unwrappedDoubleDistance(leftCenter[dim], center[dim]));
        simdSetLane(
            displacement[dim], kRightChild, unwrappedDoubleDistance(rightCenter[dim], center[dim]));
    }
    cartesian::M2M(Mi, fmmScaleDisplacement(displacement), Mo);
    for (int i = 0; i < kCount; ++i) {
        multi[i] = Mo[i][kLeftChild] + Mo[i][kRightChild];
    }
    FlopCounts counts{};
    counts.doublePrecision = 2 * cartesian::kM2MFlops<order - 2> + 18 + kCount;

    return counts;
}

/// Evaluate particle-to-multipole moments in double SIMD using native fixed-coordinate
/// differences.
template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<Position> const& center, FmmSourceView const& sources) {
    namespace cartesian = cosmictiger::gravity::cartesian;
    constexpr int kCount = FmmOrder<order>::kMultipoleCoefficients_;
    constexpr int kLanes = SimdDouble::size();
    for (int i = 0; i < kCount; ++i) {
        multi[i] = 0;
    }
    for (ParticleInteger begin = particleRange.first; begin < particleRange.second;
        begin += kLanes) {
        Vector3<SimdDouble> displacement{};
        SimdDouble mass{};
        for (int lane = 0; lane < kLanes && begin + lane < particleRange.second; ++lane) {
            auto const i = begin + lane;
            for (int dim = 0; dim < kDimensions; ++dim) {
                simdSetLane(displacement[dim], lane,
                    unwrappedDoubleDistance(sources.positions[dim][i], center[dim]));
            }
            simdSetLane(mass, lane, double(fmmSourceMass(sources, i)));
        }
        cartesian::Multipole<SimdDouble> packet{};
        cartesian::P2M(mass, fmmScaleDisplacement(displacement), packet);
        for (int i = 0; i < kCount; ++i) {
            multi[i] += simdSum(packet[i]);
        }
    }
    FlopCount const packets = (particleRange.second - particleRange.first + kLanes - 1) / kLanes;
    FlopCounts counts{};
    counts.doublePrecision = FlopCount(particleRange.second - particleRange.first) * 7 +
        packets * (kLanes * (cartesian::kP2MFlops<order - 2> + 3) + kLanes * kCount);

    return counts;
}

#endif
