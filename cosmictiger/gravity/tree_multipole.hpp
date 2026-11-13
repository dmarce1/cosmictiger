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
// order-agnostic and dispatches to exactly one of these per node through
// the build order. Both write the leading FmmOrder<order>::kMultipoleCoefficients_
// entries of `multi` and return the flop count.

// Internal node: translate the children's multipoles to `center` and sum them.
template <int order>
FlopCounts treeComputeMultipole(MultipoleBuffer& multi, MultipoleBuffer const& left,
    MultipoleBuffer const& right, Vector3<double> const& leftCenter,
    Vector3<double> const& rightCenter, Vector3<double> const& center);

// Leaf: accumulate P2M over the local particles of `particleRange` about `center`.
template <int order>
FlopCounts treeComputeMultipole(MultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<double> const& center,
    FmmSourceView const& sources = fmmSourceView(FmmParticleStore::Owned));

template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, TreeBuildMoments const& left,
    TreeBuildMoments const& right, Vector3<Position> const& leftCenter,
    Vector3<Position> const& rightCenter, Vector3<Position> const& center,
    FmmGeometry const& geometry = {});
template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<Position> const& center, FmmSourceView const& sources);
// Accuracy experiment: preserve the existing mixed leaf accumulator before its
// final narrowing. Production owned sources use the full-double overload above.
template <int order>
FlopCounts treeComputeMixedMultipole(SourceMultipoleBuffer& multi,
    Pair<ParticleInteger> particleRange, Vector3<double> const& center,
    FmmSourceView const& sources);
template <int order>
FlopCounts treeComputeWideReductionMultipole(SourceMultipoleBuffer& multi,
    Pair<ParticleInteger> particleRange, Vector3<double> const& center,
    FmmSourceView const& sources);

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
// The definitions pull in the generated kernels, so only the per-order instance
// translation units see them; tree.cpp compiles against the declarations above.

#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/math/simd_math.hpp>

/// Translate two child multipoles to the parent center in double SIMD and combine them into
/// float transport storage.
template <int order>
FlopCounts treeComputeMultipole(MultipoleBuffer& multi, MultipoleBuffer const& left,
    MultipoleBuffer const& right, Vector3<double> const& leftCenter,
    Vector3<double> const& rightCenter, Vector3<double> const& center) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    Vector3<SimdDouble> mdx{};
    cosmictiger::gravity::cartesian::Multipole<SimdDouble> Mo{};
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        simdSetLane(Mo[i], kLeftChild, left[i]);
        simdSetLane(Mo[i], kRightChild, right[i]);
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        simdSetLane(mdx[dim], kLeftChild, leftCenter[dim] - center[dim]);
        simdSetLane(mdx[dim], kRightChild, rightCenter[dim] - center[dim]);
    }
    auto const Mi = Mo;
    Mo = cosmictiger::gravity::cartesian::Multipole<SimdDouble>{};
    cosmictiger::gravity::cartesian::M2M(Mi, fmmScaleDisplacement(mdx), Mo);
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        multi[i] = Mo[i][kLeftChild] + Mo[i][kRightChild];
    }
    // M2M is double, including two input widenings and one output narrowing
    // per stored coefficient. Only the two populated SIMD lanes are counted.
    FlopCounts counts;
    counts.singlePrecision = 0;
    counts.doublePrecision =
        2 * cosmictiger::gravity::cartesian::kM2MFlops<FmmOrder<order>::kMultipoleDegree_> + 12 +
        4 * kMultipoleCoefficients;
    counts.unclassified = 0;
    return counts;
}

/// Evaluate particle moments in float SIMD and accumulate into the selected precision,
/// optionally widening lane reduction.
template <int order, class Buffer, bool wideReduction = false>
FlopCounts treeComputeMixedMultipoleImpl(Buffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<double> const& center, FmmSourceView const& sources) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    static SimdFloat const toFloatFactor = kPositionToFloat;
    cosmictiger::gravity::cartesian::Multipole<double> mValue;
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        mValue[i] = 0.0;
    }
    ParticleInteger const maxi =
        roundUp(particleRange.second - particleRange.first, (ParticleInteger) kSimdFloatSize) +
        particleRange.first;
    Vector3<SimdPosition> y{};
    for (int dim = 0; dim < kDimensions; dim++) {
        y[dim] = math::broadcast<SimdPosition>(Position(center[dim]).raw());
    }
    for (ParticleInteger i = particleRange.first; i < maxi; i += kSimdFloatSize) {
        Vector3<SimdPosition> xValue{};
        SimdFloat mask{};
        ParticleInteger const maxj = std::min(i + kSimdFloatSize, particleRange.second);
        for (ParticleInteger j = i; j < maxj; j++) {
            for (int dim = 0; dim < kDimensions; dim++) {
                simdSetLane(xValue[dim], j - i, sources.positions[dim][j].raw());
            }
            simdSetLane(mask, j - i, fmmSourceMass(sources, j));
        }
        for (ParticleInteger j = maxj; j < i + kSimdFloatSize; j++) {
            simdSetLane(mask, j - i, 0.f);
            for (int dim = 0; dim < kDimensions; dim++) {
                simdSetLane(xValue[dim], j - i, sources.positions[dim][maxj - 1].raw());
            }
        }
        Vector3<SimdFloat> dx{};
        for (int dim = 0; dim < kDimensions; dim++) {
            dx[dim] =
                simdCast<SimdFloat>(simdPositionDifference(xValue[dim], y[dim])) * toFloatFactor;
        }
        cosmictiger::gravity::cartesian::Multipole<SimdFloat> m{};
        cosmictiger::gravity::cartesian::P2M(
            math::broadcast<SimdFloat>(1), fmmScaleDisplacement(dx), m);
        for (int j = 0; j < kMultipoleCoefficients; j++) {
            m[j] *= mask;
            if constexpr (wideReduction) {
                double packetSum = double(m[j][0]);
                for (int lane = 1; lane < kSimdFloatSize; ++lane) {
                    packetSum += double(m[j][lane]);
                }
                mValue[j] += packetSum;
            } else {
                mValue[j] += simdSum(m[j]);
            }
        }
    }
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        multi[i] = mValue[i];
    }
    // Every SIMD lane executes P2M, including the zero-mass padding. Each
    // coefficient has a lane multiply and a reduction plus scalar accumulation.
    FlopCount centerConversions = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        centerConversions +=
            (kPositionBits == 64 ? 4 : 1) + FlopCount(!(center[dim] > -1.0 && center[dim] < 1.0));
    }
    FlopCount const lanes = maxi - particleRange.first;
    FlopCount const packets = lanes / kSimdFloatSize;
    // P2M, lane mass weights, and SIMD reductions are float. Each reduced
    // coefficient widens and accumulates in double, then narrows into storage.
    FlopCounts counts;
    counts.singlePrecision = lanes *
            (cosmictiger::gravity::cartesian::kP2MFlops<FmmOrder<order>::kMultipoleDegree_> + 6 +
                kMultipoleCoefficients) +
        (wideReduction ? 0 : packets * (kSimdFloatSize - 1) * kMultipoleCoefficients);
    counts.doublePrecision = centerConversions +
        ((wideReduction ? 2 * kSimdFloatSize : 2) * packets +
            int(std::is_same_v<Buffer, MultipoleBuffer>)) *
            kMultipoleCoefficients;
    counts.unclassified = 0;
    return counts;
}

/// Evaluate mixed-precision leaf moments and store the result in the float transport
/// buffer.
template <int order>
FlopCounts treeComputeMultipole(MultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<double> const& center, FmmSourceView const& sources) {
    return treeComputeMixedMultipoleImpl<order>(multi, particleRange, center, sources);
}
/// Compute mixed-precision leaf moments while retaining their double-precision
/// accumulation.
template <int order>
FlopCounts treeComputeMixedMultipole(SourceMultipoleBuffer& multi,
    Pair<ParticleInteger> particleRange, Vector3<double> const& center,
    FmmSourceView const& sources) {
    return treeComputeMixedMultipoleImpl<order>(multi, particleRange, center, sources);
}
/// Compute float SIMD moments with double-precision lane reduction and retained double
/// source storage.
template <int order>
FlopCounts treeComputeWideReductionMultipole(SourceMultipoleBuffer& multi,
    Pair<ParticleInteger> particleRange, Vector3<double> const& center,
    FmmSourceView const& sources) {
    return treeComputeMixedMultipoleImpl<order, SourceMultipoleBuffer, true>(
        multi, particleRange, center, sources);
}

/// Translate both child moments with native fixed-position displacements and retain the
/// parent sum in double precision.
template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, TreeBuildMoments const& left,
    TreeBuildMoments const& right, Vector3<Position> const& leftCenter,
    Vector3<Position> const& rightCenter, Vector3<Position> const& center,
    FmmGeometry const& geometry) {
    namespace cartesian = cosmictiger::gravity::cartesian;
    constexpr int kCount = FmmOrder<order>::kMultipoleCoefficients_;
    cartesian::Multipole<SimdDouble> Mi{}, Mo{};
    Vector3<SimdDouble> displacement{};
    FlopCounts frameFlops{};
    for (int i = 0; i < kCount; ++i) {
        simdSetLane(Mi[i], kLeftChild, left[i]);
        simdSetLane(Mi[i], kRightChild, right[i]);
    }
    for (int dim = 0; dim < kDimensions; ++dim) {
        auto difference = [&](Position point) {
            return geometry.frozenMembership ?
                detail::fmmCellDifference(
                    point, center[dim], geometry.sinkCell[dim], geometry.meshSize, &frameFlops) :
                unwrappedDoubleDistance(point, center[dim]);
        };
        simdSetLane(displacement[dim], kLeftChild, difference(leftCenter[dim]));
        simdSetLane(displacement[dim], kRightChild, difference(rightCenter[dim]));
    }
    cartesian::M2M(Mi, fmmScaleDisplacement(displacement), Mo);
    for (int i = 0; i < kCount; ++i) {
        multi[i] = Mo[i][kLeftChild] + Mo[i][kRightChild];
    }
    FlopCounts counts{};
    counts.doublePrecision = 2 * cartesian::kM2MFlops<order - 2> + 18 +
        (1 + int(!left.isDouble()) + int(!right.isDouble())) * kCount;
    if constexpr (kCountFlops) {
        if (geometry.frozenMembership) {
            counts.doublePrecision += frameFlops.doublePrecision - 2 * kDimensions;
        }
    }
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
    FlopCounts frameFlops{};
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
                    sources.geometry.frozenMembership ?
                        detail::fmmCellDifference(sources.positions[dim][i], center[dim],
                            sources.geometry.sinkCell[dim], sources.geometry.meshSize,
                            &frameFlops) :
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
    if constexpr (kCountFlops) {
        if (sources.geometry.frozenMembership) {
            counts.doublePrecision += frameFlops.doublePrecision -
                kDimensions * FlopCount(particleRange.second - particleRange.first);
        }
    }
    return counts;
}

#endif
