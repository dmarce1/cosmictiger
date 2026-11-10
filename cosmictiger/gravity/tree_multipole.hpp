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
// fmmDispatch(). Both write the leading FmmOrder<order>::kMultipoleCoefficients_
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
    Vector3<Position> const& rightCenter, Vector3<Position> const& center);
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
template <int order>
FlopCount treeSourceToMesh(double* output, SourceMultipoleBuffer const& source,
    Vector3<Position> const& sourceCenter, Vector3<Position> const& meshCenter, int meshSize);

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
// The definitions pull in the generated kernels, so only the per-order instance
// translation units see them; tree.cpp compiles against the declarations above.

#include <cosmictiger/gravity/kernels/basis_conversion.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/math/simd_math.hpp>

template <int order>
FlopCounts treeComputeMultipole(MultipoleBuffer& multi, MultipoleBuffer const& left,
    MultipoleBuffer const& right, Vector3<double> const& leftCenter,
    Vector3<double> const& rightCenter, Vector3<double> const& center) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    Vector3<SimdDouble> mdx{};
    cosmictiger::gravity::cartesian::tracefree::Multipole<SimdDouble, order> simdM{};
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        simdSetLane(simdM[i], kLeftChild, left[i]);
        simdSetLane(simdM[i], kRightChild, right[i]);
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        simdSetLane(mdx[dim], kLeftChild, leftCenter[dim] - center[dim]);
        simdSetLane(mdx[dim], kRightChild, rightCenter[dim] - center[dim]);
    }
    auto const source = simdM;
    simdM = cosmictiger::gravity::cartesian::tracefree::Multipole<SimdDouble, order>{};
    cosmictiger::gravity::cartesian::tracefree::M2M(simdM, source, fmmScaleDisplacement(mdx));
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        multi[i] = simdM[i][kLeftChild] + simdM[i][kRightChild];
    }
    // M2M is double, including two input widenings and one output narrowing
    // per stored coefficient. Only the two populated SIMD lanes are counted.
    FlopCounts counts;
    counts.singlePrecision = 0;
    counts.doublePrecision = 2 *
            cosmictiger::gravity::cartesian::tracefree::detail::kM2MFlops<
                FmmOrder<order>::kMultipoleDegree_> +
        12 + 4 * kMultipoleCoefficients;
    counts.unclassified = 0;
    return counts;
}

template <int order, class Buffer, bool wideReduction = false>
FlopCounts treeComputeMixedMultipoleImpl(Buffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<double> const& center, FmmSourceView const& sources) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    static SimdFloat const toFloatFactor = kPositionToFloat;
    cosmictiger::gravity::cartesian::tracefree::Multipole<double, order> mValue;
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
        cosmictiger::gravity::cartesian::tracefree::Multipole<SimdFloat, order> m{};
        cosmictiger::gravity::cartesian::tracefree::P2M(
            m, math::broadcast<SimdFloat>(1), fmmScaleDisplacement(dx));
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
            (cosmictiger::gravity::cartesian::tracefree::detail::kP2MFlops<
                 FmmOrder<order>::kMultipoleDegree_> +
                6 + kMultipoleCoefficients) +
        (wideReduction ? 0 : packets * (kSimdFloatSize - 1) * kMultipoleCoefficients);
    counts.doublePrecision = centerConversions +
        ((wideReduction ? 2 * kSimdFloatSize : 2) * packets +
            int(std::is_same_v<Buffer, MultipoleBuffer>)) *
            kMultipoleCoefficients;
    counts.unclassified = 0;
    return counts;
}

template <int order>
FlopCounts treeComputeMultipole(MultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<double> const& center, FmmSourceView const& sources) {
    return treeComputeMixedMultipoleImpl<order>(multi, particleRange, center, sources);
}
template <int order>
FlopCounts treeComputeMixedMultipole(SourceMultipoleBuffer& multi,
    Pair<ParticleInteger> particleRange, Vector3<double> const& center,
    FmmSourceView const& sources) {
    return treeComputeMixedMultipoleImpl<order>(multi, particleRange, center, sources);
}
template <int order>
FlopCounts treeComputeWideReductionMultipole(SourceMultipoleBuffer& multi,
    Pair<ParticleInteger> particleRange, Vector3<double> const& center,
    FmmSourceView const& sources) {
    return treeComputeMixedMultipoleImpl<order, SourceMultipoleBuffer, true>(
        multi, particleRange, center, sources);
}

template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, TreeBuildMoments const& left,
    TreeBuildMoments const& right, Vector3<Position> const& leftCenter,
    Vector3<Position> const& rightCenter, Vector3<Position> const& center) {
    namespace cartesian = cosmictiger::gravity::cartesian::tracefree;
    constexpr int kCount = FmmOrder<order>::kMultipoleCoefficients_;
    cartesian::Multipole<SimdDouble, order> source{}, translated{};
    Vector3<SimdDouble> displacement{};
    for (int i = 0; i < kCount; ++i) {
        simdSetLane(source[i], kLeftChild, left[i]);
        simdSetLane(source[i], kRightChild, right[i]);
    }
    for (int dim = 0; dim < kDimensions; ++dim) {
        simdSetLane(
            displacement[dim], kLeftChild, unwrappedDoubleDistance(leftCenter[dim], center[dim]));
        simdSetLane(
            displacement[dim], kRightChild, unwrappedDoubleDistance(rightCenter[dim], center[dim]));
    }
    cartesian::M2M(translated, source, fmmScaleDisplacement(displacement));
    for (int i = 0; i < kCount; ++i) {
        multi[i] = translated[i][kLeftChild] + translated[i][kRightChild];
    }
    FlopCounts counts{};
    counts.doublePrecision = 2 * cartesian::detail::kM2MFlops<order - 2> + 18 +
        (1 + int(!left.isDouble()) + int(!right.isDouble())) * kCount;
    return counts;
}

template <int order>
FlopCounts treeComputeMultipole(SourceMultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Vector3<Position> const& center, FmmSourceView const& sources) {
    namespace cartesian = cosmictiger::gravity::cartesian::tracefree;
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
        cartesian::Multipole<SimdDouble, order> packet{};
        cartesian::P2M(packet, mass, fmmScaleDisplacement(displacement));
        for (int i = 0; i < kCount; ++i) {
            multi[i] += simdSum(packet[i]);
        }
    }
    FlopCount const packets = (particleRange.second - particleRange.first + kLanes - 1) / kLanes;
    FlopCounts counts{};
    counts.doublePrecision = FlopCount(particleRange.second - particleRange.first) * 7 +
        packets * (kLanes * (cartesian::detail::kP2MFlops<order - 2> + 3) + kLanes * kCount);
    return counts;
}

template <int order>
FlopCount treeSourceToMesh(double* output, SourceMultipoleBuffer const& source,
    Vector3<Position> const& sourceCenter, Vector3<Position> const& meshCenter, int meshSize) {
    namespace cartesian = cosmictiger::gravity::cartesian::tracefree;
    namespace conversion = cosmictiger::gravity::basis_conversion;
    cartesian::Multipole<double, order> input{}, shifted{};
    cosmictiger::gravity::spherical::Multipole<double, order> spherical;
    for (int i = 0; i < input.size(); ++i) {
        input[i] = source[i];
    }
    Vector3<double> displacement;
    for (int dim = 0; dim < kDimensions; ++dim) {
        displacement[dim] = unwrappedDoubleDistance(sourceCenter[dim], meshCenter[dim]);
    }
    cartesian::M2M(shifted, input, fmmScaleDisplacement(displacement));
    conversion::sourceToSpherical(spherical, shifted);
    // Derived boundary conversion (docs/fmm_basis_conversion.md):
    // Sₙ(mesh) = (κh)⁻ⁿ Aₙ Cₙ(tree), h=1/meshSize, κ=kScaleFactor.
    // Symbol directory: S is a spherical moment; C is a Cartesian STF moment;
    // A is the pure source basis map; h is mesh width; κ is tree coordinate scale.
    double const ratio = double(meshSize) / kScaleFactor;
    double factor = 1;
    for (int n = 0; n <= order - 2; ++n) {
        if (n) {
            factor *= ratio;
        }
        for (int m = -n; m <= n; ++m) {
            int const slot = n * (n + 1) + m;
            output[slot] = n ? spherical[slot] * factor : spherical[slot];
        }
    }
    return cartesian::detail::kM2MFlops<order - 2> +
        conversion::kSourceToSphericalConventionalFlops<order> + 9 + input.size() + order - 2;
}

#endif
