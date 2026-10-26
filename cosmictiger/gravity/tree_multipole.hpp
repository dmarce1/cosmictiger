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

// Order-templated multipole construction for treeCreate. treeCreate stays
// order-agnostic and dispatches to exactly one of these per node through
// fmmDispatch(). Both write the leading FmmOrder<order>::kMultipoleCoefficients_
// entries of `multi` and return the flop count.

// Internal node: translate the children's multipoles to `center` and sum them.
template <int order>
FlopCount treeComputeMultipole(MultipoleBuffer& multi, MultipoleBuffer const& left,
    MultipoleBuffer const& right, Array<double, NDIM> const& leftCenter,
    Array<double, NDIM> const& rightCenter, Array<double, NDIM> const& center);

// Leaf: accumulate P2M over the local particles of `particleRange` about `center`.
template <int order>
FlopCount treeComputeMultipole(MultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Array<double, NDIM> const& center,
    FmmSourceView const& sources = fmmSourceView(FmmParticleStore::Owned));

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
// The definitions pull in the generated kernels, so only the per-order instance
// translation units see them; tree.cpp compiles against the declarations above.

#include <cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp>
#include <cosmictiger/math/simd.hpp>

template <int order>
FlopCount treeComputeMultipole(MultipoleBuffer& multi, MultipoleBuffer const& left,
    MultipoleBuffer const& right, Array<double, NDIM> const& leftCenter,
    Array<double, NDIM> const& rightCenter, Array<double, NDIM> const& center) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    Array<SimdDouble, NDIM> mdx{};
    cosmictiger::gravity::cartesian::Multipole<SimdDouble, order> simdM{};
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        simdSetLane(simdM[i], LEFT, left[i]);
        simdSetLane(simdM[i], RIGHT, right[i]);
    }
    for (int dim = 0; dim < NDIM; dim++) {
        simdSetLane(mdx[dim], LEFT, leftCenter[dim] - center[dim]);
        simdSetLane(mdx[dim], RIGHT, rightCenter[dim] - center[dim]);
    }
    auto const source = simdM;
    simdM = cosmictiger::gravity::cartesian::Multipole<SimdDouble, order>{};
    cosmictiger::gravity::cartesian::M2M(simdM, source, fmmScaleDisplacement(mdx));
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        multi[i] = simdM[i][LEFT] + simdM[i][RIGHT];
    }
    return 2 *
        cosmictiger::gravity::cartesian::detail::M2M<FmmOrder<order>::kMultipoleDegree_>::kFlops_ +
        12 + kMultipoleCoefficients;
}

template <int order>
FlopCount treeComputeMultipole(MultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Array<double, NDIM> const& center, FmmSourceView const& sources) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    static SimdFloat const toFloatFactor = kPositionToFloat;
    cosmictiger::gravity::cartesian::Multipole<double, order> mValue;
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        mValue[i] = 0.0;
    }
    ParticleInteger const maxi =
        roundUp(particleRange.second - particleRange.first, (ParticleInteger) SIMD_FLOAT_SIZE) +
        particleRange.first;
    Array<SimdPosition, NDIM> y{};
    for (int dim = 0; dim < NDIM; dim++) {
        y[dim] = simdBroadcast<SimdPosition>(Position(center[dim]).raw());
    }
    for (ParticleInteger i = particleRange.first; i < maxi; i += SIMD_FLOAT_SIZE) {
        Array<SimdPosition, NDIM> xValue{};
        SimdFloat mask{};
        ParticleInteger const maxj = std::min(i + SIMD_FLOAT_SIZE, particleRange.second);
        for (ParticleInteger j = i; j < maxj; j++) {
            for (int dim = 0; dim < NDIM; dim++) {
                simdSetLane(xValue[dim], j - i, sources.positions[dim][j].raw());
            }
            simdSetLane(mask, j - i, fmmSourceMass(sources, j));
        }
        for (ParticleInteger j = maxj; j < i + SIMD_FLOAT_SIZE; j++) {
            simdSetLane(mask, j - i, 0.f);
            for (int dim = 0; dim < NDIM; dim++) {
                simdSetLane(xValue[dim], j - i, sources.positions[dim][maxj - 1].raw());
            }
        }
        Array<SimdFloat, NDIM> dx{};
        for (int dim = 0; dim < NDIM; dim++) {
            dx[dim] =
                simdCast<SimdFloat>(simdPositionDifference(xValue[dim], y[dim])) * toFloatFactor;
        }
        cosmictiger::gravity::cartesian::Multipole<SimdFloat, order> m{};
        cosmictiger::gravity::cartesian::P2M(
            m, simdBroadcast<SimdFloat>(1), fmmScaleDisplacement(dx));
        for (int j = 0; j < kMultipoleCoefficients; j++) {
            m[j] *= mask;
            mValue[j] += simdSum(m[j]);
        }
    }
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        multi[i] = mValue[i];
    }
    // Every SIMD lane executes P2M, including the zero-mass padding. Each
    // coefficient has a lane multiply and a reduction plus scalar accumulation.
    return 3 +
        FlopCount(maxi - particleRange.first) *
        (cosmictiger::gravity::cartesian::detail::P2M<FmmOrder<order>::kMultipoleDegree_>::kFlops_ +
            6 + 2 * kMultipoleCoefficients);
}

#endif
