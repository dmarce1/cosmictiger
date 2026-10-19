/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#pragma once

#include <algorithm>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/fmm/fmm_types.hpp>
#include <cosmictiger/particles/particles.hpp>

// Order-templated multipole construction for treeCreate. treeCreate stays
// order-agnostic and dispatches to exactly one of these per node through
// fmmDispatch(). Both write the leading FmmOrder<order>::kMultipoleCoefficients_
// entries of `multi` and return the flop count.

// Internal node: translate the children's multipoles to `center` and sum them.
template <int order>
int treeComputeMultipole(MultipoleBuffer& multi, MultipoleBuffer const& left,
    MultipoleBuffer const& right, Array<double, NDIM> const& leftCenter,
    Array<double, NDIM> const& rightCenter, Array<double, NDIM> const& center);

// Leaf: accumulate P2M over the local particles of `particleRange` about `center`.
template <int order>
int treeComputeMultipole(
    MultipoleBuffer& multi, Pair<ParticleInteger> particleRange, Array<double, NDIM> const& center);

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
// The definitions pull in the generated kernels, so only the per-order instance
// translation units see them; tree.cpp compiles against the declarations above.

#include <cosmictiger/fmm/fmm_kernels.hpp>
#include <cosmictiger/math/simd.hpp>

template <int order>
int treeComputeMultipole(MultipoleBuffer& multi, MultipoleBuffer const& left,
    MultipoleBuffer const& right, Array<double, NDIM> const& leftCenter,
    Array<double, NDIM> const& rightCenter, Array<double, NDIM> const& center) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    int flops = 0;
    Array<SimdDouble, NDIM> mdx{};
    FmmMultipole<SimdDouble, order> simdM{};
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        simdSetLane(simdM[i], LEFT, left[i]);
        simdSetLane(simdM[i], RIGHT, right[i]);
    }
    for (int dim = 0; dim < NDIM; dim++) {
        simdSetLane(mdx[dim], LEFT, leftCenter[dim] - center[dim]);      // 3
        simdSetLane(mdx[dim], RIGHT, rightCenter[dim] - center[dim]);    // 3
    }
    simdM = m2M(simdM, mdx);
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        multi[i] = simdM[i][LEFT] + simdM[i][RIGHT];
    }
    flops += 2 * detail::stf::M2mKernel<FmmOrder<order>::kMultipoleDegree_>::kFlops_ + 6 +
        kMultipoleCoefficients * 2;
    return flops;
}

template <int order>
int treeComputeMultipole(MultipoleBuffer& multi, Pair<ParticleInteger> particleRange,
    Array<double, NDIM> const& center) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    static SimdFloat const toFloatFactor = kFixed2float;
    int flops = 0;
    FmmMultipole<double, order> mValue;
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        mValue[i] = 0.0;
    }
    ParticleInteger const maxi =
        roundUp(particleRange.second - particleRange.first, (ParticleInteger) SIMD_FLOAT_SIZE) +
        particleRange.first;
    Array<SimdInt, NDIM> y{};
    for (int dim = 0; dim < NDIM; dim++) {
        y[dim] = simdBroadcast<SimdInt>(Fixed32(center[dim]).raw());
    }
    for (ParticleInteger i = particleRange.first; i < maxi; i += SIMD_FLOAT_SIZE) {
        Array<SimdInt, NDIM> xValue{};
        SimdFloat mask{};
        ParticleInteger const maxj = std::min(i + SIMD_FLOAT_SIZE, particleRange.second);
        for (ParticleInteger j = i; j < maxj; j++) {
            for (int dim = 0; dim < NDIM; dim++) {
                simdSetLane(xValue[dim], j - i, particlesPos(dim, j).raw());
            }
            simdSetLane(mask, j - i, particlesMass(j));
        }
        for (ParticleInteger j = maxj; j < i + SIMD_FLOAT_SIZE; j++) {
            simdSetLane(mask, j - i, 0.f);
            for (int dim = 0; dim < NDIM; dim++) {
                simdSetLane(xValue[dim], j - i, particlesPos(dim, maxj - 1).raw());
            }
        }
        Array<SimdFloat, NDIM> dx{};
        for (int dim = 0; dim < NDIM; dim++) {
            dx[dim] = simdCast<SimdFloat>(simdPositionDifference(xValue[dim], y[dim])) *
                toFloatFactor;    // 3
        }
        auto m = p2M<order>(dx);
        for (int j = 0; j < kMultipoleCoefficients; j++) {
            m[j] *= mask;                  // kMultipoleCoefficients
            mValue[j] += simdSum(m[j]);    // kMultipoleCoefficients * 6
        }
        flops += detail::stf::P2mKernel<FmmOrder<order>::kMultipoleDegree_>::kFlops_ + 6 +
            kMultipoleCoefficients * 7;
    }
    for (int i = 0; i < kMultipoleCoefficients; i++) {
        multi[i] = mValue[i];
    }
    return flops;
}

#endif
