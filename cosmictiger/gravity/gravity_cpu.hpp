/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <algorithm>

#include <boost/align/aligned_allocator.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_kernels.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>

// Definitions of the order-templated CPU gravity operators declared in
// gravity.hpp. Include this only from fmm_cpu_instances.hpp: each per-order
// instance translation unit instantiates the three operators for its order, and
// the walk links against those instantiations.

namespace detail {
template <class T>
using SimdAlignedVector = Vector<T,
    boost::alignment::aligned_allocator<T,
        (alignof(T) > SIMD_FLOAT_SIZE * sizeof(float) ? alignof(T) :
                                                        SIMD_FLOAT_SIZE * sizeof(float))>>;

// Gathers the source multipoles and centers of `treePtrs` into SIMD lanes, padding
// the last vector with zero mass at the last source's center.
template <int order>
void gatherSourceMultipoles(Vector<TreeNode const*> const& treePtrs,
    SimdAlignedVector<FmmMultipole<SimdFloat, order>>& mValue,
    SimdAlignedVector<Array<SimdPosition, NDIM>>& yValue) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    int const numberOfSources = roundUp((int) treePtrs.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
    mValue.resize(numberOfSources);
    yValue.resize(numberOfSources);
    for (int i = 0; i < treePtrs.size(); i++) {
        int const k = i / SIMD_FLOAT_SIZE;
        int const l = i % SIMD_FLOAT_SIZE;
        auto const& source = fmmMultiPos<order>(treePtrs[i]->mpos);
        for (int j = 0; j < kMultipoleCoefficients; j++) {
            simdSetLane(mValue[k][j], l, source.multi[j]);
        }
        for (int j = 0; j < NDIM; j++) {
            simdSetLane(yValue[k][j], l, source.pos[j].raw());
        }
    }
    int const lastk = (treePtrs.size() - 1) / SIMD_FLOAT_SIZE;
    int const lastl = (treePtrs.size() - 1) % SIMD_FLOAT_SIZE;
    for (int i = treePtrs.size(); i < numberOfSources * SIMD_FLOAT_SIZE; i++) {
        int const k = i / SIMD_FLOAT_SIZE;
        int const l = i % SIMD_FLOAT_SIZE;
        for (int j = 0; j < kMultipoleCoefficients; j++) {
            simdSetLane(mValue[k][j], l, 0.0);
        }
        for (int j = 0; j < NDIM; j++) {
            simdSetLane(yValue[k][j], l, yValue[lastk][j][lastl]);
        }
    }
}
}    // namespace detail

template <int order>
void cpuGravityCc(FmmExpansion<float, order>& lValue, Vector<TreeId> const& list, TreeId self,
    bool doPhi, FmmGeometry const& geometry) {
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    FlopCounter<FlopCount> flops = 0;
    if (list.size()) {
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSources = roundUp((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local detail::SimdAlignedVector<FmmMultipole<SimdFloat, order>> mValue;
        static thread_local detail::SimdAlignedVector<Array<SimdPosition, NDIM>> yValue;
        detail::gatherSourceMultipoles<order>(treePtrs, mValue, yValue);
        Array<SimdPosition, NDIM> x{};
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = simdBroadcast<SimdPosition>(selfPtr->mpos->pos[dim].raw());
        }
        FmmExpansion<SimdFloat, order> l0;
        l0 = simdBroadcast<SimdFloat>(0.0f);
        int directFlops = 0;
        for (int j = 0; j < numberOfSources; j++) {
            Array<SimdFloat, NDIM> dx{};
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = fmmPositionDifference(x[dim], yValue[j][dim], dim, geometry);
            }
            directFlops = m2LDirect(l0, mValue[j], dx, doPhi);
        }
        // Fixed displacement and kernel work is batched outside the source loop.
        flops += FlopCount(numberOfSources) * SIMD_FLOAT_SIZE * (3 + directFlops);
        for (int i = 0; i < kLocalCoefficients; i++) {
            lValue[i] += simdSum(l0[i]);
        }
        flops += SIMD_FLOAT_SIZE * kLocalCoefficients;
    }
    addCpuFlops(flops);
}

template <int order>
void cpuGravityCp(FmmExpansion<float, order>& lValue, Vector<TreeId> const& list, TreeId self,
    bool doPhi, FmmGeometry const& geometry) {
    constexpr int kChunkSize = 32;
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    FlopCounter<FlopCount> flops = 0;
    if (list.size()) {
        TreeNode const* selfPtr = treeGetNode(self);
        for (int li = 0; li < list.size(); li += kChunkSize) {
            Array<TreeNode const*, kChunkSize> treePtrs;
            int numberOfSources = 0;
            int const maxi = std::min((int) list.size(), li + kChunkSize) - li;
            for (int i = 0; i < maxi; i++) {
                treePtrs[i] = treeGetNode(list[i + li]);
                numberOfSources += treePtrs[i]->numberOfParticles();
            }
            numberOfSources = roundUp(numberOfSources, SIMD_FLOAT_SIZE);
            Vector<Position> srcx;
            Vector<Position> srcy;
            Vector<Position> srcz;
            Vector<float> masses;
            Vector<char> types;
            srcx.resize(numberOfSources);
            srcy.resize(numberOfSources);
            srcz.resize(numberOfSources);
            masses.resize(numberOfSources);
            types.resize(numberOfSources);
            int count = 0;
            for (int i = 0; i < maxi; i++) {
                gravityReadSourcePositions(
                    *treePtrs[i], srcx.data(), srcy.data(), srcz.data(), types.data(), count);
                count += treePtrs[i]->numberOfParticles();
            }
            for (int i = 0; i < count; i++) {
                masses[i] = getOptions().hydro.enabled ?
                    (types[i] == DARK_MATTER_TYPE ? getOptions().initial.darkMatterMass :
                                                    getOptions().initial.gasMass) :
                    1.0f;
            }
            for (int i = count; i < numberOfSources; i++) {
                // A zero mass does not make an overflowing Green derivative
                // harmless: zero times infinity is NaN. Reuse an accepted
                // source position so every padded lane has finite geometry.
                srcx[i] = srcx[count - 1];
                srcy[i] = srcy[count - 1];
                srcz[i] = srcz[count - 1];
                masses[i] = 0.0;
            }
            Array<SimdPosition, NDIM> x{};
            Array<SimdPosition, NDIM> yValue{};
            for (int dim = 0; dim < NDIM; dim++) {
                x[dim] = simdBroadcast<SimdPosition>(selfPtr->mpos->pos[dim].raw());
            }
            FmmExpansion<SimdFloat, order> l0;
            l0 = simdBroadcast<SimdFloat>(0.0f);
            int greenFlops = 0;
            for (int j = 0; j < numberOfSources; j += SIMD_FLOAT_SIZE) {
                SimdFloat mass{};
                for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
                    simdSetLane(yValue[XDIM], l, srcx[j + l].raw());
                    simdSetLane(yValue[YDIM], l, srcy[j + l].raw());
                    simdSetLane(yValue[ZDIM], l, srcz[j + l].raw());
                    simdSetLane(mass, l, masses[j + l]);
                }
                Array<SimdFloat, NDIM> dx{};
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = fmmPositionDifference(x[dim], yValue[dim], dim, geometry);
                }
                FmmExpansion<SimdFloat, order> d;
                greenFlops = greensFunction(d, dx);
                for (int l = 0; l < kLocalCoefficients; l++) {
                    l0[l] += mass * d[l];
                }
            }
            // Include padded SIMD lanes; each mass-weighted accumulation is two FLOPs.
            flops += FlopCount(numberOfSources) * (greenFlops + 3 + 2 * kLocalCoefficients);
            for (int i = 0; i < kLocalCoefficients; i++) {
                lValue[i] += simdSum(l0[i]);
            }
            flops += SIMD_FLOAT_SIZE * kLocalCoefficients;
        }
    }
    addCpuFlops(flops);
}

template <int order>
void cpuGravityPc(ForceVectors& f, int doPhi, TreeId self, Vector<TreeId> const& list,
    FmmGeometry const& geometry) {
    FlopCounter<FlopCount> flops = 0;
    if (list.size()) {
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSources = roundUp((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local detail::SimdAlignedVector<FmmMultipole<SimdFloat, order>> mValue;
        static thread_local detail::SimdAlignedVector<Array<SimdPosition, NDIM>> yValue;
        detail::gatherSourceMultipoles<order>(treePtrs, mValue, yValue);
        auto const range = selfPtr->particleRange;
        Array<SimdPosition, NDIM> x{};
        int kernelFlops = 0;
        for (ParticleInteger i = range.first; i < range.second; i++) {
            Expansion2<SimdFloat> lValue2;
            lValue2(0, 0, 0) = simdBroadcast<SimdFloat>(0.0f);
            lValue2(1, 0, 0) = simdBroadcast<SimdFloat>(0.0f);
            lValue2(0, 1, 0) = simdBroadcast<SimdFloat>(0.0f);
            lValue2(0, 0, 1) = simdBroadcast<SimdFloat>(0.0f);
            for (int dim = 0; dim < NDIM; dim++) {
                x[dim] = simdBroadcast<SimdPosition>(particlesPos(dim, i).raw());
            }
            for (int j = 0; j < numberOfSources; j++) {
                Array<SimdFloat, NDIM> dx{};
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = fmmPositionDifference(x[dim], yValue[j][dim], dim, geometry);
                }
                kernelFlops = m2P(lValue2, mValue[j], dx, doPhi);
            }
            int const j = i - range.first;
            f.gx[j] -= SCALE_FACTOR2 * simdSum(lValue2(1, 0, 0));
            f.gy[j] -= SCALE_FACTOR2 * simdSum(lValue2(0, 1, 0));
            f.gz[j] -= SCALE_FACTOR2 * simdSum(lValue2(0, 0, 1));
            f.phi[j] += SCALE_FACTOR1 * simdSum(lValue2(0, 0, 0));
        }
        flops += FlopCount(range.second - range.first) *
            (FlopCount(numberOfSources) * SIMD_FLOAT_SIZE * (kernelFlops + 3) +
                4 * (SIMD_FLOAT_SIZE + 1));
    }
    addCpuFlops(flops);
}
