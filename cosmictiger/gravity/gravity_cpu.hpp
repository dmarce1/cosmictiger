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
#include <cosmictiger/fmm/fmm_kernels.hpp>
#include <cosmictiger/fmm/fmm_types.hpp>
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
using SimdAlignedVector =
    Vector<T, boost::alignment::aligned_allocator<T, SIMD_FLOAT_SIZE * sizeof(float)>>;

// Gathers the source multipoles and centers of `treePtrs` into SIMD lanes, padding
// the last vector with zero mass at the last source's center.
template <int order>
void gatherSourceMultipoles(Vector<TreeNode const*> const& treePtrs,
    SimdAlignedVector<FmmMultipole<SimdFloat, order>>& mValue,
    SimdAlignedVector<Array<SimdInt, NDIM>>& yValue) {
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
void cpuGravityCc(GravityCcType type, FmmExpansion<float, order>& lValue,
    Vector<TreeId> const& list, TreeId self, bool doPhi) {
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    FlopCounter<int> flops = 0;
    if (list.size()) {
        static SimdFloat const toFloatFactor = simdBroadcast<SimdFloat>(kFixed2float);
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSources = roundUp((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local detail::SimdAlignedVector<FmmMultipole<SimdFloat, order>> mValue;
        static thread_local detail::SimdAlignedVector<Array<SimdInt, NDIM>> yValue;
        detail::gatherSourceMultipoles<order>(treePtrs, mValue, yValue);
        Array<SimdInt, NDIM> x{};
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = simdBroadcast<SimdInt>(selfPtr->mpos->pos[dim].raw());
        }
        FmmExpansion<SimdFloat, order> l0;
        l0 = simdBroadcast<SimdFloat>(0.0f);
        for (int j = 0; j < numberOfSources; j++) {
            int const count =
                std::min(SIMD_FLOAT_SIZE, (int) (treePtrs.size() - j * SIMD_FLOAT_SIZE));
            Array<SimdFloat, NDIM> dx{};
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = simdCast<SimdFloat>(simdPositionDifference(x[dim], yValue[j][dim])) *
                    toFloatFactor;
            }
            flops += 3 * count;
            if (type == kGravityDirect) {
                flops += count * m2LDirect(l0, mValue[j], dx, doPhi);
            } else {
                Array<SimdFloat, NDIM> unwrappedPosition{};
                for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
                    int const sourceIndex =
                        std::min(j * SIMD_FLOAT_SIZE + lane, int(treePtrs.size()) - 1);
                    for (int dim = 0; dim < NDIM; dim++) {
                        simdSetLane(unwrappedPosition[dim], lane,
                            selfPtr->mpos->pos[dim].toDouble() -
                                treePtrs[sourceIndex]->mpos->pos[dim].toDouble());
                    }
                }
                FmmExpansion<SimdFloat, order> d;
                flops += count * ewaldGreensFunction(d, dx, unwrappedPosition);
                flops += count * m2L(l0, mValue[j], d, doPhi);
            }
        }
        for (int i = 0; i < kLocalCoefficients; i++) {
            lValue[i] += simdSum(l0[i]);
        }
        flops += 7 * kLocalCoefficients;
    }
    addCpuFlops(flops);
}

template <int order>
void cpuGravityCp(
    FmmExpansion<float, order>& lValue, Vector<TreeId> const& list, TreeId self, bool doPhi) {
    constexpr int kChunkSize = 32;
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    FlopCounter<int> flops = 0;
    if (list.size()) {
        static SimdFloat const toFloatFactor = simdBroadcast<SimdFloat>(kFixed2float);
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
            Vector<Fixed32> srcx;
            Vector<Fixed32> srcy;
            Vector<Fixed32> srcz;
            Vector<float> masses;
            Vector<char> types;
            srcx.resize(numberOfSources);
            srcy.resize(numberOfSources);
            srcz.resize(numberOfSources);
            masses.resize(numberOfSources);
            types.resize(numberOfSources);
            int count = 0;
            for (int i = 0; i < maxi; i++) {
                particlesGlobalReadPos(treePtrs[i]->globalParticleRange(), srcx.data(), srcy.data(),
                    srcz.data(), types.data(), count);
                count += treePtrs[i]->numberOfParticles();
            }
            for (int i = 0; i < count; i++) {
                masses[i] = getOptions().hydro.enabled ?
                    (types[i] == DARK_MATTER_TYPE ? getOptions().initial.darkMatterMass :
                                                    getOptions().initial.gasMass) :
                    1.0f;
            }
            for (int i = count; i < numberOfSources; i++) {
                srcx[i] = 0.0;
                srcy[i] = 0.0;
                srcz[i] = 0.0;
                masses[i] = 0.0;
            }
            Array<SimdInt, NDIM> x{};
            Array<SimdInt, NDIM> yValue{};
            for (int dim = 0; dim < NDIM; dim++) {
                x[dim] = simdBroadcast<SimdInt>(selfPtr->mpos->pos[dim].raw());
            }
            FmmExpansion<SimdFloat, order> l0;
            l0 = simdBroadcast<SimdFloat>(0.0f);
            for (int j = 0; j < numberOfSources; j += SIMD_FLOAT_SIZE) {
                int const cnt = std::min(count - j, SIMD_FLOAT_SIZE);
                SimdFloat mass{};
                for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
                    simdSetLane(yValue[XDIM], l, srcx[j + l].raw());
                    simdSetLane(yValue[YDIM], l, srcy[j + l].raw());
                    simdSetLane(yValue[ZDIM], l, srcz[j + l].raw());
                    simdSetLane(mass, l, masses[j + l]);
                }
                Array<SimdFloat, NDIM> dx{};
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = simdCast<SimdFloat>(simdPositionDifference(x[dim], yValue[dim])) *
                        toFloatFactor;
                }
                flops += cnt * 3;
                FmmExpansion<SimdFloat, order> d;
                flops += count * greensFunction(d, dx);
                for (int l = 0; l < kLocalCoefficients; l++) {
                    l0[l] += mass * d[l];
                }
                flops += cnt * kLocalCoefficients;
            }
            for (int i = 0; i < kLocalCoefficients; i++) {
                lValue[i] += simdSum(l0[i]);
            }
            flops += 7 * kLocalCoefficients;
        }
    }
    addCpuFlops(flops);
}

template <int order>
void cpuGravityPc(ForceVectors& f, int doPhi, TreeId self, Vector<TreeId> const& list) {
    FlopCounter<int> flops = 0;
    if (list.size()) {
        static SimdFloat const toFloatFactor = simdBroadcast<SimdFloat>(kFixed2float);
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSources = roundUp((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local detail::SimdAlignedVector<FmmMultipole<SimdFloat, order>> mValue;
        static thread_local detail::SimdAlignedVector<Array<SimdInt, NDIM>> yValue;
        detail::gatherSourceMultipoles<order>(treePtrs, mValue, yValue);
        auto const range = selfPtr->particleRange;
        Array<SimdInt, NDIM> x{};
        for (ParticleInteger i = range.first; i < range.second; i++) {
            Expansion2<SimdFloat> lValue2;
            lValue2(0, 0, 0) = simdBroadcast<SimdFloat>(0.0f);
            lValue2(1, 0, 0) = simdBroadcast<SimdFloat>(0.0f);
            lValue2(0, 1, 0) = simdBroadcast<SimdFloat>(0.0f);
            lValue2(0, 0, 1) = simdBroadcast<SimdFloat>(0.0f);
            for (int dim = 0; dim < NDIM; dim++) {
                x[dim] = simdBroadcast<SimdInt>(particlesPos(dim, i).raw());
            }
            for (int j = 0; j < numberOfSources; j++) {
                int const count =
                    std::min(SIMD_FLOAT_SIZE, (int) (treePtrs.size() - j * SIMD_FLOAT_SIZE));
                Array<SimdFloat, NDIM> dx{};
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = simdCast<SimdFloat>(simdPositionDifference(x[dim], yValue[j][dim])) *
                        toFloatFactor;
                }
                flops += 3 * count;
                flops += count * m2P(lValue2, mValue[j], dx, doPhi);
            }
            int const j = i - range.first;
            f.gx[j] -= SCALE_FACTOR2 * simdSum(lValue2(1, 0, 0));
            f.gy[j] -= SCALE_FACTOR2 * simdSum(lValue2(0, 1, 0));
            f.gz[j] -= SCALE_FACTOR2 * simdSum(lValue2(0, 0, 1));
            f.phi[j] += SCALE_FACTOR1 * simdSum(lValue2(0, 0, 0));
            flops += 28;
        }
    }
    addCpuFlops(flops);
}
