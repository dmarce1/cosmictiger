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
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

// Definitions of the order-templated CPU gravity operators declared in
// gravity.hpp. Include this only from fmm_cpu_instances.hpp: each per-order
// instance translation unit instantiates the three operators for its order, and
// the walk links against those instantiations.

namespace detail {
template <class T>
using SimdAlignedVector = Vector<T,
    boost::alignment::aligned_allocator<T,
        (alignof(T) > kSimdFloatSize * sizeof(float) ? alignof(T) :
                                                       kSimdFloatSize * sizeof(float))>>;

// Gathers the source multipoles and centers of `treePtrs` into SIMD lanes, padding
// the last vector with zero mass at the last source's center.
template <int order>
void gatherSourceMultipoles(Vector<TreeNode const*> const& treePtrs,
    SimdAlignedVector<cosmictiger::gravity::cartesian::Multipole<SimdFloat>>& mValue,
    SimdAlignedVector<Vector3<SimdPosition>>& yValue) {
    constexpr int kMultipoleCoefficients = FmmOrder<order>::kMultipoleCoefficients_;
    int const numberOfSources = roundUp((int) treePtrs.size(), kSimdFloatSize) / kSimdFloatSize;
    mValue.resize(numberOfSources);
    yValue.resize(numberOfSources);
    for (int i = 0; i < treePtrs.size(); i++) {
        int const k = i / kSimdFloatSize;
        int const l = i % kSimdFloatSize;
        auto const& source = fmmMultiPos(treePtrs[i]->mpos);
        for (int j = 0; j < kMultipoleCoefficients; j++) {
            simdSetLane(mValue[k][j], l, source.multi[j]);
        }
        for (int j = 0; j < kDimensions; j++) {
            simdSetLane(yValue[k][j], l, source.pos[j].raw());
        }
    }
    int const lastk = (treePtrs.size() - 1) / kSimdFloatSize;
    int const lastl = (treePtrs.size() - 1) % kSimdFloatSize;
    for (int i = treePtrs.size(); i < numberOfSources * kSimdFloatSize; i++) {
        int const k = i / kSimdFloatSize;
        int const l = i % kSimdFloatSize;
        for (int j = 0; j < kMultipoleCoefficients; j++) {
            simdSetLane(mValue[k][j], l, 0.0);
        }
        for (int j = 0; j < kDimensions; j++) {
            simdSetLane(yValue[k][j], l, yValue[lastk][j][lastl]);
        }
    }
}
}    // namespace detail

/// Accumulate accepted source-cell multipoles into the sink local expansion using SIMD M2L
/// and record work counts.
template <int order>
void cpuGravityCc(cosmictiger::gravity::cartesian::Expansion<float>& lValue,
    Vector<TreeId> const& list, TreeId self, bool doPhi, FmmGeometry const& geometry) {
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    FlopCounter<FlopCount> flops = 0;
    FlopCounts geometryFlops{};
    if (list.size()) {
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSources = roundUp((int) list.size(), kSimdFloatSize) / kSimdFloatSize;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local detail::SimdAlignedVector<
            cosmictiger::gravity::cartesian::Multipole<SimdFloat>>
            mValue;
        static thread_local detail::SimdAlignedVector<Vector3<SimdPosition>> yValue;
        detail::gatherSourceMultipoles<order>(treePtrs, mValue, yValue);
        Vector3<SimdPosition> position{};
        for (int dim = 0; dim < kDimensions; dim++) {
            position[dim] = math::broadcast<SimdPosition>(selfPtr->mpos->pos[dim].raw());
        }
        cosmictiger::gravity::cartesian::Expansion<SimdFloat> l0;
        l0 = math::broadcast<SimdFloat>(0.0f);
        int directFlops = 0;
        for (int j = 0; j < numberOfSources; j++) {
            Vector3<SimdFloat> dx{};
            for (int dim = 0; dim < kDimensions; dim++) {
                dx[dim] = fmmPositionDifference(
                    position[dim], yValue[j][dim], dim, geometry, &geometryFlops);
            }
            directFlops = cosmictiger::gravity::cartesian::M2L(
                mValue[j], fmmScaleDisplacement(dx), doPhi, l0);
        }
        // Coordinate scaling and kernel work is batched outside the source loop.
        flops += FlopCount(numberOfSources) * kSimdFloatSize * (3 + directFlops);
        for (int i = 0; i < kLocalCoefficients; i++) {
            lValue[i] += simdSum(l0[i]);
        }
        flops += kSimdFloatSize * kLocalCoefficients;
    }
    addCpuFlops(flops, FlopPrecision::Single);
    addCpuFlops(geometryFlops);
}

/// Gather source-leaf particles and accumulate their contributions into the sink cell's
/// local expansion.
template <int order>
void cpuGravityCp(cosmictiger::gravity::cartesian::Expansion<float>& lValue,
    Vector<TreeId> const& list, TreeId self, bool doPhi, FmmGeometry const& geometry) {
    constexpr int kChunkSize = 32;
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    FlopCounter<FlopCount> flops = 0;
    FlopCounts geometryFlops{};
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
            numberOfSources = roundUp(numberOfSources, kSimdFloatSize);
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
                gravityReadSourcePositions(*treePtrs[i], srcx.data(), srcy.data(), srcz.data(),
                    types.data(), masses.data(), count);
                count += treePtrs[i]->numberOfParticles();
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
            Vector3<SimdPosition> position{};
            Vector3<SimdPosition> yValue{};
            for (int dim = 0; dim < kDimensions; dim++) {
                position[dim] = math::broadcast<SimdPosition>(selfPtr->mpos->pos[dim].raw());
            }
            cosmictiger::gravity::cartesian::Expansion<SimdFloat> l0;
            l0 = math::broadcast<SimdFloat>(0.0f);
            int greenFlops = 0;
            for (int j = 0; j < numberOfSources; j += kSimdFloatSize) {
                SimdFloat mass{};
                for (int l = 0; l < kSimdFloatSize; l++) {
                    simdSetLane(yValue[kXi], l, srcx[j + l].raw());
                    simdSetLane(yValue[kYi], l, srcy[j + l].raw());
                    simdSetLane(yValue[kZi], l, srcz[j + l].raw());
                    simdSetLane(mass, l, masses[j + l]);
                }
                Vector3<SimdFloat> dx{};
                for (int dim = 0; dim < kDimensions; dim++) {
                    dx[dim] = fmmPositionDifference(
                        position[dim], yValue[dim], dim, geometry, &geometryFlops);
                }
                cosmictiger::gravity::cartesian::Expansion<SimdFloat> d;
                d = math::broadcast<SimdFloat>(0);
                greenFlops = cosmictiger::gravity::cartesian::P2L(
                    math::broadcast<SimdFloat>(1), fmmScaleDisplacement(dx), true, d);
                for (int l = 0; l < kLocalCoefficients; l++) {
                    l0[l] += mass * d[l];
                }
            }
            // Include padded SIMD lanes; each mass-weighted accumulation is two FLOPs.
            flops += FlopCount(numberOfSources) * (greenFlops + 3 + 2 * kLocalCoefficients);
            for (int i = 0; i < kLocalCoefficients; i++) {
                lValue[i] += simdSum(l0[i]);
            }
            flops += kSimdFloatSize * kLocalCoefficients;
        }
    }
    addCpuFlops(flops, FlopPrecision::Single);
    addCpuFlops(geometryFlops);
}

/// Evaluate accepted source multipoles at sink particles, adding potential/acceleration and
/// recording geometry/kernel work.
template <int order>
void cpuGravityPc(ForceVectors& f, int doPhi, TreeId self, Vector<TreeId> const& list,
    FmmGeometry const& geometry) {
    FlopCounter<FlopCount> flops = 0;
    FlopCounts geometryFlops{};
    if (list.size()) {
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSources = roundUp((int) list.size(), kSimdFloatSize) / kSimdFloatSize;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local detail::SimdAlignedVector<
            cosmictiger::gravity::cartesian::Multipole<SimdFloat>>
            mValue;
        static thread_local detail::SimdAlignedVector<Vector3<SimdPosition>> yValue;
        detail::gatherSourceMultipoles<order>(treePtrs, mValue, yValue);
        auto const range = selfPtr->particleRange;
        Vector3<SimdPosition> position{};
        int kernelFlops = 0;
        for (ParticleInteger i = range.first; i < range.second; i++) {
            cosmictiger::gravity::Field<SimdFloat> lValue2{};
            for (int component = 0; component < kDimensions + 1; component++) {
                lValue2[component] = math::broadcast<SimdFloat>(0.0f);
            }
            for (int dim = 0; dim < kDimensions; dim++) {
                position[dim] = math::broadcast<SimdPosition>(particlesPos(dim, i).raw());
            }
            for (int j = 0; j < numberOfSources; j++) {
                Vector3<SimdFloat> dx{};
                for (int dim = 0; dim < kDimensions; dim++) {
                    dx[dim] = fmmPositionDifference(
                        position[dim], yValue[j][dim], dim, geometry, &geometryFlops);
                }
                kernelFlops = cosmictiger::gravity::cartesian::M2P(
                    mValue[j], fmmScaleDisplacement(dx), doPhi, lValue2);
            }
            int const j = i - range.first;
            f.gx[j] += kScaleFactor2 * simdSum(lValue2.accelerationX());
            f.gy[j] += kScaleFactor2 * simdSum(lValue2.accelerationY());
            f.gz[j] += kScaleFactor2 * simdSum(lValue2.accelerationZ());
            f.phi[j] += kScaleFactor1 * simdSum(lValue2.potential());
        }
        flops += FlopCount(range.second - range.first) *
            (FlopCount(numberOfSources) * kSimdFloatSize * (kernelFlops + 3) +
                4 * (kSimdFloatSize + 1));
    }
    addCpuFlops(flops, FlopPrecision::Single);
    addCpuFlops(geometryFlops);
}
