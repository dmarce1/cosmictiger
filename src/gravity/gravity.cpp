/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

// cpuGravityCc, cpuGravityCp and cpuGravityPc are order-templated; their
// definitions live in gravity_cpu.hpp and the per-order instance translation
// units instantiate them.

void cpuGravityPp(ForceVectors& f, int doPhi, TreeId self, Vector<TreeId> const& list, float hfloat,
    FmmGeometry const& geometry) {
    FlopCounter<FlopCount> flops = 0;
    FlopCounts geometryFlops{};
    FlopCounter<FlopCount> doubleFlops = 0;
    Timer tm;
    tm.start();
    constexpr int kChunkSize = 32;
    if (list.size()) {
        SimdFloat const sinkHsoft = math::broadcast<SimdFloat>(hfloat);
        SimdFloat const hinvI = math::broadcast<SimdFloat>(1.f) / sinkHsoft;
        SimdFloat const h2invI = sqr(hinvI);
        SimdFloat const h3invI = hinvI * h2invI;
        SimdFloat const h2 = sqr(sinkHsoft);
        flops += 4 * kSimdFloatSize;
        FlopCount softenedPackets = 0;
        FlopCounts const softeningFlops =
            detail::gravitySofteningOperations(selectedGravityKernel(), SimdFloat{}, doPhi);
        SimdFloat const one = math::broadcast<SimdFloat>(1.0);
        SimdFloat const tiny = math::broadcast<SimdFloat>(1.0e-20);
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSinks = selfPtr->numberOfParticles();
        auto const range = selfPtr->particleRange;
        for (int li = 0; li < list.size(); li += kChunkSize) {
            Vector3<SimdPosition> position{};
            Vector3<SimdPosition> yValue{};
            SimdPosition srcType{};
            Array<TreeNode const*, kChunkSize> treePtrs;
            int numberOfSources = 0;
            int const maxi = std::min((int) list.size(), li + kChunkSize) - li;
            for (int i = li; i < li + maxi; i++) {
                treePtrs[i - li] = treeGetNode(list[i]);
                numberOfSources += treePtrs[i - li]->numberOfParticles();
            }
            numberOfSources = roundUp(numberOfSources, kSimdFloatSize);
            Vector<Position> srcx;
            Vector<Position> srcy;
            Vector<Position> srcz;
            Vector<float> sourceWeights;
            Vector<char> types(numberOfSources);
            srcx.resize(numberOfSources);
            srcy.resize(numberOfSources);
            srcz.resize(numberOfSources);
            sourceWeights.resize(numberOfSources);
            int count = 0;
            for (int i = 0; i < maxi; i++) {
                gravityReadSourcePositions(*treePtrs[i], srcx.data(), srcy.data(), srcz.data(),
                    types.data(), sourceWeights.data(), count);
                count += treePtrs[i]->numberOfParticles();
            }
            for (int i = count; i < numberOfSources; i++) {
                srcx[i] = 0.f;
                srcy[i] = 0.f;
                srcz[i] = 0.f;
                sourceWeights[i] = 0.0f;
            }
            //			feenableexcept (FE_DIVBYZERO);
            //			feenableexcept (FE_INVALID);
            //			feenableexcept (FE_OVERFLOW);
            for (ParticleInteger i = range.first; i < range.second; i++) {
                SimdFloat gx = math::broadcast<SimdFloat>(0.0);
                SimdFloat gy = math::broadcast<SimdFloat>(0.0);
                SimdFloat gz = math::broadcast<SimdFloat>(0.0);
                SimdFloat phi = math::broadcast<SimdFloat>(0.0);
                for (int dim = 0; dim < kDimensions; dim++) {
                    position[dim] = math::broadcast<SimdPosition>(particlesPos(dim, i).raw());
                }
                //					simd_float self_flags(0);
                for (int j = 0; j < numberOfSources; j += kSimdFloatSize) {
                    SimdFloat sourceWeight{};
                    for (int l = 0; l < kSimdFloatSize; l++) {
                        simdSetLane(yValue[kXi], l, srcx[j + l].raw());
                        simdSetLane(yValue[kYi], l, srcy[j + l].raw());
                        simdSetLane(yValue[kZi], l, srcz[j + l].raw());
                        simdSetLane(sourceWeight, l, sourceWeights[j + l]);
                    }
                    Vector3<SimdFloat> dx{};
                    for (int dim = 0; dim < kDimensions; dim++) {
                        dx[dim] = fmmPositionDifference(
                            position[dim], yValue[dim], dim, geometry, &geometryFlops);
                    }
                    SimdFloat const r2 = sqr(dx[kXi], dx[kYi], dx[kZi]);
                    SimdFloat const rinv1Far =
                        math::broadcast<SimdFloat>(1) / math::sqrt(math::max(r2, tiny));
                    SimdFloat const rinv3Far = rinv1Far * sqr(rinv1Far);
                    auto const swFar = r2 > h2;
                    SimdFloat rinv3 = rinv3Far;
                    SimdFloat rinv1 = math::broadcast<SimdFloat>(0.0f);
                    if (doPhi) {
                        rinv1 = rinv1Far;
                    }
                    if (!math::allOf(swFar)) {
                        SimdFloat rinv3Near, rinv1Near;
                        gravitySofteningKernel(
                            rinv3Near, rinv1Near, r2, hinvI, h2invI, h3invI, doPhi);
                        ++softenedPackets;
                        rinv3 = math::select(swFar, rinv3Far, rinv3Near);
                        if (doPhi) {
                            rinv1 = math::select(swFar, rinv1Far, rinv1Near);
                        }
                    }
                    rinv3 *= sourceWeight;
                    rinv1 *= sourceWeight;
                    gx = math::fma(rinv3, dx[kXi], gx);
                    gy = math::fma(rinv3, dx[kYi], gy);
                    gz = math::fma(rinv3, dx[kZi], gz);
                    phi -= rinv1;
                }
                int const j = i - range.first;
                f.gx[j] -= simdSum(gx);
                f.gy[j] -= simdSum(gy);
                f.gz[j] -= simdSum(gz);
                f.phi[j] += simdSum(phi);
            }
            // Fixed work is charged once per chunk, including padded SIMD lanes.
            // Per pair: displacement 3, norm 5, sqrt/divide 2, inverse cube 2,
            // weights 2, three FMAs 6, and potential accumulation 1.
            flops +=
                FlopCount(numberOfSinks) * (FlopCount(numberOfSources) * 18 + 4 * kSimdFloatSize);
        }
        flops += softenedPackets * softeningFlops.singlePrecision;
        doubleFlops += softenedPackets * softeningFlops.doublePrecision;
    }
    addCpuFlops(flops, FlopPrecision::Single);
    addCpuFlops(geometryFlops);
    addCpuFlops(doubleFlops, FlopPrecision::Double);
}

#ifdef USE_CUDA
void resetGravityCountersLocal();
void setGravityCounterUseLocal(bool);
Array<double, 3> gravityCounterTotalsLocal();

static Array<double, 3> gravityCounterTotals();
HPX_PLAIN_ACTION(resetGravityCounters, ResetGravityCountersAction);
HPX_PLAIN_ACTION(setGravityCounterUse, SetGravityCounterUseAction);
HPX_PLAIN_ACTION(gravityCounterTotals, GravityCounterTotalsAction);

void resetGravityCounters() {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<ResetGravityCountersAction>(child));
    }
    resetGravityCountersLocal();
    for (auto& future : pending) {
        future.get();
    }
}

void setGravityCounterUse(bool enabled) {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SetGravityCounterUseAction>(child, enabled));
    }
    setGravityCounterUseLocal(enabled);
    for (auto& future : pending) {
        future.get();
    }
}

static Array<double, 3> gravityCounterTotals() {
    Vector<hpx::future<Array<double, 3>>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<GravityCounterTotalsAction>(child));
    }
    auto totals = gravityCounterTotalsLocal();
    for (auto& future : pending) {
        auto const child = future.get();
        for (int i = 0; i < 3; ++i) {
            totals[i] += child[i];
        }
    }
    return totals;
}

void getGravityCounters(double& close, double& direct) {
    auto const totals = gravityCounterTotals();
    close = totals[2] > 0.0 ? totals[0] / totals[2] : 0.0;
    direct = totals[2] > 0.0 ? totals[1] / totals[2] : 0.0;
}
#else
void resetGravityCounters() {}
void setGravityCounterUse(bool) {}
void getGravityCounters(double& close, double& direct) {
    close = direct = 0.0;
}
#endif
