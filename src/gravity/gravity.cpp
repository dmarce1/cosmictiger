/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>

// cpuGravityCc, cpuGravityCp and cpuGravityPc are order-templated; their
// definitions live in gravity_cpu.hpp and the per-order instance translation
// units instantiate them.

void cpuGravityPp(
    ForceVectors& f, int doPhi, TreeId self, Vector<TreeId> const& list, float hfloat) {
    FlopCounter<FlopCount> flops = 0;
    Timer tm;
    tm.start();
    constexpr int kChunkSize = 32;
    if (list.size()) {
        SimdFloat const sinkHsoft = simdBroadcast<SimdFloat>(hfloat);
        SimdFloat const hinvI = simdBroadcast<SimdFloat>(1.f) / sinkHsoft;
        SimdFloat const h2invI = sqr(hinvI);
        SimdFloat const h3invI = hinvI * h2invI;
        SimdFloat const h2 = sqr(sinkHsoft);
        flops += 4 * SIMD_FLOAT_SIZE;
        FlopCount softenedPackets = 0;
        FlopCount const softeningFlops =
            detail::gravitySofteningFlops(selectedGravityKernel(), SimdFloat{}, doPhi);
        static SimdFloat const toFloatFactor = simdBroadcast<SimdFloat>(kFixed2float);
        SimdFloat const one = simdBroadcast<SimdFloat>(1.0);
        SimdFloat const tiny = simdBroadcast<SimdFloat>(1.0e-20);
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSinks = selfPtr->numberOfParticles();
        auto const range = selfPtr->particleRange;
        for (int li = 0; li < list.size(); li += kChunkSize) {
            Array<SimdInt, NDIM> x{};
            Array<SimdInt, NDIM> yValue{};
            SimdInt srcType{};
            Array<TreeNode const*, kChunkSize> treePtrs;
            int numberOfSources = 0;
            int const maxi = std::min((int) list.size(), li + kChunkSize) - li;
            for (int i = li; i < li + maxi; i++) {
                treePtrs[i - li] = treeGetNode(list[i]);
                numberOfSources += treePtrs[i - li]->numberOfParticles();
            }
            numberOfSources = roundUp(numberOfSources, SIMD_FLOAT_SIZE);
            Vector<Fixed32> srcx;
            Vector<Fixed32> srcy;
            Vector<Fixed32> srcz;
            Vector<float> sourceWeights;
            Vector<char> types(numberOfSources);
            srcx.resize(numberOfSources);
            srcy.resize(numberOfSources);
            srcz.resize(numberOfSources);
            sourceWeights.resize(numberOfSources);
            int count = 0;
            for (int i = 0; i < maxi; i++) {
                particlesGlobalReadPos(treePtrs[i]->globalParticleRange(), srcx.data(), srcy.data(),
                    srcz.data(), types.data(), count);
                count += treePtrs[i]->numberOfParticles();
            }
            for (int i = 0; i < count; i++) {
                sourceWeights[i] = getOptions().hydro.enabled ?
                    (types[i] == DARK_MATTER_TYPE ? getOptions().initial.darkMatterMass :
                                                    getOptions().initial.gasMass) :
                    1.0f;
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
                SimdFloat gx = simdBroadcast<SimdFloat>(0.0);
                SimdFloat gy = simdBroadcast<SimdFloat>(0.0);
                SimdFloat gz = simdBroadcast<SimdFloat>(0.0);
                SimdFloat phi = simdBroadcast<SimdFloat>(0.0);
                for (int dim = 0; dim < NDIM; dim++) {
                    x[dim] = simdBroadcast<SimdInt>(particlesPos(dim, i).raw());
                }
                //					simd_float self_flags(0);
                for (int j = 0; j < numberOfSources; j += SIMD_FLOAT_SIZE) {
                    SimdFloat sourceWeight{};
                    for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
                        simdSetLane(yValue[XDIM], l, srcx[j + l].raw());
                        simdSetLane(yValue[YDIM], l, srcy[j + l].raw());
                        simdSetLane(yValue[ZDIM], l, srcz[j + l].raw());
                        simdSetLane(sourceWeight, l, sourceWeights[j + l]);
                    }
                    Array<SimdFloat, NDIM> dx{};
                    for (int dim = 0; dim < NDIM; dim++) {
                        dx[dim] = simdCast<SimdFloat>(simdPositionDifference(x[dim], yValue[dim])) *
                            toFloatFactor;
                    }
                    SimdFloat const r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
                    SimdFloat const rinv1Far =
                        simdBroadcast<SimdFloat>(1) / simdSqrt(simdMax(r2, tiny));
                    SimdFloat const rinv3Far = rinv1Far * sqr(rinv1Far);
                    auto const swFar = r2 > h2;
                    SimdFloat rinv3 = rinv3Far;
                    SimdFloat rinv1 = simdBroadcast<SimdFloat>(0.0f);
                    if (doPhi) {
                        rinv1 = rinv1Far;
                    }
                    if (!simdAllOf(swFar)) {
                        SimdFloat rinv3Near, rinv1Near;
                        gravitySofteningKernel(
                            rinv3Near, rinv1Near, r2, hinvI, h2invI, h3invI, doPhi);
                        ++softenedPackets;
                        rinv3 = simdSelect(swFar, rinv3Far, rinv3Near);
                        if (doPhi) {
                            rinv1 = simdSelect(swFar, rinv1Far, rinv1Near);
                        }
                    }
                    rinv3 *= sourceWeight;
                    rinv1 *= sourceWeight;
                    gx = simdFma(rinv3, dx[XDIM], gx);
                    gy = simdFma(rinv3, dx[YDIM], gy);
                    gz = simdFma(rinv3, dx[ZDIM], gz);
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
                FlopCount(numberOfSinks) * (FlopCount(numberOfSources) * 21 + 4 * SIMD_FLOAT_SIZE);
        }
        flops += softenedPackets * softeningFlops;
    }
    addCpuFlops(flops);
}

#ifndef USE_CUDA
void resetGravityCounters() {}
void setGravityCounterUse(bool) {}
void getGravityCounters(double& close, double& direct) {
    close = direct = 0.0;
}
#endif
