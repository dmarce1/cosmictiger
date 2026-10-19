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
    FlopCounter<int> flops = 0;
    Timer tm;
    tm.start();
    constexpr int kChunkSize = 32;
    if (list.size()) {
        SimdFloat const sinkHsoft = simdBroadcast<SimdFloat>(hfloat);
        SimdFloat const hinvI = simdBroadcast<SimdFloat>(1.f) / sinkHsoft;
        SimdFloat const h2invI = sqr(hinvI);
        SimdFloat const h3invI = hinvI * h2invI;
        SimdFloat const h2 = sqr(sinkHsoft);
        flops += 7;
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
                    int const& count = maxi;
                    int const k = j / SIMD_FLOAT_SIZE;
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
                            toFloatFactor;    // 3
                    }
                    SimdFloat const r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
                    SimdFloat const rinv1Far =
                        simdBroadcast<SimdFloat>(1) / simdSqrt(simdMax(r2, tiny));    // 8
                    SimdFloat const rinv3Far = rinv1Far * sqr(rinv1Far);              // 3
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
                        rinv3 = simdSelect(swFar, rinv3Far, rinv3Near);
                        if (doPhi) {
                            rinv1 = simdSelect(swFar, rinv1Far, rinv1Near);
                            flops += 7;
                        }
                    }
                    rinv3 *= sourceWeight;                // 1
                    rinv1 *= sourceWeight;                // 1
                    gx = simdFma(rinv3, dx[XDIM], gx);    // 2
                    gy = simdFma(rinv3, dx[YDIM], gy);    // 2
                    gz = simdFma(rinv3, dx[ZDIM], gz);    // 2
                    phi -= rinv1;                         // 1
                    flops += 38;
                }
                int const j = i - range.first;
                f.gx[j] -= simdSum(gx);
                f.gy[j] -= simdSum(gy);
                f.gz[j] -= simdSum(gz);
                f.phi[j] += simdSum(phi);
                flops += 28;
            }
        }
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
