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

#include <boost/align/aligned_allocator.hpp>

#include <cosmictiger/flops.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>

void cpuGravityCc(GravityCcType type, Expansion<float>& lValue, Vector<TreeId> const& list,
    TreeId self, bool doPhi) {
    FlopCounter<int> flops = 0;
    if (list.size()) {
        static SimdFloat const toFloatFactor = simdBroadcast<SimdFloat>(kFixed2float);
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSinks = selfPtr->numberOfParticles();
        int const numberOfSources = roundUp((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local Vector<Multipole<SimdFloat>,
            boost::alignment::aligned_allocator<Multipole<SimdFloat>,
                SIMD_FLOAT_SIZE * sizeof(float)>>
            mValue;
        static thread_local Vector<Array<SimdInt, NDIM>,
            boost::alignment::aligned_allocator<Array<SimdInt, NDIM>,
                SIMD_FLOAT_SIZE * sizeof(float)>>
            yValue;
        mValue.resize(numberOfSources);
        yValue.resize(numberOfSources);
        for (int i = 0; i < treePtrs.size(); i++) {
            int const k = i / SIMD_FLOAT_SIZE;
            int const l = i % SIMD_FLOAT_SIZE;
            auto const& m = treePtrs[i]->mpos->multi;
            auto const& y = treePtrs[i]->mpos->pos;
            for (int j = 0; j < MULTIPOLE_SIZE; j++) {
                simdSetLane(mValue[k][j], l, m[j]);
            }
            for (int j = 0; j < NDIM; j++) {
                simdSetLane(yValue[k][j], l, y[j].raw());
            }
        }
        int const lastk = (treePtrs.size() - 1) / SIMD_FLOAT_SIZE;
        int const lastl = (treePtrs.size() - 1) % SIMD_FLOAT_SIZE;
        for (int i = treePtrs.size(); i < numberOfSources * SIMD_FLOAT_SIZE; i++) {
            int const k = i / SIMD_FLOAT_SIZE;
            int const l = i % SIMD_FLOAT_SIZE;
            for (int j = 0; j < MULTIPOLE_SIZE; j++) {
                simdSetLane(mValue[k][j], l, 0.0);
            }
            for (int j = 0; j < NDIM; j++) {
                simdSetLane(yValue[k][j], l, yValue[lastk][j][lastl]);
            }
        }
        Array<SimdInt, NDIM> x{};
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = simdBroadcast<SimdInt>(selfPtr->mpos->pos[dim].raw());
        }
        Expansion<SimdFloat> l0;
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
                Expansion<SimdFloat> d;
                flops += count * ewaldGreensFunction(d, dx, unwrappedPosition);
                flops += count * m2L(l0, mValue[j], d, doPhi);
            }
        }
        for (int i = 0; i < EXPANSION_SIZE; i++) {
            lValue[i] += simdSum(l0[i]);
        }
        flops += 7 * EXPANSION_SIZE;
    }
    addCpuFlops(flops);
}

void cpuGravityCp(Expansion<float>& lValue, Vector<TreeId> const& list, TreeId self, bool doPhi) {
    constexpr int kChunkSize = 32;
    FlopCounter<int> flops = 0;
    if (list.size()) {
        static SimdFloat const toFloatFactor = simdBroadcast<SimdFloat>(kFixed2float);
        SimdFloat const one = simdBroadcast<SimdFloat>(1.0);
        SimdFloat const tiny = simdBroadcast<SimdFloat>(1.0e-20);
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSinks = selfPtr->numberOfParticles();
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
            auto const range = selfPtr->particleRange;
            Array<SimdInt, NDIM> x{};
            Array<SimdInt, NDIM> yValue{};
            for (int dim = 0; dim < NDIM; dim++) {
                x[dim] = simdBroadcast<SimdInt>(selfPtr->mpos->pos[dim].raw());
            }
            Expansion<SimdFloat> l0;
            l0 = simdBroadcast<SimdFloat>(0.0f);
            for (int j = 0; j < numberOfSources; j += SIMD_FLOAT_SIZE) {
                int const cnt = std::min(count - j, SIMD_FLOAT_SIZE);
                int const k = j / SIMD_FLOAT_SIZE;
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
                Expansion<SimdFloat> d;
                flops += count * greensFunction(d, dx);
                for (int l = 0; l < EXPANSION_SIZE; l++) {
                    l0[l] += mass * d[l];
                }
                flops += cnt * EXPANSION_SIZE;
            }
            for (int i = 0; i < EXPANSION_SIZE; i++) {
                lValue[i] += simdSum(l0[i]);
            }
            flops += 7 * EXPANSION_SIZE;
        }
    }
    addCpuFlops(flops);
}

void cpuGravityPc(ForceVectors& f, int doPhi, TreeId self, Vector<TreeId> const& list) {
    FlopCounter<int> flops = 0;
    if (list.size()) {
        static SimdFloat const toFloatFactor = simdBroadcast<SimdFloat>(kFixed2float);
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSinks = selfPtr->numberOfParticles();
        int const numberOfSources = roundUp((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local Vector<Multipole<SimdFloat>,
            boost::alignment::aligned_allocator<Multipole<SimdFloat>,
                SIMD_FLOAT_SIZE * sizeof(float)>>
            mValue;
        static thread_local Vector<Array<SimdInt, NDIM>,
            boost::alignment::aligned_allocator<Array<SimdInt, NDIM>,
                SIMD_FLOAT_SIZE * sizeof(float)>>
            yValue;
        mValue.resize(numberOfSources);
        yValue.resize(numberOfSources);
        int count = 0;
        for (int i = 0; i < treePtrs.size(); i++) {
            int const k = i / SIMD_FLOAT_SIZE;
            int const l = i % SIMD_FLOAT_SIZE;
            auto const& m = treePtrs[i]->mpos->multi;
            auto const& y = treePtrs[i]->mpos->pos;
            for (int j = 0; j < MULTIPOLE_SIZE; j++) {
                simdSetLane(mValue[k][j], l, m[j]);
            }
            for (int j = 0; j < NDIM; j++) {
                simdSetLane(yValue[k][j], l, y[j].raw());
            }
        }
        int const lastk = (treePtrs.size() - 1) / SIMD_FLOAT_SIZE;
        int const lastl = (treePtrs.size() - 1) % SIMD_FLOAT_SIZE;
        for (int i = treePtrs.size(); i < numberOfSources * SIMD_FLOAT_SIZE; i++) {
            int const k = i / SIMD_FLOAT_SIZE;
            int const l = i % SIMD_FLOAT_SIZE;
            for (int j = 0; j < MULTIPOLE_SIZE; j++) {
                simdSetLane(mValue[k][j], l, 0.0);
            }
            for (int j = 0; j < NDIM; j++) {
                simdSetLane(yValue[k][j], l, yValue[lastk][j][lastl]);
            }
        }
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
