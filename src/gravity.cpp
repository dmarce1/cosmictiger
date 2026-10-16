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
        static SimdFloat const toFloatFactor(kFixed2float);
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
                mValue[k][j][l] = m[j];
            }
            for (int j = 0; j < NDIM; j++) {
                yValue[k][j][l] = y[j].raw();
            }
        }
        int const lastk = (treePtrs.size() - 1) / SIMD_FLOAT_SIZE;
        int const lastl = (treePtrs.size() - 1) % SIMD_FLOAT_SIZE;
        for (int i = treePtrs.size(); i < numberOfSources * SIMD_FLOAT_SIZE; i++) {
            int const k = i / SIMD_FLOAT_SIZE;
            int const l = i % SIMD_FLOAT_SIZE;
            for (int j = 0; j < MULTIPOLE_SIZE; j++) {
                mValue[k][j][l] = 0.0;
            }
            for (int j = 0; j < NDIM; j++) {
                yValue[k][j][l] = yValue[lastk][j][lastl];
            }
        }
        Array<SimdInt, NDIM> x;
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = selfPtr->mpos->pos[dim].raw();
        }
        Expansion<SimdFloat> l0;
        l0 = SimdFloat(0.0f);
        for (int j = 0; j < numberOfSources; j++) {
            int const count =
                std::min(SIMD_FLOAT_SIZE, (int) (treePtrs.size() - j * SIMD_FLOAT_SIZE));
            Array<SimdFloat, NDIM> dx;
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = SimdFloat(x[dim] - yValue[j][dim]) * toFloatFactor;
            }
            flops += 3 * count;
            Expansion<SimdFloat> d;
            if (type == kGravityDirect) {
                flops += count * greensFunction(d, dx);
            } else {
                flops += count * ewaldGreensFunction(d, dx);
                //				flops += count * apply_scale_factor(M[j]);
            }
            m2L(l0, mValue[j], d, doPhi);
        }
        if (type == kGravityEwald) {
            //			flops += SIMD_FLOAT_SIZE * apply_scale_factor_inv(L0);
        }
        for (int i = 0; i < EXPANSION_SIZE; i++) {
            lValue[i] += l0[i].sum();
        }
        flops += 7 * EXPANSION_SIZE;
    }
    addCpuFlops(flops);
}

void cpuGravityCp(Expansion<float>& lValue, Vector<TreeId> const& list, TreeId self, bool doPhi) {
    constexpr int kChunkSize = 32;
    FlopCounter<int> flops = 0;
    if (list.size()) {
        static SimdFloat const toFloatFactor(kFixed2float);
        SimdFloat const one(1.0);
        SimdFloat const tiny(1.0e-20);
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
                    srcz.data(), types.data(), nullptr, count);
                count += treePtrs[i]->numberOfParticles();
            }
            for (int i = 0; i < count; i++) {
                masses[i] = getOptions().sph ?
                    (types[i] == DARK_MATTER_TYPE ? getOptions().darkMatterMass :
                                                    getOptions().sphMass) :
                    1.0f;
            }
            for (int i = count; i < numberOfSources; i++) {
                srcx[i] = 0.0;
                srcy[i] = 0.0;
                srcz[i] = 0.0;
                masses[i] = 0.0;
            }
            auto const range = selfPtr->particleRange;
            Array<SimdInt, NDIM> x;
            Array<SimdInt, NDIM> yValue;
            for (int dim = 0; dim < NDIM; dim++) {
                x[dim] = selfPtr->mpos->pos[dim].raw();
            }
            Expansion<SimdFloat> l0;
            l0 = SimdFloat(0.0f);
            for (int j = 0; j < numberOfSources; j += SIMD_FLOAT_SIZE) {
                int const cnt = std::min(count - j, SIMD_FLOAT_SIZE);
                int const k = j / SIMD_FLOAT_SIZE;
                SimdFloat mass;
                for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
                    yValue[XDIM][l] = srcx[j + l].raw();
                    yValue[YDIM][l] = srcy[j + l].raw();
                    yValue[ZDIM][l] = srcz[j + l].raw();
                    mass[l] = masses[j + l];
                }
                Array<SimdFloat, NDIM> dx;
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = SimdFloat(x[dim] - yValue[dim]) * toFloatFactor;
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
                lValue[i] += l0[i].sum();
            }
            flops += 7 * EXPANSION_SIZE;
        }
    }
    addCpuFlops(flops);
}

void cpuGravityPc(ForceVectors& f, int doPhi, TreeId self, Vector<TreeId> const& list) {
    FlopCounter<int> flops = 0;
    if (list.size()) {
        static SimdFloat const toFloatFactor(kFixed2float);
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
                mValue[k][j][l] = m[j];
            }
            for (int j = 0; j < NDIM; j++) {
                yValue[k][j][l] = y[j].raw();
            }
        }
        int const lastk = (treePtrs.size() - 1) / SIMD_FLOAT_SIZE;
        int const lastl = (treePtrs.size() - 1) % SIMD_FLOAT_SIZE;
        for (int i = treePtrs.size(); i < numberOfSources * SIMD_FLOAT_SIZE; i++) {
            int const k = i / SIMD_FLOAT_SIZE;
            int const l = i % SIMD_FLOAT_SIZE;
            for (int j = 0; j < MULTIPOLE_SIZE; j++) {
                mValue[k][j][l] = 0.0;
            }
            for (int j = 0; j < NDIM; j++) {
                yValue[k][j][l] = yValue[lastk][j][lastl];
            }
        }
        auto const range = selfPtr->particleRange;
        Array<SimdInt, NDIM> x;
        for (ParticleInteger i = range.first; i < range.second; i++) {
            Expansion2<SimdFloat> lValue2;
            lValue2(0, 0, 0) = SimdFloat(0.0f);
            lValue2(1, 0, 0) = SimdFloat(0.0f);
            lValue2(0, 1, 0) = SimdFloat(0.0f);
            lValue2(0, 0, 1) = SimdFloat(0.0f);
            for (int dim = 0; dim < NDIM; dim++) {
                x[dim] = particlesPos(dim, i).raw();
            }
            for (int j = 0; j < numberOfSources; j++) {
                int const count =
                    std::min(SIMD_FLOAT_SIZE, (int) (treePtrs.size() - j * SIMD_FLOAT_SIZE));
                Array<SimdFloat, NDIM> dx;
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] = SimdFloat(x[dim] - yValue[j][dim]) * toFloatFactor;
                }
                flops += 3 * count;
                Expansion<SimdFloat> d;
                flops += count * greensFunction(d, dx);
                flops += count * m2L(lValue2, mValue[j], d, doPhi);
            }
            int const j = i - range.first;
            f.gx[j] -= SCALE_FACTOR2 * lValue2(1, 0, 0).sum();
            f.gy[j] -= SCALE_FACTOR2 * lValue2(0, 1, 0).sum();
            f.gz[j] -= SCALE_FACTOR2 * lValue2(0, 0, 1).sum();
            f.phi[j] += SCALE_FACTOR1 * lValue2(0, 0, 0).sum();
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
    size_t nearCount = 0;
    size_t farCount = 0;
    if (list.size()) {
        SimdFloat sinkHsoft;
        sinkHsoft = getOptions().hsoft;
        SimdFloat const hinvI = SimdFloat(1.f) / sinkHsoft;    // 4
        SimdFloat const h2invI = sqr(hinvI);                   // 1
        SimdFloat const h3invI = (hinvI) *h2invI;              // 1
        SimdFloat const h2 = sqr(sinkHsoft);                   // 1;
        flops += 7;
        static SimdFloat const toFloatFactor(kFixed2float);
        SimdFloat const one(1.0);
        SimdFloat const tiny(1.0e-20);
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSinks = selfPtr->numberOfParticles();
        auto const range = selfPtr->particleRange;
        for (int li = 0; li < list.size(); li += kChunkSize) {
            Array<SimdInt, NDIM> x;
            Array<SimdInt, NDIM> yValue;
            SimdInt srcType;
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
            Vector<float> zetas;
            Vector<float> hs;
            Vector<char> type;
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
                    srcz.data(), types.data(), nullptr, count);
                count += treePtrs[i]->numberOfParticles();
            }
            for (int i = 0; i < count; i++) {
                masses[i] = getOptions().sph ?
                    (types[i] == DARK_MATTER_TYPE ? getOptions().darkMatterMass :
                                                    getOptions().sphMass) :
                    1.0f;
            }
            for (int i = count; i < numberOfSources; i++) {
                srcx[i] = 0.f;
                srcy[i] = 0.f;
                srcz[i] = 0.f;
                masses[i] = 0.0f;
            }
            SimdFloat const tiny = 1.0e-15;
            //			feenableexcept (FE_DIVBYZERO);
            //			feenableexcept (FE_INVALID);
            //			feenableexcept (FE_OVERFLOW);
            for (ParticleInteger i = range.first; i < range.second; i++) {
                SimdFloat gx(0.0);
                SimdFloat gy(0.0);
                SimdFloat gz(0.0);
                SimdFloat phi(0.0);
                for (int dim = 0; dim < NDIM; dim++) {
                    x[dim] = particlesPos(dim, i).raw();
                }
                //					simd_float self_flags(0);
                for (int j = 0; j < numberOfSources; j += SIMD_FLOAT_SIZE) {
                    int const& count = maxi;
                    int const k = j / SIMD_FLOAT_SIZE;
                    SimdFloat mass;
                    for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
                        yValue[XDIM][l] = srcx[j + l].raw();
                        yValue[YDIM][l] = srcy[j + l].raw();
                        yValue[ZDIM][l] = srcz[j + l].raw();
                        mass[l] = masses[j + l];
                    }
                    Array<SimdFloat, NDIM> dx;
                    for (int dim = 0; dim < NDIM; dim++) {
                        dx[dim] = SimdFloat(x[dim] - yValue[dim]) * toFloatFactor;    // 3
                    }
                    SimdFloat const r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny);    // 6
                    SimdFloat const rinv1Far = SimdFloat(1) / sqrt(r2);                   // 8
                    SimdFloat const rinv3Far = rinv1Far * sqr(rinv1Far);                  // 3
                    SimdFloat rinv3Near;                                                  // 3
                    SimdFloat rinv1Near, rinv1(0.0f);
                    gravitySofteningKernel(rinv3Near, rinv1Near, r2, hinvI, h2invI, h3invI, doPhi);
                    SimdFloat const swFar = r2 > h2;                            // 1
                    SimdFloat const swNear = SimdFloat(1) - swFar;              // 1
                    SimdFloat rinv3 = rinv3Near * swNear + rinv3Far * swFar;    // 3
                    if (doPhi) {
                        rinv1 = rinv1Near * swNear + rinv1Far * swFar;    // 3
                        flops += 7;
                    }
                    rinv3 *= mass;                     // 1
                    rinv1 *= mass;                     // 1
                    gx = fmaf(rinv3, dx[XDIM], gx);    // 2
                    gy = fmaf(rinv3, dx[YDIM], gy);    // 2
                    gz = fmaf(rinv3, dx[ZDIM], gz);    // 2
                    phi -= rinv1;                      // 1
                    flops += 38;
                }
                int const j = i - range.first;
                f.gx[j] -= gx.sum();
                f.gy[j] -= gy.sum();
                f.gz[j] -= gz.sum();
                f.phi[j] += phi.sum();
                flops += 28;
            }
        }
    }
    addCpuFlops(flops);
}
