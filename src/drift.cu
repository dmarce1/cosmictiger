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

#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/healpix.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/timer.hpp>

#define BLOCK_SIZE 512

using ListType = DeviceVector<DeviceVector<LightconeEntry>>;

__managed__ ListType* listPtr;

__global__ void cudaDriftKernel(Fixed32* __restrict__ const x, Fixed32* __restrict__ const y,
    Fixed32* __restrict__ const z, Array<float, NDIM>* const vels, char const* const rungs,
    ParticleInteger count, int rung, double a, double dt, double tau0, double tau1, double tauMax,
    bool doLc, int nside, double drag) {
    doLc = doLc && (tauMax - tau1 < 1.0f);
    auto const& tid = threadIdx.x;
    auto const& bid = blockIdx.x;
    auto* list = doLc ? &(*listPtr)[bid] : nullptr;
    double const c0 = doLc ? 1.0 / tauMax : 0.0;
    double const t0(tau0);
    double const ainv = 1.0 / a;
    auto const& numberOfBlocks = gridDim.x;
    ParticleInteger const begin = (size_t) bid * count / numberOfBlocks;
    ParticleInteger const end = (size_t) (bid + 1) * count / numberOfBlocks;
    ParticleInteger const end0 = roundUp(end - begin, BLOCK_SIZE) + begin;
    double const c1 = dt / a;
    if (doLc) {
        list->resize(0);
    }
    for (ParticleInteger i = begin + tid; i < end0; i += BLOCK_SIZE) {
        int thisRung = -1;
        double x0 = 0.0, y0 = 0.0, z0 = 0.0, x1 = 0.0, y1 = 0.0, z1 = 0.0;
        Array<double, NDIM> x0Value;
        Array<double, NDIM> x1Value;
        float vx, vy, vz;
        float* vel;
        if (i < end) {
            thisRung = rungs[i];
            if ((rung < 0 || thisRung == rung)) {
                vel = vels[i].data();
                x0 = x[i].toDouble();
                y0 = y[i].toDouble();
                z0 = z[i].toDouble();
                x1 = x0 + double(vel[XDIM] * c1);
                y1 = y0 + double(vel[YDIM] * c1);
                z1 = z0 + double(vel[ZDIM] * c1);
                if (drag != 0.0) {
                    double const c0 = 1.0 / (1.0 + drag * dt);
                    for (int dim = 0; dim < NDIM; dim++) {
                        vel[dim] *= c0;
                    }
                }
            }
        }
        if (doLc) {
            for (int xi = -1; xi <= 0; xi++) {
                x0Value[XDIM] = x0 + (double) xi;
                x1Value[XDIM] = x1 + (double) xi;
                for (int yi = -1; yi <= 0; yi++) {
                    x0Value[YDIM] = y0 + (double) yi;
                    x1Value[YDIM] = y1 + (double) yi;
                    for (int zi = -1; zi <= 0; zi++) {
                        bool test = false;
                        LightconeEntry entry;
                        if (i < end && (rung < 0 || thisRung == rung)) {
                            x0Value[ZDIM] = z0 + (double) zi;
                            x1Value[ZDIM] = z1 + (double) zi;
                            double const dist0 = sqrt(sqr(x0Value[0], x0Value[1], x0Value[2]));
                            double const dist1 = sqrt(sqr(x1Value[0], x1Value[1], x1Value[2]));
                            double const lightconeTau0 = double(tau0) + dist0;
                            double const lightconeTau1 = double(tau1) + dist1;
                            int const i0 = (double) (lightconeTau0 * c0);
                            int const i1 = (double) (lightconeTau1 * c0);
                            if (i0 != i1 && ((dist1 <= 1.0 || dist0 <= 1.0))) {
                                vx = vel[XDIM] * ainv;
                                vy = vel[YDIM] * ainv;
                                vz = vel[ZDIM] * ainv;
                                double const& x0 = x0Value[XDIM];
                                double const& y0 = x0Value[YDIM];
                                double const& z0 = x0Value[ZDIM];
                                double const ti = double((double) (i0 + 1)) * double(tauMax);
                                double const sqrtauimtau0 = sqr(ti - t0);
                                double const tau0mtaui = t0 - ti;
                                double const aValue = 1.0 - sqr(vx, vy, vz);    // 1
                                double const b =
                                    2.0 * (tau0mtaui - (vx * x0 + vy * y0 + vz * z0));    // 2
                                double const c = sqrtauimtau0 - sqr(x0, y0, z0);          // 1
                                double const t =
                                    -(b + sqrt(b * b - 4.0 * aValue * c)) / (2.0 * aValue);    // 15
                                double const x1 = x0 + vx * t;                                 // 2
                                double const y1 = y0 + vy * t;                                 // 2
                                double const z1 = z0 + vz * t;                                 // 2
                                double const r2 = sqr(x1, y1, z1);
                                if (r2 <= 1.0) {
                                    long ipix;
                                    double xValue[NDIM] = {x1, y1, z1};
                                    vec2pixNest(nside, xValue, &ipix);
                                    entry.pix = ipix;
                                    entry.x = x1;
                                    entry.y = y1;
                                    entry.z = z1;
                                    entry.vx = vel[XDIM];
                                    entry.vy = vel[YDIM];
                                    entry.vz = vel[ZDIM];
                                    test = true;
                                }
                            }
                        }
                        int index = test;
                        int total;
                        computeIndices<BLOCK_SIZE>(index, total);
                        if (total) {
                            auto const start = list->size();
                            list->resize(start + total);
                            if (test) {
                                (*list)[start + index] = entry;
                            }
                        }
                    }
                }
            }
        }
        if (i < end && (rung < 0 || thisRung == rung)) {
            x1 -= double(x1 >= 1.0);
            y1 -= double(y1 >= 1.0);
            z1 -= double(z1 >= 1.0);
            x1 += double(x1 < 0.0);
            y1 += double(y1 < 0.0);
            z1 += double(z1 < 0.0);
            x[i] = x1;
            y[i] = y1;
            z[i] = z1;
        }
    }
}

void cudaDrift(int rung, float a, float dt, float tau0, float tau1, float tauMax, double drag) {
    static bool init = false;
    int numberOfBlocks;
    Timer tm1;
    tm1.start();
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberOfBlocks, (void const*) cudaDriftKernel, BLOCK_SIZE, 0));
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (void const*) cudaDriftKernel));
    numberOfBlocks *= cudaSmpCount();
    auto const rng = rung < 0 ? Pair<ParticleInteger>(0, particlesSize()) : particlesCurrentRange();
    if (rng.first == rng.second) {
        return;
    }
    bool const doLc = getOptions().lightcone.enabled && tauMax - tau1 < 1.0f;
    numberOfBlocks = std::min(numberOfBlocks, std::max((rng.second - rng.first) / BLOCK_SIZE, 1));
    auto cnt = rng.second - rng.first;
    if (doLc) {
        if (!init) {
            listPtr = (ListType*) cudaAllocate(sizeof(ListType));
            new (listPtr) ListType;
            init = true;
        }
        listPtr->resize(numberOfBlocks);
    }
    Fixed32* x = &particlesPos(XDIM, rng.first);
    Fixed32* y = &particlesPos(YDIM, rng.first);
    Fixed32* z = &particlesPos(ZDIM, rng.first);
    auto* vels = particlesVelData() + rng.first;
    auto const* rungs = &particlesRung(rng.first);
    drag = getOptions().initial.createGlass ? M_PI : 0.0;
    CudaStream stream(cudaGetDevice());
    cudaDriftKernel<<<numberOfBlocks, BLOCK_SIZE, 0, stream.nativeHandle()>>>(x, y, z, vels, rungs,
        cnt, rung, a, dt, tau0, tau1, tauMax, doLc, getOptions().lightcone.mapSize, drag);
    stream.checkLaunch();
    stream.wait();
    cnt = 0;
    static size_t totalCnt = 0;
    if (doLc) {
        cnt += lcAddParticles(*listPtr, a, tau0);
    }
    if (cnt != 0) {
        totalCnt += cnt;
        PRINT("%i flushed now %i flushed total\n", cnt, totalCnt);
    }

    tm1.stop();
}
