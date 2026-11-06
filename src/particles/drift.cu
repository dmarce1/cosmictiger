/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <format>
#include <iostream>
#include <numbers>

#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/math/vector3.hpp>
#endif
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/cuda/device_vector.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/particles/drift.hpp>
#include <cosmictiger/particles/particles.hpp>

constexpr int kDriftBlockSize = 512;

#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
using ListType = DeviceVector<DeviceVector<LightconeEntry>>;

__managed__ ListType* listPtr;
#endif

__global__ void cudaDriftKernel(Position* __restrict__ const x, Position* __restrict__ const y,
    Position* __restrict__ const z, ParticleVector<float>* const vels, char const* const rungs,
    char const* const types, ParticleInteger count, int rung, double a, double dt, double tau0,
    double tau1, double tauMax, bool doLc, int nside, double drag) {
    auto const& tid = threadIdx.x;
    auto const& bid = blockIdx.x;
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
    doLc = doLc && tauMax > 0.0 && (tauMax - tau1 < 1.0);
    auto* list = doLc ? &(*listPtr)[bid] : nullptr;
    double const c0 = doLc ? 1.0 / tauMax : 0.0;
    double const t0(tau0);
    double const ainv = 1.0 / a;
#endif
    auto const& numberOfBlocks = gridDim.x;
    ParticleInteger const begin = (size_t) bid * count / numberOfBlocks;
    ParticleInteger const end = (size_t) (bid + 1) * count / numberOfBlocks;
    ParticleInteger const end0 = roundUp(end - begin, ParticleInteger(kDriftBlockSize)) + begin;
    double const c1 = dt / a;
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
    if (doLc) {
        list->resize(0);
    }
#endif
    for (ParticleInteger i = begin + tid; i < end0; i += kDriftBlockSize) {
        int thisRung = -1;
        bool driftParticle = false;
        double x0 = 0.0, y0 = 0.0, z0 = 0.0, x1 = 0.0, y1 = 0.0, z1 = 0.0;
        double displacementX = 0.0, displacementY = 0.0, displacementZ = 0.0;
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
        Vector3<double> x0Value;
        Vector3<double> x1Value;
        float vx = 0.0f, vy = 0.0f, vz = 0.0f;
#endif
        float* vel;
        if (i < end) {
            thisRung = rungs[i];
            if ((rung < 0 || thisRung == rung)) {
                vel = vels[i].data();
                driftParticle = true;
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
                vx = double(vel[kXDimension]) * ainv;
                vy = double(vel[kYDimension]) * ainv;
                vz = double(vel[kZDimension]) * ainv;
#endif
                x0 = x[i].toDouble();
                y0 = kSpatialDimensions > 1 ? y[i].toDouble() : 0.5;
                z0 = kSpatialDimensions > 2 ? z[i].toDouble() : 0.5;
                displacementX = double(vel[kXDimension]) * c1;
                displacementY = kSpatialDimensions > 1 ? double(vel[kYDimension]) * c1 : 0.0;
                displacementZ = kSpatialDimensions > 2 ? double(vel[kZDimension]) * c1 : 0.0;
                x1 = x0 + displacementX;
                y1 = y0 + displacementY;
                z1 = z0 + displacementZ;
                if (drag != 0.0) {
                    double const c0 = 1.0 / (1.0 + drag * dt);
                    for (int dim = 0; dim < kSpatialDimensions; dim++) {
                        vel[dim] *= c0;
                    }
                }
            }
        }
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
        if (doLc) {
            for (int xi = -1; xi <= 0; xi++) {
                x0Value[kXDimension] = x0 + (double) xi;
                x1Value[kXDimension] = x1 + (double) xi;
                for (int yi = -1; yi <= 0; yi++) {
                    x0Value[kYDimension] = y0 + (double) yi;
                    x1Value[kYDimension] = y1 + (double) yi;
                    for (int zi = -1; zi <= 0; zi++) {
                        bool test = false;
                        LightconeEntry entry;
                        if (driftParticle) {
                            x0Value[kZDimension] = z0 + (double) zi;
                            x1Value[kZDimension] = z1 + (double) zi;
                            double const dist0 = sqrt(sqr(x0Value[0], x0Value[1], x0Value[2]));
                            double const dist1 = sqrt(sqr(x1Value[0], x1Value[1], x1Value[2]));
                            double const lightconeTau0 = double(tau0) + dist0;
                            double const lightconeTau1 = double(tau1) + dist1;
                            int const i0 = (double) (lightconeTau0 * c0);
                            int const i1 = (double) (lightconeTau1 * c0);
                            if (i0 != i1 && ((dist1 <= 1.0 || dist0 <= 1.0))) {
                                double const& x0 = x0Value[kXDimension];
                                double const& y0 = x0Value[kYDimension];
                                double const& z0 = x0Value[kZDimension];
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
                                    Vector3<double> xValue{x1, y1, z1};
                                    vec2pixNest(nside, xValue.data(), &ipix);
                                    entry.pix = ipix;
                                    entry.x = x1;
                                    entry.y = y1;
                                    entry.z = z1;
                                    entry.vx = vel[kXDimension];
                                    entry.vy = vel[kYDimension];
                                    entry.vz = vel[kZDimension];
                                    entry.type =
                                        types != nullptr ? types[i] : char(kDarkMatterType);
                                    test = true;
                                }
                            }
                        }
                        int index = test;
                        int total;
                        computeIndices<kDriftBlockSize>(index, total);
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
#endif
        if (driftParticle) {
            if constexpr (kPositionBits == 64) {
                // Preserve sub-double motion and the original low position bits.
                x[i] += Position(displacementX);
                if constexpr (kSpatialDimensions > 1) {
                    y[i] += Position(displacementY);
                }
                if constexpr (kSpatialDimensions > 2) {
                    z[i] += Position(displacementZ);
                }
            } else {
                x[i].raw() +=
                    static_cast<PositionInteger>(llrint(remainder(displacementX, 1.0) * 0x1p32));
                if constexpr (kSpatialDimensions > 1) {
                    y[i].raw() += static_cast<PositionInteger>(
                        llrint(remainder(displacementY, 1.0) * 0x1p32));
                }
                if constexpr (kSpatialDimensions > 2) {
                    z[i].raw() += static_cast<PositionInteger>(
                        llrint(remainder(displacementZ, 1.0) * 0x1p32));
                }
            }
        }
    }
}

void cudaDrift(
    int rung, double a, double dt, double tau0, double tau1, double tauMax, double drag) {
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
    static bool init = false;
#endif
    int numberOfBlocks;
    Timer tm1;
    tm1.start();
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberOfBlocks, (void const*) cudaDriftKernel, kDriftBlockSize, 0));
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (void const*) cudaDriftKernel));
    numberOfBlocks *= cudaSmpCount();
    auto const rng = rung < 0 ? Pair<ParticleInteger>(0, particlesSize()) : particlesCurrentRange();
    if (rng.first == rng.second) {
        return;
    }
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
    bool const doLc = getOptions().lightcone.enabled && tauMax > 0.0 && tauMax - tau1 < 1.0;
#else
    bool const doLc = false;
#endif
    numberOfBlocks = static_cast<int>(std::min<ParticleInteger>(
        numberOfBlocks, std::max<ParticleInteger>((rng.second - rng.first) / kDriftBlockSize, 1)));
    auto cnt = rng.second - rng.first;
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
    if (doLc) {
        if (!init) {
            listPtr = (ListType*) cudaAllocate(sizeof(ListType));
            new (listPtr) ListType;
            init = true;
        }
        listPtr->resize(numberOfBlocks);
    }
#endif
    Position* x = particlesPositionData(kXDimension, rng.first);
    Position* y = particlesPositionData(kYDimension, rng.first);
    Position* z = particlesPositionData(kZDimension, rng.first);
    auto* vels = particlesVelData() + rng.first;
    auto const* rungs = &particlesRung(rng.first);
    // particlesTy is managed memory; particlesType() returns a shared static when hydro is off.
    char const* const types = getOptions().hydro.enabled ? particlesTy + rng.first : nullptr;
    drag = getOptions().run.scenario == "glass" ? std::numbers::pi : 0.0;
    CudaStream stream(cudaGetDevice());
    cudaDriftKernel<<<numberOfBlocks, kDriftBlockSize, 0, stream.nativeHandle()>>>(x, y, z, vels,
        rungs, types, cnt, rung, a, dt, tau0, tau1, tauMax, doLc, getOptions().lightcone.mapSize,
        drag);
    stream.checkLaunch();
    stream.wait();
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
    cnt = 0;
    static size_t totalCnt = 0;
    if (doLc) {
        cnt += lcAddParticles(*listPtr, a, tau0);
    }
    if (cnt != 0) {
        totalCnt += cnt;
        std::cout << std::format("{:d} flushed now {:d} flushed total\n", cnt, totalCnt)
                  << std::flush;
    }

#endif
    tm1.stop();
}
