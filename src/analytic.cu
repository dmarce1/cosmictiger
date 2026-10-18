/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

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
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>

__global__ void analyticGravityKernel(Fixed32* sinkx, Fixed32* sinky, Fixed32* sinkz,
    Fixed32* sourcex, Fixed32* sourcey, Fixed32* sourcez, int numberOfSources, double* rphi,
    double* rgx, double* rgy, double* rgz, float hsoft);

#define ANALYTIC_BLOCK_SIZE 256

std::pair<Vector<double>, Array<Vector<double>, NDIM>> gravityAnalyticCallKernel(
    Vector<Fixed32> const& sinkx, Vector<Fixed32> const& sinky, Vector<Fixed32> const& sinkz) {
    cudaSetDevice();
    std::pair<Vector<double>, Array<Vector<double>, NDIM>> rc;
    Fixed32* devSinkx;
    Fixed32* devSinky;
    Fixed32* devSinkz;
    Fixed32* devSrcx;
    Fixed32* devSrcy;
    Fixed32* devSrcz;
    double* devPhi;
    double* devGx;
    double* devGy;
    double* devGz;
    int const numberOfSinks = sinkx.size();
    (CUDA_MALLOC(&devSinkx, numberOfSinks * sizeof(Fixed32)));
    (CUDA_MALLOC(&devSinky, numberOfSinks * sizeof(Fixed32)));
    (CUDA_MALLOC(&devSinkz, numberOfSinks * sizeof(Fixed32)));
    (CUDA_MALLOC(&devPhi, numberOfSinks * sizeof(double)));
    (CUDA_MALLOC(&devGx, numberOfSinks * sizeof(double)));
    (CUDA_MALLOC(&devGy, numberOfSinks * sizeof(double)));
    (CUDA_MALLOC(&devGz, numberOfSinks * sizeof(double)));
    Vector<double> zero(numberOfSinks, 0.0);
    if (hpxRank() == 0) {
        Vector<double> selfPhi1(numberOfSinks, selfPhi() / getOptions().gravity.supportRadius);
        CUDA_CHECK(cudaMemcpy(
            devPhi, selfPhi1.data(), numberOfSinks * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(
            devPhi, zero.data(), numberOfSinks * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(
        cudaMemcpy(devGx, zero.data(), numberOfSinks * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(devGy, zero.data(), numberOfSinks * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(devGz, zero.data(), numberOfSinks * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        devSinkx, sinkx.data(), numberOfSinks * sizeof(Fixed32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        devSinky, sinky.data(), numberOfSinks * sizeof(Fixed32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        devSinkz, sinkz.data(), numberOfSinks * sizeof(Fixed32), cudaMemcpyHostToDevice));
    PRINT("%li free\n", cudaFreeMemory());
    size_t const particlesPerLoop = (size_t) cudaFreeMemory() / (NDIM * sizeof(Fixed32)) * 85 / 100;
    int occupancy;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, analyticGravityKernel,
        ANALYTIC_BLOCK_SIZE, sizeof(double) * (NDIM + 1) * ANALYTIC_BLOCK_SIZE));
    int numberOfKernels = std::max((int) (occupancy * cudaSmpCount() / numberOfSinks), 1);
    Vector<cudaStream_t> streams(numberOfKernels);
    for (int i = 0; i < numberOfKernels; i++) {
        cudaStreamCreate(&streams[i]);
    }
    PRINT("%li particles per loop, %li kernels\n", particlesPerLoop, numberOfKernels);
    for (size_t i = 0; i < particlesSize(); i += particlesPerLoop) {
        int const totalSize =
            std::min(size_t(particlesSize()), size_t(i) + size_t(particlesPerLoop)) - size_t(i);
        (CUDA_MALLOC(&devSrcx, totalSize * sizeof(Fixed32)));
        (CUDA_MALLOC(&devSrcy, totalSize * sizeof(Fixed32)));
        (CUDA_MALLOC(&devSrcz, totalSize * sizeof(Fixed32)));
        CUDA_CHECK(cudaMemcpy(
            devSrcx, &particlesPos(0, i), totalSize * sizeof(Fixed32), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            devSrcy, &particlesPos(1, i), totalSize * sizeof(Fixed32), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            devSrcz, &particlesPos(2, i), totalSize * sizeof(Fixed32), cudaMemcpyHostToDevice));
        for (int j = 0; j < numberOfKernels; j++) {
            int const begin = size_t(j) * size_t(totalSize) / size_t(numberOfKernels);
            int const end = size_t(j + 1) * size_t(totalSize) / size_t(numberOfKernels);
            analyticGravityKernel<<<numberOfSinks, ANALYTIC_BLOCK_SIZE, 0, streams[j]>>>(devSinkx,
                devSinky, devSinkz, devSrcx + begin, devSrcy + begin, devSrcz + begin, end - begin,
                devPhi, devGx, devGy, devGz, getOptions().gravity.supportRadius);
        }
        for (int i = 0; i < numberOfKernels; i++) {
            synchronizeCudaStream(streams[i]);
        }
        CUDA_CHECK(cudaFree(devSrcx));
        CUDA_CHECK(cudaFree(devSrcy));
        CUDA_CHECK(cudaFree(devSrcz));
    }
    for (int i = 0; i < numberOfKernels; i++) {
        cudaStreamDestroy(streams[i]);
    }
    rc.first.resize(numberOfSinks);
    for (int dim = 0; dim < NDIM; dim++) {
        rc.second[dim].resize(numberOfSinks);
    }
    CUDA_CHECK(cudaMemcpy(
        rc.first.data(), devPhi, sizeof(double) * numberOfSinks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        rc.second[XDIM].data(), devGx, sizeof(double) * numberOfSinks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        rc.second[YDIM].data(), devGy, sizeof(double) * numberOfSinks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        rc.second[ZDIM].data(), devGz, sizeof(double) * numberOfSinks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(devPhi));
    CUDA_CHECK(cudaFree(devGx));
    CUDA_CHECK(cudaFree(devGy));
    CUDA_CHECK(cudaFree(devGz));
    CUDA_CHECK(cudaFree(devSinkx));
    CUDA_CHECK(cudaFree(devSinky));
    CUDA_CHECK(cudaFree(devSinkz));
    return rc;
}

__global__ void analyticGravityKernel(Fixed32* sinkx, Fixed32* sinky, Fixed32* sinkz,
    Fixed32* sourcex, Fixed32* sourcey, Fixed32* sourcez, int numberOfSources, double* rphi,
    double* rgx, double* rgy, double* rgz, float h) {
    __shared__ double phi[ANALYTIC_BLOCK_SIZE];
    __shared__ double gx[ANALYTIC_BLOCK_SIZE];
    __shared__ double gy[ANALYTIC_BLOCK_SIZE];
    __shared__ double gz[ANALYTIC_BLOCK_SIZE];

    int const& tid = threadIdx.x;
    int const& bid = blockIdx.x;

    Fixed32 const x = sinkx[bid];
    Fixed32 const y = sinky[bid];
    Fixed32 const z = sinkz[bid];
    float const cons1 = float(4.0f / sqrtf(M_PI));
    float const h2 = h * h;
    float const hinv = 1.0f / h;
    float const h2inv = hinv * hinv;
    float const h3inv = h2inv * hinv;
    phi[tid] = gx[tid] = gy[tid] = gz[tid] = 0.0f;
    for (int sourcei = tid; sourcei < numberOfSources; sourcei += ANALYTIC_BLOCK_SIZE) {
        float const xValue = distance(x, sourcex[sourcei]);
        float const yValue = distance(y, sourcey[sourcei]);
        float const zValue = distance(z, sourcez[sourcei]);
        float const r2Value = sqr(xValue, yValue, zValue);
        float fx = 0.f;
        float fy = 0.f;
        float fz = 0.f;
        float pot = 0.f;
        if (r2Value > 0.f) {
            for (int xi = -4; xi <= +4; xi++) {
                for (int yi = -4; yi <= +4; yi++) {
                    for (int zi = -4; zi <= +4; zi++) {
                        float const dx = xValue - xi;
                        float const dy = yValue - yi;
                        float const dz = zValue - zi;
                        float const r2 = sqr(dx, dy, dz);
                        if (r2 < 2.6f * 2.6f) {
                            float const r = sqrt(r2);
                            float const rinv = 1.f / r;
                            float const r2inv = rinv * rinv;
                            float const r3inv = r2inv * rinv;
                            float exp0;
                            float erfc0;
                            erfcexp(2.f * r, &erfc0, &exp0);
                            float const expfactor = cons1 * r * exp0;
                            float const d0 = -erfc0 * rinv;
                            float const d1 = (expfactor + erfc0) * r3inv;
                            pot += d0;
                            fx -= dx * d1;
                            fy -= dy * d1;
                            fz -= dz * d1;
                        }
                    }
                }
            }
            phi[tid] += float(M_PI / 4.f);
            for (int xi = -2; xi <= +2; xi++) {
                for (int yi = -2; yi <= +2; yi++) {
                    for (int zi = -2; zi <= +2; zi++) {
                        float const hx = xi;
                        float const hy = yi;
                        float const hz = zi;
                        float const h2 = sqr(hx, hy, hz);
                        if (h2 > 0.0f && h2 <= 8) {
                            float const hdotx = xValue * hx + yValue * hy + zValue * hz;
                            float const omega = float(2.0 * M_PI) * hdotx;
                            float c, s;
                            sincosf(omega, &s, &c);
                            float const c0 = -1.0f / h2 * expf(float(-M_PI * M_PI * 0.25f) * h2) *
                                float(1.f / M_PI);
                            float const c1 = -s * 2.0 * M_PI * c0;
                            pot += c0 * c;
                            fx -= c1 * hx;
                            fy -= c1 * hy;
                            fz -= c1 * hz;
                        }
                    }
                }
            }
        }
        if (r2Value == 0.f) {
            phi[tid] += 2.837291f;
            phi[tid] -= selfPhi() * hinv;
        } else if (r2Value < h2) {
            float softenedForce;
            float softenedPotential;
            gravitySofteningKernel(
                softenedForce, softenedPotential, r2Value, hinv, h2inv, h3inv, true);
            // The Ewald sum already includes the Newtonian central image.
            float const inverseRadius = rsqrtf(r2Value);
            float const forceCorrection =
                softenedForce - inverseRadius * inverseRadius * inverseRadius;
            fx -= xValue * forceCorrection;
            fy -= yValue * forceCorrection;
            fz -= zValue * forceCorrection;
            pot -= softenedPotential - inverseRadius;
        }
        phi[tid] += pot;
        gx[tid] += fx;
        gy[tid] += fy;
        gz[tid] += fz;
    }
    __syncthreads();
    for (int p = ANALYTIC_BLOCK_SIZE / 2; p >= 1; p /= 2) {
        if (tid < p) {
            gx[tid] += gx[tid + p];
            gy[tid] += gy[tid + p];
            gz[tid] += gz[tid + p];
            phi[tid] += phi[tid + p];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(rphi + bid, phi[0]);
        atomicAdd(rgx + bid, gx[0]);
        atomicAdd(rgy + bid, gy[0]);
        atomicAdd(rgz + bid, gz[0]);
    }
}
