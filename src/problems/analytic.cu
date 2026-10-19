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
#include <memory>
#include <vector>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/particles/particles.hpp>

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
    auto const allocate = []<typename T>(T*& pointer, size_t bytes) {
        CUDA_MALLOC(&pointer, bytes);
        return std::unique_ptr<T, decltype(&cudaFree)>(pointer, cudaFree);
    };
    auto sinkXOwner = allocate(devSinkx, numberOfSinks * sizeof(Fixed32));
    auto sinkYOwner = allocate(devSinky, numberOfSinks * sizeof(Fixed32));
    auto sinkZOwner = allocate(devSinkz, numberOfSinks * sizeof(Fixed32));
    auto potentialOwner = allocate(devPhi, numberOfSinks * sizeof(double));
    auto forceXOwner = allocate(devGx, numberOfSinks * sizeof(double));
    auto forceYOwner = allocate(devGy, numberOfSinks * sizeof(double));
    auto forceZOwner = allocate(devGz, numberOfSinks * sizeof(double));
    Vector<double> zero(numberOfSinks, 0.0);
    Vector<double> initialPotential(
        numberOfSinks, hpxRank() == 0 ? selfPhi() / getOptions().gravity.supportRadius : 0.0);
    {
        CudaStream copies(cudaGetDevice());
        copies.copyAsync(devPhi, initialPotential.data(), numberOfSinks * sizeof(double),
            cudaMemcpyHostToDevice);
        copies.copyAsync(
            devGx, zero.data(), numberOfSinks * sizeof(double), cudaMemcpyHostToDevice);
        copies.copyAsync(
            devGy, zero.data(), numberOfSinks * sizeof(double), cudaMemcpyHostToDevice);
        copies.copyAsync(
            devGz, zero.data(), numberOfSinks * sizeof(double), cudaMemcpyHostToDevice);
        copies.copyAsync(
            devSinkx, sinkx.data(), numberOfSinks * sizeof(Fixed32), cudaMemcpyHostToDevice);
        copies.copyAsync(
            devSinky, sinky.data(), numberOfSinks * sizeof(Fixed32), cudaMemcpyHostToDevice);
        copies.copyAsync(
            devSinkz, sinkz.data(), numberOfSinks * sizeof(Fixed32), cudaMemcpyHostToDevice);
        // Independent nonblocking kernel streams cannot inherit default-stream
        // ordering, so all shared inputs must be ready before their launches.
        copies.wait();
    }
    PRINT("%li free\n", cudaFreeMemory());
    size_t const particlesPerLoop = (size_t) cudaFreeMemory() / (NDIM * sizeof(Fixed32)) * 85 / 100;
    int occupancy;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, analyticGravityKernel,
        ANALYTIC_BLOCK_SIZE, sizeof(double) * (NDIM + 1) * ANALYTIC_BLOCK_SIZE));
    int numberOfKernels = std::max((int) (occupancy * cudaSmpCount() / numberOfSinks), 1);
    std::vector<std::unique_ptr<CudaStream>> streams;
    streams.reserve(numberOfKernels);
    for (int i = 0; i < numberOfKernels; i++) {
        streams.push_back(std::make_unique<CudaStream>(cudaGetDevice()));
    }
    PRINT("%li particles per loop, %li kernels\n", particlesPerLoop, numberOfKernels);
    for (size_t i = 0; i < particlesSize(); i += particlesPerLoop) {
        int const totalSize =
            std::min(size_t(particlesSize()), size_t(i) + size_t(particlesPerLoop)) - size_t(i);
        auto sourceXOwner = allocate(devSrcx, totalSize * sizeof(Fixed32));
        auto sourceYOwner = allocate(devSrcy, totalSize * sizeof(Fixed32));
        auto sourceZOwner = allocate(devSrcz, totalSize * sizeof(Fixed32));
        try {
            CudaStream copies(cudaGetDevice());
            copies.copyAsync(
                devSrcx, &particlesPos(0, i), totalSize * sizeof(Fixed32), cudaMemcpyHostToDevice);
            copies.copyAsync(
                devSrcy, &particlesPos(1, i), totalSize * sizeof(Fixed32), cudaMemcpyHostToDevice);
            copies.copyAsync(
                devSrcz, &particlesPos(2, i), totalSize * sizeof(Fixed32), cudaMemcpyHostToDevice);
            copies.wait();
            for (int j = 0; j < numberOfKernels; j++) {
                int const begin = size_t(j) * size_t(totalSize) / size_t(numberOfKernels);
                int const end = size_t(j + 1) * size_t(totalSize) / size_t(numberOfKernels);
                analyticGravityKernel<<<numberOfSinks, ANALYTIC_BLOCK_SIZE, 0,
                    streams[j]->nativeHandle()>>>(devSinkx, devSinky, devSinkz, devSrcx + begin,
                    devSrcy + begin, devSrcz + begin, end - begin, devPhi, devGx, devGy, devGz,
                    getOptions().gravity.supportRadius);
                streams[j]->checkLaunch();
            }
            for (auto const& stream : streams) {
                stream->wait();
            }
        } catch (...) {
            // Every concurrent reader must drain before source owners unwind.
            streams.clear();
            throw;
        }
    }
    rc.first.resize(numberOfSinks);
    for (int dim = 0; dim < NDIM; dim++) {
        rc.second[dim].resize(numberOfSinks);
    }
    CudaStream copies(cudaGetDevice());
    copies.copyAsync(
        rc.first.data(), devPhi, sizeof(double) * numberOfSinks, cudaMemcpyDeviceToHost);
    copies.copyAsync(
        rc.second[XDIM].data(), devGx, sizeof(double) * numberOfSinks, cudaMemcpyDeviceToHost);
    copies.copyAsync(
        rc.second[YDIM].data(), devGy, sizeof(double) * numberOfSinks, cudaMemcpyDeviceToHost);
    copies.copyAsync(
        rc.second[ZDIM].data(), devGz, sizeof(double) * numberOfSinks, cudaMemcpyDeviceToHost);
    copies.wait();
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
