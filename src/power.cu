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

#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>

#define BLOCK_SIZE 256

__global__ void accumulateDensityKernel(Fixed32* xValue, Fixed32* yValue, Fixed32* zValue,
    float* rho, float const* masses, int mfold, size_t numberOfParticles, int ndim,
    Range<int64_t> intbox) {
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const gsz = gridDim.x;
    ParticleInteger const ibegin = (size_t) bid * numberOfParticles / (size_t) gsz;
    ParticleInteger const iend = (size_t) (bid + 1) * numberOfParticles / (size_t) gsz;
    for (ParticleInteger i = ibegin + tid; i < iend; i += BLOCK_SIZE) {
        ALWAYS_ASSERT(i < numberOfParticles);
        float x = fmodf(mfold * xValue[i].toFloat(), 1.0f);
        float y = fmodf(mfold * yValue[i].toFloat(), 1.0f);
        float z = fmodf(mfold * zValue[i].toFloat(), 1.0f);
        int const j = (x * ndim);
        if (j >= intbox.begin[XDIM] + 1 && j < intbox.end[XDIM] - 1) {
            int const k = (y * ndim);
            if (k >= intbox.begin[YDIM] + 1 && k < intbox.end[YDIM] - 1) {
                int const l = (z * ndim);
                if (l >= intbox.begin[ZDIM] + 1 && l < intbox.end[ZDIM] - 1) {
                    float const x0 = x * ndim - j;
                    float const y0 = y * ndim - k;
                    float const z0 = z * ndim - l;
                    for (int j0 = CLOUD_MIN; j0 <= CLOUD_MAX; j0++) {
                        float wtX = cloudWeight(x0 - j0);
                        for (int k0 = CLOUD_MIN; k0 <= CLOUD_MAX; k0++) {
                            float wtY = cloudWeight(y0 - k0);
                            for (int l0 = CLOUD_MIN; l0 <= CLOUD_MAX; l0++) {
                                float wtZ = cloudWeight(z0 - l0);
                                float const mass = masses == nullptr ? 1.0f : masses[i];
                                float const wt = wtX * wtY * wtZ * mass;
                                atomicAdd(rho + intbox.index(j0 + j, k0 + k, l0 + l), wt);
                            }
                        }
                    }
                }
            }
        }
    }
}

Vector<float> accumulateDensityCuda(int m, int ndim, Range<int64_t> intbox) {
    int numberOfBlocks;
    auto const n3 = intbox.volume();
    Vector<float> rho(n3, 0.0f);
    if (particlesSize() == 0) {
        return rho;
    }
    float* devRho;
    float* masses = nullptr;
    if (getOptions().sph) {
        masses = static_cast<float*>(cudaAllocate(particlesSize() * sizeof(float)));
        for (ParticleInteger i = 0; i < particlesSize(); i++) {
            masses[i] = particlesMass(i);
        }
    }
    CUDA_CHECK(cudaMallocManaged(&devRho, n3 * sizeof(float)));
    for (int i = 0; i < n3; i++) {
        devRho[i] = 0.f;
    }
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberOfBlocks, (void const*) accumulateDensityKernel, BLOCK_SIZE, 0));
    numberOfBlocks *= cudaSmpCount();
    accumulateDensityKernel<<<numberOfBlocks, BLOCK_SIZE>>>(&particlesPos(XDIM, 0),
        &particlesPos(YDIM, 0), &particlesPos(ZDIM, 0), devRho, masses, m, particlesSize(), ndim,
        intbox);
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(rho.data(), devRho, sizeof(float) * n3);
    CUDA_CHECK(cudaFree(devRho));
    if (masses != nullptr) {
        cudaRelease(masses);
    }
    return rho;
}
