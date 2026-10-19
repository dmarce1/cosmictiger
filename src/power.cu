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

#include <memory>

#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/power_cuda.hpp>

Vector<float> accumulateDensityCuda(int m, int ndim, Range<int64_t> intbox) {
    constexpr int kBlockSize = 256;
    int numberOfBlocks;
    auto const n3 = intbox.volume();
    Vector<float> rho(n3, 0.0f);
    if (particlesSize() == 0) {
        return rho;
    }
    float* devRho;
    std::unique_ptr<float, decltype(&cudaRelease)> massOwner(nullptr, cudaRelease);
    float* masses = nullptr;
    if (getOptions().hydro.enabled) {
        massOwner.reset(static_cast<float*>(cudaAllocate(particlesSize() * sizeof(float))));
        masses = massOwner.get();
        for (ParticleInteger i = 0; i < particlesSize(); i++) {
            masses[i] = particlesMass(i);
        }
    }
    CUDA_CHECK(cudaMallocManaged(&devRho, n3 * sizeof(float)));
    std::unique_ptr<float, decltype(&cudaFree)> densityOwner(devRho, cudaFree);
    for (int i = 0; i < n3; i++) {
        devRho[i] = 0.f;
    }
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberOfBlocks, (void const*) accumulateDensityKernel, kBlockSize, 0));
    numberOfBlocks *= cudaSmpCount();
    CudaStream stream(cudaGetDevice());
    accumulateDensityKernel<<<numberOfBlocks, kBlockSize, 0, stream.nativeHandle()>>>(
        &particlesPos(XDIM, 0), &particlesPos(YDIM, 0), &particlesPos(ZDIM, 0), devRho, masses, m,
        particlesSize(), ndim, intbox);
    stream.checkLaunch();
    stream.wait();
    memcpy(rho.data(), devRho, sizeof(float) * n3);
    return rho;
}
