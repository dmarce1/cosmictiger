/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <memory>

#include <cosmictiger/analysis/power.hpp>
#include <cosmictiger/analysis/power_cuda.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>

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
