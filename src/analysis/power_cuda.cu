/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/analysis/power_cuda.hpp>
#include <cosmictiger/cosmology/fft.hpp>

__global__ void accumulateDensityKernel(Fixed32 const* xValue, Fixed32 const* yValue,
    Fixed32 const* zValue, float* rho, float const* masses, int mfold, size_t numberOfParticles,
    int ndim, Range<int64_t> intbox) {
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const gsz = gridDim.x;
    size_t const ibegin = size_t(bid) * numberOfParticles / size_t(gsz);
    size_t const iend = size_t(bid + 1) * numberOfParticles / size_t(gsz);
    for (size_t i = ibegin + tid; i < iend; i += blockDim.x) {
        ALWAYS_ASSERT(i < numberOfParticles);
        float x = fmodf(mfold * xValue[i].toFloat(), 1.0f);
        float y = fmodf(mfold * yValue[i].toFloat(), 1.0f);
        float z = fmodf(mfold * zValue[i].toFloat(), 1.0f);
        int const j = (x * ndim);
        if (j >= intbox.begin[XDIM] - CLOUD_MIN && j < intbox.end[XDIM] - CLOUD_MAX) {
            int const k = (y * ndim);
            if (k >= intbox.begin[YDIM] - CLOUD_MIN && k < intbox.end[YDIM] - CLOUD_MAX) {
                int const l = (z * ndim);
                if (l >= intbox.begin[ZDIM] - CLOUD_MIN && l < intbox.end[ZDIM] - CLOUD_MAX) {
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
