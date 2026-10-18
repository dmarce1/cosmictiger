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

#include <algorithm>

#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/power_cpu.hpp>

static void computeDensityFolded(int m);

HPX_PLAIN_ACTION(computeDensityFolded, ComputeDensityFoldedAction);

PowerSpectrumT powerSpectrumCompute(int m) {
    int const n = getOptions().initial.fourierSize;
    fft3dInit(n, -1.0f);
    computeDensityFolded(m);
    fft3dExecute();
    auto power = fft3dPowerSpectrum();
    for (int i = 0; i < power.k.size(); i++) {
        power.k[i] *= m;
    }
    fft3dDestroy();
    return power;
}

Vector<float> accumulateDensityCpu(int fold, int gridSize, Range<int64_t> box) {
    Vector<float> density(box.volume(), 0.0f);
    size_t const particleCount = particlesSize();
    if (particleCount == 0) {
        return density;
    }
    Vector<float> masses;
    if (getOptions().hydro.enabled) {
        masses.resize(particleCount);
        for (size_t index = 0; index < particleCount; index++) {
            masses[index] = particlesMass(index);
        }
    }
    accumulateDensityCpuKernel(&particlesPos(XDIM, 0), &particlesPos(YDIM, 0),
        &particlesPos(ZDIM, 0), masses.empty() ? nullptr : masses.data(), particleCount, fold,
        gridSize, box, density.data());
    return density;
}

static void computeDensityFolded(int m) {
    Vector<hpx::future<void>> futs1;
    for (auto c : hpxChildren()) {
        futs1.push_back(hpx::async<ComputeDensityFoldedAction>(c, m));
    }
    size_t const ndim = getOptions().initial.fourierSize;
    size_t const numberOfParticles = getOptions().initial.numberOfParticles;
    size_t const particlesPerRank = numberOfParticles / hpxSize();
    size_t nper = ndim;
    while (nper > 1 && sqr(nper) * nper > particlesPerRank) {
        nper /= 2;
    }
    for (size_t j = 0; j < ndim; j += nper) {
        for (size_t k = 0; k < ndim; k += nper) {
            for (size_t l = 0; l < ndim; l += nper) {
                Range<int64_t> intbox;
                intbox.begin[XDIM] = int64_t(j) + CLOUD_MIN;
                intbox.begin[YDIM] = int64_t(k) + CLOUD_MIN;
                intbox.begin[ZDIM] = int64_t(l) + CLOUD_MIN;
                intbox.end[XDIM] = std::min(j + nper, ndim) + CLOUD_MAX;
                intbox.end[YDIM] = std::min(k + nper, ndim) + CLOUD_MAX;
                intbox.end[ZDIM] = std::min(l + nper, ndim) + CLOUD_MAX;
                Vector<float> rho;
#ifdef USE_CUDA
                if (cudaEnabled()) {
                    rho = accumulateDensityCuda(m, ndim, intbox);
                } else
#endif
                {
                    rho = accumulateDensityCpu(m, ndim, intbox);
                }
                fft3dAccumulateReal(intbox, rho);
            }
        }
    }
    hpx::wait_all(futs1.begin(), futs1.end());
    for (auto& future : futs1) {
        future.get();
    }
}
