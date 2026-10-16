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

#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>

static void computeDensityFolded(int m);

HPX_PLAIN_ACTION(computeDensityFolded, ComputeDensityFoldedAction);

PowerSpectrumT powerSpectrumCompute(int m) {
    int const n = getOptions().nfour;
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

static void computeDensityFolded(int m) {
    Vector<float> rho0;
    Vector<hpx::future<void>> futs1;
    Vector<hpx::future<void>> futs2;
    for (auto c : hpxChildren()) {
        futs1.push_back(hpx::async<ComputeDensityFoldedAction>(c, m));
    }
    size_t const ndim = getOptions().nfour;
    size_t const nparts = getOptions().nparts;
    size_t const partsPerRank = nparts / hpxSize();
    size_t nper = ndim;
    while (sqr(nper) * nper > partsPerRank) {
        nper /= 2;
    }
    int const np = ndim / nper;
    for (int j = 0; j < np; j++) {
        for (int k = 0; k < np; k++) {
            for (int l = 0; l < np; l++) {
                Range<int64_t> intbox;
                intbox.begin[XDIM] = j * nper + CLOUD_MIN;
                intbox.begin[YDIM] = k * nper + CLOUD_MIN;
                intbox.begin[ZDIM] = l * nper + CLOUD_MIN;
                intbox.end[XDIM] = (j + 1) * nper + CLOUD_MAX;
                intbox.end[YDIM] = (k + 1) * nper + CLOUD_MAX;
                intbox.end[ZDIM] = (l + 1) * nper + CLOUD_MAX;
                auto rho = accumulateDensityCuda(m, ndim, intbox);
                fft3dAccumulateReal(intbox, rho);
            }
        }
    }
    hpx::wait_all(futs1.begin(), futs1.end());
}
