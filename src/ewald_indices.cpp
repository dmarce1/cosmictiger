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
#include <cmath>

#include <chealpix.h>

#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/ewald_tables.hpp>
#include <cosmictiger/fmm_order.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>

#define NHEAL 1024

HPX_PLAIN_ACTION(EwaldConst::init, EwaldConstInitAction);

#ifndef USE_CUDA

EwaldConstants ec;

#endif

void EwaldConst::init() {
    Vector<hpx::future<void>> futs;
    auto children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<EwaldConstInitAction>(c));
    }

#ifdef USE_CUDA
    EwaldConst::initGpu();
#else

    int n2max = 12;
    int nmax = std::sqrt(n2max) + 1;
    Array<float, NDIM> thisH;
    int count = 0;
    for (int i = -nmax; i <= nmax; i++) {
        for (int j = -nmax; j <= nmax; j++) {
            for (int k = -nmax; k <= nmax; k++) {
                int const i2 = i * i + j * j + k * k;
                double const x = std::max(std::abs(i) - 0.5, 0.0);
                double const y = std::max(std::abs(j) - 0.5, 0.0);
                double const z = std::max(std::abs(k) - 0.5, 0.0);
                double const distance = std::sqrt(sqr(x, y, z));
                if (distance < EWALD_REAL_CUTOFF && i2 > 0) {
                    thisH[0] = i;
                    thisH[1] = j;
                    thisH[2] = k;
                    ec.realIndices[count++] = thisH;
                }
            }
        }
    }
    auto const sortFunc = [](Array<float, NDIM> const& a, Array<float, NDIM> const& b) {
        auto const a2 = sqr(a[0], a[1], a[2]);
        auto const b2 = sqr(b[0], b[1], b[2]);
        return a2 > b2;
    };
    std::sort(ec.realIndices.begin(), ec.realIndices.end(), sortFunc);
    n2max = 8;
    nmax = std::sqrt(n2max) + 1;
    count = 0;
    for (int i = -nmax; i <= nmax; i++) {
        for (int j = -nmax; j <= nmax; j++) {
            for (int k = -nmax; k <= nmax; k++) {
                if (i * i + j * j + k * k <= n2max) {
                    thisH[0] = i;
                    thisH[1] = j;
                    thisH[2] = k;
                    auto const hdot = sqr(thisH[0]) + sqr(thisH[1]) + sqr(thisH[2]);
                    if (hdot > 0) {
                        ec.fourIndices[count++] = thisH;
                    }
                }
            }
        }
    }
    std::sort(ec.fourIndices.begin(), ec.fourIndices.end(), sortFunc);
    // Derivative tables for the runtime order (ewald_tables.hpp).
    fmmDispatch([](auto orderTag) { ewaldFillTables<decltype(orderTag)::value>(ec); });
#endif
    hpx::wait_all(futs.begin(), futs.end());
}
