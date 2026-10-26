/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>

#include <chealpix.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/dehnen/ewald_indices.hpp>
#include <cosmictiger/gravity/dehnen/ewald_tables.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/math/math.hpp>

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
    Array<float, kDimensions> thisH;
    int count = 0;
    for (int i = -nmax; i <= nmax; i++) {
        for (int j = -nmax; j <= nmax; j++) {
            for (int k = -nmax; k <= nmax; k++) {
                int const i2 = i * i + j * j + k * k;
                double const x = std::max(std::abs(i) - 0.5, 0.0);
                double const y = std::max(std::abs(j) - 0.5, 0.0);
                double const z = std::max(std::abs(k) - 0.5, 0.0);
                double const distance = std::sqrt(sqr(x, y, z));
                if (distance < kEwaldRealCutoff && i2 > 0) {
                    thisH[0] = i;
                    thisH[1] = j;
                    thisH[2] = k;
                    ec.realIndices[count++] = thisH;
                }
            }
        }
    }
    auto const sortFunc = [](Array<float, kDimensions> const& a,
                              Array<float, kDimensions> const& b) {
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
