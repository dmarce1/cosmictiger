/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/fmm/ewald_indices.hpp>
#include <cosmictiger/fmm/ewald_tables.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/math/math.hpp>

EwaldConstants ec;
__constant__ EwaldConstants ecDev;

void EwaldConst::initGpu() {
    int n2max = 12;
    int nmax = std::sqrt(n2max) + 1;
    Array<float, NDIM> thisH;
    int count = 0;
    for (int i = -nmax; i <= nmax; i++) {
        for (int j = -nmax; j <= nmax; j++) {
            for (int k = -nmax; k <= nmax; k++) {
                int const i2 = i * i + j * j + k * k;
                double const x = std::max(abs(i) - 0.5, 0.0);
                double const y = std::max(abs(j) - 0.5, 0.0);
                double const z = std::max(abs(k) - 0.5, 0.0);
                double const d = sqrt(sqr(x, y, z));
                if (d < EWALD_REAL_CUTOFF && i2 > 0) {
                    thisH[0] = i;
                    thisH[1] = j;
                    thisH[2] = k;
                    ec.realIndices[count++] = thisH;
                }
            }
        }
    }
    terminal::detailPrint("count = %i %i\n", count, NREAL);
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
    if (cudaEnabled()) {
        cudaSetDevice();
        CUDA_CHECK(cudaMemcpyToSymbol(ecDev, &ec, sizeof(EwaldConstants)));
    }
}
