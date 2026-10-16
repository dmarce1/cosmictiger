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

#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
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
    PRINT("count = %i %i\n", count, NREAL);
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
    count = 0;
    for (int i = 0; i < NFOUR; i++) {
        Array<float, NDIM> h = ec.fourIndices[i];
        auto d0 = vectorToSymTensor<float, LORDER>(h);
        float const h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);    // 5 OP
        float const c0 = -1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) / M_PI;
        Array<int, NDIM> n;
        int const signs[4] = {1, -1, -1, 1};
        for (n[0] = 0; n[0] < LORDER; n[0]++) {
            for (n[1] = 0; n[1] < LORDER - n[0]; n[1]++) {
                for (n[2] = 0; n[2] < LORDER - n[0] - n[1]; n[2]++) {
                    int const n0 = n[0] + n[1] + n[2];
                    d0(n) *= (signs[n0 % 4] * pow(2.0 * M_PI * SCALE_FACTOR_INV1, n0) * c0) *
                        SCALE_FACTOR_INV1;
                }
            }
        }
        ec.fourExpanse[count] = d0.detraceD();
        count++;
    }
    TensorSym<float, LORDER> dValue;
    for (int n = 0; n < (LORDER + 2) * (LORDER + 1) * LORDER / 6; n++) {
        dValue[n] = 0.0;
    }
    constexpr double kAlpha = 2.0;
    for (int n = 0; n < LORDER; n += 2) {
        for (int m = 0; m < LORDER - n; m += 2) {
            for (int l = 0; l < LORDER - n - m; l += 2) {
                dValue(n, m, l) = pow(SCALE_FACTOR_INV1, n + m + l + 1) *
                    pow(-2.0, (n + m + l) / 2 + 1) / ((n + m + l + 1.0) * sqrt(M_PI)) *
                    pow(kAlpha, n + m + l + 1) * doubleFactorial(n - 1) * doubleFactorial(m - 1) *
                    doubleFactorial(l - 1);
            }
        }
    }
    ec.d0 = dValue;
    cudaSetDevice();
    CUDA_CHECK(cudaMemcpyToSymbol(ecDev, &ec, sizeof(EwaldConstants)));
}

CUDA_EXPORT int EwaldConst::nfour() {
    return NFOUR;
}

CUDA_EXPORT int EwaldConst::nreal() {
    return NREAL;
}

CUDA_EXPORT TensorSym<float, LORDER> const EwaldConst::d0() {
#ifdef __CUDA_ARCH__
    return ecDev.d0;
#else
    return ec.d0;
#endif
}

CUDA_EXPORT Array<float, NDIM> const& EwaldConst::realIndex(int i) {
#ifdef __CUDA_ARCH__
    return ecDev.realIndices[i];
#else
    return ec.realIndices[i];
#endif
}

CUDA_EXPORT Array<float, NDIM> const& EwaldConst::fourIndex(int i) {
#ifdef __CUDA_ARCH__
    return ecDev.fourIndices[i];
#else
    return ec.fourIndices[i];
#endif
}

CUDA_EXPORT TensorTrlessSym<float, LORDER> const& EwaldConst::fourExpansion(int i) {
#ifdef __CUDA_ARCH__
    return ecDev.fourExpanse[i];
#else
    return ec.fourExpanse[i];
#endif
}
