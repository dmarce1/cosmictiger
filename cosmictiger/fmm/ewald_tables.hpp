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

#pragma once

#include <cmath>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/fmm/ewald_indices.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/fmm/fmm_scale.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/tensor.hpp>

// Fills the order-dependent Ewald tables: one traceless Fourier derivative tensor
// per reciprocal image and the real-space self derivatives d0. The image lists
// (realIndices, fourIndices) must already be filled; they do not depend on the
// order. Rows are written into the kFmmMaxOrder-sized buffers with a zero tail,
// and the order is recorded so the accessors can be checked against it.
//
// The fill needs only tensor.hpp, so it stays visible to every includer: the
// standalone Ewald test links ewald_indices.cpp without the per-order instance
// translation units, which also instantiate it explicitly.
template <int order>
void ewaldFillTables(EwaldConstants& constants) {
    static_assert(order >= kFmmMinOrder && order <= kFmmMaxOrder, "Unsupported FMM order");
    using Traceless = SymmetricTracelessTensor<float, order>;
    using Symmetric = SymmetricTensor<float, order>;
    static_assert(Traceless::kN_ <= kFmmMaxTracelessTensorCount);
    static_assert(Symmetric::kN_ <= kFmmMaxSymmetricTensorCount);
    for (int i = 0; i < NFOUR; i++) {
        Array<float, NDIM> const h = constants.fourIndices[i];
        auto d0Value = vectorToSymTensor<float, order>(h);
        float const h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);    // 5 OP
        float const c0 = -1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) / M_PI;
        Array<int, NDIM> n;
        int const signs[4] = {1, -1, -1, 1};
        for (n[0] = 0; n[0] < order; n[0]++) {
            for (n[1] = 0; n[1] < order - n[0]; n[1]++) {
                for (n[2] = 0; n[2] < order - n[0] - n[1]; n[2]++) {
                    int const n0 = n[0] + n[1] + n[2];
                    d0Value(n) *= (signs[n0 % 4] * pow(2.0 * M_PI * SCALE_FACTOR_INV1, n0) * c0) *
                        SCALE_FACTOR_INV1;
                }
            }
        }
        Traceless const traceless = d0Value.detraceD();
        auto& row = constants.fourExpanse[i];
        for (int j = 0; j < Traceless::kN_; j++) {
            row[j] = traceless[j];
        }
        for (int j = Traceless::kN_; j < kFmmMaxTracelessTensorCount; j++) {
            row[j] = 0.0f;
        }
    }
    Symmetric d;
    for (int n = 0; n < Symmetric::kN_; n++) {
        d[n] = 0.0f;
    }
    constexpr double kAlpha = 2.0;
    for (int n = 0; n < order; n += 2) {
        for (int m = 0; m < order - n; m += 2) {
            for (int l = 0; l < order - n - m; l += 2) {
                d(n, m, l) = pow(SCALE_FACTOR_INV1, n + m + l + 1) *
                    pow(-2.0, (n + m + l) / 2 + 1) / ((n + m + l + 1.0) * sqrt(M_PI)) *
                    pow(kAlpha, n + m + l + 1) * doubleFactorial(n - 1) * doubleFactorial(m - 1) *
                    doubleFactorial(l - 1);
            }
        }
    }
    for (int n = 0; n < Symmetric::kN_; n++) {
        constants.d0[n] = d[n];
    }
    for (int n = Symmetric::kN_; n < kFmmMaxSymmetricTensorCount; n++) {
        constants.d0[n] = 0.0f;
    }
    constants.order = order;
}
