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
// Generated Ewald correction; STF translations come from tools/generate_stf_m2l.py.
#pragma once

#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/fmm_scale.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/simd_math.hpp>
#include <cosmictiger/tensor.hpp>

namespace detail {
template <class T>
CUDA_EXPORT int ewaldCorrectionDerivatives(
    SymmetricTracelessTensor<T, 7>& d, Array<T, NDIM> position) {
    position[0] *= simdBroadcast<T>(SCALE_FACTOR);
    position[1] *= simdBroadcast<T>(SCALE_FACTOR);
    position[2] *= simdBroadcast<T>(SCALE_FACTOR);
    EwaldConst econst;
    FlopCounter<int> flops = 7;
    T r = simdSqrt(
        simdFma(position[0], position[0], simdFma(position[1], position[1], sqr(position[2]))));
    T const fouroversqrtpi = simdBroadcast<T>(2.256758334e+00);
    SymmetricTensor<T, 7> dreal;
    SymmetricTracelessTensor<T, 7> dfour;
    dreal = 0.0f;
    dfour = 0.0f;
    d = 0.0f;
    auto const realsz = econst.nreal();
    auto const zeroMask = r > simdBroadcast<T>(0.0);
    int icnt = 0;
    Array<T, NDIM> dx;
    dx = position;
    {
        T r2 = simdFma(dx[0], dx[0], simdFma(dx[1], dx[1], sqr(dx[2])));
        icnt++;
        T const r = simdSqrt(r2);
        T const n8r = simdBroadcast<T>(-8 * SCALE_FACTOR_INV2) * r;
        T const rinv =
            simdSelect(r > simdBroadcast<T>(0), simdBroadcast<T>(1), simdBroadcast<T>(0)) /
            simdMax(r, simdBroadcast<T>(1.0e-20));
        T exp0 = expnearzero(-simdBroadcast<T>(4 * SCALE_FACTOR_INV2) * r2);
        T erf0 = erfnearzero(simdBroadcast<T>(2 * SCALE_FACTOR_INV1) * r);
        T const expfactor = simdBroadcast<T>(2.256758334191025 * SCALE_FACTOR_INV1) * exp0;
        T e0 = expfactor * rinv;
        T const rinv0 = simdBroadcast<T>(1);
        T const rinv1 = rinv;
        T const rinv2 = rinv1 * rinv1;
        T const rinv3 = rinv1 * rinv2;
        T const d0 = erf0 * rinv;
        T const d1 = simdFma(simdBroadcast<T>(-1) * d0, rinv, e0);
        e0 *= n8r;
        T const d2 = simdFma(simdBroadcast<T>(-3) * d1, rinv, e0);
        e0 *= n8r;
        T const d3 = simdFma(simdBroadcast<T>(-5) * d2, rinv, e0);
        e0 *= n8r;
        T const d4 = simdFma(simdBroadcast<T>(-7) * d3, rinv, e0);
        e0 *= n8r;
        T const d5 = simdFma(simdBroadcast<T>(-9) * d4, rinv, e0);
        e0 *= n8r;
        T const d6 = simdFma(simdBroadcast<T>(-11) * d5, rinv, e0);
        T const dRinvPower0Order0 = d0 * rinv0;
        T const dRinvPower1Order0 = d1 * rinv0;
        T const dRinvPower1Order1 = d1 * rinv1;
        T const dRinvPower2Order0 = d2 * rinv0;
        T const dRinvPower2Order1 = d2 * rinv1;
        T const dRinvPower2Order2 = d2 * rinv2;
        T const dRinvPower3Order0 = d3 * rinv0;
        T const dRinvPower3Order1 = d3 * rinv1;
        T const dRinvPower3Order2 = d3 * rinv2;
        T const dRinvPower3Order3 = d3 * rinv3;
        T const dRinvPower4Order0 = d4 * rinv0;
        T const dRinvPower4Order1 = d4 * rinv1;
        T const dRinvPower4Order2 = d4 * rinv2;
        T const dRinvPower5Order0 = d5 * rinv0;
        T const dRinvPower5Order1 = d5 * rinv1;
        T const dRinvPower6Order0 = d6 * rinv0;
        Array<T, NDIM> dxrinv;
        dxrinv[0] = dx[0] * rinv;
        dxrinv[1] = dx[1] * rinv;
        dxrinv[2] = dx[2] * rinv;
        T x[84];
        x[0] = simdBroadcast<T>(1);
        x[1] = dxrinv[0];
        x[2] = dxrinv[1];
        x[3] = dxrinv[2];
        x[9] = x[3] * x[3];       // 0 0 2 | 0 0 1 | 0 0 1
        x[8] = x[2] * x[3];       // 0 1 1 | 0 0 1 | 0 1 0
        x[7] = x[2] * x[2];       // 0 2 0 | 0 1 0 | 0 1 0
        x[6] = x[1] * x[3];       // 1 0 1 | 0 0 1 | 1 0 0
        x[5] = x[1] * x[2];       // 1 1 0 | 0 1 0 | 1 0 0
        x[4] = x[1] * x[1];       // 2 0 0 | 1 0 0 | 1 0 0
        x[19] = x[9] * x[3];      // 0 0 3 | 0 0 1 | 0 0 2
        x[18] = x[8] * x[3];      // 0 1 2 | 0 0 1 | 0 1 1
        x[17] = x[8] * x[2];      // 0 2 1 | 0 1 0 | 0 1 1
        x[16] = x[7] * x[2];      // 0 3 0 | 0 1 0 | 0 2 0
        x[15] = x[6] * x[3];      // 1 0 2 | 0 0 1 | 1 0 1
        x[14] = x[5] * x[3];      // 1 1 1 | 0 0 1 | 1 1 0
        x[13] = x[5] * x[2];      // 1 2 0 | 0 1 0 | 1 1 0
        x[12] = x[6] * x[1];      // 2 0 1 | 1 0 0 | 1 0 1
        x[11] = x[5] * x[1];      // 2 1 0 | 1 0 0 | 1 1 0
        x[10] = x[4] * x[1];      // 3 0 0 | 1 0 0 | 2 0 0
        x[34] = x[9] * x[9];      // 0 0 4 | 0 0 2 | 0 0 2
        x[33] = x[8] * x[9];      // 0 1 3 | 0 0 2 | 0 1 1
        x[32] = x[8] * x[8];      // 0 2 2 | 0 1 1 | 0 1 1
        x[31] = x[8] * x[7];      // 0 3 1 | 0 2 0 | 0 1 1
        x[30] = x[7] * x[7];      // 0 4 0 | 0 2 0 | 0 2 0
        x[29] = x[6] * x[9];      // 1 0 3 | 0 0 2 | 1 0 1
        x[28] = x[5] * x[9];      // 1 1 2 | 0 0 2 | 1 1 0
        x[27] = x[5] * x[8];      // 1 2 1 | 0 1 1 | 1 1 0
        x[26] = x[5] * x[7];      // 1 3 0 | 0 2 0 | 1 1 0
        x[25] = x[6] * x[6];      // 2 0 2 | 1 0 1 | 1 0 1
        x[24] = x[5] * x[6];      // 2 1 1 | 1 0 1 | 1 1 0
        x[23] = x[5] * x[5];      // 2 2 0 | 1 1 0 | 1 1 0
        x[22] = x[6] * x[4];      // 3 0 1 | 2 0 0 | 1 0 1
        x[21] = x[5] * x[4];      // 3 1 0 | 2 0 0 | 1 1 0
        x[20] = x[4] * x[4];      // 4 0 0 | 2 0 0 | 2 0 0
        x[55] = x[19] * x[9];     // 0 0 5 | 0 0 2 | 0 0 3
        x[54] = x[18] * x[9];     // 0 1 4 | 0 0 2 | 0 1 2
        x[53] = x[17] * x[9];     // 0 2 3 | 0 0 2 | 0 2 1
        x[52] = x[17] * x[8];     // 0 3 2 | 0 1 1 | 0 2 1
        x[51] = x[17] * x[7];     // 0 4 1 | 0 2 0 | 0 2 1
        x[50] = x[16] * x[7];     // 0 5 0 | 0 2 0 | 0 3 0
        x[49] = x[15] * x[9];     // 1 0 4 | 0 0 2 | 1 0 2
        x[48] = x[14] * x[9];     // 1 1 3 | 0 0 2 | 1 1 1
        x[47] = x[14] * x[8];     // 1 2 2 | 0 1 1 | 1 1 1
        x[46] = x[14] * x[7];     // 1 3 1 | 0 2 0 | 1 1 1
        x[45] = x[13] * x[7];     // 1 4 0 | 0 2 0 | 1 2 0
        x[44] = x[12] * x[9];     // 2 0 3 | 0 0 2 | 2 0 1
        x[43] = x[14] * x[6];     // 2 1 2 | 1 0 1 | 1 1 1
        x[42] = x[14] * x[5];     // 2 2 1 | 1 1 0 | 1 1 1
        x[41] = x[11] * x[7];     // 2 3 0 | 0 2 0 | 2 1 0
        x[40] = x[12] * x[6];     // 3 0 2 | 1 0 1 | 2 0 1
        x[39] = x[14] * x[4];     // 3 1 1 | 2 0 0 | 1 1 1
        x[38] = x[11] * x[5];     // 3 2 0 | 1 1 0 | 2 1 0
        x[37] = x[12] * x[4];     // 4 0 1 | 2 0 0 | 2 0 1
        x[36] = x[11] * x[4];     // 4 1 0 | 2 0 0 | 2 1 0
        x[35] = x[10] * x[4];     // 5 0 0 | 2 0 0 | 3 0 0
        x[83] = x[19] * x[19];    // 0 0 6 | 0 0 3 | 0 0 3
        x[82] = x[18] * x[19];    // 0 1 5 | 0 0 3 | 0 1 2
        x[81] = x[17] * x[19];    // 0 2 4 | 0 0 3 | 0 2 1
        x[80] = x[17] * x[18];    // 0 3 3 | 0 1 2 | 0 2 1
        x[79] = x[17] * x[17];    // 0 4 2 | 0 2 1 | 0 2 1
        x[78] = x[17] * x[16];    // 0 5 1 | 0 3 0 | 0 2 1
        x[77] = x[16] * x[16];    // 0 6 0 | 0 3 0 | 0 3 0
        x[76] = x[15] * x[19];    // 1 0 5 | 0 0 3 | 1 0 2
        x[75] = x[14] * x[19];    // 1 1 4 | 0 0 3 | 1 1 1
        x[74] = x[14] * x[18];    // 1 2 3 | 0 1 2 | 1 1 1
        x[73] = x[14] * x[17];    // 1 3 2 | 0 2 1 | 1 1 1
        x[72] = x[14] * x[16];    // 1 4 1 | 0 3 0 | 1 1 1
        x[71] = x[13] * x[16];    // 1 5 0 | 0 3 0 | 1 2 0
        x[70] = x[12] * x[19];    // 2 0 4 | 0 0 3 | 2 0 1
        x[69] = x[14] * x[15];    // 2 1 3 | 1 0 2 | 1 1 1
        x[68] = x[14] * x[14];    // 2 2 2 | 1 1 1 | 1 1 1
        x[67] = x[14] * x[13];    // 2 3 1 | 1 2 0 | 1 1 1
        x[66] = x[11] * x[16];    // 2 4 0 | 0 3 0 | 2 1 0
        x[65] = x[12] * x[15];    // 3 0 3 | 1 0 2 | 2 0 1
        x[64] = x[14] * x[12];    // 3 1 2 | 2 0 1 | 1 1 1
        x[63] = x[14] * x[11];    // 3 2 1 | 2 1 0 | 1 1 1
        x[62] = x[11] * x[13];    // 3 3 0 | 1 2 0 | 2 1 0
        x[61] = x[12] * x[12];    // 4 0 2 | 2 0 1 | 2 0 1
        x[60] = x[14] * x[10];    // 4 1 1 | 3 0 0 | 1 1 1
        x[59] = x[11] * x[11];    // 4 2 0 | 2 1 0 | 2 1 0
        x[58] = x[12] * x[10];    // 5 0 1 | 3 0 0 | 2 0 1
        x[57] = x[11] * x[10];    // 5 1 0 | 3 0 0 | 2 1 0
        x[56] = x[10] * x[10];    // 6 0 0 | 3 0 0 | 3 0 0
        dreal[0] = simdFma(x[0], dRinvPower0Order0, dreal[0]);
        dreal[1] = simdFma(x[1], dRinvPower1Order0, dreal[1]);
        dreal[2] = simdFma(x[2], dRinvPower1Order0, dreal[2]);
        dreal[3] = simdFma(x[3], dRinvPower1Order0, dreal[3]);
        dreal[4] = simdFma(x[0], dRinvPower1Order1, dreal[4]);
        dreal[4] = simdFma(x[4], dRinvPower2Order0, dreal[4]);
        dreal[5] = simdFma(x[5], dRinvPower2Order0, dreal[5]);
        dreal[6] = simdFma(x[6], dRinvPower2Order0, dreal[6]);
        dreal[7] = simdFma(x[0], dRinvPower1Order1, dreal[7]);
        dreal[7] = simdFma(x[7], dRinvPower2Order0, dreal[7]);
        dreal[8] = simdFma(x[8], dRinvPower2Order0, dreal[8]);
        dreal[9] = simdFma(x[0], dRinvPower1Order1, dreal[9]);
        dreal[9] = simdFma(x[9], dRinvPower2Order0, dreal[9]);
        dreal[10] = simdFma(x[10], dRinvPower3Order0, dreal[10]);
        dreal[10] = simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower2Order1, dreal[10]);
        dreal[11] = simdFma(x[2], dRinvPower2Order1, dreal[11]);
        dreal[11] = simdFma(x[11], dRinvPower3Order0, dreal[11]);
        dreal[12] = simdFma(x[3], dRinvPower2Order1, dreal[12]);
        dreal[12] = simdFma(x[12], dRinvPower3Order0, dreal[12]);
        dreal[13] = simdFma(x[1], dRinvPower2Order1, dreal[13]);
        dreal[13] = simdFma(x[13], dRinvPower3Order0, dreal[13]);
        dreal[14] = simdFma(x[14], dRinvPower3Order0, dreal[14]);
        dreal[15] = simdFma(x[15], dRinvPower3Order0, dreal[15]);
        dreal[15] = simdFma(x[1], dRinvPower2Order1, dreal[15]);
        dreal[16] = simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower2Order1, dreal[16]);
        dreal[16] = simdFma(x[16], dRinvPower3Order0, dreal[16]);
        dreal[17] = simdFma(x[3], dRinvPower2Order1, dreal[17]);
        dreal[17] = simdFma(x[17], dRinvPower3Order0, dreal[17]);
        dreal[18] = simdFma(x[18], dRinvPower3Order0, dreal[18]);
        dreal[18] = simdFma(x[2], dRinvPower2Order1, dreal[18]);
        dreal[19] = simdFma(x[19], dRinvPower3Order0, dreal[19]);
        dreal[19] = simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower2Order1, dreal[19]);
        dreal[20] = simdFma(x[20], dRinvPower4Order0, dreal[20]);
        dreal[20] = simdFma(simdBroadcast<T>(6.000000000e+00), x[4] * dRinvPower3Order1, dreal[20]);
        dreal[20] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower2Order2, dreal[20]);
        dreal[21] = simdFma(x[21], dRinvPower4Order0, dreal[21]);
        dreal[21] = simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower3Order1, dreal[21]);
        dreal[22] = simdFma(x[22], dRinvPower4Order0, dreal[22]);
        dreal[22] = simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower3Order1, dreal[22]);
        dreal[23] = simdFma(x[0], dRinvPower2Order2, dreal[23]);
        dreal[23] = simdFma(x[7], dRinvPower3Order1, dreal[23]);
        dreal[23] = simdFma(x[4], dRinvPower3Order1, dreal[23]);
        dreal[23] = simdFma(x[23], dRinvPower4Order0, dreal[23]);
        dreal[24] = simdFma(x[8], dRinvPower3Order1, dreal[24]);
        dreal[24] = simdFma(x[24], dRinvPower4Order0, dreal[24]);
        dreal[25] = simdFma(x[25], dRinvPower4Order0, dreal[25]);
        dreal[25] = simdFma(x[4], dRinvPower3Order1, dreal[25]);
        dreal[25] = simdFma(x[9], dRinvPower3Order1, dreal[25]);
        dreal[25] = simdFma(x[0], dRinvPower2Order2, dreal[25]);
        dreal[26] = simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower3Order1, dreal[26]);
        dreal[26] = simdFma(x[26], dRinvPower4Order0, dreal[26]);
        dreal[27] = simdFma(x[27], dRinvPower4Order0, dreal[27]);
        dreal[27] = simdFma(x[6], dRinvPower3Order1, dreal[27]);
        dreal[28] = simdFma(x[28], dRinvPower4Order0, dreal[28]);
        dreal[28] = simdFma(x[5], dRinvPower3Order1, dreal[28]);
        dreal[29] = simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower3Order1, dreal[29]);
        dreal[29] = simdFma(x[29], dRinvPower4Order0, dreal[29]);
        dreal[30] = simdFma(x[30], dRinvPower4Order0, dreal[30]);
        dreal[30] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower2Order2, dreal[30]);
        dreal[30] = simdFma(simdBroadcast<T>(6.000000000e+00), x[7] * dRinvPower3Order1, dreal[30]);
        dreal[31] = simdFma(x[31], dRinvPower4Order0, dreal[31]);
        dreal[31] = simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower3Order1, dreal[31]);
        dreal[32] = simdFma(x[0], dRinvPower2Order2, dreal[32]);
        dreal[32] = simdFma(x[9], dRinvPower3Order1, dreal[32]);
        dreal[32] = simdFma(x[7], dRinvPower3Order1, dreal[32]);
        dreal[32] = simdFma(x[32], dRinvPower4Order0, dreal[32]);
        dreal[33] = simdFma(x[33], dRinvPower4Order0, dreal[33]);
        dreal[33] = simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower3Order1, dreal[33]);
        dreal[34] = simdFma(simdBroadcast<T>(6.000000000e+00), x[9] * dRinvPower3Order1, dreal[34]);
        dreal[34] = simdFma(x[34], dRinvPower4Order0, dreal[34]);
        dreal[34] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower2Order2, dreal[34]);
        dreal[35] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[10] * dRinvPower4Order1, dreal[35]);
        dreal[35] = simdFma(simdBroadcast<T>(1.500000000e+01), x[1] * dRinvPower3Order2, dreal[35]);
        dreal[35] = simdFma(x[35], dRinvPower5Order0, dreal[35]);
        dreal[36] = simdFma(x[36], dRinvPower5Order0, dreal[36]);
        dreal[36] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[11] * dRinvPower4Order1, dreal[36]);
        dreal[36] = simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower3Order2, dreal[36]);
        dreal[37] = simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower3Order2, dreal[37]);
        dreal[37] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[12] * dRinvPower4Order1, dreal[37]);
        dreal[37] = simdFma(x[37], dRinvPower5Order0, dreal[37]);
        dreal[38] = simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower3Order2, dreal[38]);
        dreal[38] = simdFma(x[38], dRinvPower5Order0, dreal[38]);
        dreal[38] = simdFma(x[10], dRinvPower4Order1, dreal[38]);
        dreal[38] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[13] * dRinvPower4Order1, dreal[38]);
        dreal[39] = simdFma(x[39], dRinvPower5Order0, dreal[39]);
        dreal[39] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[14] * dRinvPower4Order1, dreal[39]);
        dreal[40] = simdFma(x[40], dRinvPower5Order0, dreal[40]);
        dreal[40] = simdFma(x[10], dRinvPower4Order1, dreal[40]);
        dreal[40] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[15] * dRinvPower4Order1, dreal[40]);
        dreal[40] = simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower3Order2, dreal[40]);
        dreal[41] = simdFma(x[41], dRinvPower5Order0, dreal[41]);
        dreal[41] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[11] * dRinvPower4Order1, dreal[41]);
        dreal[41] = simdFma(x[16], dRinvPower4Order1, dreal[41]);
        dreal[41] = simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower3Order2, dreal[41]);
        dreal[42] = simdFma(x[12], dRinvPower4Order1, dreal[42]);
        dreal[42] = simdFma(x[3], dRinvPower3Order2, dreal[42]);
        dreal[42] = simdFma(x[17], dRinvPower4Order1, dreal[42]);
        dreal[42] = simdFma(x[42], dRinvPower5Order0, dreal[42]);
        dreal[43] = simdFma(x[11], dRinvPower4Order1, dreal[43]);
        dreal[43] = simdFma(x[43], dRinvPower5Order0, dreal[43]);
        dreal[43] = simdFma(x[2], dRinvPower3Order2, dreal[43]);
        dreal[43] = simdFma(x[18], dRinvPower4Order1, dreal[43]);
        dreal[44] = simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower3Order2, dreal[44]);
        dreal[44] = simdFma(x[44], dRinvPower5Order0, dreal[44]);
        dreal[44] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[12] * dRinvPower4Order1, dreal[44]);
        dreal[44] = simdFma(x[19], dRinvPower4Order1, dreal[44]);
        dreal[45] = simdFma(x[45], dRinvPower5Order0, dreal[45]);
        dreal[45] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[13] * dRinvPower4Order1, dreal[45]);
        dreal[45] = simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower3Order2, dreal[45]);
        dreal[46] = simdFma(x[46], dRinvPower5Order0, dreal[46]);
        dreal[46] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[14] * dRinvPower4Order1, dreal[46]);
        dreal[47] = simdFma(x[47], dRinvPower5Order0, dreal[47]);
        dreal[47] = simdFma(x[13], dRinvPower4Order1, dreal[47]);
        dreal[47] = simdFma(x[15], dRinvPower4Order1, dreal[47]);
        dreal[47] = simdFma(x[1], dRinvPower3Order2, dreal[47]);
        dreal[48] = simdFma(x[48], dRinvPower5Order0, dreal[48]);
        dreal[48] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[14] * dRinvPower4Order1, dreal[48]);
        dreal[49] = simdFma(x[49], dRinvPower5Order0, dreal[49]);
        dreal[49] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[15] * dRinvPower4Order1, dreal[49]);
        dreal[49] = simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower3Order2, dreal[49]);
        dreal[50] = simdFma(simdBroadcast<T>(1.500000000e+01), x[2] * dRinvPower3Order2, dreal[50]);
        dreal[50] = simdFma(x[50], dRinvPower5Order0, dreal[50]);
        dreal[50] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[16] * dRinvPower4Order1, dreal[50]);
        dreal[51] = simdFma(x[51], dRinvPower5Order0, dreal[51]);
        dreal[51] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[17] * dRinvPower4Order1, dreal[51]);
        dreal[51] = simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower3Order2, dreal[51]);
        dreal[52] = simdFma(x[16], dRinvPower4Order1, dreal[52]);
        dreal[52] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[18] * dRinvPower4Order1, dreal[52]);
        dreal[52] = simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower3Order2, dreal[52]);
        dreal[52] = simdFma(x[52], dRinvPower5Order0, dreal[52]);
        dreal[53] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[17] * dRinvPower4Order1, dreal[53]);
        dreal[53] = simdFma(x[53], dRinvPower5Order0, dreal[53]);
        dreal[53] = simdFma(x[19], dRinvPower4Order1, dreal[53]);
        dreal[53] = simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower3Order2, dreal[53]);
        dreal[54] = simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower3Order2, dreal[54]);
        dreal[54] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[18] * dRinvPower4Order1, dreal[54]);
        dreal[54] = simdFma(x[54], dRinvPower5Order0, dreal[54]);
        dreal[55] = simdFma(simdBroadcast<T>(1.500000000e+01), x[3] * dRinvPower3Order2, dreal[55]);
        dreal[55] = simdFma(x[55], dRinvPower5Order0, dreal[55]);
        dreal[55] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[19] * dRinvPower4Order1, dreal[55]);
        dreal[56] = simdFma(x[56], dRinvPower6Order0, dreal[56]);
        dreal[56] = simdFma(simdBroadcast<T>(4.500000000e+01), x[4] * dRinvPower4Order2, dreal[56]);
        dreal[56] =
            simdFma(simdBroadcast<T>(1.500000000e+01), x[20] * dRinvPower5Order1, dreal[56]);
        dreal[56] = simdFma(simdBroadcast<T>(1.500000000e+01), x[0] * dRinvPower3Order3, dreal[56]);
        dreal[57] = simdFma(simdBroadcast<T>(1.500000000e+01), x[5] * dRinvPower4Order2, dreal[57]);
        dreal[57] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[21] * dRinvPower5Order1, dreal[57]);
        dreal[57] = simdFma(x[57], dRinvPower6Order0, dreal[57]);
        dreal[58] = simdFma(simdBroadcast<T>(1.500000000e+01), x[6] * dRinvPower4Order2, dreal[58]);
        dreal[58] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[22] * dRinvPower5Order1, dreal[58]);
        dreal[58] = simdFma(x[58], dRinvPower6Order0, dreal[58]);
        dreal[59] = simdFma(x[59], dRinvPower6Order0, dreal[59]);
        dreal[59] = simdFma(x[20], dRinvPower5Order1, dreal[59]);
        dreal[59] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[23] * dRinvPower5Order1, dreal[59]);
        dreal[59] = simdFma(simdBroadcast<T>(3.000000000e+00), x[7] * dRinvPower4Order2, dreal[59]);
        dreal[59] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[59]);
        dreal[59] = simdFma(simdBroadcast<T>(6.000000000e+00), x[4] * dRinvPower4Order2, dreal[59]);
        dreal[60] = simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower4Order2, dreal[60]);
        dreal[60] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[24] * dRinvPower5Order1, dreal[60]);
        dreal[60] = simdFma(x[60], dRinvPower6Order0, dreal[60]);
        dreal[61] = simdFma(x[20], dRinvPower5Order1, dreal[61]);
        dreal[61] = simdFma(x[61], dRinvPower6Order0, dreal[61]);
        dreal[61] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[25] * dRinvPower5Order1, dreal[61]);
        dreal[61] = simdFma(simdBroadcast<T>(6.000000000e+00), x[4] * dRinvPower4Order2, dreal[61]);
        dreal[61] = simdFma(simdBroadcast<T>(3.000000000e+00), x[9] * dRinvPower4Order2, dreal[61]);
        dreal[61] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[61]);
        dreal[62] = simdFma(x[62], dRinvPower6Order0, dreal[62]);
        dreal[62] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[21] * dRinvPower5Order1, dreal[62]);
        dreal[62] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[26] * dRinvPower5Order1, dreal[62]);
        dreal[62] = simdFma(simdBroadcast<T>(9.000000000e+00), x[5] * dRinvPower4Order2, dreal[62]);
        dreal[63] = simdFma(x[63], dRinvPower6Order0, dreal[63]);
        dreal[63] = simdFma(x[22], dRinvPower5Order1, dreal[63]);
        dreal[63] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[27] * dRinvPower5Order1, dreal[63]);
        dreal[63] = simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower4Order2, dreal[63]);
        dreal[64] = simdFma(x[64], dRinvPower6Order0, dreal[64]);
        dreal[64] = simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower4Order2, dreal[64]);
        dreal[64] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[28] * dRinvPower5Order1, dreal[64]);
        dreal[64] = simdFma(x[21], dRinvPower5Order1, dreal[64]);
        dreal[65] = simdFma(simdBroadcast<T>(9.000000000e+00), x[6] * dRinvPower4Order2, dreal[65]);
        dreal[65] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[29] * dRinvPower5Order1, dreal[65]);
        dreal[65] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[22] * dRinvPower5Order1, dreal[65]);
        dreal[65] = simdFma(x[65], dRinvPower6Order0, dreal[65]);
        dreal[66] = simdFma(x[66], dRinvPower6Order0, dreal[66]);
        dreal[66] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[23] * dRinvPower5Order1, dreal[66]);
        dreal[66] = simdFma(simdBroadcast<T>(3.000000000e+00), x[4] * dRinvPower4Order2, dreal[66]);
        dreal[66] = simdFma(x[30], dRinvPower5Order1, dreal[66]);
        dreal[66] = simdFma(simdBroadcast<T>(6.000000000e+00), x[7] * dRinvPower4Order2, dreal[66]);
        dreal[66] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[66]);
        dreal[67] = simdFma(x[67], dRinvPower6Order0, dreal[67]);
        dreal[67] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[24] * dRinvPower5Order1, dreal[67]);
        dreal[67] = simdFma(x[31], dRinvPower5Order1, dreal[67]);
        dreal[67] = simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower4Order2, dreal[67]);
        dreal[68] = simdFma(x[9], dRinvPower4Order2, dreal[68]);
        dreal[68] = simdFma(x[68], dRinvPower6Order0, dreal[68]);
        dreal[68] = simdFma(x[23], dRinvPower5Order1, dreal[68]);
        dreal[68] = simdFma(x[25], dRinvPower5Order1, dreal[68]);
        dreal[68] = simdFma(x[4], dRinvPower4Order2, dreal[68]);
        dreal[68] = simdFma(x[32], dRinvPower5Order1, dreal[68]);
        dreal[68] = simdFma(x[7], dRinvPower4Order2, dreal[68]);
        dreal[68] = simdFma(x[0], dRinvPower3Order3, dreal[68]);
        dreal[69] = simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower4Order2, dreal[69]);
        dreal[69] = simdFma(x[33], dRinvPower5Order1, dreal[69]);
        dreal[69] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[24] * dRinvPower5Order1, dreal[69]);
        dreal[69] = simdFma(x[69], dRinvPower6Order0, dreal[69]);
        dreal[70] = simdFma(x[34], dRinvPower5Order1, dreal[70]);
        dreal[70] = simdFma(simdBroadcast<T>(3.000000000e+00), x[4] * dRinvPower4Order2, dreal[70]);
        dreal[70] = simdFma(simdBroadcast<T>(6.000000000e+00), x[9] * dRinvPower4Order2, dreal[70]);
        dreal[70] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[70]);
        dreal[70] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[25] * dRinvPower5Order1, dreal[70]);
        dreal[70] = simdFma(x[70], dRinvPower6Order0, dreal[70]);
        dreal[71] = simdFma(simdBroadcast<T>(1.500000000e+01), x[5] * dRinvPower4Order2, dreal[71]);
        dreal[71] = simdFma(x[71], dRinvPower6Order0, dreal[71]);
        dreal[71] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[26] * dRinvPower5Order1, dreal[71]);
        dreal[72] = simdFma(x[72], dRinvPower6Order0, dreal[72]);
        dreal[72] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[27] * dRinvPower5Order1, dreal[72]);
        dreal[72] = simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower4Order2, dreal[72]);
        dreal[73] = simdFma(x[73], dRinvPower6Order0, dreal[73]);
        dreal[73] = simdFma(x[26], dRinvPower5Order1, dreal[73]);
        dreal[73] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[28] * dRinvPower5Order1, dreal[73]);
        dreal[73] = simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower4Order2, dreal[73]);
        dreal[74] = simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower4Order2, dreal[74]);
        dreal[74] = simdFma(x[29], dRinvPower5Order1, dreal[74]);
        dreal[74] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[27] * dRinvPower5Order1, dreal[74]);
        dreal[74] = simdFma(x[74], dRinvPower6Order0, dreal[74]);
        dreal[75] = simdFma(x[75], dRinvPower6Order0, dreal[75]);
        dreal[75] = simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower4Order2, dreal[75]);
        dreal[75] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[28] * dRinvPower5Order1, dreal[75]);
        dreal[76] = simdFma(simdBroadcast<T>(1.500000000e+01), x[6] * dRinvPower4Order2, dreal[76]);
        dreal[76] = simdFma(x[76], dRinvPower6Order0, dreal[76]);
        dreal[76] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[29] * dRinvPower5Order1, dreal[76]);
        dreal[77] = simdFma(x[77], dRinvPower6Order0, dreal[77]);
        dreal[77] =
            simdFma(simdBroadcast<T>(1.500000000e+01), x[30] * dRinvPower5Order1, dreal[77]);
        dreal[77] = simdFma(simdBroadcast<T>(4.500000000e+01), x[7] * dRinvPower4Order2, dreal[77]);
        dreal[77] = simdFma(simdBroadcast<T>(1.500000000e+01), x[0] * dRinvPower3Order3, dreal[77]);
        dreal[78] = simdFma(simdBroadcast<T>(1.500000000e+01), x[8] * dRinvPower4Order2, dreal[78]);
        dreal[78] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[31] * dRinvPower5Order1, dreal[78]);
        dreal[78] = simdFma(x[78], dRinvPower6Order0, dreal[78]);
        dreal[79] = simdFma(simdBroadcast<T>(6.000000000e+00), x[7] * dRinvPower4Order2, dreal[79]);
        dreal[79] = simdFma(simdBroadcast<T>(3.000000000e+00), x[9] * dRinvPower4Order2, dreal[79]);
        dreal[79] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[79]);
        dreal[79] = simdFma(x[79], dRinvPower6Order0, dreal[79]);
        dreal[79] = simdFma(x[30], dRinvPower5Order1, dreal[79]);
        dreal[79] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[32] * dRinvPower5Order1, dreal[79]);
        dreal[80] = simdFma(x[80], dRinvPower6Order0, dreal[80]);
        dreal[80] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[31] * dRinvPower5Order1, dreal[80]);
        dreal[80] =
            simdFma(simdBroadcast<T>(3.000000000e+00), x[33] * dRinvPower5Order1, dreal[80]);
        dreal[80] = simdFma(simdBroadcast<T>(9.000000000e+00), x[8] * dRinvPower4Order2, dreal[80]);
        dreal[81] =
            simdFma(simdBroadcast<T>(6.000000000e+00), x[32] * dRinvPower5Order1, dreal[81]);
        dreal[81] = simdFma(simdBroadcast<T>(3.000000000e+00), x[7] * dRinvPower4Order2, dreal[81]);
        dreal[81] = simdFma(x[34], dRinvPower5Order1, dreal[81]);
        dreal[81] = simdFma(simdBroadcast<T>(6.000000000e+00), x[9] * dRinvPower4Order2, dreal[81]);
        dreal[81] = simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[81]);
        dreal[81] = simdFma(x[81], dRinvPower6Order0, dreal[81]);
        dreal[82] = simdFma(x[82], dRinvPower6Order0, dreal[82]);
        dreal[82] =
            simdFma(simdBroadcast<T>(1.000000000e+01), x[33] * dRinvPower5Order1, dreal[82]);
        dreal[82] = simdFma(simdBroadcast<T>(1.500000000e+01), x[8] * dRinvPower4Order2, dreal[82]);
        dreal[83] = simdFma(simdBroadcast<T>(1.500000000e+01), x[0] * dRinvPower3Order3, dreal[83]);
        dreal[83] = simdFma(simdBroadcast<T>(4.500000000e+01), x[9] * dRinvPower4Order2, dreal[83]);
        dreal[83] =
            simdFma(simdBroadcast<T>(1.500000000e+01), x[34] * dRinvPower5Order1, dreal[83]);
        dreal[83] = simdFma(x[83], dRinvPower6Order0, dreal[83]);
        auto const drz = econst.d0();
        for (int i = 0; i < 84; i++) {
            dreal[i] = simdSelect(zeroMask, dreal[i], -simdBroadcast<T>(drz[i]));
        }
    }
    flops += icnt * 843;
    for (int i = 0; i < realsz; i++) {
        auto const n = econst.realIndex(i);
        Array<T, NDIM> dx;
        for (int dim = 0; dim < NDIM; dim++) {
            dx[dim] = position[dim] - simdBroadcast<T>(SCALE_FACTOR) * n[dim];
        }
        T r2 = simdFma(dx[0], dx[0], simdFma(dx[1], dx[1], sqr(dx[2])));
        if (simdAnyOf(r2 < simdBroadcast<T>(SCALE_FACTOR2 * EWALD_REAL_CUTOFF2))) {
            icnt++;
            T const r = simdSqrt(r2);
            T const n8r = simdBroadcast<T>(-8 * SCALE_FACTOR_INV2) * r;
            T const rinv =
                simdSelect(r > simdBroadcast<T>(0), simdBroadcast<T>(1), simdBroadcast<T>(0)) /
                simdMax(r, simdBroadcast<T>(1.0e-20));
            T exp0;
            T erfc0;
            erfcexp(simdBroadcast<T>(2. * SCALE_FACTOR_INV1) * r, &erfc0, &exp0);
            T const expfactor = fouroversqrtpi * simdBroadcast<T>(SCALE_FACTOR_INV1) * exp0;
            T e0 = expfactor * rinv;
            T const rinv0 = simdBroadcast<T>(1);
            T const rinv1 = rinv;
            T const rinv2 = rinv1 * rinv1;
            T const rinv3 = rinv1 * rinv2;
            T const d0 = -erfc0 * rinv;
            T const d1 = simdFma(simdBroadcast<T>(-1) * d0, rinv, e0);
            e0 *= n8r;
            T const d2 = simdFma(simdBroadcast<T>(-3) * d1, rinv, e0);
            e0 *= n8r;
            T const d3 = simdFma(simdBroadcast<T>(-5) * d2, rinv, e0);
            e0 *= n8r;
            T const d4 = simdFma(simdBroadcast<T>(-7) * d3, rinv, e0);
            e0 *= n8r;
            T const d5 = simdFma(simdBroadcast<T>(-9) * d4, rinv, e0);
            e0 *= n8r;
            T const d6 = simdFma(simdBroadcast<T>(-11) * d5, rinv, e0);
            T const dRinvPower0Order0 = d0 * rinv0;
            T const dRinvPower1Order0 = d1 * rinv0;
            T const dRinvPower1Order1 = d1 * rinv1;
            T const dRinvPower2Order0 = d2 * rinv0;
            T const dRinvPower2Order1 = d2 * rinv1;
            T const dRinvPower2Order2 = d2 * rinv2;
            T const dRinvPower3Order0 = d3 * rinv0;
            T const dRinvPower3Order1 = d3 * rinv1;
            T const dRinvPower3Order2 = d3 * rinv2;
            T const dRinvPower3Order3 = d3 * rinv3;
            T const dRinvPower4Order0 = d4 * rinv0;
            T const dRinvPower4Order1 = d4 * rinv1;
            T const dRinvPower4Order2 = d4 * rinv2;
            T const dRinvPower5Order0 = d5 * rinv0;
            T const dRinvPower5Order1 = d5 * rinv1;
            T const dRinvPower6Order0 = d6 * rinv0;
            Array<T, NDIM> dxrinv;
            dxrinv[0] = dx[0] * rinv;
            dxrinv[1] = dx[1] * rinv;
            dxrinv[2] = dx[2] * rinv;
            T x[84];
            x[0] = simdBroadcast<T>(1);
            x[1] = dxrinv[0];
            x[2] = dxrinv[1];
            x[3] = dxrinv[2];
            x[9] = x[3] * x[3];       // 0 0 2 | 0 0 1 | 0 0 1
            x[8] = x[2] * x[3];       // 0 1 1 | 0 0 1 | 0 1 0
            x[7] = x[2] * x[2];       // 0 2 0 | 0 1 0 | 0 1 0
            x[6] = x[1] * x[3];       // 1 0 1 | 0 0 1 | 1 0 0
            x[5] = x[1] * x[2];       // 1 1 0 | 0 1 0 | 1 0 0
            x[4] = x[1] * x[1];       // 2 0 0 | 1 0 0 | 1 0 0
            x[19] = x[9] * x[3];      // 0 0 3 | 0 0 1 | 0 0 2
            x[18] = x[8] * x[3];      // 0 1 2 | 0 0 1 | 0 1 1
            x[17] = x[8] * x[2];      // 0 2 1 | 0 1 0 | 0 1 1
            x[16] = x[7] * x[2];      // 0 3 0 | 0 1 0 | 0 2 0
            x[15] = x[6] * x[3];      // 1 0 2 | 0 0 1 | 1 0 1
            x[14] = x[5] * x[3];      // 1 1 1 | 0 0 1 | 1 1 0
            x[13] = x[5] * x[2];      // 1 2 0 | 0 1 0 | 1 1 0
            x[12] = x[6] * x[1];      // 2 0 1 | 1 0 0 | 1 0 1
            x[11] = x[5] * x[1];      // 2 1 0 | 1 0 0 | 1 1 0
            x[10] = x[4] * x[1];      // 3 0 0 | 1 0 0 | 2 0 0
            x[34] = x[9] * x[9];      // 0 0 4 | 0 0 2 | 0 0 2
            x[33] = x[8] * x[9];      // 0 1 3 | 0 0 2 | 0 1 1
            x[32] = x[8] * x[8];      // 0 2 2 | 0 1 1 | 0 1 1
            x[31] = x[8] * x[7];      // 0 3 1 | 0 2 0 | 0 1 1
            x[30] = x[7] * x[7];      // 0 4 0 | 0 2 0 | 0 2 0
            x[29] = x[6] * x[9];      // 1 0 3 | 0 0 2 | 1 0 1
            x[28] = x[5] * x[9];      // 1 1 2 | 0 0 2 | 1 1 0
            x[27] = x[5] * x[8];      // 1 2 1 | 0 1 1 | 1 1 0
            x[26] = x[5] * x[7];      // 1 3 0 | 0 2 0 | 1 1 0
            x[25] = x[6] * x[6];      // 2 0 2 | 1 0 1 | 1 0 1
            x[24] = x[5] * x[6];      // 2 1 1 | 1 0 1 | 1 1 0
            x[23] = x[5] * x[5];      // 2 2 0 | 1 1 0 | 1 1 0
            x[22] = x[6] * x[4];      // 3 0 1 | 2 0 0 | 1 0 1
            x[21] = x[5] * x[4];      // 3 1 0 | 2 0 0 | 1 1 0
            x[20] = x[4] * x[4];      // 4 0 0 | 2 0 0 | 2 0 0
            x[55] = x[19] * x[9];     // 0 0 5 | 0 0 2 | 0 0 3
            x[54] = x[18] * x[9];     // 0 1 4 | 0 0 2 | 0 1 2
            x[53] = x[17] * x[9];     // 0 2 3 | 0 0 2 | 0 2 1
            x[52] = x[17] * x[8];     // 0 3 2 | 0 1 1 | 0 2 1
            x[51] = x[17] * x[7];     // 0 4 1 | 0 2 0 | 0 2 1
            x[50] = x[16] * x[7];     // 0 5 0 | 0 2 0 | 0 3 0
            x[49] = x[15] * x[9];     // 1 0 4 | 0 0 2 | 1 0 2
            x[48] = x[14] * x[9];     // 1 1 3 | 0 0 2 | 1 1 1
            x[47] = x[14] * x[8];     // 1 2 2 | 0 1 1 | 1 1 1
            x[46] = x[14] * x[7];     // 1 3 1 | 0 2 0 | 1 1 1
            x[45] = x[13] * x[7];     // 1 4 0 | 0 2 0 | 1 2 0
            x[44] = x[12] * x[9];     // 2 0 3 | 0 0 2 | 2 0 1
            x[43] = x[14] * x[6];     // 2 1 2 | 1 0 1 | 1 1 1
            x[42] = x[14] * x[5];     // 2 2 1 | 1 1 0 | 1 1 1
            x[41] = x[11] * x[7];     // 2 3 0 | 0 2 0 | 2 1 0
            x[40] = x[12] * x[6];     // 3 0 2 | 1 0 1 | 2 0 1
            x[39] = x[14] * x[4];     // 3 1 1 | 2 0 0 | 1 1 1
            x[38] = x[11] * x[5];     // 3 2 0 | 1 1 0 | 2 1 0
            x[37] = x[12] * x[4];     // 4 0 1 | 2 0 0 | 2 0 1
            x[36] = x[11] * x[4];     // 4 1 0 | 2 0 0 | 2 1 0
            x[35] = x[10] * x[4];     // 5 0 0 | 2 0 0 | 3 0 0
            x[83] = x[19] * x[19];    // 0 0 6 | 0 0 3 | 0 0 3
            x[82] = x[18] * x[19];    // 0 1 5 | 0 0 3 | 0 1 2
            x[81] = x[17] * x[19];    // 0 2 4 | 0 0 3 | 0 2 1
            x[80] = x[17] * x[18];    // 0 3 3 | 0 1 2 | 0 2 1
            x[79] = x[17] * x[17];    // 0 4 2 | 0 2 1 | 0 2 1
            x[78] = x[17] * x[16];    // 0 5 1 | 0 3 0 | 0 2 1
            x[77] = x[16] * x[16];    // 0 6 0 | 0 3 0 | 0 3 0
            x[76] = x[15] * x[19];    // 1 0 5 | 0 0 3 | 1 0 2
            x[75] = x[14] * x[19];    // 1 1 4 | 0 0 3 | 1 1 1
            x[74] = x[14] * x[18];    // 1 2 3 | 0 1 2 | 1 1 1
            x[73] = x[14] * x[17];    // 1 3 2 | 0 2 1 | 1 1 1
            x[72] = x[14] * x[16];    // 1 4 1 | 0 3 0 | 1 1 1
            x[71] = x[13] * x[16];    // 1 5 0 | 0 3 0 | 1 2 0
            x[70] = x[12] * x[19];    // 2 0 4 | 0 0 3 | 2 0 1
            x[69] = x[14] * x[15];    // 2 1 3 | 1 0 2 | 1 1 1
            x[68] = x[14] * x[14];    // 2 2 2 | 1 1 1 | 1 1 1
            x[67] = x[14] * x[13];    // 2 3 1 | 1 2 0 | 1 1 1
            x[66] = x[11] * x[16];    // 2 4 0 | 0 3 0 | 2 1 0
            x[65] = x[12] * x[15];    // 3 0 3 | 1 0 2 | 2 0 1
            x[64] = x[14] * x[12];    // 3 1 2 | 2 0 1 | 1 1 1
            x[63] = x[14] * x[11];    // 3 2 1 | 2 1 0 | 1 1 1
            x[62] = x[11] * x[13];    // 3 3 0 | 1 2 0 | 2 1 0
            x[61] = x[12] * x[12];    // 4 0 2 | 2 0 1 | 2 0 1
            x[60] = x[14] * x[10];    // 4 1 1 | 3 0 0 | 1 1 1
            x[59] = x[11] * x[11];    // 4 2 0 | 2 1 0 | 2 1 0
            x[58] = x[12] * x[10];    // 5 0 1 | 3 0 0 | 2 0 1
            x[57] = x[11] * x[10];    // 5 1 0 | 3 0 0 | 2 1 0
            x[56] = x[10] * x[10];    // 6 0 0 | 3 0 0 | 3 0 0
            dreal[0] = simdFma(x[0], dRinvPower0Order0, dreal[0]);
            dreal[1] = simdFma(x[1], dRinvPower1Order0, dreal[1]);
            dreal[2] = simdFma(x[2], dRinvPower1Order0, dreal[2]);
            dreal[3] = simdFma(x[3], dRinvPower1Order0, dreal[3]);
            dreal[4] = simdFma(x[0], dRinvPower1Order1, dreal[4]);
            dreal[4] = simdFma(x[4], dRinvPower2Order0, dreal[4]);
            dreal[5] = simdFma(x[5], dRinvPower2Order0, dreal[5]);
            dreal[6] = simdFma(x[6], dRinvPower2Order0, dreal[6]);
            dreal[7] = simdFma(x[0], dRinvPower1Order1, dreal[7]);
            dreal[7] = simdFma(x[7], dRinvPower2Order0, dreal[7]);
            dreal[8] = simdFma(x[8], dRinvPower2Order0, dreal[8]);
            dreal[9] = simdFma(x[0], dRinvPower1Order1, dreal[9]);
            dreal[9] = simdFma(x[9], dRinvPower2Order0, dreal[9]);
            dreal[10] = simdFma(x[10], dRinvPower3Order0, dreal[10]);
            dreal[10] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower2Order1, dreal[10]);
            dreal[11] = simdFma(x[2], dRinvPower2Order1, dreal[11]);
            dreal[11] = simdFma(x[11], dRinvPower3Order0, dreal[11]);
            dreal[12] = simdFma(x[3], dRinvPower2Order1, dreal[12]);
            dreal[12] = simdFma(x[12], dRinvPower3Order0, dreal[12]);
            dreal[13] = simdFma(x[1], dRinvPower2Order1, dreal[13]);
            dreal[13] = simdFma(x[13], dRinvPower3Order0, dreal[13]);
            dreal[14] = simdFma(x[14], dRinvPower3Order0, dreal[14]);
            dreal[15] = simdFma(x[15], dRinvPower3Order0, dreal[15]);
            dreal[15] = simdFma(x[1], dRinvPower2Order1, dreal[15]);
            dreal[16] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower2Order1, dreal[16]);
            dreal[16] = simdFma(x[16], dRinvPower3Order0, dreal[16]);
            dreal[17] = simdFma(x[3], dRinvPower2Order1, dreal[17]);
            dreal[17] = simdFma(x[17], dRinvPower3Order0, dreal[17]);
            dreal[18] = simdFma(x[18], dRinvPower3Order0, dreal[18]);
            dreal[18] = simdFma(x[2], dRinvPower2Order1, dreal[18]);
            dreal[19] = simdFma(x[19], dRinvPower3Order0, dreal[19]);
            dreal[19] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower2Order1, dreal[19]);
            dreal[20] = simdFma(x[20], dRinvPower4Order0, dreal[20]);
            dreal[20] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[4] * dRinvPower3Order1, dreal[20]);
            dreal[20] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower2Order2, dreal[20]);
            dreal[21] = simdFma(x[21], dRinvPower4Order0, dreal[21]);
            dreal[21] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower3Order1, dreal[21]);
            dreal[22] = simdFma(x[22], dRinvPower4Order0, dreal[22]);
            dreal[22] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower3Order1, dreal[22]);
            dreal[23] = simdFma(x[0], dRinvPower2Order2, dreal[23]);
            dreal[23] = simdFma(x[7], dRinvPower3Order1, dreal[23]);
            dreal[23] = simdFma(x[4], dRinvPower3Order1, dreal[23]);
            dreal[23] = simdFma(x[23], dRinvPower4Order0, dreal[23]);
            dreal[24] = simdFma(x[8], dRinvPower3Order1, dreal[24]);
            dreal[24] = simdFma(x[24], dRinvPower4Order0, dreal[24]);
            dreal[25] = simdFma(x[25], dRinvPower4Order0, dreal[25]);
            dreal[25] = simdFma(x[4], dRinvPower3Order1, dreal[25]);
            dreal[25] = simdFma(x[9], dRinvPower3Order1, dreal[25]);
            dreal[25] = simdFma(x[0], dRinvPower2Order2, dreal[25]);
            dreal[26] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower3Order1, dreal[26]);
            dreal[26] = simdFma(x[26], dRinvPower4Order0, dreal[26]);
            dreal[27] = simdFma(x[27], dRinvPower4Order0, dreal[27]);
            dreal[27] = simdFma(x[6], dRinvPower3Order1, dreal[27]);
            dreal[28] = simdFma(x[28], dRinvPower4Order0, dreal[28]);
            dreal[28] = simdFma(x[5], dRinvPower3Order1, dreal[28]);
            dreal[29] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower3Order1, dreal[29]);
            dreal[29] = simdFma(x[29], dRinvPower4Order0, dreal[29]);
            dreal[30] = simdFma(x[30], dRinvPower4Order0, dreal[30]);
            dreal[30] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower2Order2, dreal[30]);
            dreal[30] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[7] * dRinvPower3Order1, dreal[30]);
            dreal[31] = simdFma(x[31], dRinvPower4Order0, dreal[31]);
            dreal[31] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower3Order1, dreal[31]);
            dreal[32] = simdFma(x[0], dRinvPower2Order2, dreal[32]);
            dreal[32] = simdFma(x[9], dRinvPower3Order1, dreal[32]);
            dreal[32] = simdFma(x[7], dRinvPower3Order1, dreal[32]);
            dreal[32] = simdFma(x[32], dRinvPower4Order0, dreal[32]);
            dreal[33] = simdFma(x[33], dRinvPower4Order0, dreal[33]);
            dreal[33] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower3Order1, dreal[33]);
            dreal[34] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[9] * dRinvPower3Order1, dreal[34]);
            dreal[34] = simdFma(x[34], dRinvPower4Order0, dreal[34]);
            dreal[34] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower2Order2, dreal[34]);
            dreal[35] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[10] * dRinvPower4Order1, dreal[35]);
            dreal[35] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[1] * dRinvPower3Order2, dreal[35]);
            dreal[35] = simdFma(x[35], dRinvPower5Order0, dreal[35]);
            dreal[36] = simdFma(x[36], dRinvPower5Order0, dreal[36]);
            dreal[36] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[11] * dRinvPower4Order1, dreal[36]);
            dreal[36] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower3Order2, dreal[36]);
            dreal[37] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower3Order2, dreal[37]);
            dreal[37] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[12] * dRinvPower4Order1, dreal[37]);
            dreal[37] = simdFma(x[37], dRinvPower5Order0, dreal[37]);
            dreal[38] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower3Order2, dreal[38]);
            dreal[38] = simdFma(x[38], dRinvPower5Order0, dreal[38]);
            dreal[38] = simdFma(x[10], dRinvPower4Order1, dreal[38]);
            dreal[38] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[13] * dRinvPower4Order1, dreal[38]);
            dreal[39] = simdFma(x[39], dRinvPower5Order0, dreal[39]);
            dreal[39] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[14] * dRinvPower4Order1, dreal[39]);
            dreal[40] = simdFma(x[40], dRinvPower5Order0, dreal[40]);
            dreal[40] = simdFma(x[10], dRinvPower4Order1, dreal[40]);
            dreal[40] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[15] * dRinvPower4Order1, dreal[40]);
            dreal[40] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower3Order2, dreal[40]);
            dreal[41] = simdFma(x[41], dRinvPower5Order0, dreal[41]);
            dreal[41] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[11] * dRinvPower4Order1, dreal[41]);
            dreal[41] = simdFma(x[16], dRinvPower4Order1, dreal[41]);
            dreal[41] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower3Order2, dreal[41]);
            dreal[42] = simdFma(x[12], dRinvPower4Order1, dreal[42]);
            dreal[42] = simdFma(x[3], dRinvPower3Order2, dreal[42]);
            dreal[42] = simdFma(x[17], dRinvPower4Order1, dreal[42]);
            dreal[42] = simdFma(x[42], dRinvPower5Order0, dreal[42]);
            dreal[43] = simdFma(x[11], dRinvPower4Order1, dreal[43]);
            dreal[43] = simdFma(x[43], dRinvPower5Order0, dreal[43]);
            dreal[43] = simdFma(x[2], dRinvPower3Order2, dreal[43]);
            dreal[43] = simdFma(x[18], dRinvPower4Order1, dreal[43]);
            dreal[44] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower3Order2, dreal[44]);
            dreal[44] = simdFma(x[44], dRinvPower5Order0, dreal[44]);
            dreal[44] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[12] * dRinvPower4Order1, dreal[44]);
            dreal[44] = simdFma(x[19], dRinvPower4Order1, dreal[44]);
            dreal[45] = simdFma(x[45], dRinvPower5Order0, dreal[45]);
            dreal[45] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[13] * dRinvPower4Order1, dreal[45]);
            dreal[45] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower3Order2, dreal[45]);
            dreal[46] = simdFma(x[46], dRinvPower5Order0, dreal[46]);
            dreal[46] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[14] * dRinvPower4Order1, dreal[46]);
            dreal[47] = simdFma(x[47], dRinvPower5Order0, dreal[47]);
            dreal[47] = simdFma(x[13], dRinvPower4Order1, dreal[47]);
            dreal[47] = simdFma(x[15], dRinvPower4Order1, dreal[47]);
            dreal[47] = simdFma(x[1], dRinvPower3Order2, dreal[47]);
            dreal[48] = simdFma(x[48], dRinvPower5Order0, dreal[48]);
            dreal[48] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[14] * dRinvPower4Order1, dreal[48]);
            dreal[49] = simdFma(x[49], dRinvPower5Order0, dreal[49]);
            dreal[49] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[15] * dRinvPower4Order1, dreal[49]);
            dreal[49] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[1] * dRinvPower3Order2, dreal[49]);
            dreal[50] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[2] * dRinvPower3Order2, dreal[50]);
            dreal[50] = simdFma(x[50], dRinvPower5Order0, dreal[50]);
            dreal[50] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[16] * dRinvPower4Order1, dreal[50]);
            dreal[51] = simdFma(x[51], dRinvPower5Order0, dreal[51]);
            dreal[51] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[17] * dRinvPower4Order1, dreal[51]);
            dreal[51] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower3Order2, dreal[51]);
            dreal[52] = simdFma(x[16], dRinvPower4Order1, dreal[52]);
            dreal[52] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[18] * dRinvPower4Order1, dreal[52]);
            dreal[52] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower3Order2, dreal[52]);
            dreal[52] = simdFma(x[52], dRinvPower5Order0, dreal[52]);
            dreal[53] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[17] * dRinvPower4Order1, dreal[53]);
            dreal[53] = simdFma(x[53], dRinvPower5Order0, dreal[53]);
            dreal[53] = simdFma(x[19], dRinvPower4Order1, dreal[53]);
            dreal[53] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[3] * dRinvPower3Order2, dreal[53]);
            dreal[54] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[2] * dRinvPower3Order2, dreal[54]);
            dreal[54] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[18] * dRinvPower4Order1, dreal[54]);
            dreal[54] = simdFma(x[54], dRinvPower5Order0, dreal[54]);
            dreal[55] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[3] * dRinvPower3Order2, dreal[55]);
            dreal[55] = simdFma(x[55], dRinvPower5Order0, dreal[55]);
            dreal[55] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[19] * dRinvPower4Order1, dreal[55]);
            dreal[56] = simdFma(x[56], dRinvPower6Order0, dreal[56]);
            dreal[56] =
                simdFma(simdBroadcast<T>(4.500000000e+01), x[4] * dRinvPower4Order2, dreal[56]);
            dreal[56] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[20] * dRinvPower5Order1, dreal[56]);
            dreal[56] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[0] * dRinvPower3Order3, dreal[56]);
            dreal[57] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[5] * dRinvPower4Order2, dreal[57]);
            dreal[57] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[21] * dRinvPower5Order1, dreal[57]);
            dreal[57] = simdFma(x[57], dRinvPower6Order0, dreal[57]);
            dreal[58] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[6] * dRinvPower4Order2, dreal[58]);
            dreal[58] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[22] * dRinvPower5Order1, dreal[58]);
            dreal[58] = simdFma(x[58], dRinvPower6Order0, dreal[58]);
            dreal[59] = simdFma(x[59], dRinvPower6Order0, dreal[59]);
            dreal[59] = simdFma(x[20], dRinvPower5Order1, dreal[59]);
            dreal[59] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[23] * dRinvPower5Order1, dreal[59]);
            dreal[59] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[7] * dRinvPower4Order2, dreal[59]);
            dreal[59] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[59]);
            dreal[59] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[4] * dRinvPower4Order2, dreal[59]);
            dreal[60] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower4Order2, dreal[60]);
            dreal[60] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[24] * dRinvPower5Order1, dreal[60]);
            dreal[60] = simdFma(x[60], dRinvPower6Order0, dreal[60]);
            dreal[61] = simdFma(x[20], dRinvPower5Order1, dreal[61]);
            dreal[61] = simdFma(x[61], dRinvPower6Order0, dreal[61]);
            dreal[61] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[25] * dRinvPower5Order1, dreal[61]);
            dreal[61] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[4] * dRinvPower4Order2, dreal[61]);
            dreal[61] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[9] * dRinvPower4Order2, dreal[61]);
            dreal[61] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[61]);
            dreal[62] = simdFma(x[62], dRinvPower6Order0, dreal[62]);
            dreal[62] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[21] * dRinvPower5Order1, dreal[62]);
            dreal[62] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[26] * dRinvPower5Order1, dreal[62]);
            dreal[62] =
                simdFma(simdBroadcast<T>(9.000000000e+00), x[5] * dRinvPower4Order2, dreal[62]);
            dreal[63] = simdFma(x[63], dRinvPower6Order0, dreal[63]);
            dreal[63] = simdFma(x[22], dRinvPower5Order1, dreal[63]);
            dreal[63] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[27] * dRinvPower5Order1, dreal[63]);
            dreal[63] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower4Order2, dreal[63]);
            dreal[64] = simdFma(x[64], dRinvPower6Order0, dreal[64]);
            dreal[64] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower4Order2, dreal[64]);
            dreal[64] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[28] * dRinvPower5Order1, dreal[64]);
            dreal[64] = simdFma(x[21], dRinvPower5Order1, dreal[64]);
            dreal[65] =
                simdFma(simdBroadcast<T>(9.000000000e+00), x[6] * dRinvPower4Order2, dreal[65]);
            dreal[65] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[29] * dRinvPower5Order1, dreal[65]);
            dreal[65] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[22] * dRinvPower5Order1, dreal[65]);
            dreal[65] = simdFma(x[65], dRinvPower6Order0, dreal[65]);
            dreal[66] = simdFma(x[66], dRinvPower6Order0, dreal[66]);
            dreal[66] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[23] * dRinvPower5Order1, dreal[66]);
            dreal[66] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[4] * dRinvPower4Order2, dreal[66]);
            dreal[66] = simdFma(x[30], dRinvPower5Order1, dreal[66]);
            dreal[66] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[7] * dRinvPower4Order2, dreal[66]);
            dreal[66] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[66]);
            dreal[67] = simdFma(x[67], dRinvPower6Order0, dreal[67]);
            dreal[67] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[24] * dRinvPower5Order1, dreal[67]);
            dreal[67] = simdFma(x[31], dRinvPower5Order1, dreal[67]);
            dreal[67] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower4Order2, dreal[67]);
            dreal[68] = simdFma(x[9], dRinvPower4Order2, dreal[68]);
            dreal[68] = simdFma(x[68], dRinvPower6Order0, dreal[68]);
            dreal[68] = simdFma(x[23], dRinvPower5Order1, dreal[68]);
            dreal[68] = simdFma(x[25], dRinvPower5Order1, dreal[68]);
            dreal[68] = simdFma(x[4], dRinvPower4Order2, dreal[68]);
            dreal[68] = simdFma(x[32], dRinvPower5Order1, dreal[68]);
            dreal[68] = simdFma(x[7], dRinvPower4Order2, dreal[68]);
            dreal[68] = simdFma(x[0], dRinvPower3Order3, dreal[68]);
            dreal[69] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[8] * dRinvPower4Order2, dreal[69]);
            dreal[69] = simdFma(x[33], dRinvPower5Order1, dreal[69]);
            dreal[69] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[24] * dRinvPower5Order1, dreal[69]);
            dreal[69] = simdFma(x[69], dRinvPower6Order0, dreal[69]);
            dreal[70] = simdFma(x[34], dRinvPower5Order1, dreal[70]);
            dreal[70] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[4] * dRinvPower4Order2, dreal[70]);
            dreal[70] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[9] * dRinvPower4Order2, dreal[70]);
            dreal[70] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[70]);
            dreal[70] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[25] * dRinvPower5Order1, dreal[70]);
            dreal[70] = simdFma(x[70], dRinvPower6Order0, dreal[70]);
            dreal[71] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[5] * dRinvPower4Order2, dreal[71]);
            dreal[71] = simdFma(x[71], dRinvPower6Order0, dreal[71]);
            dreal[71] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[26] * dRinvPower5Order1, dreal[71]);
            dreal[72] = simdFma(x[72], dRinvPower6Order0, dreal[72]);
            dreal[72] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[27] * dRinvPower5Order1, dreal[72]);
            dreal[72] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower4Order2, dreal[72]);
            dreal[73] = simdFma(x[73], dRinvPower6Order0, dreal[73]);
            dreal[73] = simdFma(x[26], dRinvPower5Order1, dreal[73]);
            dreal[73] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[28] * dRinvPower5Order1, dreal[73]);
            dreal[73] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower4Order2, dreal[73]);
            dreal[74] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[6] * dRinvPower4Order2, dreal[74]);
            dreal[74] = simdFma(x[29], dRinvPower5Order1, dreal[74]);
            dreal[74] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[27] * dRinvPower5Order1, dreal[74]);
            dreal[74] = simdFma(x[74], dRinvPower6Order0, dreal[74]);
            dreal[75] = simdFma(x[75], dRinvPower6Order0, dreal[75]);
            dreal[75] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[5] * dRinvPower4Order2, dreal[75]);
            dreal[75] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[28] * dRinvPower5Order1, dreal[75]);
            dreal[76] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[6] * dRinvPower4Order2, dreal[76]);
            dreal[76] = simdFma(x[76], dRinvPower6Order0, dreal[76]);
            dreal[76] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[29] * dRinvPower5Order1, dreal[76]);
            dreal[77] = simdFma(x[77], dRinvPower6Order0, dreal[77]);
            dreal[77] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[30] * dRinvPower5Order1, dreal[77]);
            dreal[77] =
                simdFma(simdBroadcast<T>(4.500000000e+01), x[7] * dRinvPower4Order2, dreal[77]);
            dreal[77] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[0] * dRinvPower3Order3, dreal[77]);
            dreal[78] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[8] * dRinvPower4Order2, dreal[78]);
            dreal[78] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[31] * dRinvPower5Order1, dreal[78]);
            dreal[78] = simdFma(x[78], dRinvPower6Order0, dreal[78]);
            dreal[79] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[7] * dRinvPower4Order2, dreal[79]);
            dreal[79] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[9] * dRinvPower4Order2, dreal[79]);
            dreal[79] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[79]);
            dreal[79] = simdFma(x[79], dRinvPower6Order0, dreal[79]);
            dreal[79] = simdFma(x[30], dRinvPower5Order1, dreal[79]);
            dreal[79] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[32] * dRinvPower5Order1, dreal[79]);
            dreal[80] = simdFma(x[80], dRinvPower6Order0, dreal[80]);
            dreal[80] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[31] * dRinvPower5Order1, dreal[80]);
            dreal[80] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[33] * dRinvPower5Order1, dreal[80]);
            dreal[80] =
                simdFma(simdBroadcast<T>(9.000000000e+00), x[8] * dRinvPower4Order2, dreal[80]);
            dreal[81] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[32] * dRinvPower5Order1, dreal[81]);
            dreal[81] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[7] * dRinvPower4Order2, dreal[81]);
            dreal[81] = simdFma(x[34], dRinvPower5Order1, dreal[81]);
            dreal[81] =
                simdFma(simdBroadcast<T>(6.000000000e+00), x[9] * dRinvPower4Order2, dreal[81]);
            dreal[81] =
                simdFma(simdBroadcast<T>(3.000000000e+00), x[0] * dRinvPower3Order3, dreal[81]);
            dreal[81] = simdFma(x[81], dRinvPower6Order0, dreal[81]);
            dreal[82] = simdFma(x[82], dRinvPower6Order0, dreal[82]);
            dreal[82] =
                simdFma(simdBroadcast<T>(1.000000000e+01), x[33] * dRinvPower5Order1, dreal[82]);
            dreal[82] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[8] * dRinvPower4Order2, dreal[82]);
            dreal[83] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[0] * dRinvPower3Order3, dreal[83]);
            dreal[83] =
                simdFma(simdBroadcast<T>(4.500000000e+01), x[9] * dRinvPower4Order2, dreal[83]);
            dreal[83] =
                simdFma(simdBroadcast<T>(1.500000000e+01), x[34] * dRinvPower5Order1, dreal[83]);
            dreal[83] = simdFma(x[83], dRinvPower6Order0, dreal[83]);
        }
    }
    auto const foursz = econst.nfour();
    for (int i = 0; i < foursz; i++) {
        auto const& h = econst.fourIndex(i);
        auto const& d0Value = econst.fourExpansion(i);
        T const hdotx = simdFma(simdBroadcast<T>(h[0]), position[0],
            simdFma(simdBroadcast<T>(h[1]), position[1], simdBroadcast<T>(h[2]) * position[2]));
        T cn, sn;
        sincos(simdBroadcast<T>(2.0 * M_PI * SCALE_FACTOR_INV1) * hdotx, &sn, &cn);
        dfour[0] = simdFma(cn, simdBroadcast<T>(d0Value[0]), dfour[0]);
        dfour[1] = simdFma(sn, simdBroadcast<T>(d0Value[1]), dfour[1]);
        dfour[2] = simdFma(sn, simdBroadcast<T>(d0Value[2]), dfour[2]);
        dfour[3] = simdFma(cn, simdBroadcast<T>(d0Value[3]), dfour[3]);
        dfour[4] = simdFma(cn, simdBroadcast<T>(d0Value[4]), dfour[4]);
        dfour[5] = simdFma(cn, simdBroadcast<T>(d0Value[5]), dfour[5]);
        dfour[6] = simdFma(sn, simdBroadcast<T>(d0Value[6]), dfour[6]);
        dfour[7] = simdFma(sn, simdBroadcast<T>(d0Value[7]), dfour[7]);
        dfour[8] = simdFma(sn, simdBroadcast<T>(d0Value[8]), dfour[8]);
        dfour[9] = simdFma(sn, simdBroadcast<T>(d0Value[9]), dfour[9]);
        dfour[10] = simdFma(cn, simdBroadcast<T>(d0Value[10]), dfour[10]);
        dfour[11] = simdFma(cn, simdBroadcast<T>(d0Value[11]), dfour[11]);
        dfour[12] = simdFma(cn, simdBroadcast<T>(d0Value[12]), dfour[12]);
        dfour[13] = simdFma(cn, simdBroadcast<T>(d0Value[13]), dfour[13]);
        dfour[14] = simdFma(cn, simdBroadcast<T>(d0Value[14]), dfour[14]);
        dfour[15] = simdFma(sn, simdBroadcast<T>(d0Value[15]), dfour[15]);
        dfour[16] = simdFma(sn, simdBroadcast<T>(d0Value[16]), dfour[16]);
        dfour[17] = simdFma(sn, simdBroadcast<T>(d0Value[17]), dfour[17]);
        dfour[18] = simdFma(sn, simdBroadcast<T>(d0Value[18]), dfour[18]);
        dfour[19] = simdFma(sn, simdBroadcast<T>(d0Value[19]), dfour[19]);
        dfour[20] = simdFma(sn, simdBroadcast<T>(d0Value[20]), dfour[20]);
        dfour[21] = simdFma(cn, simdBroadcast<T>(d0Value[21]), dfour[21]);
        dfour[22] = simdFma(cn, simdBroadcast<T>(d0Value[22]), dfour[22]);
        dfour[23] = simdFma(cn, simdBroadcast<T>(d0Value[23]), dfour[23]);
        dfour[24] = simdFma(cn, simdBroadcast<T>(d0Value[24]), dfour[24]);
        dfour[25] = simdFma(cn, simdBroadcast<T>(d0Value[25]), dfour[25]);
        dfour[26] = simdFma(cn, simdBroadcast<T>(d0Value[26]), dfour[26]);
        dfour[27] = simdFma(cn, simdBroadcast<T>(d0Value[27]), dfour[27]);
        dfour[28] = simdFma(sn, simdBroadcast<T>(d0Value[28]), dfour[28]);
        dfour[29] = simdFma(cn, simdBroadcast<T>(d0Value[29]), dfour[29]);
        dfour[30] = simdFma(cn, simdBroadcast<T>(d0Value[30]), dfour[30]);
        dfour[31] = simdFma(sn, simdBroadcast<T>(d0Value[31]), dfour[31]);
        dfour[32] = simdFma(sn, simdBroadcast<T>(d0Value[32]), dfour[32]);
        dfour[33] = simdFma(sn, simdBroadcast<T>(d0Value[33]), dfour[33]);
        dfour[34] = simdFma(cn, simdBroadcast<T>(d0Value[34]), dfour[34]);
        dfour[35] = simdFma(cn, simdBroadcast<T>(d0Value[35]), dfour[35]);
        dfour[36] = simdFma(cn, simdBroadcast<T>(d0Value[36]), dfour[36]);
        dfour[37] = simdFma(cn, simdBroadcast<T>(d0Value[37]), dfour[37]);
        dfour[38] = simdFma(sn, simdBroadcast<T>(d0Value[38]), dfour[38]);
        dfour[39] = simdFma(sn, simdBroadcast<T>(d0Value[39]), dfour[39]);
        dfour[40] = simdFma(sn, simdBroadcast<T>(d0Value[40]), dfour[40]);
        dfour[41] = simdFma(sn, simdBroadcast<T>(d0Value[41]), dfour[41]);
        dfour[42] = simdFma(sn, simdBroadcast<T>(d0Value[42]), dfour[42]);
        dfour[43] = simdFma(cn, simdBroadcast<T>(d0Value[43]), dfour[43]);
        dfour[44] = simdFma(cn, simdBroadcast<T>(d0Value[44]), dfour[44]);
        dfour[45] = simdFma(cn, simdBroadcast<T>(d0Value[45]), dfour[45]);
        dfour[46] = simdFma(cn, simdBroadcast<T>(d0Value[46]), dfour[46]);
        dfour[47] = simdFma(cn, simdBroadcast<T>(d0Value[47]), dfour[47]);
        dfour[48] = simdFma(cn, simdBroadcast<T>(d0Value[48]), dfour[48]);
        dfour[49] = simdFma(cn, simdBroadcast<T>(d0Value[49]), dfour[49]);
    }
    T drealOrder2Trace1Index000 = dreal[9];
    T drealOrder3Trace1Index001 = dreal[19];
    T drealOrder4Trace1Index002 = dreal[34];
    T drealOrder5Trace1Index003 = dreal[55];
    T drealOrder6Trace1Index004 = dreal[83];
    T drealOrder3Trace1Index010 = dreal[18];
    T drealOrder4Trace1Index011 = dreal[33];
    T drealOrder5Trace1Index012 = dreal[54];
    T drealOrder6Trace1Index013 = dreal[82];
    T drealOrder4Trace1Index020 = dreal[32];
    T drealOrder5Trace1Index021 = dreal[53];
    T drealOrder6Trace1Index022 = dreal[81];
    T drealOrder5Trace1Index030 = dreal[52];
    T drealOrder6Trace1Index031 = dreal[80];
    T drealOrder6Trace1Index040 = dreal[79];
    T drealOrder3Trace1Index100 = dreal[15];
    T drealOrder4Trace1Index101 = dreal[29];
    T drealOrder5Trace1Index102 = dreal[49];
    T drealOrder6Trace1Index103 = dreal[76];
    T drealOrder4Trace1Index110 = dreal[28];
    T drealOrder5Trace1Index111 = dreal[48];
    T drealOrder6Trace1Index112 = dreal[75];
    T drealOrder5Trace1Index120 = dreal[47];
    T drealOrder6Trace1Index121 = dreal[74];
    T drealOrder6Trace1Index130 = dreal[73];
    T drealOrder4Trace1Index200 = dreal[25];
    T drealOrder5Trace1Index201 = dreal[44];
    T drealOrder6Trace1Index202 = dreal[70];
    T drealOrder5Trace1Index210 = dreal[43];
    T drealOrder6Trace1Index211 = dreal[69];
    T drealOrder6Trace1Index220 = dreal[68];
    T drealOrder5Trace1Index300 = dreal[40];
    T drealOrder6Trace1Index301 = dreal[65];
    T drealOrder6Trace1Index310 = dreal[64];
    T drealOrder6Trace1Index400 = dreal[61];
    T drealOrder4Trace2Index000 = dreal[34];
    T drealOrder5Trace2Index001 = dreal[55];
    T drealOrder6Trace2Index002 = dreal[83];
    T drealOrder5Trace2Index010 = dreal[54];
    T drealOrder6Trace2Index011 = dreal[82];
    T drealOrder6Trace2Index020 = dreal[81];
    T drealOrder5Trace2Index100 = dreal[49];
    T drealOrder6Trace2Index101 = dreal[76];
    T drealOrder6Trace2Index110 = dreal[75];
    T drealOrder6Trace2Index200 = dreal[70];
    T drealOrder6Trace3Index000 = dreal[83];
    drealOrder2Trace1Index000 += dreal[7];
    drealOrder6Trace1Index301 += dreal[58];
    drealOrder2Trace1Index000 += dreal[4];
    drealOrder6Trace1Index310 += dreal[62];
    drealOrder3Trace1Index001 += dreal[17];
    drealOrder6Trace1Index310 += dreal[57];
    drealOrder3Trace1Index001 += dreal[12];
    drealOrder6Trace1Index400 += dreal[59];
    drealOrder4Trace1Index002 += dreal[32];
    drealOrder6Trace1Index400 += dreal[56];
    drealOrder4Trace1Index002 += dreal[25];
    drealOrder4Trace2Index000 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[32], drealOrder4Trace2Index000);
    drealOrder5Trace1Index003 += dreal[53];
    drealOrder4Trace2Index000 += dreal[30];
    drealOrder5Trace1Index003 += dreal[44];
    drealOrder4Trace2Index000 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[25], drealOrder4Trace2Index000);
    drealOrder6Trace1Index004 += dreal[81];
    drealOrder4Trace2Index000 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[23], drealOrder4Trace2Index000);
    drealOrder6Trace1Index004 += dreal[70];
    drealOrder4Trace2Index000 += dreal[20];
    drealOrder3Trace1Index010 += dreal[16];
    drealOrder5Trace2Index001 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[53], drealOrder5Trace2Index001);
    drealOrder3Trace1Index010 += dreal[11];
    drealOrder5Trace2Index001 += dreal[51];
    drealOrder4Trace1Index011 += dreal[31];
    drealOrder5Trace2Index001 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[44], drealOrder5Trace2Index001);
    drealOrder4Trace1Index011 += dreal[24];
    drealOrder5Trace2Index001 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[42], drealOrder5Trace2Index001);
    drealOrder5Trace1Index012 += dreal[52];
    drealOrder5Trace2Index001 += dreal[37];
    drealOrder5Trace1Index012 += dreal[43];
    drealOrder6Trace2Index002 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[81], drealOrder6Trace2Index002);
    drealOrder6Trace1Index013 += dreal[80];
    drealOrder6Trace2Index002 += dreal[79];
    drealOrder6Trace1Index013 += dreal[69];
    drealOrder6Trace2Index002 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[70], drealOrder6Trace2Index002);
    drealOrder4Trace1Index020 += dreal[30];
    drealOrder6Trace2Index002 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[68], drealOrder6Trace2Index002);
    drealOrder4Trace1Index020 += dreal[23];
    drealOrder6Trace2Index002 += dreal[61];
    drealOrder5Trace1Index021 += dreal[51];
    drealOrder5Trace2Index010 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[52], drealOrder5Trace2Index010);
    drealOrder5Trace1Index021 += dreal[42];
    drealOrder5Trace2Index010 += dreal[50];
    drealOrder6Trace1Index022 += dreal[79];
    drealOrder5Trace2Index010 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[43], drealOrder5Trace2Index010);
    drealOrder6Trace1Index022 += dreal[68];
    drealOrder5Trace2Index010 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[41], drealOrder5Trace2Index010);
    drealOrder5Trace1Index030 += dreal[50];
    drealOrder5Trace2Index010 += dreal[36];
    drealOrder5Trace1Index030 += dreal[41];
    drealOrder6Trace2Index011 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[80], drealOrder6Trace2Index011);
    drealOrder6Trace1Index031 += dreal[78];
    drealOrder6Trace2Index011 += dreal[78];
    drealOrder6Trace1Index031 += dreal[67];
    drealOrder6Trace2Index011 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[69], drealOrder6Trace2Index011);
    drealOrder6Trace1Index040 += dreal[77];
    drealOrder6Trace2Index011 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[67], drealOrder6Trace2Index011);
    drealOrder6Trace1Index040 += dreal[66];
    drealOrder6Trace2Index011 += dreal[60];
    drealOrder3Trace1Index100 += dreal[13];
    drealOrder6Trace2Index020 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[79], drealOrder6Trace2Index020);
    drealOrder3Trace1Index100 += dreal[10];
    drealOrder6Trace2Index020 += dreal[77];
    drealOrder4Trace1Index101 += dreal[27];
    drealOrder6Trace2Index020 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[68], drealOrder6Trace2Index020);
    drealOrder4Trace1Index101 += dreal[22];
    drealOrder6Trace2Index020 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[66], drealOrder6Trace2Index020);
    drealOrder5Trace1Index102 += dreal[47];
    drealOrder6Trace2Index020 += dreal[59];
    drealOrder5Trace1Index102 += dreal[40];
    drealOrder5Trace2Index100 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[47], drealOrder5Trace2Index100);
    drealOrder6Trace1Index103 += dreal[74];
    drealOrder5Trace2Index100 += dreal[45];
    drealOrder6Trace1Index103 += dreal[65];
    drealOrder5Trace2Index100 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[40], drealOrder5Trace2Index100);
    drealOrder4Trace1Index110 += dreal[26];
    drealOrder5Trace2Index100 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[38], drealOrder5Trace2Index100);
    drealOrder4Trace1Index110 += dreal[21];
    drealOrder5Trace2Index100 += dreal[35];
    drealOrder5Trace1Index111 += dreal[46];
    drealOrder6Trace2Index101 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[74], drealOrder6Trace2Index101);
    drealOrder5Trace1Index111 += dreal[39];
    drealOrder6Trace2Index101 += dreal[72];
    drealOrder6Trace1Index112 += dreal[73];
    drealOrder6Trace2Index101 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[65], drealOrder6Trace2Index101);
    drealOrder6Trace1Index112 += dreal[64];
    drealOrder6Trace2Index101 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[63], drealOrder6Trace2Index101);
    drealOrder5Trace1Index120 += dreal[45];
    drealOrder6Trace2Index101 += dreal[58];
    drealOrder5Trace1Index120 += dreal[38];
    drealOrder6Trace2Index110 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[73], drealOrder6Trace2Index110);
    drealOrder6Trace1Index121 += dreal[72];
    drealOrder6Trace2Index110 += dreal[71];
    drealOrder6Trace1Index121 += dreal[63];
    drealOrder6Trace2Index110 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[64], drealOrder6Trace2Index110);
    drealOrder6Trace1Index130 += dreal[71];
    drealOrder6Trace2Index110 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[62], drealOrder6Trace2Index110);
    drealOrder6Trace1Index130 += dreal[62];
    drealOrder6Trace2Index110 += dreal[57];
    drealOrder4Trace1Index200 += dreal[23];
    drealOrder6Trace2Index200 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[68], drealOrder6Trace2Index200);
    drealOrder4Trace1Index200 += dreal[20];
    drealOrder6Trace2Index200 += dreal[66];
    drealOrder5Trace1Index201 += dreal[42];
    drealOrder6Trace2Index200 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[61], drealOrder6Trace2Index200);
    drealOrder5Trace1Index201 += dreal[37];
    drealOrder6Trace2Index200 =
        simdFma(simdBroadcast<T>(2.000000000e+00), dreal[59], drealOrder6Trace2Index200);
    drealOrder6Trace1Index202 += dreal[68];
    drealOrder6Trace2Index200 += dreal[56];
    drealOrder6Trace1Index202 += dreal[61];
    drealOrder6Trace3Index000 =
        simdFma(simdBroadcast<T>(3.000000000e+00), dreal[81], drealOrder6Trace3Index000);
    drealOrder5Trace1Index210 += dreal[41];
    drealOrder6Trace3Index000 =
        simdFma(simdBroadcast<T>(3.000000000e+00), dreal[79], drealOrder6Trace3Index000);
    drealOrder5Trace1Index210 += dreal[36];
    drealOrder6Trace3Index000 += dreal[77];
    drealOrder6Trace1Index211 += dreal[67];
    drealOrder6Trace3Index000 =
        simdFma(simdBroadcast<T>(3.000000000e+00), dreal[70], drealOrder6Trace3Index000);
    drealOrder6Trace1Index211 += dreal[60];
    drealOrder6Trace3Index000 =
        simdFma(simdBroadcast<T>(6.000000000e+00), dreal[68], drealOrder6Trace3Index000);
    drealOrder6Trace1Index220 += dreal[66];
    drealOrder6Trace3Index000 =
        simdFma(simdBroadcast<T>(3.000000000e+00), dreal[66], drealOrder6Trace3Index000);
    drealOrder6Trace1Index220 += dreal[59];
    drealOrder6Trace3Index000 =
        simdFma(simdBroadcast<T>(3.000000000e+00), dreal[61], drealOrder6Trace3Index000);
    drealOrder5Trace1Index300 += dreal[38];
    drealOrder6Trace3Index000 =
        simdFma(simdBroadcast<T>(3.000000000e+00), dreal[59], drealOrder6Trace3Index000);
    drealOrder5Trace1Index300 += dreal[35];
    drealOrder6Trace3Index000 += dreal[56];
    drealOrder6Trace1Index301 += dreal[63];
    d[0] = dreal[0];
    d[28] = dreal[3];
    d[49] = dreal[9];
    d[2] = dreal[2];
    d[30] = dreal[8];
    d[5] = dreal[7];
    d[33] = dreal[17];
    d[9] = dreal[16];
    d[37] = dreal[31];
    d[14] = dreal[30];
    d[42] = dreal[51];
    d[20] = dreal[50];
    d[48] = dreal[78];
    d[27] = dreal[77];
    d[1] = dreal[1];
    d[29] = dreal[6];
    d[4] = dreal[5];
    d[32] = dreal[14];
    d[8] = dreal[13];
    d[36] = dreal[27];
    d[13] = dreal[26];
    d[41] = dreal[46];
    d[19] = dreal[45];
    d[47] = dreal[72];
    d[26] = dreal[71];
    d[3] = dreal[4];
    d[31] = dreal[12];
    d[7] = dreal[11];
    d[35] = dreal[24];
    d[12] = dreal[23];
    d[40] = dreal[42];
    d[18] = dreal[41];
    d[46] = dreal[67];
    d[25] = dreal[66];
    d[6] = dreal[10];
    d[34] = dreal[22];
    d[11] = dreal[21];
    d[39] = dreal[39];
    d[17] = dreal[38];
    d[45] = dreal[63];
    d[24] = dreal[62];
    d[10] = dreal[20];
    d[38] = dreal[37];
    d[16] = dreal[36];
    d[44] = dreal[60];
    d[23] = dreal[59];
    d[15] = dreal[35];
    d[43] = dreal[58];
    d[22] = dreal[57];
    d[21] = dreal[56];
    d[33] = simdFma(simdBroadcast<T>(-2.000000000e-01), drealOrder3Trace1Index001, d[33]);
    d[25] = simdFma(simdBroadcast<T>(3.030303030e-02), drealOrder6Trace2Index200, d[25]);
    d[9] = simdFma(simdBroadcast<T>(-6.000000000e-01), drealOrder3Trace1Index010, d[9]);
    d[25] = simdFma(simdBroadcast<T>(-9.090909091e-02), drealOrder6Trace1Index040, d[25]);
    d[37] = simdFma(simdBroadcast<T>(-4.285714286e-01), drealOrder4Trace1Index011, d[37]);
    d[25] = simdFma(simdBroadcast<T>(6.060606061e-02), drealOrder6Trace2Index020, d[25]);
    d[14] = simdFma(simdBroadcast<T>(-8.571428571e-01), drealOrder4Trace1Index020, d[14]);
    d[25] = simdFma(simdBroadcast<T>(-4.329004329e-03), drealOrder6Trace3Index000, d[25]);
    d[14] = simdFma(simdBroadcast<T>(8.571428571e-02), drealOrder4Trace2Index000, d[14]);
    d[6] = simdFma(simdBroadcast<T>(-6.000000000e-01), drealOrder3Trace1Index100, d[6]);
    d[42] = simdFma(simdBroadcast<T>(-6.666666667e-01), drealOrder5Trace1Index021, d[42]);
    d[34] = simdFma(simdBroadcast<T>(-4.285714286e-01), drealOrder4Trace1Index101, d[34]);
    d[42] = simdFma(simdBroadcast<T>(4.761904762e-02), drealOrder5Trace2Index001, d[42]);
    d[11] = simdFma(simdBroadcast<T>(-4.285714286e-01), drealOrder4Trace1Index110, d[11]);
    d[20] = simdFma(simdBroadcast<T>(-1.111111111e+00), drealOrder5Trace1Index030, d[20]);
    d[39] = simdFma(simdBroadcast<T>(-3.333333333e-01), drealOrder5Trace1Index111, d[39]);
    d[20] = simdFma(simdBroadcast<T>(2.380952381e-01), drealOrder5Trace2Index010, d[20]);
    d[17] = simdFma(simdBroadcast<T>(-1.111111111e-01), drealOrder5Trace1Index300, d[17]);
    d[48] = simdFma(simdBroadcast<T>(-9.090909091e-01), drealOrder6Trace1Index031, d[48]);
    d[17] = simdFma(simdBroadcast<T>(-3.333333333e-01), drealOrder5Trace1Index120, d[17]);
    d[48] = simdFma(simdBroadcast<T>(1.515151515e-01), drealOrder6Trace2Index011, d[48]);
    d[17] = simdFma(simdBroadcast<T>(4.761904762e-02), drealOrder5Trace2Index100, d[17]);
    d[27] = simdFma(simdBroadcast<T>(-1.363636364e+00), drealOrder6Trace1Index040, d[27]);
    d[45] = simdFma(simdBroadcast<T>(-9.090909091e-02), drealOrder6Trace1Index301, d[45]);
    d[27] = simdFma(simdBroadcast<T>(4.545454545e-01), drealOrder6Trace2Index020, d[27]);
    d[45] = simdFma(simdBroadcast<T>(-2.727272727e-01), drealOrder6Trace1Index121, d[45]);
    d[27] = simdFma(simdBroadcast<T>(-2.164502165e-02), drealOrder6Trace3Index000, d[27]);
    d[45] = simdFma(simdBroadcast<T>(3.030303030e-02), drealOrder6Trace2Index101, d[45]);
    d[8] = simdFma(simdBroadcast<T>(-2.000000000e-01), drealOrder3Trace1Index100, d[8]);
    d[24] = simdFma(simdBroadcast<T>(-2.727272727e-01), drealOrder6Trace1Index310, d[24]);
    d[36] = simdFma(simdBroadcast<T>(-1.428571429e-01), drealOrder4Trace1Index101, d[36]);
    d[24] = simdFma(simdBroadcast<T>(-2.727272727e-01), drealOrder6Trace1Index130, d[24]);
    d[13] = simdFma(simdBroadcast<T>(-4.285714286e-01), drealOrder4Trace1Index110, d[13]);
    d[24] = simdFma(simdBroadcast<T>(9.090909091e-02), drealOrder6Trace2Index110, d[24]);
    d[41] = simdFma(simdBroadcast<T>(-3.333333333e-01), drealOrder5Trace1Index111, d[41]);
    d[10] = simdFma(simdBroadcast<T>(-8.571428571e-01), drealOrder4Trace1Index200, d[10]);
    d[19] = simdFma(simdBroadcast<T>(-6.666666667e-01), drealOrder5Trace1Index120, d[19]);
    d[10] = simdFma(simdBroadcast<T>(8.571428571e-02), drealOrder4Trace2Index000, d[10]);
    d[19] = simdFma(simdBroadcast<T>(4.761904762e-02), drealOrder5Trace2Index100, d[19]);
    d[38] = simdFma(simdBroadcast<T>(-6.666666667e-01), drealOrder5Trace1Index201, d[38]);
    d[47] = simdFma(simdBroadcast<T>(-5.454545455e-01), drealOrder6Trace1Index121, d[47]);
    d[38] = simdFma(simdBroadcast<T>(4.761904762e-02), drealOrder5Trace2Index001, d[38]);
    d[47] = simdFma(simdBroadcast<T>(3.030303030e-02), drealOrder6Trace2Index101, d[47]);
    d[16] = simdFma(simdBroadcast<T>(-6.666666667e-01), drealOrder5Trace1Index210, d[16]);
    d[26] = simdFma(simdBroadcast<T>(-9.090909091e-01), drealOrder6Trace1Index130, d[26]);
    d[16] = simdFma(simdBroadcast<T>(4.761904762e-02), drealOrder5Trace2Index010, d[16]);
    d[26] = simdFma(simdBroadcast<T>(1.515151515e-01), drealOrder6Trace2Index110, d[26]);
    d[44] = simdFma(simdBroadcast<T>(-5.454545455e-01), drealOrder6Trace1Index211, d[44]);
    d[31] = simdFma(simdBroadcast<T>(-2.000000000e-01), drealOrder3Trace1Index001, d[31]);
    d[44] = simdFma(simdBroadcast<T>(3.030303030e-02), drealOrder6Trace2Index011, d[44]);
    d[7] = simdFma(simdBroadcast<T>(-2.000000000e-01), drealOrder3Trace1Index010, d[7]);
    d[23] = simdFma(simdBroadcast<T>(-9.090909091e-02), drealOrder6Trace1Index400, d[23]);
    d[35] = simdFma(simdBroadcast<T>(-1.428571429e-01), drealOrder4Trace1Index011, d[35]);
    d[23] = simdFma(simdBroadcast<T>(-5.454545455e-01), drealOrder6Trace1Index220, d[23]);
    d[12] = simdFma(simdBroadcast<T>(-1.428571429e-01), drealOrder4Trace1Index200, d[12]);
    d[23] = simdFma(simdBroadcast<T>(6.060606061e-02), drealOrder6Trace2Index200, d[23]);
    d[12] = simdFma(simdBroadcast<T>(-1.428571429e-01), drealOrder4Trace1Index020, d[12]);
    d[23] = simdFma(simdBroadcast<T>(3.030303030e-02), drealOrder6Trace2Index020, d[23]);
    d[12] = simdFma(simdBroadcast<T>(2.857142857e-02), drealOrder4Trace2Index000, d[12]);
    d[23] = simdFma(simdBroadcast<T>(-4.329004329e-03), drealOrder6Trace3Index000, d[23]);
    d[40] = simdFma(simdBroadcast<T>(-1.111111111e-01), drealOrder5Trace1Index201, d[40]);
    d[15] = simdFma(simdBroadcast<T>(-1.111111111e+00), drealOrder5Trace1Index300, d[15]);
    d[40] = simdFma(simdBroadcast<T>(-1.111111111e-01), drealOrder5Trace1Index021, d[40]);
    d[15] = simdFma(simdBroadcast<T>(2.380952381e-01), drealOrder5Trace2Index100, d[15]);
    d[40] = simdFma(simdBroadcast<T>(1.587301587e-02), drealOrder5Trace2Index001, d[40]);
    d[43] = simdFma(simdBroadcast<T>(-9.090909091e-01), drealOrder6Trace1Index301, d[43]);
    d[18] = simdFma(simdBroadcast<T>(-3.333333333e-01), drealOrder5Trace1Index210, d[18]);
    d[43] = simdFma(simdBroadcast<T>(1.515151515e-01), drealOrder6Trace2Index101, d[43]);
    d[18] = simdFma(simdBroadcast<T>(-1.111111111e-01), drealOrder5Trace1Index030, d[18]);
    d[22] = simdFma(simdBroadcast<T>(-9.090909091e-01), drealOrder6Trace1Index310, d[22]);
    d[18] = simdFma(simdBroadcast<T>(4.761904762e-02), drealOrder5Trace2Index010, d[18]);
    d[22] = simdFma(simdBroadcast<T>(1.515151515e-01), drealOrder6Trace2Index110, d[22]);
    d[46] = simdFma(simdBroadcast<T>(-2.727272727e-01), drealOrder6Trace1Index211, d[46]);
    d[21] = simdFma(simdBroadcast<T>(-1.363636364e+00), drealOrder6Trace1Index400, d[21]);
    d[46] = simdFma(simdBroadcast<T>(-9.090909091e-02), drealOrder6Trace1Index031, d[46]);
    d[21] = simdFma(simdBroadcast<T>(4.545454545e-01), drealOrder6Trace2Index200, d[21]);
    d[46] = simdFma(simdBroadcast<T>(3.030303030e-02), drealOrder6Trace2Index011, d[46]);
    d[21] = simdFma(simdBroadcast<T>(-2.164502165e-02), drealOrder6Trace3Index000, d[21]);
    d[25] = simdFma(simdBroadcast<T>(-5.454545455e-01), drealOrder6Trace1Index220, d[25]);
    flops += 140 * foursz + 540;
    d = d + dfour;
    d[0] = simdBroadcast<T>(7.853981634e-01 * SCALE_FACTOR_INV1) + d[0];
    return flops;
}
}    // namespace detail
