/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once
#include <cosmictiger/gravity/kernels/spherical/ewald_common.hpp>

namespace cosmictiger::gravity::spherical {
#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 3) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 3)
template <>
inline constexpr int kEwaldGreenFlops<2> = 114444;
template <>
inline constexpr int kEwaldGreenConventionalFlops<2> = 119214;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
template <int degree, typename Real>
    requires(degree == 2)
CUDA_EXPORT void ewaldRegular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(2.500000000000000000E-01);
    constexpr auto k1 = static_cast<Scalar>(5.000000000000000000E-01);
    constexpr auto k2 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k3 = static_cast<Scalar>(3.000000000000000000E+00);
    output[0] = broadcast<Real>(k2);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k3) * z;
    output[6] = broadcast<Real>(k0) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    output[3] = x * broadcast<Real>(k1);
    output[1] = y * broadcast<Real>(k1);
    output[7] = z * output[3];
    output[5] = z * output[1];
    output[8] = broadcast<Real>(k0) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k0) * fma(y, output[3], x * output[1]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 2)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k01 = static_cast<Scalar>(1.980198019801980222E-02);
    constexpr auto k02 = static_cast<Scalar>(2.020202020202020374E-02);
    constexpr auto k03 = static_cast<Scalar>(2.061855670103092744E-02);
    constexpr auto k04 = static_cast<Scalar>(2.105263157894736795E-02);
    constexpr auto k05 = static_cast<Scalar>(2.150537634408602322E-02);
    constexpr auto k06 = static_cast<Scalar>(2.197802197802197974E-02);
    constexpr auto k07 = static_cast<Scalar>(2.247191011235954994E-02);
    constexpr auto k08 = static_cast<Scalar>(2.298850574712643646E-02);
    constexpr auto k09 = static_cast<Scalar>(2.352941176470588203E-02);
    constexpr auto k10 = static_cast<Scalar>(2.409638554216867595E-02);
    constexpr auto k11 = static_cast<Scalar>(2.469135802469135665E-02);
    constexpr auto k12 = static_cast<Scalar>(2.531645569620253125E-02);
    constexpr auto k13 = static_cast<Scalar>(2.597402597402597574E-02);
    constexpr auto k14 = static_cast<Scalar>(2.666666666666666838E-02);
    constexpr auto k15 = static_cast<Scalar>(2.739726027397260122E-02);
    constexpr auto k16 = static_cast<Scalar>(2.816901408450704289E-02);
    constexpr auto k17 = static_cast<Scalar>(2.898550724637681195E-02);
    constexpr auto k18 = static_cast<Scalar>(2.985074626865671585E-02);
    constexpr auto k19 = static_cast<Scalar>(3.076923076923077094E-02);
    constexpr auto k20 = static_cast<Scalar>(3.174603174603174427E-02);
    constexpr auto k21 = static_cast<Scalar>(3.278688524590164105E-02);
    constexpr auto k22 = static_cast<Scalar>(3.389830508474576259E-02);
    constexpr auto k23 = static_cast<Scalar>(3.508771929824561209E-02);
    constexpr auto k24 = static_cast<Scalar>(3.636363636363636187E-02);
    constexpr auto k25 = static_cast<Scalar>(3.773584905660377214E-02);
    constexpr auto k26 = static_cast<Scalar>(3.921568627450980338E-02);
    constexpr auto k27 = static_cast<Scalar>(4.081632653061224164E-02);
    constexpr auto k28 = static_cast<Scalar>(4.255319148936170109E-02);
    constexpr auto k29 = static_cast<Scalar>(4.444444444444444614E-02);
    constexpr auto k30 = static_cast<Scalar>(4.651162790697674354E-02);
    constexpr auto k31 = static_cast<Scalar>(4.878048780487805047E-02);
    constexpr auto k32 = static_cast<Scalar>(5.128205128205128027E-02);
    constexpr auto k33 = static_cast<Scalar>(5.405405405405405705E-02);
    constexpr auto k34 = static_cast<Scalar>(5.714285714285714107E-02);
    constexpr auto k35 = static_cast<Scalar>(6.060606060606060774E-02);
    constexpr auto k36 = static_cast<Scalar>(6.451612903225806273E-02);
    constexpr auto k37 = static_cast<Scalar>(6.896551724137930939E-02);
    constexpr auto k38 = static_cast<Scalar>(7.407407407407406996E-02);
    constexpr auto k39 = static_cast<Scalar>(8.000000000000000167E-02);
    constexpr auto k40 = static_cast<Scalar>(8.695652173913043237E-02);
    constexpr auto k41 = static_cast<Scalar>(9.523809523809523281E-02);
    constexpr auto k42 = static_cast<Scalar>(1.052631578947368363E-01);
    constexpr auto k43 = static_cast<Scalar>(1.176470588235294101E-01);
    constexpr auto k44 = static_cast<Scalar>(1.333333333333333315E-01);
    constexpr auto k45 = static_cast<Scalar>(1.538461538461538547E-01);
    constexpr auto k46 = static_cast<Scalar>(1.818181818181818232E-01);
    constexpr auto k47 = static_cast<Scalar>(2.222222222222222099E-01);
    constexpr auto k48 = static_cast<Scalar>(2.857142857142856984E-01);
    constexpr auto k49 = static_cast<Scalar>(4.000000000000000222E-01);
    constexpr auto k50 = static_cast<Scalar>(6.366197723675813824E-01);
    constexpr auto k51 = static_cast<Scalar>(6.666666666666666297E-01);
    constexpr auto k52 = static_cast<Scalar>(7.853981633974482790E-01);
    constexpr auto k53 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k54 = static_cast<Scalar>(2.000000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(2.256758334191025117E+00);
    constexpr auto k56 = static_cast<Scalar>(2.467401100272339498E+00);
    constexpr auto k57 = static_cast<Scalar>(4.000000000000000000E+00);
    constexpr auto k58 = static_cast<Scalar>(4.188790204786390525E+00);
    constexpr auto k59 = static_cast<Scalar>(6.000000000000000000E+00);
    constexpr auto k60 = static_cast<Scalar>(6.018022224509399720E+00);
    constexpr auto k61 = static_cast<Scalar>(6.283185307179586232E+00);
    constexpr auto k62 = static_cast<Scalar>(9.628835559215042750E+00);
    constexpr auto k63 = static_cast<Scalar>(1.315947253478581125E+01);
    constexpr auto k64 = static_cast<Scalar>(2.400000000000000000E+01);
    Real harmonics[9];
    Real compensation[9];
    Real imagePair[9];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k57) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k53);
    series0 = fma(series0, argument * broadcast<Real>(k03), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k04), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k05), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k06), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k07), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k08), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k09), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k53));
    series0 = fma(series0, argument * broadcast<Real>(k51), broadcast<Real>(k53));
    Real const radial0 = broadcast<Real>(k55) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k00);
    Real series1 = broadcast<Real>(k53);
    series1 = fma(series1, argument * broadcast<Real>(k02), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k03), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k04), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k05), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k06), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k07), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k08), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k09), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k53));
    series1 = fma(series1, argument * broadcast<Real>(k49), broadcast<Real>(k53));
    Real const radial1 = -broadcast<Real>(k60) * exponential * series1;
    output[1] = broadcast<Real>(k54) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k00);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k00);
    output[3] = broadcast<Real>(k54) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k00);
    Real series2 = broadcast<Real>(k53);
    series2 = fma(series2, argument * broadcast<Real>(k01), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k02), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k03), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k04), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k05), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k06), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k07), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k08), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k09), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k53));
    series2 = fma(series2, argument * broadcast<Real>(k48), broadcast<Real>(k53));
    Real const radial2 = broadcast<Real>(k62) * exponential * series2;
    output[4] = broadcast<Real>(k64) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k00);
    output[5] = broadcast<Real>(k59) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k00);
    output[6] = broadcast<Real>(k57) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k00);
    output[7] = broadcast<Real>(k59) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k00);
    output[8] = broadcast<Real>(k64) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k00);
    output[0] += broadcast<Real>(k52);
    greenTrace = broadcast<Real>(k58);
    for (int ix = -6; ix <= 6; ++ix) {
        for (int iy = -6; iy <= 6; ++iy) {
            for (int iz = -6; iz <= 6; ++iz) {
                int const latticeSquared = ix * ix + iy * iy + iz * iz;
                if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 || (ix == 0 && iy < 0) ||
                    (ix == 0 && iy == 0 && iz < 0)) {
                    continue;
                }
                for (int sign = -1; sign <= 1; sign += 2) {
                    Real const imageX = x + broadcast<Real>(sign * ix);
                    Real const imageY = y + broadcast<Real>(sign * iy);
                    Real const imageZ = z + broadcast<Real>(sign * iz);
                    Real const imageSquared =
                        fma(imageX, imageX, fma(imageY, imageY, imageZ * imageZ));
                    Real const imageRadius = sqrt(imageSquared);
                    Real const iR = broadcast<Real>(k53) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k57) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k54) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k55) * imageRadius * imageExp;
                    Real inversePower = iR;
                    Real const imageRadial0 = -screening * inversePower;
                    Real const imageValue0 = imageRadial0 * harmonics[0];
                    if (sign < 0) {
                        imagePair[0] = imageValue0;
                    } else {
                        imagePair[0] += imageValue0;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k51);
                    Real const imageRadial1 = screening * inversePower;
                    Real const imageValue1 = broadcast<Real>(k54) * imageRadial1 * harmonics[1];
                    if (sign < 0) {
                        imagePair[1] = imageValue1;
                    } else {
                        imagePair[1] += imageValue1;
                    }
                    Real const imageValue2 = imageRadial1 * harmonics[2];
                    if (sign < 0) {
                        imagePair[2] = imageValue2;
                    } else {
                        imagePair[2] += imageValue2;
                    }
                    Real const imageValue3 = broadcast<Real>(k54) * imageRadial1 * harmonics[3];
                    if (sign < 0) {
                        imagePair[3] = imageValue3;
                    } else {
                        imagePair[3] += imageValue3;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k64) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k59) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k57) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k59) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k64) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                }
                detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                Real const waveX = broadcast<Real>(ix);
                Real const waveY = broadcast<Real>(iy);
                Real const waveZ = broadcast<Real>(iz);
                Real const waveSquared = broadcast<Real>(latticeSquared);
                ewaldRegular<degree>(harmonics, waveX, waveY, waveZ, waveSquared);
                Real const phase = broadcast<Real>(k61) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k50) *
                    detail::ewaldExp(-broadcast<Real>(k56) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k53) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k61) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k54) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k54) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k63) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k64) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k59) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k57) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k59) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k64) * waveRadial2 * harmonics[8]);
            }
        }
    }
}
/// Evaluate the periodic correction with promoted working precision, then convert the result.
template <int degree, typename Real, typename Output>
    requires(degree == 2)
CUDA_EXPORT void evaluateEwaldGreen(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    using Working = typename detail::EwaldWorkingType<Real>::Type;
    Working temporary[9];
    Working trace;
    evaluateEwaldWorking<degree, Working>(temporary, trace, detail::ewaldConvert<Working>(x),
        detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
    for (int i = 0; i < 9; ++i) {
        output[i] = detail::ewaldConvert<Real>(temporary[i]);
    }
    greenTrace = detail::ewaldConvert<Real>(trace);
}
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 4) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 4)
template <>
inline constexpr int kEwaldGreenFlops<3> = 194963;
template <>
inline constexpr int kEwaldGreenConventionalFlops<3> = 206716;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
template <int degree, typename Real>
    requires(degree == 3)
CUDA_EXPORT void ewaldRegular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(1.111111111111111049E-01);
    constexpr auto k1 = static_cast<Scalar>(1.250000000000000000E-01);
    constexpr auto k2 = static_cast<Scalar>(1.666666666666666574E-01);
    constexpr auto k3 = static_cast<Scalar>(2.500000000000000000E-01);
    constexpr auto k4 = static_cast<Scalar>(5.000000000000000000E-01);
    constexpr auto k5 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k6 = static_cast<Scalar>(3.000000000000000000E+00);
    constexpr auto k7 = static_cast<Scalar>(5.000000000000000000E+00);
    output[0] = broadcast<Real>(k5);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k6) * z;
    output[6] = broadcast<Real>(k3) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k7) * z;
    output[12] = broadcast<Real>(k0) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    output[3] = x * broadcast<Real>(k4);
    output[1] = y * broadcast<Real>(k4);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k7) * z;
    output[13] = broadcast<Real>(k1) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k1) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    output[8] = broadcast<Real>(k3) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k3) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    output[15] = broadcast<Real>(k2) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k2) * fma(y, output[8], x * output[4]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 3)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k01 = static_cast<Scalar>(1.941747572815533812E-02);
    constexpr auto k02 = static_cast<Scalar>(1.980198019801980222E-02);
    constexpr auto k03 = static_cast<Scalar>(2.020202020202020374E-02);
    constexpr auto k04 = static_cast<Scalar>(2.061855670103092744E-02);
    constexpr auto k05 = static_cast<Scalar>(2.105263157894736795E-02);
    constexpr auto k06 = static_cast<Scalar>(2.150537634408602322E-02);
    constexpr auto k07 = static_cast<Scalar>(2.197802197802197974E-02);
    constexpr auto k08 = static_cast<Scalar>(2.247191011235954994E-02);
    constexpr auto k09 = static_cast<Scalar>(2.298850574712643646E-02);
    constexpr auto k10 = static_cast<Scalar>(2.352941176470588203E-02);
    constexpr auto k11 = static_cast<Scalar>(2.409638554216867595E-02);
    constexpr auto k12 = static_cast<Scalar>(2.469135802469135665E-02);
    constexpr auto k13 = static_cast<Scalar>(2.531645569620253125E-02);
    constexpr auto k14 = static_cast<Scalar>(2.597402597402597574E-02);
    constexpr auto k15 = static_cast<Scalar>(2.666666666666666838E-02);
    constexpr auto k16 = static_cast<Scalar>(2.739726027397260122E-02);
    constexpr auto k17 = static_cast<Scalar>(2.816901408450704289E-02);
    constexpr auto k18 = static_cast<Scalar>(2.898550724637681195E-02);
    constexpr auto k19 = static_cast<Scalar>(2.985074626865671585E-02);
    constexpr auto k20 = static_cast<Scalar>(3.076923076923077094E-02);
    constexpr auto k21 = static_cast<Scalar>(3.174603174603174427E-02);
    constexpr auto k22 = static_cast<Scalar>(3.278688524590164105E-02);
    constexpr auto k23 = static_cast<Scalar>(3.389830508474576259E-02);
    constexpr auto k24 = static_cast<Scalar>(3.508771929824561209E-02);
    constexpr auto k25 = static_cast<Scalar>(3.636363636363636187E-02);
    constexpr auto k26 = static_cast<Scalar>(3.773584905660377214E-02);
    constexpr auto k27 = static_cast<Scalar>(3.921568627450980338E-02);
    constexpr auto k28 = static_cast<Scalar>(4.081632653061224164E-02);
    constexpr auto k29 = static_cast<Scalar>(4.255319148936170109E-02);
    constexpr auto k30 = static_cast<Scalar>(4.444444444444444614E-02);
    constexpr auto k31 = static_cast<Scalar>(4.651162790697674354E-02);
    constexpr auto k32 = static_cast<Scalar>(4.878048780487805047E-02);
    constexpr auto k33 = static_cast<Scalar>(5.128205128205128027E-02);
    constexpr auto k34 = static_cast<Scalar>(5.405405405405405705E-02);
    constexpr auto k35 = static_cast<Scalar>(5.714285714285714107E-02);
    constexpr auto k36 = static_cast<Scalar>(6.060606060606060774E-02);
    constexpr auto k37 = static_cast<Scalar>(6.451612903225806273E-02);
    constexpr auto k38 = static_cast<Scalar>(6.896551724137930939E-02);
    constexpr auto k39 = static_cast<Scalar>(7.407407407407406996E-02);
    constexpr auto k40 = static_cast<Scalar>(8.000000000000000167E-02);
    constexpr auto k41 = static_cast<Scalar>(8.695652173913043237E-02);
    constexpr auto k42 = static_cast<Scalar>(9.523809523809523281E-02);
    constexpr auto k43 = static_cast<Scalar>(1.052631578947368363E-01);
    constexpr auto k44 = static_cast<Scalar>(1.176470588235294101E-01);
    constexpr auto k45 = static_cast<Scalar>(1.333333333333333315E-01);
    constexpr auto k46 = static_cast<Scalar>(1.538461538461538547E-01);
    constexpr auto k47 = static_cast<Scalar>(1.818181818181818232E-01);
    constexpr auto k48 = static_cast<Scalar>(2.222222222222222099E-01);
    constexpr auto k49 = static_cast<Scalar>(2.857142857142856984E-01);
    constexpr auto k50 = static_cast<Scalar>(4.000000000000000222E-01);
    constexpr auto k51 = static_cast<Scalar>(6.366197723675813824E-01);
    constexpr auto k52 = static_cast<Scalar>(6.666666666666666297E-01);
    constexpr auto k53 = static_cast<Scalar>(7.853981633974482790E-01);
    constexpr auto k54 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(2.000000000000000000E+00);
    constexpr auto k56 = static_cast<Scalar>(2.256758334191025117E+00);
    constexpr auto k57 = static_cast<Scalar>(2.467401100272339498E+00);
    constexpr auto k58 = static_cast<Scalar>(4.000000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(4.188790204786390525E+00);
    constexpr auto k60 = static_cast<Scalar>(6.000000000000000000E+00);
    constexpr auto k61 = static_cast<Scalar>(6.018022224509399720E+00);
    constexpr auto k62 = static_cast<Scalar>(6.283185307179586232E+00);
    constexpr auto k63 = static_cast<Scalar>(9.628835559215042750E+00);
    constexpr auto k64 = static_cast<Scalar>(1.100438349624576340E+01);
    constexpr auto k65 = static_cast<Scalar>(1.315947253478581125E+01);
    constexpr auto k66 = static_cast<Scalar>(1.653668089615990056E+01);
    constexpr auto k67 = static_cast<Scalar>(2.400000000000000000E+01);
    constexpr auto k68 = static_cast<Scalar>(3.600000000000000000E+01);
    constexpr auto k69 = static_cast<Scalar>(4.800000000000000000E+01);
    constexpr auto k70 = static_cast<Scalar>(1.200000000000000000E+02);
    constexpr auto k71 = static_cast<Scalar>(7.200000000000000000E+02);
    Real harmonics[16];
    Real compensation[16];
    Real imagePair[16];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k58) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k54);
    series0 = fma(series0, argument * broadcast<Real>(k04), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k05), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k06), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k07), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k08), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k09), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k50), broadcast<Real>(k54));
    series0 = fma(series0, argument * broadcast<Real>(k52), broadcast<Real>(k54));
    Real const radial0 = broadcast<Real>(k56) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k00);
    Real series1 = broadcast<Real>(k54);
    series1 = fma(series1, argument * broadcast<Real>(k03), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k04), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k05), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k06), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k07), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k08), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k09), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k49), broadcast<Real>(k54));
    series1 = fma(series1, argument * broadcast<Real>(k50), broadcast<Real>(k54));
    Real const radial1 = -broadcast<Real>(k61) * exponential * series1;
    output[1] = broadcast<Real>(k55) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k00);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k00);
    output[3] = broadcast<Real>(k55) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k00);
    Real series2 = broadcast<Real>(k54);
    series2 = fma(series2, argument * broadcast<Real>(k02), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k03), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k04), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k05), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k06), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k07), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k08), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k09), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k48), broadcast<Real>(k54));
    series2 = fma(series2, argument * broadcast<Real>(k49), broadcast<Real>(k54));
    Real const radial2 = broadcast<Real>(k63) * exponential * series2;
    output[4] = broadcast<Real>(k67) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k00);
    output[5] = broadcast<Real>(k60) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k00);
    output[6] = broadcast<Real>(k58) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k00);
    output[7] = broadcast<Real>(k60) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k00);
    output[8] = broadcast<Real>(k67) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k00);
    Real series3 = broadcast<Real>(k54);
    series3 = fma(series3, argument * broadcast<Real>(k01), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k02), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k03), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k04), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k05), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k06), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k07), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k08), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k09), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k47), broadcast<Real>(k54));
    series3 = fma(series3, argument * broadcast<Real>(k48), broadcast<Real>(k54));
    Real const radial3 = -broadcast<Real>(k64) * exponential * series3;
    output[9] = broadcast<Real>(k71) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k00);
    output[10] = broadcast<Real>(k70) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k00);
    output[11] = broadcast<Real>(k69) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k00);
    output[12] = broadcast<Real>(k68) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k00);
    output[13] = broadcast<Real>(k69) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k00);
    output[14] = broadcast<Real>(k70) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k00);
    output[15] = broadcast<Real>(k71) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k00);
    output[0] += broadcast<Real>(k53);
    greenTrace = broadcast<Real>(k59);
    for (int ix = -6; ix <= 6; ++ix) {
        for (int iy = -6; iy <= 6; ++iy) {
            for (int iz = -6; iz <= 6; ++iz) {
                int const latticeSquared = ix * ix + iy * iy + iz * iz;
                if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 || (ix == 0 && iy < 0) ||
                    (ix == 0 && iy == 0 && iz < 0)) {
                    continue;
                }
                for (int sign = -1; sign <= 1; sign += 2) {
                    Real const imageX = x + broadcast<Real>(sign * ix);
                    Real const imageY = y + broadcast<Real>(sign * iy);
                    Real const imageZ = z + broadcast<Real>(sign * iz);
                    Real const imageSquared =
                        fma(imageX, imageX, fma(imageY, imageY, imageZ * imageZ));
                    Real const imageRadius = sqrt(imageSquared);
                    Real const iR = broadcast<Real>(k54) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k58) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k55) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k56) * imageRadius * imageExp;
                    Real inversePower = iR;
                    Real const imageRadial0 = -screening * inversePower;
                    Real const imageValue0 = imageRadial0 * harmonics[0];
                    if (sign < 0) {
                        imagePair[0] = imageValue0;
                    } else {
                        imagePair[0] += imageValue0;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k52);
                    Real const imageRadial1 = screening * inversePower;
                    Real const imageValue1 = broadcast<Real>(k55) * imageRadial1 * harmonics[1];
                    if (sign < 0) {
                        imagePair[1] = imageValue1;
                    } else {
                        imagePair[1] += imageValue1;
                    }
                    Real const imageValue2 = imageRadial1 * harmonics[2];
                    if (sign < 0) {
                        imagePair[2] = imageValue2;
                    } else {
                        imagePair[2] += imageValue2;
                    }
                    Real const imageValue3 = broadcast<Real>(k55) * imageRadial1 * harmonics[3];
                    if (sign < 0) {
                        imagePair[3] = imageValue3;
                    } else {
                        imagePair[3] += imageValue3;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k50);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k67) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k60) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k58) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k60) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k67) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k71) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k70) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k69) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k68) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k69) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k70) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k71) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                }
                detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                Real const waveX = broadcast<Real>(ix);
                Real const waveY = broadcast<Real>(iy);
                Real const waveZ = broadcast<Real>(iz);
                Real const waveSquared = broadcast<Real>(latticeSquared);
                ewaldRegular<degree>(harmonics, waveX, waveY, waveZ, waveSquared);
                Real const phase = broadcast<Real>(k62) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k51) *
                    detail::ewaldExp(-broadcast<Real>(k57) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k54) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k62) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k55) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k55) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k65) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k67) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k60) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k58) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k60) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k67) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k66) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k71) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k70) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k69) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k68) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k69) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k70) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k71) * waveRadial3 * harmonics[15]);
            }
        }
    }
}
/// Evaluate the periodic correction with promoted working precision, then convert the result.
template <int degree, typename Real, typename Output>
    requires(degree == 3)
CUDA_EXPORT void evaluateEwaldGreen(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    using Working = typename detail::EwaldWorkingType<Real>::Type;
    Working temporary[16];
    Working trace;
    evaluateEwaldWorking<degree, Working>(temporary, trace, detail::ewaldConvert<Working>(x),
        detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
    for (int i = 0; i < 16; ++i) {
        output[i] = detail::ewaldConvert<Real>(temporary[i]);
    }
    greenTrace = detail::ewaldConvert<Real>(trace);
}
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 5) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 5)
template <>
inline constexpr int kEwaldGreenFlops<4> = 299055;
template <>
inline constexpr int kEwaldGreenConventionalFlops<4> = 320565;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
template <int degree, typename Real>
    requires(degree == 4)
CUDA_EXPORT void ewaldRegular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(6.250000000000000000E-02);
    constexpr auto k01 = static_cast<Scalar>(6.666666666666666574E-02);
    constexpr auto k02 = static_cast<Scalar>(8.333333333333332871E-02);
    constexpr auto k03 = static_cast<Scalar>(1.111111111111111049E-01);
    constexpr auto k04 = static_cast<Scalar>(1.250000000000000000E-01);
    constexpr auto k05 = static_cast<Scalar>(1.666666666666666574E-01);
    constexpr auto k06 = static_cast<Scalar>(2.500000000000000000E-01);
    constexpr auto k07 = static_cast<Scalar>(5.000000000000000000E-01);
    constexpr auto k08 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k09 = static_cast<Scalar>(3.000000000000000000E+00);
    constexpr auto k10 = static_cast<Scalar>(5.000000000000000000E+00);
    constexpr auto k11 = static_cast<Scalar>(7.000000000000000000E+00);
    output[0] = broadcast<Real>(k08);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k09) * z;
    output[6] = broadcast<Real>(k06) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k10) * z;
    output[12] = broadcast<Real>(k03) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k11) * z;
    output[20] = broadcast<Real>(k00) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    output[3] = x * broadcast<Real>(k07);
    output[1] = y * broadcast<Real>(k07);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k10) * z;
    output[13] = broadcast<Real>(k04) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k04) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k11) * z;
    output[21] = broadcast<Real>(k01) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k01) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    output[8] = broadcast<Real>(k06) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k06) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k11) * z;
    output[22] = broadcast<Real>(k02) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k02) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    output[15] = broadcast<Real>(k05) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k05) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    output[24] = broadcast<Real>(k04) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k04) * fma(y, output[15], x * output[9]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 4)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k01 = static_cast<Scalar>(1.904761904761904934E-02);
    constexpr auto k02 = static_cast<Scalar>(1.941747572815533812E-02);
    constexpr auto k03 = static_cast<Scalar>(1.980198019801980222E-02);
    constexpr auto k04 = static_cast<Scalar>(2.020202020202020374E-02);
    constexpr auto k05 = static_cast<Scalar>(2.061855670103092744E-02);
    constexpr auto k06 = static_cast<Scalar>(2.105263157894736795E-02);
    constexpr auto k07 = static_cast<Scalar>(2.150537634408602322E-02);
    constexpr auto k08 = static_cast<Scalar>(2.197802197802197974E-02);
    constexpr auto k09 = static_cast<Scalar>(2.247191011235954994E-02);
    constexpr auto k10 = static_cast<Scalar>(2.298850574712643646E-02);
    constexpr auto k11 = static_cast<Scalar>(2.352941176470588203E-02);
    constexpr auto k12 = static_cast<Scalar>(2.409638554216867595E-02);
    constexpr auto k13 = static_cast<Scalar>(2.469135802469135665E-02);
    constexpr auto k14 = static_cast<Scalar>(2.531645569620253125E-02);
    constexpr auto k15 = static_cast<Scalar>(2.597402597402597574E-02);
    constexpr auto k16 = static_cast<Scalar>(2.666666666666666838E-02);
    constexpr auto k17 = static_cast<Scalar>(2.739726027397260122E-02);
    constexpr auto k18 = static_cast<Scalar>(2.816901408450704289E-02);
    constexpr auto k19 = static_cast<Scalar>(2.898550724637681195E-02);
    constexpr auto k20 = static_cast<Scalar>(2.985074626865671585E-02);
    constexpr auto k21 = static_cast<Scalar>(3.076923076923077094E-02);
    constexpr auto k22 = static_cast<Scalar>(3.174603174603174427E-02);
    constexpr auto k23 = static_cast<Scalar>(3.278688524590164105E-02);
    constexpr auto k24 = static_cast<Scalar>(3.389830508474576259E-02);
    constexpr auto k25 = static_cast<Scalar>(3.508771929824561209E-02);
    constexpr auto k26 = static_cast<Scalar>(3.636363636363636187E-02);
    constexpr auto k27 = static_cast<Scalar>(3.773584905660377214E-02);
    constexpr auto k28 = static_cast<Scalar>(3.921568627450980338E-02);
    constexpr auto k29 = static_cast<Scalar>(4.081632653061224164E-02);
    constexpr auto k30 = static_cast<Scalar>(4.255319148936170109E-02);
    constexpr auto k31 = static_cast<Scalar>(4.444444444444444614E-02);
    constexpr auto k32 = static_cast<Scalar>(4.651162790697674354E-02);
    constexpr auto k33 = static_cast<Scalar>(4.878048780487805047E-02);
    constexpr auto k34 = static_cast<Scalar>(5.128205128205128027E-02);
    constexpr auto k35 = static_cast<Scalar>(5.405405405405405705E-02);
    constexpr auto k36 = static_cast<Scalar>(5.714285714285714107E-02);
    constexpr auto k37 = static_cast<Scalar>(6.060606060606060774E-02);
    constexpr auto k38 = static_cast<Scalar>(6.451612903225806273E-02);
    constexpr auto k39 = static_cast<Scalar>(6.896551724137930939E-02);
    constexpr auto k40 = static_cast<Scalar>(7.407407407407406996E-02);
    constexpr auto k41 = static_cast<Scalar>(8.000000000000000167E-02);
    constexpr auto k42 = static_cast<Scalar>(8.695652173913043237E-02);
    constexpr auto k43 = static_cast<Scalar>(9.523809523809523281E-02);
    constexpr auto k44 = static_cast<Scalar>(1.052631578947368363E-01);
    constexpr auto k45 = static_cast<Scalar>(1.176470588235294101E-01);
    constexpr auto k46 = static_cast<Scalar>(1.333333333333333315E-01);
    constexpr auto k47 = static_cast<Scalar>(1.538461538461538547E-01);
    constexpr auto k48 = static_cast<Scalar>(1.818181818181818232E-01);
    constexpr auto k49 = static_cast<Scalar>(2.222222222222222099E-01);
    constexpr auto k50 = static_cast<Scalar>(2.857142857142856984E-01);
    constexpr auto k51 = static_cast<Scalar>(4.000000000000000222E-01);
    constexpr auto k52 = static_cast<Scalar>(6.366197723675813824E-01);
    constexpr auto k53 = static_cast<Scalar>(6.666666666666666297E-01);
    constexpr auto k54 = static_cast<Scalar>(7.853981633974482790E-01);
    constexpr auto k55 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k56 = static_cast<Scalar>(2.000000000000000000E+00);
    constexpr auto k57 = static_cast<Scalar>(2.256758334191025117E+00);
    constexpr auto k58 = static_cast<Scalar>(2.467401100272339498E+00);
    constexpr auto k59 = static_cast<Scalar>(4.000000000000000000E+00);
    constexpr auto k60 = static_cast<Scalar>(4.188790204786390525E+00);
    constexpr auto k61 = static_cast<Scalar>(6.000000000000000000E+00);
    constexpr auto k62 = static_cast<Scalar>(6.018022224509399720E+00);
    constexpr auto k63 = static_cast<Scalar>(6.283185307179586232E+00);
    constexpr auto k64 = static_cast<Scalar>(9.628835559215042750E+00);
    constexpr auto k65 = static_cast<Scalar>(9.781674218885120453E+00);
    constexpr auto k66 = static_cast<Scalar>(1.100438349624576340E+01);
    constexpr auto k67 = static_cast<Scalar>(1.315947253478581125E+01);
    constexpr auto k68 = static_cast<Scalar>(1.484329006232417747E+01);
    constexpr auto k69 = static_cast<Scalar>(1.653668089615990056E+01);
    constexpr auto k70 = static_cast<Scalar>(2.400000000000000000E+01);
    constexpr auto k71 = static_cast<Scalar>(3.600000000000000000E+01);
    constexpr auto k72 = static_cast<Scalar>(4.800000000000000000E+01);
    constexpr auto k73 = static_cast<Scalar>(1.200000000000000000E+02);
    constexpr auto k74 = static_cast<Scalar>(5.760000000000000000E+02);
    constexpr auto k75 = static_cast<Scalar>(7.200000000000000000E+02);
    constexpr auto k76 = static_cast<Scalar>(1.440000000000000000E+03);
    constexpr auto k77 = static_cast<Scalar>(5.040000000000000000E+03);
    constexpr auto k78 = static_cast<Scalar>(4.032000000000000000E+04);
    Real harmonics[25];
    Real compensation[25];
    Real imagePair[25];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k59) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k55);
    series0 = fma(series0, argument * broadcast<Real>(k05), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k06), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k07), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k08), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k09), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k50), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k51), broadcast<Real>(k55));
    series0 = fma(series0, argument * broadcast<Real>(k53), broadcast<Real>(k55));
    Real const radial0 = broadcast<Real>(k57) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k00);
    Real series1 = broadcast<Real>(k55);
    series1 = fma(series1, argument * broadcast<Real>(k04), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k05), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k06), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k07), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k08), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k09), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k49), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k50), broadcast<Real>(k55));
    series1 = fma(series1, argument * broadcast<Real>(k51), broadcast<Real>(k55));
    Real const radial1 = -broadcast<Real>(k62) * exponential * series1;
    output[1] = broadcast<Real>(k56) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k00);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k00);
    output[3] = broadcast<Real>(k56) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k00);
    Real series2 = broadcast<Real>(k55);
    series2 = fma(series2, argument * broadcast<Real>(k03), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k04), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k05), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k06), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k07), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k08), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k09), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k48), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k49), broadcast<Real>(k55));
    series2 = fma(series2, argument * broadcast<Real>(k50), broadcast<Real>(k55));
    Real const radial2 = broadcast<Real>(k64) * exponential * series2;
    output[4] = broadcast<Real>(k70) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k00);
    output[5] = broadcast<Real>(k61) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k00);
    output[6] = broadcast<Real>(k59) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k00);
    output[7] = broadcast<Real>(k61) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k00);
    output[8] = broadcast<Real>(k70) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k00);
    Real series3 = broadcast<Real>(k55);
    series3 = fma(series3, argument * broadcast<Real>(k02), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k03), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k04), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k05), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k06), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k07), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k08), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k09), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k47), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k48), broadcast<Real>(k55));
    series3 = fma(series3, argument * broadcast<Real>(k49), broadcast<Real>(k55));
    Real const radial3 = -broadcast<Real>(k66) * exponential * series3;
    output[9] = broadcast<Real>(k75) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k00);
    output[10] = broadcast<Real>(k73) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k00);
    output[11] = broadcast<Real>(k72) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k00);
    output[12] = broadcast<Real>(k71) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k00);
    output[13] = broadcast<Real>(k72) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k00);
    output[14] = broadcast<Real>(k73) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k00);
    output[15] = broadcast<Real>(k75) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k00);
    Real series4 = broadcast<Real>(k55);
    series4 = fma(series4, argument * broadcast<Real>(k01), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k02), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k03), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k04), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k05), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k06), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k07), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k08), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k09), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k10), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k11), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k12), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k13), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k14), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k15), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k16), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k17), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k18), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k19), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k20), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k21), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k22), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k23), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k24), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k25), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k26), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k27), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k28), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k29), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k30), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k31), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k32), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k33), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k34), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k35), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k36), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k37), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k38), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k39), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k40), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k41), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k42), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k43), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k44), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k45), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k46), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k47), broadcast<Real>(k55));
    series4 = fma(series4, argument * broadcast<Real>(k48), broadcast<Real>(k55));
    Real const radial4 = broadcast<Real>(k65) * exponential * series4;
    output[16] = broadcast<Real>(k78) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k00);
    output[17] = broadcast<Real>(k77) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k00);
    output[18] = broadcast<Real>(k76) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k00);
    output[19] = broadcast<Real>(k75) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k00);
    output[20] = broadcast<Real>(k74) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k00);
    output[21] = broadcast<Real>(k75) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k00);
    output[22] = broadcast<Real>(k76) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k00);
    output[23] = broadcast<Real>(k77) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k00);
    output[24] = broadcast<Real>(k78) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k00);
    output[0] += broadcast<Real>(k54);
    greenTrace = broadcast<Real>(k60);
    for (int ix = -6; ix <= 6; ++ix) {
        for (int iy = -6; iy <= 6; ++iy) {
            for (int iz = -6; iz <= 6; ++iz) {
                int const latticeSquared = ix * ix + iy * iy + iz * iz;
                if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 || (ix == 0 && iy < 0) ||
                    (ix == 0 && iy == 0 && iz < 0)) {
                    continue;
                }
                for (int sign = -1; sign <= 1; sign += 2) {
                    Real const imageX = x + broadcast<Real>(sign * ix);
                    Real const imageY = y + broadcast<Real>(sign * iy);
                    Real const imageZ = z + broadcast<Real>(sign * iz);
                    Real const imageSquared =
                        fma(imageX, imageX, fma(imageY, imageY, imageZ * imageZ));
                    Real const imageRadius = sqrt(imageSquared);
                    Real const iR = broadcast<Real>(k55) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k59) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k56) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k57) * imageRadius * imageExp;
                    Real inversePower = iR;
                    Real const imageRadial0 = -screening * inversePower;
                    Real const imageValue0 = imageRadial0 * harmonics[0];
                    if (sign < 0) {
                        imagePair[0] = imageValue0;
                    } else {
                        imagePair[0] += imageValue0;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k53);
                    Real const imageRadial1 = screening * inversePower;
                    Real const imageValue1 = broadcast<Real>(k56) * imageRadial1 * harmonics[1];
                    if (sign < 0) {
                        imagePair[1] = imageValue1;
                    } else {
                        imagePair[1] += imageValue1;
                    }
                    Real const imageValue2 = imageRadial1 * harmonics[2];
                    if (sign < 0) {
                        imagePair[2] = imageValue2;
                    } else {
                        imagePair[2] += imageValue2;
                    }
                    Real const imageValue3 = broadcast<Real>(k56) * imageRadial1 * harmonics[3];
                    if (sign < 0) {
                        imagePair[3] = imageValue3;
                    } else {
                        imagePair[3] += imageValue3;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k51);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k70) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k61) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k59) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k61) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k70) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k50);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k75) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k73) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k72) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k71) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k72) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k73) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k75) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k78) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k77) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k76) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k75) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k74) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k75) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k76) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k77) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k78) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                }
                detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                Real const waveX = broadcast<Real>(ix);
                Real const waveY = broadcast<Real>(iy);
                Real const waveZ = broadcast<Real>(iz);
                Real const waveSquared = broadcast<Real>(latticeSquared);
                ewaldRegular<degree>(harmonics, waveX, waveY, waveZ, waveSquared);
                Real const phase = broadcast<Real>(k63) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k52) *
                    detail::ewaldExp(-broadcast<Real>(k58) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k55) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k63) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k56) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k56) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k67) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k70) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k61) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k59) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k61) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k70) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k69) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k75) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k73) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k72) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k71) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k72) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k73) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k75) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k68) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k78) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k77) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k76) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k75) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k74) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k75) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k76) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k77) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k78) * waveRadial4 * harmonics[24]);
            }
        }
    }
}
/// Evaluate the periodic correction with promoted working precision, then convert the result.
template <int degree, typename Real, typename Output>
    requires(degree == 4)
CUDA_EXPORT void evaluateEwaldGreen(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    using Working = typename detail::EwaldWorkingType<Real>::Type;
    Working temporary[25];
    Working trace;
    evaluateEwaldWorking<degree, Working>(temporary, trace, detail::ewaldConvert<Working>(x),
        detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
    for (int i = 0; i < 25; ++i) {
        output[i] = detail::ewaldConvert<Real>(temporary[i]);
    }
    greenTrace = detail::ewaldConvert<Real>(trace);
}
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 6) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 6)
template <>
inline constexpr int kEwaldGreenFlops<5> = 426720;
template <>
inline constexpr int kEwaldGreenConventionalFlops<5> = 460761;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
template <int degree, typename Real>
    requires(degree == 5)
CUDA_EXPORT void ewaldRegular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(4.000000000000000083E-02);
    constexpr auto k01 = static_cast<Scalar>(4.166666666666666435E-02);
    constexpr auto k02 = static_cast<Scalar>(4.761904761904761640E-02);
    constexpr auto k03 = static_cast<Scalar>(6.250000000000000000E-02);
    constexpr auto k04 = static_cast<Scalar>(6.666666666666666574E-02);
    constexpr auto k05 = static_cast<Scalar>(8.333333333333332871E-02);
    constexpr auto k06 = static_cast<Scalar>(1.000000000000000056E-01);
    constexpr auto k07 = static_cast<Scalar>(1.111111111111111049E-01);
    constexpr auto k08 = static_cast<Scalar>(1.250000000000000000E-01);
    constexpr auto k09 = static_cast<Scalar>(1.666666666666666574E-01);
    constexpr auto k10 = static_cast<Scalar>(2.500000000000000000E-01);
    constexpr auto k11 = static_cast<Scalar>(5.000000000000000000E-01);
    constexpr auto k12 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k13 = static_cast<Scalar>(3.000000000000000000E+00);
    constexpr auto k14 = static_cast<Scalar>(5.000000000000000000E+00);
    constexpr auto k15 = static_cast<Scalar>(7.000000000000000000E+00);
    constexpr auto k16 = static_cast<Scalar>(9.000000000000000000E+00);
    output[0] = broadcast<Real>(k12);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k13) * z;
    output[6] = broadcast<Real>(k10) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k14) * z;
    output[12] = broadcast<Real>(k07) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k15) * z;
    output[20] = broadcast<Real>(k03) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    Real const zFactor5_0 = broadcast<Real>(k16) * z;
    output[30] = broadcast<Real>(k00) * fma(zFactor5_0, output[20], -radiusSquared * output[12]);
    output[3] = x * broadcast<Real>(k11);
    output[1] = y * broadcast<Real>(k11);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k14) * z;
    output[13] = broadcast<Real>(k08) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k08) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k15) * z;
    output[21] = broadcast<Real>(k04) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k04) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    Real const zFactor5_1 = broadcast<Real>(k16) * z;
    output[31] = broadcast<Real>(k01) * fma(zFactor5_1, output[21], -radiusSquared * output[13]);
    output[29] = broadcast<Real>(k01) * fma(zFactor5_1, output[19], -radiusSquared * output[11]);
    output[8] = broadcast<Real>(k10) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k10) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k15) * z;
    output[22] = broadcast<Real>(k05) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k05) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    Real const zFactor5_2 = broadcast<Real>(k16) * z;
    output[32] = broadcast<Real>(k02) * fma(zFactor5_2, output[22], -radiusSquared * output[14]);
    output[28] = broadcast<Real>(k02) * fma(zFactor5_2, output[18], -radiusSquared * output[10]);
    output[15] = broadcast<Real>(k09) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k09) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    Real const zFactor5_3 = broadcast<Real>(k16) * z;
    output[33] = broadcast<Real>(k03) * fma(zFactor5_3, output[23], -radiusSquared * output[15]);
    output[27] = broadcast<Real>(k03) * fma(zFactor5_3, output[17], -radiusSquared * output[9]);
    output[24] = broadcast<Real>(k08) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k08) * fma(y, output[15], x * output[9]);
    output[34] = z * output[24];
    output[26] = z * output[16];
    output[35] = broadcast<Real>(k06) * fma(x, output[24], -y * output[16]);
    output[25] = broadcast<Real>(k06) * fma(y, output[24], x * output[16]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 5)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k01 = static_cast<Scalar>(1.869158878504672758E-02);
    constexpr auto k02 = static_cast<Scalar>(1.904761904761904934E-02);
    constexpr auto k03 = static_cast<Scalar>(1.941747572815533812E-02);
    constexpr auto k04 = static_cast<Scalar>(1.980198019801980222E-02);
    constexpr auto k05 = static_cast<Scalar>(2.020202020202020374E-02);
    constexpr auto k06 = static_cast<Scalar>(2.061855670103092744E-02);
    constexpr auto k07 = static_cast<Scalar>(2.105263157894736795E-02);
    constexpr auto k08 = static_cast<Scalar>(2.150537634408602322E-02);
    constexpr auto k09 = static_cast<Scalar>(2.197802197802197974E-02);
    constexpr auto k10 = static_cast<Scalar>(2.247191011235954994E-02);
    constexpr auto k11 = static_cast<Scalar>(2.298850574712643646E-02);
    constexpr auto k12 = static_cast<Scalar>(2.352941176470588203E-02);
    constexpr auto k13 = static_cast<Scalar>(2.409638554216867595E-02);
    constexpr auto k14 = static_cast<Scalar>(2.469135802469135665E-02);
    constexpr auto k15 = static_cast<Scalar>(2.531645569620253125E-02);
    constexpr auto k16 = static_cast<Scalar>(2.597402597402597574E-02);
    constexpr auto k17 = static_cast<Scalar>(2.666666666666666838E-02);
    constexpr auto k18 = static_cast<Scalar>(2.739726027397260122E-02);
    constexpr auto k19 = static_cast<Scalar>(2.816901408450704289E-02);
    constexpr auto k20 = static_cast<Scalar>(2.898550724637681195E-02);
    constexpr auto k21 = static_cast<Scalar>(2.985074626865671585E-02);
    constexpr auto k22 = static_cast<Scalar>(3.076923076923077094E-02);
    constexpr auto k23 = static_cast<Scalar>(3.174603174603174427E-02);
    constexpr auto k24 = static_cast<Scalar>(3.278688524590164105E-02);
    constexpr auto k25 = static_cast<Scalar>(3.389830508474576259E-02);
    constexpr auto k26 = static_cast<Scalar>(3.508771929824561209E-02);
    constexpr auto k27 = static_cast<Scalar>(3.636363636363636187E-02);
    constexpr auto k28 = static_cast<Scalar>(3.773584905660377214E-02);
    constexpr auto k29 = static_cast<Scalar>(3.921568627450980338E-02);
    constexpr auto k30 = static_cast<Scalar>(4.081632653061224164E-02);
    constexpr auto k31 = static_cast<Scalar>(4.255319148936170109E-02);
    constexpr auto k32 = static_cast<Scalar>(4.444444444444444614E-02);
    constexpr auto k33 = static_cast<Scalar>(4.651162790697674354E-02);
    constexpr auto k34 = static_cast<Scalar>(4.878048780487805047E-02);
    constexpr auto k35 = static_cast<Scalar>(5.128205128205128027E-02);
    constexpr auto k36 = static_cast<Scalar>(5.405405405405405705E-02);
    constexpr auto k37 = static_cast<Scalar>(5.714285714285714107E-02);
    constexpr auto k38 = static_cast<Scalar>(6.060606060606060774E-02);
    constexpr auto k39 = static_cast<Scalar>(6.451612903225806273E-02);
    constexpr auto k40 = static_cast<Scalar>(6.896551724137930939E-02);
    constexpr auto k41 = static_cast<Scalar>(7.407407407407406996E-02);
    constexpr auto k42 = static_cast<Scalar>(8.000000000000000167E-02);
    constexpr auto k43 = static_cast<Scalar>(8.695652173913043237E-02);
    constexpr auto k44 = static_cast<Scalar>(9.523809523809523281E-02);
    constexpr auto k45 = static_cast<Scalar>(1.052631578947368363E-01);
    constexpr auto k46 = static_cast<Scalar>(1.176470588235294101E-01);
    constexpr auto k47 = static_cast<Scalar>(1.333333333333333315E-01);
    constexpr auto k48 = static_cast<Scalar>(1.538461538461538547E-01);
    constexpr auto k49 = static_cast<Scalar>(1.818181818181818232E-01);
    constexpr auto k50 = static_cast<Scalar>(2.222222222222222099E-01);
    constexpr auto k51 = static_cast<Scalar>(2.857142857142856984E-01);
    constexpr auto k52 = static_cast<Scalar>(4.000000000000000222E-01);
    constexpr auto k53 = static_cast<Scalar>(6.366197723675813824E-01);
    constexpr auto k54 = static_cast<Scalar>(6.666666666666666297E-01);
    constexpr auto k55 = static_cast<Scalar>(7.853981633974482790E-01);
    constexpr auto k56 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k57 = static_cast<Scalar>(2.000000000000000000E+00);
    constexpr auto k58 = static_cast<Scalar>(2.256758334191025117E+00);
    constexpr auto k59 = static_cast<Scalar>(2.467401100272339498E+00);
    constexpr auto k60 = static_cast<Scalar>(4.000000000000000000E+00);
    constexpr auto k61 = static_cast<Scalar>(4.188790204786390525E+00);
    constexpr auto k62 = static_cast<Scalar>(6.000000000000000000E+00);
    constexpr auto k63 = static_cast<Scalar>(6.018022224509399720E+00);
    constexpr auto k64 = static_cast<Scalar>(6.283185307179586232E+00);
    constexpr auto k65 = static_cast<Scalar>(7.113944886461904815E+00);
    constexpr auto k66 = static_cast<Scalar>(9.628835559215042750E+00);
    constexpr auto k67 = static_cast<Scalar>(9.781674218885120453E+00);
    constexpr auto k68 = static_cast<Scalar>(1.036257133664445007E+01);
    constexpr auto k69 = static_cast<Scalar>(1.100438349624576340E+01);
    constexpr auto k70 = static_cast<Scalar>(1.315947253478581125E+01);
    constexpr auto k71 = static_cast<Scalar>(1.484329006232417747E+01);
    constexpr auto k72 = static_cast<Scalar>(1.653668089615990056E+01);
    constexpr auto k73 = static_cast<Scalar>(2.400000000000000000E+01);
    constexpr auto k74 = static_cast<Scalar>(3.600000000000000000E+01);
    constexpr auto k75 = static_cast<Scalar>(4.800000000000000000E+01);
    constexpr auto k76 = static_cast<Scalar>(1.200000000000000000E+02);
    constexpr auto k77 = static_cast<Scalar>(5.760000000000000000E+02);
    constexpr auto k78 = static_cast<Scalar>(7.200000000000000000E+02);
    constexpr auto k79 = static_cast<Scalar>(1.440000000000000000E+03);
    constexpr auto k80 = static_cast<Scalar>(5.040000000000000000E+03);
    constexpr auto k81 = static_cast<Scalar>(1.440000000000000000E+04);
    constexpr auto k82 = static_cast<Scalar>(1.728000000000000000E+04);
    constexpr auto k83 = static_cast<Scalar>(3.024000000000000000E+04);
    constexpr auto k84 = static_cast<Scalar>(4.032000000000000000E+04);
    constexpr auto k85 = static_cast<Scalar>(8.064000000000000000E+04);
    constexpr auto k86 = static_cast<Scalar>(3.628800000000000000E+05);
    constexpr auto k87 = static_cast<Scalar>(3.628800000000000000E+06);
    Real harmonics[36];
    Real compensation[36];
    Real imagePair[36];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k60) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k56);
    series0 = fma(series0, argument * broadcast<Real>(k06), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k07), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k08), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k09), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k50), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k51), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k52), broadcast<Real>(k56));
    series0 = fma(series0, argument * broadcast<Real>(k54), broadcast<Real>(k56));
    Real const radial0 = broadcast<Real>(k58) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k00);
    Real series1 = broadcast<Real>(k56);
    series1 = fma(series1, argument * broadcast<Real>(k05), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k06), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k07), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k08), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k09), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k49), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k50), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k51), broadcast<Real>(k56));
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k56));
    Real const radial1 = -broadcast<Real>(k63) * exponential * series1;
    output[1] = broadcast<Real>(k57) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k00);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k00);
    output[3] = broadcast<Real>(k57) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k00);
    Real series2 = broadcast<Real>(k56);
    series2 = fma(series2, argument * broadcast<Real>(k04), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k05), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k06), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k07), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k08), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k09), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k48), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k49), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k50), broadcast<Real>(k56));
    series2 = fma(series2, argument * broadcast<Real>(k51), broadcast<Real>(k56));
    Real const radial2 = broadcast<Real>(k66) * exponential * series2;
    output[4] = broadcast<Real>(k73) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k00);
    output[5] = broadcast<Real>(k62) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k00);
    output[6] = broadcast<Real>(k60) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k00);
    output[7] = broadcast<Real>(k62) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k00);
    output[8] = broadcast<Real>(k73) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k00);
    Real series3 = broadcast<Real>(k56);
    series3 = fma(series3, argument * broadcast<Real>(k03), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k04), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k05), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k06), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k07), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k08), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k09), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k47), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k48), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k49), broadcast<Real>(k56));
    series3 = fma(series3, argument * broadcast<Real>(k50), broadcast<Real>(k56));
    Real const radial3 = -broadcast<Real>(k69) * exponential * series3;
    output[9] = broadcast<Real>(k78) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k00);
    output[10] = broadcast<Real>(k76) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k00);
    output[11] = broadcast<Real>(k75) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k00);
    output[12] = broadcast<Real>(k74) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k00);
    output[13] = broadcast<Real>(k75) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k00);
    output[14] = broadcast<Real>(k76) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k00);
    output[15] = broadcast<Real>(k78) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k00);
    Real series4 = broadcast<Real>(k56);
    series4 = fma(series4, argument * broadcast<Real>(k02), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k03), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k04), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k05), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k06), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k07), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k08), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k09), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k10), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k11), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k12), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k13), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k14), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k15), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k16), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k17), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k18), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k19), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k20), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k21), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k22), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k23), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k24), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k25), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k26), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k27), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k28), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k29), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k30), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k31), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k32), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k33), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k34), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k35), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k36), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k37), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k38), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k39), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k40), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k41), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k42), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k43), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k44), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k45), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k46), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k47), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k48), broadcast<Real>(k56));
    series4 = fma(series4, argument * broadcast<Real>(k49), broadcast<Real>(k56));
    Real const radial4 = broadcast<Real>(k67) * exponential * series4;
    output[16] = broadcast<Real>(k84) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k00);
    output[17] = broadcast<Real>(k80) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k00);
    output[18] = broadcast<Real>(k79) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k00);
    output[19] = broadcast<Real>(k78) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k00);
    output[20] = broadcast<Real>(k77) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k00);
    output[21] = broadcast<Real>(k78) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k00);
    output[22] = broadcast<Real>(k79) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k00);
    output[23] = broadcast<Real>(k80) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k00);
    output[24] = broadcast<Real>(k84) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k00);
    Real series5 = broadcast<Real>(k56);
    series5 = fma(series5, argument * broadcast<Real>(k01), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k02), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k03), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k04), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k05), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k06), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k07), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k08), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k09), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k10), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k11), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k12), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k13), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k14), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k15), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k16), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k17), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k18), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k19), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k20), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k21), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k22), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k23), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k24), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k25), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k26), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k27), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k28), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k29), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k30), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k31), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k32), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k33), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k34), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k35), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k36), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k37), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k38), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k39), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k40), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k41), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k42), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k43), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k44), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k45), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k46), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k47), broadcast<Real>(k56));
    series5 = fma(series5, argument * broadcast<Real>(k48), broadcast<Real>(k56));
    Real const radial5 = -broadcast<Real>(k65) * exponential * series5;
    output[25] = broadcast<Real>(k87) * radial5 * harmonics[25];
    compensation[25] = broadcast<Real>(k00);
    output[26] = broadcast<Real>(k86) * radial5 * harmonics[26];
    compensation[26] = broadcast<Real>(k00);
    output[27] = broadcast<Real>(k85) * radial5 * harmonics[27];
    compensation[27] = broadcast<Real>(k00);
    output[28] = broadcast<Real>(k83) * radial5 * harmonics[28];
    compensation[28] = broadcast<Real>(k00);
    output[29] = broadcast<Real>(k82) * radial5 * harmonics[29];
    compensation[29] = broadcast<Real>(k00);
    output[30] = broadcast<Real>(k81) * radial5 * harmonics[30];
    compensation[30] = broadcast<Real>(k00);
    output[31] = broadcast<Real>(k82) * radial5 * harmonics[31];
    compensation[31] = broadcast<Real>(k00);
    output[32] = broadcast<Real>(k83) * radial5 * harmonics[32];
    compensation[32] = broadcast<Real>(k00);
    output[33] = broadcast<Real>(k85) * radial5 * harmonics[33];
    compensation[33] = broadcast<Real>(k00);
    output[34] = broadcast<Real>(k86) * radial5 * harmonics[34];
    compensation[34] = broadcast<Real>(k00);
    output[35] = broadcast<Real>(k87) * radial5 * harmonics[35];
    compensation[35] = broadcast<Real>(k00);
    output[0] += broadcast<Real>(k55);
    greenTrace = broadcast<Real>(k61);
    for (int ix = -6; ix <= 6; ++ix) {
        for (int iy = -6; iy <= 6; ++iy) {
            for (int iz = -6; iz <= 6; ++iz) {
                int const latticeSquared = ix * ix + iy * iy + iz * iz;
                if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 || (ix == 0 && iy < 0) ||
                    (ix == 0 && iy == 0 && iz < 0)) {
                    continue;
                }
                for (int sign = -1; sign <= 1; sign += 2) {
                    Real const imageX = x + broadcast<Real>(sign * ix);
                    Real const imageY = y + broadcast<Real>(sign * iy);
                    Real const imageZ = z + broadcast<Real>(sign * iz);
                    Real const imageSquared =
                        fma(imageX, imageX, fma(imageY, imageY, imageZ * imageZ));
                    Real const imageRadius = sqrt(imageSquared);
                    Real const iR = broadcast<Real>(k56) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k60) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k57) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k58) * imageRadius * imageExp;
                    Real inversePower = iR;
                    Real const imageRadial0 = -screening * inversePower;
                    Real const imageValue0 = imageRadial0 * harmonics[0];
                    if (sign < 0) {
                        imagePair[0] = imageValue0;
                    } else {
                        imagePair[0] += imageValue0;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k54);
                    Real const imageRadial1 = screening * inversePower;
                    Real const imageValue1 = broadcast<Real>(k57) * imageRadial1 * harmonics[1];
                    if (sign < 0) {
                        imagePair[1] = imageValue1;
                    } else {
                        imagePair[1] += imageValue1;
                    }
                    Real const imageValue2 = imageRadial1 * harmonics[2];
                    if (sign < 0) {
                        imagePair[2] = imageValue2;
                    } else {
                        imagePair[2] += imageValue2;
                    }
                    Real const imageValue3 = broadcast<Real>(k57) * imageRadial1 * harmonics[3];
                    if (sign < 0) {
                        imagePair[3] = imageValue3;
                    } else {
                        imagePair[3] += imageValue3;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k52);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k73) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k62) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k60) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k62) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k73) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k51);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k78) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k76) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k75) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k74) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k75) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k76) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k78) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k50);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k84) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k80) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k79) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k78) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k77) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k78) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k79) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k80) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k84) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k87) * imageRadial5 * harmonics[25];
                    if (sign < 0) {
                        imagePair[25] = imageValue25;
                    } else {
                        imagePair[25] += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k86) * imageRadial5 * harmonics[26];
                    if (sign < 0) {
                        imagePair[26] = imageValue26;
                    } else {
                        imagePair[26] += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k85) * imageRadial5 * harmonics[27];
                    if (sign < 0) {
                        imagePair[27] = imageValue27;
                    } else {
                        imagePair[27] += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k83) * imageRadial5 * harmonics[28];
                    if (sign < 0) {
                        imagePair[28] = imageValue28;
                    } else {
                        imagePair[28] += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k82) * imageRadial5 * harmonics[29];
                    if (sign < 0) {
                        imagePair[29] = imageValue29;
                    } else {
                        imagePair[29] += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k81) * imageRadial5 * harmonics[30];
                    if (sign < 0) {
                        imagePair[30] = imageValue30;
                    } else {
                        imagePair[30] += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k82) * imageRadial5 * harmonics[31];
                    if (sign < 0) {
                        imagePair[31] = imageValue31;
                    } else {
                        imagePair[31] += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k83) * imageRadial5 * harmonics[32];
                    if (sign < 0) {
                        imagePair[32] = imageValue32;
                    } else {
                        imagePair[32] += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k85) * imageRadial5 * harmonics[33];
                    if (sign < 0) {
                        imagePair[33] = imageValue33;
                    } else {
                        imagePair[33] += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k86) * imageRadial5 * harmonics[34];
                    if (sign < 0) {
                        imagePair[34] = imageValue34;
                    } else {
                        imagePair[34] += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k87) * imageRadial5 * harmonics[35];
                    if (sign < 0) {
                        imagePair[35] = imageValue35;
                    } else {
                        imagePair[35] += imageValue35;
                    }
                }
                detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                detail::ewaldAccumulate(output[25], compensation[25], imagePair[25]);
                detail::ewaldAccumulate(output[26], compensation[26], imagePair[26]);
                detail::ewaldAccumulate(output[27], compensation[27], imagePair[27]);
                detail::ewaldAccumulate(output[28], compensation[28], imagePair[28]);
                detail::ewaldAccumulate(output[29], compensation[29], imagePair[29]);
                detail::ewaldAccumulate(output[30], compensation[30], imagePair[30]);
                detail::ewaldAccumulate(output[31], compensation[31], imagePair[31]);
                detail::ewaldAccumulate(output[32], compensation[32], imagePair[32]);
                detail::ewaldAccumulate(output[33], compensation[33], imagePair[33]);
                detail::ewaldAccumulate(output[34], compensation[34], imagePair[34]);
                detail::ewaldAccumulate(output[35], compensation[35], imagePair[35]);
                Real const waveX = broadcast<Real>(ix);
                Real const waveY = broadcast<Real>(iy);
                Real const waveZ = broadcast<Real>(iz);
                Real const waveSquared = broadcast<Real>(latticeSquared);
                ewaldRegular<degree>(harmonics, waveX, waveY, waveZ, waveSquared);
                Real const phase = broadcast<Real>(k64) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k53) *
                    detail::ewaldExp(-broadcast<Real>(k59) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k56) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k64) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k57) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k57) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k70) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k73) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k62) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k60) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k62) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k73) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k72) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k78) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k76) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k75) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k74) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k75) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k76) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k78) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k71) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k84) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k80) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k79) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k78) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k77) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k78) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k79) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k80) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k84) * waveRadial4 * harmonics[24]);
                Real const waveRadial5 = broadcast<Real>(k68) * wavePotential * -sine;
                detail::ewaldAccumulate(output[25], compensation[25],
                    broadcast<Real>(k87) * waveRadial5 * harmonics[25]);
                detail::ewaldAccumulate(output[26], compensation[26],
                    broadcast<Real>(k86) * waveRadial5 * harmonics[26]);
                detail::ewaldAccumulate(output[27], compensation[27],
                    broadcast<Real>(k85) * waveRadial5 * harmonics[27]);
                detail::ewaldAccumulate(output[28], compensation[28],
                    broadcast<Real>(k83) * waveRadial5 * harmonics[28]);
                detail::ewaldAccumulate(output[29], compensation[29],
                    broadcast<Real>(k82) * waveRadial5 * harmonics[29]);
                detail::ewaldAccumulate(output[30], compensation[30],
                    broadcast<Real>(k81) * waveRadial5 * harmonics[30]);
                detail::ewaldAccumulate(output[31], compensation[31],
                    broadcast<Real>(k82) * waveRadial5 * harmonics[31]);
                detail::ewaldAccumulate(output[32], compensation[32],
                    broadcast<Real>(k83) * waveRadial5 * harmonics[32]);
                detail::ewaldAccumulate(output[33], compensation[33],
                    broadcast<Real>(k85) * waveRadial5 * harmonics[33]);
                detail::ewaldAccumulate(output[34], compensation[34],
                    broadcast<Real>(k86) * waveRadial5 * harmonics[34]);
                detail::ewaldAccumulate(output[35], compensation[35],
                    broadcast<Real>(k87) * waveRadial5 * harmonics[35]);
            }
        }
    }
}
/// Evaluate the periodic correction with promoted working precision, then convert the result.
template <int degree, typename Real, typename Output>
    requires(degree == 5)
CUDA_EXPORT void evaluateEwaldGreen(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    using Working = typename detail::EwaldWorkingType<Real>::Type;
    Working temporary[36];
    Working trace;
    evaluateEwaldWorking<degree, Working>(temporary, trace, detail::ewaldConvert<Working>(x),
        detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
    for (int i = 0; i < 36; ++i) {
        output[i] = detail::ewaldConvert<Real>(temporary[i]);
    }
    greenTrace = detail::ewaldConvert<Real>(trace);
}
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 7) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 7)
template <>
inline constexpr int kEwaldGreenFlops<6> = 577958;
template <>
inline constexpr int kEwaldGreenConventionalFlops<6> = 627304;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
template <int degree, typename Real>
    requires(degree == 6)
CUDA_EXPORT void ewaldRegular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(2.777777777777777624E-02);
    constexpr auto k01 = static_cast<Scalar>(2.857142857142857054E-02);
    constexpr auto k02 = static_cast<Scalar>(3.125000000000000000E-02);
    constexpr auto k03 = static_cast<Scalar>(3.703703703703703498E-02);
    constexpr auto k04 = static_cast<Scalar>(4.000000000000000083E-02);
    constexpr auto k05 = static_cast<Scalar>(4.166666666666666435E-02);
    constexpr auto k06 = static_cast<Scalar>(4.761904761904761640E-02);
    constexpr auto k07 = static_cast<Scalar>(5.000000000000000278E-02);
    constexpr auto k08 = static_cast<Scalar>(6.250000000000000000E-02);
    constexpr auto k09 = static_cast<Scalar>(6.666666666666666574E-02);
    constexpr auto k10 = static_cast<Scalar>(8.333333333333332871E-02);
    constexpr auto k11 = static_cast<Scalar>(1.000000000000000056E-01);
    constexpr auto k12 = static_cast<Scalar>(1.111111111111111049E-01);
    constexpr auto k13 = static_cast<Scalar>(1.250000000000000000E-01);
    constexpr auto k14 = static_cast<Scalar>(1.666666666666666574E-01);
    constexpr auto k15 = static_cast<Scalar>(2.500000000000000000E-01);
    constexpr auto k16 = static_cast<Scalar>(5.000000000000000000E-01);
    constexpr auto k17 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k18 = static_cast<Scalar>(3.000000000000000000E+00);
    constexpr auto k19 = static_cast<Scalar>(5.000000000000000000E+00);
    constexpr auto k20 = static_cast<Scalar>(7.000000000000000000E+00);
    constexpr auto k21 = static_cast<Scalar>(9.000000000000000000E+00);
    constexpr auto k22 = static_cast<Scalar>(1.100000000000000000E+01);
    output[0] = broadcast<Real>(k17);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k18) * z;
    output[6] = broadcast<Real>(k15) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k19) * z;
    output[12] = broadcast<Real>(k12) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k20) * z;
    output[20] = broadcast<Real>(k08) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    Real const zFactor5_0 = broadcast<Real>(k21) * z;
    output[30] = broadcast<Real>(k04) * fma(zFactor5_0, output[20], -radiusSquared * output[12]);
    Real const zFactor6_0 = broadcast<Real>(k22) * z;
    output[42] = broadcast<Real>(k00) * fma(zFactor6_0, output[30], -radiusSquared * output[20]);
    output[3] = x * broadcast<Real>(k16);
    output[1] = y * broadcast<Real>(k16);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k19) * z;
    output[13] = broadcast<Real>(k13) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k13) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k20) * z;
    output[21] = broadcast<Real>(k09) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k09) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    Real const zFactor5_1 = broadcast<Real>(k21) * z;
    output[31] = broadcast<Real>(k05) * fma(zFactor5_1, output[21], -radiusSquared * output[13]);
    output[29] = broadcast<Real>(k05) * fma(zFactor5_1, output[19], -radiusSquared * output[11]);
    Real const zFactor6_1 = broadcast<Real>(k22) * z;
    output[43] = broadcast<Real>(k01) * fma(zFactor6_1, output[31], -radiusSquared * output[21]);
    output[41] = broadcast<Real>(k01) * fma(zFactor6_1, output[29], -radiusSquared * output[19]);
    output[8] = broadcast<Real>(k15) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k15) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k20) * z;
    output[22] = broadcast<Real>(k10) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k10) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    Real const zFactor5_2 = broadcast<Real>(k21) * z;
    output[32] = broadcast<Real>(k06) * fma(zFactor5_2, output[22], -radiusSquared * output[14]);
    output[28] = broadcast<Real>(k06) * fma(zFactor5_2, output[18], -radiusSquared * output[10]);
    Real const zFactor6_2 = broadcast<Real>(k22) * z;
    output[44] = broadcast<Real>(k02) * fma(zFactor6_2, output[32], -radiusSquared * output[22]);
    output[40] = broadcast<Real>(k02) * fma(zFactor6_2, output[28], -radiusSquared * output[18]);
    output[15] = broadcast<Real>(k14) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k14) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    Real const zFactor5_3 = broadcast<Real>(k21) * z;
    output[33] = broadcast<Real>(k08) * fma(zFactor5_3, output[23], -radiusSquared * output[15]);
    output[27] = broadcast<Real>(k08) * fma(zFactor5_3, output[17], -radiusSquared * output[9]);
    Real const zFactor6_3 = broadcast<Real>(k22) * z;
    output[45] = broadcast<Real>(k03) * fma(zFactor6_3, output[33], -radiusSquared * output[23]);
    output[39] = broadcast<Real>(k03) * fma(zFactor6_3, output[27], -radiusSquared * output[17]);
    output[24] = broadcast<Real>(k13) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k13) * fma(y, output[15], x * output[9]);
    output[34] = z * output[24];
    output[26] = z * output[16];
    Real const zFactor6_4 = broadcast<Real>(k22) * z;
    output[46] = broadcast<Real>(k07) * fma(zFactor6_4, output[34], -radiusSquared * output[24]);
    output[38] = broadcast<Real>(k07) * fma(zFactor6_4, output[26], -radiusSquared * output[16]);
    output[35] = broadcast<Real>(k11) * fma(x, output[24], -y * output[16]);
    output[25] = broadcast<Real>(k11) * fma(y, output[24], x * output[16]);
    output[47] = z * output[35];
    output[37] = z * output[25];
    output[48] = broadcast<Real>(k10) * fma(x, output[35], -y * output[25]);
    output[36] = broadcast<Real>(k10) * fma(y, output[35], x * output[25]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 6)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k01 = static_cast<Scalar>(1.834862385321101019E-02);
    constexpr auto k02 = static_cast<Scalar>(1.869158878504672758E-02);
    constexpr auto k03 = static_cast<Scalar>(1.904761904761904934E-02);
    constexpr auto k04 = static_cast<Scalar>(1.941747572815533812E-02);
    constexpr auto k05 = static_cast<Scalar>(1.980198019801980222E-02);
    constexpr auto k06 = static_cast<Scalar>(2.020202020202020374E-02);
    constexpr auto k07 = static_cast<Scalar>(2.061855670103092744E-02);
    constexpr auto k08 = static_cast<Scalar>(2.105263157894736795E-02);
    constexpr auto k09 = static_cast<Scalar>(2.150537634408602322E-02);
    constexpr auto k10 = static_cast<Scalar>(2.197802197802197974E-02);
    constexpr auto k11 = static_cast<Scalar>(2.247191011235954994E-02);
    constexpr auto k12 = static_cast<Scalar>(2.298850574712643646E-02);
    constexpr auto k13 = static_cast<Scalar>(2.352941176470588203E-02);
    constexpr auto k14 = static_cast<Scalar>(2.409638554216867595E-02);
    constexpr auto k15 = static_cast<Scalar>(2.469135802469135665E-02);
    constexpr auto k16 = static_cast<Scalar>(2.531645569620253125E-02);
    constexpr auto k17 = static_cast<Scalar>(2.597402597402597574E-02);
    constexpr auto k18 = static_cast<Scalar>(2.666666666666666838E-02);
    constexpr auto k19 = static_cast<Scalar>(2.739726027397260122E-02);
    constexpr auto k20 = static_cast<Scalar>(2.816901408450704289E-02);
    constexpr auto k21 = static_cast<Scalar>(2.898550724637681195E-02);
    constexpr auto k22 = static_cast<Scalar>(2.985074626865671585E-02);
    constexpr auto k23 = static_cast<Scalar>(3.076923076923077094E-02);
    constexpr auto k24 = static_cast<Scalar>(3.174603174603174427E-02);
    constexpr auto k25 = static_cast<Scalar>(3.278688524590164105E-02);
    constexpr auto k26 = static_cast<Scalar>(3.389830508474576259E-02);
    constexpr auto k27 = static_cast<Scalar>(3.508771929824561209E-02);
    constexpr auto k28 = static_cast<Scalar>(3.636363636363636187E-02);
    constexpr auto k29 = static_cast<Scalar>(3.773584905660377214E-02);
    constexpr auto k30 = static_cast<Scalar>(3.921568627450980338E-02);
    constexpr auto k31 = static_cast<Scalar>(4.081632653061224164E-02);
    constexpr auto k32 = static_cast<Scalar>(4.255319148936170109E-02);
    constexpr auto k33 = static_cast<Scalar>(4.444444444444444614E-02);
    constexpr auto k34 = static_cast<Scalar>(4.651162790697674354E-02);
    constexpr auto k35 = static_cast<Scalar>(4.878048780487805047E-02);
    constexpr auto k36 = static_cast<Scalar>(5.128205128205128027E-02);
    constexpr auto k37 = static_cast<Scalar>(5.405405405405405705E-02);
    constexpr auto k38 = static_cast<Scalar>(5.714285714285714107E-02);
    constexpr auto k39 = static_cast<Scalar>(6.060606060606060774E-02);
    constexpr auto k40 = static_cast<Scalar>(6.451612903225806273E-02);
    constexpr auto k41 = static_cast<Scalar>(6.896551724137930939E-02);
    constexpr auto k42 = static_cast<Scalar>(7.407407407407406996E-02);
    constexpr auto k43 = static_cast<Scalar>(8.000000000000000167E-02);
    constexpr auto k44 = static_cast<Scalar>(8.695652173913043237E-02);
    constexpr auto k45 = static_cast<Scalar>(9.523809523809523281E-02);
    constexpr auto k46 = static_cast<Scalar>(1.052631578947368363E-01);
    constexpr auto k47 = static_cast<Scalar>(1.176470588235294101E-01);
    constexpr auto k48 = static_cast<Scalar>(1.333333333333333315E-01);
    constexpr auto k49 = static_cast<Scalar>(1.538461538461538547E-01);
    constexpr auto k50 = static_cast<Scalar>(1.818181818181818232E-01);
    constexpr auto k51 = static_cast<Scalar>(2.222222222222222099E-01);
    constexpr auto k52 = static_cast<Scalar>(2.857142857142856984E-01);
    constexpr auto k53 = static_cast<Scalar>(4.000000000000000222E-01);
    constexpr auto k54 = static_cast<Scalar>(6.366197723675813824E-01);
    constexpr auto k55 = static_cast<Scalar>(6.666666666666666297E-01);
    constexpr auto k56 = static_cast<Scalar>(7.853981633974482790E-01);
    constexpr auto k57 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k58 = static_cast<Scalar>(2.000000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(2.256758334191025117E+00);
    constexpr auto k60 = static_cast<Scalar>(2.467401100272339498E+00);
    constexpr auto k61 = static_cast<Scalar>(4.000000000000000000E+00);
    constexpr auto k62 = static_cast<Scalar>(4.188790204786390525E+00);
    constexpr auto k63 = static_cast<Scalar>(4.377812237822711339E+00);
    constexpr auto k64 = static_cast<Scalar>(5.919086906091339806E+00);
    constexpr auto k65 = static_cast<Scalar>(6.000000000000000000E+00);
    constexpr auto k66 = static_cast<Scalar>(6.018022224509399720E+00);
    constexpr auto k67 = static_cast<Scalar>(6.283185307179586232E+00);
    constexpr auto k68 = static_cast<Scalar>(7.113944886461904815E+00);
    constexpr auto k69 = static_cast<Scalar>(9.628835559215042750E+00);
    constexpr auto k70 = static_cast<Scalar>(9.781674218885120453E+00);
    constexpr auto k71 = static_cast<Scalar>(1.036257133664445007E+01);
    constexpr auto k72 = static_cast<Scalar>(1.100438349624576340E+01);
    constexpr auto k73 = static_cast<Scalar>(1.315947253478581125E+01);
    constexpr auto k74 = static_cast<Scalar>(1.484329006232417747E+01);
    constexpr auto k75 = static_cast<Scalar>(1.653668089615990056E+01);
    constexpr auto k76 = static_cast<Scalar>(2.400000000000000000E+01);
    constexpr auto k77 = static_cast<Scalar>(3.600000000000000000E+01);
    constexpr auto k78 = static_cast<Scalar>(4.800000000000000000E+01);
    constexpr auto k79 = static_cast<Scalar>(1.200000000000000000E+02);
    constexpr auto k80 = static_cast<Scalar>(5.760000000000000000E+02);
    constexpr auto k81 = static_cast<Scalar>(7.200000000000000000E+02);
    constexpr auto k82 = static_cast<Scalar>(1.440000000000000000E+03);
    constexpr auto k83 = static_cast<Scalar>(5.040000000000000000E+03);
    constexpr auto k84 = static_cast<Scalar>(1.440000000000000000E+04);
    constexpr auto k85 = static_cast<Scalar>(1.728000000000000000E+04);
    constexpr auto k86 = static_cast<Scalar>(3.024000000000000000E+04);
    constexpr auto k87 = static_cast<Scalar>(4.032000000000000000E+04);
    constexpr auto k88 = static_cast<Scalar>(8.064000000000000000E+04);
    constexpr auto k89 = static_cast<Scalar>(3.628800000000000000E+05);
    constexpr auto k90 = static_cast<Scalar>(5.184000000000000000E+05);
    constexpr auto k91 = static_cast<Scalar>(6.048000000000000000E+05);
    constexpr auto k92 = static_cast<Scalar>(9.676800000000000000E+05);
    constexpr auto k93 = static_cast<Scalar>(2.177280000000000000E+06);
    constexpr auto k94 = static_cast<Scalar>(3.628800000000000000E+06);
    constexpr auto k95 = static_cast<Scalar>(7.257600000000000000E+06);
    constexpr auto k96 = static_cast<Scalar>(3.991680000000000000E+07);
    constexpr auto k97 = static_cast<Scalar>(4.790016000000000000E+08);
    Real harmonics[49];
    Real compensation[49];
    Real imagePair[49];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k61) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k57);
    series0 = fma(series0, argument * broadcast<Real>(k07), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k08), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k09), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k50), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k51), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k52), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k53), broadcast<Real>(k57));
    series0 = fma(series0, argument * broadcast<Real>(k55), broadcast<Real>(k57));
    Real const radial0 = broadcast<Real>(k59) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k00);
    Real series1 = broadcast<Real>(k57);
    series1 = fma(series1, argument * broadcast<Real>(k06), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k07), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k08), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k09), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k49), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k50), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k51), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k57));
    series1 = fma(series1, argument * broadcast<Real>(k53), broadcast<Real>(k57));
    Real const radial1 = -broadcast<Real>(k66) * exponential * series1;
    output[1] = broadcast<Real>(k58) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k00);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k00);
    output[3] = broadcast<Real>(k58) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k00);
    Real series2 = broadcast<Real>(k57);
    series2 = fma(series2, argument * broadcast<Real>(k05), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k06), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k07), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k08), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k09), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k48), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k49), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k50), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k51), broadcast<Real>(k57));
    series2 = fma(series2, argument * broadcast<Real>(k52), broadcast<Real>(k57));
    Real const radial2 = broadcast<Real>(k69) * exponential * series2;
    output[4] = broadcast<Real>(k76) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k00);
    output[5] = broadcast<Real>(k65) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k00);
    output[6] = broadcast<Real>(k61) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k00);
    output[7] = broadcast<Real>(k65) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k00);
    output[8] = broadcast<Real>(k76) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k00);
    Real series3 = broadcast<Real>(k57);
    series3 = fma(series3, argument * broadcast<Real>(k04), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k05), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k06), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k07), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k08), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k09), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k47), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k48), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k49), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k50), broadcast<Real>(k57));
    series3 = fma(series3, argument * broadcast<Real>(k51), broadcast<Real>(k57));
    Real const radial3 = -broadcast<Real>(k72) * exponential * series3;
    output[9] = broadcast<Real>(k81) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k00);
    output[10] = broadcast<Real>(k79) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k00);
    output[11] = broadcast<Real>(k78) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k00);
    output[12] = broadcast<Real>(k77) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k00);
    output[13] = broadcast<Real>(k78) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k00);
    output[14] = broadcast<Real>(k79) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k00);
    output[15] = broadcast<Real>(k81) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k00);
    Real series4 = broadcast<Real>(k57);
    series4 = fma(series4, argument * broadcast<Real>(k03), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k04), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k05), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k06), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k07), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k08), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k09), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k10), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k11), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k12), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k13), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k14), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k15), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k16), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k17), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k18), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k19), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k20), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k21), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k22), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k23), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k24), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k25), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k26), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k27), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k28), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k29), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k30), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k31), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k32), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k33), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k34), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k35), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k36), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k37), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k38), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k39), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k40), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k41), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k42), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k43), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k44), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k45), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k46), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k47), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k48), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k49), broadcast<Real>(k57));
    series4 = fma(series4, argument * broadcast<Real>(k50), broadcast<Real>(k57));
    Real const radial4 = broadcast<Real>(k70) * exponential * series4;
    output[16] = broadcast<Real>(k87) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k00);
    output[17] = broadcast<Real>(k83) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k00);
    output[18] = broadcast<Real>(k82) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k00);
    output[19] = broadcast<Real>(k81) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k00);
    output[20] = broadcast<Real>(k80) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k00);
    output[21] = broadcast<Real>(k81) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k00);
    output[22] = broadcast<Real>(k82) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k00);
    output[23] = broadcast<Real>(k83) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k00);
    output[24] = broadcast<Real>(k87) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k00);
    Real series5 = broadcast<Real>(k57);
    series5 = fma(series5, argument * broadcast<Real>(k02), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k03), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k04), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k05), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k06), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k07), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k08), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k09), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k10), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k11), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k12), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k13), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k14), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k15), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k16), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k17), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k18), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k19), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k20), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k21), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k22), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k23), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k24), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k25), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k26), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k27), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k28), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k29), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k30), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k31), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k32), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k33), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k34), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k35), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k36), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k37), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k38), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k39), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k40), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k41), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k42), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k43), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k44), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k45), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k46), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k47), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k48), broadcast<Real>(k57));
    series5 = fma(series5, argument * broadcast<Real>(k49), broadcast<Real>(k57));
    Real const radial5 = -broadcast<Real>(k68) * exponential * series5;
    output[25] = broadcast<Real>(k94) * radial5 * harmonics[25];
    compensation[25] = broadcast<Real>(k00);
    output[26] = broadcast<Real>(k89) * radial5 * harmonics[26];
    compensation[26] = broadcast<Real>(k00);
    output[27] = broadcast<Real>(k88) * radial5 * harmonics[27];
    compensation[27] = broadcast<Real>(k00);
    output[28] = broadcast<Real>(k86) * radial5 * harmonics[28];
    compensation[28] = broadcast<Real>(k00);
    output[29] = broadcast<Real>(k85) * radial5 * harmonics[29];
    compensation[29] = broadcast<Real>(k00);
    output[30] = broadcast<Real>(k84) * radial5 * harmonics[30];
    compensation[30] = broadcast<Real>(k00);
    output[31] = broadcast<Real>(k85) * radial5 * harmonics[31];
    compensation[31] = broadcast<Real>(k00);
    output[32] = broadcast<Real>(k86) * radial5 * harmonics[32];
    compensation[32] = broadcast<Real>(k00);
    output[33] = broadcast<Real>(k88) * radial5 * harmonics[33];
    compensation[33] = broadcast<Real>(k00);
    output[34] = broadcast<Real>(k89) * radial5 * harmonics[34];
    compensation[34] = broadcast<Real>(k00);
    output[35] = broadcast<Real>(k94) * radial5 * harmonics[35];
    compensation[35] = broadcast<Real>(k00);
    Real series6 = broadcast<Real>(k57);
    series6 = fma(series6, argument * broadcast<Real>(k01), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k02), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k03), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k04), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k05), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k06), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k07), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k08), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k09), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k10), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k11), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k12), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k13), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k14), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k15), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k16), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k17), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k18), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k19), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k20), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k21), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k22), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k23), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k24), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k25), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k26), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k27), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k28), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k29), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k30), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k31), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k32), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k33), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k34), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k35), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k36), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k37), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k38), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k39), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k40), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k41), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k42), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k43), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k44), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k45), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k46), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k47), broadcast<Real>(k57));
    series6 = fma(series6, argument * broadcast<Real>(k48), broadcast<Real>(k57));
    Real const radial6 = broadcast<Real>(k63) * exponential * series6;
    output[36] = broadcast<Real>(k97) * radial6 * harmonics[36];
    compensation[36] = broadcast<Real>(k00);
    output[37] = broadcast<Real>(k96) * radial6 * harmonics[37];
    compensation[37] = broadcast<Real>(k00);
    output[38] = broadcast<Real>(k95) * radial6 * harmonics[38];
    compensation[38] = broadcast<Real>(k00);
    output[39] = broadcast<Real>(k93) * radial6 * harmonics[39];
    compensation[39] = broadcast<Real>(k00);
    output[40] = broadcast<Real>(k92) * radial6 * harmonics[40];
    compensation[40] = broadcast<Real>(k00);
    output[41] = broadcast<Real>(k91) * radial6 * harmonics[41];
    compensation[41] = broadcast<Real>(k00);
    output[42] = broadcast<Real>(k90) * radial6 * harmonics[42];
    compensation[42] = broadcast<Real>(k00);
    output[43] = broadcast<Real>(k91) * radial6 * harmonics[43];
    compensation[43] = broadcast<Real>(k00);
    output[44] = broadcast<Real>(k92) * radial6 * harmonics[44];
    compensation[44] = broadcast<Real>(k00);
    output[45] = broadcast<Real>(k93) * radial6 * harmonics[45];
    compensation[45] = broadcast<Real>(k00);
    output[46] = broadcast<Real>(k95) * radial6 * harmonics[46];
    compensation[46] = broadcast<Real>(k00);
    output[47] = broadcast<Real>(k96) * radial6 * harmonics[47];
    compensation[47] = broadcast<Real>(k00);
    output[48] = broadcast<Real>(k97) * radial6 * harmonics[48];
    compensation[48] = broadcast<Real>(k00);
    output[0] += broadcast<Real>(k56);
    greenTrace = broadcast<Real>(k62);
    for (int ix = -6; ix <= 6; ++ix) {
        for (int iy = -6; iy <= 6; ++iy) {
            for (int iz = -6; iz <= 6; ++iz) {
                int const latticeSquared = ix * ix + iy * iy + iz * iz;
                if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 || (ix == 0 && iy < 0) ||
                    (ix == 0 && iy == 0 && iz < 0)) {
                    continue;
                }
                for (int sign = -1; sign <= 1; sign += 2) {
                    Real const imageX = x + broadcast<Real>(sign * ix);
                    Real const imageY = y + broadcast<Real>(sign * iy);
                    Real const imageZ = z + broadcast<Real>(sign * iz);
                    Real const imageSquared =
                        fma(imageX, imageX, fma(imageY, imageY, imageZ * imageZ));
                    Real const imageRadius = sqrt(imageSquared);
                    Real const iR = broadcast<Real>(k57) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k61) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k58) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k59) * imageRadius * imageExp;
                    Real inversePower = iR;
                    Real const imageRadial0 = -screening * inversePower;
                    Real const imageValue0 = imageRadial0 * harmonics[0];
                    if (sign < 0) {
                        imagePair[0] = imageValue0;
                    } else {
                        imagePair[0] += imageValue0;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k55);
                    Real const imageRadial1 = screening * inversePower;
                    Real const imageValue1 = broadcast<Real>(k58) * imageRadial1 * harmonics[1];
                    if (sign < 0) {
                        imagePair[1] = imageValue1;
                    } else {
                        imagePair[1] += imageValue1;
                    }
                    Real const imageValue2 = imageRadial1 * harmonics[2];
                    if (sign < 0) {
                        imagePair[2] = imageValue2;
                    } else {
                        imagePair[2] += imageValue2;
                    }
                    Real const imageValue3 = broadcast<Real>(k58) * imageRadial1 * harmonics[3];
                    if (sign < 0) {
                        imagePair[3] = imageValue3;
                    } else {
                        imagePair[3] += imageValue3;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k53);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k76) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k65) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k61) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k65) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k76) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k52);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k81) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k79) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k78) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k77) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k78) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k79) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k81) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k51);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k87) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k83) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k82) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k81) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k80) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k81) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k82) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k83) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k87) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k50);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k94) * imageRadial5 * harmonics[25];
                    if (sign < 0) {
                        imagePair[25] = imageValue25;
                    } else {
                        imagePair[25] += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k89) * imageRadial5 * harmonics[26];
                    if (sign < 0) {
                        imagePair[26] = imageValue26;
                    } else {
                        imagePair[26] += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k88) * imageRadial5 * harmonics[27];
                    if (sign < 0) {
                        imagePair[27] = imageValue27;
                    } else {
                        imagePair[27] += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k86) * imageRadial5 * harmonics[28];
                    if (sign < 0) {
                        imagePair[28] = imageValue28;
                    } else {
                        imagePair[28] += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k85) * imageRadial5 * harmonics[29];
                    if (sign < 0) {
                        imagePair[29] = imageValue29;
                    } else {
                        imagePair[29] += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k84) * imageRadial5 * harmonics[30];
                    if (sign < 0) {
                        imagePair[30] = imageValue30;
                    } else {
                        imagePair[30] += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k85) * imageRadial5 * harmonics[31];
                    if (sign < 0) {
                        imagePair[31] = imageValue31;
                    } else {
                        imagePair[31] += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k86) * imageRadial5 * harmonics[32];
                    if (sign < 0) {
                        imagePair[32] = imageValue32;
                    } else {
                        imagePair[32] += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k88) * imageRadial5 * harmonics[33];
                    if (sign < 0) {
                        imagePair[33] = imageValue33;
                    } else {
                        imagePair[33] += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k89) * imageRadial5 * harmonics[34];
                    if (sign < 0) {
                        imagePair[34] = imageValue34;
                    } else {
                        imagePair[34] += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k94) * imageRadial5 * harmonics[35];
                    if (sign < 0) {
                        imagePair[35] = imageValue35;
                    } else {
                        imagePair[35] += imageValue35;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
                    Real const imageRadial6 = -screening * inversePower;
                    Real const imageValue36 = broadcast<Real>(k97) * imageRadial6 * harmonics[36];
                    if (sign < 0) {
                        imagePair[36] = imageValue36;
                    } else {
                        imagePair[36] += imageValue36;
                    }
                    Real const imageValue37 = broadcast<Real>(k96) * imageRadial6 * harmonics[37];
                    if (sign < 0) {
                        imagePair[37] = imageValue37;
                    } else {
                        imagePair[37] += imageValue37;
                    }
                    Real const imageValue38 = broadcast<Real>(k95) * imageRadial6 * harmonics[38];
                    if (sign < 0) {
                        imagePair[38] = imageValue38;
                    } else {
                        imagePair[38] += imageValue38;
                    }
                    Real const imageValue39 = broadcast<Real>(k93) * imageRadial6 * harmonics[39];
                    if (sign < 0) {
                        imagePair[39] = imageValue39;
                    } else {
                        imagePair[39] += imageValue39;
                    }
                    Real const imageValue40 = broadcast<Real>(k92) * imageRadial6 * harmonics[40];
                    if (sign < 0) {
                        imagePair[40] = imageValue40;
                    } else {
                        imagePair[40] += imageValue40;
                    }
                    Real const imageValue41 = broadcast<Real>(k91) * imageRadial6 * harmonics[41];
                    if (sign < 0) {
                        imagePair[41] = imageValue41;
                    } else {
                        imagePair[41] += imageValue41;
                    }
                    Real const imageValue42 = broadcast<Real>(k90) * imageRadial6 * harmonics[42];
                    if (sign < 0) {
                        imagePair[42] = imageValue42;
                    } else {
                        imagePair[42] += imageValue42;
                    }
                    Real const imageValue43 = broadcast<Real>(k91) * imageRadial6 * harmonics[43];
                    if (sign < 0) {
                        imagePair[43] = imageValue43;
                    } else {
                        imagePair[43] += imageValue43;
                    }
                    Real const imageValue44 = broadcast<Real>(k92) * imageRadial6 * harmonics[44];
                    if (sign < 0) {
                        imagePair[44] = imageValue44;
                    } else {
                        imagePair[44] += imageValue44;
                    }
                    Real const imageValue45 = broadcast<Real>(k93) * imageRadial6 * harmonics[45];
                    if (sign < 0) {
                        imagePair[45] = imageValue45;
                    } else {
                        imagePair[45] += imageValue45;
                    }
                    Real const imageValue46 = broadcast<Real>(k95) * imageRadial6 * harmonics[46];
                    if (sign < 0) {
                        imagePair[46] = imageValue46;
                    } else {
                        imagePair[46] += imageValue46;
                    }
                    Real const imageValue47 = broadcast<Real>(k96) * imageRadial6 * harmonics[47];
                    if (sign < 0) {
                        imagePair[47] = imageValue47;
                    } else {
                        imagePair[47] += imageValue47;
                    }
                    Real const imageValue48 = broadcast<Real>(k97) * imageRadial6 * harmonics[48];
                    if (sign < 0) {
                        imagePair[48] = imageValue48;
                    } else {
                        imagePair[48] += imageValue48;
                    }
                }
                detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                detail::ewaldAccumulate(output[25], compensation[25], imagePair[25]);
                detail::ewaldAccumulate(output[26], compensation[26], imagePair[26]);
                detail::ewaldAccumulate(output[27], compensation[27], imagePair[27]);
                detail::ewaldAccumulate(output[28], compensation[28], imagePair[28]);
                detail::ewaldAccumulate(output[29], compensation[29], imagePair[29]);
                detail::ewaldAccumulate(output[30], compensation[30], imagePair[30]);
                detail::ewaldAccumulate(output[31], compensation[31], imagePair[31]);
                detail::ewaldAccumulate(output[32], compensation[32], imagePair[32]);
                detail::ewaldAccumulate(output[33], compensation[33], imagePair[33]);
                detail::ewaldAccumulate(output[34], compensation[34], imagePair[34]);
                detail::ewaldAccumulate(output[35], compensation[35], imagePair[35]);
                detail::ewaldAccumulate(output[36], compensation[36], imagePair[36]);
                detail::ewaldAccumulate(output[37], compensation[37], imagePair[37]);
                detail::ewaldAccumulate(output[38], compensation[38], imagePair[38]);
                detail::ewaldAccumulate(output[39], compensation[39], imagePair[39]);
                detail::ewaldAccumulate(output[40], compensation[40], imagePair[40]);
                detail::ewaldAccumulate(output[41], compensation[41], imagePair[41]);
                detail::ewaldAccumulate(output[42], compensation[42], imagePair[42]);
                detail::ewaldAccumulate(output[43], compensation[43], imagePair[43]);
                detail::ewaldAccumulate(output[44], compensation[44], imagePair[44]);
                detail::ewaldAccumulate(output[45], compensation[45], imagePair[45]);
                detail::ewaldAccumulate(output[46], compensation[46], imagePair[46]);
                detail::ewaldAccumulate(output[47], compensation[47], imagePair[47]);
                detail::ewaldAccumulate(output[48], compensation[48], imagePair[48]);
                Real const waveX = broadcast<Real>(ix);
                Real const waveY = broadcast<Real>(iy);
                Real const waveZ = broadcast<Real>(iz);
                Real const waveSquared = broadcast<Real>(latticeSquared);
                ewaldRegular<degree>(harmonics, waveX, waveY, waveZ, waveSquared);
                Real const phase = broadcast<Real>(k67) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k54) *
                    detail::ewaldExp(-broadcast<Real>(k60) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k57) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k67) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k58) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k58) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k73) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k76) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k65) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k61) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k65) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k76) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k75) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k81) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k79) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k78) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k77) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k78) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k79) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k81) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k74) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k87) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k83) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k82) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k81) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k80) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k81) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k82) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k83) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k87) * waveRadial4 * harmonics[24]);
                Real const waveRadial5 = broadcast<Real>(k71) * wavePotential * -sine;
                detail::ewaldAccumulate(output[25], compensation[25],
                    broadcast<Real>(k94) * waveRadial5 * harmonics[25]);
                detail::ewaldAccumulate(output[26], compensation[26],
                    broadcast<Real>(k89) * waveRadial5 * harmonics[26]);
                detail::ewaldAccumulate(output[27], compensation[27],
                    broadcast<Real>(k88) * waveRadial5 * harmonics[27]);
                detail::ewaldAccumulate(output[28], compensation[28],
                    broadcast<Real>(k86) * waveRadial5 * harmonics[28]);
                detail::ewaldAccumulate(output[29], compensation[29],
                    broadcast<Real>(k85) * waveRadial5 * harmonics[29]);
                detail::ewaldAccumulate(output[30], compensation[30],
                    broadcast<Real>(k84) * waveRadial5 * harmonics[30]);
                detail::ewaldAccumulate(output[31], compensation[31],
                    broadcast<Real>(k85) * waveRadial5 * harmonics[31]);
                detail::ewaldAccumulate(output[32], compensation[32],
                    broadcast<Real>(k86) * waveRadial5 * harmonics[32]);
                detail::ewaldAccumulate(output[33], compensation[33],
                    broadcast<Real>(k88) * waveRadial5 * harmonics[33]);
                detail::ewaldAccumulate(output[34], compensation[34],
                    broadcast<Real>(k89) * waveRadial5 * harmonics[34]);
                detail::ewaldAccumulate(output[35], compensation[35],
                    broadcast<Real>(k94) * waveRadial5 * harmonics[35]);
                Real const waveRadial6 = broadcast<Real>(k64) * wavePotential * -cosine;
                detail::ewaldAccumulate(output[36], compensation[36],
                    broadcast<Real>(k97) * waveRadial6 * harmonics[36]);
                detail::ewaldAccumulate(output[37], compensation[37],
                    broadcast<Real>(k96) * waveRadial6 * harmonics[37]);
                detail::ewaldAccumulate(output[38], compensation[38],
                    broadcast<Real>(k95) * waveRadial6 * harmonics[38]);
                detail::ewaldAccumulate(output[39], compensation[39],
                    broadcast<Real>(k93) * waveRadial6 * harmonics[39]);
                detail::ewaldAccumulate(output[40], compensation[40],
                    broadcast<Real>(k92) * waveRadial6 * harmonics[40]);
                detail::ewaldAccumulate(output[41], compensation[41],
                    broadcast<Real>(k91) * waveRadial6 * harmonics[41]);
                detail::ewaldAccumulate(output[42], compensation[42],
                    broadcast<Real>(k90) * waveRadial6 * harmonics[42]);
                detail::ewaldAccumulate(output[43], compensation[43],
                    broadcast<Real>(k91) * waveRadial6 * harmonics[43]);
                detail::ewaldAccumulate(output[44], compensation[44],
                    broadcast<Real>(k92) * waveRadial6 * harmonics[44]);
                detail::ewaldAccumulate(output[45], compensation[45],
                    broadcast<Real>(k93) * waveRadial6 * harmonics[45]);
                detail::ewaldAccumulate(output[46], compensation[46],
                    broadcast<Real>(k95) * waveRadial6 * harmonics[46]);
                detail::ewaldAccumulate(output[47], compensation[47],
                    broadcast<Real>(k96) * waveRadial6 * harmonics[47]);
                detail::ewaldAccumulate(output[48], compensation[48],
                    broadcast<Real>(k97) * waveRadial6 * harmonics[48]);
            }
        }
    }
}
/// Evaluate the periodic correction with promoted working precision, then convert the result.
template <int degree, typename Real, typename Output>
    requires(degree == 6)
CUDA_EXPORT void evaluateEwaldGreen(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    using Working = typename detail::EwaldWorkingType<Real>::Type;
    Working temporary[49];
    Working trace;
    evaluateEwaldWorking<degree, Working>(temporary, trace, detail::ewaldConvert<Working>(x),
        detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
    for (int i = 0; i < 49; ++i) {
        output[i] = detail::ewaldConvert<Real>(temporary[i]);
    }
    greenTrace = detail::ewaldConvert<Real>(trace);
}
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 8) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 8)
template <>
inline constexpr int kEwaldGreenFlops<7> = 752769;
template <>
inline constexpr int kEwaldGreenConventionalFlops<7> = 820194;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
template <int degree, typename Real>
    requires(degree == 7)
CUDA_EXPORT void ewaldRegular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(2.040816326530612082E-02);
    constexpr auto k01 = static_cast<Scalar>(2.083333333333333218E-02);
    constexpr auto k02 = static_cast<Scalar>(2.222222222222222307E-02);
    constexpr auto k03 = static_cast<Scalar>(2.500000000000000139E-02);
    constexpr auto k04 = static_cast<Scalar>(2.777777777777777624E-02);
    constexpr auto k05 = static_cast<Scalar>(2.857142857142857054E-02);
    constexpr auto k06 = static_cast<Scalar>(3.030303030303030387E-02);
    constexpr auto k07 = static_cast<Scalar>(3.125000000000000000E-02);
    constexpr auto k08 = static_cast<Scalar>(3.703703703703703498E-02);
    constexpr auto k09 = static_cast<Scalar>(4.000000000000000083E-02);
    constexpr auto k10 = static_cast<Scalar>(4.166666666666666435E-02);
    constexpr auto k11 = static_cast<Scalar>(4.761904761904761640E-02);
    constexpr auto k12 = static_cast<Scalar>(5.000000000000000278E-02);
    constexpr auto k13 = static_cast<Scalar>(6.250000000000000000E-02);
    constexpr auto k14 = static_cast<Scalar>(6.666666666666666574E-02);
    constexpr auto k15 = static_cast<Scalar>(7.142857142857142461E-02);
    constexpr auto k16 = static_cast<Scalar>(8.333333333333332871E-02);
    constexpr auto k17 = static_cast<Scalar>(1.000000000000000056E-01);
    constexpr auto k18 = static_cast<Scalar>(1.111111111111111049E-01);
    constexpr auto k19 = static_cast<Scalar>(1.250000000000000000E-01);
    constexpr auto k20 = static_cast<Scalar>(1.666666666666666574E-01);
    constexpr auto k21 = static_cast<Scalar>(2.500000000000000000E-01);
    constexpr auto k22 = static_cast<Scalar>(5.000000000000000000E-01);
    constexpr auto k23 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k24 = static_cast<Scalar>(3.000000000000000000E+00);
    constexpr auto k25 = static_cast<Scalar>(5.000000000000000000E+00);
    constexpr auto k26 = static_cast<Scalar>(7.000000000000000000E+00);
    constexpr auto k27 = static_cast<Scalar>(9.000000000000000000E+00);
    constexpr auto k28 = static_cast<Scalar>(1.100000000000000000E+01);
    constexpr auto k29 = static_cast<Scalar>(1.300000000000000000E+01);
    output[0] = broadcast<Real>(k23);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k24) * z;
    output[6] = broadcast<Real>(k21) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k25) * z;
    output[12] = broadcast<Real>(k18) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k26) * z;
    output[20] = broadcast<Real>(k13) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    Real const zFactor5_0 = broadcast<Real>(k27) * z;
    output[30] = broadcast<Real>(k09) * fma(zFactor5_0, output[20], -radiusSquared * output[12]);
    Real const zFactor6_0 = broadcast<Real>(k28) * z;
    output[42] = broadcast<Real>(k04) * fma(zFactor6_0, output[30], -radiusSquared * output[20]);
    Real const zFactor7_0 = broadcast<Real>(k29) * z;
    output[56] = broadcast<Real>(k00) * fma(zFactor7_0, output[42], -radiusSquared * output[30]);
    output[3] = x * broadcast<Real>(k22);
    output[1] = y * broadcast<Real>(k22);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k25) * z;
    output[13] = broadcast<Real>(k19) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k19) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k26) * z;
    output[21] = broadcast<Real>(k14) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k14) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    Real const zFactor5_1 = broadcast<Real>(k27) * z;
    output[31] = broadcast<Real>(k10) * fma(zFactor5_1, output[21], -radiusSquared * output[13]);
    output[29] = broadcast<Real>(k10) * fma(zFactor5_1, output[19], -radiusSquared * output[11]);
    Real const zFactor6_1 = broadcast<Real>(k28) * z;
    output[43] = broadcast<Real>(k05) * fma(zFactor6_1, output[31], -radiusSquared * output[21]);
    output[41] = broadcast<Real>(k05) * fma(zFactor6_1, output[29], -radiusSquared * output[19]);
    Real const zFactor7_1 = broadcast<Real>(k29) * z;
    output[57] = broadcast<Real>(k01) * fma(zFactor7_1, output[43], -radiusSquared * output[31]);
    output[55] = broadcast<Real>(k01) * fma(zFactor7_1, output[41], -radiusSquared * output[29]);
    output[8] = broadcast<Real>(k21) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k21) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k26) * z;
    output[22] = broadcast<Real>(k16) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k16) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    Real const zFactor5_2 = broadcast<Real>(k27) * z;
    output[32] = broadcast<Real>(k11) * fma(zFactor5_2, output[22], -radiusSquared * output[14]);
    output[28] = broadcast<Real>(k11) * fma(zFactor5_2, output[18], -radiusSquared * output[10]);
    Real const zFactor6_2 = broadcast<Real>(k28) * z;
    output[44] = broadcast<Real>(k07) * fma(zFactor6_2, output[32], -radiusSquared * output[22]);
    output[40] = broadcast<Real>(k07) * fma(zFactor6_2, output[28], -radiusSquared * output[18]);
    Real const zFactor7_2 = broadcast<Real>(k29) * z;
    output[58] = broadcast<Real>(k02) * fma(zFactor7_2, output[44], -radiusSquared * output[32]);
    output[54] = broadcast<Real>(k02) * fma(zFactor7_2, output[40], -radiusSquared * output[28]);
    output[15] = broadcast<Real>(k20) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k20) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    Real const zFactor5_3 = broadcast<Real>(k27) * z;
    output[33] = broadcast<Real>(k13) * fma(zFactor5_3, output[23], -radiusSquared * output[15]);
    output[27] = broadcast<Real>(k13) * fma(zFactor5_3, output[17], -radiusSquared * output[9]);
    Real const zFactor6_3 = broadcast<Real>(k28) * z;
    output[45] = broadcast<Real>(k08) * fma(zFactor6_3, output[33], -radiusSquared * output[23]);
    output[39] = broadcast<Real>(k08) * fma(zFactor6_3, output[27], -radiusSquared * output[17]);
    Real const zFactor7_3 = broadcast<Real>(k29) * z;
    output[59] = broadcast<Real>(k03) * fma(zFactor7_3, output[45], -radiusSquared * output[33]);
    output[53] = broadcast<Real>(k03) * fma(zFactor7_3, output[39], -radiusSquared * output[27]);
    output[24] = broadcast<Real>(k19) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k19) * fma(y, output[15], x * output[9]);
    output[34] = z * output[24];
    output[26] = z * output[16];
    Real const zFactor6_4 = broadcast<Real>(k28) * z;
    output[46] = broadcast<Real>(k12) * fma(zFactor6_4, output[34], -radiusSquared * output[24]);
    output[38] = broadcast<Real>(k12) * fma(zFactor6_4, output[26], -radiusSquared * output[16]);
    Real const zFactor7_4 = broadcast<Real>(k29) * z;
    output[60] = broadcast<Real>(k06) * fma(zFactor7_4, output[46], -radiusSquared * output[34]);
    output[52] = broadcast<Real>(k06) * fma(zFactor7_4, output[38], -radiusSquared * output[26]);
    output[35] = broadcast<Real>(k17) * fma(x, output[24], -y * output[16]);
    output[25] = broadcast<Real>(k17) * fma(y, output[24], x * output[16]);
    output[47] = z * output[35];
    output[37] = z * output[25];
    Real const zFactor7_5 = broadcast<Real>(k29) * z;
    output[61] = broadcast<Real>(k10) * fma(zFactor7_5, output[47], -radiusSquared * output[35]);
    output[51] = broadcast<Real>(k10) * fma(zFactor7_5, output[37], -radiusSquared * output[25]);
    output[48] = broadcast<Real>(k16) * fma(x, output[35], -y * output[25]);
    output[36] = broadcast<Real>(k16) * fma(y, output[35], x * output[25]);
    output[62] = z * output[48];
    output[50] = z * output[36];
    output[63] = broadcast<Real>(k15) * fma(x, output[48], -y * output[36]);
    output[49] = broadcast<Real>(k15) * fma(y, output[48], x * output[36]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 7)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k000 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k001 = static_cast<Scalar>(1.801801801801801786E-02);
    constexpr auto k002 = static_cast<Scalar>(1.834862385321101019E-02);
    constexpr auto k003 = static_cast<Scalar>(1.869158878504672758E-02);
    constexpr auto k004 = static_cast<Scalar>(1.904761904761904934E-02);
    constexpr auto k005 = static_cast<Scalar>(1.941747572815533812E-02);
    constexpr auto k006 = static_cast<Scalar>(1.980198019801980222E-02);
    constexpr auto k007 = static_cast<Scalar>(2.020202020202020374E-02);
    constexpr auto k008 = static_cast<Scalar>(2.061855670103092744E-02);
    constexpr auto k009 = static_cast<Scalar>(2.105263157894736795E-02);
    constexpr auto k010 = static_cast<Scalar>(2.150537634408602322E-02);
    constexpr auto k011 = static_cast<Scalar>(2.197802197802197974E-02);
    constexpr auto k012 = static_cast<Scalar>(2.247191011235954994E-02);
    constexpr auto k013 = static_cast<Scalar>(2.298850574712643646E-02);
    constexpr auto k014 = static_cast<Scalar>(2.352941176470588203E-02);
    constexpr auto k015 = static_cast<Scalar>(2.409638554216867595E-02);
    constexpr auto k016 = static_cast<Scalar>(2.469135802469135665E-02);
    constexpr auto k017 = static_cast<Scalar>(2.531645569620253125E-02);
    constexpr auto k018 = static_cast<Scalar>(2.597402597402597574E-02);
    constexpr auto k019 = static_cast<Scalar>(2.666666666666666838E-02);
    constexpr auto k020 = static_cast<Scalar>(2.739726027397260122E-02);
    constexpr auto k021 = static_cast<Scalar>(2.816901408450704289E-02);
    constexpr auto k022 = static_cast<Scalar>(2.898550724637681195E-02);
    constexpr auto k023 = static_cast<Scalar>(2.985074626865671585E-02);
    constexpr auto k024 = static_cast<Scalar>(3.076923076923077094E-02);
    constexpr auto k025 = static_cast<Scalar>(3.174603174603174427E-02);
    constexpr auto k026 = static_cast<Scalar>(3.278688524590164105E-02);
    constexpr auto k027 = static_cast<Scalar>(3.389830508474576259E-02);
    constexpr auto k028 = static_cast<Scalar>(3.508771929824561209E-02);
    constexpr auto k029 = static_cast<Scalar>(3.636363636363636187E-02);
    constexpr auto k030 = static_cast<Scalar>(3.773584905660377214E-02);
    constexpr auto k031 = static_cast<Scalar>(3.921568627450980338E-02);
    constexpr auto k032 = static_cast<Scalar>(4.081632653061224164E-02);
    constexpr auto k033 = static_cast<Scalar>(4.255319148936170109E-02);
    constexpr auto k034 = static_cast<Scalar>(4.444444444444444614E-02);
    constexpr auto k035 = static_cast<Scalar>(4.651162790697674354E-02);
    constexpr auto k036 = static_cast<Scalar>(4.878048780487805047E-02);
    constexpr auto k037 = static_cast<Scalar>(5.128205128205128027E-02);
    constexpr auto k038 = static_cast<Scalar>(5.405405405405405705E-02);
    constexpr auto k039 = static_cast<Scalar>(5.714285714285714107E-02);
    constexpr auto k040 = static_cast<Scalar>(6.060606060606060774E-02);
    constexpr auto k041 = static_cast<Scalar>(6.451612903225806273E-02);
    constexpr auto k042 = static_cast<Scalar>(6.896551724137930939E-02);
    constexpr auto k043 = static_cast<Scalar>(7.407407407407406996E-02);
    constexpr auto k044 = static_cast<Scalar>(8.000000000000000167E-02);
    constexpr auto k045 = static_cast<Scalar>(8.695652173913043237E-02);
    constexpr auto k046 = static_cast<Scalar>(9.523809523809523281E-02);
    constexpr auto k047 = static_cast<Scalar>(1.052631578947368363E-01);
    constexpr auto k048 = static_cast<Scalar>(1.176470588235294101E-01);
    constexpr auto k049 = static_cast<Scalar>(1.333333333333333315E-01);
    constexpr auto k050 = static_cast<Scalar>(1.538461538461538547E-01);
    constexpr auto k051 = static_cast<Scalar>(1.818181818181818232E-01);
    constexpr auto k052 = static_cast<Scalar>(2.222222222222222099E-01);
    constexpr auto k053 = static_cast<Scalar>(2.857142857142856984E-01);
    constexpr auto k054 = static_cast<Scalar>(4.000000000000000222E-01);
    constexpr auto k055 = static_cast<Scalar>(6.366197723675813824E-01);
    constexpr auto k056 = static_cast<Scalar>(6.666666666666666297E-01);
    constexpr auto k057 = static_cast<Scalar>(7.853981633974482790E-01);
    constexpr auto k058 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k059 = static_cast<Scalar>(2.000000000000000000E+00);
    constexpr auto k060 = static_cast<Scalar>(2.256758334191025117E+00);
    constexpr auto k061 = static_cast<Scalar>(2.334833193505446047E+00);
    constexpr auto k062 = static_cast<Scalar>(2.467401100272339498E+00);
    constexpr auto k063 = static_cast<Scalar>(2.860824606174782847E+00);
    constexpr auto k064 = static_cast<Scalar>(4.000000000000000000E+00);
    constexpr auto k065 = static_cast<Scalar>(4.188790204786390525E+00);
    constexpr auto k066 = static_cast<Scalar>(4.377812237822711339E+00);
    constexpr auto k067 = static_cast<Scalar>(5.919086906091339806E+00);
    constexpr auto k068 = static_cast<Scalar>(6.000000000000000000E+00);
    constexpr auto k069 = static_cast<Scalar>(6.018022224509399720E+00);
    constexpr auto k070 = static_cast<Scalar>(6.283185307179586232E+00);
    constexpr auto k071 = static_cast<Scalar>(7.113944886461904815E+00);
    constexpr auto k072 = static_cast<Scalar>(9.628835559215042750E+00);
    constexpr auto k073 = static_cast<Scalar>(9.781674218885120453E+00);
    constexpr auto k074 = static_cast<Scalar>(1.036257133664445007E+01);
    constexpr auto k075 = static_cast<Scalar>(1.100438349624576340E+01);
    constexpr auto k076 = static_cast<Scalar>(1.315947253478581125E+01);
    constexpr auto k077 = static_cast<Scalar>(1.484329006232417747E+01);
    constexpr auto k078 = static_cast<Scalar>(1.653668089615990056E+01);
    constexpr auto k079 = static_cast<Scalar>(2.400000000000000000E+01);
    constexpr auto k080 = static_cast<Scalar>(3.600000000000000000E+01);
    constexpr auto k081 = static_cast<Scalar>(4.800000000000000000E+01);
    constexpr auto k082 = static_cast<Scalar>(1.200000000000000000E+02);
    constexpr auto k083 = static_cast<Scalar>(5.760000000000000000E+02);
    constexpr auto k084 = static_cast<Scalar>(7.200000000000000000E+02);
    constexpr auto k085 = static_cast<Scalar>(1.440000000000000000E+03);
    constexpr auto k086 = static_cast<Scalar>(5.040000000000000000E+03);
    constexpr auto k087 = static_cast<Scalar>(1.440000000000000000E+04);
    constexpr auto k088 = static_cast<Scalar>(1.728000000000000000E+04);
    constexpr auto k089 = static_cast<Scalar>(3.024000000000000000E+04);
    constexpr auto k090 = static_cast<Scalar>(4.032000000000000000E+04);
    constexpr auto k091 = static_cast<Scalar>(8.064000000000000000E+04);
    constexpr auto k092 = static_cast<Scalar>(3.628800000000000000E+05);
    constexpr auto k093 = static_cast<Scalar>(5.184000000000000000E+05);
    constexpr auto k094 = static_cast<Scalar>(6.048000000000000000E+05);
    constexpr auto k095 = static_cast<Scalar>(9.676800000000000000E+05);
    constexpr auto k096 = static_cast<Scalar>(2.177280000000000000E+06);
    constexpr auto k097 = static_cast<Scalar>(3.628800000000000000E+06);
    constexpr auto k098 = static_cast<Scalar>(7.257600000000000000E+06);
    constexpr auto k099 = static_cast<Scalar>(2.540160000000000000E+07);
    constexpr auto k100 = static_cast<Scalar>(2.903040000000000000E+07);
    constexpr auto k101 = static_cast<Scalar>(3.991680000000000000E+07);
    constexpr auto k102 = static_cast<Scalar>(4.354560000000000000E+07);
    constexpr auto k103 = static_cast<Scalar>(8.709120000000000000E+07);
    constexpr auto k104 = static_cast<Scalar>(2.395008000000000000E+08);
    constexpr auto k105 = static_cast<Scalar>(4.790016000000000000E+08);
    constexpr auto k106 = static_cast<Scalar>(9.580032000000000000E+08);
    constexpr auto k107 = static_cast<Scalar>(6.227020800000000000E+09);
    constexpr auto k108 = static_cast<Scalar>(8.717829120000000000E+10);
    Real harmonics[64];
    Real compensation[64];
    Real imagePair[64];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k064) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k058);
    series0 = fma(series0, argument * broadcast<Real>(k008), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k009), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k010), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k011), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k012), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k013), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k014), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k015), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k016), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k017), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k018), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k019), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k020), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k021), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k022), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k023), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k024), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k025), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k026), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k027), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k028), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k029), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k030), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k031), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k032), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k033), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k034), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k035), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k036), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k037), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k038), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k039), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k040), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k041), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k042), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k043), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k044), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k045), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k046), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k047), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k048), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k049), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k050), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k051), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k052), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k053), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k054), broadcast<Real>(k058));
    series0 = fma(series0, argument * broadcast<Real>(k056), broadcast<Real>(k058));
    Real const radial0 = broadcast<Real>(k060) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k000);
    Real series1 = broadcast<Real>(k058);
    series1 = fma(series1, argument * broadcast<Real>(k007), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k008), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k009), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k010), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k011), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k012), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k013), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k014), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k015), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k016), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k017), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k018), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k019), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k020), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k021), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k022), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k023), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k024), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k025), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k026), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k027), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k028), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k029), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k030), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k031), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k032), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k033), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k034), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k035), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k036), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k037), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k038), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k039), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k040), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k041), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k042), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k043), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k044), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k045), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k046), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k047), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k048), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k049), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k050), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k051), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k052), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k053), broadcast<Real>(k058));
    series1 = fma(series1, argument * broadcast<Real>(k054), broadcast<Real>(k058));
    Real const radial1 = -broadcast<Real>(k069) * exponential * series1;
    output[1] = broadcast<Real>(k059) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k000);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k000);
    output[3] = broadcast<Real>(k059) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k000);
    Real series2 = broadcast<Real>(k058);
    series2 = fma(series2, argument * broadcast<Real>(k006), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k007), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k008), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k009), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k010), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k011), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k012), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k013), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k014), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k015), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k016), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k017), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k018), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k019), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k020), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k021), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k022), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k023), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k024), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k025), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k026), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k027), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k028), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k029), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k030), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k031), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k032), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k033), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k034), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k035), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k036), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k037), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k038), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k039), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k040), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k041), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k042), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k043), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k044), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k045), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k046), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k047), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k048), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k049), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k050), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k051), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k052), broadcast<Real>(k058));
    series2 = fma(series2, argument * broadcast<Real>(k053), broadcast<Real>(k058));
    Real const radial2 = broadcast<Real>(k072) * exponential * series2;
    output[4] = broadcast<Real>(k079) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k000);
    output[5] = broadcast<Real>(k068) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k000);
    output[6] = broadcast<Real>(k064) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k000);
    output[7] = broadcast<Real>(k068) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k000);
    output[8] = broadcast<Real>(k079) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k000);
    Real series3 = broadcast<Real>(k058);
    series3 = fma(series3, argument * broadcast<Real>(k005), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k006), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k007), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k008), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k009), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k010), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k011), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k012), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k013), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k014), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k015), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k016), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k017), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k018), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k019), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k020), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k021), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k022), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k023), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k024), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k025), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k026), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k027), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k028), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k029), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k030), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k031), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k032), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k033), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k034), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k035), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k036), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k037), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k038), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k039), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k040), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k041), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k042), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k043), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k044), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k045), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k046), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k047), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k048), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k049), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k050), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k051), broadcast<Real>(k058));
    series3 = fma(series3, argument * broadcast<Real>(k052), broadcast<Real>(k058));
    Real const radial3 = -broadcast<Real>(k075) * exponential * series3;
    output[9] = broadcast<Real>(k084) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k000);
    output[10] = broadcast<Real>(k082) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k000);
    output[11] = broadcast<Real>(k081) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k000);
    output[12] = broadcast<Real>(k080) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k000);
    output[13] = broadcast<Real>(k081) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k000);
    output[14] = broadcast<Real>(k082) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k000);
    output[15] = broadcast<Real>(k084) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k000);
    Real series4 = broadcast<Real>(k058);
    series4 = fma(series4, argument * broadcast<Real>(k004), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k005), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k006), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k007), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k008), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k009), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k010), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k011), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k012), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k013), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k014), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k015), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k016), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k017), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k018), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k019), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k020), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k021), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k022), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k023), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k024), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k025), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k026), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k027), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k028), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k029), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k030), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k031), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k032), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k033), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k034), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k035), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k036), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k037), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k038), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k039), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k040), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k041), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k042), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k043), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k044), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k045), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k046), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k047), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k048), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k049), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k050), broadcast<Real>(k058));
    series4 = fma(series4, argument * broadcast<Real>(k051), broadcast<Real>(k058));
    Real const radial4 = broadcast<Real>(k073) * exponential * series4;
    output[16] = broadcast<Real>(k090) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k000);
    output[17] = broadcast<Real>(k086) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k000);
    output[18] = broadcast<Real>(k085) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k000);
    output[19] = broadcast<Real>(k084) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k000);
    output[20] = broadcast<Real>(k083) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k000);
    output[21] = broadcast<Real>(k084) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k000);
    output[22] = broadcast<Real>(k085) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k000);
    output[23] = broadcast<Real>(k086) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k000);
    output[24] = broadcast<Real>(k090) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k000);
    Real series5 = broadcast<Real>(k058);
    series5 = fma(series5, argument * broadcast<Real>(k003), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k004), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k005), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k006), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k007), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k008), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k009), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k010), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k011), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k012), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k013), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k014), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k015), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k016), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k017), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k018), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k019), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k020), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k021), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k022), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k023), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k024), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k025), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k026), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k027), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k028), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k029), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k030), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k031), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k032), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k033), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k034), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k035), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k036), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k037), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k038), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k039), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k040), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k041), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k042), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k043), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k044), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k045), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k046), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k047), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k048), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k049), broadcast<Real>(k058));
    series5 = fma(series5, argument * broadcast<Real>(k050), broadcast<Real>(k058));
    Real const radial5 = -broadcast<Real>(k071) * exponential * series5;
    output[25] = broadcast<Real>(k097) * radial5 * harmonics[25];
    compensation[25] = broadcast<Real>(k000);
    output[26] = broadcast<Real>(k092) * radial5 * harmonics[26];
    compensation[26] = broadcast<Real>(k000);
    output[27] = broadcast<Real>(k091) * radial5 * harmonics[27];
    compensation[27] = broadcast<Real>(k000);
    output[28] = broadcast<Real>(k089) * radial5 * harmonics[28];
    compensation[28] = broadcast<Real>(k000);
    output[29] = broadcast<Real>(k088) * radial5 * harmonics[29];
    compensation[29] = broadcast<Real>(k000);
    output[30] = broadcast<Real>(k087) * radial5 * harmonics[30];
    compensation[30] = broadcast<Real>(k000);
    output[31] = broadcast<Real>(k088) * radial5 * harmonics[31];
    compensation[31] = broadcast<Real>(k000);
    output[32] = broadcast<Real>(k089) * radial5 * harmonics[32];
    compensation[32] = broadcast<Real>(k000);
    output[33] = broadcast<Real>(k091) * radial5 * harmonics[33];
    compensation[33] = broadcast<Real>(k000);
    output[34] = broadcast<Real>(k092) * radial5 * harmonics[34];
    compensation[34] = broadcast<Real>(k000);
    output[35] = broadcast<Real>(k097) * radial5 * harmonics[35];
    compensation[35] = broadcast<Real>(k000);
    Real series6 = broadcast<Real>(k058);
    series6 = fma(series6, argument * broadcast<Real>(k002), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k003), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k004), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k005), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k006), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k007), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k008), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k009), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k010), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k011), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k012), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k013), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k014), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k015), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k016), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k017), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k018), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k019), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k020), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k021), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k022), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k023), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k024), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k025), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k026), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k027), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k028), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k029), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k030), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k031), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k032), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k033), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k034), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k035), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k036), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k037), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k038), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k039), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k040), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k041), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k042), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k043), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k044), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k045), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k046), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k047), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k048), broadcast<Real>(k058));
    series6 = fma(series6, argument * broadcast<Real>(k049), broadcast<Real>(k058));
    Real const radial6 = broadcast<Real>(k066) * exponential * series6;
    output[36] = broadcast<Real>(k105) * radial6 * harmonics[36];
    compensation[36] = broadcast<Real>(k000);
    output[37] = broadcast<Real>(k101) * radial6 * harmonics[37];
    compensation[37] = broadcast<Real>(k000);
    output[38] = broadcast<Real>(k098) * radial6 * harmonics[38];
    compensation[38] = broadcast<Real>(k000);
    output[39] = broadcast<Real>(k096) * radial6 * harmonics[39];
    compensation[39] = broadcast<Real>(k000);
    output[40] = broadcast<Real>(k095) * radial6 * harmonics[40];
    compensation[40] = broadcast<Real>(k000);
    output[41] = broadcast<Real>(k094) * radial6 * harmonics[41];
    compensation[41] = broadcast<Real>(k000);
    output[42] = broadcast<Real>(k093) * radial6 * harmonics[42];
    compensation[42] = broadcast<Real>(k000);
    output[43] = broadcast<Real>(k094) * radial6 * harmonics[43];
    compensation[43] = broadcast<Real>(k000);
    output[44] = broadcast<Real>(k095) * radial6 * harmonics[44];
    compensation[44] = broadcast<Real>(k000);
    output[45] = broadcast<Real>(k096) * radial6 * harmonics[45];
    compensation[45] = broadcast<Real>(k000);
    output[46] = broadcast<Real>(k098) * radial6 * harmonics[46];
    compensation[46] = broadcast<Real>(k000);
    output[47] = broadcast<Real>(k101) * radial6 * harmonics[47];
    compensation[47] = broadcast<Real>(k000);
    output[48] = broadcast<Real>(k105) * radial6 * harmonics[48];
    compensation[48] = broadcast<Real>(k000);
    Real series7 = broadcast<Real>(k058);
    series7 = fma(series7, argument * broadcast<Real>(k001), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k002), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k003), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k004), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k005), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k006), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k007), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k008), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k009), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k010), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k011), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k012), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k013), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k014), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k015), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k016), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k017), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k018), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k019), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k020), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k021), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k022), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k023), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k024), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k025), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k026), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k027), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k028), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k029), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k030), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k031), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k032), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k033), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k034), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k035), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k036), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k037), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k038), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k039), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k040), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k041), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k042), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k043), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k044), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k045), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k046), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k047), broadcast<Real>(k058));
    series7 = fma(series7, argument * broadcast<Real>(k048), broadcast<Real>(k058));
    Real const radial7 = -broadcast<Real>(k061) * exponential * series7;
    output[49] = broadcast<Real>(k108) * radial7 * harmonics[49];
    compensation[49] = broadcast<Real>(k000);
    output[50] = broadcast<Real>(k107) * radial7 * harmonics[50];
    compensation[50] = broadcast<Real>(k000);
    output[51] = broadcast<Real>(k106) * radial7 * harmonics[51];
    compensation[51] = broadcast<Real>(k000);
    output[52] = broadcast<Real>(k104) * radial7 * harmonics[52];
    compensation[52] = broadcast<Real>(k000);
    output[53] = broadcast<Real>(k103) * radial7 * harmonics[53];
    compensation[53] = broadcast<Real>(k000);
    output[54] = broadcast<Real>(k102) * radial7 * harmonics[54];
    compensation[54] = broadcast<Real>(k000);
    output[55] = broadcast<Real>(k100) * radial7 * harmonics[55];
    compensation[55] = broadcast<Real>(k000);
    output[56] = broadcast<Real>(k099) * radial7 * harmonics[56];
    compensation[56] = broadcast<Real>(k000);
    output[57] = broadcast<Real>(k100) * radial7 * harmonics[57];
    compensation[57] = broadcast<Real>(k000);
    output[58] = broadcast<Real>(k102) * radial7 * harmonics[58];
    compensation[58] = broadcast<Real>(k000);
    output[59] = broadcast<Real>(k103) * radial7 * harmonics[59];
    compensation[59] = broadcast<Real>(k000);
    output[60] = broadcast<Real>(k104) * radial7 * harmonics[60];
    compensation[60] = broadcast<Real>(k000);
    output[61] = broadcast<Real>(k106) * radial7 * harmonics[61];
    compensation[61] = broadcast<Real>(k000);
    output[62] = broadcast<Real>(k107) * radial7 * harmonics[62];
    compensation[62] = broadcast<Real>(k000);
    output[63] = broadcast<Real>(k108) * radial7 * harmonics[63];
    compensation[63] = broadcast<Real>(k000);
    output[0] += broadcast<Real>(k057);
    greenTrace = broadcast<Real>(k065);
    for (int ix = -6; ix <= 6; ++ix) {
        for (int iy = -6; iy <= 6; ++iy) {
            for (int iz = -6; iz <= 6; ++iz) {
                int const latticeSquared = ix * ix + iy * iy + iz * iz;
                if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 || (ix == 0 && iy < 0) ||
                    (ix == 0 && iy == 0 && iz < 0)) {
                    continue;
                }
                for (int sign = -1; sign <= 1; sign += 2) {
                    Real const imageX = x + broadcast<Real>(sign * ix);
                    Real const imageY = y + broadcast<Real>(sign * iy);
                    Real const imageZ = z + broadcast<Real>(sign * iz);
                    Real const imageSquared =
                        fma(imageX, imageX, fma(imageY, imageY, imageZ * imageZ));
                    Real const imageRadius = sqrt(imageSquared);
                    Real const iR = broadcast<Real>(k058) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k064) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k059) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k060) * imageRadius * imageExp;
                    Real inversePower = iR;
                    Real const imageRadial0 = -screening * inversePower;
                    Real const imageValue0 = imageRadial0 * harmonics[0];
                    if (sign < 0) {
                        imagePair[0] = imageValue0;
                    } else {
                        imagePair[0] += imageValue0;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k056);
                    Real const imageRadial1 = screening * inversePower;
                    Real const imageValue1 = broadcast<Real>(k059) * imageRadial1 * harmonics[1];
                    if (sign < 0) {
                        imagePair[1] = imageValue1;
                    } else {
                        imagePair[1] += imageValue1;
                    }
                    Real const imageValue2 = imageRadial1 * harmonics[2];
                    if (sign < 0) {
                        imagePair[2] = imageValue2;
                    } else {
                        imagePair[2] += imageValue2;
                    }
                    Real const imageValue3 = broadcast<Real>(k059) * imageRadial1 * harmonics[3];
                    if (sign < 0) {
                        imagePair[3] = imageValue3;
                    } else {
                        imagePair[3] += imageValue3;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k054);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k079) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k068) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k064) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k068) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k079) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k053);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k084) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k082) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k081) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k080) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k081) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k082) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k084) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k052);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k090) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k086) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k085) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k084) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k083) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k084) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k085) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k086) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k090) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k051);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k097) * imageRadial5 * harmonics[25];
                    if (sign < 0) {
                        imagePair[25] = imageValue25;
                    } else {
                        imagePair[25] += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k092) * imageRadial5 * harmonics[26];
                    if (sign < 0) {
                        imagePair[26] = imageValue26;
                    } else {
                        imagePair[26] += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k091) * imageRadial5 * harmonics[27];
                    if (sign < 0) {
                        imagePair[27] = imageValue27;
                    } else {
                        imagePair[27] += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k089) * imageRadial5 * harmonics[28];
                    if (sign < 0) {
                        imagePair[28] = imageValue28;
                    } else {
                        imagePair[28] += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k088) * imageRadial5 * harmonics[29];
                    if (sign < 0) {
                        imagePair[29] = imageValue29;
                    } else {
                        imagePair[29] += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k087) * imageRadial5 * harmonics[30];
                    if (sign < 0) {
                        imagePair[30] = imageValue30;
                    } else {
                        imagePair[30] += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k088) * imageRadial5 * harmonics[31];
                    if (sign < 0) {
                        imagePair[31] = imageValue31;
                    } else {
                        imagePair[31] += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k089) * imageRadial5 * harmonics[32];
                    if (sign < 0) {
                        imagePair[32] = imageValue32;
                    } else {
                        imagePair[32] += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k091) * imageRadial5 * harmonics[33];
                    if (sign < 0) {
                        imagePair[33] = imageValue33;
                    } else {
                        imagePair[33] += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k092) * imageRadial5 * harmonics[34];
                    if (sign < 0) {
                        imagePair[34] = imageValue34;
                    } else {
                        imagePair[34] += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k097) * imageRadial5 * harmonics[35];
                    if (sign < 0) {
                        imagePair[35] = imageValue35;
                    } else {
                        imagePair[35] += imageValue35;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k050);
                    Real const imageRadial6 = -screening * inversePower;
                    Real const imageValue36 = broadcast<Real>(k105) * imageRadial6 * harmonics[36];
                    if (sign < 0) {
                        imagePair[36] = imageValue36;
                    } else {
                        imagePair[36] += imageValue36;
                    }
                    Real const imageValue37 = broadcast<Real>(k101) * imageRadial6 * harmonics[37];
                    if (sign < 0) {
                        imagePair[37] = imageValue37;
                    } else {
                        imagePair[37] += imageValue37;
                    }
                    Real const imageValue38 = broadcast<Real>(k098) * imageRadial6 * harmonics[38];
                    if (sign < 0) {
                        imagePair[38] = imageValue38;
                    } else {
                        imagePair[38] += imageValue38;
                    }
                    Real const imageValue39 = broadcast<Real>(k096) * imageRadial6 * harmonics[39];
                    if (sign < 0) {
                        imagePair[39] = imageValue39;
                    } else {
                        imagePair[39] += imageValue39;
                    }
                    Real const imageValue40 = broadcast<Real>(k095) * imageRadial6 * harmonics[40];
                    if (sign < 0) {
                        imagePair[40] = imageValue40;
                    } else {
                        imagePair[40] += imageValue40;
                    }
                    Real const imageValue41 = broadcast<Real>(k094) * imageRadial6 * harmonics[41];
                    if (sign < 0) {
                        imagePair[41] = imageValue41;
                    } else {
                        imagePair[41] += imageValue41;
                    }
                    Real const imageValue42 = broadcast<Real>(k093) * imageRadial6 * harmonics[42];
                    if (sign < 0) {
                        imagePair[42] = imageValue42;
                    } else {
                        imagePair[42] += imageValue42;
                    }
                    Real const imageValue43 = broadcast<Real>(k094) * imageRadial6 * harmonics[43];
                    if (sign < 0) {
                        imagePair[43] = imageValue43;
                    } else {
                        imagePair[43] += imageValue43;
                    }
                    Real const imageValue44 = broadcast<Real>(k095) * imageRadial6 * harmonics[44];
                    if (sign < 0) {
                        imagePair[44] = imageValue44;
                    } else {
                        imagePair[44] += imageValue44;
                    }
                    Real const imageValue45 = broadcast<Real>(k096) * imageRadial6 * harmonics[45];
                    if (sign < 0) {
                        imagePair[45] = imageValue45;
                    } else {
                        imagePair[45] += imageValue45;
                    }
                    Real const imageValue46 = broadcast<Real>(k098) * imageRadial6 * harmonics[46];
                    if (sign < 0) {
                        imagePair[46] = imageValue46;
                    } else {
                        imagePair[46] += imageValue46;
                    }
                    Real const imageValue47 = broadcast<Real>(k101) * imageRadial6 * harmonics[47];
                    if (sign < 0) {
                        imagePair[47] = imageValue47;
                    } else {
                        imagePair[47] += imageValue47;
                    }
                    Real const imageValue48 = broadcast<Real>(k105) * imageRadial6 * harmonics[48];
                    if (sign < 0) {
                        imagePair[48] = imageValue48;
                    } else {
                        imagePair[48] += imageValue48;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k049);
                    Real const imageRadial7 = screening * inversePower;
                    Real const imageValue49 = broadcast<Real>(k108) * imageRadial7 * harmonics[49];
                    if (sign < 0) {
                        imagePair[49] = imageValue49;
                    } else {
                        imagePair[49] += imageValue49;
                    }
                    Real const imageValue50 = broadcast<Real>(k107) * imageRadial7 * harmonics[50];
                    if (sign < 0) {
                        imagePair[50] = imageValue50;
                    } else {
                        imagePair[50] += imageValue50;
                    }
                    Real const imageValue51 = broadcast<Real>(k106) * imageRadial7 * harmonics[51];
                    if (sign < 0) {
                        imagePair[51] = imageValue51;
                    } else {
                        imagePair[51] += imageValue51;
                    }
                    Real const imageValue52 = broadcast<Real>(k104) * imageRadial7 * harmonics[52];
                    if (sign < 0) {
                        imagePair[52] = imageValue52;
                    } else {
                        imagePair[52] += imageValue52;
                    }
                    Real const imageValue53 = broadcast<Real>(k103) * imageRadial7 * harmonics[53];
                    if (sign < 0) {
                        imagePair[53] = imageValue53;
                    } else {
                        imagePair[53] += imageValue53;
                    }
                    Real const imageValue54 = broadcast<Real>(k102) * imageRadial7 * harmonics[54];
                    if (sign < 0) {
                        imagePair[54] = imageValue54;
                    } else {
                        imagePair[54] += imageValue54;
                    }
                    Real const imageValue55 = broadcast<Real>(k100) * imageRadial7 * harmonics[55];
                    if (sign < 0) {
                        imagePair[55] = imageValue55;
                    } else {
                        imagePair[55] += imageValue55;
                    }
                    Real const imageValue56 = broadcast<Real>(k099) * imageRadial7 * harmonics[56];
                    if (sign < 0) {
                        imagePair[56] = imageValue56;
                    } else {
                        imagePair[56] += imageValue56;
                    }
                    Real const imageValue57 = broadcast<Real>(k100) * imageRadial7 * harmonics[57];
                    if (sign < 0) {
                        imagePair[57] = imageValue57;
                    } else {
                        imagePair[57] += imageValue57;
                    }
                    Real const imageValue58 = broadcast<Real>(k102) * imageRadial7 * harmonics[58];
                    if (sign < 0) {
                        imagePair[58] = imageValue58;
                    } else {
                        imagePair[58] += imageValue58;
                    }
                    Real const imageValue59 = broadcast<Real>(k103) * imageRadial7 * harmonics[59];
                    if (sign < 0) {
                        imagePair[59] = imageValue59;
                    } else {
                        imagePair[59] += imageValue59;
                    }
                    Real const imageValue60 = broadcast<Real>(k104) * imageRadial7 * harmonics[60];
                    if (sign < 0) {
                        imagePair[60] = imageValue60;
                    } else {
                        imagePair[60] += imageValue60;
                    }
                    Real const imageValue61 = broadcast<Real>(k106) * imageRadial7 * harmonics[61];
                    if (sign < 0) {
                        imagePair[61] = imageValue61;
                    } else {
                        imagePair[61] += imageValue61;
                    }
                    Real const imageValue62 = broadcast<Real>(k107) * imageRadial7 * harmonics[62];
                    if (sign < 0) {
                        imagePair[62] = imageValue62;
                    } else {
                        imagePair[62] += imageValue62;
                    }
                    Real const imageValue63 = broadcast<Real>(k108) * imageRadial7 * harmonics[63];
                    if (sign < 0) {
                        imagePair[63] = imageValue63;
                    } else {
                        imagePair[63] += imageValue63;
                    }
                }
                detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                detail::ewaldAccumulate(output[25], compensation[25], imagePair[25]);
                detail::ewaldAccumulate(output[26], compensation[26], imagePair[26]);
                detail::ewaldAccumulate(output[27], compensation[27], imagePair[27]);
                detail::ewaldAccumulate(output[28], compensation[28], imagePair[28]);
                detail::ewaldAccumulate(output[29], compensation[29], imagePair[29]);
                detail::ewaldAccumulate(output[30], compensation[30], imagePair[30]);
                detail::ewaldAccumulate(output[31], compensation[31], imagePair[31]);
                detail::ewaldAccumulate(output[32], compensation[32], imagePair[32]);
                detail::ewaldAccumulate(output[33], compensation[33], imagePair[33]);
                detail::ewaldAccumulate(output[34], compensation[34], imagePair[34]);
                detail::ewaldAccumulate(output[35], compensation[35], imagePair[35]);
                detail::ewaldAccumulate(output[36], compensation[36], imagePair[36]);
                detail::ewaldAccumulate(output[37], compensation[37], imagePair[37]);
                detail::ewaldAccumulate(output[38], compensation[38], imagePair[38]);
                detail::ewaldAccumulate(output[39], compensation[39], imagePair[39]);
                detail::ewaldAccumulate(output[40], compensation[40], imagePair[40]);
                detail::ewaldAccumulate(output[41], compensation[41], imagePair[41]);
                detail::ewaldAccumulate(output[42], compensation[42], imagePair[42]);
                detail::ewaldAccumulate(output[43], compensation[43], imagePair[43]);
                detail::ewaldAccumulate(output[44], compensation[44], imagePair[44]);
                detail::ewaldAccumulate(output[45], compensation[45], imagePair[45]);
                detail::ewaldAccumulate(output[46], compensation[46], imagePair[46]);
                detail::ewaldAccumulate(output[47], compensation[47], imagePair[47]);
                detail::ewaldAccumulate(output[48], compensation[48], imagePair[48]);
                detail::ewaldAccumulate(output[49], compensation[49], imagePair[49]);
                detail::ewaldAccumulate(output[50], compensation[50], imagePair[50]);
                detail::ewaldAccumulate(output[51], compensation[51], imagePair[51]);
                detail::ewaldAccumulate(output[52], compensation[52], imagePair[52]);
                detail::ewaldAccumulate(output[53], compensation[53], imagePair[53]);
                detail::ewaldAccumulate(output[54], compensation[54], imagePair[54]);
                detail::ewaldAccumulate(output[55], compensation[55], imagePair[55]);
                detail::ewaldAccumulate(output[56], compensation[56], imagePair[56]);
                detail::ewaldAccumulate(output[57], compensation[57], imagePair[57]);
                detail::ewaldAccumulate(output[58], compensation[58], imagePair[58]);
                detail::ewaldAccumulate(output[59], compensation[59], imagePair[59]);
                detail::ewaldAccumulate(output[60], compensation[60], imagePair[60]);
                detail::ewaldAccumulate(output[61], compensation[61], imagePair[61]);
                detail::ewaldAccumulate(output[62], compensation[62], imagePair[62]);
                detail::ewaldAccumulate(output[63], compensation[63], imagePair[63]);
                Real const waveX = broadcast<Real>(ix);
                Real const waveY = broadcast<Real>(iy);
                Real const waveZ = broadcast<Real>(iz);
                Real const waveSquared = broadcast<Real>(latticeSquared);
                ewaldRegular<degree>(harmonics, waveX, waveY, waveZ, waveSquared);
                Real const phase = broadcast<Real>(k070) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k055) *
                    detail::ewaldExp(-broadcast<Real>(k062) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k058) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k070) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k059) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k059) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k076) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k079) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k068) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k064) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k068) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k079) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k078) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k084) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k082) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k081) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k080) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k081) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k082) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k084) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k077) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k090) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k086) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k085) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k084) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k083) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k084) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k085) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k086) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k090) * waveRadial4 * harmonics[24]);
                Real const waveRadial5 = broadcast<Real>(k074) * wavePotential * -sine;
                detail::ewaldAccumulate(output[25], compensation[25],
                    broadcast<Real>(k097) * waveRadial5 * harmonics[25]);
                detail::ewaldAccumulate(output[26], compensation[26],
                    broadcast<Real>(k092) * waveRadial5 * harmonics[26]);
                detail::ewaldAccumulate(output[27], compensation[27],
                    broadcast<Real>(k091) * waveRadial5 * harmonics[27]);
                detail::ewaldAccumulate(output[28], compensation[28],
                    broadcast<Real>(k089) * waveRadial5 * harmonics[28]);
                detail::ewaldAccumulate(output[29], compensation[29],
                    broadcast<Real>(k088) * waveRadial5 * harmonics[29]);
                detail::ewaldAccumulate(output[30], compensation[30],
                    broadcast<Real>(k087) * waveRadial5 * harmonics[30]);
                detail::ewaldAccumulate(output[31], compensation[31],
                    broadcast<Real>(k088) * waveRadial5 * harmonics[31]);
                detail::ewaldAccumulate(output[32], compensation[32],
                    broadcast<Real>(k089) * waveRadial5 * harmonics[32]);
                detail::ewaldAccumulate(output[33], compensation[33],
                    broadcast<Real>(k091) * waveRadial5 * harmonics[33]);
                detail::ewaldAccumulate(output[34], compensation[34],
                    broadcast<Real>(k092) * waveRadial5 * harmonics[34]);
                detail::ewaldAccumulate(output[35], compensation[35],
                    broadcast<Real>(k097) * waveRadial5 * harmonics[35]);
                Real const waveRadial6 = broadcast<Real>(k067) * wavePotential * -cosine;
                detail::ewaldAccumulate(output[36], compensation[36],
                    broadcast<Real>(k105) * waveRadial6 * harmonics[36]);
                detail::ewaldAccumulate(output[37], compensation[37],
                    broadcast<Real>(k101) * waveRadial6 * harmonics[37]);
                detail::ewaldAccumulate(output[38], compensation[38],
                    broadcast<Real>(k098) * waveRadial6 * harmonics[38]);
                detail::ewaldAccumulate(output[39], compensation[39],
                    broadcast<Real>(k096) * waveRadial6 * harmonics[39]);
                detail::ewaldAccumulate(output[40], compensation[40],
                    broadcast<Real>(k095) * waveRadial6 * harmonics[40]);
                detail::ewaldAccumulate(output[41], compensation[41],
                    broadcast<Real>(k094) * waveRadial6 * harmonics[41]);
                detail::ewaldAccumulate(output[42], compensation[42],
                    broadcast<Real>(k093) * waveRadial6 * harmonics[42]);
                detail::ewaldAccumulate(output[43], compensation[43],
                    broadcast<Real>(k094) * waveRadial6 * harmonics[43]);
                detail::ewaldAccumulate(output[44], compensation[44],
                    broadcast<Real>(k095) * waveRadial6 * harmonics[44]);
                detail::ewaldAccumulate(output[45], compensation[45],
                    broadcast<Real>(k096) * waveRadial6 * harmonics[45]);
                detail::ewaldAccumulate(output[46], compensation[46],
                    broadcast<Real>(k098) * waveRadial6 * harmonics[46]);
                detail::ewaldAccumulate(output[47], compensation[47],
                    broadcast<Real>(k101) * waveRadial6 * harmonics[47]);
                detail::ewaldAccumulate(output[48], compensation[48],
                    broadcast<Real>(k105) * waveRadial6 * harmonics[48]);
                Real const waveRadial7 = broadcast<Real>(k063) * wavePotential * sine;
                detail::ewaldAccumulate(output[49], compensation[49],
                    broadcast<Real>(k108) * waveRadial7 * harmonics[49]);
                detail::ewaldAccumulate(output[50], compensation[50],
                    broadcast<Real>(k107) * waveRadial7 * harmonics[50]);
                detail::ewaldAccumulate(output[51], compensation[51],
                    broadcast<Real>(k106) * waveRadial7 * harmonics[51]);
                detail::ewaldAccumulate(output[52], compensation[52],
                    broadcast<Real>(k104) * waveRadial7 * harmonics[52]);
                detail::ewaldAccumulate(output[53], compensation[53],
                    broadcast<Real>(k103) * waveRadial7 * harmonics[53]);
                detail::ewaldAccumulate(output[54], compensation[54],
                    broadcast<Real>(k102) * waveRadial7 * harmonics[54]);
                detail::ewaldAccumulate(output[55], compensation[55],
                    broadcast<Real>(k100) * waveRadial7 * harmonics[55]);
                detail::ewaldAccumulate(output[56], compensation[56],
                    broadcast<Real>(k099) * waveRadial7 * harmonics[56]);
                detail::ewaldAccumulate(output[57], compensation[57],
                    broadcast<Real>(k100) * waveRadial7 * harmonics[57]);
                detail::ewaldAccumulate(output[58], compensation[58],
                    broadcast<Real>(k102) * waveRadial7 * harmonics[58]);
                detail::ewaldAccumulate(output[59], compensation[59],
                    broadcast<Real>(k103) * waveRadial7 * harmonics[59]);
                detail::ewaldAccumulate(output[60], compensation[60],
                    broadcast<Real>(k104) * waveRadial7 * harmonics[60]);
                detail::ewaldAccumulate(output[61], compensation[61],
                    broadcast<Real>(k106) * waveRadial7 * harmonics[61]);
                detail::ewaldAccumulate(output[62], compensation[62],
                    broadcast<Real>(k107) * waveRadial7 * harmonics[62]);
                detail::ewaldAccumulate(output[63], compensation[63],
                    broadcast<Real>(k108) * waveRadial7 * harmonics[63]);
            }
        }
    }
}
/// Evaluate the periodic correction with promoted working precision, then convert the result.
template <int degree, typename Real, typename Output>
    requires(degree == 7)
CUDA_EXPORT void evaluateEwaldGreen(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    using Working = typename detail::EwaldWorkingType<Real>::Type;
    Working temporary[64];
    Working trace;
    evaluateEwaldWorking<degree, Working>(temporary, trace, detail::ewaldConvert<Working>(x),
        detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
    for (int i = 0; i < 64; ++i) {
        output[i] = detail::ewaldConvert<Real>(temporary[i]);
    }
    greenTrace = detail::ewaldConvert<Real>(trace);
}
#endif

#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= 9) && \
    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= 9)
template <>
inline constexpr int kEwaldGreenFlops<8> = 951153;
template <>
inline constexpr int kEwaldGreenConventionalFlops<8> = 1039431;
/// Assign regular solid harmonics using the supplied coordinates and squared radius.
template <int degree, typename Real>
    requires(degree == 8)
CUDA_EXPORT void ewaldRegular(Real* output, Real x, Real y, Real z, Real radiusSquared) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(1.562500000000000000E-02);
    constexpr auto k01 = static_cast<Scalar>(1.587301587301587213E-02);
    constexpr auto k02 = static_cast<Scalar>(1.666666666666666644E-02);
    constexpr auto k03 = static_cast<Scalar>(1.818181818181818094E-02);
    constexpr auto k04 = static_cast<Scalar>(2.040816326530612082E-02);
    constexpr auto k05 = static_cast<Scalar>(2.083333333333333218E-02);
    constexpr auto k06 = static_cast<Scalar>(2.222222222222222307E-02);
    constexpr auto k07 = static_cast<Scalar>(2.500000000000000139E-02);
    constexpr auto k08 = static_cast<Scalar>(2.564102564102564014E-02);
    constexpr auto k09 = static_cast<Scalar>(2.777777777777777624E-02);
    constexpr auto k10 = static_cast<Scalar>(2.857142857142857054E-02);
    constexpr auto k11 = static_cast<Scalar>(3.030303030303030387E-02);
    constexpr auto k12 = static_cast<Scalar>(3.125000000000000000E-02);
    constexpr auto k13 = static_cast<Scalar>(3.571428571428571230E-02);
    constexpr auto k14 = static_cast<Scalar>(3.703703703703703498E-02);
    constexpr auto k15 = static_cast<Scalar>(4.000000000000000083E-02);
    constexpr auto k16 = static_cast<Scalar>(4.166666666666666435E-02);
    constexpr auto k17 = static_cast<Scalar>(4.761904761904761640E-02);
    constexpr auto k18 = static_cast<Scalar>(5.000000000000000278E-02);
    constexpr auto k19 = static_cast<Scalar>(6.250000000000000000E-02);
    constexpr auto k20 = static_cast<Scalar>(6.666666666666666574E-02);
    constexpr auto k21 = static_cast<Scalar>(7.142857142857142461E-02);
    constexpr auto k22 = static_cast<Scalar>(8.333333333333332871E-02);
    constexpr auto k23 = static_cast<Scalar>(1.000000000000000056E-01);
    constexpr auto k24 = static_cast<Scalar>(1.111111111111111049E-01);
    constexpr auto k25 = static_cast<Scalar>(1.250000000000000000E-01);
    constexpr auto k26 = static_cast<Scalar>(1.666666666666666574E-01);
    constexpr auto k27 = static_cast<Scalar>(2.500000000000000000E-01);
    constexpr auto k28 = static_cast<Scalar>(5.000000000000000000E-01);
    constexpr auto k29 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k30 = static_cast<Scalar>(3.000000000000000000E+00);
    constexpr auto k31 = static_cast<Scalar>(5.000000000000000000E+00);
    constexpr auto k32 = static_cast<Scalar>(7.000000000000000000E+00);
    constexpr auto k33 = static_cast<Scalar>(9.000000000000000000E+00);
    constexpr auto k34 = static_cast<Scalar>(1.100000000000000000E+01);
    constexpr auto k35 = static_cast<Scalar>(1.300000000000000000E+01);
    constexpr auto k36 = static_cast<Scalar>(1.500000000000000000E+01);
    output[0] = broadcast<Real>(k29);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k30) * z;
    output[6] = broadcast<Real>(k27) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k31) * z;
    output[12] = broadcast<Real>(k24) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k32) * z;
    output[20] = broadcast<Real>(k19) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    Real const zFactor5_0 = broadcast<Real>(k33) * z;
    output[30] = broadcast<Real>(k15) * fma(zFactor5_0, output[20], -radiusSquared * output[12]);
    Real const zFactor6_0 = broadcast<Real>(k34) * z;
    output[42] = broadcast<Real>(k09) * fma(zFactor6_0, output[30], -radiusSquared * output[20]);
    Real const zFactor7_0 = broadcast<Real>(k35) * z;
    output[56] = broadcast<Real>(k04) * fma(zFactor7_0, output[42], -radiusSquared * output[30]);
    Real const zFactor8_0 = broadcast<Real>(k36) * z;
    output[72] = broadcast<Real>(k00) * fma(zFactor8_0, output[56], -radiusSquared * output[42]);
    output[3] = x * broadcast<Real>(k28);
    output[1] = y * broadcast<Real>(k28);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k31) * z;
    output[13] = broadcast<Real>(k25) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k25) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k32) * z;
    output[21] = broadcast<Real>(k20) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k20) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    Real const zFactor5_1 = broadcast<Real>(k33) * z;
    output[31] = broadcast<Real>(k16) * fma(zFactor5_1, output[21], -radiusSquared * output[13]);
    output[29] = broadcast<Real>(k16) * fma(zFactor5_1, output[19], -radiusSquared * output[11]);
    Real const zFactor6_1 = broadcast<Real>(k34) * z;
    output[43] = broadcast<Real>(k10) * fma(zFactor6_1, output[31], -radiusSquared * output[21]);
    output[41] = broadcast<Real>(k10) * fma(zFactor6_1, output[29], -radiusSquared * output[19]);
    Real const zFactor7_1 = broadcast<Real>(k35) * z;
    output[57] = broadcast<Real>(k05) * fma(zFactor7_1, output[43], -radiusSquared * output[31]);
    output[55] = broadcast<Real>(k05) * fma(zFactor7_1, output[41], -radiusSquared * output[29]);
    Real const zFactor8_1 = broadcast<Real>(k36) * z;
    output[73] = broadcast<Real>(k01) * fma(zFactor8_1, output[57], -radiusSquared * output[43]);
    output[71] = broadcast<Real>(k01) * fma(zFactor8_1, output[55], -radiusSquared * output[41]);
    output[8] = broadcast<Real>(k27) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k27) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k32) * z;
    output[22] = broadcast<Real>(k22) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k22) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    Real const zFactor5_2 = broadcast<Real>(k33) * z;
    output[32] = broadcast<Real>(k17) * fma(zFactor5_2, output[22], -radiusSquared * output[14]);
    output[28] = broadcast<Real>(k17) * fma(zFactor5_2, output[18], -radiusSquared * output[10]);
    Real const zFactor6_2 = broadcast<Real>(k34) * z;
    output[44] = broadcast<Real>(k12) * fma(zFactor6_2, output[32], -radiusSquared * output[22]);
    output[40] = broadcast<Real>(k12) * fma(zFactor6_2, output[28], -radiusSquared * output[18]);
    Real const zFactor7_2 = broadcast<Real>(k35) * z;
    output[58] = broadcast<Real>(k06) * fma(zFactor7_2, output[44], -radiusSquared * output[32]);
    output[54] = broadcast<Real>(k06) * fma(zFactor7_2, output[40], -radiusSquared * output[28]);
    Real const zFactor8_2 = broadcast<Real>(k36) * z;
    output[74] = broadcast<Real>(k02) * fma(zFactor8_2, output[58], -radiusSquared * output[44]);
    output[70] = broadcast<Real>(k02) * fma(zFactor8_2, output[54], -radiusSquared * output[40]);
    output[15] = broadcast<Real>(k26) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k26) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    Real const zFactor5_3 = broadcast<Real>(k33) * z;
    output[33] = broadcast<Real>(k19) * fma(zFactor5_3, output[23], -radiusSquared * output[15]);
    output[27] = broadcast<Real>(k19) * fma(zFactor5_3, output[17], -radiusSquared * output[9]);
    Real const zFactor6_3 = broadcast<Real>(k34) * z;
    output[45] = broadcast<Real>(k14) * fma(zFactor6_3, output[33], -radiusSquared * output[23]);
    output[39] = broadcast<Real>(k14) * fma(zFactor6_3, output[27], -radiusSquared * output[17]);
    Real const zFactor7_3 = broadcast<Real>(k35) * z;
    output[59] = broadcast<Real>(k07) * fma(zFactor7_3, output[45], -radiusSquared * output[33]);
    output[53] = broadcast<Real>(k07) * fma(zFactor7_3, output[39], -radiusSquared * output[27]);
    Real const zFactor8_3 = broadcast<Real>(k36) * z;
    output[75] = broadcast<Real>(k03) * fma(zFactor8_3, output[59], -radiusSquared * output[45]);
    output[69] = broadcast<Real>(k03) * fma(zFactor8_3, output[53], -radiusSquared * output[39]);
    output[24] = broadcast<Real>(k25) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k25) * fma(y, output[15], x * output[9]);
    output[34] = z * output[24];
    output[26] = z * output[16];
    Real const zFactor6_4 = broadcast<Real>(k34) * z;
    output[46] = broadcast<Real>(k18) * fma(zFactor6_4, output[34], -radiusSquared * output[24]);
    output[38] = broadcast<Real>(k18) * fma(zFactor6_4, output[26], -radiusSquared * output[16]);
    Real const zFactor7_4 = broadcast<Real>(k35) * z;
    output[60] = broadcast<Real>(k11) * fma(zFactor7_4, output[46], -radiusSquared * output[34]);
    output[52] = broadcast<Real>(k11) * fma(zFactor7_4, output[38], -radiusSquared * output[26]);
    Real const zFactor8_4 = broadcast<Real>(k36) * z;
    output[76] = broadcast<Real>(k05) * fma(zFactor8_4, output[60], -radiusSquared * output[46]);
    output[68] = broadcast<Real>(k05) * fma(zFactor8_4, output[52], -radiusSquared * output[38]);
    output[35] = broadcast<Real>(k23) * fma(x, output[24], -y * output[16]);
    output[25] = broadcast<Real>(k23) * fma(y, output[24], x * output[16]);
    output[47] = z * output[35];
    output[37] = z * output[25];
    Real const zFactor7_5 = broadcast<Real>(k35) * z;
    output[61] = broadcast<Real>(k16) * fma(zFactor7_5, output[47], -radiusSquared * output[35]);
    output[51] = broadcast<Real>(k16) * fma(zFactor7_5, output[37], -radiusSquared * output[25]);
    Real const zFactor8_5 = broadcast<Real>(k36) * z;
    output[77] = broadcast<Real>(k08) * fma(zFactor8_5, output[61], -radiusSquared * output[47]);
    output[67] = broadcast<Real>(k08) * fma(zFactor8_5, output[51], -radiusSquared * output[37]);
    output[48] = broadcast<Real>(k22) * fma(x, output[35], -y * output[25]);
    output[36] = broadcast<Real>(k22) * fma(y, output[35], x * output[25]);
    output[62] = z * output[48];
    output[50] = z * output[36];
    Real const zFactor8_6 = broadcast<Real>(k36) * z;
    output[78] = broadcast<Real>(k13) * fma(zFactor8_6, output[62], -radiusSquared * output[48]);
    output[66] = broadcast<Real>(k13) * fma(zFactor8_6, output[50], -radiusSquared * output[36]);
    output[63] = broadcast<Real>(k21) * fma(x, output[48], -y * output[36]);
    output[49] = broadcast<Real>(k21) * fma(y, output[48], x * output[36]);
    output[79] = z * output[63];
    output[65] = z * output[49];
    output[80] = broadcast<Real>(k19) * fma(x, output[63], -y * output[49]);
    output[64] = broadcast<Real>(k19) * fma(y, output[63], x * output[49]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 8)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k000 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k001 = static_cast<Scalar>(1.769911504424778736E-02);
    constexpr auto k002 = static_cast<Scalar>(1.801801801801801786E-02);
    constexpr auto k003 = static_cast<Scalar>(1.834862385321101019E-02);
    constexpr auto k004 = static_cast<Scalar>(1.869158878504672758E-02);
    constexpr auto k005 = static_cast<Scalar>(1.904761904761904934E-02);
    constexpr auto k006 = static_cast<Scalar>(1.941747572815533812E-02);
    constexpr auto k007 = static_cast<Scalar>(1.980198019801980222E-02);
    constexpr auto k008 = static_cast<Scalar>(2.020202020202020374E-02);
    constexpr auto k009 = static_cast<Scalar>(2.061855670103092744E-02);
    constexpr auto k010 = static_cast<Scalar>(2.105263157894736795E-02);
    constexpr auto k011 = static_cast<Scalar>(2.150537634408602322E-02);
    constexpr auto k012 = static_cast<Scalar>(2.197802197802197974E-02);
    constexpr auto k013 = static_cast<Scalar>(2.247191011235954994E-02);
    constexpr auto k014 = static_cast<Scalar>(2.298850574712643646E-02);
    constexpr auto k015 = static_cast<Scalar>(2.352941176470588203E-02);
    constexpr auto k016 = static_cast<Scalar>(2.409638554216867595E-02);
    constexpr auto k017 = static_cast<Scalar>(2.469135802469135665E-02);
    constexpr auto k018 = static_cast<Scalar>(2.531645569620253125E-02);
    constexpr auto k019 = static_cast<Scalar>(2.597402597402597574E-02);
    constexpr auto k020 = static_cast<Scalar>(2.666666666666666838E-02);
    constexpr auto k021 = static_cast<Scalar>(2.739726027397260122E-02);
    constexpr auto k022 = static_cast<Scalar>(2.816901408450704289E-02);
    constexpr auto k023 = static_cast<Scalar>(2.898550724637681195E-02);
    constexpr auto k024 = static_cast<Scalar>(2.985074626865671585E-02);
    constexpr auto k025 = static_cast<Scalar>(3.076923076923077094E-02);
    constexpr auto k026 = static_cast<Scalar>(3.174603174603174427E-02);
    constexpr auto k027 = static_cast<Scalar>(3.278688524590164105E-02);
    constexpr auto k028 = static_cast<Scalar>(3.389830508474576259E-02);
    constexpr auto k029 = static_cast<Scalar>(3.508771929824561209E-02);
    constexpr auto k030 = static_cast<Scalar>(3.636363636363636187E-02);
    constexpr auto k031 = static_cast<Scalar>(3.773584905660377214E-02);
    constexpr auto k032 = static_cast<Scalar>(3.921568627450980338E-02);
    constexpr auto k033 = static_cast<Scalar>(4.081632653061224164E-02);
    constexpr auto k034 = static_cast<Scalar>(4.255319148936170109E-02);
    constexpr auto k035 = static_cast<Scalar>(4.444444444444444614E-02);
    constexpr auto k036 = static_cast<Scalar>(4.651162790697674354E-02);
    constexpr auto k037 = static_cast<Scalar>(4.878048780487805047E-02);
    constexpr auto k038 = static_cast<Scalar>(5.128205128205128027E-02);
    constexpr auto k039 = static_cast<Scalar>(5.405405405405405705E-02);
    constexpr auto k040 = static_cast<Scalar>(5.714285714285714107E-02);
    constexpr auto k041 = static_cast<Scalar>(6.060606060606060774E-02);
    constexpr auto k042 = static_cast<Scalar>(6.451612903225806273E-02);
    constexpr auto k043 = static_cast<Scalar>(6.896551724137930939E-02);
    constexpr auto k044 = static_cast<Scalar>(7.407407407407406996E-02);
    constexpr auto k045 = static_cast<Scalar>(8.000000000000000167E-02);
    constexpr auto k046 = static_cast<Scalar>(8.695652173913043237E-02);
    constexpr auto k047 = static_cast<Scalar>(9.523809523809523281E-02);
    constexpr auto k048 = static_cast<Scalar>(1.052631578947368363E-01);
    constexpr auto k049 = static_cast<Scalar>(1.176470588235294101E-01);
    constexpr auto k050 = static_cast<Scalar>(1.333333333333333315E-01);
    constexpr auto k051 = static_cast<Scalar>(1.538461538461538547E-01);
    constexpr auto k052 = static_cast<Scalar>(1.818181818181818232E-01);
    constexpr auto k053 = static_cast<Scalar>(2.222222222222222099E-01);
    constexpr auto k054 = static_cast<Scalar>(2.857142857142856984E-01);
    constexpr auto k055 = static_cast<Scalar>(4.000000000000000222E-01);
    constexpr auto k056 = static_cast<Scalar>(6.366197723675813824E-01);
    constexpr auto k057 = static_cast<Scalar>(6.666666666666666297E-01);
    constexpr auto k058 = static_cast<Scalar>(7.853981633974482790E-01);
    constexpr auto k059 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k060 = static_cast<Scalar>(1.098745032237856911E+00);
    constexpr auto k061 = static_cast<Scalar>(1.198339408795681349E+00);
    constexpr auto k062 = static_cast<Scalar>(2.000000000000000000E+00);
    constexpr auto k063 = static_cast<Scalar>(2.256758334191025117E+00);
    constexpr auto k064 = static_cast<Scalar>(2.334833193505446047E+00);
    constexpr auto k065 = static_cast<Scalar>(2.467401100272339498E+00);
    constexpr auto k066 = static_cast<Scalar>(2.860824606174782847E+00);
    constexpr auto k067 = static_cast<Scalar>(4.000000000000000000E+00);
    constexpr auto k068 = static_cast<Scalar>(4.188790204786390525E+00);
    constexpr auto k069 = static_cast<Scalar>(4.377812237822711339E+00);
    constexpr auto k070 = static_cast<Scalar>(5.919086906091339806E+00);
    constexpr auto k071 = static_cast<Scalar>(6.000000000000000000E+00);
    constexpr auto k072 = static_cast<Scalar>(6.018022224509399720E+00);
    constexpr auto k073 = static_cast<Scalar>(6.283185307179586232E+00);
    constexpr auto k074 = static_cast<Scalar>(7.113944886461904815E+00);
    constexpr auto k075 = static_cast<Scalar>(9.628835559215042750E+00);
    constexpr auto k076 = static_cast<Scalar>(9.781674218885120453E+00);
    constexpr auto k077 = static_cast<Scalar>(1.036257133664445007E+01);
    constexpr auto k078 = static_cast<Scalar>(1.100438349624576340E+01);
    constexpr auto k079 = static_cast<Scalar>(1.315947253478581125E+01);
    constexpr auto k080 = static_cast<Scalar>(1.484329006232417747E+01);
    constexpr auto k081 = static_cast<Scalar>(1.653668089615990056E+01);
    constexpr auto k082 = static_cast<Scalar>(2.400000000000000000E+01);
    constexpr auto k083 = static_cast<Scalar>(3.600000000000000000E+01);
    constexpr auto k084 = static_cast<Scalar>(4.800000000000000000E+01);
    constexpr auto k085 = static_cast<Scalar>(1.200000000000000000E+02);
    constexpr auto k086 = static_cast<Scalar>(5.760000000000000000E+02);
    constexpr auto k087 = static_cast<Scalar>(7.200000000000000000E+02);
    constexpr auto k088 = static_cast<Scalar>(1.440000000000000000E+03);
    constexpr auto k089 = static_cast<Scalar>(5.040000000000000000E+03);
    constexpr auto k090 = static_cast<Scalar>(1.440000000000000000E+04);
    constexpr auto k091 = static_cast<Scalar>(1.728000000000000000E+04);
    constexpr auto k092 = static_cast<Scalar>(3.024000000000000000E+04);
    constexpr auto k093 = static_cast<Scalar>(4.032000000000000000E+04);
    constexpr auto k094 = static_cast<Scalar>(8.064000000000000000E+04);
    constexpr auto k095 = static_cast<Scalar>(3.628800000000000000E+05);
    constexpr auto k096 = static_cast<Scalar>(5.184000000000000000E+05);
    constexpr auto k097 = static_cast<Scalar>(6.048000000000000000E+05);
    constexpr auto k098 = static_cast<Scalar>(9.676800000000000000E+05);
    constexpr auto k099 = static_cast<Scalar>(2.177280000000000000E+06);
    constexpr auto k100 = static_cast<Scalar>(3.628800000000000000E+06);
    constexpr auto k101 = static_cast<Scalar>(7.257600000000000000E+06);
    constexpr auto k102 = static_cast<Scalar>(2.540160000000000000E+07);
    constexpr auto k103 = static_cast<Scalar>(2.903040000000000000E+07);
    constexpr auto k104 = static_cast<Scalar>(3.991680000000000000E+07);
    constexpr auto k105 = static_cast<Scalar>(4.354560000000000000E+07);
    constexpr auto k106 = static_cast<Scalar>(8.709120000000000000E+07);
    constexpr auto k107 = static_cast<Scalar>(2.395008000000000000E+08);
    constexpr auto k108 = static_cast<Scalar>(4.790016000000000000E+08);
    constexpr auto k109 = static_cast<Scalar>(9.580032000000000000E+08);
    constexpr auto k110 = static_cast<Scalar>(1.625702400000000000E+09);
    constexpr auto k111 = static_cast<Scalar>(1.828915200000000000E+09);
    constexpr auto k112 = static_cast<Scalar>(2.612736000000000000E+09);
    constexpr auto k113 = static_cast<Scalar>(4.790016000000000000E+09);
    constexpr auto k114 = static_cast<Scalar>(6.227020800000000000E+09);
    constexpr auto k115 = static_cast<Scalar>(1.149603840000000000E+10);
    constexpr auto k116 = static_cast<Scalar>(3.736212480000000000E+10);
    constexpr auto k117 = static_cast<Scalar>(8.717829120000000000E+10);
    constexpr auto k118 = static_cast<Scalar>(1.743565824000000000E+11);
    constexpr auto k119 = static_cast<Scalar>(1.307674368000000000E+12);
    constexpr auto k120 = static_cast<Scalar>(2.092278988800000000E+13);
    Real harmonics[81];
    Real compensation[81];
    Real imagePair[81];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k067) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k059);
    series0 = fma(series0, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k049), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k050), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k051), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k052), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k053), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k054), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k055), broadcast<Real>(k059));
    series0 = fma(series0, argument * broadcast<Real>(k057), broadcast<Real>(k059));
    Real const radial0 = broadcast<Real>(k063) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k000);
    Real series1 = broadcast<Real>(k059);
    series1 = fma(series1, argument * broadcast<Real>(k008), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k049), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k050), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k051), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k052), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k053), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k054), broadcast<Real>(k059));
    series1 = fma(series1, argument * broadcast<Real>(k055), broadcast<Real>(k059));
    Real const radial1 = -broadcast<Real>(k072) * exponential * series1;
    output[1] = broadcast<Real>(k062) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k000);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k000);
    output[3] = broadcast<Real>(k062) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k000);
    Real series2 = broadcast<Real>(k059);
    series2 = fma(series2, argument * broadcast<Real>(k007), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k008), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k049), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k050), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k051), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k052), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k053), broadcast<Real>(k059));
    series2 = fma(series2, argument * broadcast<Real>(k054), broadcast<Real>(k059));
    Real const radial2 = broadcast<Real>(k075) * exponential * series2;
    output[4] = broadcast<Real>(k082) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k000);
    output[5] = broadcast<Real>(k071) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k000);
    output[6] = broadcast<Real>(k067) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k000);
    output[7] = broadcast<Real>(k071) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k000);
    output[8] = broadcast<Real>(k082) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k000);
    Real series3 = broadcast<Real>(k059);
    series3 = fma(series3, argument * broadcast<Real>(k006), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k007), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k008), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k049), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k050), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k051), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k052), broadcast<Real>(k059));
    series3 = fma(series3, argument * broadcast<Real>(k053), broadcast<Real>(k059));
    Real const radial3 = -broadcast<Real>(k078) * exponential * series3;
    output[9] = broadcast<Real>(k087) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k000);
    output[10] = broadcast<Real>(k085) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k000);
    output[11] = broadcast<Real>(k084) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k000);
    output[12] = broadcast<Real>(k083) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k000);
    output[13] = broadcast<Real>(k084) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k000);
    output[14] = broadcast<Real>(k085) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k000);
    output[15] = broadcast<Real>(k087) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k000);
    Real series4 = broadcast<Real>(k059);
    series4 = fma(series4, argument * broadcast<Real>(k005), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k006), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k007), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k008), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k049), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k050), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k051), broadcast<Real>(k059));
    series4 = fma(series4, argument * broadcast<Real>(k052), broadcast<Real>(k059));
    Real const radial4 = broadcast<Real>(k076) * exponential * series4;
    output[16] = broadcast<Real>(k093) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k000);
    output[17] = broadcast<Real>(k089) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k000);
    output[18] = broadcast<Real>(k088) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k000);
    output[19] = broadcast<Real>(k087) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k000);
    output[20] = broadcast<Real>(k086) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k000);
    output[21] = broadcast<Real>(k087) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k000);
    output[22] = broadcast<Real>(k088) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k000);
    output[23] = broadcast<Real>(k089) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k000);
    output[24] = broadcast<Real>(k093) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k000);
    Real series5 = broadcast<Real>(k059);
    series5 = fma(series5, argument * broadcast<Real>(k004), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k005), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k006), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k007), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k008), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k049), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k050), broadcast<Real>(k059));
    series5 = fma(series5, argument * broadcast<Real>(k051), broadcast<Real>(k059));
    Real const radial5 = -broadcast<Real>(k074) * exponential * series5;
    output[25] = broadcast<Real>(k100) * radial5 * harmonics[25];
    compensation[25] = broadcast<Real>(k000);
    output[26] = broadcast<Real>(k095) * radial5 * harmonics[26];
    compensation[26] = broadcast<Real>(k000);
    output[27] = broadcast<Real>(k094) * radial5 * harmonics[27];
    compensation[27] = broadcast<Real>(k000);
    output[28] = broadcast<Real>(k092) * radial5 * harmonics[28];
    compensation[28] = broadcast<Real>(k000);
    output[29] = broadcast<Real>(k091) * radial5 * harmonics[29];
    compensation[29] = broadcast<Real>(k000);
    output[30] = broadcast<Real>(k090) * radial5 * harmonics[30];
    compensation[30] = broadcast<Real>(k000);
    output[31] = broadcast<Real>(k091) * radial5 * harmonics[31];
    compensation[31] = broadcast<Real>(k000);
    output[32] = broadcast<Real>(k092) * radial5 * harmonics[32];
    compensation[32] = broadcast<Real>(k000);
    output[33] = broadcast<Real>(k094) * radial5 * harmonics[33];
    compensation[33] = broadcast<Real>(k000);
    output[34] = broadcast<Real>(k095) * radial5 * harmonics[34];
    compensation[34] = broadcast<Real>(k000);
    output[35] = broadcast<Real>(k100) * radial5 * harmonics[35];
    compensation[35] = broadcast<Real>(k000);
    Real series6 = broadcast<Real>(k059);
    series6 = fma(series6, argument * broadcast<Real>(k003), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k004), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k005), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k006), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k007), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k008), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k049), broadcast<Real>(k059));
    series6 = fma(series6, argument * broadcast<Real>(k050), broadcast<Real>(k059));
    Real const radial6 = broadcast<Real>(k069) * exponential * series6;
    output[36] = broadcast<Real>(k108) * radial6 * harmonics[36];
    compensation[36] = broadcast<Real>(k000);
    output[37] = broadcast<Real>(k104) * radial6 * harmonics[37];
    compensation[37] = broadcast<Real>(k000);
    output[38] = broadcast<Real>(k101) * radial6 * harmonics[38];
    compensation[38] = broadcast<Real>(k000);
    output[39] = broadcast<Real>(k099) * radial6 * harmonics[39];
    compensation[39] = broadcast<Real>(k000);
    output[40] = broadcast<Real>(k098) * radial6 * harmonics[40];
    compensation[40] = broadcast<Real>(k000);
    output[41] = broadcast<Real>(k097) * radial6 * harmonics[41];
    compensation[41] = broadcast<Real>(k000);
    output[42] = broadcast<Real>(k096) * radial6 * harmonics[42];
    compensation[42] = broadcast<Real>(k000);
    output[43] = broadcast<Real>(k097) * radial6 * harmonics[43];
    compensation[43] = broadcast<Real>(k000);
    output[44] = broadcast<Real>(k098) * radial6 * harmonics[44];
    compensation[44] = broadcast<Real>(k000);
    output[45] = broadcast<Real>(k099) * radial6 * harmonics[45];
    compensation[45] = broadcast<Real>(k000);
    output[46] = broadcast<Real>(k101) * radial6 * harmonics[46];
    compensation[46] = broadcast<Real>(k000);
    output[47] = broadcast<Real>(k104) * radial6 * harmonics[47];
    compensation[47] = broadcast<Real>(k000);
    output[48] = broadcast<Real>(k108) * radial6 * harmonics[48];
    compensation[48] = broadcast<Real>(k000);
    Real series7 = broadcast<Real>(k059);
    series7 = fma(series7, argument * broadcast<Real>(k002), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k003), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k004), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k005), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k006), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k007), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k008), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    series7 = fma(series7, argument * broadcast<Real>(k049), broadcast<Real>(k059));
    Real const radial7 = -broadcast<Real>(k064) * exponential * series7;
    output[49] = broadcast<Real>(k117) * radial7 * harmonics[49];
    compensation[49] = broadcast<Real>(k000);
    output[50] = broadcast<Real>(k114) * radial7 * harmonics[50];
    compensation[50] = broadcast<Real>(k000);
    output[51] = broadcast<Real>(k109) * radial7 * harmonics[51];
    compensation[51] = broadcast<Real>(k000);
    output[52] = broadcast<Real>(k107) * radial7 * harmonics[52];
    compensation[52] = broadcast<Real>(k000);
    output[53] = broadcast<Real>(k106) * radial7 * harmonics[53];
    compensation[53] = broadcast<Real>(k000);
    output[54] = broadcast<Real>(k105) * radial7 * harmonics[54];
    compensation[54] = broadcast<Real>(k000);
    output[55] = broadcast<Real>(k103) * radial7 * harmonics[55];
    compensation[55] = broadcast<Real>(k000);
    output[56] = broadcast<Real>(k102) * radial7 * harmonics[56];
    compensation[56] = broadcast<Real>(k000);
    output[57] = broadcast<Real>(k103) * radial7 * harmonics[57];
    compensation[57] = broadcast<Real>(k000);
    output[58] = broadcast<Real>(k105) * radial7 * harmonics[58];
    compensation[58] = broadcast<Real>(k000);
    output[59] = broadcast<Real>(k106) * radial7 * harmonics[59];
    compensation[59] = broadcast<Real>(k000);
    output[60] = broadcast<Real>(k107) * radial7 * harmonics[60];
    compensation[60] = broadcast<Real>(k000);
    output[61] = broadcast<Real>(k109) * radial7 * harmonics[61];
    compensation[61] = broadcast<Real>(k000);
    output[62] = broadcast<Real>(k114) * radial7 * harmonics[62];
    compensation[62] = broadcast<Real>(k000);
    output[63] = broadcast<Real>(k117) * radial7 * harmonics[63];
    compensation[63] = broadcast<Real>(k000);
    Real series8 = broadcast<Real>(k059);
    series8 = fma(series8, argument * broadcast<Real>(k001), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k002), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k003), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k004), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k005), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k006), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k007), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k008), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k009), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k010), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k011), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k012), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k013), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k014), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k015), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k016), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k017), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k018), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k019), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k020), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k021), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k022), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k023), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k024), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k025), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k026), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k027), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k028), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k029), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k030), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k031), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k032), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k033), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k034), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k035), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k036), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k037), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k038), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k039), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k040), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k041), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k042), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k043), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k044), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k045), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k046), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k047), broadcast<Real>(k059));
    series8 = fma(series8, argument * broadcast<Real>(k048), broadcast<Real>(k059));
    Real const radial8 = broadcast<Real>(k060) * exponential * series8;
    output[64] = broadcast<Real>(k120) * radial8 * harmonics[64];
    compensation[64] = broadcast<Real>(k000);
    output[65] = broadcast<Real>(k119) * radial8 * harmonics[65];
    compensation[65] = broadcast<Real>(k000);
    output[66] = broadcast<Real>(k118) * radial8 * harmonics[66];
    compensation[66] = broadcast<Real>(k000);
    output[67] = broadcast<Real>(k116) * radial8 * harmonics[67];
    compensation[67] = broadcast<Real>(k000);
    output[68] = broadcast<Real>(k115) * radial8 * harmonics[68];
    compensation[68] = broadcast<Real>(k000);
    output[69] = broadcast<Real>(k113) * radial8 * harmonics[69];
    compensation[69] = broadcast<Real>(k000);
    output[70] = broadcast<Real>(k112) * radial8 * harmonics[70];
    compensation[70] = broadcast<Real>(k000);
    output[71] = broadcast<Real>(k111) * radial8 * harmonics[71];
    compensation[71] = broadcast<Real>(k000);
    output[72] = broadcast<Real>(k110) * radial8 * harmonics[72];
    compensation[72] = broadcast<Real>(k000);
    output[73] = broadcast<Real>(k111) * radial8 * harmonics[73];
    compensation[73] = broadcast<Real>(k000);
    output[74] = broadcast<Real>(k112) * radial8 * harmonics[74];
    compensation[74] = broadcast<Real>(k000);
    output[75] = broadcast<Real>(k113) * radial8 * harmonics[75];
    compensation[75] = broadcast<Real>(k000);
    output[76] = broadcast<Real>(k115) * radial8 * harmonics[76];
    compensation[76] = broadcast<Real>(k000);
    output[77] = broadcast<Real>(k116) * radial8 * harmonics[77];
    compensation[77] = broadcast<Real>(k000);
    output[78] = broadcast<Real>(k118) * radial8 * harmonics[78];
    compensation[78] = broadcast<Real>(k000);
    output[79] = broadcast<Real>(k119) * radial8 * harmonics[79];
    compensation[79] = broadcast<Real>(k000);
    output[80] = broadcast<Real>(k120) * radial8 * harmonics[80];
    compensation[80] = broadcast<Real>(k000);
    output[0] += broadcast<Real>(k058);
    greenTrace = broadcast<Real>(k068);
    for (int ix = -6; ix <= 6; ++ix) {
        for (int iy = -6; iy <= 6; ++iy) {
            for (int iz = -6; iz <= 6; ++iz) {
                int const latticeSquared = ix * ix + iy * iy + iz * iz;
                if (latticeSquared == 0 || latticeSquared > 36 || ix < 0 || (ix == 0 && iy < 0) ||
                    (ix == 0 && iy == 0 && iz < 0)) {
                    continue;
                }
                for (int sign = -1; sign <= 1; sign += 2) {
                    Real const imageX = x + broadcast<Real>(sign * ix);
                    Real const imageY = y + broadcast<Real>(sign * iy);
                    Real const imageZ = z + broadcast<Real>(sign * iz);
                    Real const imageSquared =
                        fma(imageX, imageX, fma(imageY, imageY, imageZ * imageZ));
                    Real const imageRadius = sqrt(imageSquared);
                    Real const iR = broadcast<Real>(k059) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k067) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k062) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k063) * imageRadius * imageExp;
                    Real inversePower = iR;
                    Real const imageRadial0 = -screening * inversePower;
                    Real const imageValue0 = imageRadial0 * harmonics[0];
                    if (sign < 0) {
                        imagePair[0] = imageValue0;
                    } else {
                        imagePair[0] += imageValue0;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k057);
                    Real const imageRadial1 = screening * inversePower;
                    Real const imageValue1 = broadcast<Real>(k062) * imageRadial1 * harmonics[1];
                    if (sign < 0) {
                        imagePair[1] = imageValue1;
                    } else {
                        imagePair[1] += imageValue1;
                    }
                    Real const imageValue2 = imageRadial1 * harmonics[2];
                    if (sign < 0) {
                        imagePair[2] = imageValue2;
                    } else {
                        imagePair[2] += imageValue2;
                    }
                    Real const imageValue3 = broadcast<Real>(k062) * imageRadial1 * harmonics[3];
                    if (sign < 0) {
                        imagePair[3] = imageValue3;
                    } else {
                        imagePair[3] += imageValue3;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k055);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k082) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k071) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k067) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k071) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k082) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k054);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k087) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k085) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k084) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k083) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k084) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k085) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k087) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k053);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k093) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k089) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k088) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k087) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k086) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k087) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k088) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k089) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k093) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k052);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k100) * imageRadial5 * harmonics[25];
                    if (sign < 0) {
                        imagePair[25] = imageValue25;
                    } else {
                        imagePair[25] += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k095) * imageRadial5 * harmonics[26];
                    if (sign < 0) {
                        imagePair[26] = imageValue26;
                    } else {
                        imagePair[26] += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k094) * imageRadial5 * harmonics[27];
                    if (sign < 0) {
                        imagePair[27] = imageValue27;
                    } else {
                        imagePair[27] += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k092) * imageRadial5 * harmonics[28];
                    if (sign < 0) {
                        imagePair[28] = imageValue28;
                    } else {
                        imagePair[28] += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k091) * imageRadial5 * harmonics[29];
                    if (sign < 0) {
                        imagePair[29] = imageValue29;
                    } else {
                        imagePair[29] += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k090) * imageRadial5 * harmonics[30];
                    if (sign < 0) {
                        imagePair[30] = imageValue30;
                    } else {
                        imagePair[30] += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k091) * imageRadial5 * harmonics[31];
                    if (sign < 0) {
                        imagePair[31] = imageValue31;
                    } else {
                        imagePair[31] += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k092) * imageRadial5 * harmonics[32];
                    if (sign < 0) {
                        imagePair[32] = imageValue32;
                    } else {
                        imagePair[32] += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k094) * imageRadial5 * harmonics[33];
                    if (sign < 0) {
                        imagePair[33] = imageValue33;
                    } else {
                        imagePair[33] += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k095) * imageRadial5 * harmonics[34];
                    if (sign < 0) {
                        imagePair[34] = imageValue34;
                    } else {
                        imagePair[34] += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k100) * imageRadial5 * harmonics[35];
                    if (sign < 0) {
                        imagePair[35] = imageValue35;
                    } else {
                        imagePair[35] += imageValue35;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k051);
                    Real const imageRadial6 = -screening * inversePower;
                    Real const imageValue36 = broadcast<Real>(k108) * imageRadial6 * harmonics[36];
                    if (sign < 0) {
                        imagePair[36] = imageValue36;
                    } else {
                        imagePair[36] += imageValue36;
                    }
                    Real const imageValue37 = broadcast<Real>(k104) * imageRadial6 * harmonics[37];
                    if (sign < 0) {
                        imagePair[37] = imageValue37;
                    } else {
                        imagePair[37] += imageValue37;
                    }
                    Real const imageValue38 = broadcast<Real>(k101) * imageRadial6 * harmonics[38];
                    if (sign < 0) {
                        imagePair[38] = imageValue38;
                    } else {
                        imagePair[38] += imageValue38;
                    }
                    Real const imageValue39 = broadcast<Real>(k099) * imageRadial6 * harmonics[39];
                    if (sign < 0) {
                        imagePair[39] = imageValue39;
                    } else {
                        imagePair[39] += imageValue39;
                    }
                    Real const imageValue40 = broadcast<Real>(k098) * imageRadial6 * harmonics[40];
                    if (sign < 0) {
                        imagePair[40] = imageValue40;
                    } else {
                        imagePair[40] += imageValue40;
                    }
                    Real const imageValue41 = broadcast<Real>(k097) * imageRadial6 * harmonics[41];
                    if (sign < 0) {
                        imagePair[41] = imageValue41;
                    } else {
                        imagePair[41] += imageValue41;
                    }
                    Real const imageValue42 = broadcast<Real>(k096) * imageRadial6 * harmonics[42];
                    if (sign < 0) {
                        imagePair[42] = imageValue42;
                    } else {
                        imagePair[42] += imageValue42;
                    }
                    Real const imageValue43 = broadcast<Real>(k097) * imageRadial6 * harmonics[43];
                    if (sign < 0) {
                        imagePair[43] = imageValue43;
                    } else {
                        imagePair[43] += imageValue43;
                    }
                    Real const imageValue44 = broadcast<Real>(k098) * imageRadial6 * harmonics[44];
                    if (sign < 0) {
                        imagePair[44] = imageValue44;
                    } else {
                        imagePair[44] += imageValue44;
                    }
                    Real const imageValue45 = broadcast<Real>(k099) * imageRadial6 * harmonics[45];
                    if (sign < 0) {
                        imagePair[45] = imageValue45;
                    } else {
                        imagePair[45] += imageValue45;
                    }
                    Real const imageValue46 = broadcast<Real>(k101) * imageRadial6 * harmonics[46];
                    if (sign < 0) {
                        imagePair[46] = imageValue46;
                    } else {
                        imagePair[46] += imageValue46;
                    }
                    Real const imageValue47 = broadcast<Real>(k104) * imageRadial6 * harmonics[47];
                    if (sign < 0) {
                        imagePair[47] = imageValue47;
                    } else {
                        imagePair[47] += imageValue47;
                    }
                    Real const imageValue48 = broadcast<Real>(k108) * imageRadial6 * harmonics[48];
                    if (sign < 0) {
                        imagePair[48] = imageValue48;
                    } else {
                        imagePair[48] += imageValue48;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k050);
                    Real const imageRadial7 = screening * inversePower;
                    Real const imageValue49 = broadcast<Real>(k117) * imageRadial7 * harmonics[49];
                    if (sign < 0) {
                        imagePair[49] = imageValue49;
                    } else {
                        imagePair[49] += imageValue49;
                    }
                    Real const imageValue50 = broadcast<Real>(k114) * imageRadial7 * harmonics[50];
                    if (sign < 0) {
                        imagePair[50] = imageValue50;
                    } else {
                        imagePair[50] += imageValue50;
                    }
                    Real const imageValue51 = broadcast<Real>(k109) * imageRadial7 * harmonics[51];
                    if (sign < 0) {
                        imagePair[51] = imageValue51;
                    } else {
                        imagePair[51] += imageValue51;
                    }
                    Real const imageValue52 = broadcast<Real>(k107) * imageRadial7 * harmonics[52];
                    if (sign < 0) {
                        imagePair[52] = imageValue52;
                    } else {
                        imagePair[52] += imageValue52;
                    }
                    Real const imageValue53 = broadcast<Real>(k106) * imageRadial7 * harmonics[53];
                    if (sign < 0) {
                        imagePair[53] = imageValue53;
                    } else {
                        imagePair[53] += imageValue53;
                    }
                    Real const imageValue54 = broadcast<Real>(k105) * imageRadial7 * harmonics[54];
                    if (sign < 0) {
                        imagePair[54] = imageValue54;
                    } else {
                        imagePair[54] += imageValue54;
                    }
                    Real const imageValue55 = broadcast<Real>(k103) * imageRadial7 * harmonics[55];
                    if (sign < 0) {
                        imagePair[55] = imageValue55;
                    } else {
                        imagePair[55] += imageValue55;
                    }
                    Real const imageValue56 = broadcast<Real>(k102) * imageRadial7 * harmonics[56];
                    if (sign < 0) {
                        imagePair[56] = imageValue56;
                    } else {
                        imagePair[56] += imageValue56;
                    }
                    Real const imageValue57 = broadcast<Real>(k103) * imageRadial7 * harmonics[57];
                    if (sign < 0) {
                        imagePair[57] = imageValue57;
                    } else {
                        imagePair[57] += imageValue57;
                    }
                    Real const imageValue58 = broadcast<Real>(k105) * imageRadial7 * harmonics[58];
                    if (sign < 0) {
                        imagePair[58] = imageValue58;
                    } else {
                        imagePair[58] += imageValue58;
                    }
                    Real const imageValue59 = broadcast<Real>(k106) * imageRadial7 * harmonics[59];
                    if (sign < 0) {
                        imagePair[59] = imageValue59;
                    } else {
                        imagePair[59] += imageValue59;
                    }
                    Real const imageValue60 = broadcast<Real>(k107) * imageRadial7 * harmonics[60];
                    if (sign < 0) {
                        imagePair[60] = imageValue60;
                    } else {
                        imagePair[60] += imageValue60;
                    }
                    Real const imageValue61 = broadcast<Real>(k109) * imageRadial7 * harmonics[61];
                    if (sign < 0) {
                        imagePair[61] = imageValue61;
                    } else {
                        imagePair[61] += imageValue61;
                    }
                    Real const imageValue62 = broadcast<Real>(k114) * imageRadial7 * harmonics[62];
                    if (sign < 0) {
                        imagePair[62] = imageValue62;
                    } else {
                        imagePair[62] += imageValue62;
                    }
                    Real const imageValue63 = broadcast<Real>(k117) * imageRadial7 * harmonics[63];
                    if (sign < 0) {
                        imagePair[63] = imageValue63;
                    } else {
                        imagePair[63] += imageValue63;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k049);
                    Real const imageRadial8 = -screening * inversePower;
                    Real const imageValue64 = broadcast<Real>(k120) * imageRadial8 * harmonics[64];
                    if (sign < 0) {
                        imagePair[64] = imageValue64;
                    } else {
                        imagePair[64] += imageValue64;
                    }
                    Real const imageValue65 = broadcast<Real>(k119) * imageRadial8 * harmonics[65];
                    if (sign < 0) {
                        imagePair[65] = imageValue65;
                    } else {
                        imagePair[65] += imageValue65;
                    }
                    Real const imageValue66 = broadcast<Real>(k118) * imageRadial8 * harmonics[66];
                    if (sign < 0) {
                        imagePair[66] = imageValue66;
                    } else {
                        imagePair[66] += imageValue66;
                    }
                    Real const imageValue67 = broadcast<Real>(k116) * imageRadial8 * harmonics[67];
                    if (sign < 0) {
                        imagePair[67] = imageValue67;
                    } else {
                        imagePair[67] += imageValue67;
                    }
                    Real const imageValue68 = broadcast<Real>(k115) * imageRadial8 * harmonics[68];
                    if (sign < 0) {
                        imagePair[68] = imageValue68;
                    } else {
                        imagePair[68] += imageValue68;
                    }
                    Real const imageValue69 = broadcast<Real>(k113) * imageRadial8 * harmonics[69];
                    if (sign < 0) {
                        imagePair[69] = imageValue69;
                    } else {
                        imagePair[69] += imageValue69;
                    }
                    Real const imageValue70 = broadcast<Real>(k112) * imageRadial8 * harmonics[70];
                    if (sign < 0) {
                        imagePair[70] = imageValue70;
                    } else {
                        imagePair[70] += imageValue70;
                    }
                    Real const imageValue71 = broadcast<Real>(k111) * imageRadial8 * harmonics[71];
                    if (sign < 0) {
                        imagePair[71] = imageValue71;
                    } else {
                        imagePair[71] += imageValue71;
                    }
                    Real const imageValue72 = broadcast<Real>(k110) * imageRadial8 * harmonics[72];
                    if (sign < 0) {
                        imagePair[72] = imageValue72;
                    } else {
                        imagePair[72] += imageValue72;
                    }
                    Real const imageValue73 = broadcast<Real>(k111) * imageRadial8 * harmonics[73];
                    if (sign < 0) {
                        imagePair[73] = imageValue73;
                    } else {
                        imagePair[73] += imageValue73;
                    }
                    Real const imageValue74 = broadcast<Real>(k112) * imageRadial8 * harmonics[74];
                    if (sign < 0) {
                        imagePair[74] = imageValue74;
                    } else {
                        imagePair[74] += imageValue74;
                    }
                    Real const imageValue75 = broadcast<Real>(k113) * imageRadial8 * harmonics[75];
                    if (sign < 0) {
                        imagePair[75] = imageValue75;
                    } else {
                        imagePair[75] += imageValue75;
                    }
                    Real const imageValue76 = broadcast<Real>(k115) * imageRadial8 * harmonics[76];
                    if (sign < 0) {
                        imagePair[76] = imageValue76;
                    } else {
                        imagePair[76] += imageValue76;
                    }
                    Real const imageValue77 = broadcast<Real>(k116) * imageRadial8 * harmonics[77];
                    if (sign < 0) {
                        imagePair[77] = imageValue77;
                    } else {
                        imagePair[77] += imageValue77;
                    }
                    Real const imageValue78 = broadcast<Real>(k118) * imageRadial8 * harmonics[78];
                    if (sign < 0) {
                        imagePair[78] = imageValue78;
                    } else {
                        imagePair[78] += imageValue78;
                    }
                    Real const imageValue79 = broadcast<Real>(k119) * imageRadial8 * harmonics[79];
                    if (sign < 0) {
                        imagePair[79] = imageValue79;
                    } else {
                        imagePair[79] += imageValue79;
                    }
                    Real const imageValue80 = broadcast<Real>(k120) * imageRadial8 * harmonics[80];
                    if (sign < 0) {
                        imagePair[80] = imageValue80;
                    } else {
                        imagePair[80] += imageValue80;
                    }
                }
                detail::ewaldAccumulate(output[0], compensation[0], imagePair[0]);
                detail::ewaldAccumulate(output[1], compensation[1], imagePair[1]);
                detail::ewaldAccumulate(output[2], compensation[2], imagePair[2]);
                detail::ewaldAccumulate(output[3], compensation[3], imagePair[3]);
                detail::ewaldAccumulate(output[4], compensation[4], imagePair[4]);
                detail::ewaldAccumulate(output[5], compensation[5], imagePair[5]);
                detail::ewaldAccumulate(output[6], compensation[6], imagePair[6]);
                detail::ewaldAccumulate(output[7], compensation[7], imagePair[7]);
                detail::ewaldAccumulate(output[8], compensation[8], imagePair[8]);
                detail::ewaldAccumulate(output[9], compensation[9], imagePair[9]);
                detail::ewaldAccumulate(output[10], compensation[10], imagePair[10]);
                detail::ewaldAccumulate(output[11], compensation[11], imagePair[11]);
                detail::ewaldAccumulate(output[12], compensation[12], imagePair[12]);
                detail::ewaldAccumulate(output[13], compensation[13], imagePair[13]);
                detail::ewaldAccumulate(output[14], compensation[14], imagePair[14]);
                detail::ewaldAccumulate(output[15], compensation[15], imagePair[15]);
                detail::ewaldAccumulate(output[16], compensation[16], imagePair[16]);
                detail::ewaldAccumulate(output[17], compensation[17], imagePair[17]);
                detail::ewaldAccumulate(output[18], compensation[18], imagePair[18]);
                detail::ewaldAccumulate(output[19], compensation[19], imagePair[19]);
                detail::ewaldAccumulate(output[20], compensation[20], imagePair[20]);
                detail::ewaldAccumulate(output[21], compensation[21], imagePair[21]);
                detail::ewaldAccumulate(output[22], compensation[22], imagePair[22]);
                detail::ewaldAccumulate(output[23], compensation[23], imagePair[23]);
                detail::ewaldAccumulate(output[24], compensation[24], imagePair[24]);
                detail::ewaldAccumulate(output[25], compensation[25], imagePair[25]);
                detail::ewaldAccumulate(output[26], compensation[26], imagePair[26]);
                detail::ewaldAccumulate(output[27], compensation[27], imagePair[27]);
                detail::ewaldAccumulate(output[28], compensation[28], imagePair[28]);
                detail::ewaldAccumulate(output[29], compensation[29], imagePair[29]);
                detail::ewaldAccumulate(output[30], compensation[30], imagePair[30]);
                detail::ewaldAccumulate(output[31], compensation[31], imagePair[31]);
                detail::ewaldAccumulate(output[32], compensation[32], imagePair[32]);
                detail::ewaldAccumulate(output[33], compensation[33], imagePair[33]);
                detail::ewaldAccumulate(output[34], compensation[34], imagePair[34]);
                detail::ewaldAccumulate(output[35], compensation[35], imagePair[35]);
                detail::ewaldAccumulate(output[36], compensation[36], imagePair[36]);
                detail::ewaldAccumulate(output[37], compensation[37], imagePair[37]);
                detail::ewaldAccumulate(output[38], compensation[38], imagePair[38]);
                detail::ewaldAccumulate(output[39], compensation[39], imagePair[39]);
                detail::ewaldAccumulate(output[40], compensation[40], imagePair[40]);
                detail::ewaldAccumulate(output[41], compensation[41], imagePair[41]);
                detail::ewaldAccumulate(output[42], compensation[42], imagePair[42]);
                detail::ewaldAccumulate(output[43], compensation[43], imagePair[43]);
                detail::ewaldAccumulate(output[44], compensation[44], imagePair[44]);
                detail::ewaldAccumulate(output[45], compensation[45], imagePair[45]);
                detail::ewaldAccumulate(output[46], compensation[46], imagePair[46]);
                detail::ewaldAccumulate(output[47], compensation[47], imagePair[47]);
                detail::ewaldAccumulate(output[48], compensation[48], imagePair[48]);
                detail::ewaldAccumulate(output[49], compensation[49], imagePair[49]);
                detail::ewaldAccumulate(output[50], compensation[50], imagePair[50]);
                detail::ewaldAccumulate(output[51], compensation[51], imagePair[51]);
                detail::ewaldAccumulate(output[52], compensation[52], imagePair[52]);
                detail::ewaldAccumulate(output[53], compensation[53], imagePair[53]);
                detail::ewaldAccumulate(output[54], compensation[54], imagePair[54]);
                detail::ewaldAccumulate(output[55], compensation[55], imagePair[55]);
                detail::ewaldAccumulate(output[56], compensation[56], imagePair[56]);
                detail::ewaldAccumulate(output[57], compensation[57], imagePair[57]);
                detail::ewaldAccumulate(output[58], compensation[58], imagePair[58]);
                detail::ewaldAccumulate(output[59], compensation[59], imagePair[59]);
                detail::ewaldAccumulate(output[60], compensation[60], imagePair[60]);
                detail::ewaldAccumulate(output[61], compensation[61], imagePair[61]);
                detail::ewaldAccumulate(output[62], compensation[62], imagePair[62]);
                detail::ewaldAccumulate(output[63], compensation[63], imagePair[63]);
                detail::ewaldAccumulate(output[64], compensation[64], imagePair[64]);
                detail::ewaldAccumulate(output[65], compensation[65], imagePair[65]);
                detail::ewaldAccumulate(output[66], compensation[66], imagePair[66]);
                detail::ewaldAccumulate(output[67], compensation[67], imagePair[67]);
                detail::ewaldAccumulate(output[68], compensation[68], imagePair[68]);
                detail::ewaldAccumulate(output[69], compensation[69], imagePair[69]);
                detail::ewaldAccumulate(output[70], compensation[70], imagePair[70]);
                detail::ewaldAccumulate(output[71], compensation[71], imagePair[71]);
                detail::ewaldAccumulate(output[72], compensation[72], imagePair[72]);
                detail::ewaldAccumulate(output[73], compensation[73], imagePair[73]);
                detail::ewaldAccumulate(output[74], compensation[74], imagePair[74]);
                detail::ewaldAccumulate(output[75], compensation[75], imagePair[75]);
                detail::ewaldAccumulate(output[76], compensation[76], imagePair[76]);
                detail::ewaldAccumulate(output[77], compensation[77], imagePair[77]);
                detail::ewaldAccumulate(output[78], compensation[78], imagePair[78]);
                detail::ewaldAccumulate(output[79], compensation[79], imagePair[79]);
                detail::ewaldAccumulate(output[80], compensation[80], imagePair[80]);
                Real const waveX = broadcast<Real>(ix);
                Real const waveY = broadcast<Real>(iy);
                Real const waveZ = broadcast<Real>(iz);
                Real const waveSquared = broadcast<Real>(latticeSquared);
                ewaldRegular<degree>(harmonics, waveX, waveY, waveZ, waveSquared);
                Real const phase = broadcast<Real>(k073) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k056) *
                    detail::ewaldExp(-broadcast<Real>(k065) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k059) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k073) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k062) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k062) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k079) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k082) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k071) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k067) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k071) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k082) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k081) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k087) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k085) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k084) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k083) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k084) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k085) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k087) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k080) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k093) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k089) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k088) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k087) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k086) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k087) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k088) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k089) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k093) * waveRadial4 * harmonics[24]);
                Real const waveRadial5 = broadcast<Real>(k077) * wavePotential * -sine;
                detail::ewaldAccumulate(output[25], compensation[25],
                    broadcast<Real>(k100) * waveRadial5 * harmonics[25]);
                detail::ewaldAccumulate(output[26], compensation[26],
                    broadcast<Real>(k095) * waveRadial5 * harmonics[26]);
                detail::ewaldAccumulate(output[27], compensation[27],
                    broadcast<Real>(k094) * waveRadial5 * harmonics[27]);
                detail::ewaldAccumulate(output[28], compensation[28],
                    broadcast<Real>(k092) * waveRadial5 * harmonics[28]);
                detail::ewaldAccumulate(output[29], compensation[29],
                    broadcast<Real>(k091) * waveRadial5 * harmonics[29]);
                detail::ewaldAccumulate(output[30], compensation[30],
                    broadcast<Real>(k090) * waveRadial5 * harmonics[30]);
                detail::ewaldAccumulate(output[31], compensation[31],
                    broadcast<Real>(k091) * waveRadial5 * harmonics[31]);
                detail::ewaldAccumulate(output[32], compensation[32],
                    broadcast<Real>(k092) * waveRadial5 * harmonics[32]);
                detail::ewaldAccumulate(output[33], compensation[33],
                    broadcast<Real>(k094) * waveRadial5 * harmonics[33]);
                detail::ewaldAccumulate(output[34], compensation[34],
                    broadcast<Real>(k095) * waveRadial5 * harmonics[34]);
                detail::ewaldAccumulate(output[35], compensation[35],
                    broadcast<Real>(k100) * waveRadial5 * harmonics[35]);
                Real const waveRadial6 = broadcast<Real>(k070) * wavePotential * -cosine;
                detail::ewaldAccumulate(output[36], compensation[36],
                    broadcast<Real>(k108) * waveRadial6 * harmonics[36]);
                detail::ewaldAccumulate(output[37], compensation[37],
                    broadcast<Real>(k104) * waveRadial6 * harmonics[37]);
                detail::ewaldAccumulate(output[38], compensation[38],
                    broadcast<Real>(k101) * waveRadial6 * harmonics[38]);
                detail::ewaldAccumulate(output[39], compensation[39],
                    broadcast<Real>(k099) * waveRadial6 * harmonics[39]);
                detail::ewaldAccumulate(output[40], compensation[40],
                    broadcast<Real>(k098) * waveRadial6 * harmonics[40]);
                detail::ewaldAccumulate(output[41], compensation[41],
                    broadcast<Real>(k097) * waveRadial6 * harmonics[41]);
                detail::ewaldAccumulate(output[42], compensation[42],
                    broadcast<Real>(k096) * waveRadial6 * harmonics[42]);
                detail::ewaldAccumulate(output[43], compensation[43],
                    broadcast<Real>(k097) * waveRadial6 * harmonics[43]);
                detail::ewaldAccumulate(output[44], compensation[44],
                    broadcast<Real>(k098) * waveRadial6 * harmonics[44]);
                detail::ewaldAccumulate(output[45], compensation[45],
                    broadcast<Real>(k099) * waveRadial6 * harmonics[45]);
                detail::ewaldAccumulate(output[46], compensation[46],
                    broadcast<Real>(k101) * waveRadial6 * harmonics[46]);
                detail::ewaldAccumulate(output[47], compensation[47],
                    broadcast<Real>(k104) * waveRadial6 * harmonics[47]);
                detail::ewaldAccumulate(output[48], compensation[48],
                    broadcast<Real>(k108) * waveRadial6 * harmonics[48]);
                Real const waveRadial7 = broadcast<Real>(k066) * wavePotential * sine;
                detail::ewaldAccumulate(output[49], compensation[49],
                    broadcast<Real>(k117) * waveRadial7 * harmonics[49]);
                detail::ewaldAccumulate(output[50], compensation[50],
                    broadcast<Real>(k114) * waveRadial7 * harmonics[50]);
                detail::ewaldAccumulate(output[51], compensation[51],
                    broadcast<Real>(k109) * waveRadial7 * harmonics[51]);
                detail::ewaldAccumulate(output[52], compensation[52],
                    broadcast<Real>(k107) * waveRadial7 * harmonics[52]);
                detail::ewaldAccumulate(output[53], compensation[53],
                    broadcast<Real>(k106) * waveRadial7 * harmonics[53]);
                detail::ewaldAccumulate(output[54], compensation[54],
                    broadcast<Real>(k105) * waveRadial7 * harmonics[54]);
                detail::ewaldAccumulate(output[55], compensation[55],
                    broadcast<Real>(k103) * waveRadial7 * harmonics[55]);
                detail::ewaldAccumulate(output[56], compensation[56],
                    broadcast<Real>(k102) * waveRadial7 * harmonics[56]);
                detail::ewaldAccumulate(output[57], compensation[57],
                    broadcast<Real>(k103) * waveRadial7 * harmonics[57]);
                detail::ewaldAccumulate(output[58], compensation[58],
                    broadcast<Real>(k105) * waveRadial7 * harmonics[58]);
                detail::ewaldAccumulate(output[59], compensation[59],
                    broadcast<Real>(k106) * waveRadial7 * harmonics[59]);
                detail::ewaldAccumulate(output[60], compensation[60],
                    broadcast<Real>(k107) * waveRadial7 * harmonics[60]);
                detail::ewaldAccumulate(output[61], compensation[61],
                    broadcast<Real>(k109) * waveRadial7 * harmonics[61]);
                detail::ewaldAccumulate(output[62], compensation[62],
                    broadcast<Real>(k114) * waveRadial7 * harmonics[62]);
                detail::ewaldAccumulate(output[63], compensation[63],
                    broadcast<Real>(k117) * waveRadial7 * harmonics[63]);
                Real const waveRadial8 = broadcast<Real>(k061) * wavePotential * cosine;
                detail::ewaldAccumulate(output[64], compensation[64],
                    broadcast<Real>(k120) * waveRadial8 * harmonics[64]);
                detail::ewaldAccumulate(output[65], compensation[65],
                    broadcast<Real>(k119) * waveRadial8 * harmonics[65]);
                detail::ewaldAccumulate(output[66], compensation[66],
                    broadcast<Real>(k118) * waveRadial8 * harmonics[66]);
                detail::ewaldAccumulate(output[67], compensation[67],
                    broadcast<Real>(k116) * waveRadial8 * harmonics[67]);
                detail::ewaldAccumulate(output[68], compensation[68],
                    broadcast<Real>(k115) * waveRadial8 * harmonics[68]);
                detail::ewaldAccumulate(output[69], compensation[69],
                    broadcast<Real>(k113) * waveRadial8 * harmonics[69]);
                detail::ewaldAccumulate(output[70], compensation[70],
                    broadcast<Real>(k112) * waveRadial8 * harmonics[70]);
                detail::ewaldAccumulate(output[71], compensation[71],
                    broadcast<Real>(k111) * waveRadial8 * harmonics[71]);
                detail::ewaldAccumulate(output[72], compensation[72],
                    broadcast<Real>(k110) * waveRadial8 * harmonics[72]);
                detail::ewaldAccumulate(output[73], compensation[73],
                    broadcast<Real>(k111) * waveRadial8 * harmonics[73]);
                detail::ewaldAccumulate(output[74], compensation[74],
                    broadcast<Real>(k112) * waveRadial8 * harmonics[74]);
                detail::ewaldAccumulate(output[75], compensation[75],
                    broadcast<Real>(k113) * waveRadial8 * harmonics[75]);
                detail::ewaldAccumulate(output[76], compensation[76],
                    broadcast<Real>(k115) * waveRadial8 * harmonics[76]);
                detail::ewaldAccumulate(output[77], compensation[77],
                    broadcast<Real>(k116) * waveRadial8 * harmonics[77]);
                detail::ewaldAccumulate(output[78], compensation[78],
                    broadcast<Real>(k118) * waveRadial8 * harmonics[78]);
                detail::ewaldAccumulate(output[79], compensation[79],
                    broadcast<Real>(k119) * waveRadial8 * harmonics[79]);
                detail::ewaldAccumulate(output[80], compensation[80],
                    broadcast<Real>(k120) * waveRadial8 * harmonics[80]);
            }
        }
    }
}
/// Evaluate the periodic correction with promoted working precision, then convert the result.
template <int degree, typename Real, typename Output>
    requires(degree == 8)
CUDA_EXPORT void evaluateEwaldGreen(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    using Working = typename detail::EwaldWorkingType<Real>::Type;
    Working temporary[81];
    Working trace;
    evaluateEwaldWorking<degree, Working>(temporary, trace, detail::ewaldConvert<Working>(x),
        detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));
    for (int i = 0; i < 81; ++i) {
        output[i] = detail::ewaldConvert<Real>(temporary[i]);
    }
    greenTrace = detail::ewaldConvert<Real>(trace);
}
#endif
}    // namespace cosmictiger::gravity::spherical
