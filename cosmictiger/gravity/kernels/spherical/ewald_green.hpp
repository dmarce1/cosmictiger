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
    constexpr auto k0 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+3.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.50000000000000000E-01);
    constexpr auto k3 = static_cast<Scalar>(+5.00000000000000000E-01);
    output[0] = broadcast<Real>(k0);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k1) * z;
    output[6] = broadcast<Real>(k2) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    output[3] = x * broadcast<Real>(k3);
    output[1] = y * broadcast<Real>(k3);
    output[7] = z * output[3];
    output[5] = z * output[1];
    output[8] = broadcast<Real>(k2) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k2) * fma(y, output[3], x * output[1]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 2)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(+4.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.06185567010309274E-02);
    constexpr auto k3 = static_cast<Scalar>(+2.10526315789473679E-02);
    constexpr auto k4 = static_cast<Scalar>(+2.15053763440860232E-02);
    constexpr auto k5 = static_cast<Scalar>(+2.19780219780219797E-02);
    constexpr auto k6 = static_cast<Scalar>(+2.24719101123595499E-02);
    constexpr auto k7 = static_cast<Scalar>(+2.29885057471264365E-02);
    constexpr auto k8 = static_cast<Scalar>(+2.35294117647058820E-02);
    constexpr auto k9 = static_cast<Scalar>(+2.40963855421686760E-02);
    constexpr auto k10 = static_cast<Scalar>(+2.46913580246913567E-02);
    constexpr auto k11 = static_cast<Scalar>(+2.53164556962025313E-02);
    constexpr auto k12 = static_cast<Scalar>(+2.59740259740259757E-02);
    constexpr auto k13 = static_cast<Scalar>(+2.66666666666666684E-02);
    constexpr auto k14 = static_cast<Scalar>(+2.73972602739726012E-02);
    constexpr auto k15 = static_cast<Scalar>(+2.81690140845070429E-02);
    constexpr auto k16 = static_cast<Scalar>(+2.89855072463768119E-02);
    constexpr auto k17 = static_cast<Scalar>(+2.98507462686567158E-02);
    constexpr auto k18 = static_cast<Scalar>(+3.07692307692307709E-02);
    constexpr auto k19 = static_cast<Scalar>(+3.17460317460317443E-02);
    constexpr auto k20 = static_cast<Scalar>(+3.27868852459016411E-02);
    constexpr auto k21 = static_cast<Scalar>(+3.38983050847457626E-02);
    constexpr auto k22 = static_cast<Scalar>(+3.50877192982456121E-02);
    constexpr auto k23 = static_cast<Scalar>(+3.63636363636363619E-02);
    constexpr auto k24 = static_cast<Scalar>(+3.77358490566037721E-02);
    constexpr auto k25 = static_cast<Scalar>(+3.92156862745098034E-02);
    constexpr auto k26 = static_cast<Scalar>(+4.08163265306122416E-02);
    constexpr auto k27 = static_cast<Scalar>(+4.25531914893617011E-02);
    constexpr auto k28 = static_cast<Scalar>(+4.44444444444444461E-02);
    constexpr auto k29 = static_cast<Scalar>(+4.65116279069767435E-02);
    constexpr auto k30 = static_cast<Scalar>(+4.87804878048780505E-02);
    constexpr auto k31 = static_cast<Scalar>(+5.12820512820512803E-02);
    constexpr auto k32 = static_cast<Scalar>(+5.40540540540540571E-02);
    constexpr auto k33 = static_cast<Scalar>(+5.71428571428571411E-02);
    constexpr auto k34 = static_cast<Scalar>(+6.06060606060606077E-02);
    constexpr auto k35 = static_cast<Scalar>(+6.45161290322580627E-02);
    constexpr auto k36 = static_cast<Scalar>(+6.89655172413793094E-02);
    constexpr auto k37 = static_cast<Scalar>(+7.40740740740740700E-02);
    constexpr auto k38 = static_cast<Scalar>(+8.00000000000000017E-02);
    constexpr auto k39 = static_cast<Scalar>(+8.69565217391304324E-02);
    constexpr auto k40 = static_cast<Scalar>(+9.52380952380952328E-02);
    constexpr auto k41 = static_cast<Scalar>(+1.05263157894736836E-01);
    constexpr auto k42 = static_cast<Scalar>(+1.17647058823529410E-01);
    constexpr auto k43 = static_cast<Scalar>(+1.33333333333333331E-01);
    constexpr auto k44 = static_cast<Scalar>(+1.53846153846153855E-01);
    constexpr auto k45 = static_cast<Scalar>(+1.81818181818181823E-01);
    constexpr auto k46 = static_cast<Scalar>(+2.22222222222222210E-01);
    constexpr auto k47 = static_cast<Scalar>(+2.85714285714285698E-01);
    constexpr auto k48 = static_cast<Scalar>(+4.00000000000000022E-01);
    constexpr auto k49 = static_cast<Scalar>(+6.66666666666666630E-01);
    constexpr auto k50 = static_cast<Scalar>(+2.25675833419102512E+00);
    constexpr auto k51 = static_cast<Scalar>(+0.00000000000000000E+00);
    constexpr auto k52 = static_cast<Scalar>(+2.02020202020202037E-02);
    constexpr auto k53 = static_cast<Scalar>(+6.01802222450939972E+00);
    constexpr auto k54 = static_cast<Scalar>(+2.00000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(+1.98019801980198022E-02);
    constexpr auto k56 = static_cast<Scalar>(+9.62883555921504275E+00);
    constexpr auto k57 = static_cast<Scalar>(+2.40000000000000000E+01);
    constexpr auto k58 = static_cast<Scalar>(+6.00000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(+7.85398163397448279E-01);
    constexpr auto k60 = static_cast<Scalar>(+4.18879020478639053E+00);
    constexpr auto k61 = static_cast<Scalar>(+6.28318530717958623E+00);
    constexpr auto k62 = static_cast<Scalar>(+6.36619772367581382E-01);
    constexpr auto k63 = static_cast<Scalar>(+2.46740110027233950E+00);
    constexpr auto k64 = static_cast<Scalar>(+1.31594725347858112E+01);
    Real harmonics[9];
    Real compensation[9];
    Real imagePair[9];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k0) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k1);
    series0 = fma(series0, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k1));
    Real const radial0 = broadcast<Real>(k50) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k51);
    Real series1 = broadcast<Real>(k1);
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    Real const radial1 = -broadcast<Real>(k53) * exponential * series1;
    output[1] = broadcast<Real>(k54) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k51);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k51);
    output[3] = broadcast<Real>(k54) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k51);
    Real series2 = broadcast<Real>(k1);
    series2 = fma(series2, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    Real const radial2 = broadcast<Real>(k56) * exponential * series2;
    output[4] = broadcast<Real>(k57) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k51);
    output[5] = broadcast<Real>(k58) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k51);
    output[6] = broadcast<Real>(k0) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k51);
    output[7] = broadcast<Real>(k58) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k51);
    output[8] = broadcast<Real>(k57) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k51);
    output[0] += broadcast<Real>(k59);
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
                    Real const iR = broadcast<Real>(k1) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k0) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k54) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k50) * imageRadius * imageExp;
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
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
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
                    gammaTerm *= imageArgument * broadcast<Real>(k48);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k57) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k58) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k0) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k58) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k57) * imageRadial2 * harmonics[8];
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
                Real const wavePotential = -broadcast<Real>(k62) *
                    detail::ewaldExp(-broadcast<Real>(k63) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k1) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k61) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k54) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k54) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k64) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k57) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k58) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k0) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k58) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k57) * waveRadial2 * harmonics[8]);
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
    constexpr auto k0 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+3.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.50000000000000000E-01);
    constexpr auto k3 = static_cast<Scalar>(+5.00000000000000000E+00);
    constexpr auto k4 = static_cast<Scalar>(+1.11111111111111105E-01);
    constexpr auto k5 = static_cast<Scalar>(+5.00000000000000000E-01);
    constexpr auto k6 = static_cast<Scalar>(+1.25000000000000000E-01);
    constexpr auto k7 = static_cast<Scalar>(+1.66666666666666657E-01);
    output[0] = broadcast<Real>(k0);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k1) * z;
    output[6] = broadcast<Real>(k2) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k3) * z;
    output[12] = broadcast<Real>(k4) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    output[3] = x * broadcast<Real>(k5);
    output[1] = y * broadcast<Real>(k5);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k3) * z;
    output[13] = broadcast<Real>(k6) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k6) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    output[8] = broadcast<Real>(k2) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k2) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    output[15] = broadcast<Real>(k7) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k7) * fma(y, output[8], x * output[4]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 3)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(+4.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.06185567010309274E-02);
    constexpr auto k3 = static_cast<Scalar>(+2.10526315789473679E-02);
    constexpr auto k4 = static_cast<Scalar>(+2.15053763440860232E-02);
    constexpr auto k5 = static_cast<Scalar>(+2.19780219780219797E-02);
    constexpr auto k6 = static_cast<Scalar>(+2.24719101123595499E-02);
    constexpr auto k7 = static_cast<Scalar>(+2.29885057471264365E-02);
    constexpr auto k8 = static_cast<Scalar>(+2.35294117647058820E-02);
    constexpr auto k9 = static_cast<Scalar>(+2.40963855421686760E-02);
    constexpr auto k10 = static_cast<Scalar>(+2.46913580246913567E-02);
    constexpr auto k11 = static_cast<Scalar>(+2.53164556962025313E-02);
    constexpr auto k12 = static_cast<Scalar>(+2.59740259740259757E-02);
    constexpr auto k13 = static_cast<Scalar>(+2.66666666666666684E-02);
    constexpr auto k14 = static_cast<Scalar>(+2.73972602739726012E-02);
    constexpr auto k15 = static_cast<Scalar>(+2.81690140845070429E-02);
    constexpr auto k16 = static_cast<Scalar>(+2.89855072463768119E-02);
    constexpr auto k17 = static_cast<Scalar>(+2.98507462686567158E-02);
    constexpr auto k18 = static_cast<Scalar>(+3.07692307692307709E-02);
    constexpr auto k19 = static_cast<Scalar>(+3.17460317460317443E-02);
    constexpr auto k20 = static_cast<Scalar>(+3.27868852459016411E-02);
    constexpr auto k21 = static_cast<Scalar>(+3.38983050847457626E-02);
    constexpr auto k22 = static_cast<Scalar>(+3.50877192982456121E-02);
    constexpr auto k23 = static_cast<Scalar>(+3.63636363636363619E-02);
    constexpr auto k24 = static_cast<Scalar>(+3.77358490566037721E-02);
    constexpr auto k25 = static_cast<Scalar>(+3.92156862745098034E-02);
    constexpr auto k26 = static_cast<Scalar>(+4.08163265306122416E-02);
    constexpr auto k27 = static_cast<Scalar>(+4.25531914893617011E-02);
    constexpr auto k28 = static_cast<Scalar>(+4.44444444444444461E-02);
    constexpr auto k29 = static_cast<Scalar>(+4.65116279069767435E-02);
    constexpr auto k30 = static_cast<Scalar>(+4.87804878048780505E-02);
    constexpr auto k31 = static_cast<Scalar>(+5.12820512820512803E-02);
    constexpr auto k32 = static_cast<Scalar>(+5.40540540540540571E-02);
    constexpr auto k33 = static_cast<Scalar>(+5.71428571428571411E-02);
    constexpr auto k34 = static_cast<Scalar>(+6.06060606060606077E-02);
    constexpr auto k35 = static_cast<Scalar>(+6.45161290322580627E-02);
    constexpr auto k36 = static_cast<Scalar>(+6.89655172413793094E-02);
    constexpr auto k37 = static_cast<Scalar>(+7.40740740740740700E-02);
    constexpr auto k38 = static_cast<Scalar>(+8.00000000000000017E-02);
    constexpr auto k39 = static_cast<Scalar>(+8.69565217391304324E-02);
    constexpr auto k40 = static_cast<Scalar>(+9.52380952380952328E-02);
    constexpr auto k41 = static_cast<Scalar>(+1.05263157894736836E-01);
    constexpr auto k42 = static_cast<Scalar>(+1.17647058823529410E-01);
    constexpr auto k43 = static_cast<Scalar>(+1.33333333333333331E-01);
    constexpr auto k44 = static_cast<Scalar>(+1.53846153846153855E-01);
    constexpr auto k45 = static_cast<Scalar>(+1.81818181818181823E-01);
    constexpr auto k46 = static_cast<Scalar>(+2.22222222222222210E-01);
    constexpr auto k47 = static_cast<Scalar>(+2.85714285714285698E-01);
    constexpr auto k48 = static_cast<Scalar>(+4.00000000000000022E-01);
    constexpr auto k49 = static_cast<Scalar>(+6.66666666666666630E-01);
    constexpr auto k50 = static_cast<Scalar>(+2.25675833419102512E+00);
    constexpr auto k51 = static_cast<Scalar>(+0.00000000000000000E+00);
    constexpr auto k52 = static_cast<Scalar>(+2.02020202020202037E-02);
    constexpr auto k53 = static_cast<Scalar>(+6.01802222450939972E+00);
    constexpr auto k54 = static_cast<Scalar>(+2.00000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(+1.98019801980198022E-02);
    constexpr auto k56 = static_cast<Scalar>(+9.62883555921504275E+00);
    constexpr auto k57 = static_cast<Scalar>(+2.40000000000000000E+01);
    constexpr auto k58 = static_cast<Scalar>(+6.00000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(+1.94174757281553381E-02);
    constexpr auto k60 = static_cast<Scalar>(+1.10043834962457634E+01);
    constexpr auto k61 = static_cast<Scalar>(+7.20000000000000000E+02);
    constexpr auto k62 = static_cast<Scalar>(+1.20000000000000000E+02);
    constexpr auto k63 = static_cast<Scalar>(+4.80000000000000000E+01);
    constexpr auto k64 = static_cast<Scalar>(+3.60000000000000000E+01);
    constexpr auto k65 = static_cast<Scalar>(+7.85398163397448279E-01);
    constexpr auto k66 = static_cast<Scalar>(+4.18879020478639053E+00);
    constexpr auto k67 = static_cast<Scalar>(+6.28318530717958623E+00);
    constexpr auto k68 = static_cast<Scalar>(+6.36619772367581382E-01);
    constexpr auto k69 = static_cast<Scalar>(+2.46740110027233950E+00);
    constexpr auto k70 = static_cast<Scalar>(+1.31594725347858112E+01);
    constexpr auto k71 = static_cast<Scalar>(+1.65366808961599006E+01);
    Real harmonics[16];
    Real compensation[16];
    Real imagePair[16];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k0) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k1);
    series0 = fma(series0, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k1));
    Real const radial0 = broadcast<Real>(k50) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k51);
    Real series1 = broadcast<Real>(k1);
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    Real const radial1 = -broadcast<Real>(k53) * exponential * series1;
    output[1] = broadcast<Real>(k54) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k51);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k51);
    output[3] = broadcast<Real>(k54) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k51);
    Real series2 = broadcast<Real>(k1);
    series2 = fma(series2, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    Real const radial2 = broadcast<Real>(k56) * exponential * series2;
    output[4] = broadcast<Real>(k57) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k51);
    output[5] = broadcast<Real>(k58) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k51);
    output[6] = broadcast<Real>(k0) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k51);
    output[7] = broadcast<Real>(k58) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k51);
    output[8] = broadcast<Real>(k57) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k51);
    Real series3 = broadcast<Real>(k1);
    series3 = fma(series3, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    Real const radial3 = -broadcast<Real>(k60) * exponential * series3;
    output[9] = broadcast<Real>(k61) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k51);
    output[10] = broadcast<Real>(k62) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k51);
    output[11] = broadcast<Real>(k63) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k51);
    output[12] = broadcast<Real>(k64) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k51);
    output[13] = broadcast<Real>(k63) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k51);
    output[14] = broadcast<Real>(k62) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k51);
    output[15] = broadcast<Real>(k61) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k51);
    output[0] += broadcast<Real>(k65);
    greenTrace = broadcast<Real>(k66);
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
                    Real const iR = broadcast<Real>(k1) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k0) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k54) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k50) * imageRadius * imageExp;
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
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
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
                    gammaTerm *= imageArgument * broadcast<Real>(k48);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k57) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k58) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k0) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k58) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k57) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k47);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k61) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k62) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k63) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k64) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k63) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k62) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k61) * imageRadial3 * harmonics[15];
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
                Real const phase = broadcast<Real>(k67) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k68) *
                    detail::ewaldExp(-broadcast<Real>(k69) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k1) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k67) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k54) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k54) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k70) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k57) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k58) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k0) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k58) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k57) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k71) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k61) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k64) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k61) * waveRadial3 * harmonics[15]);
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
    constexpr auto k0 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+3.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.50000000000000000E-01);
    constexpr auto k3 = static_cast<Scalar>(+5.00000000000000000E+00);
    constexpr auto k4 = static_cast<Scalar>(+1.11111111111111105E-01);
    constexpr auto k5 = static_cast<Scalar>(+7.00000000000000000E+00);
    constexpr auto k6 = static_cast<Scalar>(+6.25000000000000000E-02);
    constexpr auto k7 = static_cast<Scalar>(+5.00000000000000000E-01);
    constexpr auto k8 = static_cast<Scalar>(+1.25000000000000000E-01);
    constexpr auto k9 = static_cast<Scalar>(+6.66666666666666657E-02);
    constexpr auto k10 = static_cast<Scalar>(+8.33333333333333287E-02);
    constexpr auto k11 = static_cast<Scalar>(+1.66666666666666657E-01);
    output[0] = broadcast<Real>(k0);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k1) * z;
    output[6] = broadcast<Real>(k2) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k3) * z;
    output[12] = broadcast<Real>(k4) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k5) * z;
    output[20] = broadcast<Real>(k6) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    output[3] = x * broadcast<Real>(k7);
    output[1] = y * broadcast<Real>(k7);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k3) * z;
    output[13] = broadcast<Real>(k8) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k8) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k5) * z;
    output[21] = broadcast<Real>(k9) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k9) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    output[8] = broadcast<Real>(k2) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k2) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k5) * z;
    output[22] = broadcast<Real>(k10) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k10) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    output[15] = broadcast<Real>(k11) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k11) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    output[24] = broadcast<Real>(k8) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k8) * fma(y, output[15], x * output[9]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 4)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(+4.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.06185567010309274E-02);
    constexpr auto k3 = static_cast<Scalar>(+2.10526315789473679E-02);
    constexpr auto k4 = static_cast<Scalar>(+2.15053763440860232E-02);
    constexpr auto k5 = static_cast<Scalar>(+2.19780219780219797E-02);
    constexpr auto k6 = static_cast<Scalar>(+2.24719101123595499E-02);
    constexpr auto k7 = static_cast<Scalar>(+2.29885057471264365E-02);
    constexpr auto k8 = static_cast<Scalar>(+2.35294117647058820E-02);
    constexpr auto k9 = static_cast<Scalar>(+2.40963855421686760E-02);
    constexpr auto k10 = static_cast<Scalar>(+2.46913580246913567E-02);
    constexpr auto k11 = static_cast<Scalar>(+2.53164556962025313E-02);
    constexpr auto k12 = static_cast<Scalar>(+2.59740259740259757E-02);
    constexpr auto k13 = static_cast<Scalar>(+2.66666666666666684E-02);
    constexpr auto k14 = static_cast<Scalar>(+2.73972602739726012E-02);
    constexpr auto k15 = static_cast<Scalar>(+2.81690140845070429E-02);
    constexpr auto k16 = static_cast<Scalar>(+2.89855072463768119E-02);
    constexpr auto k17 = static_cast<Scalar>(+2.98507462686567158E-02);
    constexpr auto k18 = static_cast<Scalar>(+3.07692307692307709E-02);
    constexpr auto k19 = static_cast<Scalar>(+3.17460317460317443E-02);
    constexpr auto k20 = static_cast<Scalar>(+3.27868852459016411E-02);
    constexpr auto k21 = static_cast<Scalar>(+3.38983050847457626E-02);
    constexpr auto k22 = static_cast<Scalar>(+3.50877192982456121E-02);
    constexpr auto k23 = static_cast<Scalar>(+3.63636363636363619E-02);
    constexpr auto k24 = static_cast<Scalar>(+3.77358490566037721E-02);
    constexpr auto k25 = static_cast<Scalar>(+3.92156862745098034E-02);
    constexpr auto k26 = static_cast<Scalar>(+4.08163265306122416E-02);
    constexpr auto k27 = static_cast<Scalar>(+4.25531914893617011E-02);
    constexpr auto k28 = static_cast<Scalar>(+4.44444444444444461E-02);
    constexpr auto k29 = static_cast<Scalar>(+4.65116279069767435E-02);
    constexpr auto k30 = static_cast<Scalar>(+4.87804878048780505E-02);
    constexpr auto k31 = static_cast<Scalar>(+5.12820512820512803E-02);
    constexpr auto k32 = static_cast<Scalar>(+5.40540540540540571E-02);
    constexpr auto k33 = static_cast<Scalar>(+5.71428571428571411E-02);
    constexpr auto k34 = static_cast<Scalar>(+6.06060606060606077E-02);
    constexpr auto k35 = static_cast<Scalar>(+6.45161290322580627E-02);
    constexpr auto k36 = static_cast<Scalar>(+6.89655172413793094E-02);
    constexpr auto k37 = static_cast<Scalar>(+7.40740740740740700E-02);
    constexpr auto k38 = static_cast<Scalar>(+8.00000000000000017E-02);
    constexpr auto k39 = static_cast<Scalar>(+8.69565217391304324E-02);
    constexpr auto k40 = static_cast<Scalar>(+9.52380952380952328E-02);
    constexpr auto k41 = static_cast<Scalar>(+1.05263157894736836E-01);
    constexpr auto k42 = static_cast<Scalar>(+1.17647058823529410E-01);
    constexpr auto k43 = static_cast<Scalar>(+1.33333333333333331E-01);
    constexpr auto k44 = static_cast<Scalar>(+1.53846153846153855E-01);
    constexpr auto k45 = static_cast<Scalar>(+1.81818181818181823E-01);
    constexpr auto k46 = static_cast<Scalar>(+2.22222222222222210E-01);
    constexpr auto k47 = static_cast<Scalar>(+2.85714285714285698E-01);
    constexpr auto k48 = static_cast<Scalar>(+4.00000000000000022E-01);
    constexpr auto k49 = static_cast<Scalar>(+6.66666666666666630E-01);
    constexpr auto k50 = static_cast<Scalar>(+2.25675833419102512E+00);
    constexpr auto k51 = static_cast<Scalar>(+0.00000000000000000E+00);
    constexpr auto k52 = static_cast<Scalar>(+2.02020202020202037E-02);
    constexpr auto k53 = static_cast<Scalar>(+6.01802222450939972E+00);
    constexpr auto k54 = static_cast<Scalar>(+2.00000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(+1.98019801980198022E-02);
    constexpr auto k56 = static_cast<Scalar>(+9.62883555921504275E+00);
    constexpr auto k57 = static_cast<Scalar>(+2.40000000000000000E+01);
    constexpr auto k58 = static_cast<Scalar>(+6.00000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(+1.94174757281553381E-02);
    constexpr auto k60 = static_cast<Scalar>(+1.10043834962457634E+01);
    constexpr auto k61 = static_cast<Scalar>(+7.20000000000000000E+02);
    constexpr auto k62 = static_cast<Scalar>(+1.20000000000000000E+02);
    constexpr auto k63 = static_cast<Scalar>(+4.80000000000000000E+01);
    constexpr auto k64 = static_cast<Scalar>(+3.60000000000000000E+01);
    constexpr auto k65 = static_cast<Scalar>(+1.90476190476190493E-02);
    constexpr auto k66 = static_cast<Scalar>(+9.78167421888512045E+00);
    constexpr auto k67 = static_cast<Scalar>(+4.03200000000000000E+04);
    constexpr auto k68 = static_cast<Scalar>(+5.04000000000000000E+03);
    constexpr auto k69 = static_cast<Scalar>(+1.44000000000000000E+03);
    constexpr auto k70 = static_cast<Scalar>(+5.76000000000000000E+02);
    constexpr auto k71 = static_cast<Scalar>(+7.85398163397448279E-01);
    constexpr auto k72 = static_cast<Scalar>(+4.18879020478639053E+00);
    constexpr auto k73 = static_cast<Scalar>(+6.28318530717958623E+00);
    constexpr auto k74 = static_cast<Scalar>(+6.36619772367581382E-01);
    constexpr auto k75 = static_cast<Scalar>(+2.46740110027233950E+00);
    constexpr auto k76 = static_cast<Scalar>(+1.31594725347858112E+01);
    constexpr auto k77 = static_cast<Scalar>(+1.65366808961599006E+01);
    constexpr auto k78 = static_cast<Scalar>(+1.48432900623241775E+01);
    Real harmonics[25];
    Real compensation[25];
    Real imagePair[25];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k0) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k1);
    series0 = fma(series0, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k1));
    Real const radial0 = broadcast<Real>(k50) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k51);
    Real series1 = broadcast<Real>(k1);
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    Real const radial1 = -broadcast<Real>(k53) * exponential * series1;
    output[1] = broadcast<Real>(k54) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k51);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k51);
    output[3] = broadcast<Real>(k54) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k51);
    Real series2 = broadcast<Real>(k1);
    series2 = fma(series2, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    Real const radial2 = broadcast<Real>(k56) * exponential * series2;
    output[4] = broadcast<Real>(k57) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k51);
    output[5] = broadcast<Real>(k58) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k51);
    output[6] = broadcast<Real>(k0) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k51);
    output[7] = broadcast<Real>(k58) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k51);
    output[8] = broadcast<Real>(k57) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k51);
    Real series3 = broadcast<Real>(k1);
    series3 = fma(series3, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    Real const radial3 = -broadcast<Real>(k60) * exponential * series3;
    output[9] = broadcast<Real>(k61) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k51);
    output[10] = broadcast<Real>(k62) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k51);
    output[11] = broadcast<Real>(k63) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k51);
    output[12] = broadcast<Real>(k64) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k51);
    output[13] = broadcast<Real>(k63) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k51);
    output[14] = broadcast<Real>(k62) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k51);
    output[15] = broadcast<Real>(k61) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k51);
    Real series4 = broadcast<Real>(k1);
    series4 = fma(series4, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    Real const radial4 = broadcast<Real>(k66) * exponential * series4;
    output[16] = broadcast<Real>(k67) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k51);
    output[17] = broadcast<Real>(k68) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k51);
    output[18] = broadcast<Real>(k69) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k51);
    output[19] = broadcast<Real>(k61) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k51);
    output[20] = broadcast<Real>(k70) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k51);
    output[21] = broadcast<Real>(k61) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k51);
    output[22] = broadcast<Real>(k69) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k51);
    output[23] = broadcast<Real>(k68) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k51);
    output[24] = broadcast<Real>(k67) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k51);
    output[0] += broadcast<Real>(k71);
    greenTrace = broadcast<Real>(k72);
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
                    Real const iR = broadcast<Real>(k1) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k0) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k54) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k50) * imageRadius * imageExp;
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
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
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
                    gammaTerm *= imageArgument * broadcast<Real>(k48);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k57) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k58) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k0) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k58) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k57) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k47);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k61) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k62) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k63) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k64) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k63) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k62) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k61) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k46);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k67) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k68) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k69) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k61) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k70) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k61) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k69) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k68) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k67) * imageRadial4 * harmonics[24];
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
                Real const phase = broadcast<Real>(k73) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k74) *
                    detail::ewaldExp(-broadcast<Real>(k75) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k1) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k73) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k54) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k54) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k76) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k57) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k58) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k0) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k58) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k57) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k77) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k61) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k64) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k61) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k78) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k70) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[24]);
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
    constexpr auto k0 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+3.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.50000000000000000E-01);
    constexpr auto k3 = static_cast<Scalar>(+5.00000000000000000E+00);
    constexpr auto k4 = static_cast<Scalar>(+1.11111111111111105E-01);
    constexpr auto k5 = static_cast<Scalar>(+7.00000000000000000E+00);
    constexpr auto k6 = static_cast<Scalar>(+6.25000000000000000E-02);
    constexpr auto k7 = static_cast<Scalar>(+9.00000000000000000E+00);
    constexpr auto k8 = static_cast<Scalar>(+4.00000000000000008E-02);
    constexpr auto k9 = static_cast<Scalar>(+5.00000000000000000E-01);
    constexpr auto k10 = static_cast<Scalar>(+1.25000000000000000E-01);
    constexpr auto k11 = static_cast<Scalar>(+6.66666666666666657E-02);
    constexpr auto k12 = static_cast<Scalar>(+4.16666666666666644E-02);
    constexpr auto k13 = static_cast<Scalar>(+8.33333333333333287E-02);
    constexpr auto k14 = static_cast<Scalar>(+4.76190476190476164E-02);
    constexpr auto k15 = static_cast<Scalar>(+1.66666666666666657E-01);
    constexpr auto k16 = static_cast<Scalar>(+1.00000000000000006E-01);
    output[0] = broadcast<Real>(k0);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k1) * z;
    output[6] = broadcast<Real>(k2) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k3) * z;
    output[12] = broadcast<Real>(k4) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k5) * z;
    output[20] = broadcast<Real>(k6) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    Real const zFactor5_0 = broadcast<Real>(k7) * z;
    output[30] = broadcast<Real>(k8) * fma(zFactor5_0, output[20], -radiusSquared * output[12]);
    output[3] = x * broadcast<Real>(k9);
    output[1] = y * broadcast<Real>(k9);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k3) * z;
    output[13] = broadcast<Real>(k10) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k10) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k5) * z;
    output[21] = broadcast<Real>(k11) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k11) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    Real const zFactor5_1 = broadcast<Real>(k7) * z;
    output[31] = broadcast<Real>(k12) * fma(zFactor5_1, output[21], -radiusSquared * output[13]);
    output[29] = broadcast<Real>(k12) * fma(zFactor5_1, output[19], -radiusSquared * output[11]);
    output[8] = broadcast<Real>(k2) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k2) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k5) * z;
    output[22] = broadcast<Real>(k13) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k13) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    Real const zFactor5_2 = broadcast<Real>(k7) * z;
    output[32] = broadcast<Real>(k14) * fma(zFactor5_2, output[22], -radiusSquared * output[14]);
    output[28] = broadcast<Real>(k14) * fma(zFactor5_2, output[18], -radiusSquared * output[10]);
    output[15] = broadcast<Real>(k15) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k15) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    Real const zFactor5_3 = broadcast<Real>(k7) * z;
    output[33] = broadcast<Real>(k6) * fma(zFactor5_3, output[23], -radiusSquared * output[15]);
    output[27] = broadcast<Real>(k6) * fma(zFactor5_3, output[17], -radiusSquared * output[9]);
    output[24] = broadcast<Real>(k10) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k10) * fma(y, output[15], x * output[9]);
    output[34] = z * output[24];
    output[26] = z * output[16];
    output[35] = broadcast<Real>(k16) * fma(x, output[24], -y * output[16]);
    output[25] = broadcast<Real>(k16) * fma(y, output[24], x * output[16]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 5)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(+4.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.06185567010309274E-02);
    constexpr auto k3 = static_cast<Scalar>(+2.10526315789473679E-02);
    constexpr auto k4 = static_cast<Scalar>(+2.15053763440860232E-02);
    constexpr auto k5 = static_cast<Scalar>(+2.19780219780219797E-02);
    constexpr auto k6 = static_cast<Scalar>(+2.24719101123595499E-02);
    constexpr auto k7 = static_cast<Scalar>(+2.29885057471264365E-02);
    constexpr auto k8 = static_cast<Scalar>(+2.35294117647058820E-02);
    constexpr auto k9 = static_cast<Scalar>(+2.40963855421686760E-02);
    constexpr auto k10 = static_cast<Scalar>(+2.46913580246913567E-02);
    constexpr auto k11 = static_cast<Scalar>(+2.53164556962025313E-02);
    constexpr auto k12 = static_cast<Scalar>(+2.59740259740259757E-02);
    constexpr auto k13 = static_cast<Scalar>(+2.66666666666666684E-02);
    constexpr auto k14 = static_cast<Scalar>(+2.73972602739726012E-02);
    constexpr auto k15 = static_cast<Scalar>(+2.81690140845070429E-02);
    constexpr auto k16 = static_cast<Scalar>(+2.89855072463768119E-02);
    constexpr auto k17 = static_cast<Scalar>(+2.98507462686567158E-02);
    constexpr auto k18 = static_cast<Scalar>(+3.07692307692307709E-02);
    constexpr auto k19 = static_cast<Scalar>(+3.17460317460317443E-02);
    constexpr auto k20 = static_cast<Scalar>(+3.27868852459016411E-02);
    constexpr auto k21 = static_cast<Scalar>(+3.38983050847457626E-02);
    constexpr auto k22 = static_cast<Scalar>(+3.50877192982456121E-02);
    constexpr auto k23 = static_cast<Scalar>(+3.63636363636363619E-02);
    constexpr auto k24 = static_cast<Scalar>(+3.77358490566037721E-02);
    constexpr auto k25 = static_cast<Scalar>(+3.92156862745098034E-02);
    constexpr auto k26 = static_cast<Scalar>(+4.08163265306122416E-02);
    constexpr auto k27 = static_cast<Scalar>(+4.25531914893617011E-02);
    constexpr auto k28 = static_cast<Scalar>(+4.44444444444444461E-02);
    constexpr auto k29 = static_cast<Scalar>(+4.65116279069767435E-02);
    constexpr auto k30 = static_cast<Scalar>(+4.87804878048780505E-02);
    constexpr auto k31 = static_cast<Scalar>(+5.12820512820512803E-02);
    constexpr auto k32 = static_cast<Scalar>(+5.40540540540540571E-02);
    constexpr auto k33 = static_cast<Scalar>(+5.71428571428571411E-02);
    constexpr auto k34 = static_cast<Scalar>(+6.06060606060606077E-02);
    constexpr auto k35 = static_cast<Scalar>(+6.45161290322580627E-02);
    constexpr auto k36 = static_cast<Scalar>(+6.89655172413793094E-02);
    constexpr auto k37 = static_cast<Scalar>(+7.40740740740740700E-02);
    constexpr auto k38 = static_cast<Scalar>(+8.00000000000000017E-02);
    constexpr auto k39 = static_cast<Scalar>(+8.69565217391304324E-02);
    constexpr auto k40 = static_cast<Scalar>(+9.52380952380952328E-02);
    constexpr auto k41 = static_cast<Scalar>(+1.05263157894736836E-01);
    constexpr auto k42 = static_cast<Scalar>(+1.17647058823529410E-01);
    constexpr auto k43 = static_cast<Scalar>(+1.33333333333333331E-01);
    constexpr auto k44 = static_cast<Scalar>(+1.53846153846153855E-01);
    constexpr auto k45 = static_cast<Scalar>(+1.81818181818181823E-01);
    constexpr auto k46 = static_cast<Scalar>(+2.22222222222222210E-01);
    constexpr auto k47 = static_cast<Scalar>(+2.85714285714285698E-01);
    constexpr auto k48 = static_cast<Scalar>(+4.00000000000000022E-01);
    constexpr auto k49 = static_cast<Scalar>(+6.66666666666666630E-01);
    constexpr auto k50 = static_cast<Scalar>(+2.25675833419102512E+00);
    constexpr auto k51 = static_cast<Scalar>(+0.00000000000000000E+00);
    constexpr auto k52 = static_cast<Scalar>(+2.02020202020202037E-02);
    constexpr auto k53 = static_cast<Scalar>(+6.01802222450939972E+00);
    constexpr auto k54 = static_cast<Scalar>(+2.00000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(+1.98019801980198022E-02);
    constexpr auto k56 = static_cast<Scalar>(+9.62883555921504275E+00);
    constexpr auto k57 = static_cast<Scalar>(+2.40000000000000000E+01);
    constexpr auto k58 = static_cast<Scalar>(+6.00000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(+1.94174757281553381E-02);
    constexpr auto k60 = static_cast<Scalar>(+1.10043834962457634E+01);
    constexpr auto k61 = static_cast<Scalar>(+7.20000000000000000E+02);
    constexpr auto k62 = static_cast<Scalar>(+1.20000000000000000E+02);
    constexpr auto k63 = static_cast<Scalar>(+4.80000000000000000E+01);
    constexpr auto k64 = static_cast<Scalar>(+3.60000000000000000E+01);
    constexpr auto k65 = static_cast<Scalar>(+1.90476190476190493E-02);
    constexpr auto k66 = static_cast<Scalar>(+9.78167421888512045E+00);
    constexpr auto k67 = static_cast<Scalar>(+4.03200000000000000E+04);
    constexpr auto k68 = static_cast<Scalar>(+5.04000000000000000E+03);
    constexpr auto k69 = static_cast<Scalar>(+1.44000000000000000E+03);
    constexpr auto k70 = static_cast<Scalar>(+5.76000000000000000E+02);
    constexpr auto k71 = static_cast<Scalar>(+1.86915887850467276E-02);
    constexpr auto k72 = static_cast<Scalar>(+7.11394488646190482E+00);
    constexpr auto k73 = static_cast<Scalar>(+3.62880000000000000E+06);
    constexpr auto k74 = static_cast<Scalar>(+3.62880000000000000E+05);
    constexpr auto k75 = static_cast<Scalar>(+8.06400000000000000E+04);
    constexpr auto k76 = static_cast<Scalar>(+3.02400000000000000E+04);
    constexpr auto k77 = static_cast<Scalar>(+1.72800000000000000E+04);
    constexpr auto k78 = static_cast<Scalar>(+1.44000000000000000E+04);
    constexpr auto k79 = static_cast<Scalar>(+7.85398163397448279E-01);
    constexpr auto k80 = static_cast<Scalar>(+4.18879020478639053E+00);
    constexpr auto k81 = static_cast<Scalar>(+6.28318530717958623E+00);
    constexpr auto k82 = static_cast<Scalar>(+6.36619772367581382E-01);
    constexpr auto k83 = static_cast<Scalar>(+2.46740110027233950E+00);
    constexpr auto k84 = static_cast<Scalar>(+1.31594725347858112E+01);
    constexpr auto k85 = static_cast<Scalar>(+1.65366808961599006E+01);
    constexpr auto k86 = static_cast<Scalar>(+1.48432900623241775E+01);
    constexpr auto k87 = static_cast<Scalar>(+1.03625713366444501E+01);
    Real harmonics[36];
    Real compensation[36];
    Real imagePair[36];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k0) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k1);
    series0 = fma(series0, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k1));
    Real const radial0 = broadcast<Real>(k50) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k51);
    Real series1 = broadcast<Real>(k1);
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    Real const radial1 = -broadcast<Real>(k53) * exponential * series1;
    output[1] = broadcast<Real>(k54) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k51);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k51);
    output[3] = broadcast<Real>(k54) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k51);
    Real series2 = broadcast<Real>(k1);
    series2 = fma(series2, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    Real const radial2 = broadcast<Real>(k56) * exponential * series2;
    output[4] = broadcast<Real>(k57) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k51);
    output[5] = broadcast<Real>(k58) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k51);
    output[6] = broadcast<Real>(k0) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k51);
    output[7] = broadcast<Real>(k58) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k51);
    output[8] = broadcast<Real>(k57) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k51);
    Real series3 = broadcast<Real>(k1);
    series3 = fma(series3, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    Real const radial3 = -broadcast<Real>(k60) * exponential * series3;
    output[9] = broadcast<Real>(k61) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k51);
    output[10] = broadcast<Real>(k62) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k51);
    output[11] = broadcast<Real>(k63) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k51);
    output[12] = broadcast<Real>(k64) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k51);
    output[13] = broadcast<Real>(k63) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k51);
    output[14] = broadcast<Real>(k62) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k51);
    output[15] = broadcast<Real>(k61) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k51);
    Real series4 = broadcast<Real>(k1);
    series4 = fma(series4, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    Real const radial4 = broadcast<Real>(k66) * exponential * series4;
    output[16] = broadcast<Real>(k67) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k51);
    output[17] = broadcast<Real>(k68) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k51);
    output[18] = broadcast<Real>(k69) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k51);
    output[19] = broadcast<Real>(k61) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k51);
    output[20] = broadcast<Real>(k70) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k51);
    output[21] = broadcast<Real>(k61) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k51);
    output[22] = broadcast<Real>(k69) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k51);
    output[23] = broadcast<Real>(k68) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k51);
    output[24] = broadcast<Real>(k67) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k51);
    Real series5 = broadcast<Real>(k1);
    series5 = fma(series5, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    Real const radial5 = -broadcast<Real>(k72) * exponential * series5;
    output[25] = broadcast<Real>(k73) * radial5 * harmonics[25];
    compensation[25] = broadcast<Real>(k51);
    output[26] = broadcast<Real>(k74) * radial5 * harmonics[26];
    compensation[26] = broadcast<Real>(k51);
    output[27] = broadcast<Real>(k75) * radial5 * harmonics[27];
    compensation[27] = broadcast<Real>(k51);
    output[28] = broadcast<Real>(k76) * radial5 * harmonics[28];
    compensation[28] = broadcast<Real>(k51);
    output[29] = broadcast<Real>(k77) * radial5 * harmonics[29];
    compensation[29] = broadcast<Real>(k51);
    output[30] = broadcast<Real>(k78) * radial5 * harmonics[30];
    compensation[30] = broadcast<Real>(k51);
    output[31] = broadcast<Real>(k77) * radial5 * harmonics[31];
    compensation[31] = broadcast<Real>(k51);
    output[32] = broadcast<Real>(k76) * radial5 * harmonics[32];
    compensation[32] = broadcast<Real>(k51);
    output[33] = broadcast<Real>(k75) * radial5 * harmonics[33];
    compensation[33] = broadcast<Real>(k51);
    output[34] = broadcast<Real>(k74) * radial5 * harmonics[34];
    compensation[34] = broadcast<Real>(k51);
    output[35] = broadcast<Real>(k73) * radial5 * harmonics[35];
    compensation[35] = broadcast<Real>(k51);
    output[0] += broadcast<Real>(k79);
    greenTrace = broadcast<Real>(k80);
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
                    Real const iR = broadcast<Real>(k1) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k0) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k54) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k50) * imageRadius * imageExp;
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
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
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
                    gammaTerm *= imageArgument * broadcast<Real>(k48);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k57) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k58) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k0) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k58) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k57) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k47);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k61) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k62) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k63) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k64) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k63) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k62) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k61) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k46);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k67) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k68) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k69) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k61) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k70) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k61) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k69) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k68) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k67) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k45);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k73) * imageRadial5 * harmonics[25];
                    if (sign < 0) {
                        imagePair[25] = imageValue25;
                    } else {
                        imagePair[25] += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k74) * imageRadial5 * harmonics[26];
                    if (sign < 0) {
                        imagePair[26] = imageValue26;
                    } else {
                        imagePair[26] += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k75) * imageRadial5 * harmonics[27];
                    if (sign < 0) {
                        imagePair[27] = imageValue27;
                    } else {
                        imagePair[27] += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k76) * imageRadial5 * harmonics[28];
                    if (sign < 0) {
                        imagePair[28] = imageValue28;
                    } else {
                        imagePair[28] += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k77) * imageRadial5 * harmonics[29];
                    if (sign < 0) {
                        imagePair[29] = imageValue29;
                    } else {
                        imagePair[29] += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k78) * imageRadial5 * harmonics[30];
                    if (sign < 0) {
                        imagePair[30] = imageValue30;
                    } else {
                        imagePair[30] += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k77) * imageRadial5 * harmonics[31];
                    if (sign < 0) {
                        imagePair[31] = imageValue31;
                    } else {
                        imagePair[31] += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k76) * imageRadial5 * harmonics[32];
                    if (sign < 0) {
                        imagePair[32] = imageValue32;
                    } else {
                        imagePair[32] += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k75) * imageRadial5 * harmonics[33];
                    if (sign < 0) {
                        imagePair[33] = imageValue33;
                    } else {
                        imagePair[33] += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k74) * imageRadial5 * harmonics[34];
                    if (sign < 0) {
                        imagePair[34] = imageValue34;
                    } else {
                        imagePair[34] += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k73) * imageRadial5 * harmonics[35];
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
                Real const phase = broadcast<Real>(k81) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k82) *
                    detail::ewaldExp(-broadcast<Real>(k83) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k1) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k81) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k54) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k54) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k84) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k57) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k58) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k0) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k58) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k57) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k85) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k61) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k64) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k61) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k86) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k70) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[24]);
                Real const waveRadial5 = broadcast<Real>(k87) * wavePotential * -sine;
                detail::ewaldAccumulate(output[25], compensation[25],
                    broadcast<Real>(k73) * waveRadial5 * harmonics[25]);
                detail::ewaldAccumulate(output[26], compensation[26],
                    broadcast<Real>(k74) * waveRadial5 * harmonics[26]);
                detail::ewaldAccumulate(output[27], compensation[27],
                    broadcast<Real>(k75) * waveRadial5 * harmonics[27]);
                detail::ewaldAccumulate(output[28], compensation[28],
                    broadcast<Real>(k76) * waveRadial5 * harmonics[28]);
                detail::ewaldAccumulate(output[29], compensation[29],
                    broadcast<Real>(k77) * waveRadial5 * harmonics[29]);
                detail::ewaldAccumulate(output[30], compensation[30],
                    broadcast<Real>(k78) * waveRadial5 * harmonics[30]);
                detail::ewaldAccumulate(output[31], compensation[31],
                    broadcast<Real>(k77) * waveRadial5 * harmonics[31]);
                detail::ewaldAccumulate(output[32], compensation[32],
                    broadcast<Real>(k76) * waveRadial5 * harmonics[32]);
                detail::ewaldAccumulate(output[33], compensation[33],
                    broadcast<Real>(k75) * waveRadial5 * harmonics[33]);
                detail::ewaldAccumulate(output[34], compensation[34],
                    broadcast<Real>(k74) * waveRadial5 * harmonics[34]);
                detail::ewaldAccumulate(output[35], compensation[35],
                    broadcast<Real>(k73) * waveRadial5 * harmonics[35]);
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
    constexpr auto k0 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+3.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.50000000000000000E-01);
    constexpr auto k3 = static_cast<Scalar>(+5.00000000000000000E+00);
    constexpr auto k4 = static_cast<Scalar>(+1.11111111111111105E-01);
    constexpr auto k5 = static_cast<Scalar>(+7.00000000000000000E+00);
    constexpr auto k6 = static_cast<Scalar>(+6.25000000000000000E-02);
    constexpr auto k7 = static_cast<Scalar>(+9.00000000000000000E+00);
    constexpr auto k8 = static_cast<Scalar>(+4.00000000000000008E-02);
    constexpr auto k9 = static_cast<Scalar>(+1.10000000000000000E+01);
    constexpr auto k10 = static_cast<Scalar>(+2.77777777777777762E-02);
    constexpr auto k11 = static_cast<Scalar>(+5.00000000000000000E-01);
    constexpr auto k12 = static_cast<Scalar>(+1.25000000000000000E-01);
    constexpr auto k13 = static_cast<Scalar>(+6.66666666666666657E-02);
    constexpr auto k14 = static_cast<Scalar>(+4.16666666666666644E-02);
    constexpr auto k15 = static_cast<Scalar>(+2.85714285714285705E-02);
    constexpr auto k16 = static_cast<Scalar>(+8.33333333333333287E-02);
    constexpr auto k17 = static_cast<Scalar>(+4.76190476190476164E-02);
    constexpr auto k18 = static_cast<Scalar>(+3.12500000000000000E-02);
    constexpr auto k19 = static_cast<Scalar>(+1.66666666666666657E-01);
    constexpr auto k20 = static_cast<Scalar>(+3.70370370370370350E-02);
    constexpr auto k21 = static_cast<Scalar>(+5.00000000000000028E-02);
    constexpr auto k22 = static_cast<Scalar>(+1.00000000000000006E-01);
    output[0] = broadcast<Real>(k0);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k1) * z;
    output[6] = broadcast<Real>(k2) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k3) * z;
    output[12] = broadcast<Real>(k4) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k5) * z;
    output[20] = broadcast<Real>(k6) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    Real const zFactor5_0 = broadcast<Real>(k7) * z;
    output[30] = broadcast<Real>(k8) * fma(zFactor5_0, output[20], -radiusSquared * output[12]);
    Real const zFactor6_0 = broadcast<Real>(k9) * z;
    output[42] = broadcast<Real>(k10) * fma(zFactor6_0, output[30], -radiusSquared * output[20]);
    output[3] = x * broadcast<Real>(k11);
    output[1] = y * broadcast<Real>(k11);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k3) * z;
    output[13] = broadcast<Real>(k12) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k12) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k5) * z;
    output[21] = broadcast<Real>(k13) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k13) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    Real const zFactor5_1 = broadcast<Real>(k7) * z;
    output[31] = broadcast<Real>(k14) * fma(zFactor5_1, output[21], -radiusSquared * output[13]);
    output[29] = broadcast<Real>(k14) * fma(zFactor5_1, output[19], -radiusSquared * output[11]);
    Real const zFactor6_1 = broadcast<Real>(k9) * z;
    output[43] = broadcast<Real>(k15) * fma(zFactor6_1, output[31], -radiusSquared * output[21]);
    output[41] = broadcast<Real>(k15) * fma(zFactor6_1, output[29], -radiusSquared * output[19]);
    output[8] = broadcast<Real>(k2) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k2) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k5) * z;
    output[22] = broadcast<Real>(k16) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k16) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    Real const zFactor5_2 = broadcast<Real>(k7) * z;
    output[32] = broadcast<Real>(k17) * fma(zFactor5_2, output[22], -radiusSquared * output[14]);
    output[28] = broadcast<Real>(k17) * fma(zFactor5_2, output[18], -radiusSquared * output[10]);
    Real const zFactor6_2 = broadcast<Real>(k9) * z;
    output[44] = broadcast<Real>(k18) * fma(zFactor6_2, output[32], -radiusSquared * output[22]);
    output[40] = broadcast<Real>(k18) * fma(zFactor6_2, output[28], -radiusSquared * output[18]);
    output[15] = broadcast<Real>(k19) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k19) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    Real const zFactor5_3 = broadcast<Real>(k7) * z;
    output[33] = broadcast<Real>(k6) * fma(zFactor5_3, output[23], -radiusSquared * output[15]);
    output[27] = broadcast<Real>(k6) * fma(zFactor5_3, output[17], -radiusSquared * output[9]);
    Real const zFactor6_3 = broadcast<Real>(k9) * z;
    output[45] = broadcast<Real>(k20) * fma(zFactor6_3, output[33], -radiusSquared * output[23]);
    output[39] = broadcast<Real>(k20) * fma(zFactor6_3, output[27], -radiusSquared * output[17]);
    output[24] = broadcast<Real>(k12) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k12) * fma(y, output[15], x * output[9]);
    output[34] = z * output[24];
    output[26] = z * output[16];
    Real const zFactor6_4 = broadcast<Real>(k9) * z;
    output[46] = broadcast<Real>(k21) * fma(zFactor6_4, output[34], -radiusSquared * output[24]);
    output[38] = broadcast<Real>(k21) * fma(zFactor6_4, output[26], -radiusSquared * output[16]);
    output[35] = broadcast<Real>(k22) * fma(x, output[24], -y * output[16]);
    output[25] = broadcast<Real>(k22) * fma(y, output[24], x * output[16]);
    output[47] = z * output[35];
    output[37] = z * output[25];
    output[48] = broadcast<Real>(k16) * fma(x, output[35], -y * output[25]);
    output[36] = broadcast<Real>(k16) * fma(y, output[35], x * output[25]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 6)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(+4.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.06185567010309274E-02);
    constexpr auto k3 = static_cast<Scalar>(+2.10526315789473679E-02);
    constexpr auto k4 = static_cast<Scalar>(+2.15053763440860232E-02);
    constexpr auto k5 = static_cast<Scalar>(+2.19780219780219797E-02);
    constexpr auto k6 = static_cast<Scalar>(+2.24719101123595499E-02);
    constexpr auto k7 = static_cast<Scalar>(+2.29885057471264365E-02);
    constexpr auto k8 = static_cast<Scalar>(+2.35294117647058820E-02);
    constexpr auto k9 = static_cast<Scalar>(+2.40963855421686760E-02);
    constexpr auto k10 = static_cast<Scalar>(+2.46913580246913567E-02);
    constexpr auto k11 = static_cast<Scalar>(+2.53164556962025313E-02);
    constexpr auto k12 = static_cast<Scalar>(+2.59740259740259757E-02);
    constexpr auto k13 = static_cast<Scalar>(+2.66666666666666684E-02);
    constexpr auto k14 = static_cast<Scalar>(+2.73972602739726012E-02);
    constexpr auto k15 = static_cast<Scalar>(+2.81690140845070429E-02);
    constexpr auto k16 = static_cast<Scalar>(+2.89855072463768119E-02);
    constexpr auto k17 = static_cast<Scalar>(+2.98507462686567158E-02);
    constexpr auto k18 = static_cast<Scalar>(+3.07692307692307709E-02);
    constexpr auto k19 = static_cast<Scalar>(+3.17460317460317443E-02);
    constexpr auto k20 = static_cast<Scalar>(+3.27868852459016411E-02);
    constexpr auto k21 = static_cast<Scalar>(+3.38983050847457626E-02);
    constexpr auto k22 = static_cast<Scalar>(+3.50877192982456121E-02);
    constexpr auto k23 = static_cast<Scalar>(+3.63636363636363619E-02);
    constexpr auto k24 = static_cast<Scalar>(+3.77358490566037721E-02);
    constexpr auto k25 = static_cast<Scalar>(+3.92156862745098034E-02);
    constexpr auto k26 = static_cast<Scalar>(+4.08163265306122416E-02);
    constexpr auto k27 = static_cast<Scalar>(+4.25531914893617011E-02);
    constexpr auto k28 = static_cast<Scalar>(+4.44444444444444461E-02);
    constexpr auto k29 = static_cast<Scalar>(+4.65116279069767435E-02);
    constexpr auto k30 = static_cast<Scalar>(+4.87804878048780505E-02);
    constexpr auto k31 = static_cast<Scalar>(+5.12820512820512803E-02);
    constexpr auto k32 = static_cast<Scalar>(+5.40540540540540571E-02);
    constexpr auto k33 = static_cast<Scalar>(+5.71428571428571411E-02);
    constexpr auto k34 = static_cast<Scalar>(+6.06060606060606077E-02);
    constexpr auto k35 = static_cast<Scalar>(+6.45161290322580627E-02);
    constexpr auto k36 = static_cast<Scalar>(+6.89655172413793094E-02);
    constexpr auto k37 = static_cast<Scalar>(+7.40740740740740700E-02);
    constexpr auto k38 = static_cast<Scalar>(+8.00000000000000017E-02);
    constexpr auto k39 = static_cast<Scalar>(+8.69565217391304324E-02);
    constexpr auto k40 = static_cast<Scalar>(+9.52380952380952328E-02);
    constexpr auto k41 = static_cast<Scalar>(+1.05263157894736836E-01);
    constexpr auto k42 = static_cast<Scalar>(+1.17647058823529410E-01);
    constexpr auto k43 = static_cast<Scalar>(+1.33333333333333331E-01);
    constexpr auto k44 = static_cast<Scalar>(+1.53846153846153855E-01);
    constexpr auto k45 = static_cast<Scalar>(+1.81818181818181823E-01);
    constexpr auto k46 = static_cast<Scalar>(+2.22222222222222210E-01);
    constexpr auto k47 = static_cast<Scalar>(+2.85714285714285698E-01);
    constexpr auto k48 = static_cast<Scalar>(+4.00000000000000022E-01);
    constexpr auto k49 = static_cast<Scalar>(+6.66666666666666630E-01);
    constexpr auto k50 = static_cast<Scalar>(+2.25675833419102512E+00);
    constexpr auto k51 = static_cast<Scalar>(+0.00000000000000000E+00);
    constexpr auto k52 = static_cast<Scalar>(+2.02020202020202037E-02);
    constexpr auto k53 = static_cast<Scalar>(+6.01802222450939972E+00);
    constexpr auto k54 = static_cast<Scalar>(+2.00000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(+1.98019801980198022E-02);
    constexpr auto k56 = static_cast<Scalar>(+9.62883555921504275E+00);
    constexpr auto k57 = static_cast<Scalar>(+2.40000000000000000E+01);
    constexpr auto k58 = static_cast<Scalar>(+6.00000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(+1.94174757281553381E-02);
    constexpr auto k60 = static_cast<Scalar>(+1.10043834962457634E+01);
    constexpr auto k61 = static_cast<Scalar>(+7.20000000000000000E+02);
    constexpr auto k62 = static_cast<Scalar>(+1.20000000000000000E+02);
    constexpr auto k63 = static_cast<Scalar>(+4.80000000000000000E+01);
    constexpr auto k64 = static_cast<Scalar>(+3.60000000000000000E+01);
    constexpr auto k65 = static_cast<Scalar>(+1.90476190476190493E-02);
    constexpr auto k66 = static_cast<Scalar>(+9.78167421888512045E+00);
    constexpr auto k67 = static_cast<Scalar>(+4.03200000000000000E+04);
    constexpr auto k68 = static_cast<Scalar>(+5.04000000000000000E+03);
    constexpr auto k69 = static_cast<Scalar>(+1.44000000000000000E+03);
    constexpr auto k70 = static_cast<Scalar>(+5.76000000000000000E+02);
    constexpr auto k71 = static_cast<Scalar>(+1.86915887850467276E-02);
    constexpr auto k72 = static_cast<Scalar>(+7.11394488646190482E+00);
    constexpr auto k73 = static_cast<Scalar>(+3.62880000000000000E+06);
    constexpr auto k74 = static_cast<Scalar>(+3.62880000000000000E+05);
    constexpr auto k75 = static_cast<Scalar>(+8.06400000000000000E+04);
    constexpr auto k76 = static_cast<Scalar>(+3.02400000000000000E+04);
    constexpr auto k77 = static_cast<Scalar>(+1.72800000000000000E+04);
    constexpr auto k78 = static_cast<Scalar>(+1.44000000000000000E+04);
    constexpr auto k79 = static_cast<Scalar>(+1.83486238532110102E-02);
    constexpr auto k80 = static_cast<Scalar>(+4.37781223782271134E+00);
    constexpr auto k81 = static_cast<Scalar>(+4.79001600000000000E+08);
    constexpr auto k82 = static_cast<Scalar>(+3.99168000000000000E+07);
    constexpr auto k83 = static_cast<Scalar>(+7.25760000000000000E+06);
    constexpr auto k84 = static_cast<Scalar>(+2.17728000000000000E+06);
    constexpr auto k85 = static_cast<Scalar>(+9.67680000000000000E+05);
    constexpr auto k86 = static_cast<Scalar>(+6.04800000000000000E+05);
    constexpr auto k87 = static_cast<Scalar>(+5.18400000000000000E+05);
    constexpr auto k88 = static_cast<Scalar>(+7.85398163397448279E-01);
    constexpr auto k89 = static_cast<Scalar>(+4.18879020478639053E+00);
    constexpr auto k90 = static_cast<Scalar>(+6.28318530717958623E+00);
    constexpr auto k91 = static_cast<Scalar>(+6.36619772367581382E-01);
    constexpr auto k92 = static_cast<Scalar>(+2.46740110027233950E+00);
    constexpr auto k93 = static_cast<Scalar>(+1.31594725347858112E+01);
    constexpr auto k94 = static_cast<Scalar>(+1.65366808961599006E+01);
    constexpr auto k95 = static_cast<Scalar>(+1.48432900623241775E+01);
    constexpr auto k96 = static_cast<Scalar>(+1.03625713366444501E+01);
    constexpr auto k97 = static_cast<Scalar>(+5.91908690609133981E+00);
    Real harmonics[49];
    Real compensation[49];
    Real imagePair[49];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k0) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k1);
    series0 = fma(series0, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k1));
    Real const radial0 = broadcast<Real>(k50) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k51);
    Real series1 = broadcast<Real>(k1);
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    Real const radial1 = -broadcast<Real>(k53) * exponential * series1;
    output[1] = broadcast<Real>(k54) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k51);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k51);
    output[3] = broadcast<Real>(k54) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k51);
    Real series2 = broadcast<Real>(k1);
    series2 = fma(series2, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    Real const radial2 = broadcast<Real>(k56) * exponential * series2;
    output[4] = broadcast<Real>(k57) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k51);
    output[5] = broadcast<Real>(k58) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k51);
    output[6] = broadcast<Real>(k0) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k51);
    output[7] = broadcast<Real>(k58) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k51);
    output[8] = broadcast<Real>(k57) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k51);
    Real series3 = broadcast<Real>(k1);
    series3 = fma(series3, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    Real const radial3 = -broadcast<Real>(k60) * exponential * series3;
    output[9] = broadcast<Real>(k61) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k51);
    output[10] = broadcast<Real>(k62) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k51);
    output[11] = broadcast<Real>(k63) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k51);
    output[12] = broadcast<Real>(k64) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k51);
    output[13] = broadcast<Real>(k63) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k51);
    output[14] = broadcast<Real>(k62) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k51);
    output[15] = broadcast<Real>(k61) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k51);
    Real series4 = broadcast<Real>(k1);
    series4 = fma(series4, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    Real const radial4 = broadcast<Real>(k66) * exponential * series4;
    output[16] = broadcast<Real>(k67) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k51);
    output[17] = broadcast<Real>(k68) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k51);
    output[18] = broadcast<Real>(k69) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k51);
    output[19] = broadcast<Real>(k61) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k51);
    output[20] = broadcast<Real>(k70) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k51);
    output[21] = broadcast<Real>(k61) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k51);
    output[22] = broadcast<Real>(k69) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k51);
    output[23] = broadcast<Real>(k68) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k51);
    output[24] = broadcast<Real>(k67) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k51);
    Real series5 = broadcast<Real>(k1);
    series5 = fma(series5, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    Real const radial5 = -broadcast<Real>(k72) * exponential * series5;
    output[25] = broadcast<Real>(k73) * radial5 * harmonics[25];
    compensation[25] = broadcast<Real>(k51);
    output[26] = broadcast<Real>(k74) * radial5 * harmonics[26];
    compensation[26] = broadcast<Real>(k51);
    output[27] = broadcast<Real>(k75) * radial5 * harmonics[27];
    compensation[27] = broadcast<Real>(k51);
    output[28] = broadcast<Real>(k76) * radial5 * harmonics[28];
    compensation[28] = broadcast<Real>(k51);
    output[29] = broadcast<Real>(k77) * radial5 * harmonics[29];
    compensation[29] = broadcast<Real>(k51);
    output[30] = broadcast<Real>(k78) * radial5 * harmonics[30];
    compensation[30] = broadcast<Real>(k51);
    output[31] = broadcast<Real>(k77) * radial5 * harmonics[31];
    compensation[31] = broadcast<Real>(k51);
    output[32] = broadcast<Real>(k76) * radial5 * harmonics[32];
    compensation[32] = broadcast<Real>(k51);
    output[33] = broadcast<Real>(k75) * radial5 * harmonics[33];
    compensation[33] = broadcast<Real>(k51);
    output[34] = broadcast<Real>(k74) * radial5 * harmonics[34];
    compensation[34] = broadcast<Real>(k51);
    output[35] = broadcast<Real>(k73) * radial5 * harmonics[35];
    compensation[35] = broadcast<Real>(k51);
    Real series6 = broadcast<Real>(k1);
    series6 = fma(series6, argument * broadcast<Real>(k79), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    Real const radial6 = broadcast<Real>(k80) * exponential * series6;
    output[36] = broadcast<Real>(k81) * radial6 * harmonics[36];
    compensation[36] = broadcast<Real>(k51);
    output[37] = broadcast<Real>(k82) * radial6 * harmonics[37];
    compensation[37] = broadcast<Real>(k51);
    output[38] = broadcast<Real>(k83) * radial6 * harmonics[38];
    compensation[38] = broadcast<Real>(k51);
    output[39] = broadcast<Real>(k84) * radial6 * harmonics[39];
    compensation[39] = broadcast<Real>(k51);
    output[40] = broadcast<Real>(k85) * radial6 * harmonics[40];
    compensation[40] = broadcast<Real>(k51);
    output[41] = broadcast<Real>(k86) * radial6 * harmonics[41];
    compensation[41] = broadcast<Real>(k51);
    output[42] = broadcast<Real>(k87) * radial6 * harmonics[42];
    compensation[42] = broadcast<Real>(k51);
    output[43] = broadcast<Real>(k86) * radial6 * harmonics[43];
    compensation[43] = broadcast<Real>(k51);
    output[44] = broadcast<Real>(k85) * radial6 * harmonics[44];
    compensation[44] = broadcast<Real>(k51);
    output[45] = broadcast<Real>(k84) * radial6 * harmonics[45];
    compensation[45] = broadcast<Real>(k51);
    output[46] = broadcast<Real>(k83) * radial6 * harmonics[46];
    compensation[46] = broadcast<Real>(k51);
    output[47] = broadcast<Real>(k82) * radial6 * harmonics[47];
    compensation[47] = broadcast<Real>(k51);
    output[48] = broadcast<Real>(k81) * radial6 * harmonics[48];
    compensation[48] = broadcast<Real>(k51);
    output[0] += broadcast<Real>(k88);
    greenTrace = broadcast<Real>(k89);
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
                    Real const iR = broadcast<Real>(k1) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k0) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k54) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k50) * imageRadius * imageExp;
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
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
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
                    gammaTerm *= imageArgument * broadcast<Real>(k48);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k57) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k58) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k0) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k58) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k57) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k47);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k61) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k62) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k63) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k64) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k63) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k62) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k61) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k46);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k67) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k68) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k69) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k61) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k70) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k61) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k69) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k68) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k67) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k45);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k73) * imageRadial5 * harmonics[25];
                    if (sign < 0) {
                        imagePair[25] = imageValue25;
                    } else {
                        imagePair[25] += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k74) * imageRadial5 * harmonics[26];
                    if (sign < 0) {
                        imagePair[26] = imageValue26;
                    } else {
                        imagePair[26] += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k75) * imageRadial5 * harmonics[27];
                    if (sign < 0) {
                        imagePair[27] = imageValue27;
                    } else {
                        imagePair[27] += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k76) * imageRadial5 * harmonics[28];
                    if (sign < 0) {
                        imagePair[28] = imageValue28;
                    } else {
                        imagePair[28] += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k77) * imageRadial5 * harmonics[29];
                    if (sign < 0) {
                        imagePair[29] = imageValue29;
                    } else {
                        imagePair[29] += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k78) * imageRadial5 * harmonics[30];
                    if (sign < 0) {
                        imagePair[30] = imageValue30;
                    } else {
                        imagePair[30] += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k77) * imageRadial5 * harmonics[31];
                    if (sign < 0) {
                        imagePair[31] = imageValue31;
                    } else {
                        imagePair[31] += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k76) * imageRadial5 * harmonics[32];
                    if (sign < 0) {
                        imagePair[32] = imageValue32;
                    } else {
                        imagePair[32] += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k75) * imageRadial5 * harmonics[33];
                    if (sign < 0) {
                        imagePair[33] = imageValue33;
                    } else {
                        imagePair[33] += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k74) * imageRadial5 * harmonics[34];
                    if (sign < 0) {
                        imagePair[34] = imageValue34;
                    } else {
                        imagePair[34] += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k73) * imageRadial5 * harmonics[35];
                    if (sign < 0) {
                        imagePair[35] = imageValue35;
                    } else {
                        imagePair[35] += imageValue35;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k44);
                    Real const imageRadial6 = -screening * inversePower;
                    Real const imageValue36 = broadcast<Real>(k81) * imageRadial6 * harmonics[36];
                    if (sign < 0) {
                        imagePair[36] = imageValue36;
                    } else {
                        imagePair[36] += imageValue36;
                    }
                    Real const imageValue37 = broadcast<Real>(k82) * imageRadial6 * harmonics[37];
                    if (sign < 0) {
                        imagePair[37] = imageValue37;
                    } else {
                        imagePair[37] += imageValue37;
                    }
                    Real const imageValue38 = broadcast<Real>(k83) * imageRadial6 * harmonics[38];
                    if (sign < 0) {
                        imagePair[38] = imageValue38;
                    } else {
                        imagePair[38] += imageValue38;
                    }
                    Real const imageValue39 = broadcast<Real>(k84) * imageRadial6 * harmonics[39];
                    if (sign < 0) {
                        imagePair[39] = imageValue39;
                    } else {
                        imagePair[39] += imageValue39;
                    }
                    Real const imageValue40 = broadcast<Real>(k85) * imageRadial6 * harmonics[40];
                    if (sign < 0) {
                        imagePair[40] = imageValue40;
                    } else {
                        imagePair[40] += imageValue40;
                    }
                    Real const imageValue41 = broadcast<Real>(k86) * imageRadial6 * harmonics[41];
                    if (sign < 0) {
                        imagePair[41] = imageValue41;
                    } else {
                        imagePair[41] += imageValue41;
                    }
                    Real const imageValue42 = broadcast<Real>(k87) * imageRadial6 * harmonics[42];
                    if (sign < 0) {
                        imagePair[42] = imageValue42;
                    } else {
                        imagePair[42] += imageValue42;
                    }
                    Real const imageValue43 = broadcast<Real>(k86) * imageRadial6 * harmonics[43];
                    if (sign < 0) {
                        imagePair[43] = imageValue43;
                    } else {
                        imagePair[43] += imageValue43;
                    }
                    Real const imageValue44 = broadcast<Real>(k85) * imageRadial6 * harmonics[44];
                    if (sign < 0) {
                        imagePair[44] = imageValue44;
                    } else {
                        imagePair[44] += imageValue44;
                    }
                    Real const imageValue45 = broadcast<Real>(k84) * imageRadial6 * harmonics[45];
                    if (sign < 0) {
                        imagePair[45] = imageValue45;
                    } else {
                        imagePair[45] += imageValue45;
                    }
                    Real const imageValue46 = broadcast<Real>(k83) * imageRadial6 * harmonics[46];
                    if (sign < 0) {
                        imagePair[46] = imageValue46;
                    } else {
                        imagePair[46] += imageValue46;
                    }
                    Real const imageValue47 = broadcast<Real>(k82) * imageRadial6 * harmonics[47];
                    if (sign < 0) {
                        imagePair[47] = imageValue47;
                    } else {
                        imagePair[47] += imageValue47;
                    }
                    Real const imageValue48 = broadcast<Real>(k81) * imageRadial6 * harmonics[48];
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
                Real const phase = broadcast<Real>(k90) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k91) *
                    detail::ewaldExp(-broadcast<Real>(k92) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k1) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k90) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k54) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k54) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k93) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k57) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k58) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k0) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k58) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k57) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k94) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k61) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k64) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k61) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k95) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k70) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[24]);
                Real const waveRadial5 = broadcast<Real>(k96) * wavePotential * -sine;
                detail::ewaldAccumulate(output[25], compensation[25],
                    broadcast<Real>(k73) * waveRadial5 * harmonics[25]);
                detail::ewaldAccumulate(output[26], compensation[26],
                    broadcast<Real>(k74) * waveRadial5 * harmonics[26]);
                detail::ewaldAccumulate(output[27], compensation[27],
                    broadcast<Real>(k75) * waveRadial5 * harmonics[27]);
                detail::ewaldAccumulate(output[28], compensation[28],
                    broadcast<Real>(k76) * waveRadial5 * harmonics[28]);
                detail::ewaldAccumulate(output[29], compensation[29],
                    broadcast<Real>(k77) * waveRadial5 * harmonics[29]);
                detail::ewaldAccumulate(output[30], compensation[30],
                    broadcast<Real>(k78) * waveRadial5 * harmonics[30]);
                detail::ewaldAccumulate(output[31], compensation[31],
                    broadcast<Real>(k77) * waveRadial5 * harmonics[31]);
                detail::ewaldAccumulate(output[32], compensation[32],
                    broadcast<Real>(k76) * waveRadial5 * harmonics[32]);
                detail::ewaldAccumulate(output[33], compensation[33],
                    broadcast<Real>(k75) * waveRadial5 * harmonics[33]);
                detail::ewaldAccumulate(output[34], compensation[34],
                    broadcast<Real>(k74) * waveRadial5 * harmonics[34]);
                detail::ewaldAccumulate(output[35], compensation[35],
                    broadcast<Real>(k73) * waveRadial5 * harmonics[35]);
                Real const waveRadial6 = broadcast<Real>(k97) * wavePotential * -cosine;
                detail::ewaldAccumulate(output[36], compensation[36],
                    broadcast<Real>(k81) * waveRadial6 * harmonics[36]);
                detail::ewaldAccumulate(output[37], compensation[37],
                    broadcast<Real>(k82) * waveRadial6 * harmonics[37]);
                detail::ewaldAccumulate(output[38], compensation[38],
                    broadcast<Real>(k83) * waveRadial6 * harmonics[38]);
                detail::ewaldAccumulate(output[39], compensation[39],
                    broadcast<Real>(k84) * waveRadial6 * harmonics[39]);
                detail::ewaldAccumulate(output[40], compensation[40],
                    broadcast<Real>(k85) * waveRadial6 * harmonics[40]);
                detail::ewaldAccumulate(output[41], compensation[41],
                    broadcast<Real>(k86) * waveRadial6 * harmonics[41]);
                detail::ewaldAccumulate(output[42], compensation[42],
                    broadcast<Real>(k87) * waveRadial6 * harmonics[42]);
                detail::ewaldAccumulate(output[43], compensation[43],
                    broadcast<Real>(k86) * waveRadial6 * harmonics[43]);
                detail::ewaldAccumulate(output[44], compensation[44],
                    broadcast<Real>(k85) * waveRadial6 * harmonics[44]);
                detail::ewaldAccumulate(output[45], compensation[45],
                    broadcast<Real>(k84) * waveRadial6 * harmonics[45]);
                detail::ewaldAccumulate(output[46], compensation[46],
                    broadcast<Real>(k83) * waveRadial6 * harmonics[46]);
                detail::ewaldAccumulate(output[47], compensation[47],
                    broadcast<Real>(k82) * waveRadial6 * harmonics[47]);
                detail::ewaldAccumulate(output[48], compensation[48],
                    broadcast<Real>(k81) * waveRadial6 * harmonics[48]);
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
    constexpr auto k0 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+3.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.50000000000000000E-01);
    constexpr auto k3 = static_cast<Scalar>(+5.00000000000000000E+00);
    constexpr auto k4 = static_cast<Scalar>(+1.11111111111111105E-01);
    constexpr auto k5 = static_cast<Scalar>(+7.00000000000000000E+00);
    constexpr auto k6 = static_cast<Scalar>(+6.25000000000000000E-02);
    constexpr auto k7 = static_cast<Scalar>(+9.00000000000000000E+00);
    constexpr auto k8 = static_cast<Scalar>(+4.00000000000000008E-02);
    constexpr auto k9 = static_cast<Scalar>(+1.10000000000000000E+01);
    constexpr auto k10 = static_cast<Scalar>(+2.77777777777777762E-02);
    constexpr auto k11 = static_cast<Scalar>(+1.30000000000000000E+01);
    constexpr auto k12 = static_cast<Scalar>(+2.04081632653061208E-02);
    constexpr auto k13 = static_cast<Scalar>(+5.00000000000000000E-01);
    constexpr auto k14 = static_cast<Scalar>(+1.25000000000000000E-01);
    constexpr auto k15 = static_cast<Scalar>(+6.66666666666666657E-02);
    constexpr auto k16 = static_cast<Scalar>(+4.16666666666666644E-02);
    constexpr auto k17 = static_cast<Scalar>(+2.85714285714285705E-02);
    constexpr auto k18 = static_cast<Scalar>(+2.08333333333333322E-02);
    constexpr auto k19 = static_cast<Scalar>(+8.33333333333333287E-02);
    constexpr auto k20 = static_cast<Scalar>(+4.76190476190476164E-02);
    constexpr auto k21 = static_cast<Scalar>(+3.12500000000000000E-02);
    constexpr auto k22 = static_cast<Scalar>(+2.22222222222222231E-02);
    constexpr auto k23 = static_cast<Scalar>(+1.66666666666666657E-01);
    constexpr auto k24 = static_cast<Scalar>(+3.70370370370370350E-02);
    constexpr auto k25 = static_cast<Scalar>(+2.50000000000000014E-02);
    constexpr auto k26 = static_cast<Scalar>(+5.00000000000000028E-02);
    constexpr auto k27 = static_cast<Scalar>(+3.03030303030303039E-02);
    constexpr auto k28 = static_cast<Scalar>(+1.00000000000000006E-01);
    constexpr auto k29 = static_cast<Scalar>(+7.14285714285714246E-02);
    output[0] = broadcast<Real>(k0);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k1) * z;
    output[6] = broadcast<Real>(k2) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k3) * z;
    output[12] = broadcast<Real>(k4) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k5) * z;
    output[20] = broadcast<Real>(k6) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    Real const zFactor5_0 = broadcast<Real>(k7) * z;
    output[30] = broadcast<Real>(k8) * fma(zFactor5_0, output[20], -radiusSquared * output[12]);
    Real const zFactor6_0 = broadcast<Real>(k9) * z;
    output[42] = broadcast<Real>(k10) * fma(zFactor6_0, output[30], -radiusSquared * output[20]);
    Real const zFactor7_0 = broadcast<Real>(k11) * z;
    output[56] = broadcast<Real>(k12) * fma(zFactor7_0, output[42], -radiusSquared * output[30]);
    output[3] = x * broadcast<Real>(k13);
    output[1] = y * broadcast<Real>(k13);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k3) * z;
    output[13] = broadcast<Real>(k14) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k14) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k5) * z;
    output[21] = broadcast<Real>(k15) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k15) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    Real const zFactor5_1 = broadcast<Real>(k7) * z;
    output[31] = broadcast<Real>(k16) * fma(zFactor5_1, output[21], -radiusSquared * output[13]);
    output[29] = broadcast<Real>(k16) * fma(zFactor5_1, output[19], -radiusSquared * output[11]);
    Real const zFactor6_1 = broadcast<Real>(k9) * z;
    output[43] = broadcast<Real>(k17) * fma(zFactor6_1, output[31], -radiusSquared * output[21]);
    output[41] = broadcast<Real>(k17) * fma(zFactor6_1, output[29], -radiusSquared * output[19]);
    Real const zFactor7_1 = broadcast<Real>(k11) * z;
    output[57] = broadcast<Real>(k18) * fma(zFactor7_1, output[43], -radiusSquared * output[31]);
    output[55] = broadcast<Real>(k18) * fma(zFactor7_1, output[41], -radiusSquared * output[29]);
    output[8] = broadcast<Real>(k2) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k2) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k5) * z;
    output[22] = broadcast<Real>(k19) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k19) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    Real const zFactor5_2 = broadcast<Real>(k7) * z;
    output[32] = broadcast<Real>(k20) * fma(zFactor5_2, output[22], -radiusSquared * output[14]);
    output[28] = broadcast<Real>(k20) * fma(zFactor5_2, output[18], -radiusSquared * output[10]);
    Real const zFactor6_2 = broadcast<Real>(k9) * z;
    output[44] = broadcast<Real>(k21) * fma(zFactor6_2, output[32], -radiusSquared * output[22]);
    output[40] = broadcast<Real>(k21) * fma(zFactor6_2, output[28], -radiusSquared * output[18]);
    Real const zFactor7_2 = broadcast<Real>(k11) * z;
    output[58] = broadcast<Real>(k22) * fma(zFactor7_2, output[44], -radiusSquared * output[32]);
    output[54] = broadcast<Real>(k22) * fma(zFactor7_2, output[40], -radiusSquared * output[28]);
    output[15] = broadcast<Real>(k23) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k23) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    Real const zFactor5_3 = broadcast<Real>(k7) * z;
    output[33] = broadcast<Real>(k6) * fma(zFactor5_3, output[23], -radiusSquared * output[15]);
    output[27] = broadcast<Real>(k6) * fma(zFactor5_3, output[17], -radiusSquared * output[9]);
    Real const zFactor6_3 = broadcast<Real>(k9) * z;
    output[45] = broadcast<Real>(k24) * fma(zFactor6_3, output[33], -radiusSquared * output[23]);
    output[39] = broadcast<Real>(k24) * fma(zFactor6_3, output[27], -radiusSquared * output[17]);
    Real const zFactor7_3 = broadcast<Real>(k11) * z;
    output[59] = broadcast<Real>(k25) * fma(zFactor7_3, output[45], -radiusSquared * output[33]);
    output[53] = broadcast<Real>(k25) * fma(zFactor7_3, output[39], -radiusSquared * output[27]);
    output[24] = broadcast<Real>(k14) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k14) * fma(y, output[15], x * output[9]);
    output[34] = z * output[24];
    output[26] = z * output[16];
    Real const zFactor6_4 = broadcast<Real>(k9) * z;
    output[46] = broadcast<Real>(k26) * fma(zFactor6_4, output[34], -radiusSquared * output[24]);
    output[38] = broadcast<Real>(k26) * fma(zFactor6_4, output[26], -radiusSquared * output[16]);
    Real const zFactor7_4 = broadcast<Real>(k11) * z;
    output[60] = broadcast<Real>(k27) * fma(zFactor7_4, output[46], -radiusSquared * output[34]);
    output[52] = broadcast<Real>(k27) * fma(zFactor7_4, output[38], -radiusSquared * output[26]);
    output[35] = broadcast<Real>(k28) * fma(x, output[24], -y * output[16]);
    output[25] = broadcast<Real>(k28) * fma(y, output[24], x * output[16]);
    output[47] = z * output[35];
    output[37] = z * output[25];
    Real const zFactor7_5 = broadcast<Real>(k11) * z;
    output[61] = broadcast<Real>(k16) * fma(zFactor7_5, output[47], -radiusSquared * output[35]);
    output[51] = broadcast<Real>(k16) * fma(zFactor7_5, output[37], -radiusSquared * output[25]);
    output[48] = broadcast<Real>(k19) * fma(x, output[35], -y * output[25]);
    output[36] = broadcast<Real>(k19) * fma(y, output[35], x * output[25]);
    output[62] = z * output[48];
    output[50] = z * output[36];
    output[63] = broadcast<Real>(k29) * fma(x, output[48], -y * output[36]);
    output[49] = broadcast<Real>(k29) * fma(y, output[48], x * output[36]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 7)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(+4.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.06185567010309274E-02);
    constexpr auto k3 = static_cast<Scalar>(+2.10526315789473679E-02);
    constexpr auto k4 = static_cast<Scalar>(+2.15053763440860232E-02);
    constexpr auto k5 = static_cast<Scalar>(+2.19780219780219797E-02);
    constexpr auto k6 = static_cast<Scalar>(+2.24719101123595499E-02);
    constexpr auto k7 = static_cast<Scalar>(+2.29885057471264365E-02);
    constexpr auto k8 = static_cast<Scalar>(+2.35294117647058820E-02);
    constexpr auto k9 = static_cast<Scalar>(+2.40963855421686760E-02);
    constexpr auto k10 = static_cast<Scalar>(+2.46913580246913567E-02);
    constexpr auto k11 = static_cast<Scalar>(+2.53164556962025313E-02);
    constexpr auto k12 = static_cast<Scalar>(+2.59740259740259757E-02);
    constexpr auto k13 = static_cast<Scalar>(+2.66666666666666684E-02);
    constexpr auto k14 = static_cast<Scalar>(+2.73972602739726012E-02);
    constexpr auto k15 = static_cast<Scalar>(+2.81690140845070429E-02);
    constexpr auto k16 = static_cast<Scalar>(+2.89855072463768119E-02);
    constexpr auto k17 = static_cast<Scalar>(+2.98507462686567158E-02);
    constexpr auto k18 = static_cast<Scalar>(+3.07692307692307709E-02);
    constexpr auto k19 = static_cast<Scalar>(+3.17460317460317443E-02);
    constexpr auto k20 = static_cast<Scalar>(+3.27868852459016411E-02);
    constexpr auto k21 = static_cast<Scalar>(+3.38983050847457626E-02);
    constexpr auto k22 = static_cast<Scalar>(+3.50877192982456121E-02);
    constexpr auto k23 = static_cast<Scalar>(+3.63636363636363619E-02);
    constexpr auto k24 = static_cast<Scalar>(+3.77358490566037721E-02);
    constexpr auto k25 = static_cast<Scalar>(+3.92156862745098034E-02);
    constexpr auto k26 = static_cast<Scalar>(+4.08163265306122416E-02);
    constexpr auto k27 = static_cast<Scalar>(+4.25531914893617011E-02);
    constexpr auto k28 = static_cast<Scalar>(+4.44444444444444461E-02);
    constexpr auto k29 = static_cast<Scalar>(+4.65116279069767435E-02);
    constexpr auto k30 = static_cast<Scalar>(+4.87804878048780505E-02);
    constexpr auto k31 = static_cast<Scalar>(+5.12820512820512803E-02);
    constexpr auto k32 = static_cast<Scalar>(+5.40540540540540571E-02);
    constexpr auto k33 = static_cast<Scalar>(+5.71428571428571411E-02);
    constexpr auto k34 = static_cast<Scalar>(+6.06060606060606077E-02);
    constexpr auto k35 = static_cast<Scalar>(+6.45161290322580627E-02);
    constexpr auto k36 = static_cast<Scalar>(+6.89655172413793094E-02);
    constexpr auto k37 = static_cast<Scalar>(+7.40740740740740700E-02);
    constexpr auto k38 = static_cast<Scalar>(+8.00000000000000017E-02);
    constexpr auto k39 = static_cast<Scalar>(+8.69565217391304324E-02);
    constexpr auto k40 = static_cast<Scalar>(+9.52380952380952328E-02);
    constexpr auto k41 = static_cast<Scalar>(+1.05263157894736836E-01);
    constexpr auto k42 = static_cast<Scalar>(+1.17647058823529410E-01);
    constexpr auto k43 = static_cast<Scalar>(+1.33333333333333331E-01);
    constexpr auto k44 = static_cast<Scalar>(+1.53846153846153855E-01);
    constexpr auto k45 = static_cast<Scalar>(+1.81818181818181823E-01);
    constexpr auto k46 = static_cast<Scalar>(+2.22222222222222210E-01);
    constexpr auto k47 = static_cast<Scalar>(+2.85714285714285698E-01);
    constexpr auto k48 = static_cast<Scalar>(+4.00000000000000022E-01);
    constexpr auto k49 = static_cast<Scalar>(+6.66666666666666630E-01);
    constexpr auto k50 = static_cast<Scalar>(+2.25675833419102512E+00);
    constexpr auto k51 = static_cast<Scalar>(+0.00000000000000000E+00);
    constexpr auto k52 = static_cast<Scalar>(+2.02020202020202037E-02);
    constexpr auto k53 = static_cast<Scalar>(+6.01802222450939972E+00);
    constexpr auto k54 = static_cast<Scalar>(+2.00000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(+1.98019801980198022E-02);
    constexpr auto k56 = static_cast<Scalar>(+9.62883555921504275E+00);
    constexpr auto k57 = static_cast<Scalar>(+2.40000000000000000E+01);
    constexpr auto k58 = static_cast<Scalar>(+6.00000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(+1.94174757281553381E-02);
    constexpr auto k60 = static_cast<Scalar>(+1.10043834962457634E+01);
    constexpr auto k61 = static_cast<Scalar>(+7.20000000000000000E+02);
    constexpr auto k62 = static_cast<Scalar>(+1.20000000000000000E+02);
    constexpr auto k63 = static_cast<Scalar>(+4.80000000000000000E+01);
    constexpr auto k64 = static_cast<Scalar>(+3.60000000000000000E+01);
    constexpr auto k65 = static_cast<Scalar>(+1.90476190476190493E-02);
    constexpr auto k66 = static_cast<Scalar>(+9.78167421888512045E+00);
    constexpr auto k67 = static_cast<Scalar>(+4.03200000000000000E+04);
    constexpr auto k68 = static_cast<Scalar>(+5.04000000000000000E+03);
    constexpr auto k69 = static_cast<Scalar>(+1.44000000000000000E+03);
    constexpr auto k70 = static_cast<Scalar>(+5.76000000000000000E+02);
    constexpr auto k71 = static_cast<Scalar>(+1.86915887850467276E-02);
    constexpr auto k72 = static_cast<Scalar>(+7.11394488646190482E+00);
    constexpr auto k73 = static_cast<Scalar>(+3.62880000000000000E+06);
    constexpr auto k74 = static_cast<Scalar>(+3.62880000000000000E+05);
    constexpr auto k75 = static_cast<Scalar>(+8.06400000000000000E+04);
    constexpr auto k76 = static_cast<Scalar>(+3.02400000000000000E+04);
    constexpr auto k77 = static_cast<Scalar>(+1.72800000000000000E+04);
    constexpr auto k78 = static_cast<Scalar>(+1.44000000000000000E+04);
    constexpr auto k79 = static_cast<Scalar>(+1.83486238532110102E-02);
    constexpr auto k80 = static_cast<Scalar>(+4.37781223782271134E+00);
    constexpr auto k81 = static_cast<Scalar>(+4.79001600000000000E+08);
    constexpr auto k82 = static_cast<Scalar>(+3.99168000000000000E+07);
    constexpr auto k83 = static_cast<Scalar>(+7.25760000000000000E+06);
    constexpr auto k84 = static_cast<Scalar>(+2.17728000000000000E+06);
    constexpr auto k85 = static_cast<Scalar>(+9.67680000000000000E+05);
    constexpr auto k86 = static_cast<Scalar>(+6.04800000000000000E+05);
    constexpr auto k87 = static_cast<Scalar>(+5.18400000000000000E+05);
    constexpr auto k88 = static_cast<Scalar>(+1.80180180180180179E-02);
    constexpr auto k89 = static_cast<Scalar>(+2.33483319350544605E+00);
    constexpr auto k90 = static_cast<Scalar>(+8.71782912000000000E+10);
    constexpr auto k91 = static_cast<Scalar>(+6.22702080000000000E+09);
    constexpr auto k92 = static_cast<Scalar>(+9.58003200000000000E+08);
    constexpr auto k93 = static_cast<Scalar>(+2.39500800000000000E+08);
    constexpr auto k94 = static_cast<Scalar>(+8.70912000000000000E+07);
    constexpr auto k95 = static_cast<Scalar>(+4.35456000000000000E+07);
    constexpr auto k96 = static_cast<Scalar>(+2.90304000000000000E+07);
    constexpr auto k97 = static_cast<Scalar>(+2.54016000000000000E+07);
    constexpr auto k98 = static_cast<Scalar>(+7.85398163397448279E-01);
    constexpr auto k99 = static_cast<Scalar>(+4.18879020478639053E+00);
    constexpr auto k100 = static_cast<Scalar>(+6.28318530717958623E+00);
    constexpr auto k101 = static_cast<Scalar>(+6.36619772367581382E-01);
    constexpr auto k102 = static_cast<Scalar>(+2.46740110027233950E+00);
    constexpr auto k103 = static_cast<Scalar>(+1.31594725347858112E+01);
    constexpr auto k104 = static_cast<Scalar>(+1.65366808961599006E+01);
    constexpr auto k105 = static_cast<Scalar>(+1.48432900623241775E+01);
    constexpr auto k106 = static_cast<Scalar>(+1.03625713366444501E+01);
    constexpr auto k107 = static_cast<Scalar>(+5.91908690609133981E+00);
    constexpr auto k108 = static_cast<Scalar>(+2.86082460617478285E+00);
    Real harmonics[64];
    Real compensation[64];
    Real imagePair[64];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k0) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k1);
    series0 = fma(series0, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k1));
    Real const radial0 = broadcast<Real>(k50) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k51);
    Real series1 = broadcast<Real>(k1);
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    Real const radial1 = -broadcast<Real>(k53) * exponential * series1;
    output[1] = broadcast<Real>(k54) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k51);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k51);
    output[3] = broadcast<Real>(k54) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k51);
    Real series2 = broadcast<Real>(k1);
    series2 = fma(series2, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    Real const radial2 = broadcast<Real>(k56) * exponential * series2;
    output[4] = broadcast<Real>(k57) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k51);
    output[5] = broadcast<Real>(k58) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k51);
    output[6] = broadcast<Real>(k0) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k51);
    output[7] = broadcast<Real>(k58) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k51);
    output[8] = broadcast<Real>(k57) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k51);
    Real series3 = broadcast<Real>(k1);
    series3 = fma(series3, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    Real const radial3 = -broadcast<Real>(k60) * exponential * series3;
    output[9] = broadcast<Real>(k61) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k51);
    output[10] = broadcast<Real>(k62) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k51);
    output[11] = broadcast<Real>(k63) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k51);
    output[12] = broadcast<Real>(k64) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k51);
    output[13] = broadcast<Real>(k63) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k51);
    output[14] = broadcast<Real>(k62) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k51);
    output[15] = broadcast<Real>(k61) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k51);
    Real series4 = broadcast<Real>(k1);
    series4 = fma(series4, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    Real const radial4 = broadcast<Real>(k66) * exponential * series4;
    output[16] = broadcast<Real>(k67) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k51);
    output[17] = broadcast<Real>(k68) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k51);
    output[18] = broadcast<Real>(k69) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k51);
    output[19] = broadcast<Real>(k61) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k51);
    output[20] = broadcast<Real>(k70) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k51);
    output[21] = broadcast<Real>(k61) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k51);
    output[22] = broadcast<Real>(k69) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k51);
    output[23] = broadcast<Real>(k68) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k51);
    output[24] = broadcast<Real>(k67) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k51);
    Real series5 = broadcast<Real>(k1);
    series5 = fma(series5, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    Real const radial5 = -broadcast<Real>(k72) * exponential * series5;
    output[25] = broadcast<Real>(k73) * radial5 * harmonics[25];
    compensation[25] = broadcast<Real>(k51);
    output[26] = broadcast<Real>(k74) * radial5 * harmonics[26];
    compensation[26] = broadcast<Real>(k51);
    output[27] = broadcast<Real>(k75) * radial5 * harmonics[27];
    compensation[27] = broadcast<Real>(k51);
    output[28] = broadcast<Real>(k76) * radial5 * harmonics[28];
    compensation[28] = broadcast<Real>(k51);
    output[29] = broadcast<Real>(k77) * radial5 * harmonics[29];
    compensation[29] = broadcast<Real>(k51);
    output[30] = broadcast<Real>(k78) * radial5 * harmonics[30];
    compensation[30] = broadcast<Real>(k51);
    output[31] = broadcast<Real>(k77) * radial5 * harmonics[31];
    compensation[31] = broadcast<Real>(k51);
    output[32] = broadcast<Real>(k76) * radial5 * harmonics[32];
    compensation[32] = broadcast<Real>(k51);
    output[33] = broadcast<Real>(k75) * radial5 * harmonics[33];
    compensation[33] = broadcast<Real>(k51);
    output[34] = broadcast<Real>(k74) * radial5 * harmonics[34];
    compensation[34] = broadcast<Real>(k51);
    output[35] = broadcast<Real>(k73) * radial5 * harmonics[35];
    compensation[35] = broadcast<Real>(k51);
    Real series6 = broadcast<Real>(k1);
    series6 = fma(series6, argument * broadcast<Real>(k79), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    Real const radial6 = broadcast<Real>(k80) * exponential * series6;
    output[36] = broadcast<Real>(k81) * radial6 * harmonics[36];
    compensation[36] = broadcast<Real>(k51);
    output[37] = broadcast<Real>(k82) * radial6 * harmonics[37];
    compensation[37] = broadcast<Real>(k51);
    output[38] = broadcast<Real>(k83) * radial6 * harmonics[38];
    compensation[38] = broadcast<Real>(k51);
    output[39] = broadcast<Real>(k84) * radial6 * harmonics[39];
    compensation[39] = broadcast<Real>(k51);
    output[40] = broadcast<Real>(k85) * radial6 * harmonics[40];
    compensation[40] = broadcast<Real>(k51);
    output[41] = broadcast<Real>(k86) * radial6 * harmonics[41];
    compensation[41] = broadcast<Real>(k51);
    output[42] = broadcast<Real>(k87) * radial6 * harmonics[42];
    compensation[42] = broadcast<Real>(k51);
    output[43] = broadcast<Real>(k86) * radial6 * harmonics[43];
    compensation[43] = broadcast<Real>(k51);
    output[44] = broadcast<Real>(k85) * radial6 * harmonics[44];
    compensation[44] = broadcast<Real>(k51);
    output[45] = broadcast<Real>(k84) * radial6 * harmonics[45];
    compensation[45] = broadcast<Real>(k51);
    output[46] = broadcast<Real>(k83) * radial6 * harmonics[46];
    compensation[46] = broadcast<Real>(k51);
    output[47] = broadcast<Real>(k82) * radial6 * harmonics[47];
    compensation[47] = broadcast<Real>(k51);
    output[48] = broadcast<Real>(k81) * radial6 * harmonics[48];
    compensation[48] = broadcast<Real>(k51);
    Real series7 = broadcast<Real>(k1);
    series7 = fma(series7, argument * broadcast<Real>(k88), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k79), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    Real const radial7 = -broadcast<Real>(k89) * exponential * series7;
    output[49] = broadcast<Real>(k90) * radial7 * harmonics[49];
    compensation[49] = broadcast<Real>(k51);
    output[50] = broadcast<Real>(k91) * radial7 * harmonics[50];
    compensation[50] = broadcast<Real>(k51);
    output[51] = broadcast<Real>(k92) * radial7 * harmonics[51];
    compensation[51] = broadcast<Real>(k51);
    output[52] = broadcast<Real>(k93) * radial7 * harmonics[52];
    compensation[52] = broadcast<Real>(k51);
    output[53] = broadcast<Real>(k94) * radial7 * harmonics[53];
    compensation[53] = broadcast<Real>(k51);
    output[54] = broadcast<Real>(k95) * radial7 * harmonics[54];
    compensation[54] = broadcast<Real>(k51);
    output[55] = broadcast<Real>(k96) * radial7 * harmonics[55];
    compensation[55] = broadcast<Real>(k51);
    output[56] = broadcast<Real>(k97) * radial7 * harmonics[56];
    compensation[56] = broadcast<Real>(k51);
    output[57] = broadcast<Real>(k96) * radial7 * harmonics[57];
    compensation[57] = broadcast<Real>(k51);
    output[58] = broadcast<Real>(k95) * radial7 * harmonics[58];
    compensation[58] = broadcast<Real>(k51);
    output[59] = broadcast<Real>(k94) * radial7 * harmonics[59];
    compensation[59] = broadcast<Real>(k51);
    output[60] = broadcast<Real>(k93) * radial7 * harmonics[60];
    compensation[60] = broadcast<Real>(k51);
    output[61] = broadcast<Real>(k92) * radial7 * harmonics[61];
    compensation[61] = broadcast<Real>(k51);
    output[62] = broadcast<Real>(k91) * radial7 * harmonics[62];
    compensation[62] = broadcast<Real>(k51);
    output[63] = broadcast<Real>(k90) * radial7 * harmonics[63];
    compensation[63] = broadcast<Real>(k51);
    output[0] += broadcast<Real>(k98);
    greenTrace = broadcast<Real>(k99);
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
                    Real const iR = broadcast<Real>(k1) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k0) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k54) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k50) * imageRadius * imageExp;
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
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
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
                    gammaTerm *= imageArgument * broadcast<Real>(k48);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k57) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k58) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k0) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k58) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k57) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k47);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k61) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k62) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k63) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k64) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k63) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k62) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k61) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k46);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k67) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k68) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k69) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k61) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k70) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k61) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k69) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k68) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k67) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k45);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k73) * imageRadial5 * harmonics[25];
                    if (sign < 0) {
                        imagePair[25] = imageValue25;
                    } else {
                        imagePair[25] += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k74) * imageRadial5 * harmonics[26];
                    if (sign < 0) {
                        imagePair[26] = imageValue26;
                    } else {
                        imagePair[26] += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k75) * imageRadial5 * harmonics[27];
                    if (sign < 0) {
                        imagePair[27] = imageValue27;
                    } else {
                        imagePair[27] += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k76) * imageRadial5 * harmonics[28];
                    if (sign < 0) {
                        imagePair[28] = imageValue28;
                    } else {
                        imagePair[28] += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k77) * imageRadial5 * harmonics[29];
                    if (sign < 0) {
                        imagePair[29] = imageValue29;
                    } else {
                        imagePair[29] += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k78) * imageRadial5 * harmonics[30];
                    if (sign < 0) {
                        imagePair[30] = imageValue30;
                    } else {
                        imagePair[30] += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k77) * imageRadial5 * harmonics[31];
                    if (sign < 0) {
                        imagePair[31] = imageValue31;
                    } else {
                        imagePair[31] += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k76) * imageRadial5 * harmonics[32];
                    if (sign < 0) {
                        imagePair[32] = imageValue32;
                    } else {
                        imagePair[32] += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k75) * imageRadial5 * harmonics[33];
                    if (sign < 0) {
                        imagePair[33] = imageValue33;
                    } else {
                        imagePair[33] += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k74) * imageRadial5 * harmonics[34];
                    if (sign < 0) {
                        imagePair[34] = imageValue34;
                    } else {
                        imagePair[34] += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k73) * imageRadial5 * harmonics[35];
                    if (sign < 0) {
                        imagePair[35] = imageValue35;
                    } else {
                        imagePair[35] += imageValue35;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k44);
                    Real const imageRadial6 = -screening * inversePower;
                    Real const imageValue36 = broadcast<Real>(k81) * imageRadial6 * harmonics[36];
                    if (sign < 0) {
                        imagePair[36] = imageValue36;
                    } else {
                        imagePair[36] += imageValue36;
                    }
                    Real const imageValue37 = broadcast<Real>(k82) * imageRadial6 * harmonics[37];
                    if (sign < 0) {
                        imagePair[37] = imageValue37;
                    } else {
                        imagePair[37] += imageValue37;
                    }
                    Real const imageValue38 = broadcast<Real>(k83) * imageRadial6 * harmonics[38];
                    if (sign < 0) {
                        imagePair[38] = imageValue38;
                    } else {
                        imagePair[38] += imageValue38;
                    }
                    Real const imageValue39 = broadcast<Real>(k84) * imageRadial6 * harmonics[39];
                    if (sign < 0) {
                        imagePair[39] = imageValue39;
                    } else {
                        imagePair[39] += imageValue39;
                    }
                    Real const imageValue40 = broadcast<Real>(k85) * imageRadial6 * harmonics[40];
                    if (sign < 0) {
                        imagePair[40] = imageValue40;
                    } else {
                        imagePair[40] += imageValue40;
                    }
                    Real const imageValue41 = broadcast<Real>(k86) * imageRadial6 * harmonics[41];
                    if (sign < 0) {
                        imagePair[41] = imageValue41;
                    } else {
                        imagePair[41] += imageValue41;
                    }
                    Real const imageValue42 = broadcast<Real>(k87) * imageRadial6 * harmonics[42];
                    if (sign < 0) {
                        imagePair[42] = imageValue42;
                    } else {
                        imagePair[42] += imageValue42;
                    }
                    Real const imageValue43 = broadcast<Real>(k86) * imageRadial6 * harmonics[43];
                    if (sign < 0) {
                        imagePair[43] = imageValue43;
                    } else {
                        imagePair[43] += imageValue43;
                    }
                    Real const imageValue44 = broadcast<Real>(k85) * imageRadial6 * harmonics[44];
                    if (sign < 0) {
                        imagePair[44] = imageValue44;
                    } else {
                        imagePair[44] += imageValue44;
                    }
                    Real const imageValue45 = broadcast<Real>(k84) * imageRadial6 * harmonics[45];
                    if (sign < 0) {
                        imagePair[45] = imageValue45;
                    } else {
                        imagePair[45] += imageValue45;
                    }
                    Real const imageValue46 = broadcast<Real>(k83) * imageRadial6 * harmonics[46];
                    if (sign < 0) {
                        imagePair[46] = imageValue46;
                    } else {
                        imagePair[46] += imageValue46;
                    }
                    Real const imageValue47 = broadcast<Real>(k82) * imageRadial6 * harmonics[47];
                    if (sign < 0) {
                        imagePair[47] = imageValue47;
                    } else {
                        imagePair[47] += imageValue47;
                    }
                    Real const imageValue48 = broadcast<Real>(k81) * imageRadial6 * harmonics[48];
                    if (sign < 0) {
                        imagePair[48] = imageValue48;
                    } else {
                        imagePair[48] += imageValue48;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k43);
                    Real const imageRadial7 = screening * inversePower;
                    Real const imageValue49 = broadcast<Real>(k90) * imageRadial7 * harmonics[49];
                    if (sign < 0) {
                        imagePair[49] = imageValue49;
                    } else {
                        imagePair[49] += imageValue49;
                    }
                    Real const imageValue50 = broadcast<Real>(k91) * imageRadial7 * harmonics[50];
                    if (sign < 0) {
                        imagePair[50] = imageValue50;
                    } else {
                        imagePair[50] += imageValue50;
                    }
                    Real const imageValue51 = broadcast<Real>(k92) * imageRadial7 * harmonics[51];
                    if (sign < 0) {
                        imagePair[51] = imageValue51;
                    } else {
                        imagePair[51] += imageValue51;
                    }
                    Real const imageValue52 = broadcast<Real>(k93) * imageRadial7 * harmonics[52];
                    if (sign < 0) {
                        imagePair[52] = imageValue52;
                    } else {
                        imagePair[52] += imageValue52;
                    }
                    Real const imageValue53 = broadcast<Real>(k94) * imageRadial7 * harmonics[53];
                    if (sign < 0) {
                        imagePair[53] = imageValue53;
                    } else {
                        imagePair[53] += imageValue53;
                    }
                    Real const imageValue54 = broadcast<Real>(k95) * imageRadial7 * harmonics[54];
                    if (sign < 0) {
                        imagePair[54] = imageValue54;
                    } else {
                        imagePair[54] += imageValue54;
                    }
                    Real const imageValue55 = broadcast<Real>(k96) * imageRadial7 * harmonics[55];
                    if (sign < 0) {
                        imagePair[55] = imageValue55;
                    } else {
                        imagePair[55] += imageValue55;
                    }
                    Real const imageValue56 = broadcast<Real>(k97) * imageRadial7 * harmonics[56];
                    if (sign < 0) {
                        imagePair[56] = imageValue56;
                    } else {
                        imagePair[56] += imageValue56;
                    }
                    Real const imageValue57 = broadcast<Real>(k96) * imageRadial7 * harmonics[57];
                    if (sign < 0) {
                        imagePair[57] = imageValue57;
                    } else {
                        imagePair[57] += imageValue57;
                    }
                    Real const imageValue58 = broadcast<Real>(k95) * imageRadial7 * harmonics[58];
                    if (sign < 0) {
                        imagePair[58] = imageValue58;
                    } else {
                        imagePair[58] += imageValue58;
                    }
                    Real const imageValue59 = broadcast<Real>(k94) * imageRadial7 * harmonics[59];
                    if (sign < 0) {
                        imagePair[59] = imageValue59;
                    } else {
                        imagePair[59] += imageValue59;
                    }
                    Real const imageValue60 = broadcast<Real>(k93) * imageRadial7 * harmonics[60];
                    if (sign < 0) {
                        imagePair[60] = imageValue60;
                    } else {
                        imagePair[60] += imageValue60;
                    }
                    Real const imageValue61 = broadcast<Real>(k92) * imageRadial7 * harmonics[61];
                    if (sign < 0) {
                        imagePair[61] = imageValue61;
                    } else {
                        imagePair[61] += imageValue61;
                    }
                    Real const imageValue62 = broadcast<Real>(k91) * imageRadial7 * harmonics[62];
                    if (sign < 0) {
                        imagePair[62] = imageValue62;
                    } else {
                        imagePair[62] += imageValue62;
                    }
                    Real const imageValue63 = broadcast<Real>(k90) * imageRadial7 * harmonics[63];
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
                Real const phase = broadcast<Real>(k100) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k101) *
                    detail::ewaldExp(-broadcast<Real>(k102) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k1) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k100) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k54) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k54) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k103) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k57) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k58) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k0) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k58) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k57) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k104) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k61) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k64) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k61) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k105) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k70) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[24]);
                Real const waveRadial5 = broadcast<Real>(k106) * wavePotential * -sine;
                detail::ewaldAccumulate(output[25], compensation[25],
                    broadcast<Real>(k73) * waveRadial5 * harmonics[25]);
                detail::ewaldAccumulate(output[26], compensation[26],
                    broadcast<Real>(k74) * waveRadial5 * harmonics[26]);
                detail::ewaldAccumulate(output[27], compensation[27],
                    broadcast<Real>(k75) * waveRadial5 * harmonics[27]);
                detail::ewaldAccumulate(output[28], compensation[28],
                    broadcast<Real>(k76) * waveRadial5 * harmonics[28]);
                detail::ewaldAccumulate(output[29], compensation[29],
                    broadcast<Real>(k77) * waveRadial5 * harmonics[29]);
                detail::ewaldAccumulate(output[30], compensation[30],
                    broadcast<Real>(k78) * waveRadial5 * harmonics[30]);
                detail::ewaldAccumulate(output[31], compensation[31],
                    broadcast<Real>(k77) * waveRadial5 * harmonics[31]);
                detail::ewaldAccumulate(output[32], compensation[32],
                    broadcast<Real>(k76) * waveRadial5 * harmonics[32]);
                detail::ewaldAccumulate(output[33], compensation[33],
                    broadcast<Real>(k75) * waveRadial5 * harmonics[33]);
                detail::ewaldAccumulate(output[34], compensation[34],
                    broadcast<Real>(k74) * waveRadial5 * harmonics[34]);
                detail::ewaldAccumulate(output[35], compensation[35],
                    broadcast<Real>(k73) * waveRadial5 * harmonics[35]);
                Real const waveRadial6 = broadcast<Real>(k107) * wavePotential * -cosine;
                detail::ewaldAccumulate(output[36], compensation[36],
                    broadcast<Real>(k81) * waveRadial6 * harmonics[36]);
                detail::ewaldAccumulate(output[37], compensation[37],
                    broadcast<Real>(k82) * waveRadial6 * harmonics[37]);
                detail::ewaldAccumulate(output[38], compensation[38],
                    broadcast<Real>(k83) * waveRadial6 * harmonics[38]);
                detail::ewaldAccumulate(output[39], compensation[39],
                    broadcast<Real>(k84) * waveRadial6 * harmonics[39]);
                detail::ewaldAccumulate(output[40], compensation[40],
                    broadcast<Real>(k85) * waveRadial6 * harmonics[40]);
                detail::ewaldAccumulate(output[41], compensation[41],
                    broadcast<Real>(k86) * waveRadial6 * harmonics[41]);
                detail::ewaldAccumulate(output[42], compensation[42],
                    broadcast<Real>(k87) * waveRadial6 * harmonics[42]);
                detail::ewaldAccumulate(output[43], compensation[43],
                    broadcast<Real>(k86) * waveRadial6 * harmonics[43]);
                detail::ewaldAccumulate(output[44], compensation[44],
                    broadcast<Real>(k85) * waveRadial6 * harmonics[44]);
                detail::ewaldAccumulate(output[45], compensation[45],
                    broadcast<Real>(k84) * waveRadial6 * harmonics[45]);
                detail::ewaldAccumulate(output[46], compensation[46],
                    broadcast<Real>(k83) * waveRadial6 * harmonics[46]);
                detail::ewaldAccumulate(output[47], compensation[47],
                    broadcast<Real>(k82) * waveRadial6 * harmonics[47]);
                detail::ewaldAccumulate(output[48], compensation[48],
                    broadcast<Real>(k81) * waveRadial6 * harmonics[48]);
                Real const waveRadial7 = broadcast<Real>(k108) * wavePotential * sine;
                detail::ewaldAccumulate(output[49], compensation[49],
                    broadcast<Real>(k90) * waveRadial7 * harmonics[49]);
                detail::ewaldAccumulate(output[50], compensation[50],
                    broadcast<Real>(k91) * waveRadial7 * harmonics[50]);
                detail::ewaldAccumulate(output[51], compensation[51],
                    broadcast<Real>(k92) * waveRadial7 * harmonics[51]);
                detail::ewaldAccumulate(output[52], compensation[52],
                    broadcast<Real>(k93) * waveRadial7 * harmonics[52]);
                detail::ewaldAccumulate(output[53], compensation[53],
                    broadcast<Real>(k94) * waveRadial7 * harmonics[53]);
                detail::ewaldAccumulate(output[54], compensation[54],
                    broadcast<Real>(k95) * waveRadial7 * harmonics[54]);
                detail::ewaldAccumulate(output[55], compensation[55],
                    broadcast<Real>(k96) * waveRadial7 * harmonics[55]);
                detail::ewaldAccumulate(output[56], compensation[56],
                    broadcast<Real>(k97) * waveRadial7 * harmonics[56]);
                detail::ewaldAccumulate(output[57], compensation[57],
                    broadcast<Real>(k96) * waveRadial7 * harmonics[57]);
                detail::ewaldAccumulate(output[58], compensation[58],
                    broadcast<Real>(k95) * waveRadial7 * harmonics[58]);
                detail::ewaldAccumulate(output[59], compensation[59],
                    broadcast<Real>(k94) * waveRadial7 * harmonics[59]);
                detail::ewaldAccumulate(output[60], compensation[60],
                    broadcast<Real>(k93) * waveRadial7 * harmonics[60]);
                detail::ewaldAccumulate(output[61], compensation[61],
                    broadcast<Real>(k92) * waveRadial7 * harmonics[61]);
                detail::ewaldAccumulate(output[62], compensation[62],
                    broadcast<Real>(k91) * waveRadial7 * harmonics[62]);
                detail::ewaldAccumulate(output[63], compensation[63],
                    broadcast<Real>(k90) * waveRadial7 * harmonics[63]);
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
    constexpr auto k0 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+3.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.50000000000000000E-01);
    constexpr auto k3 = static_cast<Scalar>(+5.00000000000000000E+00);
    constexpr auto k4 = static_cast<Scalar>(+1.11111111111111105E-01);
    constexpr auto k5 = static_cast<Scalar>(+7.00000000000000000E+00);
    constexpr auto k6 = static_cast<Scalar>(+6.25000000000000000E-02);
    constexpr auto k7 = static_cast<Scalar>(+9.00000000000000000E+00);
    constexpr auto k8 = static_cast<Scalar>(+4.00000000000000008E-02);
    constexpr auto k9 = static_cast<Scalar>(+1.10000000000000000E+01);
    constexpr auto k10 = static_cast<Scalar>(+2.77777777777777762E-02);
    constexpr auto k11 = static_cast<Scalar>(+1.30000000000000000E+01);
    constexpr auto k12 = static_cast<Scalar>(+2.04081632653061208E-02);
    constexpr auto k13 = static_cast<Scalar>(+1.50000000000000000E+01);
    constexpr auto k14 = static_cast<Scalar>(+1.56250000000000000E-02);
    constexpr auto k15 = static_cast<Scalar>(+5.00000000000000000E-01);
    constexpr auto k16 = static_cast<Scalar>(+1.25000000000000000E-01);
    constexpr auto k17 = static_cast<Scalar>(+6.66666666666666657E-02);
    constexpr auto k18 = static_cast<Scalar>(+4.16666666666666644E-02);
    constexpr auto k19 = static_cast<Scalar>(+2.85714285714285705E-02);
    constexpr auto k20 = static_cast<Scalar>(+2.08333333333333322E-02);
    constexpr auto k21 = static_cast<Scalar>(+1.58730158730158721E-02);
    constexpr auto k22 = static_cast<Scalar>(+8.33333333333333287E-02);
    constexpr auto k23 = static_cast<Scalar>(+4.76190476190476164E-02);
    constexpr auto k24 = static_cast<Scalar>(+3.12500000000000000E-02);
    constexpr auto k25 = static_cast<Scalar>(+2.22222222222222231E-02);
    constexpr auto k26 = static_cast<Scalar>(+1.66666666666666664E-02);
    constexpr auto k27 = static_cast<Scalar>(+1.66666666666666657E-01);
    constexpr auto k28 = static_cast<Scalar>(+3.70370370370370350E-02);
    constexpr auto k29 = static_cast<Scalar>(+2.50000000000000014E-02);
    constexpr auto k30 = static_cast<Scalar>(+1.81818181818181809E-02);
    constexpr auto k31 = static_cast<Scalar>(+5.00000000000000028E-02);
    constexpr auto k32 = static_cast<Scalar>(+3.03030303030303039E-02);
    constexpr auto k33 = static_cast<Scalar>(+1.00000000000000006E-01);
    constexpr auto k34 = static_cast<Scalar>(+2.56410256410256401E-02);
    constexpr auto k35 = static_cast<Scalar>(+3.57142857142857123E-02);
    constexpr auto k36 = static_cast<Scalar>(+7.14285714285714246E-02);
    output[0] = broadcast<Real>(k0);
    output[2] = z * output[0];
    Real const zFactor2_0 = broadcast<Real>(k1) * z;
    output[6] = broadcast<Real>(k2) * fma(zFactor2_0, output[2], -radiusSquared * output[0]);
    Real const zFactor3_0 = broadcast<Real>(k3) * z;
    output[12] = broadcast<Real>(k4) * fma(zFactor3_0, output[6], -radiusSquared * output[2]);
    Real const zFactor4_0 = broadcast<Real>(k5) * z;
    output[20] = broadcast<Real>(k6) * fma(zFactor4_0, output[12], -radiusSquared * output[6]);
    Real const zFactor5_0 = broadcast<Real>(k7) * z;
    output[30] = broadcast<Real>(k8) * fma(zFactor5_0, output[20], -radiusSquared * output[12]);
    Real const zFactor6_0 = broadcast<Real>(k9) * z;
    output[42] = broadcast<Real>(k10) * fma(zFactor6_0, output[30], -radiusSquared * output[20]);
    Real const zFactor7_0 = broadcast<Real>(k11) * z;
    output[56] = broadcast<Real>(k12) * fma(zFactor7_0, output[42], -radiusSquared * output[30]);
    Real const zFactor8_0 = broadcast<Real>(k13) * z;
    output[72] = broadcast<Real>(k14) * fma(zFactor8_0, output[56], -radiusSquared * output[42]);
    output[3] = x * broadcast<Real>(k15);
    output[1] = y * broadcast<Real>(k15);
    output[7] = z * output[3];
    output[5] = z * output[1];
    Real const zFactor3_1 = broadcast<Real>(k3) * z;
    output[13] = broadcast<Real>(k16) * fma(zFactor3_1, output[7], -radiusSquared * output[3]);
    output[11] = broadcast<Real>(k16) * fma(zFactor3_1, output[5], -radiusSquared * output[1]);
    Real const zFactor4_1 = broadcast<Real>(k5) * z;
    output[21] = broadcast<Real>(k17) * fma(zFactor4_1, output[13], -radiusSquared * output[7]);
    output[19] = broadcast<Real>(k17) * fma(zFactor4_1, output[11], -radiusSquared * output[5]);
    Real const zFactor5_1 = broadcast<Real>(k7) * z;
    output[31] = broadcast<Real>(k18) * fma(zFactor5_1, output[21], -radiusSquared * output[13]);
    output[29] = broadcast<Real>(k18) * fma(zFactor5_1, output[19], -radiusSquared * output[11]);
    Real const zFactor6_1 = broadcast<Real>(k9) * z;
    output[43] = broadcast<Real>(k19) * fma(zFactor6_1, output[31], -radiusSquared * output[21]);
    output[41] = broadcast<Real>(k19) * fma(zFactor6_1, output[29], -radiusSquared * output[19]);
    Real const zFactor7_1 = broadcast<Real>(k11) * z;
    output[57] = broadcast<Real>(k20) * fma(zFactor7_1, output[43], -radiusSquared * output[31]);
    output[55] = broadcast<Real>(k20) * fma(zFactor7_1, output[41], -radiusSquared * output[29]);
    Real const zFactor8_1 = broadcast<Real>(k13) * z;
    output[73] = broadcast<Real>(k21) * fma(zFactor8_1, output[57], -radiusSquared * output[43]);
    output[71] = broadcast<Real>(k21) * fma(zFactor8_1, output[55], -radiusSquared * output[41]);
    output[8] = broadcast<Real>(k2) * fma(x, output[3], -y * output[1]);
    output[4] = broadcast<Real>(k2) * fma(y, output[3], x * output[1]);
    output[14] = z * output[8];
    output[10] = z * output[4];
    Real const zFactor4_2 = broadcast<Real>(k5) * z;
    output[22] = broadcast<Real>(k22) * fma(zFactor4_2, output[14], -radiusSquared * output[8]);
    output[18] = broadcast<Real>(k22) * fma(zFactor4_2, output[10], -radiusSquared * output[4]);
    Real const zFactor5_2 = broadcast<Real>(k7) * z;
    output[32] = broadcast<Real>(k23) * fma(zFactor5_2, output[22], -radiusSquared * output[14]);
    output[28] = broadcast<Real>(k23) * fma(zFactor5_2, output[18], -radiusSquared * output[10]);
    Real const zFactor6_2 = broadcast<Real>(k9) * z;
    output[44] = broadcast<Real>(k24) * fma(zFactor6_2, output[32], -radiusSquared * output[22]);
    output[40] = broadcast<Real>(k24) * fma(zFactor6_2, output[28], -radiusSquared * output[18]);
    Real const zFactor7_2 = broadcast<Real>(k11) * z;
    output[58] = broadcast<Real>(k25) * fma(zFactor7_2, output[44], -radiusSquared * output[32]);
    output[54] = broadcast<Real>(k25) * fma(zFactor7_2, output[40], -radiusSquared * output[28]);
    Real const zFactor8_2 = broadcast<Real>(k13) * z;
    output[74] = broadcast<Real>(k26) * fma(zFactor8_2, output[58], -radiusSquared * output[44]);
    output[70] = broadcast<Real>(k26) * fma(zFactor8_2, output[54], -radiusSquared * output[40]);
    output[15] = broadcast<Real>(k27) * fma(x, output[8], -y * output[4]);
    output[9] = broadcast<Real>(k27) * fma(y, output[8], x * output[4]);
    output[23] = z * output[15];
    output[17] = z * output[9];
    Real const zFactor5_3 = broadcast<Real>(k7) * z;
    output[33] = broadcast<Real>(k6) * fma(zFactor5_3, output[23], -radiusSquared * output[15]);
    output[27] = broadcast<Real>(k6) * fma(zFactor5_3, output[17], -radiusSquared * output[9]);
    Real const zFactor6_3 = broadcast<Real>(k9) * z;
    output[45] = broadcast<Real>(k28) * fma(zFactor6_3, output[33], -radiusSquared * output[23]);
    output[39] = broadcast<Real>(k28) * fma(zFactor6_3, output[27], -radiusSquared * output[17]);
    Real const zFactor7_3 = broadcast<Real>(k11) * z;
    output[59] = broadcast<Real>(k29) * fma(zFactor7_3, output[45], -radiusSquared * output[33]);
    output[53] = broadcast<Real>(k29) * fma(zFactor7_3, output[39], -radiusSquared * output[27]);
    Real const zFactor8_3 = broadcast<Real>(k13) * z;
    output[75] = broadcast<Real>(k30) * fma(zFactor8_3, output[59], -radiusSquared * output[45]);
    output[69] = broadcast<Real>(k30) * fma(zFactor8_3, output[53], -radiusSquared * output[39]);
    output[24] = broadcast<Real>(k16) * fma(x, output[15], -y * output[9]);
    output[16] = broadcast<Real>(k16) * fma(y, output[15], x * output[9]);
    output[34] = z * output[24];
    output[26] = z * output[16];
    Real const zFactor6_4 = broadcast<Real>(k9) * z;
    output[46] = broadcast<Real>(k31) * fma(zFactor6_4, output[34], -radiusSquared * output[24]);
    output[38] = broadcast<Real>(k31) * fma(zFactor6_4, output[26], -radiusSquared * output[16]);
    Real const zFactor7_4 = broadcast<Real>(k11) * z;
    output[60] = broadcast<Real>(k32) * fma(zFactor7_4, output[46], -radiusSquared * output[34]);
    output[52] = broadcast<Real>(k32) * fma(zFactor7_4, output[38], -radiusSquared * output[26]);
    Real const zFactor8_4 = broadcast<Real>(k13) * z;
    output[76] = broadcast<Real>(k20) * fma(zFactor8_4, output[60], -radiusSquared * output[46]);
    output[68] = broadcast<Real>(k20) * fma(zFactor8_4, output[52], -radiusSquared * output[38]);
    output[35] = broadcast<Real>(k33) * fma(x, output[24], -y * output[16]);
    output[25] = broadcast<Real>(k33) * fma(y, output[24], x * output[16]);
    output[47] = z * output[35];
    output[37] = z * output[25];
    Real const zFactor7_5 = broadcast<Real>(k11) * z;
    output[61] = broadcast<Real>(k18) * fma(zFactor7_5, output[47], -radiusSquared * output[35]);
    output[51] = broadcast<Real>(k18) * fma(zFactor7_5, output[37], -radiusSquared * output[25]);
    Real const zFactor8_5 = broadcast<Real>(k13) * z;
    output[77] = broadcast<Real>(k34) * fma(zFactor8_5, output[61], -radiusSquared * output[47]);
    output[67] = broadcast<Real>(k34) * fma(zFactor8_5, output[51], -radiusSquared * output[37]);
    output[48] = broadcast<Real>(k22) * fma(x, output[35], -y * output[25]);
    output[36] = broadcast<Real>(k22) * fma(y, output[35], x * output[25]);
    output[62] = z * output[48];
    output[50] = z * output[36];
    Real const zFactor8_6 = broadcast<Real>(k13) * z;
    output[78] = broadcast<Real>(k35) * fma(zFactor8_6, output[62], -radiusSquared * output[48]);
    output[66] = broadcast<Real>(k35) * fma(zFactor8_6, output[50], -radiusSquared * output[36]);
    output[63] = broadcast<Real>(k36) * fma(x, output[48], -y * output[36]);
    output[49] = broadcast<Real>(k36) * fma(y, output[48], x * output[36]);
    output[79] = z * output[63];
    output[65] = z * output[49];
    output[80] = broadcast<Real>(k6) * fma(x, output[63], -y * output[49]);
    output[64] = broadcast<Real>(k6) * fma(y, output[63], x * output[49]);
}
/// Assign periodic Ewald correction coefficients and trace using compensated lattice sums.
template <int degree, typename Real, typename Output>
    requires(degree == 8)
CUDA_EXPORT void evaluateEwaldWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(+4.00000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(+1.00000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(+2.06185567010309274E-02);
    constexpr auto k3 = static_cast<Scalar>(+2.10526315789473679E-02);
    constexpr auto k4 = static_cast<Scalar>(+2.15053763440860232E-02);
    constexpr auto k5 = static_cast<Scalar>(+2.19780219780219797E-02);
    constexpr auto k6 = static_cast<Scalar>(+2.24719101123595499E-02);
    constexpr auto k7 = static_cast<Scalar>(+2.29885057471264365E-02);
    constexpr auto k8 = static_cast<Scalar>(+2.35294117647058820E-02);
    constexpr auto k9 = static_cast<Scalar>(+2.40963855421686760E-02);
    constexpr auto k10 = static_cast<Scalar>(+2.46913580246913567E-02);
    constexpr auto k11 = static_cast<Scalar>(+2.53164556962025313E-02);
    constexpr auto k12 = static_cast<Scalar>(+2.59740259740259757E-02);
    constexpr auto k13 = static_cast<Scalar>(+2.66666666666666684E-02);
    constexpr auto k14 = static_cast<Scalar>(+2.73972602739726012E-02);
    constexpr auto k15 = static_cast<Scalar>(+2.81690140845070429E-02);
    constexpr auto k16 = static_cast<Scalar>(+2.89855072463768119E-02);
    constexpr auto k17 = static_cast<Scalar>(+2.98507462686567158E-02);
    constexpr auto k18 = static_cast<Scalar>(+3.07692307692307709E-02);
    constexpr auto k19 = static_cast<Scalar>(+3.17460317460317443E-02);
    constexpr auto k20 = static_cast<Scalar>(+3.27868852459016411E-02);
    constexpr auto k21 = static_cast<Scalar>(+3.38983050847457626E-02);
    constexpr auto k22 = static_cast<Scalar>(+3.50877192982456121E-02);
    constexpr auto k23 = static_cast<Scalar>(+3.63636363636363619E-02);
    constexpr auto k24 = static_cast<Scalar>(+3.77358490566037721E-02);
    constexpr auto k25 = static_cast<Scalar>(+3.92156862745098034E-02);
    constexpr auto k26 = static_cast<Scalar>(+4.08163265306122416E-02);
    constexpr auto k27 = static_cast<Scalar>(+4.25531914893617011E-02);
    constexpr auto k28 = static_cast<Scalar>(+4.44444444444444461E-02);
    constexpr auto k29 = static_cast<Scalar>(+4.65116279069767435E-02);
    constexpr auto k30 = static_cast<Scalar>(+4.87804878048780505E-02);
    constexpr auto k31 = static_cast<Scalar>(+5.12820512820512803E-02);
    constexpr auto k32 = static_cast<Scalar>(+5.40540540540540571E-02);
    constexpr auto k33 = static_cast<Scalar>(+5.71428571428571411E-02);
    constexpr auto k34 = static_cast<Scalar>(+6.06060606060606077E-02);
    constexpr auto k35 = static_cast<Scalar>(+6.45161290322580627E-02);
    constexpr auto k36 = static_cast<Scalar>(+6.89655172413793094E-02);
    constexpr auto k37 = static_cast<Scalar>(+7.40740740740740700E-02);
    constexpr auto k38 = static_cast<Scalar>(+8.00000000000000017E-02);
    constexpr auto k39 = static_cast<Scalar>(+8.69565217391304324E-02);
    constexpr auto k40 = static_cast<Scalar>(+9.52380952380952328E-02);
    constexpr auto k41 = static_cast<Scalar>(+1.05263157894736836E-01);
    constexpr auto k42 = static_cast<Scalar>(+1.17647058823529410E-01);
    constexpr auto k43 = static_cast<Scalar>(+1.33333333333333331E-01);
    constexpr auto k44 = static_cast<Scalar>(+1.53846153846153855E-01);
    constexpr auto k45 = static_cast<Scalar>(+1.81818181818181823E-01);
    constexpr auto k46 = static_cast<Scalar>(+2.22222222222222210E-01);
    constexpr auto k47 = static_cast<Scalar>(+2.85714285714285698E-01);
    constexpr auto k48 = static_cast<Scalar>(+4.00000000000000022E-01);
    constexpr auto k49 = static_cast<Scalar>(+6.66666666666666630E-01);
    constexpr auto k50 = static_cast<Scalar>(+2.25675833419102512E+00);
    constexpr auto k51 = static_cast<Scalar>(+0.00000000000000000E+00);
    constexpr auto k52 = static_cast<Scalar>(+2.02020202020202037E-02);
    constexpr auto k53 = static_cast<Scalar>(+6.01802222450939972E+00);
    constexpr auto k54 = static_cast<Scalar>(+2.00000000000000000E+00);
    constexpr auto k55 = static_cast<Scalar>(+1.98019801980198022E-02);
    constexpr auto k56 = static_cast<Scalar>(+9.62883555921504275E+00);
    constexpr auto k57 = static_cast<Scalar>(+2.40000000000000000E+01);
    constexpr auto k58 = static_cast<Scalar>(+6.00000000000000000E+00);
    constexpr auto k59 = static_cast<Scalar>(+1.94174757281553381E-02);
    constexpr auto k60 = static_cast<Scalar>(+1.10043834962457634E+01);
    constexpr auto k61 = static_cast<Scalar>(+7.20000000000000000E+02);
    constexpr auto k62 = static_cast<Scalar>(+1.20000000000000000E+02);
    constexpr auto k63 = static_cast<Scalar>(+4.80000000000000000E+01);
    constexpr auto k64 = static_cast<Scalar>(+3.60000000000000000E+01);
    constexpr auto k65 = static_cast<Scalar>(+1.90476190476190493E-02);
    constexpr auto k66 = static_cast<Scalar>(+9.78167421888512045E+00);
    constexpr auto k67 = static_cast<Scalar>(+4.03200000000000000E+04);
    constexpr auto k68 = static_cast<Scalar>(+5.04000000000000000E+03);
    constexpr auto k69 = static_cast<Scalar>(+1.44000000000000000E+03);
    constexpr auto k70 = static_cast<Scalar>(+5.76000000000000000E+02);
    constexpr auto k71 = static_cast<Scalar>(+1.86915887850467276E-02);
    constexpr auto k72 = static_cast<Scalar>(+7.11394488646190482E+00);
    constexpr auto k73 = static_cast<Scalar>(+3.62880000000000000E+06);
    constexpr auto k74 = static_cast<Scalar>(+3.62880000000000000E+05);
    constexpr auto k75 = static_cast<Scalar>(+8.06400000000000000E+04);
    constexpr auto k76 = static_cast<Scalar>(+3.02400000000000000E+04);
    constexpr auto k77 = static_cast<Scalar>(+1.72800000000000000E+04);
    constexpr auto k78 = static_cast<Scalar>(+1.44000000000000000E+04);
    constexpr auto k79 = static_cast<Scalar>(+1.83486238532110102E-02);
    constexpr auto k80 = static_cast<Scalar>(+4.37781223782271134E+00);
    constexpr auto k81 = static_cast<Scalar>(+4.79001600000000000E+08);
    constexpr auto k82 = static_cast<Scalar>(+3.99168000000000000E+07);
    constexpr auto k83 = static_cast<Scalar>(+7.25760000000000000E+06);
    constexpr auto k84 = static_cast<Scalar>(+2.17728000000000000E+06);
    constexpr auto k85 = static_cast<Scalar>(+9.67680000000000000E+05);
    constexpr auto k86 = static_cast<Scalar>(+6.04800000000000000E+05);
    constexpr auto k87 = static_cast<Scalar>(+5.18400000000000000E+05);
    constexpr auto k88 = static_cast<Scalar>(+1.80180180180180179E-02);
    constexpr auto k89 = static_cast<Scalar>(+2.33483319350544605E+00);
    constexpr auto k90 = static_cast<Scalar>(+8.71782912000000000E+10);
    constexpr auto k91 = static_cast<Scalar>(+6.22702080000000000E+09);
    constexpr auto k92 = static_cast<Scalar>(+9.58003200000000000E+08);
    constexpr auto k93 = static_cast<Scalar>(+2.39500800000000000E+08);
    constexpr auto k94 = static_cast<Scalar>(+8.70912000000000000E+07);
    constexpr auto k95 = static_cast<Scalar>(+4.35456000000000000E+07);
    constexpr auto k96 = static_cast<Scalar>(+2.90304000000000000E+07);
    constexpr auto k97 = static_cast<Scalar>(+2.54016000000000000E+07);
    constexpr auto k98 = static_cast<Scalar>(+1.76991150442477874E-02);
    constexpr auto k99 = static_cast<Scalar>(+1.09874503223785691E+00);
    constexpr auto k100 = static_cast<Scalar>(+2.09227898880000000E+13);
    constexpr auto k101 = static_cast<Scalar>(+1.30767436800000000E+12);
    constexpr auto k102 = static_cast<Scalar>(+1.74356582400000000E+11);
    constexpr auto k103 = static_cast<Scalar>(+3.73621248000000000E+10);
    constexpr auto k104 = static_cast<Scalar>(+1.14960384000000000E+10);
    constexpr auto k105 = static_cast<Scalar>(+4.79001600000000000E+09);
    constexpr auto k106 = static_cast<Scalar>(+2.61273600000000000E+09);
    constexpr auto k107 = static_cast<Scalar>(+1.82891520000000000E+09);
    constexpr auto k108 = static_cast<Scalar>(+1.62570240000000000E+09);
    constexpr auto k109 = static_cast<Scalar>(+7.85398163397448279E-01);
    constexpr auto k110 = static_cast<Scalar>(+4.18879020478639053E+00);
    constexpr auto k111 = static_cast<Scalar>(+6.28318530717958623E+00);
    constexpr auto k112 = static_cast<Scalar>(+6.36619772367581382E-01);
    constexpr auto k113 = static_cast<Scalar>(+2.46740110027233950E+00);
    constexpr auto k114 = static_cast<Scalar>(+1.31594725347858112E+01);
    constexpr auto k115 = static_cast<Scalar>(+1.65366808961599006E+01);
    constexpr auto k116 = static_cast<Scalar>(+1.48432900623241775E+01);
    constexpr auto k117 = static_cast<Scalar>(+1.03625713366444501E+01);
    constexpr auto k118 = static_cast<Scalar>(+5.91908690609133981E+00);
    constexpr auto k119 = static_cast<Scalar>(+2.86082460617478285E+00);
    constexpr auto k120 = static_cast<Scalar>(+1.19833940879568135E+00);
    Real harmonics[81];
    Real compensation[81];
    Real imagePair[81];
    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular<degree>(harmonics, x, y, z, radiusSquared);
    Real const argument = broadcast<Real>(k0) * radiusSquared;
    Real const exponential = detail::ewaldExp(-argument);
    Real series0 = broadcast<Real>(k1);
    series0 = fma(series0, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    series0 = fma(series0, argument * broadcast<Real>(k49), broadcast<Real>(k1));
    Real const radial0 = broadcast<Real>(k50) * exponential * series0;
    output[0] = radial0 * harmonics[0];
    compensation[0] = broadcast<Real>(k51);
    Real series1 = broadcast<Real>(k1);
    series1 = fma(series1, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    series1 = fma(series1, argument * broadcast<Real>(k48), broadcast<Real>(k1));
    Real const radial1 = -broadcast<Real>(k53) * exponential * series1;
    output[1] = broadcast<Real>(k54) * radial1 * harmonics[1];
    compensation[1] = broadcast<Real>(k51);
    output[2] = radial1 * harmonics[2];
    compensation[2] = broadcast<Real>(k51);
    output[3] = broadcast<Real>(k54) * radial1 * harmonics[3];
    compensation[3] = broadcast<Real>(k51);
    Real series2 = broadcast<Real>(k1);
    series2 = fma(series2, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    series2 = fma(series2, argument * broadcast<Real>(k47), broadcast<Real>(k1));
    Real const radial2 = broadcast<Real>(k56) * exponential * series2;
    output[4] = broadcast<Real>(k57) * radial2 * harmonics[4];
    compensation[4] = broadcast<Real>(k51);
    output[5] = broadcast<Real>(k58) * radial2 * harmonics[5];
    compensation[5] = broadcast<Real>(k51);
    output[6] = broadcast<Real>(k0) * radial2 * harmonics[6];
    compensation[6] = broadcast<Real>(k51);
    output[7] = broadcast<Real>(k58) * radial2 * harmonics[7];
    compensation[7] = broadcast<Real>(k51);
    output[8] = broadcast<Real>(k57) * radial2 * harmonics[8];
    compensation[8] = broadcast<Real>(k51);
    Real series3 = broadcast<Real>(k1);
    series3 = fma(series3, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    series3 = fma(series3, argument * broadcast<Real>(k46), broadcast<Real>(k1));
    Real const radial3 = -broadcast<Real>(k60) * exponential * series3;
    output[9] = broadcast<Real>(k61) * radial3 * harmonics[9];
    compensation[9] = broadcast<Real>(k51);
    output[10] = broadcast<Real>(k62) * radial3 * harmonics[10];
    compensation[10] = broadcast<Real>(k51);
    output[11] = broadcast<Real>(k63) * radial3 * harmonics[11];
    compensation[11] = broadcast<Real>(k51);
    output[12] = broadcast<Real>(k64) * radial3 * harmonics[12];
    compensation[12] = broadcast<Real>(k51);
    output[13] = broadcast<Real>(k63) * radial3 * harmonics[13];
    compensation[13] = broadcast<Real>(k51);
    output[14] = broadcast<Real>(k62) * radial3 * harmonics[14];
    compensation[14] = broadcast<Real>(k51);
    output[15] = broadcast<Real>(k61) * radial3 * harmonics[15];
    compensation[15] = broadcast<Real>(k51);
    Real series4 = broadcast<Real>(k1);
    series4 = fma(series4, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    series4 = fma(series4, argument * broadcast<Real>(k45), broadcast<Real>(k1));
    Real const radial4 = broadcast<Real>(k66) * exponential * series4;
    output[16] = broadcast<Real>(k67) * radial4 * harmonics[16];
    compensation[16] = broadcast<Real>(k51);
    output[17] = broadcast<Real>(k68) * radial4 * harmonics[17];
    compensation[17] = broadcast<Real>(k51);
    output[18] = broadcast<Real>(k69) * radial4 * harmonics[18];
    compensation[18] = broadcast<Real>(k51);
    output[19] = broadcast<Real>(k61) * radial4 * harmonics[19];
    compensation[19] = broadcast<Real>(k51);
    output[20] = broadcast<Real>(k70) * radial4 * harmonics[20];
    compensation[20] = broadcast<Real>(k51);
    output[21] = broadcast<Real>(k61) * radial4 * harmonics[21];
    compensation[21] = broadcast<Real>(k51);
    output[22] = broadcast<Real>(k69) * radial4 * harmonics[22];
    compensation[22] = broadcast<Real>(k51);
    output[23] = broadcast<Real>(k68) * radial4 * harmonics[23];
    compensation[23] = broadcast<Real>(k51);
    output[24] = broadcast<Real>(k67) * radial4 * harmonics[24];
    compensation[24] = broadcast<Real>(k51);
    Real series5 = broadcast<Real>(k1);
    series5 = fma(series5, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    series5 = fma(series5, argument * broadcast<Real>(k44), broadcast<Real>(k1));
    Real const radial5 = -broadcast<Real>(k72) * exponential * series5;
    output[25] = broadcast<Real>(k73) * radial5 * harmonics[25];
    compensation[25] = broadcast<Real>(k51);
    output[26] = broadcast<Real>(k74) * radial5 * harmonics[26];
    compensation[26] = broadcast<Real>(k51);
    output[27] = broadcast<Real>(k75) * radial5 * harmonics[27];
    compensation[27] = broadcast<Real>(k51);
    output[28] = broadcast<Real>(k76) * radial5 * harmonics[28];
    compensation[28] = broadcast<Real>(k51);
    output[29] = broadcast<Real>(k77) * radial5 * harmonics[29];
    compensation[29] = broadcast<Real>(k51);
    output[30] = broadcast<Real>(k78) * radial5 * harmonics[30];
    compensation[30] = broadcast<Real>(k51);
    output[31] = broadcast<Real>(k77) * radial5 * harmonics[31];
    compensation[31] = broadcast<Real>(k51);
    output[32] = broadcast<Real>(k76) * radial5 * harmonics[32];
    compensation[32] = broadcast<Real>(k51);
    output[33] = broadcast<Real>(k75) * radial5 * harmonics[33];
    compensation[33] = broadcast<Real>(k51);
    output[34] = broadcast<Real>(k74) * radial5 * harmonics[34];
    compensation[34] = broadcast<Real>(k51);
    output[35] = broadcast<Real>(k73) * radial5 * harmonics[35];
    compensation[35] = broadcast<Real>(k51);
    Real series6 = broadcast<Real>(k1);
    series6 = fma(series6, argument * broadcast<Real>(k79), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    series6 = fma(series6, argument * broadcast<Real>(k43), broadcast<Real>(k1));
    Real const radial6 = broadcast<Real>(k80) * exponential * series6;
    output[36] = broadcast<Real>(k81) * radial6 * harmonics[36];
    compensation[36] = broadcast<Real>(k51);
    output[37] = broadcast<Real>(k82) * radial6 * harmonics[37];
    compensation[37] = broadcast<Real>(k51);
    output[38] = broadcast<Real>(k83) * radial6 * harmonics[38];
    compensation[38] = broadcast<Real>(k51);
    output[39] = broadcast<Real>(k84) * radial6 * harmonics[39];
    compensation[39] = broadcast<Real>(k51);
    output[40] = broadcast<Real>(k85) * radial6 * harmonics[40];
    compensation[40] = broadcast<Real>(k51);
    output[41] = broadcast<Real>(k86) * radial6 * harmonics[41];
    compensation[41] = broadcast<Real>(k51);
    output[42] = broadcast<Real>(k87) * radial6 * harmonics[42];
    compensation[42] = broadcast<Real>(k51);
    output[43] = broadcast<Real>(k86) * radial6 * harmonics[43];
    compensation[43] = broadcast<Real>(k51);
    output[44] = broadcast<Real>(k85) * radial6 * harmonics[44];
    compensation[44] = broadcast<Real>(k51);
    output[45] = broadcast<Real>(k84) * radial6 * harmonics[45];
    compensation[45] = broadcast<Real>(k51);
    output[46] = broadcast<Real>(k83) * radial6 * harmonics[46];
    compensation[46] = broadcast<Real>(k51);
    output[47] = broadcast<Real>(k82) * radial6 * harmonics[47];
    compensation[47] = broadcast<Real>(k51);
    output[48] = broadcast<Real>(k81) * radial6 * harmonics[48];
    compensation[48] = broadcast<Real>(k51);
    Real series7 = broadcast<Real>(k1);
    series7 = fma(series7, argument * broadcast<Real>(k88), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k79), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    series7 = fma(series7, argument * broadcast<Real>(k42), broadcast<Real>(k1));
    Real const radial7 = -broadcast<Real>(k89) * exponential * series7;
    output[49] = broadcast<Real>(k90) * radial7 * harmonics[49];
    compensation[49] = broadcast<Real>(k51);
    output[50] = broadcast<Real>(k91) * radial7 * harmonics[50];
    compensation[50] = broadcast<Real>(k51);
    output[51] = broadcast<Real>(k92) * radial7 * harmonics[51];
    compensation[51] = broadcast<Real>(k51);
    output[52] = broadcast<Real>(k93) * radial7 * harmonics[52];
    compensation[52] = broadcast<Real>(k51);
    output[53] = broadcast<Real>(k94) * radial7 * harmonics[53];
    compensation[53] = broadcast<Real>(k51);
    output[54] = broadcast<Real>(k95) * radial7 * harmonics[54];
    compensation[54] = broadcast<Real>(k51);
    output[55] = broadcast<Real>(k96) * radial7 * harmonics[55];
    compensation[55] = broadcast<Real>(k51);
    output[56] = broadcast<Real>(k97) * radial7 * harmonics[56];
    compensation[56] = broadcast<Real>(k51);
    output[57] = broadcast<Real>(k96) * radial7 * harmonics[57];
    compensation[57] = broadcast<Real>(k51);
    output[58] = broadcast<Real>(k95) * radial7 * harmonics[58];
    compensation[58] = broadcast<Real>(k51);
    output[59] = broadcast<Real>(k94) * radial7 * harmonics[59];
    compensation[59] = broadcast<Real>(k51);
    output[60] = broadcast<Real>(k93) * radial7 * harmonics[60];
    compensation[60] = broadcast<Real>(k51);
    output[61] = broadcast<Real>(k92) * radial7 * harmonics[61];
    compensation[61] = broadcast<Real>(k51);
    output[62] = broadcast<Real>(k91) * radial7 * harmonics[62];
    compensation[62] = broadcast<Real>(k51);
    output[63] = broadcast<Real>(k90) * radial7 * harmonics[63];
    compensation[63] = broadcast<Real>(k51);
    Real series8 = broadcast<Real>(k1);
    series8 = fma(series8, argument * broadcast<Real>(k98), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k88), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k79), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k71), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k65), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k59), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k55), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k52), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k2), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k3), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k4), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k5), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k6), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k7), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k8), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k9), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k10), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k11), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k12), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k13), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k14), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k15), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k16), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k17), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k18), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k19), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k20), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k21), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k22), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k23), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k24), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k25), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k26), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k27), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k28), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k29), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k30), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k31), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k32), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k33), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k34), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k35), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k36), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k37), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k38), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k39), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k40), broadcast<Real>(k1));
    series8 = fma(series8, argument * broadcast<Real>(k41), broadcast<Real>(k1));
    Real const radial8 = broadcast<Real>(k99) * exponential * series8;
    output[64] = broadcast<Real>(k100) * radial8 * harmonics[64];
    compensation[64] = broadcast<Real>(k51);
    output[65] = broadcast<Real>(k101) * radial8 * harmonics[65];
    compensation[65] = broadcast<Real>(k51);
    output[66] = broadcast<Real>(k102) * radial8 * harmonics[66];
    compensation[66] = broadcast<Real>(k51);
    output[67] = broadcast<Real>(k103) * radial8 * harmonics[67];
    compensation[67] = broadcast<Real>(k51);
    output[68] = broadcast<Real>(k104) * radial8 * harmonics[68];
    compensation[68] = broadcast<Real>(k51);
    output[69] = broadcast<Real>(k105) * radial8 * harmonics[69];
    compensation[69] = broadcast<Real>(k51);
    output[70] = broadcast<Real>(k106) * radial8 * harmonics[70];
    compensation[70] = broadcast<Real>(k51);
    output[71] = broadcast<Real>(k107) * radial8 * harmonics[71];
    compensation[71] = broadcast<Real>(k51);
    output[72] = broadcast<Real>(k108) * radial8 * harmonics[72];
    compensation[72] = broadcast<Real>(k51);
    output[73] = broadcast<Real>(k107) * radial8 * harmonics[73];
    compensation[73] = broadcast<Real>(k51);
    output[74] = broadcast<Real>(k106) * radial8 * harmonics[74];
    compensation[74] = broadcast<Real>(k51);
    output[75] = broadcast<Real>(k105) * radial8 * harmonics[75];
    compensation[75] = broadcast<Real>(k51);
    output[76] = broadcast<Real>(k104) * radial8 * harmonics[76];
    compensation[76] = broadcast<Real>(k51);
    output[77] = broadcast<Real>(k103) * radial8 * harmonics[77];
    compensation[77] = broadcast<Real>(k51);
    output[78] = broadcast<Real>(k102) * radial8 * harmonics[78];
    compensation[78] = broadcast<Real>(k51);
    output[79] = broadcast<Real>(k101) * radial8 * harmonics[79];
    compensation[79] = broadcast<Real>(k51);
    output[80] = broadcast<Real>(k100) * radial8 * harmonics[80];
    compensation[80] = broadcast<Real>(k51);
    output[0] += broadcast<Real>(k109);
    greenTrace = broadcast<Real>(k110);
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
                    Real const iR = broadcast<Real>(k1) / imageRadius;
                    Real const inverseSquared = iR * iR;
                    ewaldRegular<degree>(harmonics, imageX, imageY, imageZ, imageSquared);
                    Real const imageArgument = broadcast<Real>(k0) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k54) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k50) * imageRadius * imageExp;
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
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
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
                    gammaTerm *= imageArgument * broadcast<Real>(k48);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k57) * imageRadial2 * harmonics[4];
                    if (sign < 0) {
                        imagePair[4] = imageValue4;
                    } else {
                        imagePair[4] += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k58) * imageRadial2 * harmonics[5];
                    if (sign < 0) {
                        imagePair[5] = imageValue5;
                    } else {
                        imagePair[5] += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k0) * imageRadial2 * harmonics[6];
                    if (sign < 0) {
                        imagePair[6] = imageValue6;
                    } else {
                        imagePair[6] += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k58) * imageRadial2 * harmonics[7];
                    if (sign < 0) {
                        imagePair[7] = imageValue7;
                    } else {
                        imagePair[7] += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k57) * imageRadial2 * harmonics[8];
                    if (sign < 0) {
                        imagePair[8] = imageValue8;
                    } else {
                        imagePair[8] += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k47);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k61) * imageRadial3 * harmonics[9];
                    if (sign < 0) {
                        imagePair[9] = imageValue9;
                    } else {
                        imagePair[9] += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k62) * imageRadial3 * harmonics[10];
                    if (sign < 0) {
                        imagePair[10] = imageValue10;
                    } else {
                        imagePair[10] += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k63) * imageRadial3 * harmonics[11];
                    if (sign < 0) {
                        imagePair[11] = imageValue11;
                    } else {
                        imagePair[11] += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k64) * imageRadial3 * harmonics[12];
                    if (sign < 0) {
                        imagePair[12] = imageValue12;
                    } else {
                        imagePair[12] += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k63) * imageRadial3 * harmonics[13];
                    if (sign < 0) {
                        imagePair[13] = imageValue13;
                    } else {
                        imagePair[13] += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k62) * imageRadial3 * harmonics[14];
                    if (sign < 0) {
                        imagePair[14] = imageValue14;
                    } else {
                        imagePair[14] += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k61) * imageRadial3 * harmonics[15];
                    if (sign < 0) {
                        imagePair[15] = imageValue15;
                    } else {
                        imagePair[15] += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k46);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k67) * imageRadial4 * harmonics[16];
                    if (sign < 0) {
                        imagePair[16] = imageValue16;
                    } else {
                        imagePair[16] += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k68) * imageRadial4 * harmonics[17];
                    if (sign < 0) {
                        imagePair[17] = imageValue17;
                    } else {
                        imagePair[17] += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k69) * imageRadial4 * harmonics[18];
                    if (sign < 0) {
                        imagePair[18] = imageValue18;
                    } else {
                        imagePair[18] += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k61) * imageRadial4 * harmonics[19];
                    if (sign < 0) {
                        imagePair[19] = imageValue19;
                    } else {
                        imagePair[19] += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k70) * imageRadial4 * harmonics[20];
                    if (sign < 0) {
                        imagePair[20] = imageValue20;
                    } else {
                        imagePair[20] += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k61) * imageRadial4 * harmonics[21];
                    if (sign < 0) {
                        imagePair[21] = imageValue21;
                    } else {
                        imagePair[21] += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k69) * imageRadial4 * harmonics[22];
                    if (sign < 0) {
                        imagePair[22] = imageValue22;
                    } else {
                        imagePair[22] += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k68) * imageRadial4 * harmonics[23];
                    if (sign < 0) {
                        imagePair[23] = imageValue23;
                    } else {
                        imagePair[23] += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k67) * imageRadial4 * harmonics[24];
                    if (sign < 0) {
                        imagePair[24] = imageValue24;
                    } else {
                        imagePair[24] += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k45);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k73) * imageRadial5 * harmonics[25];
                    if (sign < 0) {
                        imagePair[25] = imageValue25;
                    } else {
                        imagePair[25] += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k74) * imageRadial5 * harmonics[26];
                    if (sign < 0) {
                        imagePair[26] = imageValue26;
                    } else {
                        imagePair[26] += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k75) * imageRadial5 * harmonics[27];
                    if (sign < 0) {
                        imagePair[27] = imageValue27;
                    } else {
                        imagePair[27] += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k76) * imageRadial5 * harmonics[28];
                    if (sign < 0) {
                        imagePair[28] = imageValue28;
                    } else {
                        imagePair[28] += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k77) * imageRadial5 * harmonics[29];
                    if (sign < 0) {
                        imagePair[29] = imageValue29;
                    } else {
                        imagePair[29] += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k78) * imageRadial5 * harmonics[30];
                    if (sign < 0) {
                        imagePair[30] = imageValue30;
                    } else {
                        imagePair[30] += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k77) * imageRadial5 * harmonics[31];
                    if (sign < 0) {
                        imagePair[31] = imageValue31;
                    } else {
                        imagePair[31] += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k76) * imageRadial5 * harmonics[32];
                    if (sign < 0) {
                        imagePair[32] = imageValue32;
                    } else {
                        imagePair[32] += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k75) * imageRadial5 * harmonics[33];
                    if (sign < 0) {
                        imagePair[33] = imageValue33;
                    } else {
                        imagePair[33] += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k74) * imageRadial5 * harmonics[34];
                    if (sign < 0) {
                        imagePair[34] = imageValue34;
                    } else {
                        imagePair[34] += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k73) * imageRadial5 * harmonics[35];
                    if (sign < 0) {
                        imagePair[35] = imageValue35;
                    } else {
                        imagePair[35] += imageValue35;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k44);
                    Real const imageRadial6 = -screening * inversePower;
                    Real const imageValue36 = broadcast<Real>(k81) * imageRadial6 * harmonics[36];
                    if (sign < 0) {
                        imagePair[36] = imageValue36;
                    } else {
                        imagePair[36] += imageValue36;
                    }
                    Real const imageValue37 = broadcast<Real>(k82) * imageRadial6 * harmonics[37];
                    if (sign < 0) {
                        imagePair[37] = imageValue37;
                    } else {
                        imagePair[37] += imageValue37;
                    }
                    Real const imageValue38 = broadcast<Real>(k83) * imageRadial6 * harmonics[38];
                    if (sign < 0) {
                        imagePair[38] = imageValue38;
                    } else {
                        imagePair[38] += imageValue38;
                    }
                    Real const imageValue39 = broadcast<Real>(k84) * imageRadial6 * harmonics[39];
                    if (sign < 0) {
                        imagePair[39] = imageValue39;
                    } else {
                        imagePair[39] += imageValue39;
                    }
                    Real const imageValue40 = broadcast<Real>(k85) * imageRadial6 * harmonics[40];
                    if (sign < 0) {
                        imagePair[40] = imageValue40;
                    } else {
                        imagePair[40] += imageValue40;
                    }
                    Real const imageValue41 = broadcast<Real>(k86) * imageRadial6 * harmonics[41];
                    if (sign < 0) {
                        imagePair[41] = imageValue41;
                    } else {
                        imagePair[41] += imageValue41;
                    }
                    Real const imageValue42 = broadcast<Real>(k87) * imageRadial6 * harmonics[42];
                    if (sign < 0) {
                        imagePair[42] = imageValue42;
                    } else {
                        imagePair[42] += imageValue42;
                    }
                    Real const imageValue43 = broadcast<Real>(k86) * imageRadial6 * harmonics[43];
                    if (sign < 0) {
                        imagePair[43] = imageValue43;
                    } else {
                        imagePair[43] += imageValue43;
                    }
                    Real const imageValue44 = broadcast<Real>(k85) * imageRadial6 * harmonics[44];
                    if (sign < 0) {
                        imagePair[44] = imageValue44;
                    } else {
                        imagePair[44] += imageValue44;
                    }
                    Real const imageValue45 = broadcast<Real>(k84) * imageRadial6 * harmonics[45];
                    if (sign < 0) {
                        imagePair[45] = imageValue45;
                    } else {
                        imagePair[45] += imageValue45;
                    }
                    Real const imageValue46 = broadcast<Real>(k83) * imageRadial6 * harmonics[46];
                    if (sign < 0) {
                        imagePair[46] = imageValue46;
                    } else {
                        imagePair[46] += imageValue46;
                    }
                    Real const imageValue47 = broadcast<Real>(k82) * imageRadial6 * harmonics[47];
                    if (sign < 0) {
                        imagePair[47] = imageValue47;
                    } else {
                        imagePair[47] += imageValue47;
                    }
                    Real const imageValue48 = broadcast<Real>(k81) * imageRadial6 * harmonics[48];
                    if (sign < 0) {
                        imagePair[48] = imageValue48;
                    } else {
                        imagePair[48] += imageValue48;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k43);
                    Real const imageRadial7 = screening * inversePower;
                    Real const imageValue49 = broadcast<Real>(k90) * imageRadial7 * harmonics[49];
                    if (sign < 0) {
                        imagePair[49] = imageValue49;
                    } else {
                        imagePair[49] += imageValue49;
                    }
                    Real const imageValue50 = broadcast<Real>(k91) * imageRadial7 * harmonics[50];
                    if (sign < 0) {
                        imagePair[50] = imageValue50;
                    } else {
                        imagePair[50] += imageValue50;
                    }
                    Real const imageValue51 = broadcast<Real>(k92) * imageRadial7 * harmonics[51];
                    if (sign < 0) {
                        imagePair[51] = imageValue51;
                    } else {
                        imagePair[51] += imageValue51;
                    }
                    Real const imageValue52 = broadcast<Real>(k93) * imageRadial7 * harmonics[52];
                    if (sign < 0) {
                        imagePair[52] = imageValue52;
                    } else {
                        imagePair[52] += imageValue52;
                    }
                    Real const imageValue53 = broadcast<Real>(k94) * imageRadial7 * harmonics[53];
                    if (sign < 0) {
                        imagePair[53] = imageValue53;
                    } else {
                        imagePair[53] += imageValue53;
                    }
                    Real const imageValue54 = broadcast<Real>(k95) * imageRadial7 * harmonics[54];
                    if (sign < 0) {
                        imagePair[54] = imageValue54;
                    } else {
                        imagePair[54] += imageValue54;
                    }
                    Real const imageValue55 = broadcast<Real>(k96) * imageRadial7 * harmonics[55];
                    if (sign < 0) {
                        imagePair[55] = imageValue55;
                    } else {
                        imagePair[55] += imageValue55;
                    }
                    Real const imageValue56 = broadcast<Real>(k97) * imageRadial7 * harmonics[56];
                    if (sign < 0) {
                        imagePair[56] = imageValue56;
                    } else {
                        imagePair[56] += imageValue56;
                    }
                    Real const imageValue57 = broadcast<Real>(k96) * imageRadial7 * harmonics[57];
                    if (sign < 0) {
                        imagePair[57] = imageValue57;
                    } else {
                        imagePair[57] += imageValue57;
                    }
                    Real const imageValue58 = broadcast<Real>(k95) * imageRadial7 * harmonics[58];
                    if (sign < 0) {
                        imagePair[58] = imageValue58;
                    } else {
                        imagePair[58] += imageValue58;
                    }
                    Real const imageValue59 = broadcast<Real>(k94) * imageRadial7 * harmonics[59];
                    if (sign < 0) {
                        imagePair[59] = imageValue59;
                    } else {
                        imagePair[59] += imageValue59;
                    }
                    Real const imageValue60 = broadcast<Real>(k93) * imageRadial7 * harmonics[60];
                    if (sign < 0) {
                        imagePair[60] = imageValue60;
                    } else {
                        imagePair[60] += imageValue60;
                    }
                    Real const imageValue61 = broadcast<Real>(k92) * imageRadial7 * harmonics[61];
                    if (sign < 0) {
                        imagePair[61] = imageValue61;
                    } else {
                        imagePair[61] += imageValue61;
                    }
                    Real const imageValue62 = broadcast<Real>(k91) * imageRadial7 * harmonics[62];
                    if (sign < 0) {
                        imagePair[62] = imageValue62;
                    } else {
                        imagePair[62] += imageValue62;
                    }
                    Real const imageValue63 = broadcast<Real>(k90) * imageRadial7 * harmonics[63];
                    if (sign < 0) {
                        imagePair[63] = imageValue63;
                    } else {
                        imagePair[63] += imageValue63;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k42);
                    Real const imageRadial8 = -screening * inversePower;
                    Real const imageValue64 = broadcast<Real>(k100) * imageRadial8 * harmonics[64];
                    if (sign < 0) {
                        imagePair[64] = imageValue64;
                    } else {
                        imagePair[64] += imageValue64;
                    }
                    Real const imageValue65 = broadcast<Real>(k101) * imageRadial8 * harmonics[65];
                    if (sign < 0) {
                        imagePair[65] = imageValue65;
                    } else {
                        imagePair[65] += imageValue65;
                    }
                    Real const imageValue66 = broadcast<Real>(k102) * imageRadial8 * harmonics[66];
                    if (sign < 0) {
                        imagePair[66] = imageValue66;
                    } else {
                        imagePair[66] += imageValue66;
                    }
                    Real const imageValue67 = broadcast<Real>(k103) * imageRadial8 * harmonics[67];
                    if (sign < 0) {
                        imagePair[67] = imageValue67;
                    } else {
                        imagePair[67] += imageValue67;
                    }
                    Real const imageValue68 = broadcast<Real>(k104) * imageRadial8 * harmonics[68];
                    if (sign < 0) {
                        imagePair[68] = imageValue68;
                    } else {
                        imagePair[68] += imageValue68;
                    }
                    Real const imageValue69 = broadcast<Real>(k105) * imageRadial8 * harmonics[69];
                    if (sign < 0) {
                        imagePair[69] = imageValue69;
                    } else {
                        imagePair[69] += imageValue69;
                    }
                    Real const imageValue70 = broadcast<Real>(k106) * imageRadial8 * harmonics[70];
                    if (sign < 0) {
                        imagePair[70] = imageValue70;
                    } else {
                        imagePair[70] += imageValue70;
                    }
                    Real const imageValue71 = broadcast<Real>(k107) * imageRadial8 * harmonics[71];
                    if (sign < 0) {
                        imagePair[71] = imageValue71;
                    } else {
                        imagePair[71] += imageValue71;
                    }
                    Real const imageValue72 = broadcast<Real>(k108) * imageRadial8 * harmonics[72];
                    if (sign < 0) {
                        imagePair[72] = imageValue72;
                    } else {
                        imagePair[72] += imageValue72;
                    }
                    Real const imageValue73 = broadcast<Real>(k107) * imageRadial8 * harmonics[73];
                    if (sign < 0) {
                        imagePair[73] = imageValue73;
                    } else {
                        imagePair[73] += imageValue73;
                    }
                    Real const imageValue74 = broadcast<Real>(k106) * imageRadial8 * harmonics[74];
                    if (sign < 0) {
                        imagePair[74] = imageValue74;
                    } else {
                        imagePair[74] += imageValue74;
                    }
                    Real const imageValue75 = broadcast<Real>(k105) * imageRadial8 * harmonics[75];
                    if (sign < 0) {
                        imagePair[75] = imageValue75;
                    } else {
                        imagePair[75] += imageValue75;
                    }
                    Real const imageValue76 = broadcast<Real>(k104) * imageRadial8 * harmonics[76];
                    if (sign < 0) {
                        imagePair[76] = imageValue76;
                    } else {
                        imagePair[76] += imageValue76;
                    }
                    Real const imageValue77 = broadcast<Real>(k103) * imageRadial8 * harmonics[77];
                    if (sign < 0) {
                        imagePair[77] = imageValue77;
                    } else {
                        imagePair[77] += imageValue77;
                    }
                    Real const imageValue78 = broadcast<Real>(k102) * imageRadial8 * harmonics[78];
                    if (sign < 0) {
                        imagePair[78] = imageValue78;
                    } else {
                        imagePair[78] += imageValue78;
                    }
                    Real const imageValue79 = broadcast<Real>(k101) * imageRadial8 * harmonics[79];
                    if (sign < 0) {
                        imagePair[79] = imageValue79;
                    } else {
                        imagePair[79] += imageValue79;
                    }
                    Real const imageValue80 = broadcast<Real>(k100) * imageRadial8 * harmonics[80];
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
                Real const phase = broadcast<Real>(k111) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k112) *
                    detail::ewaldExp(-broadcast<Real>(k113) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k1) * wavePotential * cosine;
                detail::ewaldAccumulate(output[0], compensation[0], waveRadial0 * harmonics[0]);
                Real const waveRadial1 = broadcast<Real>(k111) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    output[1], compensation[1], broadcast<Real>(k54) * waveRadial1 * harmonics[1]);
                detail::ewaldAccumulate(output[2], compensation[2], waveRadial1 * harmonics[2]);
                detail::ewaldAccumulate(
                    output[3], compensation[3], broadcast<Real>(k54) * waveRadial1 * harmonics[3]);
                Real const waveRadial2 = broadcast<Real>(k114) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    output[4], compensation[4], broadcast<Real>(k57) * waveRadial2 * harmonics[4]);
                detail::ewaldAccumulate(
                    output[5], compensation[5], broadcast<Real>(k58) * waveRadial2 * harmonics[5]);
                detail::ewaldAccumulate(
                    output[6], compensation[6], broadcast<Real>(k0) * waveRadial2 * harmonics[6]);
                detail::ewaldAccumulate(
                    output[7], compensation[7], broadcast<Real>(k58) * waveRadial2 * harmonics[7]);
                detail::ewaldAccumulate(
                    output[8], compensation[8], broadcast<Real>(k57) * waveRadial2 * harmonics[8]);
                Real const waveRadial3 = broadcast<Real>(k115) * wavePotential * sine;
                detail::ewaldAccumulate(
                    output[9], compensation[9], broadcast<Real>(k61) * waveRadial3 * harmonics[9]);
                detail::ewaldAccumulate(output[10], compensation[10],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[10]);
                detail::ewaldAccumulate(output[11], compensation[11],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[11]);
                detail::ewaldAccumulate(output[12], compensation[12],
                    broadcast<Real>(k64) * waveRadial3 * harmonics[12]);
                detail::ewaldAccumulate(output[13], compensation[13],
                    broadcast<Real>(k63) * waveRadial3 * harmonics[13]);
                detail::ewaldAccumulate(output[14], compensation[14],
                    broadcast<Real>(k62) * waveRadial3 * harmonics[14]);
                detail::ewaldAccumulate(output[15], compensation[15],
                    broadcast<Real>(k61) * waveRadial3 * harmonics[15]);
                Real const waveRadial4 = broadcast<Real>(k116) * wavePotential * cosine;
                detail::ewaldAccumulate(output[16], compensation[16],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[16]);
                detail::ewaldAccumulate(output[17], compensation[17],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[17]);
                detail::ewaldAccumulate(output[18], compensation[18],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[18]);
                detail::ewaldAccumulate(output[19], compensation[19],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[19]);
                detail::ewaldAccumulate(output[20], compensation[20],
                    broadcast<Real>(k70) * waveRadial4 * harmonics[20]);
                detail::ewaldAccumulate(output[21], compensation[21],
                    broadcast<Real>(k61) * waveRadial4 * harmonics[21]);
                detail::ewaldAccumulate(output[22], compensation[22],
                    broadcast<Real>(k69) * waveRadial4 * harmonics[22]);
                detail::ewaldAccumulate(output[23], compensation[23],
                    broadcast<Real>(k68) * waveRadial4 * harmonics[23]);
                detail::ewaldAccumulate(output[24], compensation[24],
                    broadcast<Real>(k67) * waveRadial4 * harmonics[24]);
                Real const waveRadial5 = broadcast<Real>(k117) * wavePotential * -sine;
                detail::ewaldAccumulate(output[25], compensation[25],
                    broadcast<Real>(k73) * waveRadial5 * harmonics[25]);
                detail::ewaldAccumulate(output[26], compensation[26],
                    broadcast<Real>(k74) * waveRadial5 * harmonics[26]);
                detail::ewaldAccumulate(output[27], compensation[27],
                    broadcast<Real>(k75) * waveRadial5 * harmonics[27]);
                detail::ewaldAccumulate(output[28], compensation[28],
                    broadcast<Real>(k76) * waveRadial5 * harmonics[28]);
                detail::ewaldAccumulate(output[29], compensation[29],
                    broadcast<Real>(k77) * waveRadial5 * harmonics[29]);
                detail::ewaldAccumulate(output[30], compensation[30],
                    broadcast<Real>(k78) * waveRadial5 * harmonics[30]);
                detail::ewaldAccumulate(output[31], compensation[31],
                    broadcast<Real>(k77) * waveRadial5 * harmonics[31]);
                detail::ewaldAccumulate(output[32], compensation[32],
                    broadcast<Real>(k76) * waveRadial5 * harmonics[32]);
                detail::ewaldAccumulate(output[33], compensation[33],
                    broadcast<Real>(k75) * waveRadial5 * harmonics[33]);
                detail::ewaldAccumulate(output[34], compensation[34],
                    broadcast<Real>(k74) * waveRadial5 * harmonics[34]);
                detail::ewaldAccumulate(output[35], compensation[35],
                    broadcast<Real>(k73) * waveRadial5 * harmonics[35]);
                Real const waveRadial6 = broadcast<Real>(k118) * wavePotential * -cosine;
                detail::ewaldAccumulate(output[36], compensation[36],
                    broadcast<Real>(k81) * waveRadial6 * harmonics[36]);
                detail::ewaldAccumulate(output[37], compensation[37],
                    broadcast<Real>(k82) * waveRadial6 * harmonics[37]);
                detail::ewaldAccumulate(output[38], compensation[38],
                    broadcast<Real>(k83) * waveRadial6 * harmonics[38]);
                detail::ewaldAccumulate(output[39], compensation[39],
                    broadcast<Real>(k84) * waveRadial6 * harmonics[39]);
                detail::ewaldAccumulate(output[40], compensation[40],
                    broadcast<Real>(k85) * waveRadial6 * harmonics[40]);
                detail::ewaldAccumulate(output[41], compensation[41],
                    broadcast<Real>(k86) * waveRadial6 * harmonics[41]);
                detail::ewaldAccumulate(output[42], compensation[42],
                    broadcast<Real>(k87) * waveRadial6 * harmonics[42]);
                detail::ewaldAccumulate(output[43], compensation[43],
                    broadcast<Real>(k86) * waveRadial6 * harmonics[43]);
                detail::ewaldAccumulate(output[44], compensation[44],
                    broadcast<Real>(k85) * waveRadial6 * harmonics[44]);
                detail::ewaldAccumulate(output[45], compensation[45],
                    broadcast<Real>(k84) * waveRadial6 * harmonics[45]);
                detail::ewaldAccumulate(output[46], compensation[46],
                    broadcast<Real>(k83) * waveRadial6 * harmonics[46]);
                detail::ewaldAccumulate(output[47], compensation[47],
                    broadcast<Real>(k82) * waveRadial6 * harmonics[47]);
                detail::ewaldAccumulate(output[48], compensation[48],
                    broadcast<Real>(k81) * waveRadial6 * harmonics[48]);
                Real const waveRadial7 = broadcast<Real>(k119) * wavePotential * sine;
                detail::ewaldAccumulate(output[49], compensation[49],
                    broadcast<Real>(k90) * waveRadial7 * harmonics[49]);
                detail::ewaldAccumulate(output[50], compensation[50],
                    broadcast<Real>(k91) * waveRadial7 * harmonics[50]);
                detail::ewaldAccumulate(output[51], compensation[51],
                    broadcast<Real>(k92) * waveRadial7 * harmonics[51]);
                detail::ewaldAccumulate(output[52], compensation[52],
                    broadcast<Real>(k93) * waveRadial7 * harmonics[52]);
                detail::ewaldAccumulate(output[53], compensation[53],
                    broadcast<Real>(k94) * waveRadial7 * harmonics[53]);
                detail::ewaldAccumulate(output[54], compensation[54],
                    broadcast<Real>(k95) * waveRadial7 * harmonics[54]);
                detail::ewaldAccumulate(output[55], compensation[55],
                    broadcast<Real>(k96) * waveRadial7 * harmonics[55]);
                detail::ewaldAccumulate(output[56], compensation[56],
                    broadcast<Real>(k97) * waveRadial7 * harmonics[56]);
                detail::ewaldAccumulate(output[57], compensation[57],
                    broadcast<Real>(k96) * waveRadial7 * harmonics[57]);
                detail::ewaldAccumulate(output[58], compensation[58],
                    broadcast<Real>(k95) * waveRadial7 * harmonics[58]);
                detail::ewaldAccumulate(output[59], compensation[59],
                    broadcast<Real>(k94) * waveRadial7 * harmonics[59]);
                detail::ewaldAccumulate(output[60], compensation[60],
                    broadcast<Real>(k93) * waveRadial7 * harmonics[60]);
                detail::ewaldAccumulate(output[61], compensation[61],
                    broadcast<Real>(k92) * waveRadial7 * harmonics[61]);
                detail::ewaldAccumulate(output[62], compensation[62],
                    broadcast<Real>(k91) * waveRadial7 * harmonics[62]);
                detail::ewaldAccumulate(output[63], compensation[63],
                    broadcast<Real>(k90) * waveRadial7 * harmonics[63]);
                Real const waveRadial8 = broadcast<Real>(k120) * wavePotential * cosine;
                detail::ewaldAccumulate(output[64], compensation[64],
                    broadcast<Real>(k100) * waveRadial8 * harmonics[64]);
                detail::ewaldAccumulate(output[65], compensation[65],
                    broadcast<Real>(k101) * waveRadial8 * harmonics[65]);
                detail::ewaldAccumulate(output[66], compensation[66],
                    broadcast<Real>(k102) * waveRadial8 * harmonics[66]);
                detail::ewaldAccumulate(output[67], compensation[67],
                    broadcast<Real>(k103) * waveRadial8 * harmonics[67]);
                detail::ewaldAccumulate(output[68], compensation[68],
                    broadcast<Real>(k104) * waveRadial8 * harmonics[68]);
                detail::ewaldAccumulate(output[69], compensation[69],
                    broadcast<Real>(k105) * waveRadial8 * harmonics[69]);
                detail::ewaldAccumulate(output[70], compensation[70],
                    broadcast<Real>(k106) * waveRadial8 * harmonics[70]);
                detail::ewaldAccumulate(output[71], compensation[71],
                    broadcast<Real>(k107) * waveRadial8 * harmonics[71]);
                detail::ewaldAccumulate(output[72], compensation[72],
                    broadcast<Real>(k108) * waveRadial8 * harmonics[72]);
                detail::ewaldAccumulate(output[73], compensation[73],
                    broadcast<Real>(k107) * waveRadial8 * harmonics[73]);
                detail::ewaldAccumulate(output[74], compensation[74],
                    broadcast<Real>(k106) * waveRadial8 * harmonics[74]);
                detail::ewaldAccumulate(output[75], compensation[75],
                    broadcast<Real>(k105) * waveRadial8 * harmonics[75]);
                detail::ewaldAccumulate(output[76], compensation[76],
                    broadcast<Real>(k104) * waveRadial8 * harmonics[76]);
                detail::ewaldAccumulate(output[77], compensation[77],
                    broadcast<Real>(k103) * waveRadial8 * harmonics[77]);
                detail::ewaldAccumulate(output[78], compensation[78],
                    broadcast<Real>(k102) * waveRadial8 * harmonics[78]);
                detail::ewaldAccumulate(output[79], compensation[79],
                    broadcast<Real>(k101) * waveRadial8 * harmonics[79]);
                detail::ewaldAccumulate(output[80], compensation[80],
                    broadcast<Real>(k100) * waveRadial8 * harmonics[80]);
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
