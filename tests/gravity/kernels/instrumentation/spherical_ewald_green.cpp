/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

// Generated counting-scalar definitions. Tests only; not a library source.
#include <cosmictiger/gravity/kernels/spherical/ewald_green.hpp>
namespace cosmictiger::gravity::spherical {
template <typename Real>
void ewaldRegular(Real x, Real y, Real z, Real radiusSquared, Real* output) {
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
    detail::CoefficientView<Real> const U{output};

    U(0, 0) = broadcast<Real>(k17);
    U(1, 0) = z * U(0, 0);
    Real const zFactor2_0 = broadcast<Real>(k18) * z;
    U(2, 0) = broadcast<Real>(k15) * fma(zFactor2_0, U(1, 0), -radiusSquared * U(0, 0));
    Real const zFactor3_0 = broadcast<Real>(k19) * z;
    U(3, 0) = broadcast<Real>(k12) * fma(zFactor3_0, U(2, 0), -radiusSquared * U(1, 0));
    Real const zFactor4_0 = broadcast<Real>(k20) * z;
    U(4, 0) = broadcast<Real>(k08) * fma(zFactor4_0, U(3, 0), -radiusSquared * U(2, 0));
    Real const zFactor5_0 = broadcast<Real>(k21) * z;
    U(5, 0) = broadcast<Real>(k04) * fma(zFactor5_0, U(4, 0), -radiusSquared * U(3, 0));
    Real const zFactor6_0 = broadcast<Real>(k22) * z;
    U(6, 0) = broadcast<Real>(k00) * fma(zFactor6_0, U(5, 0), -radiusSquared * U(4, 0));
    U(1, 1) = x * broadcast<Real>(k16);
    U(1, -1) = y * broadcast<Real>(k16);
    U(2, 1) = z * U(1, 1);
    U(2, -1) = z * U(1, -1);
    Real const zFactor3_1 = broadcast<Real>(k19) * z;
    U(3, 1) = broadcast<Real>(k13) * fma(zFactor3_1, U(2, 1), -radiusSquared * U(1, 1));
    U(3, -1) = broadcast<Real>(k13) * fma(zFactor3_1, U(2, -1), -radiusSquared * U(1, -1));
    Real const zFactor4_1 = broadcast<Real>(k20) * z;
    U(4, 1) = broadcast<Real>(k09) * fma(zFactor4_1, U(3, 1), -radiusSquared * U(2, 1));
    U(4, -1) = broadcast<Real>(k09) * fma(zFactor4_1, U(3, -1), -radiusSquared * U(2, -1));
    Real const zFactor5_1 = broadcast<Real>(k21) * z;
    U(5, 1) = broadcast<Real>(k05) * fma(zFactor5_1, U(4, 1), -radiusSquared * U(3, 1));
    U(5, -1) = broadcast<Real>(k05) * fma(zFactor5_1, U(4, -1), -radiusSquared * U(3, -1));
    Real const zFactor6_1 = broadcast<Real>(k22) * z;
    U(6, 1) = broadcast<Real>(k01) * fma(zFactor6_1, U(5, 1), -radiusSquared * U(4, 1));
    U(6, -1) = broadcast<Real>(k01) * fma(zFactor6_1, U(5, -1), -radiusSquared * U(4, -1));
    U(2, 2) = broadcast<Real>(k15) * fma(x, U(1, 1), -y * U(1, -1));
    U(2, -2) = broadcast<Real>(k15) * fma(y, U(1, 1), x * U(1, -1));
    U(3, 2) = z * U(2, 2);
    U(3, -2) = z * U(2, -2);
    Real const zFactor4_2 = broadcast<Real>(k20) * z;
    U(4, 2) = broadcast<Real>(k10) * fma(zFactor4_2, U(3, 2), -radiusSquared * U(2, 2));
    U(4, -2) = broadcast<Real>(k10) * fma(zFactor4_2, U(3, -2), -radiusSquared * U(2, -2));
    Real const zFactor5_2 = broadcast<Real>(k21) * z;
    U(5, 2) = broadcast<Real>(k06) * fma(zFactor5_2, U(4, 2), -radiusSquared * U(3, 2));
    U(5, -2) = broadcast<Real>(k06) * fma(zFactor5_2, U(4, -2), -radiusSquared * U(3, -2));
    Real const zFactor6_2 = broadcast<Real>(k22) * z;
    U(6, 2) = broadcast<Real>(k02) * fma(zFactor6_2, U(5, 2), -radiusSquared * U(4, 2));
    U(6, -2) = broadcast<Real>(k02) * fma(zFactor6_2, U(5, -2), -radiusSquared * U(4, -2));
    U(3, 3) = broadcast<Real>(k14) * fma(x, U(2, 2), -y * U(2, -2));
    U(3, -3) = broadcast<Real>(k14) * fma(y, U(2, 2), x * U(2, -2));
    U(4, 3) = z * U(3, 3);
    U(4, -3) = z * U(3, -3);
    Real const zFactor5_3 = broadcast<Real>(k21) * z;
    U(5, 3) = broadcast<Real>(k08) * fma(zFactor5_3, U(4, 3), -radiusSquared * U(3, 3));
    U(5, -3) = broadcast<Real>(k08) * fma(zFactor5_3, U(4, -3), -radiusSquared * U(3, -3));
    Real const zFactor6_3 = broadcast<Real>(k22) * z;
    U(6, 3) = broadcast<Real>(k03) * fma(zFactor6_3, U(5, 3), -radiusSquared * U(4, 3));
    U(6, -3) = broadcast<Real>(k03) * fma(zFactor6_3, U(5, -3), -radiusSquared * U(4, -3));
    U(4, 4) = broadcast<Real>(k13) * fma(x, U(3, 3), -y * U(3, -3));
    U(4, -4) = broadcast<Real>(k13) * fma(y, U(3, 3), x * U(3, -3));
    U(5, 4) = z * U(4, 4);
    U(5, -4) = z * U(4, -4);
    Real const zFactor6_4 = broadcast<Real>(k22) * z;
    U(6, 4) = broadcast<Real>(k07) * fma(zFactor6_4, U(5, 4), -radiusSquared * U(4, 4));
    U(6, -4) = broadcast<Real>(k07) * fma(zFactor6_4, U(5, -4), -radiusSquared * U(4, -4));
    U(5, 5) = broadcast<Real>(k11) * fma(x, U(4, 4), -y * U(4, -4));
    U(5, -5) = broadcast<Real>(k11) * fma(y, U(4, 4), x * U(4, -4));
    U(6, 5) = z * U(5, 5);
    U(6, -5) = z * U(5, -5);
    U(6, 6) = broadcast<Real>(k10) * fma(x, U(5, 5), -y * U(5, -5));
    U(6, -6) = broadcast<Real>(k10) * fma(y, U(5, 5), x * U(5, -5));
}
}    // namespace cosmictiger::gravity::spherical
namespace cosmictiger::gravity::spherical {
template <typename Real>
void evaluateEwaldWorking(Real& greenTrace, Real x, Real y, Real z, Real* output) {
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
    detail::CoefficientView<Real> const G{output};

    Real harmonics[49];
    detail::CoefficientView<Real> const U{harmonics};

    Real compensation[49];
    detail::CoefficientView<Real> const C{compensation};

    Real imagePair[49];
    detail::CoefficientView<Real> const I{imagePair};

    Real const radiusSquared = fma(x, x, fma(y, y, z * z));
    ewaldRegular(x, y, z, radiusSquared, harmonics);
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
    G(0, 0) = radial0 * U(0, 0);
    C(0, 0) = broadcast<Real>(k00);
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
    G(1, -1) = broadcast<Real>(k58) * radial1 * U(1, -1);
    C(1, -1) = broadcast<Real>(k00);
    G(1, 0) = radial1 * U(1, 0);
    C(1, 0) = broadcast<Real>(k00);
    G(1, 1) = broadcast<Real>(k58) * radial1 * U(1, 1);
    C(1, 1) = broadcast<Real>(k00);
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
    G(2, -2) = broadcast<Real>(k76) * radial2 * U(2, -2);
    C(2, -2) = broadcast<Real>(k00);
    G(2, -1) = broadcast<Real>(k65) * radial2 * U(2, -1);
    C(2, -1) = broadcast<Real>(k00);
    G(2, 0) = broadcast<Real>(k61) * radial2 * U(2, 0);
    C(2, 0) = broadcast<Real>(k00);
    G(2, 1) = broadcast<Real>(k65) * radial2 * U(2, 1);
    C(2, 1) = broadcast<Real>(k00);
    G(2, 2) = broadcast<Real>(k76) * radial2 * U(2, 2);
    C(2, 2) = broadcast<Real>(k00);
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
    G(3, -3) = broadcast<Real>(k81) * radial3 * U(3, -3);
    C(3, -3) = broadcast<Real>(k00);
    G(3, -2) = broadcast<Real>(k79) * radial3 * U(3, -2);
    C(3, -2) = broadcast<Real>(k00);
    G(3, -1) = broadcast<Real>(k78) * radial3 * U(3, -1);
    C(3, -1) = broadcast<Real>(k00);
    G(3, 0) = broadcast<Real>(k77) * radial3 * U(3, 0);
    C(3, 0) = broadcast<Real>(k00);
    G(3, 1) = broadcast<Real>(k78) * radial3 * U(3, 1);
    C(3, 1) = broadcast<Real>(k00);
    G(3, 2) = broadcast<Real>(k79) * radial3 * U(3, 2);
    C(3, 2) = broadcast<Real>(k00);
    G(3, 3) = broadcast<Real>(k81) * radial3 * U(3, 3);
    C(3, 3) = broadcast<Real>(k00);
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
    G(4, -4) = broadcast<Real>(k87) * radial4 * U(4, -4);
    C(4, -4) = broadcast<Real>(k00);
    G(4, -3) = broadcast<Real>(k83) * radial4 * U(4, -3);
    C(4, -3) = broadcast<Real>(k00);
    G(4, -2) = broadcast<Real>(k82) * radial4 * U(4, -2);
    C(4, -2) = broadcast<Real>(k00);
    G(4, -1) = broadcast<Real>(k81) * radial4 * U(4, -1);
    C(4, -1) = broadcast<Real>(k00);
    G(4, 0) = broadcast<Real>(k80) * radial4 * U(4, 0);
    C(4, 0) = broadcast<Real>(k00);
    G(4, 1) = broadcast<Real>(k81) * radial4 * U(4, 1);
    C(4, 1) = broadcast<Real>(k00);
    G(4, 2) = broadcast<Real>(k82) * radial4 * U(4, 2);
    C(4, 2) = broadcast<Real>(k00);
    G(4, 3) = broadcast<Real>(k83) * radial4 * U(4, 3);
    C(4, 3) = broadcast<Real>(k00);
    G(4, 4) = broadcast<Real>(k87) * radial4 * U(4, 4);
    C(4, 4) = broadcast<Real>(k00);
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
    G(5, -5) = broadcast<Real>(k94) * radial5 * U(5, -5);
    C(5, -5) = broadcast<Real>(k00);
    G(5, -4) = broadcast<Real>(k89) * radial5 * U(5, -4);
    C(5, -4) = broadcast<Real>(k00);
    G(5, -3) = broadcast<Real>(k88) * radial5 * U(5, -3);
    C(5, -3) = broadcast<Real>(k00);
    G(5, -2) = broadcast<Real>(k86) * radial5 * U(5, -2);
    C(5, -2) = broadcast<Real>(k00);
    G(5, -1) = broadcast<Real>(k85) * radial5 * U(5, -1);
    C(5, -1) = broadcast<Real>(k00);
    G(5, 0) = broadcast<Real>(k84) * radial5 * U(5, 0);
    C(5, 0) = broadcast<Real>(k00);
    G(5, 1) = broadcast<Real>(k85) * radial5 * U(5, 1);
    C(5, 1) = broadcast<Real>(k00);
    G(5, 2) = broadcast<Real>(k86) * radial5 * U(5, 2);
    C(5, 2) = broadcast<Real>(k00);
    G(5, 3) = broadcast<Real>(k88) * radial5 * U(5, 3);
    C(5, 3) = broadcast<Real>(k00);
    G(5, 4) = broadcast<Real>(k89) * radial5 * U(5, 4);
    C(5, 4) = broadcast<Real>(k00);
    G(5, 5) = broadcast<Real>(k94) * radial5 * U(5, 5);
    C(5, 5) = broadcast<Real>(k00);
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
    G(6, -6) = broadcast<Real>(k97) * radial6 * U(6, -6);
    C(6, -6) = broadcast<Real>(k00);
    G(6, -5) = broadcast<Real>(k96) * radial6 * U(6, -5);
    C(6, -5) = broadcast<Real>(k00);
    G(6, -4) = broadcast<Real>(k95) * radial6 * U(6, -4);
    C(6, -4) = broadcast<Real>(k00);
    G(6, -3) = broadcast<Real>(k93) * radial6 * U(6, -3);
    C(6, -3) = broadcast<Real>(k00);
    G(6, -2) = broadcast<Real>(k92) * radial6 * U(6, -2);
    C(6, -2) = broadcast<Real>(k00);
    G(6, -1) = broadcast<Real>(k91) * radial6 * U(6, -1);
    C(6, -1) = broadcast<Real>(k00);
    G(6, 0) = broadcast<Real>(k90) * radial6 * U(6, 0);
    C(6, 0) = broadcast<Real>(k00);
    G(6, 1) = broadcast<Real>(k91) * radial6 * U(6, 1);
    C(6, 1) = broadcast<Real>(k00);
    G(6, 2) = broadcast<Real>(k92) * radial6 * U(6, 2);
    C(6, 2) = broadcast<Real>(k00);
    G(6, 3) = broadcast<Real>(k93) * radial6 * U(6, 3);
    C(6, 3) = broadcast<Real>(k00);
    G(6, 4) = broadcast<Real>(k95) * radial6 * U(6, 4);
    C(6, 4) = broadcast<Real>(k00);
    G(6, 5) = broadcast<Real>(k96) * radial6 * U(6, 5);
    C(6, 5) = broadcast<Real>(k00);
    G(6, 6) = broadcast<Real>(k97) * radial6 * U(6, 6);
    C(6, 6) = broadcast<Real>(k00);
    G(0, 0) += broadcast<Real>(k56);
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
                    ewaldRegular(imageX, imageY, imageZ, imageSquared, harmonics);
                    Real const imageArgument = broadcast<Real>(k61) * imageSquared;
                    Real const imageExp = detail::ewaldExp(-imageArgument);
                    Real screening = detail::ewaldErfc(broadcast<Real>(k58) * imageRadius);
                    Real gammaTerm = broadcast<Real>(k59) * imageRadius * imageExp;
                    Real inversePower = iR;
                    Real const imageRadial0 = -screening * inversePower;
                    Real const imageValue0 = imageRadial0 * U(0, 0);
                    if (sign < 0) {
                        I(0, 0) = imageValue0;
                    } else {
                        I(0, 0) += imageValue0;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k55);
                    Real const imageRadial1 = screening * inversePower;
                    Real const imageValue1 = broadcast<Real>(k58) * imageRadial1 * U(1, -1);
                    if (sign < 0) {
                        I(1, -1) = imageValue1;
                    } else {
                        I(1, -1) += imageValue1;
                    }
                    Real const imageValue2 = imageRadial1 * U(1, 0);
                    if (sign < 0) {
                        I(1, 0) = imageValue2;
                    } else {
                        I(1, 0) += imageValue2;
                    }
                    Real const imageValue3 = broadcast<Real>(k58) * imageRadial1 * U(1, 1);
                    if (sign < 0) {
                        I(1, 1) = imageValue3;
                    } else {
                        I(1, 1) += imageValue3;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k53);
                    Real const imageRadial2 = -screening * inversePower;
                    Real const imageValue4 = broadcast<Real>(k76) * imageRadial2 * U(2, -2);
                    if (sign < 0) {
                        I(2, -2) = imageValue4;
                    } else {
                        I(2, -2) += imageValue4;
                    }
                    Real const imageValue5 = broadcast<Real>(k65) * imageRadial2 * U(2, -1);
                    if (sign < 0) {
                        I(2, -1) = imageValue5;
                    } else {
                        I(2, -1) += imageValue5;
                    }
                    Real const imageValue6 = broadcast<Real>(k61) * imageRadial2 * U(2, 0);
                    if (sign < 0) {
                        I(2, 0) = imageValue6;
                    } else {
                        I(2, 0) += imageValue6;
                    }
                    Real const imageValue7 = broadcast<Real>(k65) * imageRadial2 * U(2, 1);
                    if (sign < 0) {
                        I(2, 1) = imageValue7;
                    } else {
                        I(2, 1) += imageValue7;
                    }
                    Real const imageValue8 = broadcast<Real>(k76) * imageRadial2 * U(2, 2);
                    if (sign < 0) {
                        I(2, 2) = imageValue8;
                    } else {
                        I(2, 2) += imageValue8;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k52);
                    Real const imageRadial3 = screening * inversePower;
                    Real const imageValue9 = broadcast<Real>(k81) * imageRadial3 * U(3, -3);
                    if (sign < 0) {
                        I(3, -3) = imageValue9;
                    } else {
                        I(3, -3) += imageValue9;
                    }
                    Real const imageValue10 = broadcast<Real>(k79) * imageRadial3 * U(3, -2);
                    if (sign < 0) {
                        I(3, -2) = imageValue10;
                    } else {
                        I(3, -2) += imageValue10;
                    }
                    Real const imageValue11 = broadcast<Real>(k78) * imageRadial3 * U(3, -1);
                    if (sign < 0) {
                        I(3, -1) = imageValue11;
                    } else {
                        I(3, -1) += imageValue11;
                    }
                    Real const imageValue12 = broadcast<Real>(k77) * imageRadial3 * U(3, 0);
                    if (sign < 0) {
                        I(3, 0) = imageValue12;
                    } else {
                        I(3, 0) += imageValue12;
                    }
                    Real const imageValue13 = broadcast<Real>(k78) * imageRadial3 * U(3, 1);
                    if (sign < 0) {
                        I(3, 1) = imageValue13;
                    } else {
                        I(3, 1) += imageValue13;
                    }
                    Real const imageValue14 = broadcast<Real>(k79) * imageRadial3 * U(3, 2);
                    if (sign < 0) {
                        I(3, 2) = imageValue14;
                    } else {
                        I(3, 2) += imageValue14;
                    }
                    Real const imageValue15 = broadcast<Real>(k81) * imageRadial3 * U(3, 3);
                    if (sign < 0) {
                        I(3, 3) = imageValue15;
                    } else {
                        I(3, 3) += imageValue15;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k51);
                    Real const imageRadial4 = -screening * inversePower;
                    Real const imageValue16 = broadcast<Real>(k87) * imageRadial4 * U(4, -4);
                    if (sign < 0) {
                        I(4, -4) = imageValue16;
                    } else {
                        I(4, -4) += imageValue16;
                    }
                    Real const imageValue17 = broadcast<Real>(k83) * imageRadial4 * U(4, -3);
                    if (sign < 0) {
                        I(4, -3) = imageValue17;
                    } else {
                        I(4, -3) += imageValue17;
                    }
                    Real const imageValue18 = broadcast<Real>(k82) * imageRadial4 * U(4, -2);
                    if (sign < 0) {
                        I(4, -2) = imageValue18;
                    } else {
                        I(4, -2) += imageValue18;
                    }
                    Real const imageValue19 = broadcast<Real>(k81) * imageRadial4 * U(4, -1);
                    if (sign < 0) {
                        I(4, -1) = imageValue19;
                    } else {
                        I(4, -1) += imageValue19;
                    }
                    Real const imageValue20 = broadcast<Real>(k80) * imageRadial4 * U(4, 0);
                    if (sign < 0) {
                        I(4, 0) = imageValue20;
                    } else {
                        I(4, 0) += imageValue20;
                    }
                    Real const imageValue21 = broadcast<Real>(k81) * imageRadial4 * U(4, 1);
                    if (sign < 0) {
                        I(4, 1) = imageValue21;
                    } else {
                        I(4, 1) += imageValue21;
                    }
                    Real const imageValue22 = broadcast<Real>(k82) * imageRadial4 * U(4, 2);
                    if (sign < 0) {
                        I(4, 2) = imageValue22;
                    } else {
                        I(4, 2) += imageValue22;
                    }
                    Real const imageValue23 = broadcast<Real>(k83) * imageRadial4 * U(4, 3);
                    if (sign < 0) {
                        I(4, 3) = imageValue23;
                    } else {
                        I(4, 3) += imageValue23;
                    }
                    Real const imageValue24 = broadcast<Real>(k87) * imageRadial4 * U(4, 4);
                    if (sign < 0) {
                        I(4, 4) = imageValue24;
                    } else {
                        I(4, 4) += imageValue24;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k50);
                    Real const imageRadial5 = screening * inversePower;
                    Real const imageValue25 = broadcast<Real>(k94) * imageRadial5 * U(5, -5);
                    if (sign < 0) {
                        I(5, -5) = imageValue25;
                    } else {
                        I(5, -5) += imageValue25;
                    }
                    Real const imageValue26 = broadcast<Real>(k89) * imageRadial5 * U(5, -4);
                    if (sign < 0) {
                        I(5, -4) = imageValue26;
                    } else {
                        I(5, -4) += imageValue26;
                    }
                    Real const imageValue27 = broadcast<Real>(k88) * imageRadial5 * U(5, -3);
                    if (sign < 0) {
                        I(5, -3) = imageValue27;
                    } else {
                        I(5, -3) += imageValue27;
                    }
                    Real const imageValue28 = broadcast<Real>(k86) * imageRadial5 * U(5, -2);
                    if (sign < 0) {
                        I(5, -2) = imageValue28;
                    } else {
                        I(5, -2) += imageValue28;
                    }
                    Real const imageValue29 = broadcast<Real>(k85) * imageRadial5 * U(5, -1);
                    if (sign < 0) {
                        I(5, -1) = imageValue29;
                    } else {
                        I(5, -1) += imageValue29;
                    }
                    Real const imageValue30 = broadcast<Real>(k84) * imageRadial5 * U(5, 0);
                    if (sign < 0) {
                        I(5, 0) = imageValue30;
                    } else {
                        I(5, 0) += imageValue30;
                    }
                    Real const imageValue31 = broadcast<Real>(k85) * imageRadial5 * U(5, 1);
                    if (sign < 0) {
                        I(5, 1) = imageValue31;
                    } else {
                        I(5, 1) += imageValue31;
                    }
                    Real const imageValue32 = broadcast<Real>(k86) * imageRadial5 * U(5, 2);
                    if (sign < 0) {
                        I(5, 2) = imageValue32;
                    } else {
                        I(5, 2) += imageValue32;
                    }
                    Real const imageValue33 = broadcast<Real>(k88) * imageRadial5 * U(5, 3);
                    if (sign < 0) {
                        I(5, 3) = imageValue33;
                    } else {
                        I(5, 3) += imageValue33;
                    }
                    Real const imageValue34 = broadcast<Real>(k89) * imageRadial5 * U(5, 4);
                    if (sign < 0) {
                        I(5, 4) = imageValue34;
                    } else {
                        I(5, 4) += imageValue34;
                    }
                    Real const imageValue35 = broadcast<Real>(k94) * imageRadial5 * U(5, 5);
                    if (sign < 0) {
                        I(5, 5) = imageValue35;
                    } else {
                        I(5, 5) += imageValue35;
                    }
                    inversePower *= inverseSquared;
                    screening += gammaTerm;
                    gammaTerm *= imageArgument * broadcast<Real>(k49);
                    Real const imageRadial6 = -screening * inversePower;
                    Real const imageValue36 = broadcast<Real>(k97) * imageRadial6 * U(6, -6);
                    if (sign < 0) {
                        I(6, -6) = imageValue36;
                    } else {
                        I(6, -6) += imageValue36;
                    }
                    Real const imageValue37 = broadcast<Real>(k96) * imageRadial6 * U(6, -5);
                    if (sign < 0) {
                        I(6, -5) = imageValue37;
                    } else {
                        I(6, -5) += imageValue37;
                    }
                    Real const imageValue38 = broadcast<Real>(k95) * imageRadial6 * U(6, -4);
                    if (sign < 0) {
                        I(6, -4) = imageValue38;
                    } else {
                        I(6, -4) += imageValue38;
                    }
                    Real const imageValue39 = broadcast<Real>(k93) * imageRadial6 * U(6, -3);
                    if (sign < 0) {
                        I(6, -3) = imageValue39;
                    } else {
                        I(6, -3) += imageValue39;
                    }
                    Real const imageValue40 = broadcast<Real>(k92) * imageRadial6 * U(6, -2);
                    if (sign < 0) {
                        I(6, -2) = imageValue40;
                    } else {
                        I(6, -2) += imageValue40;
                    }
                    Real const imageValue41 = broadcast<Real>(k91) * imageRadial6 * U(6, -1);
                    if (sign < 0) {
                        I(6, -1) = imageValue41;
                    } else {
                        I(6, -1) += imageValue41;
                    }
                    Real const imageValue42 = broadcast<Real>(k90) * imageRadial6 * U(6, 0);
                    if (sign < 0) {
                        I(6, 0) = imageValue42;
                    } else {
                        I(6, 0) += imageValue42;
                    }
                    Real const imageValue43 = broadcast<Real>(k91) * imageRadial6 * U(6, 1);
                    if (sign < 0) {
                        I(6, 1) = imageValue43;
                    } else {
                        I(6, 1) += imageValue43;
                    }
                    Real const imageValue44 = broadcast<Real>(k92) * imageRadial6 * U(6, 2);
                    if (sign < 0) {
                        I(6, 2) = imageValue44;
                    } else {
                        I(6, 2) += imageValue44;
                    }
                    Real const imageValue45 = broadcast<Real>(k93) * imageRadial6 * U(6, 3);
                    if (sign < 0) {
                        I(6, 3) = imageValue45;
                    } else {
                        I(6, 3) += imageValue45;
                    }
                    Real const imageValue46 = broadcast<Real>(k95) * imageRadial6 * U(6, 4);
                    if (sign < 0) {
                        I(6, 4) = imageValue46;
                    } else {
                        I(6, 4) += imageValue46;
                    }
                    Real const imageValue47 = broadcast<Real>(k96) * imageRadial6 * U(6, 5);
                    if (sign < 0) {
                        I(6, 5) = imageValue47;
                    } else {
                        I(6, 5) += imageValue47;
                    }
                    Real const imageValue48 = broadcast<Real>(k97) * imageRadial6 * U(6, 6);
                    if (sign < 0) {
                        I(6, 6) = imageValue48;
                    } else {
                        I(6, 6) += imageValue48;
                    }
                }
                detail::ewaldAccumulate(G(0, 0), C(0, 0), I(0, 0));
                detail::ewaldAccumulate(G(1, -1), C(1, -1), I(1, -1));
                detail::ewaldAccumulate(G(1, 0), C(1, 0), I(1, 0));
                detail::ewaldAccumulate(G(1, 1), C(1, 1), I(1, 1));
                detail::ewaldAccumulate(G(2, -2), C(2, -2), I(2, -2));
                detail::ewaldAccumulate(G(2, -1), C(2, -1), I(2, -1));
                detail::ewaldAccumulate(G(2, 0), C(2, 0), I(2, 0));
                detail::ewaldAccumulate(G(2, 1), C(2, 1), I(2, 1));
                detail::ewaldAccumulate(G(2, 2), C(2, 2), I(2, 2));
                detail::ewaldAccumulate(G(3, -3), C(3, -3), I(3, -3));
                detail::ewaldAccumulate(G(3, -2), C(3, -2), I(3, -2));
                detail::ewaldAccumulate(G(3, -1), C(3, -1), I(3, -1));
                detail::ewaldAccumulate(G(3, 0), C(3, 0), I(3, 0));
                detail::ewaldAccumulate(G(3, 1), C(3, 1), I(3, 1));
                detail::ewaldAccumulate(G(3, 2), C(3, 2), I(3, 2));
                detail::ewaldAccumulate(G(3, 3), C(3, 3), I(3, 3));
                detail::ewaldAccumulate(G(4, -4), C(4, -4), I(4, -4));
                detail::ewaldAccumulate(G(4, -3), C(4, -3), I(4, -3));
                detail::ewaldAccumulate(G(4, -2), C(4, -2), I(4, -2));
                detail::ewaldAccumulate(G(4, -1), C(4, -1), I(4, -1));
                detail::ewaldAccumulate(G(4, 0), C(4, 0), I(4, 0));
                detail::ewaldAccumulate(G(4, 1), C(4, 1), I(4, 1));
                detail::ewaldAccumulate(G(4, 2), C(4, 2), I(4, 2));
                detail::ewaldAccumulate(G(4, 3), C(4, 3), I(4, 3));
                detail::ewaldAccumulate(G(4, 4), C(4, 4), I(4, 4));
                detail::ewaldAccumulate(G(5, -5), C(5, -5), I(5, -5));
                detail::ewaldAccumulate(G(5, -4), C(5, -4), I(5, -4));
                detail::ewaldAccumulate(G(5, -3), C(5, -3), I(5, -3));
                detail::ewaldAccumulate(G(5, -2), C(5, -2), I(5, -2));
                detail::ewaldAccumulate(G(5, -1), C(5, -1), I(5, -1));
                detail::ewaldAccumulate(G(5, 0), C(5, 0), I(5, 0));
                detail::ewaldAccumulate(G(5, 1), C(5, 1), I(5, 1));
                detail::ewaldAccumulate(G(5, 2), C(5, 2), I(5, 2));
                detail::ewaldAccumulate(G(5, 3), C(5, 3), I(5, 3));
                detail::ewaldAccumulate(G(5, 4), C(5, 4), I(5, 4));
                detail::ewaldAccumulate(G(5, 5), C(5, 5), I(5, 5));
                detail::ewaldAccumulate(G(6, -6), C(6, -6), I(6, -6));
                detail::ewaldAccumulate(G(6, -5), C(6, -5), I(6, -5));
                detail::ewaldAccumulate(G(6, -4), C(6, -4), I(6, -4));
                detail::ewaldAccumulate(G(6, -3), C(6, -3), I(6, -3));
                detail::ewaldAccumulate(G(6, -2), C(6, -2), I(6, -2));
                detail::ewaldAccumulate(G(6, -1), C(6, -1), I(6, -1));
                detail::ewaldAccumulate(G(6, 0), C(6, 0), I(6, 0));
                detail::ewaldAccumulate(G(6, 1), C(6, 1), I(6, 1));
                detail::ewaldAccumulate(G(6, 2), C(6, 2), I(6, 2));
                detail::ewaldAccumulate(G(6, 3), C(6, 3), I(6, 3));
                detail::ewaldAccumulate(G(6, 4), C(6, 4), I(6, 4));
                detail::ewaldAccumulate(G(6, 5), C(6, 5), I(6, 5));
                detail::ewaldAccumulate(G(6, 6), C(6, 6), I(6, 6));
                Real const waveX = broadcast<Real>(ix);
                Real const waveY = broadcast<Real>(iy);
                Real const waveZ = broadcast<Real>(iz);
                Real const waveSquared = broadcast<Real>(latticeSquared);
                ewaldRegular(waveX, waveY, waveZ, waveSquared, harmonics);
                Real const phase = broadcast<Real>(k67) * fma(waveX, x, fma(waveY, y, waveZ * z));
                Real const cosine = detail::ewaldCos(phase);
                Real const sine = detail::ewaldSin(phase);
                Real const wavePotential = -broadcast<Real>(k54) *
                    detail::ewaldExp(-broadcast<Real>(k60) * waveSquared) / waveSquared;
                Real const waveRadial0 = broadcast<Real>(k57) * wavePotential * cosine;
                detail::ewaldAccumulate(G(0, 0), C(0, 0), waveRadial0 * U(0, 0));
                Real const waveRadial1 = broadcast<Real>(k67) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    G(1, -1), C(1, -1), broadcast<Real>(k58) * waveRadial1 * U(1, -1));
                detail::ewaldAccumulate(G(1, 0), C(1, 0), waveRadial1 * U(1, 0));
                detail::ewaldAccumulate(
                    G(1, 1), C(1, 1), broadcast<Real>(k58) * waveRadial1 * U(1, 1));
                Real const waveRadial2 = broadcast<Real>(k73) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    G(2, -2), C(2, -2), broadcast<Real>(k76) * waveRadial2 * U(2, -2));
                detail::ewaldAccumulate(
                    G(2, -1), C(2, -1), broadcast<Real>(k65) * waveRadial2 * U(2, -1));
                detail::ewaldAccumulate(
                    G(2, 0), C(2, 0), broadcast<Real>(k61) * waveRadial2 * U(2, 0));
                detail::ewaldAccumulate(
                    G(2, 1), C(2, 1), broadcast<Real>(k65) * waveRadial2 * U(2, 1));
                detail::ewaldAccumulate(
                    G(2, 2), C(2, 2), broadcast<Real>(k76) * waveRadial2 * U(2, 2));
                Real const waveRadial3 = broadcast<Real>(k75) * wavePotential * sine;
                detail::ewaldAccumulate(
                    G(3, -3), C(3, -3), broadcast<Real>(k81) * waveRadial3 * U(3, -3));
                detail::ewaldAccumulate(
                    G(3, -2), C(3, -2), broadcast<Real>(k79) * waveRadial3 * U(3, -2));
                detail::ewaldAccumulate(
                    G(3, -1), C(3, -1), broadcast<Real>(k78) * waveRadial3 * U(3, -1));
                detail::ewaldAccumulate(
                    G(3, 0), C(3, 0), broadcast<Real>(k77) * waveRadial3 * U(3, 0));
                detail::ewaldAccumulate(
                    G(3, 1), C(3, 1), broadcast<Real>(k78) * waveRadial3 * U(3, 1));
                detail::ewaldAccumulate(
                    G(3, 2), C(3, 2), broadcast<Real>(k79) * waveRadial3 * U(3, 2));
                detail::ewaldAccumulate(
                    G(3, 3), C(3, 3), broadcast<Real>(k81) * waveRadial3 * U(3, 3));
                Real const waveRadial4 = broadcast<Real>(k74) * wavePotential * cosine;
                detail::ewaldAccumulate(
                    G(4, -4), C(4, -4), broadcast<Real>(k87) * waveRadial4 * U(4, -4));
                detail::ewaldAccumulate(
                    G(4, -3), C(4, -3), broadcast<Real>(k83) * waveRadial4 * U(4, -3));
                detail::ewaldAccumulate(
                    G(4, -2), C(4, -2), broadcast<Real>(k82) * waveRadial4 * U(4, -2));
                detail::ewaldAccumulate(
                    G(4, -1), C(4, -1), broadcast<Real>(k81) * waveRadial4 * U(4, -1));
                detail::ewaldAccumulate(
                    G(4, 0), C(4, 0), broadcast<Real>(k80) * waveRadial4 * U(4, 0));
                detail::ewaldAccumulate(
                    G(4, 1), C(4, 1), broadcast<Real>(k81) * waveRadial4 * U(4, 1));
                detail::ewaldAccumulate(
                    G(4, 2), C(4, 2), broadcast<Real>(k82) * waveRadial4 * U(4, 2));
                detail::ewaldAccumulate(
                    G(4, 3), C(4, 3), broadcast<Real>(k83) * waveRadial4 * U(4, 3));
                detail::ewaldAccumulate(
                    G(4, 4), C(4, 4), broadcast<Real>(k87) * waveRadial4 * U(4, 4));
                Real const waveRadial5 = broadcast<Real>(k71) * wavePotential * -sine;
                detail::ewaldAccumulate(
                    G(5, -5), C(5, -5), broadcast<Real>(k94) * waveRadial5 * U(5, -5));
                detail::ewaldAccumulate(
                    G(5, -4), C(5, -4), broadcast<Real>(k89) * waveRadial5 * U(5, -4));
                detail::ewaldAccumulate(
                    G(5, -3), C(5, -3), broadcast<Real>(k88) * waveRadial5 * U(5, -3));
                detail::ewaldAccumulate(
                    G(5, -2), C(5, -2), broadcast<Real>(k86) * waveRadial5 * U(5, -2));
                detail::ewaldAccumulate(
                    G(5, -1), C(5, -1), broadcast<Real>(k85) * waveRadial5 * U(5, -1));
                detail::ewaldAccumulate(
                    G(5, 0), C(5, 0), broadcast<Real>(k84) * waveRadial5 * U(5, 0));
                detail::ewaldAccumulate(
                    G(5, 1), C(5, 1), broadcast<Real>(k85) * waveRadial5 * U(5, 1));
                detail::ewaldAccumulate(
                    G(5, 2), C(5, 2), broadcast<Real>(k86) * waveRadial5 * U(5, 2));
                detail::ewaldAccumulate(
                    G(5, 3), C(5, 3), broadcast<Real>(k88) * waveRadial5 * U(5, 3));
                detail::ewaldAccumulate(
                    G(5, 4), C(5, 4), broadcast<Real>(k89) * waveRadial5 * U(5, 4));
                detail::ewaldAccumulate(
                    G(5, 5), C(5, 5), broadcast<Real>(k94) * waveRadial5 * U(5, 5));
                Real const waveRadial6 = broadcast<Real>(k64) * wavePotential * -cosine;
                detail::ewaldAccumulate(
                    G(6, -6), C(6, -6), broadcast<Real>(k97) * waveRadial6 * U(6, -6));
                detail::ewaldAccumulate(
                    G(6, -5), C(6, -5), broadcast<Real>(k96) * waveRadial6 * U(6, -5));
                detail::ewaldAccumulate(
                    G(6, -4), C(6, -4), broadcast<Real>(k95) * waveRadial6 * U(6, -4));
                detail::ewaldAccumulate(
                    G(6, -3), C(6, -3), broadcast<Real>(k93) * waveRadial6 * U(6, -3));
                detail::ewaldAccumulate(
                    G(6, -2), C(6, -2), broadcast<Real>(k92) * waveRadial6 * U(6, -2));
                detail::ewaldAccumulate(
                    G(6, -1), C(6, -1), broadcast<Real>(k91) * waveRadial6 * U(6, -1));
                detail::ewaldAccumulate(
                    G(6, 0), C(6, 0), broadcast<Real>(k90) * waveRadial6 * U(6, 0));
                detail::ewaldAccumulate(
                    G(6, 1), C(6, 1), broadcast<Real>(k91) * waveRadial6 * U(6, 1));
                detail::ewaldAccumulate(
                    G(6, 2), C(6, 2), broadcast<Real>(k92) * waveRadial6 * U(6, 2));
                detail::ewaldAccumulate(
                    G(6, 3), C(6, 3), broadcast<Real>(k93) * waveRadial6 * U(6, 3));
                detail::ewaldAccumulate(
                    G(6, 4), C(6, 4), broadcast<Real>(k95) * waveRadial6 * U(6, 4));
                detail::ewaldAccumulate(
                    G(6, 5), C(6, 5), broadcast<Real>(k96) * waveRadial6 * U(6, 5));
                detail::ewaldAccumulate(
                    G(6, 6), C(6, 6), broadcast<Real>(k97) * waveRadial6 * U(6, 6));
            }
        }
    }
}
}    // namespace cosmictiger::gravity::spherical
namespace cosmictiger::gravity::spherical {
template <typename Real>
void evaluateEwaldGreen(Real& greenTrace, Real x, Real y, Real z, Real* output) {
    using namespace math;
    using Working = typename detail::EwaldWorkingType<Real>::Type;
    Working temporary[49];
    Working trace;
    evaluateEwaldWorking(trace, detail::ewaldConvert<Working>(x), detail::ewaldConvert<Working>(y),
        detail::ewaldConvert<Working>(z), temporary);
    for (int i = 0; i < 49; ++i) {
        output[i] = detail::ewaldConvert<Real>(temporary[i]);
    }
    greenTrace = detail::ewaldConvert<Real>(trace);
}
}    // namespace cosmictiger::gravity::spherical
