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
    Real harmonics[49];
    Real compensation[49];
    Real imagePair[49];
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
                    ewaldRegular(imageX, imageY, imageZ, imageSquared, harmonics);
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
                ewaldRegular(waveX, waveY, waveZ, waveSquared, harmonics);
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
