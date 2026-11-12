/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

// Generated counting-scalar definitions. Tests only; not a library source.
#include <cosmictiger/gravity/kernels/cartesian_tracefree/ewald_green.hpp>
namespace cosmictiger::gravity::cartesian {
template <class Real>
void accumulateEwaldRadial(Vector3<Real> const& x, Real const* q, Green<Real>& target) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k0 = static_cast<Scalar>(3.000000000000000000E+00);
    constexpr auto k1 = static_cast<Scalar>(6.000000000000000000E+00);
    constexpr auto k2 = static_cast<Scalar>(9.000000000000000000E+00);
    constexpr auto k3 = static_cast<Scalar>(1.000000000000000000E+01);
    constexpr auto k4 = static_cast<Scalar>(1.500000000000000000E+01);
    constexpr auto k5 = static_cast<Scalar>(4.500000000000000000E+01);
    Real const t10 = x[1] * x[1];
    Real const t13 = fma(t10, q[2], q[1]);
    Real const t14 = x[0] * q[2];
    Real const t16 = x[0] * x[0];
    Real const t19 = fma(x[0], t14, q[1]);
    Real const t20 = x[1] * q[2];
    Real const t23 = x[1] * t10;
    Real const t27 = t20 * broadcast<Real>(k0);
    Real const t28 = fma(t23, q[3], t27);
    Real const t29 = x[0] * q[3];
    Real const t31 = fma(t10, t29, t14);
    Real const t32 = t16 * q[3];
    Real const t35 = fma(x[1], t32, t20);
    Real const t36 = x[0] * t16;
    Real const t40 = t14 * broadcast<Real>(k0);
    Real const t41 = fma(t16, t29, t40);
    Real const t42 = t10 * q[3];
    Real const t44 = x[2] * q[2];
    Real const t45 = fma(x[2], t42, t44);
    Real const t46 = x[1] * t29;
    Real const t49 = fma(x[2], t32, t44);
    Real const t50 = t10 * t10;
    Real const t56 = q[2] * broadcast<Real>(k0);
    Real const t57 = fma(t42, broadcast<Real>(k1), t56);
    Real const t58 = fma(t50, q[4], t57);
    Real const t59 = x[0] * q[4];
    Real const t61 = broadcast<Real>(k0) * t46;
    Real const t62 = fma(t23, t59, t61);
    Real const t63 = t16 * q[4];
    Real const t66 = q[2] + t32;
    Real const t67 = t42 + t66;
    Real const t68 = fma(t10, t63, t67);
    Real const t69 = t36 * q[4];
    Real const t73 = fma(x[1], t69, t61);
    Real const t74 = t16 * t16;
    Real const t79 = fma(t32, broadcast<Real>(k1), t56);
    Real const t80 = fma(t16, t63, t79);
    Real const t81 = t23 * q[4];
    Real const t83 = x[1] * q[3];
    Real const t84 = x[2] * t83;
    Real const t85 = broadcast<Real>(k0) * t84;
    Real const t86 = fma(x[2], t81, t85);
    Real const t87 = t10 * t59;
    Real const t89 = x[2] * t29;
    Real const t90 = fma(x[2], t87, t89);
    Real const t91 = x[1] * t63;
    Real const t93 = fma(x[2], t91, t84);
    Real const t95 = broadcast<Real>(k0) * t89;
    Real const t96 = fma(x[2], t69, t95);
    Real const t97 = t10 * t23;
    Real const t105 = t83 * broadcast<Real>(k4);
    Real const t106 = fma(t81, broadcast<Real>(k3), t105);
    Real const t107 = fma(t97, q[5], t106);
    Real const t108 = x[0] * q[5];
    Real const t111 = broadcast<Real>(k0) * t29;
    Real const t112 = fma(broadcast<Real>(k1), t87, t111);
    Real const t113 = fma(t50, t108, t112);
    Real const t114 = t16 * q[5];
    Real const t118 = broadcast<Real>(k0) * t83;
    Real const t119 = t81 + t118;
    Real const t120 = fma(broadcast<Real>(k0), t91, t119);
    Real const t121 = fma(t23, t114, t120);
    Real const t122 = t36 * q[5];
    Real const t127 = t69 + t111;
    Real const t128 = fma(broadcast<Real>(k0), t87, t127);
    Real const t129 = fma(t10, t122, t128);
    Real const t130 = t74 * q[5];
    Real const t134 = fma(broadcast<Real>(k1), t91, t118);
    Real const t135 = fma(x[1], t130, t134);
    Real const t136 = t16 * t36;
    Real const t143 = t29 * broadcast<Real>(k4);
    Real const t144 = fma(t69, broadcast<Real>(k3), t143);
    Real const t145 = fma(t36, t114, t144);
    Real const t146 = x[2] * q[3];
    Real const t147 = broadcast<Real>(k0) * t146;
    Real const t148 = t50 * q[5];
    Real const t150 = t10 * q[4];
    Real const t151 = x[2] * t150;
    Real const t152 = broadcast<Real>(k1) * t151;
    Real const t153 = fma(x[2], t148, t152);
    Real const t154 = t147 + t153;
    Real const t155 = t23 * t108;
    Real const t157 = x[1] * t59;
    Real const t158 = x[2] * t157;
    Real const t159 = broadcast<Real>(k0) * t158;
    Real const t160 = fma(x[2], t155, t159);
    Real const t161 = x[2] * t63;
    Real const t162 = t10 * t114;
    Real const t164 = t146 + t151;
    Real const t165 = fma(x[2], t162, t164);
    Real const t166 = t161 + t165;
    Real const t167 = x[1] * t122;
    Real const t169 = fma(x[2], t167, t159);
    Real const t172 = fma(broadcast<Real>(k1), t161, t147);
    Real const t173 = fma(x[2], t130, t172);
    Real const t174 = t23 * t23;
    Real const t182 = q[3] * broadcast<Real>(k4);
    Real const t183 = fma(broadcast<Real>(k4), t148, t182);
    Real const t184 = fma(t150, broadcast<Real>(k5), t183);
    Real const t185 = fma(t174, q[6], t184);
    Real const t186 = x[0] * q[6];
    Real const t189 = broadcast<Real>(k4) * t157;
    Real const t190 = fma(broadcast<Real>(k3), t155, t189);
    Real const t191 = fma(t97, t186, t190);
    Real const t192 = t16 * q[6];
    Real const t195 = broadcast<Real>(k1) * t162;
    Real const t198 = q[3] * broadcast<Real>(k0);
    Real const t199 = fma(broadcast<Real>(k0), t63, t198);
    Real const t200 = t148 + t199;
    Real const t201 = fma(broadcast<Real>(k1), t150, t200);
    Real const t202 = t195 + t201;
    Real const t203 = fma(t50, t192, t202);
    Real const t204 = t36 * q[6];
    Real const t211 = t157 * broadcast<Real>(k2);
    Real const t212 = fma(broadcast<Real>(k0), t155, t211);
    Real const t213 = fma(broadcast<Real>(k0), t167, t212);
    Real const t214 = fma(t23, t204, t213);
    Real const t215 = t74 * q[6];
    Real const t220 = fma(broadcast<Real>(k1), t63, t198);
    Real const t221 = t130 + t220;
    Real const t222 = fma(broadcast<Real>(k0), t150, t221);
    Real const t223 = t195 + t222;
    Real const t224 = fma(t10, t215, t223);
    Real const t225 = t136 * q[6];
    Real const t230 = fma(broadcast<Real>(k3), t167, t189);
    Real const t231 = fma(x[1], t225, t230);
    Real const t239 = fma(t63, broadcast<Real>(k5), t182);
    Real const t240 = fma(broadcast<Real>(k4), t130, t239);
    Real const t241 = fma(t36, t204, t240);
    Real const t242 = x[1] * q[4];
    Real const t243 = x[2] * t242;
    Real const t245 = t97 * q[6];
    Real const t247 = t23 * q[5];
    Real const t248 = x[2] * t247;
    Real const t249 = broadcast<Real>(k3) * t248;
    Real const t250 = fma(x[2], t245, t249);
    Real const t251 = fma(broadcast<Real>(k4), t243, t250);
    Real const t252 = x[2] * t59;
    Real const t253 = broadcast<Real>(k0) * t252;
    Real const t254 = t50 * t186;
    Real const t256 = t10 * t108;
    Real const t257 = x[2] * t256;
    Real const t258 = broadcast<Real>(k1) * t257;
    Real const t259 = fma(x[2], t254, t258);
    Real const t260 = t253 + t259;
    Real const t261 = x[1] * t114;
    Real const t262 = x[2] * t261;
    Real const t264 = t23 * t192;
    Real const t266 = broadcast<Real>(k0) * t243;
    Real const t267 = t248 + t266;
    Real const t268 = fma(x[2], t264, t267);
    Real const t269 = fma(broadcast<Real>(k0), t262, t268);
    Real const t270 = x[2] * t122;
    Real const t271 = t10 * t204;
    Real const t274 = fma(broadcast<Real>(k0), t257, t253);
    Real const t275 = fma(x[2], t271, t274);
    Real const t276 = t270 + t275;
    Real const t277 = x[1] * t215;
    Real const t280 = fma(broadcast<Real>(k1), t262, t266);
    Real const t281 = fma(x[2], t277, t280);
    Real const t284 = broadcast<Real>(k4) * t252;
    Real const t285 = fma(broadcast<Real>(k3), t270, t284);
    Real const t286 = fma(x[2], t225, t285);
    target[0] += q[0];
    target[1] = fma(x[1], q[1], target[1]);
    target[2] = fma(q[1], x[0], target[2]);
    target[3] = fma(q[1], x[2], target[3]);
    target[4] += t13;
    target[5] = fma(x[1], t14, target[5]);
    target[6] += t19;
    target[7] = fma(x[2], t20, target[7]);
    target[8] = fma(x[2], t14, target[8]);
    target[9] += t28;
    target[10] += t31;
    target[11] += t35;
    target[12] += t41;
    target[13] += t45;
    target[14] = fma(x[2], t46, target[14]);
    target[15] += t49;
    target[16] += t58;
    target[17] += t62;
    target[18] += t68;
    target[19] += t73;
    target[20] += t80;
    target[21] += t86;
    target[22] += t90;
    target[23] += t93;
    target[24] += t96;
    target[25] += t107;
    target[26] += t113;
    target[27] += t121;
    target[28] += t129;
    target[29] += t135;
    target[30] += t145;
    target[31] += t154;
    target[32] += t160;
    target[33] += t166;
    target[34] += t169;
    target[35] += t173;
    target[36] += t185;
    target[37] += t191;
    target[38] += t203;
    target[39] += t214;
    target[40] += t224;
    target[41] += t231;
    target[42] += t241;
    target[43] += t251;
    target[44] += t260;
    target[45] += t269;
    target[46] += t276;
    target[47] += t281;
    target[48] += t286;
}
}    // namespace cosmictiger::gravity::cartesian
namespace cosmictiger::gravity::cartesian {
template <class Real>
void evaluateEwaldGreen(Vector3<Real> const& distance, Green<Real>& output) {
    using namespace math;
    Fma const fma{};
    using Scalar = SimdScalar<Real>;
    constexpr auto k00 = static_cast<Scalar>(0.000000000000000000E+00);
    constexpr auto k01 = static_cast<Scalar>(3.278688524590164105E-02);
    constexpr auto k02 = static_cast<Scalar>(3.389830508474576259E-02);
    constexpr auto k03 = static_cast<Scalar>(3.508771929824561209E-02);
    constexpr auto k04 = static_cast<Scalar>(3.636363636363636187E-02);
    constexpr auto k05 = static_cast<Scalar>(3.773584905660377214E-02);
    constexpr auto k06 = static_cast<Scalar>(3.921568627450980338E-02);
    constexpr auto k07 = static_cast<Scalar>(4.081632653061224164E-02);
    constexpr auto k08 = static_cast<Scalar>(4.255319148936170109E-02);
    constexpr auto k09 = static_cast<Scalar>(4.444444444444444614E-02);
    constexpr auto k10 = static_cast<Scalar>(4.651162790697674354E-02);
    constexpr auto k11 = static_cast<Scalar>(4.878048780487805047E-02);
    constexpr auto k12 = static_cast<Scalar>(5.128205128205128027E-02);
    constexpr auto k13 = static_cast<Scalar>(5.405405405405405705E-02);
    constexpr auto k14 = static_cast<Scalar>(5.714285714285714107E-02);
    constexpr auto k15 = static_cast<Scalar>(6.060606060606060774E-02);
    constexpr auto k16 = static_cast<Scalar>(6.451612903225806273E-02);
    constexpr auto k17 = static_cast<Scalar>(6.896551724137930939E-02);
    constexpr auto k18 = static_cast<Scalar>(7.407407407407406996E-02);
    constexpr auto k19 = static_cast<Scalar>(7.692307692307692735E-02);
    constexpr auto k20 = static_cast<Scalar>(8.000000000000000167E-02);
    constexpr auto k21 = static_cast<Scalar>(8.695652173913043237E-02);
    constexpr auto k22 = static_cast<Scalar>(9.090909090909091161E-02);
    constexpr auto k23 = static_cast<Scalar>(9.523809523809523281E-02);
    constexpr auto k24 = static_cast<Scalar>(1.052631578947368363E-01);
    constexpr auto k25 = static_cast<Scalar>(1.111111111111111049E-01);
    constexpr auto k26 = static_cast<Scalar>(1.176470588235294101E-01);
    constexpr auto k27 = static_cast<Scalar>(1.333333333333333315E-01);
    constexpr auto k28 = static_cast<Scalar>(1.428571428571428492E-01);
    constexpr auto k29 = static_cast<Scalar>(2.000000000000000111E-01);
    constexpr auto k30 = static_cast<Scalar>(3.333333333333333148E-01);
    constexpr auto k31 = static_cast<Scalar>(7.853981633974482790E-01);
    constexpr auto k32 = static_cast<Scalar>(1.000000000000000000E+00);
    constexpr auto k33 = static_cast<Scalar>(2.000000000000000000E+00);
    constexpr auto k34 = static_cast<Scalar>(2.256758334191025117E+00);
    constexpr auto k35 = static_cast<Scalar>(2.256758334191025148E+00);
    constexpr auto k36 = static_cast<Scalar>(3.000000000000000000E+00);
    constexpr auto k37 = static_cast<Scalar>(4.000000000000000000E+00);
    constexpr auto k38 = static_cast<Scalar>(5.000000000000000000E+00);
    constexpr auto k39 = static_cast<Scalar>(6.283185307179586232E+00);
    constexpr auto k40 = static_cast<Scalar>(7.000000000000000000E+00);
    constexpr auto k41 = static_cast<Scalar>(8.000000000000000000E+00);
    constexpr auto k42 = static_cast<Scalar>(9.000000000000000000E+00);
    constexpr auto k43 = static_cast<Scalar>(1.100000000000000000E+01);
    constexpr auto k44 = static_cast<Scalar>(1.805406667352820094E+01);
    constexpr auto k45 = static_cast<Scalar>(1.444325333882256075E+02);
    constexpr auto k46 = static_cast<Scalar>(1.155460267105804860E+03);
    constexpr auto k47 = static_cast<Scalar>(9.243682136846438880E+03);
    constexpr auto k48 = static_cast<Scalar>(7.394945709477151104E+04);
    constexpr auto k49 = static_cast<Scalar>(5.915956567581720883E+05);
    using namespace math;
    Real const one = broadcast<Real>(k32);
    Vector3<Real> x = distance;
    Real const r2 = x.dot(x);
    Real const y = broadcast<Real>(k37) * r2;
    Real const exponential = detail::ewaldExp(-y);
    Real series = one;
    series = fma(series, y * broadcast<Real>(k01), one);
    series = fma(series, y * broadcast<Real>(k02), one);
    series = fma(series, y * broadcast<Real>(k03), one);
    series = fma(series, y * broadcast<Real>(k04), one);
    series = fma(series, y * broadcast<Real>(k05), one);
    series = fma(series, y * broadcast<Real>(k06), one);
    series = fma(series, y * broadcast<Real>(k07), one);
    series = fma(series, y * broadcast<Real>(k08), one);
    series = fma(series, y * broadcast<Real>(k09), one);
    series = fma(series, y * broadcast<Real>(k10), one);
    series = fma(series, y * broadcast<Real>(k11), one);
    series = fma(series, y * broadcast<Real>(k12), one);
    series = fma(series, y * broadcast<Real>(k13), one);
    series = fma(series, y * broadcast<Real>(k14), one);
    series = fma(series, y * broadcast<Real>(k15), one);
    series = fma(series, y * broadcast<Real>(k16), one);
    series = fma(series, y * broadcast<Real>(k17), one);
    series = fma(series, y * broadcast<Real>(k18), one);
    series = fma(series, y * broadcast<Real>(k20), one);
    series = fma(series, y * broadcast<Real>(k21), one);
    series = fma(series, y * broadcast<Real>(k23), one);
    series = fma(series, y * broadcast<Real>(k24), one);
    series = fma(series, y * broadcast<Real>(k26), one);
    series = fma(series, y * broadcast<Real>(k27), one);
    Real s = exponential * series * broadcast<Real>(k19);
    Real q[7];
    q[6] = broadcast<Real>(k49) * s;
    s = fma(broadcast<Real>(k33) * y, s, exponential) * broadcast<Real>(k22);
    q[5] = -broadcast<Real>(k48) * s;
    s = fma(broadcast<Real>(k33) * y, s, exponential) * broadcast<Real>(k25);
    q[4] = broadcast<Real>(k47) * s;
    s = fma(broadcast<Real>(k33) * y, s, exponential) * broadcast<Real>(k28);
    q[3] = -broadcast<Real>(k46) * s;
    s = fma(broadcast<Real>(k33) * y, s, exponential) * broadcast<Real>(k29);
    q[2] = broadcast<Real>(k45) * s;
    s = fma(broadcast<Real>(k33) * y, s, exponential) * broadcast<Real>(k30);
    q[1] = -broadcast<Real>(k44) * s;
    s = fma(broadcast<Real>(k33) * y, s, exponential) * broadcast<Real>(k32);
    q[0] = broadcast<Real>(k34) * s;
    output = broadcast<Real>(k00);
    accumulateEwaldRadial(x, q, output);
    for (int image = 0; image < 125; ++image) {
        Green<Real> pair{};
        for (int sign = -1; sign <= 1; sign += 2) {
            for (int dim = 0; dim < 3; ++dim) {
                x[dim] = distance[dim] + broadcast<Real>(sign * detail::kEwaldImages[image][dim]);
            }
            Real const radiusSquared = x.dot(x);
            Real const inverseRadius = one / sqrt(radiusSquared);
            Real const inverseSquared = inverseRadius * inverseRadius;
            Real exponentialTerm =
                broadcast<Real>(k35) * detail::ewaldExp(-broadcast<Real>(k37) * radiusSquared);
            q[0] = -detail::ewaldErfc(broadcast<Real>(k33) / inverseRadius) * inverseRadius;
            q[1] = fma(-broadcast<Real>(k32), q[0], exponentialTerm) * inverseSquared;
            exponentialTerm *= -broadcast<Real>(k41);
            q[2] = fma(-broadcast<Real>(k36), q[1], exponentialTerm) * inverseSquared;
            exponentialTerm *= -broadcast<Real>(k41);
            q[3] = fma(-broadcast<Real>(k38), q[2], exponentialTerm) * inverseSquared;
            exponentialTerm *= -broadcast<Real>(k41);
            q[4] = fma(-broadcast<Real>(k40), q[3], exponentialTerm) * inverseSquared;
            exponentialTerm *= -broadcast<Real>(k41);
            q[5] = fma(-broadcast<Real>(k42), q[4], exponentialTerm) * inverseSquared;
            exponentialTerm *= -broadcast<Real>(k41);
            q[6] = fma(-broadcast<Real>(k43), q[5], exponentialTerm) * inverseSquared;
            accumulateEwaldRadial(x, q, pair);
        }
        for (int i = 0; i < 49; ++i) {
            output[i] += pair[i];
        }
    }
    for (int wave = 0; wave < 128; ++wave) {
        Real phase = broadcast<Real>(k00);
        for (int dim = 0; dim < 3; ++dim) {
            phase = fma(broadcast<Real>(detail::kEwaldWaves[wave][dim]), distance[dim], phase);
        }
        phase *= broadcast<Real>(k39);
        Real const cosine = detail::ewaldCos(phase);
        Real const sine = -detail::ewaldSin(phase);
        output[0] = fma(broadcast<Real>(detail::kEwaldWaves[wave][3]), cosine, output[0]);
        output[1] = fma(broadcast<Real>(detail::kEwaldWaves[wave][4]), sine, output[1]);
        output[2] = fma(broadcast<Real>(detail::kEwaldWaves[wave][5]), sine, output[2]);
        output[3] = fma(broadcast<Real>(detail::kEwaldWaves[wave][6]), sine, output[3]);
        output[4] = fma(broadcast<Real>(detail::kEwaldWaves[wave][7]), cosine, output[4]);
        output[5] = fma(broadcast<Real>(detail::kEwaldWaves[wave][8]), cosine, output[5]);
        output[6] = fma(broadcast<Real>(detail::kEwaldWaves[wave][9]), cosine, output[6]);
        output[7] = fma(broadcast<Real>(detail::kEwaldWaves[wave][10]), cosine, output[7]);
        output[8] = fma(broadcast<Real>(detail::kEwaldWaves[wave][11]), cosine, output[8]);
        output[9] = fma(broadcast<Real>(detail::kEwaldWaves[wave][12]), sine, output[9]);
        output[10] = fma(broadcast<Real>(detail::kEwaldWaves[wave][13]), sine, output[10]);
        output[11] = fma(broadcast<Real>(detail::kEwaldWaves[wave][14]), sine, output[11]);
        output[12] = fma(broadcast<Real>(detail::kEwaldWaves[wave][15]), sine, output[12]);
        output[13] = fma(broadcast<Real>(detail::kEwaldWaves[wave][16]), sine, output[13]);
        output[14] = fma(broadcast<Real>(detail::kEwaldWaves[wave][17]), sine, output[14]);
        output[15] = fma(broadcast<Real>(detail::kEwaldWaves[wave][18]), sine, output[15]);
        output[16] = fma(broadcast<Real>(detail::kEwaldWaves[wave][19]), cosine, output[16]);
        output[17] = fma(broadcast<Real>(detail::kEwaldWaves[wave][20]), cosine, output[17]);
        output[18] = fma(broadcast<Real>(detail::kEwaldWaves[wave][21]), cosine, output[18]);
        output[19] = fma(broadcast<Real>(detail::kEwaldWaves[wave][22]), cosine, output[19]);
        output[20] = fma(broadcast<Real>(detail::kEwaldWaves[wave][23]), cosine, output[20]);
        output[21] = fma(broadcast<Real>(detail::kEwaldWaves[wave][24]), cosine, output[21]);
        output[22] = fma(broadcast<Real>(detail::kEwaldWaves[wave][25]), cosine, output[22]);
        output[23] = fma(broadcast<Real>(detail::kEwaldWaves[wave][26]), cosine, output[23]);
        output[24] = fma(broadcast<Real>(detail::kEwaldWaves[wave][27]), cosine, output[24]);
        output[25] = fma(broadcast<Real>(detail::kEwaldWaves[wave][28]), sine, output[25]);
        output[26] = fma(broadcast<Real>(detail::kEwaldWaves[wave][29]), sine, output[26]);
        output[27] = fma(broadcast<Real>(detail::kEwaldWaves[wave][30]), sine, output[27]);
        output[28] = fma(broadcast<Real>(detail::kEwaldWaves[wave][31]), sine, output[28]);
        output[29] = fma(broadcast<Real>(detail::kEwaldWaves[wave][32]), sine, output[29]);
        output[30] = fma(broadcast<Real>(detail::kEwaldWaves[wave][33]), sine, output[30]);
        output[31] = fma(broadcast<Real>(detail::kEwaldWaves[wave][34]), sine, output[31]);
        output[32] = fma(broadcast<Real>(detail::kEwaldWaves[wave][35]), sine, output[32]);
        output[33] = fma(broadcast<Real>(detail::kEwaldWaves[wave][36]), sine, output[33]);
        output[34] = fma(broadcast<Real>(detail::kEwaldWaves[wave][37]), sine, output[34]);
        output[35] = fma(broadcast<Real>(detail::kEwaldWaves[wave][38]), sine, output[35]);
        output[36] = fma(broadcast<Real>(detail::kEwaldWaves[wave][39]), cosine, output[36]);
        output[37] = fma(broadcast<Real>(detail::kEwaldWaves[wave][40]), cosine, output[37]);
        output[38] = fma(broadcast<Real>(detail::kEwaldWaves[wave][41]), cosine, output[38]);
        output[39] = fma(broadcast<Real>(detail::kEwaldWaves[wave][42]), cosine, output[39]);
        output[40] = fma(broadcast<Real>(detail::kEwaldWaves[wave][43]), cosine, output[40]);
        output[41] = fma(broadcast<Real>(detail::kEwaldWaves[wave][44]), cosine, output[41]);
        output[42] = fma(broadcast<Real>(detail::kEwaldWaves[wave][45]), cosine, output[42]);
        output[43] = fma(broadcast<Real>(detail::kEwaldWaves[wave][46]), cosine, output[43]);
        output[44] = fma(broadcast<Real>(detail::kEwaldWaves[wave][47]), cosine, output[44]);
        output[45] = fma(broadcast<Real>(detail::kEwaldWaves[wave][48]), cosine, output[45]);
        output[46] = fma(broadcast<Real>(detail::kEwaldWaves[wave][49]), cosine, output[46]);
        output[47] = fma(broadcast<Real>(detail::kEwaldWaves[wave][50]), cosine, output[47]);
        output[48] = fma(broadcast<Real>(detail::kEwaldWaves[wave][51]), cosine, output[48]);
    }
    output[0] += broadcast<Real>(k31);
    output[0] *= broadcast<Real>(kScaleFactorInverse1);
    output[1] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1);
    output[2] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1);
    output[3] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1);
    output[4] *=
        broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[5] *=
        broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[6] *=
        broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[7] *=
        broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[8] *=
        broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[9] *= broadcast<Real>(
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[10] *= broadcast<Real>(
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[11] *= broadcast<Real>(
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[12] *= broadcast<Real>(
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[13] *= broadcast<Real>(
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[14] *= broadcast<Real>(
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[15] *= broadcast<Real>(
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[16] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[17] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[18] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[19] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[20] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[21] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[22] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[23] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[24] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[25] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[26] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[27] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[28] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[29] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[30] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[31] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[32] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[33] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[34] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[35] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1);
    output[36] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[37] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[38] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[39] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[40] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[41] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[42] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[43] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[44] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[45] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[46] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[47] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
    output[48] *= broadcast<Real>(kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 * kScaleFactorInverse1 *
        kScaleFactorInverse1);
}
}    // namespace cosmictiger::gravity::cartesian
