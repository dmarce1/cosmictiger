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

#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/timer.hpp>

struct ChemistryParams {
    double codeToCm;
    double codeToG;
    double codeToS;
    bool stars;
    float a;
    float g;
};

#define CHEM_BLOCK_SIZE 96

#define NCOOL 14

__managed__ Array<double, NCOOL> coolingTotals;

__device__ float testElectronFraction(float ne, SpeciesT n0Value, SpeciesT& nValue, float t,
    float dt, float z, float k1, float k2, float k3, float k4, float k5, float k6, float k7,
    float k8, float k9, float k11, float k12, float k14, float k16, float sigma20, float sigma21,
    float sigma22, int& flops) {
    //	const int tid = threadIdx.x;
    //	auto tm = clock64();
    float& h = nValue.h;
    float& hp = nValue.hp;
    float& hn = nValue.hn;
    float& he = nValue.he;
    float& hep = nValue.hep;
    float& hepp = nValue.hepp;
    float& h2 = nValue.h2;
    float hp0 = n0Value.hp;
    float hep0 = n0Value.hep;
    float hepp0 = n0Value.hepp;
    float h20 = n0Value.h2;
    float hall = h + hp + 2.f * h2 + hn;    // 4
    float heall = he + hep + hepp;          // 2
    nValue = n0Value;
#define Power(a, b) powf(a, b)
#define Sqrt(a) sqrtf(a)
#define Rule(a, b) a = b
#define List(...) __VA_ARGS__
    //	PRINT("in %e %e %e %e %e %e %e\n", H, Hp, Hn, H2, He, Hep, Hepp);
    hn = (h * k7 * ne) / fmaf(hp, k16, fmaf(h, k8, k14 * ne));    // 11
    hp = hall - (h + fmaf(2.f, h2, hn));
    hp = fmaxf(hp, 0.0f);                                             // 1
                                                                      // 4
    float const twoH2 = 2.f * h2;                                     // 1
    float const tmp1 = (twoH2 - hall + hn);                           // 2
    float const hpnum = (hp0 - dt * tmp1 * fmaf(ne, k1, sigma20));    // 5
    float const hpden = fmaf(dt, fmaf(ne, k1 + k2, sigma20), 1.f);    // 5
    hp = hpnum / hpden;                                               // 4
    h = hall - (hp + fmaf(2.f, h2, hn));                              // 4
    hn = (h * k7 * ne) / fmaf(hp, k16, fmaf(h, k8, k14 * ne));        // 11
    h = hall - (hp + fmaf(2.f, h2, hn));                              // 4
    float const tmp2 = k7 * k8;                                       // 1
    float const h2num = fmaf(
        h20, fmaf(hp, k16, fmaf(h, k8, k14 * ne)), dt * h * tmp2 * ne * fmaf(2.f, h20, h));    // 13
    float const k14ne = k14 * ne;                                                              // 1
    float const tmp3 = hp * k16 + fmaf(h, k8, k14ne);                                          // 4
    float const h2den =
        fmaf(dt, fmaf(tmp3, (fmaf(hp, k11, ne * k12)), ne * 2.f * h * tmp2), tmp3);    // 10
    h2 = h2num / h2den;                                                                // 4
    h = hall - (hp + fmaf(2.f, h2, hn));                                               // 4
    hn = (h * k7 * ne) / fmaf(hp, k16, fmaf(h, k8, k14 * ne));                         // 11
    h = hall - (hp + fmaf(2.f, h2, hn));
    float const k6ne = k6 * ne;                         // 1
    float const tmp4 = fmaf(dt, k6ne, 1.f);             // 2
    float const tmp5 = fmaf(k3, ne, sigma21 - k6ne);    // 3
    float const hepNum = fmaf(
        -hepp0, dt * tmp5, tmp4 * (fmaf(dt, fmaf(heall, k3 * ne, heall * sigma21), hep0)));    // 11
    float const k345 = k3 + k4 + k5;                                                           // 2
    float const hepDen = fmaf(tmp4, fmaf(dt, fmaf(k345, ne, sigma21 + sigma22), 1.f),
        (sqr(dt) * tmp5 * fmaf(k5, ne, sigma22)));    // 12
    hep = hepNum / hepDen;                            // 4
    float const heppNum = fmaf(dt,
        fmaf(dt, heall * fmaf(k5, ne * fmaf(k3, ne, sigma21), fmaf(k3, ne, sigma21) * sigma22),
            fmaf(ne, fmaf(hepp0, k345, hep0 * k5), fmaf(hepp0, sigma21, (hep0 + hepp0) * sigma22))),
        hepp0);    // 22
    float const heppDen = fmaf(dt,
        (dt *
                fmaf(k3, sqr(ne) * (k5 + k6),
                    fmaf(ne, fmaf(k6, fmaf(k4, ne, sigma21), k5 * sigma21),
                        fmaf(k3, ne, sigma21) * sigma22)) +
            fmaf((k345 + k6), ne, sigma21) + sigma22),
        1.f);                      // 21
    hepp = (heppNum / heppDen);    // 4
    he = heall - hep - hepp;       // 2
    h = fmaxf(h, 0.0f);            // 1
    hp = fmaxf(hp, 0.0f);          // 1
    hn = fmaxf(hn, 0.0f);          // 1
    h2 = fmaxf(h2, 0.0f);          // 1
    he = fmaxf(he, 0.0f);          // 1
    hep = fmaxf(hep, 0.0f);        // 1
    hepp = fmaxf(hepp, 0.0f);      // 1
    flops += 200;
    //	tm = clock64() - tm;
    //	if (tid == 0) {
    //		atomicAdd(&timer3, (double) tm);
    //	}
    return hp - hn + hep + 2 * hepp - ne;
}

__device__ float testTemperature(SpeciesT n0Value, SpeciesT& nValue, float t0, float t, float dt,
    float z, int& flops, float* dedtPtr, bool addTotals) {
    //	const int tid = threadIdx.x;
    //	auto tm = clock64();
    auto const oT = t;
    t = fmaxf(TMIN, t);                         // 1
    t = fminf(TMAX, t);                         // 1
    float const tev = t / constants::kEvtoK;    // 4
    float const tevinv = 1.f / tev;             // 4
    float const lnTev = log(tev);               // 8
    float const log10T = log10(t);              // 8
    float const t3 = t * 1e-3f;                 // 1
    float const sqrtTinv = rsqrtf(t);           // 4
    float const sqrtT = sqrtf(t);               // 4
    float const tinv = 1.f / t;                 // 4
    float const t6 = t * 1e-6f;                 // 1
    float k1, k2, k3, k4, k5, k6, k7, k8, k9, k11, k12, k14, k16;
    {
        float const c0 = -32.71396786f;
        float const c1 = 13.536556f;
        float const c2 = -5.73932875f;
        float const c3 = 1.56315498f;
        float const c4 = -0.2877056f;
        float const c5 = 3.48255977e-2f;
        float const c6 = -2.63197617e-3f;
        float const c7 = 1.11954395e-4f;
        float const c8 = -2.03914985e-6f;
        float k = c8;
        k = fmaf(k, lnTev, c7);
        k = fmaf(k, lnTev, c6);
        k = fmaf(k, lnTev, c5);
        k = fmaf(k, lnTev, c4);
        k = fmaf(k, lnTev, c3);
        k = fmaf(k, lnTev, c2);
        k = fmaf(k, lnTev, c1);
        k = fmaf(k, lnTev, c0);
        k1 = expf(k);
    }
    {
        float const c0 = -28.6130338f;
        float const c1 = -0.72411256f;
        float const c2 = -2.02604473e-2f;
        float const c3 = -2.38086188e-3f;
        float const c4 = -3.21260521e-4f;
        float const c5 = -1.42150291e-5f;
        float const c6 = 4.98910892e-6f;
        float const c7 = 5.75561414e-7f;
        float const c8 = -1.85676704e-8f;
        float const c9 = -3.07113524e-9f;
        float k = c9;
        k = fmaf(k, lnTev, c8);
        k = fmaf(k, lnTev, c7);
        k = fmaf(k, lnTev, c6);
        k = fmaf(k, lnTev, c5);
        k = fmaf(k, lnTev, c4);
        k = fmaf(k, lnTev, c3);
        k = fmaf(k, lnTev, c2);
        k = fmaf(k, lnTev, c1);
        k = fmaf(k, lnTev, c0);
        k2 = expf(k);
    }
    {
        float const c0 = -44.09864886f;
        float const c1 = 23.91596563f;
        float const c2 = -10.7532302f;
        float const c3 = 3.05803875f;
        float const c4 = -0.56851189f;
        float const c5 = 6.79539123e-2f;
        float const c6 = -5.00905610e-3f;
        float const c7 = 2.06723616e-4f;
        float const c8 = -3.64916141e-6f;
        float k = c8;
        k = fmaf(k, lnTev, c7);
        k = fmaf(k, lnTev, c6);
        k = fmaf(k, lnTev, c5);
        k = fmaf(k, lnTev, c4);
        k = fmaf(k, lnTev, c3);
        k = fmaf(k, lnTev, c2);
        k = fmaf(k, lnTev, c1);
        k = fmaf(k, lnTev, c0);
        k3 = expf(k);
    }
    {
        k4 = 3.925E-13f * powf(tev, -0.6353f);    // 5
        k4 += 1.544E-9f * powf(tev, -1.5f) *
            (0.3f * expf(-48.596f * tevinv) + expf(-40.496f * tevinv));    // 18
    }
    {
        float const c0 = -68.71040990f;
        float const c1 = 43.93347633f;
        float const c2 = -18.4806699f;
        float const c3 = 4.70162649f;
        float const c4 = -0.76924663f;
        float const c5 = 8.113042E-2f;
        float const c6 = -5.32402063E-3f;
        float const c7 = 1.97570531E-4f;
        float const c8 = -3.16558106E-6f;
        float k = c8;
        k = fmaf(k, lnTev, c7);
        k = fmaf(k, lnTev, c6);
        k = fmaf(k, lnTev, c5);
        k = fmaf(k, lnTev, c4);
        k = fmaf(k, lnTev, c3);
        k = fmaf(k, lnTev, c2);
        k = fmaf(k, lnTev, c1);
        k = fmaf(k, lnTev, c0);
        k5 = expf(k);
    }
    {
        k6 = 3.36e-10f * sqrtTinv * powf(t3, -0.2f) / (1.0f + powf(t6, 0.7));
    }
    {
        if (t < 6000.f) {    // 1
            float const c0 = 1.429e-18f;
            float const c1 = powf(t, 0.762f);                      // 8
            float const c2 = powf(t, 0.1523f * log10T);            // 9
            float const c3 = powf(t, -3.247e-2f * sqr(log10T));    // 10
            k7 = c0 * c1 * c2 * c3;                                // 3
            flops += 31;
        } else {
            float const c0 = 3.802e-17;
            float const c1 = powf(t, 0.1998f * log10T);                   // 9
            float const c2 = powf(10.f, 4.0415e-5f * powf(log10T, 6));    // 17
            float const c3 = powf(10.f, -5.447e-3f * powf(log10T, 4));    // 17
            k7 = c0 * c1 * c2 * c3;                                       // 3
            flops += 47;
        }
    }
    {
        if (tev > 0.1) {
            float const c0 = -20.06913897f;
            float const c1 = 0.22898f;
            float const c2 = 3.5998377E-2f;
            float const c3 = -4.55512E-3f;
            float const c4 = -3.10511544E-4f;
            float const c5 = 1.0732940E-4f;
            float const c6 = -8.36671960E-6f;
            float const c7 = 2.23830623E-7f;
            float k = c7;
            k = fmaf(k, lnTev, c6);
            k = fmaf(k, lnTev, c5);
            k = fmaf(k, lnTev, c4);
            k = fmaf(k, lnTev, c3);
            k = fmaf(k, lnTev, c2);
            k = fmaf(k, lnTev, c1);
            k = fmaf(k, lnTev, c0);
            k8 = expf(k);
            flops += 22;
        } else {
            k8 = 1.428e-9f;
        }
    }
    {
        float const c0 = -24.24914687f;
        float const c1 = 3.40082444f;
        float const c2 = -3.89800396f;
        float const c3 = 2.04558782f;
        float const c4 = -0.541618285f;
        float const c5 = 8.41077503E-2f;
        float const c6 = -7.87902615E-3f;
        float const c7 = 4.13839842E-4f;
        float const c8 = -9.36345888E-6f;
        float k = c8;
        k = fmaf(k, lnTev, c7);
        k = fmaf(k, lnTev, c6);
        k = fmaf(k, lnTev, c5);
        k = fmaf(k, lnTev, c4);
        k = fmaf(k, lnTev, c3);
        k = fmaf(k, lnTev, c2);
        k = fmaf(k, lnTev, c1);
        k = fmaf(k, lnTev, c0);
        k11 = expf(k);
    }
    {
        k12 = 5.6e-11f * sqrtT * expf(-102124.f * tinv);    // 14
    }
    {
        float const c0 = -18.01849334f;
        float const c1 = 2.3608522f;
        float const c2 = -0.28274430f;
        float const c3 = 1.62331664E-2f;
        float const c4 = -3.36501203E-2f;
        float const c5 = 1.17832978E-2f;
        float const c6 = -1.65619470E-3f;
        float const c7 = 1.06827520E-4f;
        float const c8 = -2.63128581E-6f;
        float k = c8;
        k = fmaf(k, lnTev, c7);
        k = fmaf(k, lnTev, c6);
        k = fmaf(k, lnTev, c5);
        k = fmaf(k, lnTev, c4);
        k = fmaf(k, lnTev, c3);
        k = fmaf(k, lnTev, c2);
        k = fmaf(k, lnTev, c1);
        k = fmaf(k, lnTev, c0);
        k14 = expf(k);
    }
    {
        k16 = 7e-7f * sqrtTinv;    // 5
    }
    t = oT;
    float cool = 0.f;
    float const ne = nValue.hp - nValue.hn + nValue.hep + 2 * nValue.hepp;    // 4
    float const t5 = t * 1e-5f;                                               // 1
    float const tmp1 = 1.0f / (1.0f + sqrtf(t5));                             // 9
    float const tmp2 = powf(t, -.1687f);                                      // 8
    float const tmp3 = powf(t3, -.2f);                                        // 8
    float const tmp4 = tmp3 / (1.0f + powf(t6, .7f));
    Array<float, NCOOL> c;
    c[0] = 7.5e-19f * tmp1 * expf(-118348.f * tinv) * ne * nValue.h;                         // 13
    c[1] = 9.1e-27f * tmp1 * expf(-13179.f * tinv) * tmp2 * sqr(ne) * nValue.he;             // 15
    c[2] = 5.54e-17f * tmp1 * expf(-473638.f * tinv) * powf(t, -.397f) * ne * nValue.hep;    // 22
    c[3] = 1.27e-21f * sqrtT * tmp1 * expf(-157809.1f * tinv) * ne * nValue.h;               // 4
    c[4] = 9.38e-22f * sqrtT * tmp1 * expf(-285335.4f * tinv) * ne * nValue.he;              // 4
    c[5] = 4.95e-22f * sqrtT * tmp1 * expf(-631515.f * tinv) * ne * nValue.hep;              // 4
    c[6] = 5.01e-27f * tmp1 * tmp2 * expf(-55338.f * tinv) * sqr(ne) * nValue.hep;           // 16
    c[7] = 8.7e-27f * sqrtT * tmp4 * ne * nValue.hp;                                         // 18
    c[8] = 1.55e-26f * powf(t, 0.3647f) * ne * nValue.hep;                                   // 12
    c[9] = 1.24e-13f * powf(t, -1.5f) * (1.f + .3f * expf(-94000.f * tinv)) *
        expf(-470000.f * tinv) * ne * nValue.hep;                             // 34
    c[10] = 3.48e-26f * sqrtT * tmp4 * ne * nValue.hepp;                      // 18
    float const qn = powf(nValue.h2, 0.77f) + 1.2 * powf(nValue.h, 0.77f);    // 18
    float lrL, lrH;
    float const x = log10T - 4.f;    // 1
    float const x2 = sqr(x);         // 1
    float lvL;
    if (t > 4031.f) {    // 1
        lrL = 1.38e-22f * expf(-9243.f * tinv);
        flops += 10;
    } else {
        lrL = powf(10.0f, -22.90f - 0.553f * x - 1.148f * x2);
        flops += 12;
    }
    lrL *= qn;           // 1
    if (t > 1087.f) {    // 1
        lrH = 3.9e-19f * expf(-6118.f * tinv);
        flops += 10;
    } else {
        lrH = powf(10.f, -19.24f + 0.474f * x - 1.247 * x2);
        flops += 12;
    }
    if (t > 1635.f) {    // 1
        lvL = 1e-12f * sqrtT * expf(-1000.f * tinv);
        flops += 11;
    } else {
        lvL = 1.4e-13f * expf(t * (1.f / 125.f) - sqr(t * (1.f / 577.f)));
        flops += 13;
    }
    lvL *= nValue.h * 8.18e-13f;                                                // 2
    float const lvH = 1.1e-13f * expf(-6744.f * tinv);                          // 10
    c[11] = nValue.h2 * (lrH * lrL / (lrL + lrH) + lvH * lvL / (lvL + lvH));    // 15
    c[12] = 1.43e-27f * sqrtT * (fmaf(0.34f, expf(-sqr(5.5f - log10T) * (1.f / 3.f)), 1.1f)) * ne *
        (nValue.hp + nValue.hep + nValue.hepp);                                 // 21
    c[13] = 5.65e-36f * sqr(sqr(1.f + z)) * (fmaf(-2.73f, (1 + z), t)) * ne;    // 10
    constexpr float kExpmax = 88.0f;
    float const tmp5 = powf(1.f + z, 0.43f);                                                // 9
    float const tmp6 = powf(1.f + z, 0.3f);                                                 // 9
    float const tmp8 = sqr(z - 2.3f);                                                       // 1
    float const tmp7 = tmp5 * expf(-fminf((1.f / 1.95f) * tmp8, kExpmax));                  // 12
    float const sigma20 = 5.6e-13f * tmp7;                                                  // 1
    float const sigma22 = 9.6e-15f * tmp6 * expf(-fminf((1.f / 2.6f) * tmp8, kExpmax));     // 12
    float const sigma21 = 3.2e-13f * tmp7;                                                  // 1
    float const sigmaH = 3.9e-24f * tmp7;                                                   // 1
    float const sigmaHe = 6.4e-24f * tmp5 * expf(-fminf((1.f / 2.1f) * tmp8, kExpmax));     // 12
    float const sigmaHep = 8.6e-26f * tmp6 * expf(-fminf((1.f / 2.7f) * tmp8, kExpmax));    // 12
    float const heat = sigmaH * nValue.h + sigmaHe * nValue.he + sigmaHep * nValue.hep;     // 5
    for (int i = 0; i < NCOOL; i++) {
        cool += c[i];
        if (addTotals) {
            atomicAdd(&coolingTotals[i], c[i] * dt);
        }
    }
    float const dedt = heat - cool;    // 1

    flops += 625;

    float const n0 = nValue.h + 2.0f * nValue.hp + nValue.h2 + nValue.he + 2.0f * nValue.hep +
        3.0f * nValue.hepp;                                              // 8
    float const cv0 = (1.5f * (1.0f - nValue.h2) + 2.5f * nValue.h2);    // 4

    float const htot = nValue.h + nValue.hp + nValue.hn + 2.f * nValue.h2;    // 4
    float const hetot = nValue.he + nValue.hep + nValue.hepp;                 // 2
    float neMax = htot + 4.f * hetot;                                         // 2
    float neMin = neMax * 1e-20f;                                             // 1
    for (int i = 0; i < 29; i++) {
        float neMid = sqrtf(neMax * neMin);
        float feMax, feMid;
        if (i == 0) {
            feMax = testElectronFraction(neMax, n0Value, nValue, t, dt, z, k1, k2, k3, k4, k5, k6,
                k7, k8, k9, k11, k12, k14, k16, sigma20, sigma21, sigma22, flops);
        }
        feMid = testElectronFraction(neMid, n0Value, nValue, t, dt, z, k1, k2, k3, k4, k5, k6, k7,
            k8, k9, k11, k12, k14, k16, sigma20, sigma21, sigma22, flops);
        if (copysignf(1.f, feMax) != copysignf(1.f, feMid)) {
            neMin = neMid;
        } else {
            neMax = neMid;
            feMax = feMid;
        }
        flops += 8;
    }

    float const n1 = nValue.h + 2.0f * nValue.hp + nValue.h2 + nValue.he + 2.0f * nValue.hep +
        3.0f * nValue.hepp;                  // 8
    float const cv1 = (1.5f + nValue.h2);    // 4
    *dedtPtr = dedt;
    return constants::kB * (cv1 * n1 * t - cv0 * n0 * t0) - dt * dedt;    // 8
}

__global__ void chemistryKernel(
    ChemistryParams params, ChemAttribs* chems, int nchems, int* nextIndex, double* totalFlops) {
    //	const int tid = threadIdx.x;
    //	auto tm = clock64();
    int index;
    index = atomicAdd(nextIndex, 1);
    double const codeToEnergyDensity =
        params.codeToG / (params.codeToCm * sqr(params.codeToS));              // 7
    double const codeToEnergy = sqr(params.codeToCm) / sqr(params.codeToS);    // 7
    double const codeToDensity = pow(params.codeToCm, -3) * params.codeToG;    // 10
    int flops = 18;
    double myflops = 0.0;
    while (index < nchems) {
        ChemAttribs& attr = chems[index];
        SpeciesT nValue2;
        SpeciesT n0Value2;
        nValue2.h = attr.h;
        nValue2.hp = attr.hp;
        nValue2.hn = attr.hn;
        nValue2.h2 = attr.h2;
        nValue2.he = attr.he;    // 2
        nValue2.hep = attr.hep;
        nValue2.hepp = attr.hepp;
        if (nValue2.h < 0.0) {
            if (nValue2.h < -5.0e-7) {
                PRINT("Negative H !\n", nValue2.h);
            } else {
                nValue2.hp += nValue2.h;
                nValue2.h = 0.f;
            }
        }
        double dt = (double) attr.dt * (double) params.a;
        dt *= (double) params.codeToS;    // 1
        double rho = (double) attr.rho * (double) codeToDensity * pow((double) params.a, -3.0);
        float const hotMass = 1.0f - attr.coldMass;
        rho *= hotMass;
        double const rhoavo = rho * constants::kAvo;    // 1
        nValue2.h *= rhoavo;                            // 1
        nValue2.hp *= rhoavo;                           // 1
        nValue2.hn *= rhoavo;                           // 1
        nValue2.h2 *= 0.5 * rhoavo;                     // 2
        nValue2.he *= 0.25 * rhoavo;                    // 2
        nValue2.hep *= 0.25 * rhoavo;                   // 2
        nValue2.hepp *= 0.25 * rhoavo;                  // 2
        double n0 = (double) nValue2.h + (double) nValue2.hp + (double) nValue2.hn +
            (double) nValue2.h2 + (double) nValue2.he + (double) nValue2.hep +
            (double) nValue2.hepp;    // 8
        double n = (double) nValue2.h + 2.0 * (double) nValue2.hp + (double) nValue2.h2 +
            (double) nValue2.he + 2.0 * (double) nValue2.hep + 3.0 * (double) nValue2.hepp;
        n *= hotMass;
        double cv = (1.50 + (double) nValue2.h2 / n0);
        cv = 1.5;    // 4
                     //		double gamma = 1.0 + 1.0 / cv;
                     //// 5
        double gamma = 5.0 / 3.0;
        cv *= constants::kB;    // 1
        double eint = attr.eint;
        eint *= codeToEnergy;
        eint /= sqr(params.a);
        double t0 = (eint * rho) / (n * cv);
        double tmax = TMAX;
        float tmin = TMIN;
        n0Value2 = nValue2;
        double z = 1.0 / (double) params.a - 1.0;    // 2
        double tmid;
        float dedt;
        float dedt0;

        auto const testInstability = [rho, dt, z, &flops](float rhoTh, float t, SpeciesT nValue) {
            if (rhoTh < 1.0e-37f) {
                return 1.0f;
            }
            float const factor = rhoTh / rho;
            nValue.h *= factor;
            nValue.hp *= factor;
            nValue.hn *= factor;
            nValue.h2 *= factor;
            nValue.he *= factor;
            nValue.hep *= factor;
            nValue.hepp *= factor;
            float t1 = t;
            float t2 = 1.01f * t;
            SpeciesT n0Value2 = nValue;
            float dedt1;
            float dedt2;
            testTemperature(n0Value2, nValue, t1, t1, 0.f, z, flops, &dedt1, false);
            testTemperature(n0Value2, nValue, t2, t2, 0.f, z, flops, &dedt2, false);
            if (-dedt2 / t2 <= -dedt1 / t1) {
                return 1.0f;
            } else {
                return 0.0f;
            }
        };
        bool const unstable = testInstability(rho, t0, nValue2) > 0.f;
        testTemperature(n0Value2, nValue2, t0, t0, dt, z, flops, &dedt0, false);
        float const tcool = -eint * rho / (dedt0 * (params.a));
        // const float tdyn = sqrt(double(4.0 / 3.0 * M_PI) / (double(params.G) * double(rho))) /
        // params.a;
        /*	if (params.stars && unstable && dedt0 < 0.f) {
                float hot_mass = 1.f - attr.cold_mass;
                float hotmass0 = hot_mass;
                float factor = expf(-dt / tcool);
                hot_mass = fmaxf(hot_mass * factor, 0.5f * (2.5e-7f + hot_mass));
                attr.cold_mass = 1.f - hot_mass;
                ALWAYS_ASSERT(attr.cold_mass < 1.f);
                const double rhoavoinv = 1.0 / rhoavo;
           // 4 N.H *= (double) rhoavoinv;
           // 1 N.Hp *= (double) rhoavoinv;
           // 1 N.Hn *= (double) rhoavoinv;
           // 1 N.H2 *= 2.0 * (double) rhoavoinv;
           // 1 N.He *= 4.0 * (double) rhoavoinv;
           // 1 N.Hep *= 4.0 * (double) rhoavoinv;
           // 1 N.Hepp *= 4.0 * (double) rhoavoinv; attr.H = N.H;
           // 1 attr.H2 = N.H2; attr.Hep = N.Hep; attr.Hepp = N.Hepp; attr.Hn = N.Hn; attr.Hp =
           N.Hp; attr.He = N.He; } else {*/
        for (int i = 0; i < 27; i++) {
            float fMid, fMax;
            tmid = sqrtf(tmax * tmin);    // 5
            nValue2 = nValue2.numberDensityToFractions();
            nValue2 = n0Value2;
            if (i == 0) {
                fMax = testTemperature(n0Value2, nValue2, t0, tmax, dt, z, flops, &dedt, false);
            }
            fMid = testTemperature(n0Value2, nValue2, t0, tmid, dt, z, flops, &dedt, false);
            if (copysignf(1.f, fMid) != copysignf(1.f, fMax)) {
                tmin = tmid;
            } else {
                tmax = tmid;
                fMax = fMid;
            }

            flops += 7;
        }
        tmid = sqrtf(tmax * tmin);
        testTemperature(n0Value2, nValue2, t0, t0, dt, z, flops, &dedt0, false);
        testTemperature(n0Value2, nValue2, t0, tmid, dt, z, flops, &dedt, true);
        dedt = 0.5f * dedt + 0.5f * dedt0;
        float t = tmid;
        n0 = (double) nValue2.h + (double) nValue2.h2 + (double) nValue2.he + (double) nValue2.hep +
            (double) nValue2.hepp + (double) nValue2.hp + (double) nValue2.hn;
        n = (double) nValue2.h + 2.0 * (double) nValue2.hp + (double) nValue2.h2 +
            (double) nValue2.he + 2.0 * (double) nValue2.hep + 3.0 * (double) nValue2.hepp;    // 8
        double const rhoavoinv = 1.0 / rhoavo;                                                 // 4
                                                  //			cv = (1.5 + N.H2 / n0);
        cv = 1.5;                                    // 4
        nValue2.h *= (double) rhoavoinv;             // 1
        nValue2.hp *= (double) rhoavoinv;            // 1
        nValue2.hn *= (double) rhoavoinv;            // 1
        nValue2.h2 *= 2.0 * (double) rhoavoinv;      // 1
        nValue2.he *= 4.0 * (double) rhoavoinv;      // 1
        nValue2.hep *= 4.0 * (double) rhoavoinv;     // 1
        nValue2.hepp *= 4.0 * (double) rhoavoinv;    // 1
        //			gamma = 1.0 + 1.0 / cv;
        //// 5
        gamma = 5.0 / 3.0;
        cv *= constants::kB;
        eint = cv * n * t / rho;    // 1
        eint *= sqr(params.a);
        eint /= codeToEnergy;
        attr.h = nValue2.h;
        attr.h2 = nValue2.h2;
        attr.hep = nValue2.hep;
        attr.hepp = nValue2.hepp;
        attr.hn = nValue2.hn;
        attr.hp = nValue2.hp;
        attr.he = nValue2.he;
        attr.eint = eint;
        attr.tcool = tcool;
        /*		}*/
        //		attr.rho_th = rho_th / code_to_density;
        // PRINT( "%e\n", eint);
        flops += 136;
        myflops += flops;
        flops = 0;
        index = atomicAdd(nextIndex, 1);
    }
    atomicAdd(totalFlops, myflops);
}

double cudaChemistryStep(Vector<ChemAttribs>& chems, float scale) {
    Timer tm;
    static auto const opts = getOptions();
    double* flops;
    int* index;
    ChemAttribs* devChems;
    ChemistryParams params;
    int nblocks;
    auto stream = cudaGetStream();
    for (int i = 0; i < NCOOL; i++) {
        coolingTotals[i] = 0.f;
    }
    CUDA_CHECK(cudaMalloc(&devChems, sizeof(ChemAttribs) * chems.size()));
    CUDA_CHECK(cudaMallocManaged(&flops, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&index, sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(devChems, chems.data(), sizeof(ChemAttribs) * chems.size(),
        cudaMemcpyHostToDevice, stream));
    *index = 0;
    *flops = 0;
    params.a = scale;
    params.codeToCm = opts.codeToCm;
    params.g = constants::kG;
    params.stars = opts.stars;
    params.codeToG = opts.codeToG;
    params.codeToS = opts.codeToS;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nblocks, (void const*) chemistryKernel, CHEM_BLOCK_SIZE, 0));
    nblocks *= cudaSmpCount();
    tm.start();
    chemistryKernel<<<nblocks, CHEM_BLOCK_SIZE, 0, stream>>>(
        params, devChems, chems.size(), index, flops);
    synchronizeCudaStream(stream);
    tm.stop();
    //	PRINT("CHEM GFLOPS = %e\n", *flops / 1024.0 / 1024.0 / 1024.0 / tm.read());
    CUDA_CHECK(cudaMemcpyAsync(chems.data(), devChems, sizeof(ChemAttribs) * chems.size(),
        cudaMemcpyDeviceToHost, stream));
    auto rflops = *flops;
    CUDA_CHECK(cudaFree(flops));
    CUDA_CHECK(cudaFree(index));
    synchronizeCudaStream(stream);
    CUDA_CHECK(cudaFree(devChems));
    cudaEndStream(stream);
    double ctot = 0.0;
    for (int i = 0; i < NCOOL; i++) {
        ctot += coolingTotals[i];
    }
    if (ctot > 0.0) {
        for (int i = 0; i < NCOOL; i++) {
            //		PRINT("C%i = %e\n", i, cooling_totals[i] / ctot);
        }
    }
    return rflops;
}

void testCudaChemistryKernel() {
    /*	static const auto opts = get_options();
     const double code_to_energy_density = opts.code_to_g / (opts.code_to_cm * sqr(opts.code_to_s));
     // 7 const double code_to_density = pow(opts.code_to_cm, -3) * opts.code_to_g; const double Y =
     get_options().Y0;		// 10 PRINT("%e %e %e %e %e\n", opts.code_to_g, opts.code_to_cm,
     opts.code_to_s, code_to_energy_density, code_to_density); const int N = 10000000;
     vector<chem_attribs> chem0;
     vector<chem_attribs> chems;
     float z = 0.5;
     for (int i = 0; i < N; i++) {
     float T = pow(10.0, 3.0 + rand1() * 5.0);
     //		PRINT("%e\n", T);
     float rho = pow(10.0, -(24.0 + 3.0 * rand1())) * (z + 1);
     species_t N;
     N.H = rand1();
     N.H2 = rand1() * 0.1;
     N.Hp = rand1();
     N.Hn = 0.0;
     N.He = rand1();
     N.Hep = rand1();
     N.Hepp = rand1();
     float h0 = (N.H + 2.0 * N.H2 + N.Hp + N.Hn) / (1.0 - opts.Y0);
     float he0 = (N.He + N.Hep + N.Hepp) / opts.Y0;
     N.H /= h0;
     N.H2 /= h0;
     N.Hp /= h0;
     N.Hn /= h0;
     N.He /= he0;
     N.Hep /= he0;
     N.Hepp /= he0;
     chem_attribs chem;
     float n0 = N.H + N.Hp + 0.5f * N.H2 + N.Hn + 0.25f * N.He + 0.25f * N.Hep + 0.25f * N.Hepp;
     float n = (N.H + 2.f * N.Hp + 0.5f * N.H2 + 0.25f * N.He + 0.5f * N.Hep + .75f * N.Hepp) * rho
     * constants::avo; float cv = (1.5 + 0.5f * N.H2 / n0); float gamma = 1.0 + 1.0 / cv; cv *=
     constants::kb; float energy = cv * n * T; double K = (double) energy * (gamma - 1.0) *
     pow((double) rho, -gamma); K /= code_to_energy_density; K *= pow(code_to_density, gamma);
     //		PRINT("!!!!!!!!!!!1 %e %e %e %e\n", energy, gamma, rho, K);
     rho /= code_to_density;
     chem.dt = 1e15 / opts.code_to_s;
     chem.He = N.He;
     chem.H2 = N.H2;
     chem.Hp = N.Hp;
     chem.Hep = N.Hep;
     chem.Hepp = N.Hepp;
     chem.rho = rho;
     chem.K = K;
     chem.Hn = N.Hn;
     chem.rho = rho;
     chems.push_back(chem);
     chem0.push_back(chem);
     }
     PRINT("Starting\n");
     cuda_chemistry_step(chems, 1.0);

     for (int i = 0; i < N; i++) {
     //PRINT("%e %e | %e %e %e | %e %e %e | \n", chem0[i].K, chems[i].K, 1 - Y - chem0[i].Hp -
     chem0[i].H2 * 2 - chem0[i].Hn, chem0[i].Hp, chem0[i].H2,
     //1 - Y - chems[i].Hp - chems[i].H2 * 2 - chems[i].Hn, chems[i].Hp, chems[i].H2);
     }*/
}

CUDA_EXPORT
SpeciesT SpeciesT::numberDensityToFractions() const {
    SpeciesT f;
    f = *this;
    f.he *= 4.0f;
    f.hep *= 4.0f;
    f.hepp *= 4.0f;
    f.h2 *= 2.0f;
    float rho = 0.f;
    for (int i = 0; i < NSPECIES; i++) {
        rho += f.n[i];
    }
    float rhoinv = 1.f / rho;
    for (int i = 0; i < NSPECIES; i++) {
        f.n[i] *= rhoinv;
    }
    return f;
}
