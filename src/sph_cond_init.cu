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

#include <cosmictiger/sph_cuda.hpp>

struct CondInitRecord1 {
    Fixed32 x;
    Fixed32 y;
    Fixed32 z;
    float h;
};
struct CondInitRecord2 {
#ifdef ENTROPY
    float entr;
#else
    float eint;
#endif
    float cfrac;
    float rho;
    char star;
};

struct CondInitWorkspace {
    DeviceVector<CondInitRecord1> rec1;
    DeviceVector<CondInitRecord2> rec2;
    DeviceVector<int> neighbors;
};

__global__ void sphCudaCondInit(
    SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ CondInitWorkspace ws;
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    new (&ws) CondInitWorkspace();
    int flops = 0;
    Array<Fixed32, NDIM> x;
    float const gamma0 = data.defGamma;
    float const codeToEnergy = sqr(params.codeToCm) / sqr(params.codeToS);    // 5
    float const codeToDensity = params.codeToG / pow(params.codeToCm, 3.);    // 12
    float const colog0 =
        log(1.5 * pow(constants::kB, 1.5) * pow(constants::kE, -3) * pow(M_PI, -0.5));    // 35
    float const kappa0 = 20.0 * pow(2.0 / M_PI, 1.5) * pow(constants::kB, 2.5) *
        pow(constants::kMe, -0.5) * pow(constants::kE, -4.0) * params.codeToS * params.codeToCm /
        (params.codeToG * constants::kAvo);    // 49
    float const propc0 = 0.4 * (gamma0 - 1.0) *
        sqrtf(2.0 * constants::kB / (M_PI * constants::kMe)) / constants::kC;    // 17
    float const cv0 = constants::kB / (gamma0 - 1.0f);                           // 5
    float const invgm1 = 1.f / (gamma0 - 1.0);
    flops += 128;
    while (index < data.numberTargetNodes) {
        __syncthreads();
        int flops = 0;
        ws.rec1.resize(0);
        ws.rec2.resize(0);
        SphTreeNode const& self = data.trees[data.selfs[index]];
        for (int ni = self.neighborRange.first; ni < self.neighborRange.second; ni++) {
            SphTreeNode const& other = data.trees[data.neighbors[ni]];
            int const maxpi = roundUp(other.partRange.second - other.partRange.first, blockSize) +
                other.partRange.first;
            for (int pi = other.partRange.first + tid; pi < maxpi; pi += blockSize) {
                bool contains = false;
                int j;
                int total;
                float h;
                int rung;
                if (pi < other.partRange.second) {
                    h = data.h[pi];
                    rung = data.rungs[pi];
                    if (rung >= params.minRung) {
                        x[XDIM] = data.x[pi];
                        x[YDIM] = data.y[pi];
                        x[ZDIM] = data.z[pi];
                        contains = (self.outerBox.contains(x));
                        if (!contains) {
                            contains = true;
                            float const& h = data.h[pi];
                            for (int dim = 0; dim < NDIM; dim++) {
                                if (distance(x[dim], self.innerBox.begin[dim]) + h < 0.f) {
                                    flops += 3;
                                    contains = false;
                                    break;
                                }
                                if (distance(self.innerBox.end[dim], x[dim]) + h < 0.f) {
                                    flops += 3;
                                    contains = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                j = contains;
                computeIndices<COND_INIT_BLOCK_SIZE>(j, total);
                int const offset = ws.rec1.size();
                __syncthreads();
                int const nextSize = offset + total;
                ws.rec1.resize(nextSize);
                ws.rec2.resize(nextSize);
                if (contains) {
                    int const k = offset + j;
                    ws.rec1[k].x = x[XDIM];
                    ws.rec1[k].y = x[YDIM];
                    ws.rec1[k].z = x[ZDIM];
                    ws.rec1[k].h = h;
#ifdef ENTROPY
                    ws.rec2[k].entr = data.entr[pi];
#else
                    ws.rec2[k].eint = data.eint[pi];
#endif
                    ws.rec2[k].rho = data.rho[pi];
                    if (params.stars) {
                        ws.rec2[k].cfrac = data.coldFrac[pi];
                        ws.rec2[k].star = data.stars[pi];
                    } else {
                        ws.rec2[k].cfrac = 0.f;
                        ws.rec2[k].star = false;
                    }
                }
            }
        }
        float const& m = data.m;
        flops++;
        for (int i = self.partRange.first; i < self.partRange.second; i++) {
            __syncthreads();
            int const snki = self.sinkPartRange.first - self.partRange.first + i;
            bool const active = data.rungs[i] >= params.minRung;
            bool const semiactive = !active && data.saSnk[snki];
            flops += 2;
            if (semiactive || active) {
                float cfracI = 0.f;
                bool starI = false;
                if (params.stars) {
                    cfracI = data.coldFrac[i];
                    starI = data.stars[i];
                }
                float const& rhoI = data.rho[i];    // 2
                float const& hI = data.h[i];
#ifdef ENTROPY
                float const& aI = data.entr[i];
#else
                float const& eintI = data.eint[i];
#endif
                auto const& xI = data.x[i];
                auto const& yI = data.y[i];
                auto const& zI = data.z[i];
                float const hfracI = 1.f - cfracI;
                float const hinvI = 1.f / hI;
                float const h2I = sqr(hI);
                float const h3invI = (sqr(hinvI) * hinvI);    // 2
#ifdef ENTROPY
                float const eintI = aI * powf(rhoI * hfracI, gamma0 - 1.0f);    // 11
#endif
                float gradx = 0.0f;
                float grady = 0.0f;
                float gradz = 0.0f;
                float const omegaI = data.omegaSnk[snki];
                flops += 11;
                ws.neighbors.resize(0);
                float const jmax = roundUp(ws.rec1.size(), blockSize);
                for (int j = tid; j < jmax; j += blockSize) {
                    int k;
                    int total;
                    bool use = false;
                    if (j < ws.rec1.size()) {
                        auto const rec1 = ws.rec1[j];
                        Fixed32 const xJ = rec1.x;
                        Fixed32 const yJ = rec1.y;
                        Fixed32 const zJ = rec1.z;
                        float const hJ = rec1.h;
                        float const xIj = distance(xI, xJ);     // 1
                        float const yIj = distance(yI, yJ);     // 1
                        float const zIj = distance(zI, zJ);     // 1
                        float const r2 = sqr(xIj, yIj, zIj);    // 5
                        if (r2 < fmaxf(sqr(hI), sqr(hJ))) {     // 4
                            use = true;
                        }
                        flops += 12;
                    }
                    k = use;
                    computeIndices<COND_INIT_BLOCK_SIZE>(k, total);
                    int const offset = ws.neighbors.size();
                    __syncthreads();
                    int const nextSize = offset + total;
                    ws.neighbors.resize(nextSize);
                    if (use) {
                        int const l = offset + k;
                        ws.neighbors[l] = j;
                    }
                }
                __syncthreads();
                if (!starI) {
                    for (int j = tid; j < ws.neighbors.size(); j += COND_INIT_BLOCK_SIZE) {
                        int const kk = ws.neighbors[j];
                        auto const& rec1 = ws.rec1[kk];
                        auto const& rec2 = ws.rec2[kk];
                        auto const& starJ = rec2.star;
                        if (!starJ) {
#ifdef ENTROPY
                            float const& rhoJ = rec2.rho;    // 2
                            float const& aJ = rec2.entr;
#else
                            float const& eintJ = rec2.eint;
#endif
                            float const& cfracJ = rec2.cfrac;
                            float const hfracJ = 1.f - cfracJ;
                            Fixed32 const xJ = rec1.x;
                            Fixed32 const yJ = rec1.y;
                            Fixed32 const zJ = rec1.z;
                            float const xIj = distance(xI, xJ);     // 1
                            float const yIj = distance(yI, yJ);     // 1
                            float const zIj = distance(zI, zJ);     // 1
                            float const r2 = sqr(xIj, yIj, zIj);    // 5
                            flops += 10;
                            if (r2 < h2I && r2 != 0.0f) {    // 2
                                float const hJ = rec1.h;
                                float const r = sqrtf(r2);                  // 4
                                float const rinv = 1.f / r;                 // 4
                                float const q = r * hinvI;                  // 1
                                float const hinvJ = 1.f / hJ;               // 4
                                float const h3invJ = sqr(hinvJ) * hinvJ;    // 2
#ifdef ENTROPY
                                float const eintJ =
                                    aJ * powf(rhoJ * hfracJ, gamma0 - 1.0f);    // 11
#endif
                                float const dwdq = dkernelWDq(q);
                                float const dWdrI = dwdq * h3invI * hinvI / omegaI;    // 3
                                float const tmp = dWdrI * rinv * (eintJ - eintI);      // 14
                                gradx = fmaf(tmp, xIj, gradx);                         // 2
                                grady = fmaf(tmp, yIj, grady);                         // 2
                                gradz = fmaf(tmp, zIj, gradz);                         // 2
                                flops += 49;
                            }
                        }
                    }
                    float const tmp = m / (params.a * rhoI * eintI);    // 5
                    gradx *= tmp;                                       // 1
                    grady *= tmp;                                       // 1
                    gradz *= tmp;                                       // 1
                    flops += 8;
                    sharedReduceAdd<float, COND_INIT_BLOCK_SIZE>(gradx);    // 31
                    sharedReduceAdd<float, COND_INIT_BLOCK_SIZE>(grady);    // 31
                    sharedReduceAdd<float, COND_INIT_BLOCK_SIZE>(gradz);    // 31
                    if (tid == 0) {
                        flops += 93;
                        auto const& fracI = data.rec1Snk[snki].frac;
#ifdef ENTROPY
                        float const& aI = data.rec2Snk[snki].entropy;
#else
                        float const& eintI = data.rec2Snk[snki].eint;
#endif
                        float const& hValue = fracI[CHEM_H];
                        float const& hp = fracI[CHEM_HP];
                        float const& hn = fracI[CHEM_HN];
                        float const& h2 = fracI[CHEM_H2];
                        float const& he = fracI[CHEM_HE];
                        float const& hep = fracI[CHEM_HEP];
                        float const& hepp = fracI[CHEM_HEPP];
                        float const grad2 = sqr(gradx, grady, gradz);                     // 5
                        float const gradToT = sqrtf(grad2);                               // 4
                        float const rho0 = rhoI * hfracI / (sqr(params.a) * params.a);    // 7
                        float n0 = (hValue +
                            fmaf(2.f, hp,
                                fmaf(.5f, h2,
                                    fmaf(.25f, he, fmaf(.5f, hep, .75f * hepp)))));    // 10
                        float const mmwI = 1.0f / n0;                                  // 4
                        float const neI = fmaxf((hp - hn + fmaf(0.25f, hep, 0.5f * hepp)) * rho0 *
                                (constants::kAvo * codeToDensity),
                            1e-30f);    // 8
#ifdef ENTROPY
                        float const eint =
                            codeToEnergy * aI * powf(rho0 * hfracI, gamma0 - 1.0) * invgm1;    // 13
#else
                        float const eint = eintI;
#endif
                        float const tI = mmwI * eint / (cv0 * constants::kAvo);              // 6
                        float const cologI = colog0 + 1.5f * logf(tI) - 0.5f * logf(neI);    // 20
                        float kappaI =
                            mmwI * (gamma0 - 1.f) * kappa0 * powf(tI, 2.5f) / cologI;    // 15
                        float const sigmaxI = propc0 * sqrtf(tI);                        // 5
                        float const rValue =
                            2.f * mmwI * kappaI * gradToT / (rhoI * sigmaxI);    // 7
                        float const phi = (2.f + 3.f * rValue) /
                            (2.f + 3.f * rValue + 3.f * sqr(rValue));    // 11
                        kappaI *= phi;                                   // 1
                        ALWAYS_ASSERT(kappaI >= 0.0f);
                        data.kapSnk[snki] = kappaI;
#ifdef ENTROPY
                        data.entr0Snk[snki] = aI;
#else
                        data.eint0Snk[snki] = eintI;
#endif
                    }
                }
            }
        }
        sharedReduceAdd<int, COND_INIT_BLOCK_SIZE>(flops);
        if (tid == 0) {
            atomicAdd(&reduce->flops, (float) flops);
            index = atomicAdd(&reduce->counter, 1);
        }
        flops = 0;
        __syncthreads();
    }
    (&ws)->~CondInitWorkspace();
}
