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

struct HydroRecord1 {
    Fixed32 x;
    Fixed32 y;
    Fixed32 z;
    float h;
};

struct HydroRecord2 {
    Array<float, NCHEMFRACS> chem;
    float vx;
    float vy;
    float vz;
#ifdef ENTROPY
    float entr;
#else
    float eint;
#endif
    float alpha;
    float omega;
    float omegaP;
    float pre;
    float shearv;
    float coldFrac;
    float kappa;
    float rho;
    char star;
};

struct HydroWorkspace {
    DeviceVector<HydroRecord1> rec1;
    DeviceVector<HydroRecord2> rec2;
    DeviceVector<int> neighbors;
};

#ifdef ENTROPY
__device__ inline float pressure(float a, float rho, float gamma) {
    return a * powf(rho, gamma);
}
#else
__device__ inline float pressure(float eint, float rho, float gamma) {
    return (gamma - 1.0f) * eint * rho;
}
#endif

__global__ void sphCudaHydro(SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ HydroWorkspace ws;
    new (&ws) HydroWorkspace();
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    Array<Fixed32, NDIM> x;
    float totalVsigMax = 0.;
    int maxRungHydro = 0;
    int maxRungGrav = 0;
    int maxRung = 0;
    int flops = 0;
    float const ainv = 1.0f / params.a;
    float const gamma0 = data.defGamma;
    float const invgamma0 = 1.f / gamma0;
    float const log2fparamst0 = log2f(params.t0);
    flops += 16;
    while (index < data.numberTargetNodes) {
        __syncthreads();
        SphTreeNode const& self = data.trees[data.selfs[index]];
        ws.rec1.resize(0);
        ws.rec2.resize(0);
        for (int ni = self.neighborRange.first; ni < self.neighborRange.second; ni++) {
            SphTreeNode const& other = data.trees[data.neighbors[ni]];
            int const maxpi = roundUp(other.partRange.second - other.partRange.first, blockSize) +
                other.partRange.first;
            for (int pi = other.partRange.first + tid; pi < maxpi; pi += blockSize) {
                bool contains = false;
                int j;
                int total;
                if (pi < other.partRange.second) {
                    x[XDIM] = data.x[pi];
                    x[YDIM] = data.y[pi];
                    x[ZDIM] = data.z[pi];
                    if (self.outerBox.contains(x)) {
                        contains = true;
                    }
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
                j = contains;
                computeIndices<HYDRO_BLOCK_SIZE>(j, total);
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
                    ws.rec1[k].h = data.h[pi];
                    ws.rec2[k].vx = data.vx[pi];
                    ws.rec2[k].vy = data.vy[pi];
                    ws.rec2[k].vz = data.vz[pi];
#ifdef ENTROPY
                    ws.rec2[k].entr = data.entr[pi];
#else
                    ws.rec2[k].eint = data.eint[pi];
#endif
                    ws.rec2[k].alpha = data.alpha[pi];
                    ws.rec2[k].omega = data.omega[pi];
                    ws.rec2[k].omegaP = data.omegaP[pi];
                    ws.rec2[k].rho = data.rho[pi];
                    ws.rec2[k].pre = data.pre[pi];
                    if (params.diffusion && params.phase == 1) {
                        ws.rec2[k].shearv = data.shearv[pi];
                    }
                    if (params.stars) {
                        ws.rec2[k].coldFrac = data.coldFrac[pi];
                        ws.rec2[k].star = data.stars[pi];
                    } else {
                        ws.rec2[k].coldFrac = 0.f;
                        ws.rec2[k].star = false;
                    }
                    if (params.diffusion & params.phase == 1) {
                        if (data.chemistry) {
                            ws.rec2[k].chem = data.chem[pi];
                        }
                    }
                }
            }
        }
        for (int i = self.partRange.first; i < self.partRange.second; i++) {
            __syncthreads();
            int const snki = self.sinkPartRange.first - self.partRange.first + i;
            int rungI = data.rungsSnk[data.dmIndexSnk[snki]];
            bool use = rungI >= params.minRung;
            float const& m = data.m;
            float const minv = 1.f / m;
            if (use) {
                float difcoI;
                float cfracI;
                auto const& xI = data.x[i];
                auto const& yI = data.y[i];
                auto const& zI = data.z[i];
                float const& omegaI = data.omega[i];
                float const& omegaPI = data.omegaP[i];
                auto const& vxI = data.vx[i];
                auto const& vyI = data.vy[i];
                auto const& vzI = data.vz[i];
                float const& hI = data.h[i];
#ifdef ENTROPY
                float const& aI = data.entr[i];
#else
                float const& eintI = data.eint[i];
#endif
                float const rhoI = data.rho[i];    // 2
                float const alphaI = data.alpha[i];
                float const hinvI = 1.f / hI;                 // 4
                float const h3invI = (sqr(hinvI) * hinvI);    // 3
                bool starI;
                if (params.stars) {
                    cfracI = data.coldFrac[i];
                    starI = data.stars[i];
                } else {
                    cfracI = 0.f;
                    starI = false;
                }
                float const hfracI = 1.f - cfracI;
                float const rhoinvI = 1.f / rhoI;    // 4
                Array<float, NCHEMFRACS> fracI;
                if ((params.diffusion && params.phase == 1) && data.chemistry) {
                    fracI = data.chem[i];
                }
                if (params.diffusion && params.phase == 1) {
                    float const shearvI = data.shearv[i];
                    difcoI = SPH_DIFFUSION_C * sqr(hI) * shearvI;    // 3
                }
#ifdef HOPKINS
                float const& preI = data.pre[i];
#else
#ifdef ENTROPY
                float const preI = pressure(aI, rhoI, gamma0);
#else
                float const preI = pressure(eintI, rhoI, gamma0);
#endif
#endif
#ifdef ENTROPY
                float const eint = aI * powf(rhoI, gamma0 - 1.f) / (gamma0 - 1.f);
                float const deDt0 = aI / eint;    // 12
                float const cI = sqrtf(
                    gamma0 * powf(aI, 1.0f / gamma0) * powf(preI, gamma0 - 1.f) / hfracI);    // 22
#else
                float const deDt0 = 1.f;                                    // 12
                float const cI = sqrtf(gamma0 * (gamma0 - 1.f) * eintI);    // 22
#endif
                flops += 55;
                float ax = 0.f;
                float ay = 0.f;
                float az = 0.f;
                float deDt1 = 0.f;
                float deDt2 = 0.f;
                float dcmDt = 0.f;
                Array<float, NCHEMFRACS> dfracDt;
                if (data.chemistry) {
                    for (int fi = 0; fi < NCHEMFRACS; fi++) {
                        dfracDt[fi] = 0.f;
                    }
                }
                float dtinvCfl = 0.f;
                float dtinvVisc = 0.f;
                //				float one = 0.0f;
                float const& adot = params.adot;
                float dd = 0.f;
                ws.neighbors.resize(0);
                float const jmax = roundUp(ws.rec1.size(), blockSize);
                for (int j = tid; j < jmax; j += blockSize) {
                    int k;
                    int total;
                    bool use = false;
                    if (j < ws.rec1.size()) {
                        auto const& rec1 = ws.rec1[j];
                        Fixed32 const& xJ = rec1.x;
                        Fixed32 const& yJ = rec1.y;
                        Fixed32 const& zJ = rec1.z;
                        float const& hJ = rec1.h;
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
                    computeIndices<HYDRO_BLOCK_SIZE>(k, total);
                    int const offset = ws.neighbors.size();
                    __syncthreads();
                    int const nextSize = offset + total;
                    ws.neighbors.resize(nextSize);
                    if (use) {
                        int const l = offset + k;
                        ws.neighbors[l] = j;
                    }
                }
                float dvxDx = 0.f;
                float dvxDy = 0.f;
                float dvxDz = 0.f;
                float dvyDx = 0.f;
                float dvyDy = 0.f;
                float dvyDz = 0.f;
                float dvzDx = 0.f;
                float dvzDy = 0.f;
                float dvzDz = 0.f;
                float vsig = 0.f;
                __syncthreads();
                if (!starI) {
                    for (int j = tid; j < ws.neighbors.size(); j += blockSize) {
                        int const kk = ws.neighbors[j];
                        auto const& rec1 = ws.rec1[kk];
                        auto const& rec2 = ws.rec2[kk];
                        auto const& starJ = rec2.star;
                        if (!starJ) {
                            float const& hJ = rec1.h;
                            Fixed32 const& xJ = rec1.x;
                            Fixed32 const& yJ = rec1.y;
                            Fixed32 const& zJ = rec1.z;
                            float const& cfracJ = rec2.coldFrac;
                            float const& shearvJ = rec2.shearv;
                            auto const& fracJ = rec2.chem;
                            float const& vxJ = rec2.vx;
                            float const& vyJ = rec2.vy;
                            float const& vzJ = rec2.vz;
                            float const& omegaJ = rec2.omega;
                            float const& omegaPJ = rec2.omegaP;
                            float const& alphaJ = fmaxf(rec2.alpha, params.alpha0);
                            float const& rhoJ = rec2.rho;    // 2
#ifdef ENTROPY
                            float const& aJ = rec2.entr;
#else
                            float const& eintJ = rec2.eint;
#endif
#ifdef HOPKINS
                            float const& preJ = rec2.pre;
#ifdef ENTROPY
                            float const fIj = (1.f -
                                smoothX(hJ, params.hmin, params.hmax) * powf(aJ, -1.0f / gamma0) /
                                    omegaPI);
                            float const fJi = (1.f -
                                smoothX(hI, params.hmin, params.hmax) * powf(aI, -1.0f / gamma0) /
                                    omegaPJ);
#else
                            float const fIj = (1.f -
                                smoothX(hJ, params.hmin, params.hmax) /
                                    ((gamma0 - 1.0f) * eintJ * omegaPI));
                            float const fJi = (1.f -
                                smoothX(hI, params.hmin, params.hmax) /
                                    ((gamma0 - 1.0f) * eintI * omegaPJ));
#endif
#else
#ifdef ENTROPY
                            float const preJ = pressure(aJ, rhoJ, gamma0);
#else
                            float const preJ = pressure(eintJ, rhoJ, gamma0);
#endif
#endif
                            float const hfracJ = 1.f - cfracJ;
                            float const hinvJ = 1.f / hJ;               // 4
                            float const h3invJ = sqr(hinvJ) * hinvJ;    // 3
                            float const xIj = distance(xI, xJ);         // 1
                            float const yIj = distance(yI, yJ);         // 1
                            float const zIj = distance(zI, zJ);         // 1
                            float const r2 = sqr(xIj, yIj, zIj);        // 5
                            float const r = sqrtf(r2);                  // 4
                            float const qI = r * hinvI;                 // 1
                            float const qJ = r * hinvJ;                 // 1
#ifdef ENTROPY
                            float const cJ = sqrtf(gamma0 * powf(aJ, 1.0f / gamma0) *
                                powf(preJ, gamma0 - 1.f));    // 23
#else
                            float const cJ = sqrtf(gamma0 * (gamma0 - 1.f) * eintJ);    // 23
#endif
                            float const vx0Ij = vxI - vxJ;                      // 3
                            float const vy0Ij = vyI - vyJ;                      // 3
                            float const vz0Ij = vzI - vzJ;                      // 3
                            float const vxIj = vx0Ij + xIj * adot;              // 3
                            float const vyIj = vy0Ij + yIj * adot;              // 3
                            float const vzIj = vz0Ij + zIj * adot;              // 3
                            float const rinv = 1.0f / (r > 0.f ? r : 1e37f);    // 5
                            float const vdotxIj =
                                fmaf(xIj, vxIj, fmaf(yIj, vyIj, zIj * vzIj));        // 5
                            float const hIj = 0.5f * (hI + hJ);                      // 2
                            float const wIj = fminf(vdotxIj * rinv, 0.f);            // 2
                            float const muIj = wIj * hIj / sqrtf(r2 + ETA * hIj);    // 12
                            float const rhoIj = 0.5f * (rhoI + rhoJ);                // 2
                            float const cIj = 0.5f * (cI + cJ);                      // 2
                            float const hfracIj =
                                2.0f * hfracI * hfracJ / (hfracI + hfracJ + 1e-36f);
                            float const alphaIj = 0.5f * hfracIj * (alphaI + alphaJ);     // 2
                            float const vsigIj = alphaIj * (cIj - params.beta * muIj);    // 4
                            float const dWdrI = dkernelWDq(qI) * hinvI * h3invI;          // 2
                            float const dWdrJ = dkernelWDq(qJ) * hinvJ * h3invJ;          // 2
                            //				ALWAYS_ASSERT(omega_i > 0.f);
                            if (omegaJ <= 0.0) {
                                PRINT("%i %e %e %e\n", data.selfs[index], omegaJ, qI, qJ);
                            }
                            ALWAYS_ASSERT(omegaJ > 0.f);
                            float const dWdrIj = 0.5f * (dWdrI + dWdrJ);    // 4
                            float const dWdrIRinv = dWdrI * rinv;           // 2
                            float const dWdrJRinv = dWdrJ * rinv;           // 2
                            float const dWdrIjRinv = dWdrIj * rinv;         // 1
                            float const dWdrXIj = xIj * dWdrIjRinv;         // 1
                            float const dWdrYIj = yIj * dWdrIjRinv;         // 1
                            float const dWdrZIj = zIj * dWdrIjRinv;         // 1
                            float const dWdrXI = dWdrIRinv * xIj;           // 1
                            float const dWdrYI = dWdrIRinv * yIj;           // 1
                            float const dWdrZI = dWdrIRinv * zIj;           // 1
                            float const dWdrXJ = dWdrJRinv * xIj;           // 1
                            float const dWdrYJ = dWdrJRinv * yIj;           // 1
                            float const dWdrZJ = dWdrJRinv * zIj;           // 1
                            float const mainv = m * ainv;                   // 1
#ifdef HOPKINS
#ifdef ENTROPY
                            float const dpI = fIj * mainv * powf(aI * aJ, 1.f / gamma0) *
                                powf(preI, 1.f - 2.f / gamma0);
                            float const dpJ = fJi * mainv * powf(aJ * aI, 1.f / gamma0) *
                                powf(preJ, 1.f - 2.f / gamma0);
#else
                            float const dpI =
                                fIj * mainv * sqr(gamma0 - 1.f) * eintI * eintJ / preI;
                            float const dpJ =
                                fJi * mainv * sqr(gamma0 - 1.f) * eintI * eintJ / preJ;
#endif
#else
                            float const dpI = mainv * preI / sqr(rhoI) / omegaI;
                            float const dpJ = mainv * preJ / sqr(rhoJ) / omegaJ;
#endif
                            float const piIj = -mainv * wIj * hIj * rinv * vsigIj / rhoIj;    // 9
                            ax -= dpI * dWdrXI + dpJ * dWdrXJ;                                // 4
                            ay -= dpI * dWdrYI + dpJ * dWdrYJ;                                // 4
                            az -= dpI * dWdrZI + dpJ * dWdrZJ;                                // 4
                            ax -= piIj * dWdrXIj;                                             // 2
                            ay -= piIj * dWdrYIj;                                             // 2
                            az -= piIj * dWdrZIj;                                             // 2
                            float const vdWIj =
                                fmaf(vxIj, dWdrXIj, fmaf(vyIj, dWdrYIj, vzIj * dWdrZIj));    // 5
                            deDt2 = fmaf(deDt0, 0.5f * piIj * vdWIj, deDt2);                 // 4
#ifndef ENTROPY
                            float const v0dWI =
                                fmaf(vx0Ij, dWdrXI, fmaf(vy0Ij, dWdrYI, vz0Ij * dWdrZI));
#ifdef HOPKINS    // 5
                            deDt2 = fmaf(mainv * fIj * v0dWI,
                                sqr(gamma0 - 1.f) * eintI * eintJ / preI, deDt2);    // 4
#else
                            deDt2 = fmaf(
                                mainv * v0dWI, (gamma0 - 1.f) * eintI * rhoinvI / omegaI, deDt2);
#endif
#endif
                            dvxDx -= m * rhoinvI * vxIj * dWdrXI;    // 2
                            dvyDy -= m * rhoinvI * vyIj * dWdrYI;    // 2
                            dvzDz -= m * rhoinvI * vzIj * dWdrZI;    // 2
                            vsig = fmaxf(vsig, -wIj);                // 1
                            if (params.phase == 0) {
                                dvyDx -= m * rhoinvI * vyIj * dWdrXI;    // 2
                                dvzDx -= m * rhoinvI * vzIj * dWdrXI;    // 2
                                dvxDy -= m * rhoinvI * vxIj * dWdrYI;    // 2
                                dvzDy -= m * rhoinvI * vzIj * dWdrYI;    // 2
                                dvxDz -= m * rhoinvI * vxIj * dWdrZI;    // 2
                                dvyDz -= m * rhoinvI * vyIj * dWdrZI;    // 2
                            } else if (params.phase == 1) {
                                float const h = fminf(hI, hJ * (1 << MAX_RUNG_DIF));    // 3
                                dtinvCfl = fmaxf(dtinvCfl, cIj / h);                    // 1
                                dtinvVisc = fmaxf(dtinvVisc, 0.6f * vsigIj / h);
                                if (params.diffusion) {
                                    flops += 29;
                                    float difMax = 0.f;
                                    float const difcoJ =
                                        SPH_DIFFUSION_C * sqr(hJ) * shearvJ;           // 3
                                    float const difcoIj = 0.5f * (difcoI + difcoJ);    // 2
                                    float const dIj =
                                        -2.f * m / rhoIj * difcoIj * dWdrIj * rinv;    // 7
                                    ALWAYS_ASSERT(dIj >= 0.f);
#ifdef ENTROPY
                                    float const a0J =
                                        aJ * powf(rhoJ * rhoinvI / hfracI * hfracJ, gamma0 - 1.f);
                                    difMax = fabs(a0J - aI) / fmaxf(a0J, aI);
                                    deDt1 -= dIj * (aI - a0J);    // 16;
#else
                                    difMax = fabs(eintJ - eintI) / fmaxf(eintJ, eintI);
                                    deDt1 -= dIj * (eintI - eintJ);    // 16;
#endif
                                    if (params.stars) {
                                        dcmDt -= dIj * (cfracI - cfracJ);    // 3
                                        difMax = fmaxf(
                                            difMax, fabs(cfracJ - cfracI) / fmaxf(cfracJ, cfracI));
                                        flops += 3;
                                    }
                                    if (data.chemistry) {
                                        for (int fi = 0; fi < NCHEMFRACS; fi++) {
                                            ALWAYS_ASSERT(fracJ[fi] >= 0.f);
                                            dfracDt[fi] -= dIj * (fracI[fi] - fracJ[fi]);    // 5
                                            difMax = fmaxf(difMax,
                                                fabs(fracI[fi] - fracJ[fi]) /
                                                    fmaxf(fracI[fi], fracJ[fi]));
                                            flops += 5;
                                        }
                                    }
                                    dd += dIj * difMax;    // 1
                                }
                            }
                        }
                    }
                }
                if (params.phase == 0) {
                    sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvxDy);    // 127
                    sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvxDz);    // 127
                    sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvyDz);    // 127
                    sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvzDx);    // 127
                    sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvzDy);    // 127
                    sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvyDx);    // 127
                } else if (params.phase == 1) {
                    sharedReduceMax<HYDRO_BLOCK_SIZE>(dtinvCfl);        // 31
                    sharedReduceMax<HYDRO_BLOCK_SIZE>(dtinvVisc);       // 31
                    sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(deDt1);    // 31
                }
                sharedReduceMax<HYDRO_BLOCK_SIZE>(vsig);            // 31
                sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvxDx);    // 127
                sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvyDy);    // 127
                sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dvzDz);    // 127
                sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(deDt2);    // 31
                sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(ax);       // 31
                sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(ay);       // 31
                sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(az);       // 31
                if (params.diffusion && params.phase == 1) {
                    sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dd);
                    if (params.stars) {
                        sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dcmDt);    // 31
                    }
                    if (data.chemistry) {
                        for (int fi = 0; fi < NCHEMFRACS; fi++) {
                            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(dfracDt[fi]);    // 31
                        }
                    }
                }
                if (tid == 0) {
                    float gxI;
                    float gyI;
                    float gzI;
                    if (data.gravity) {
                        gxI = data.gxSnk[snki];
                        gyI = data.gySnk[snki];
                        gzI = data.gzSnk[snki];
                    } else {
                        gxI = 0.f;
                        gyI = 0.f;
                        gzI = 0.f;
                    }
                    ax += gxI;    // 1
                    ay += gyI;    // 1
                    az += gzI;    // 1
                    data.rec6Snk[snki].dvel[XDIM] = ax;
                    data.rec6Snk[snki].dvel[YDIM] = ay;
                    data.rec6Snk[snki].dvel[ZDIM] = az;
#ifdef ENTROPY
                    if (params.phase == 1) {
                        data.dentr1Snk[snki] = deDt1;
                    }
                    data.dentr2Snk[snki] = deDt2;
#else
                    if (params.phase == 1) {
                        data.deint1Snk[snki] = deDt1;
                    }
                    data.deint2Snk[snki] = deDt2;
#endif
                    if (params.stars) {
                        data.rec5Snk[snki].dfcold = dcmDt;
                    }
                    if (data.chemistry) {
                        data.rec5Snk[snki].dfrac = dfracDt;
                    }
                    if (params.phase == 0 && params.tau > 0.f) {
                        float divV, curlVx, curlVy, curlVz;
                        divV = dvxDx + dvyDy + dvzDz;    // 2
                        curlVx = dvzDy - dvyDz;          // 1
                        curlVy = -dvzDx + dvxDz;         // 2
                        curlVz = dvyDx - dvxDy;
                        char& rung = data.rungsSnk[data.dmIndexSnk[snki]];
                        float const dt1 = params.t0 * rungDt[rung];
                        float const curlv = sqrtf(sqr(curlVx, curlVy, curlVz));    // 9
                        float const divV0 = data.divvSnk[snki];
                        float& alpha = data.rec1Snk[snki].alpha;
                        float const ddivvDt =
                            (divV - divV0) / dt1 - params.adot * ainv * divV;    // 8
                        float const limiter = sqr(divV) /
                            (sqr(divV) + sqr(curlv) + 1.0e-4f * sqr(cI / hI * ainv));    // 14
                        float const s =
                            limiter * sqr(hI) * fmaxf(0.f, -ddivvDt) * sqr(params.a);    // 6
                        vsig += cI;
                        float const alphaTarg = params.alpha0 +
                            (params.alpha1 - params.alpha0) * s / (s + sqr(cI));          // 8
                        float const lambda0 = params.alphaDecay * vsig * hinvI * ainv;    // 3
                        float const lambda1 = vsig * hinvI * ainv / params.cfl;           // 3
                        if (alpha < alphaTarg) {                                          // 2
                            alpha = (alpha + lambda1 * dt1 * alphaTarg) / (1.f + lambda1 * dt1);
                            flops++;
                        } else {
                            alpha = (alpha + lambda0 * dt1 * alphaTarg) / (1.f + lambda0 * dt1);
                            flops += 10;
                        }
                    } else if (params.phase == 1) {
                        float divV;
                        divV = dvxDx + dvyDy + dvzDz;    // 2
                        data.divvSnk[snki] = divV;
                        float dtinvHydro = 0.f;
                        float const hI = data.rec2Snk[snki].h;
                        float const dloghdt = fabsf(divV - 3.f * params.adot * ainv) /
                            (3.f + dlogsmoothXDlogh(hI, params.hmin, params.hmax));    // 5
                        float const dtinvDivv = params.a * (dloghdt);
                        float const dtDivv = params.cfl * params.a / (dtinvDivv + 1e-33f);    // 5
                        float dtinvDiff = 0.0f;
                        if (params.diffusion) {
                            dtinvDiff = params.a * dd * (1.0f / 3.0f);
                        }
                        float const a2 = sqr(ax, ay, az);    // 5
                        float dtinvAcc = sqrtf(sqrtf(a2) * hinvI) * params.a * params.cfl /
                            (data.eta * sqrtf(params.a));                        // 11
                        dtinvHydro = fmaxf(dtinvHydro, dtinvDivv);               // 1
                        dtinvHydro = fmaxf(dtinvHydro, dtinvCfl + dtinvVisc);    // 1
                        dtinvHydro = fmaxf(dtinvHydro, dtinvDiff);               // 1
                        dtinvHydro = fmaxf(dtinvHydro, dtinvAcc);                // 1
                        float dthydro = params.cfl * params.a / dtinvHydro;      // 6
                        float const g2 = sqr(gxI, gyI, gzI);                     // 5
                        float const dtinvGrav = sqrtf(sqrtf(g2));                // 8
                        float dtgrav = data.eta *
                            sqrtf(params.a * (params.vsoft ? hI : data.gsoft)) /
                            (dtinvGrav + 1e-30f);                               // 11
                        dthydro = fminf(dthydro, params.maxDt);                 // 1
                        dtgrav = fminf(dtgrav, params.maxDt);                   // 1
                        totalVsigMax = fmaxf(totalVsigMax, dtinvHydro * hI);    // 2
                        char& rung = data.rungsSnk[data.dmIndexSnk[snki]];
                        float const lastDt = rungDt[rung] * params.t0;                  // 1
                        int const rungHydro = ceilf(log2fparamst0 - log2f(dthydro));    // 10
                        int const rungGrav = ceilf(log2fparamst0 - log2f(dtgrav));      // 10
                        maxRungHydro = max(maxRungHydro, rungHydro);
                        maxRungGrav = max(maxRungGrav, rungGrav);
                        if (data.gravity) {
                            rung = max(max((int) max(rungHydro, rungGrav),
                                           max(params.minRung, (int) rung - 1)),
                                1);
                        } else {
                            rung =
                                max(max((int) (rungHydro), max(params.minRung, (int) rung - 1)), 1);
                        }
                        maxRung = max(maxRung, rung);
                        atomicMax(&reduce->dtinvCfl, dtinvCfl);
                        atomicMax(&reduce->dtinvVisc, dtinvVisc);
                        atomicMax(&reduce->dtinvDiff, dtinvDiff);
                        atomicMax(&reduce->dtinvAcc, dtinvAcc);
                        atomicMax(&reduce->dtinvDivv, dtinvDivv);
                        if (rung < 0 || rung > 10) {
                            if (tid == 0) {
                                PRINT("Rung out of range %i %e %e %e %e |%e %e %e %e %e\n",
                                    rungHydro, vsig, cfracI, hI, cI, dtinvCfl, dtinvVisc, dtinvDiff,
                                    dtinvAcc, dtinvDivv);
                                //		__trap();
                            }
                        }
                    }
                }
            }
        }
        sharedReduceAdd<int, HYDRO_BLOCK_SIZE>(flops);
        if (tid == 0) {
            atomicAdd(&reduce->flops, (float) flops);
            index = atomicAdd(&reduce->counter, 1);
        }
        flops = 0;
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax(&reduce->vsigMax, totalVsigMax);
        atomicMax(&reduce->maxRung, maxRung);
        atomicMax(&reduce->maxRungHydro, maxRungHydro);
        atomicMax(&reduce->maxRungGrav, maxRungGrav);
    }
    (&ws)->~HydroWorkspace();
}
