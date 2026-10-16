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

struct ConductionRecord1 {
    Fixed32 x;
    Fixed32 y;
    Fixed32 z;
    float h;
    char rung;
};
struct ConductionRecord2 {
#ifdef ENTROPY
    float entr;
#else
    float eint;
#endif
    float kappa;
    float omega;
    float cfrac;
    float rho;
    char star;
};

struct ConductionWorkspace {
    DeviceVector<int> neighbors;
    DeviceVector<ConductionRecord1> rec1;
    DeviceVector<ConductionRecord2> rec2;
};

__global__ void sphCudaConduction(SphRunParams params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ ConductionWorkspace ws;
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    new (&ws) ConductionWorkspace();
    float const gamma0 = data.defGamma;
    float const m = data.m;
    Array<Fixed32, NDIM> x;
    float const cons = -2.f * sqr(params.a) * m;
    int flops = 4;
    while (index < data.nselfs) {
        __syncthreads();
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
                    ALWAYS_ASSERT(h > 0.f);
                    rung = data.rungs[pi];
                    if (rung >= params.minRung) {
                        x[XDIM] = data.x[pi];
                        x[YDIM] = data.y[pi];
                        x[ZDIM] = data.z[pi];
                        contains = (self.outerBox.contains(x));
                        if (!contains) {
                            contains = true;
                            for (int dim = 0; dim < NDIM; dim++) {
                                if (distance(x[dim], self.innerBox.begin[dim]) + h < 0.f) {
                                    contains = false;
                                    break;
                                }
                                if (distance(self.innerBox.end[dim], x[dim]) + h < 0.f) {
                                    contains = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                j = contains;
                computeIndices<CONDUCTION_BLOCK_SIZE>(j, total);
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
                    ALWAYS_ASSERT(h > 0.f);
                    ws.rec1[k].h = h;
                    ws.rec1[k].rung = data.rungs[pi];
#ifdef ENTROPY
                    ws.rec2[k].entr = data.entr[pi];
#else
                    ws.rec2[k].eint = data.eint[pi];
#endif
                    ws.rec2[k].kappa = data.kappa[pi];
                    ws.rec2[k].omega = data.omega[pi];
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
        __syncthreads();
        for (int i = self.partRange.first; i < self.partRange.second; i++) {
            int const snki = self.sinkPartRange.first - self.partRange.first + i;
            bool const active = data.rungs[i] >= params.minRung;
            bool const semiactive = !active && data.saSnk[snki];
            bool const converged = data.convergedSnk[snki];
            if ((active || semiactive) && !converged) {
                ws.neighbors.resize(0);
                float const jmax = roundUp(ws.rec1.size(), blockSize);
                bool starI = false;
                if (params.stars) {
                    starI = data.stars[i];
                }
                auto const& xI = data.x[i];
                auto const& yI = data.y[i];
                auto const& zI = data.z[i];
                float const& hI = data.h[i];
#ifdef ENTROPY
                float const& aI = data.entr[i];
#else
                float const& eintI = data.eint[i];
#endif
                float cfracI = 0.f;
                if (params.stars) {
                    cfracI = data.coldFrac[i];
                }
                float const hfracI = 1.f - cfracI;
                for (int j = tid; j < jmax; j += blockSize) {
                    int k;
                    int total;
                    bool use = false;
                    if (j < ws.rec1.size()) {
                        auto const rec1 = ws.rec1[j];
                        Fixed32 const xJ = rec1.x;
                        Fixed32 const yJ = rec1.y;
                        Fixed32 const zJ = rec1.z;
                        auto const rungJ = rec1.rung;
                        float const hJ = rec1.h;
                        float const xIj = distance(xI, xJ);                // 1
                        float const yIj = distance(yI, yJ);                // 1
                        float const zIj = distance(zI, zJ);                // 1
                        float const r2 = sqr(xIj, yIj, zIj);               // 5
                        if (r2 < fmaxf(sqr(hI), sqr(hJ)) && r2 > 0.f) {    // 4
                            use = active || rungJ >= params.minRung;
                        }
                        flops += 12;
                    }
                    k = use;
                    computeIndices<CONDUCTION_BLOCK_SIZE>(k, total);
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

                float const& omegaI = data.omega[i];
                auto const& rungI = data.rungs[i];
                float const& kappaI = data.kappa[i];
                float const& rhoI = data.rho[i];
                float const h2I = sqr(hI);                                                      // 1
                float const hinvI = 1.f / hI;                                                   // 4
                float const h3invI = (sqr(hinvI) * hinvI);                                      // 2
                float const dtI = rungI >= params.minRung ? rungDt[rungI] * params.t0 : 0.f;    // 1
                float const rhoinvI = 1.f / (rhoI);                                             // 5
                float den = 0.f;
                float num = 0.f;
                flops += 28;
                if (!starI) {
                    for (int j = tid; j < ws.neighbors.size(); j += blockSize) {
                        int const kk = ws.neighbors[j];
                        auto const& rec1 = ws.rec1[kk];
                        auto const& rec2 = ws.rec2[kk];
                        auto const& starJ = rec2.star;
                        if (!starJ) {
                            Fixed32 const xJ = rec1.x;
                            Fixed32 const yJ = rec1.y;
                            Fixed32 const zJ = rec1.z;
                            float const hJ = rec1.h;
                            auto const rungJ = rec1.rung;
                            float const xIj = distance(xI, xJ);     // 1
                            float const yIj = distance(yI, yJ);     // 1
                            float const zIj = distance(zI, zJ);     // 1
                            float const r2 = sqr(xIj, yIj, zIj);    // 5
                            float const h2J = sqr(hJ);              // 1
                            flops += 11;
                            if (r2 < fmaxf(h2J, h2I)) {    // 2
                                float const& cfracJ = rec2.cfrac;
#ifdef ENTROPY
                                float const& aJ = rec2.entr;
#else
                                float const& eintJ = rec2.eint;
#endif
                                float const& rhoJ = rec2.rho;    // 2
                                float const hfracJ = 1.f - cfracJ;
                                float const r = sqrtf(r2);     // 4
                                float const rinv = 1.f / r;    // 4
                                float const qI = r * hinvI;    // 1
                                flops += 10;
                                float const omegaJ = rec2.omega;
                                float const kappaJ = rec2.kappa;
                                float const hinvJ = 1.f / hJ;               // 4
                                float const h3invJ = sqr(hinvJ) * hinvJ;    // 2
                                float const qJ = r * hinvJ;                 // 1
                                float const dWdrI =
                                    dkernelWDq(qI) * h3invI * hinvI / omegaI;    // 3
                                float const dWdrJ =
                                    dkernelWDq(qJ) * h3invJ * hinvJ / omegaJ;    // 3
                                float const dWdrIj = 0.5f * (dWdrI + dWdrJ);     // 22
                                float const kappaIj =
                                    2.f * kappaI * kappaJ / (kappaI + kappaJ + 1.0e-35f);    // 8
                                float const dtJ = rungJ >= params.minRung ?
                                    rungDt[rungJ] * params.t0 :
                                    0.f;                                  // 1
                                float const dtIj = 0.5f * (dtI + dtJ);    // 2
                                float const dIj =
                                    cons * kappaIj * dWdrIj / (rhoI * rhoJ) * dtIj * rinv;    // 10
                                ALWAYS_ASSERT(dIj >= 0.f);
#ifdef ENTROPY
                                num = fmaf(dIj,
                                    aJ * powf(rhoJ * rhoinvI * hfracJ / hfracI, gamma0 - 1.f),
                                    num);    // 14
#else
                                num = fmaf(dIj, eintJ, num);    // 14
#endif
                                den += dIj;    // 1
                                flops += 69;
                            }
                        }
                    }
                }
                sharedReduceAdd<float, CONDUCTION_BLOCK_SIZE>(num);    // 31
                sharedReduceAdd<float, CONDUCTION_BLOCK_SIZE>(den);    // 31
                if (tid == 0) {
                    flops += (CONDUCTION_BLOCK_SIZE - 1) * 2;
                    flops += 7;
#ifdef ENTROPY
                    float const a0 = data.entr0Snk[snki];
                    float const a1 = (a0 + num) / (1.f + den);
                    float const dA = a1 - aI;
                    data.dentr1Snk[snki] = dA;
                    ALWAYS_ASSERT(aI + dA >= 0.0);
#else
                    float const eint0 = data.eint0Snk[snki];
                    float const eint1 = (eint0 + num) / (1.f + den);
                    float const deint = eint1 - eintI;
                    data.deint1Snk[snki] = deint;
                    ALWAYS_ASSERT(eintI + deint >= 0.0);
#endif
                }
            }
        }

        sharedReduceAdd<int, CONDUCTION_BLOCK_SIZE>(flops);
        if (tid == 0) {
            index = atomicAdd(&reduce->counter, 1);
            atomicAdd(&reduce->flops, (float) flops);
        }
        flops = 0;
        __syncthreads();
    }
    (&ws)->~ConductionWorkspace();
}
