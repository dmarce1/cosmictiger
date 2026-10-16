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

struct Prehydro1Workspace {
    DeviceVector<SoftlensRecord> rec1;
};

__global__ void sphCudaPrehydro1(
    SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ Prehydro1Workspace ws;
    __syncthreads();
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    new (&ws) Prehydro1Workspace();
    Array<Fixed32, NDIM> x;
    while (index < data.numberTargetNodes) {
        int flops = 0;
        __syncthreads();
        ws.rec1.resize(0);
        SphTreeNode const& self = data.trees[data.selfs[index]];
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
                    contains = (self.outerBox.contains(x));
                }
                j = contains;
                computeIndices<PREHYDRO1_BLOCK_SIZE>(j, total);
                int const offset = ws.rec1.size();
                __syncthreads();
                int const nextSize = offset + total;
                ws.rec1.resize(nextSize);
                if (contains) {
                    int const k = offset + j;
                    ws.rec1[k].x = x[XDIM];
                    ws.rec1[k].y = x[YDIM];
                    ws.rec1[k].z = x[ZDIM];
                    ws.rec1[k].type = SPH_TYPE;
                }
            }
        }
        float hminAll = 1e+20;
        float hmaxAll = 0.0;
        float const w0 = kernelW(0.f);
        __syncthreads();
        for (int i = self.partRange.first; i < self.partRange.second; i++) {
            __syncthreads();
            int const snki = self.sinkPartRange.first - self.partRange.first + i;
            auto const rung = data.rungsSnk[data.dmIndexSnk[snki]];
            bool const active = rung >= params.minRung;
            auto& converged = data.convergedSnk[snki];
            bool const use = active && !converged;
            if (use) {
                x[XDIM] = data.x[i];
                x[YDIM] = data.y[i];
                x[ZDIM] = data.z[i];
                float& h = data.rec2Snk[snki].h;
                float wcount;
                if (!computeSoftlens<PREHYDRO1_BLOCK_SIZE>(h, params.hmin, params.hmax, data.n,
                        ws.rec1, x, self.outerBox, SPH_TYPE, wcount)) {
                    if (tid == 0) {
                        atomicAdd(&reduce->flag, 1);
                        if (tid == 0) {
                            converged = false;
                        }
                    }
                } else {
                    if (tid == 0) {
                        if (tid == 0) {
                            converged = true;
                        }
                        hminAll = fminf(hminAll, h);
                        hmaxAll = fmaxf(hmaxAll, h);
                    }
                }
            }
        }
        sharedReduceAdd<int, PREHYDRO1_BLOCK_SIZE>(flops);
        if (tid == 0) {
            atomicMax(&reduce->hmax, hmaxAll);
            atomicMin(&reduce->hmin, hminAll);
            atomicAdd(&reduce->flops, (double) flops);
            index = atomicAdd(&reduce->counter, 1);
        }
        flops = 0;
        __syncthreads();
    }
    (&ws)->~Prehydro1Workspace();
}

struct Prehydro2Record2 {
    float vx;
    float vy;
    float vz;
#ifdef ENTROPY
    float entr;
#else
    float eint;
#endif
    float h;
    char star;
    char rung;
};

struct Prehydro2Workspace {
    DeviceVector<SoftlensRecord> rec1;
    DeviceVector<Prehydro2Record2> rec2;
    DeviceVector<int> neighbors;
};

__global__ void sphCudaPrehydro2(
    SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ Prehydro2Workspace ws;
    __syncthreads();
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    new (&ws) Prehydro2Workspace();
    Array<Fixed32, NDIM> x;
    while (index < data.numberTargetNodes) {
        int flops = 0;
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
                if (pi < other.partRange.second) {
                    x[XDIM] = data.x[pi];
                    x[YDIM] = data.y[pi];
                    x[ZDIM] = data.z[pi];
                    contains = (self.outerBox.contains(x));
                    if (!contains) {
                        contains = true;
                        float const& h = data.h[pi];
                        for (int dim = 0; dim < NDIM; dim++) {
                            if (distance(x[dim], self.innerBox.begin[dim]) + 1.01f * h < 0.f) {
                                flops += 3;
                                contains = false;
                                break;
                            }
                            if (distance(self.innerBox.end[dim], x[dim]) + 1.01f * h < 0.f) {
                                flops += 3;
                                contains = false;
                                break;
                            }
                        }
                    }
                }
                j = contains;
                computeIndices<PREHYDRO2_BLOCK_SIZE>(j, total);
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
                    ws.rec1[k].type = SPH_TYPE;
                    ws.rec2[k].h = data.h[pi];
                    if (params.stars) {
                        ws.rec2[k].star = data.stars[pi];
                    } else {
                        ws.rec2[k].star = false;
                    }
                    ws.rec2[k].vx = data.vx[pi];
                    ws.rec2[k].vy = data.vy[pi];
                    ws.rec2[k].vz = data.vz[pi];
#ifdef ENTROPY
                    ws.rec2[k].entr = data.entr[pi];
#else
                    ws.rec2[k].eint = data.eint[pi];
#endif
                    ws.rec2[k].rung = data.rungs[pi];
                }
            }
        }
        __syncthreads();
        float hmin = 1e+20;
        float hmax = 0.0;
        float const w0 = kernelW(0.f);
        __syncthreads();
        for (int i = self.partRange.first; i < self.partRange.second; i++) {
            __syncthreads();
            int const snki = self.sinkPartRange.first - self.partRange.first + i;
            auto const rung = data.rungsSnk[data.dmIndexSnk[snki]];
            bool const active = rung >= params.minRung;
            int semiactive = 0;
            x[XDIM] = data.x[i];
            x[YDIM] = data.y[i];
            x[ZDIM] = data.z[i];
            Fixed32 const& xI = x[XDIM];
            Fixed32 const& yI = x[YDIM];
            Fixed32 const& zI = x[ZDIM];
            float const& hI = data.rec2Snk[snki].h;
            float const hinvI = 1.f / hI;    // 4
            float const h2I = sqr(hI);       // 1
            if (active) {
                if (tid == 0) {
                    data.saSnk[snki] = true;
                }
            } else {
                int const jmax = roundUp(ws.rec1.size(), blockSize);
                if (tid == 0) {
                    data.saSnk[snki] = false;
                }
                for (int j = tid; j < jmax; j += blockSize) {
                    if (j < ws.rec1.size()) {
                        auto const& rec1 = ws.rec1[j];
                        auto const& rec2 = ws.rec2[j];
                        if (rec2.rung >= params.minRung) {
                            auto const& xJ = rec1.x;
                            auto const& yJ = rec1.y;
                            auto const& zJ = rec1.z;
                            auto const& hJ = rec2.h;
                            auto const h2J = sqr(hJ);               // 1
                            float const xIj = distance(xI, xJ);     // 1
                            float const yIj = distance(yI, yJ);     // 1
                            float const zIj = distance(zI, zJ);     // 1
                            float const r2 = sqr(xIj, yIj, zIj);    // 5
                            if (r2 < fmaxf(sqr(hI), sqr(hJ))) {     // 2
                                semiactive++;
                            }
                        }
                        flops += 11;
                    }
                    sharedReduceAdd<int, PREHYDRO2_BLOCK_SIZE>(semiactive);
                    if (semiactive) {
                        if (tid == 0) {
                            data.saSnk[snki] = true;
                        }
                        break;
                    }
                }
            }
            int boxXceeded = false;
            auto& converged = data.convergedSnk[snki];
            bool path;
            if (semiactive && !params.vsoft && !converged) {
                path = true;
                x[XDIM] = data.x[i];
                x[YDIM] = data.y[i];
                x[ZDIM] = data.z[i];
                float& h = data.rec2Snk[snki].h;
                float wcount;
                if (!computeSoftlens<PREHYDRO2_BLOCK_SIZE>(h, params.hmin, params.hmax, data.n,
                        ws.rec1, x, self.outerBox, SPH_TYPE, wcount)) {
                    boxXceeded = true;
                    if (tid == 0) {
                        atomicAdd(&reduce->flag, 1);
                        converged = false;
                    }
                } else {
                    if (tid == 0) {
                        converged = true;
                    }
                }
            } else {
                path = false;
            }
            if ((active || semiactive) && !boxXceeded) {
                __syncthreads();
                int const snki = self.sinkPartRange.first - self.partRange.first + i;
                float const& hI = data.rec2Snk[snki].h;
                ALWAYS_ASSERT(hI >= params.hmin);
                float const& vxI = data.vx[i];
                float const& vyI = data.vy[i];
                float const& vzI = data.vz[i];
                Fixed32 const& xI = x[XDIM];
                Fixed32 const& yI = x[YDIM];
                Fixed32 const& zI = x[ZDIM];
                float const hinvI = 1.f / hI;    // 4
                float const h3invI = sqr(hinvI) * hinvI;
                float const h2I = sqr(hI);    // 1
                float dwSum, wSum;
                dwSum = 0.f;
                wSum = 0.f;
                float rhoh30 = (3.0f * data.n) / (4.0f * float(M_PI));    // 5
                float dvxDx = 0.f;
                float dvxDy = 0.f;
                float dvxDz = 0.f;
                float dvyDx = 0.f;
                float dvyDy = 0.f;
                float dvyDz = 0.f;
                float dvzDx = 0.f;
                float dvzDy = 0.f;
                float dvzDz = 0.f;
                float rhoI = 0.f;
#ifdef HOPKINS
                float pre = 0.f;
                float dpredh = 0.f;
#endif
                ws.neighbors.resize(0);
                __syncthreads();
                flops += 10;
                int const jmax = roundUp(ws.rec1.size(), PREHYDRO2_BLOCK_SIZE);
                for (int j = tid; j < jmax; j += blockSize) {
                    bool contains = false;
                    if (j < ws.rec1.size()) {
                        auto const& rec1 = ws.rec1[j];
                        auto const& rec2 = ws.rec2[j];
                        Fixed32 const& xJ = rec1.x;
                        Fixed32 const& yJ = rec1.y;
                        Fixed32 const& zJ = rec1.z;
                        auto const& starJ = rec2.star;
#ifdef ENTROPY
                        float const& aJ = rec2.entr;
#else
                        float const& eintJ = rec2.eint;
#endif
                        float const xIj = distance(xI, xJ);    // 1
                        float const yIj = distance(yI, yJ);    // 1
                        float const zIj = distance(zI, zJ);    // 1
                        float const r2 = sqr(xIj, yIj, zIj);
                        float const r = sqrtf(r2);    // 4
                        float const q = r * hinvI;    // 1
                        if (q < 1.f) {                // 1
                            float const w = kernelW(q);
                            float const dwdq = dkernelWDq(q);
                            dwSum -= q * dwdq;    // 2
                            wSum += w;
                            if (!starJ) {
#ifdef HOPKINS
#ifdef ENTROPY
                                pre += powf(aJ, 1.f / data.defGamma) * w;
                                dpredh -= powf(aJ, 1.f / data.defGamma) * q * dwdq;
#else
                                pre += (data.defGamma - 1.0f) * eintJ * w;
                                dpredh -= (data.defGamma - 1.0f) * eintJ * q * dwdq;
#endif
#endif
                                rhoI += data.m * w * h3invI;
                            }
                            contains = true;
                            flops += 2;
                        }
                        flops += 9;
                    }
                    int k = contains;
                    int total;
                    computeIndices<PREHYDRO2_BLOCK_SIZE>(k, total);
                    int const offset = ws.neighbors.size();
                    __syncthreads();
                    int const nextSize = offset + total;
                    ws.neighbors.resize(nextSize);
                    if (contains) {
                        int const l = offset + k;
                        ws.neighbors[l] = j;
                    }
                }
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dwSum);
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(wSum);
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(rhoI);
#ifdef HOPKINS
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(pre);
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dpredh);
#endif
                data.rhoSnk[snki] = rhoI;
                auto const one = wSum / (3.0 / 4.0 / M_PI * data.n);
                //	ALWAYS_ASSERT(fabs(one - 1.0) < .001);
                float const a = 0.33333333333f * dwSum / wSum;
                float f, dfdh;
                dsmoothXDh(hI, params.hmin, params.hmax, f, dfdh);
                float const b = 0.33333333333f * hI / f * dfdh;
                float const omegaI = (a + b) / (1.0f + b);
                data.omegaSnk[snki] = omegaI;
#ifdef HOPKINS
                data.omegaPSnk[snki] =
                    1.f / ((-pre + 0.33333333333333f * dpredh) / (f * wSum)) * omegaI;
#ifdef ENTROPY
                data.preSnk[snki] = powf(data.m * pre * h3invI, data.defGamma);
#else
                data.preSnk[snki] = data.m * pre * h3invI;
#endif
                if (tid == 0) {
                    //				PRINT( "%e %e %e\n", data.pre_snk[snki], rho_i, x_i.to_float());
                }
#else
                data.omegaPSnk[snki] = 1.f / ((-1.f + entropy * f * wSum) / (f * wSum)) * omegaI;
#endif
                //	PRINT( "%e\n", 	data.omegaP_snk[snki] );
                if (!(omegaI > 0.0)) {
                    PRINT("%e %e %e %e %e %e\n", omegaI, a, b, hI, wSum, dwSum);
                }
                ALWAYS_ASSERT(omegaI > 0.f);
                __syncthreads();
                float const h4invI = h3invI * hinvI;    // 1
                flops += 16;
                for (int j = tid; j < ws.neighbors.size(); j += blockSize) {
                    int const kk = ws.neighbors[j];
                    auto const& rec1 = ws.rec1[kk];
                    auto const& rec2 = ws.rec2[kk];
                    auto const& starJ = rec2.star;
                    if (!starJ) {
                        Fixed32 const& xJ = rec1.x;
                        Fixed32 const& yJ = rec1.y;
                        Fixed32 const& zJ = rec1.z;
                        float const& vxJ = rec2.vx;
                        float const& vyJ = rec2.vy;
                        float const& vzJ = rec2.vz;
                        float const xIj = distance(xI, xJ);                  // 1
                        float const yIj = distance(yI, yJ);                  // 1
                        float const zIj = distance(zI, zJ);                  // 1
                        float const r2 = sqr(xIj, yIj, zIj);                 // 5
                        float const r = sqrtf(r2);                           // 4
                        float const q = r * hinvI;                           // 1
                        float const vxIj = vxI - vxJ + xIj * params.adot;    // 3
                        float const vyIj = vyI - vyJ + yIj * params.adot;    // 3
                        float const vzIj = vzI - vzJ + zIj * params.adot;    // 3
                        float const rinv = 1.0f / (1.0e-30f + r);            // 5
                        float const dwdq = dkernelWDq(q);
                        float const dWdrI = dwdq * h4invI / omegaI;    // 2
                        float const dWdrIRinv = dWdrI * rinv;          // 1
                        float const dWdrIX = dWdrIRinv * xIj;          // 1
                        float const dWdrIY = dWdrIRinv * yIj;          // 1
                        float const dWdrIZ = dWdrIRinv * zIj;          // 1
                        dvxDx -= vxIj * dWdrIX;                        // 2
                        dvyDx -= vyIj * dWdrIX;                        // 2
                        dvzDx -= vzIj * dWdrIX;                        // 2
                        dvxDy -= vxIj * dWdrIY;                        // 2
                        dvyDy -= vyIj * dWdrIY;                        // 2
                        dvzDy -= vzIj * dWdrIY;                        // 2
                        dvxDz -= vxIj * dWdrIZ;                        // 2
                        dvyDz -= vyIj * dWdrIZ;                        // 2
                        dvzDz -= vzIj * dWdrIZ;                        // 2
                    }
                    flops += 68;
                }
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvxDx);    // 127
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvxDy);    // 127
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvxDz);    // 127
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvyDx);    // 127
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvyDy);    // 127
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvyDz);    // 127
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvzDx);    // 127
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvzDy);    // 127
                sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(dvzDz);    // 127
                if (tid == 0) {
                    flops += 38 + (PREHYDRO2_BLOCK_SIZE - 1) * 11;
                    float divV;
                    float const mrhoinv = data.m / rhoI;    // 4
                    dvxDx *= mrhoinv;                       // 1
                    dvxDy *= mrhoinv;                       // 1
                    dvxDz *= mrhoinv;                       // 1
                    dvyDx *= mrhoinv;                       // 1
                    dvyDy *= mrhoinv;                       // 1
                    dvyDz *= mrhoinv;                       // 1
                    dvzDx *= mrhoinv;                       // 1
                    dvzDy *= mrhoinv;                       // 1
                    dvzDz *= mrhoinv;                       // 1
                    divV = dvxDx + dvyDy + dvzDz;           // 2
                    float shearXx, shearXy, shearXz, shearYy, shearYz, shearZz;
                    shearXx = dvxDx - (1.f / 3.f) * divV;    // 2
                    shearYy = dvyDy - (1.f / 3.f) * divV;    // 2
                    shearZz = dvzDz - (1.f / 3.f) * divV;    // 2
                    shearXy = 0.5f * (dvxDy + dvyDx);        // 2
                    shearXz = 0.5f * (dvxDz + dvzDx);        // 2
                    shearYz = 0.5f * (dvyDz + dvzDy);        // 2
                    float const shearv = sqrtf(sqr(shearXx) + sqr(shearYy) + sqr(shearZz) +
                        2.0f * (sqr(shearXy) + sqr(shearXz) + sqr(shearYz)));    // 16
                    data.shearSnk[snki] = shearv;
                }
            }
        }
        sharedReduceAdd<int, PREHYDRO2_BLOCK_SIZE>(flops);
        if (tid == 0) {
            atomicMax(&reduce->hmax, hmax);
            atomicMin(&reduce->hmin, hmin);
            atomicAdd(&reduce->flops, (double) flops);
            index = atomicAdd(&reduce->counter, 1);
        }
        flops = 0;
        __syncthreads();
    }
    (&ws)->~Prehydro2Workspace();
}
