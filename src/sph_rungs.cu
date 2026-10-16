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

namespace detail {

struct RungsWorkspace {
    DeviceVector<Fixed32> x;
    DeviceVector<Fixed32> y;
    DeviceVector<Fixed32> z;
    DeviceVector<float> h;
    DeviceVector<char> rungs;
};

}    // namespace detail

__global__ void sphCudaRungs(SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ detail::RungsWorkspace ws;
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    new (&ws) detail::RungsWorkspace();
    int flops = 0;
    Array<Fixed32, NDIM> x;
    int changes = 0;
    while (index < data.numberOfTargetNodes) {
        __syncthreads();
        ws.x.resize(0);
        ws.y.resize(0);
        ws.z.resize(0);
        ws.h.resize(0);
        ws.rungs.resize(0);
        SphTreeNode const& self = data.trees[data.selfs[index]];
        for (int ni = self.neighborRange.first; ni < self.neighborRange.second; ni++) {
            SphTreeNode const& other = data.trees[data.neighbors[ni]];
            int const maxpi =
                roundUp(other.particleRange.second - other.particleRange.first, blockSize) +
                other.particleRange.first;
            for (int pi = other.particleRange.first + tid; pi < maxpi; pi += blockSize) {
                bool contains = false;
                int j;
                int total;
                int rung;
                if (pi < other.particleRange.second) {
                    rung = data.rungs[pi];
                    if (rung >= params.minRung) {
                        x[XDIM] = data.x[pi];
                        x[YDIM] = data.y[pi];
                        x[ZDIM] = data.z[pi];
                        if (self.outerBox.contains(x)) {
                            contains = true;
                        } else {
                            contains = sphSupportIntersects(self.innerBox, x, data.h[pi]);
                        }
                    }
                }
                j = contains;
                computeIndices<RUNGS_BLOCK_SIZE>(j, total);
                int const offset = ws.x.size();
                __syncthreads();
                int const nextSize = offset + total;
                ws.x.resize(nextSize);
                ws.y.resize(nextSize);
                ws.z.resize(nextSize);
                ws.h.resize(nextSize);
                ws.rungs.resize(nextSize);
                if (contains) {
                    ASSERT(j < total);
                    int const k = offset + j;
                    ws.x[k] = x[XDIM];
                    ws.y[k] = x[YDIM];
                    ws.z[k] = x[ZDIM];
                    ws.h[k] = data.h[pi];
                    ws.rungs[k] = rung;
                }
            }
        }
        for (int i = self.particleRange.first; i < self.particleRange.second; i++) {
            __syncthreads();
            if (data.rungs[i] >= params.minRung) {
                int const sinkIndex = self.sinkParticleRange.first - self.particleRange.first + i;
                auto const xI = data.x[i];
                auto const yI = data.y[i];
                auto const zI = data.z[i];
                auto const hI = data.rec2Snk[sinkIndex].h;
                int const rungI = data.rungs[i];
                auto const h2I = sqr(hI);
                int const jmax = roundUp(ws.x.size(), blockSize);
                int maxRungJ = 0;
                for (int j = tid; j < jmax; j += blockSize) {
                    if (j < ws.x.size()) {
                        auto const xJ = ws.x[j];
                        auto const yJ = ws.y[j];
                        auto const zJ = ws.z[j];
                        int const rungJ = ws.rungs[j];
                        float const xIj = distance(xI, xJ);     // 1
                        float const yIj = distance(yI, yJ);     // 1
                        float const zIj = distance(zI, zJ);     // 1
                        float const r2 = sqr(xIj, yIj, zIj);    // 5
                        if (r2 < fmaxf(h2I, sqr(ws.h[j]))) {
                            maxRungJ = max(maxRungJ, rungJ);
                        }
                        flops += 9;
                    }
                }
                sharedReduceMax<RUNGS_BLOCK_SIZE>(maxRungJ);
                if (tid == 0) {
                    if (rungI < maxRungJ - MAX_RUNG_DIF) {
                        changes++;
                        data.rungsSnk[data.dmIndexSnk[sinkIndex]] = maxRungJ - MAX_RUNG_DIF;
                    }
                }
            }
        }
        sharedReduceAdd<int, RUNGS_BLOCK_SIZE>(flops);
        if (tid == 0) {
            index = atomicAdd(&reduce->counter, 1);
            atomicAdd(&reduce->flops, flops);
        }
        flops = 0;
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&reduce->flag, changes);
    }
    (&ws)->~RungsWorkspace();
}
