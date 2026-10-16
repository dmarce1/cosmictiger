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

struct Prehydro1Workspace {
    DeviceVector<SoftlensRecord> positions;
};

struct Prehydro2Record {
    float h;
    char rung;
};

struct Prehydro2Workspace {
    DeviceVector<SoftlensRecord> positions;
    DeviceVector<Prehydro2Record> particles;
};

}    // namespace detail

__global__ void sphCudaPrehydro1(
    SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ detail::Prehydro1Workspace workspace;
    new (&workspace) detail::Prehydro1Workspace();
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    while (index < data.numberOfTargetNodes) {
        SphTreeNode const& self = data.trees[data.selfs[index]];
        workspace.positions.resize(0);
        for (int ni = self.neighborRange.first; ni < self.neighborRange.second; ni++) {
            auto const& other = data.trees[data.neighbors[ni]];
            int const end = other.particleRange.first +
                roundUp(other.particleRange.second - other.particleRange.first, blockSize);
            for (int pi = other.particleRange.first + tid; pi < end; pi += blockSize) {
                bool contains = false;
                Array<Fixed32, NDIM> position;
                if (pi < other.particleRange.second) {
                    position[XDIM] = data.x[pi];
                    position[YDIM] = data.y[pi];
                    position[ZDIM] = data.z[pi];
                    contains = self.outerBox.contains(position);
                }
                int slot = contains;
                int total;
                computeIndices<PREHYDRO1_BLOCK_SIZE>(slot, total);
                int const offset = workspace.positions.size();
                __syncthreads();
                workspace.positions.resize(offset + total);
                if (contains) {
                    workspace.positions[offset + slot] = {
                        position[XDIM], position[YDIM], position[ZDIM], SPH_TYPE};
                }
            }
        }
        for (int i = self.particleRange.first; i < self.particleRange.second; i++) {
            __syncthreads();
            int const sink = self.sinkParticleRange.first - self.particleRange.first + i;
            bool const active = data.rungsSnk[data.sinkParticleIndices[sink]] >= params.minRung;
            if (active && !data.convergedSnk[sink]) {
                Array<Fixed32, NDIM> position;
                position[XDIM] = data.x[i];
                position[YDIM] = data.y[i];
                position[ZDIM] = data.z[i];
                float& h = data.rec2Snk[sink].h;
                float weightedCount;
                bool const converged =
                    computeSoftlens<PREHYDRO1_BLOCK_SIZE>(h, params.hmin, params.hmax, data.n,
                        workspace.positions, position, self.outerBox, SPH_TYPE, weightedCount);
                if (tid == 0) {
                    data.convergedSnk[sink] = converged;
                    if (converged) {
                        atomicMin(&reduce->hmin, h);
                        atomicMax(&reduce->hmax, h);
                    } else {
                        atomicAdd(&reduce->flag, 1);
                    }
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            index = atomicAdd(&reduce->counter, 1);
        }
        __syncthreads();
    }
    (&workspace)->~Prehydro1Workspace();
}

__global__ void sphCudaPrehydro2(
    SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ detail::Prehydro2Workspace workspace;
    new (&workspace) detail::Prehydro2Workspace();
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    while (index < data.numberOfTargetNodes) {
        SphTreeNode const& self = data.trees[data.selfs[index]];
        workspace.positions.resize(0);
        workspace.particles.resize(0);
        for (int ni = self.neighborRange.first; ni < self.neighborRange.second; ni++) {
            auto const& other = data.trees[data.neighbors[ni]];
            int const end = other.particleRange.first +
                roundUp(other.particleRange.second - other.particleRange.first, blockSize);
            for (int pi = other.particleRange.first + tid; pi < end; pi += blockSize) {
                bool contains = false;
                Array<Fixed32, NDIM> position;
                if (pi < other.particleRange.second) {
                    position[XDIM] = data.x[pi];
                    position[YDIM] = data.y[pi];
                    position[ZDIM] = data.z[pi];
                    contains = self.outerBox.contains(position);
                    if (!contains) {
                        contains = sphSupportIntersects(self.innerBox, position, data.h[pi]);
                    }
                }
                int slot = contains;
                int total;
                computeIndices<PREHYDRO2_BLOCK_SIZE>(slot, total);
                int const offset = workspace.positions.size();
                __syncthreads();
                workspace.positions.resize(offset + total);
                workspace.particles.resize(offset + total);
                if (contains) {
                    workspace.positions[offset + slot] = {
                        position[XDIM], position[YDIM], position[ZDIM], SPH_TYPE};
                    workspace.particles[offset + slot] = {data.h[pi], data.rungs[pi]};
                }
            }
        }
        for (int i = self.particleRange.first; i < self.particleRange.second; i++) {
            __syncthreads();
            int const sink = self.sinkParticleRange.first - self.particleRange.first + i;
            bool const active = data.rungsSnk[data.sinkParticleIndices[sink]] >= params.minRung;
            Array<Fixed32, NDIM> position;
            position[XDIM] = data.x[i];
            position[YDIM] = data.y[i];
            position[ZDIM] = data.z[i];
            float& h = data.rec2Snk[sink].h;
            int semiactive = active;
            if (!active) {
                for (int j = tid; j < workspace.positions.size(); j += blockSize) {
                    auto const& neighbor = workspace.positions[j];
                    auto const& particle = workspace.particles[j];
                    if (particle.rung >= params.minRung) {
                        float const dx = distance(position[XDIM], neighbor.x);
                        float const dy = distance(position[YDIM], neighbor.y);
                        float const dz = distance(position[ZDIM], neighbor.z);
                        if (sqr(dx, dy, dz) < fmaxf(sqr(h), sqr(particle.h))) {
                            semiactive++;
                        }
                    }
                }
                sharedReduceAdd<int, PREHYDRO2_BLOCK_SIZE>(semiactive);
            }
            if (tid == 0) {
                data.saSnk[sink] = semiactive != 0;
            }
            if (!semiactive) {
                continue;
            }
            bool converged = true;
            if (!active && !params.vsoft && !data.convergedSnk[sink]) {
                float weightedCount;
                converged = computeSoftlens<PREHYDRO2_BLOCK_SIZE>(h, params.hmin, params.hmax,
                    data.n, workspace.positions, position, self.outerBox, SPH_TYPE, weightedCount);
            }
            if (!converged) {
                if (tid == 0) {
                    data.convergedSnk[sink] = false;
                    atomicAdd(&reduce->flag, 1);
                }
                continue;
            }
            float const hinv = 1.0f / h;
            float const normalization = data.m * hinv * hinv * hinv;
            float density = 0.0f;
            for (int j = tid; j < workspace.positions.size(); j += blockSize) {
                auto const& neighbor = workspace.positions[j];
                float const dx = distance(position[XDIM], neighbor.x);
                float const dy = distance(position[YDIM], neighbor.y);
                float const dz = distance(position[ZDIM], neighbor.z);
                float const q = sqrtf(sqr(dx, dy, dz)) * hinv;
                density += normalization * kernelW(q);
            }
            sharedReduceAdd<float, PREHYDRO2_BLOCK_SIZE>(density);
            if (tid == 0) {
                ALWAYS_ASSERT(h > 0.0f && density > 0.0f);
                data.rhoSnk[sink] = density;
                data.preSnk[sink] = (data.defGamma - 1.0f) * density * data.eint[i];
                // Classical SPH uses the symmetric kernel gradient directly.
                data.omegaSnk[sink] = 1.0f;
                data.omegaPSnk[sink] = 1.0f;
                data.shearSnk[sink] = 0.0f;
                data.convergedSnk[sink] = true;
                atomicMin(&reduce->hmin, h);
                atomicMax(&reduce->hmax, h);
            }
        }
        __syncthreads();
        if (tid == 0) {
            index = atomicAdd(&reduce->counter, 1);
        }
        __syncthreads();
    }
    (&workspace)->~Prehydro2Workspace();
}
