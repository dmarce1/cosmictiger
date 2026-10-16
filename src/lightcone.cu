/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

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

#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/options.hpp>

#define BLOCK_SIZE 32

__global__ void cudaLightconeKernel(LightconeParticleMapType* partMapPtr,
    LightconeTreeMapType* treeMapPtr, int* rcPtr, LightconeTreeId const* leaves, int n,
    int* indexPtr, LightconeGroup* nextIdPtr, double linkLen, int hpxSize, int hpxRank) {
    int const& tid = threadIdx.x;
    __shared__ int index;
    float const linkLen2 = sqr(linkLen);
    auto& partMap = *partMapPtr;
    auto& treeMap = *treeMapPtr;
    if (tid == 0) {
        index = atomicAdd(indexPtr, 1);
    }
    auto const generateId = [&nextIdPtr, hpxRank, hpxSize]() {
        LightconeGroup id = atomicAdd(nextIdPtr, 1);
        id *= hpxSize;
        id += hpxRank + 1;
        return id;
    };
    __syncwarp();
    while (index < n) {
        auto const& thisLeaf = leaves[index];
        auto const& self = treeMap[thisLeaf.pix][thisLeaf.index];
        if (self.lastActive) {
            Range<double> mybox = self.box.pad(1.01 * linkLen);
            int foundAnyLink = false;
            auto const& neighbors = self.neighbors;
            auto& selfParts = partMap[self.pix];
            for (int li = 0; li < neighbors.size(); li++) {
                if (thisLeaf != neighbors[li]) {
                    auto const ni = neighbors[li];
                    auto const& other = treeMap[ni.pix][ni.index];
                    auto const& otherParts = partMap[other.pix];
                    for (int pi = other.partRange.first; pi < other.partRange.second; pi++) {
                        auto const bValue = otherParts.group[pi];
                        Array<double, NDIM> b;
                        b[XDIM] = otherParts.pos[XDIM][pi].toDouble();
                        b[YDIM] = otherParts.pos[YDIM][pi].toDouble();
                        b[ZDIM] = otherParts.pos[ZDIM][pi].toDouble();
                        if (mybox.contains(b)) {
                            for (int pj = self.partRange.first + tid; pj < self.partRange.second;
                                pj += BLOCK_SIZE) {
                                auto& a = selfParts.group[pj];
                                float const dx =
                                    distance(selfParts.pos[XDIM][pj], otherParts.pos[XDIM][pi]);
                                float const dy =
                                    distance(selfParts.pos[YDIM][pj], otherParts.pos[YDIM][pi]);
                                float const dz =
                                    distance(selfParts.pos[ZDIM][pj], otherParts.pos[ZDIM][pi]);
                                float const r2 = sqr(dx, dy, dz);
                                if (r2 < linkLen2) {
                                    if (a == LC_NO_GROUP) {
                                        if (bValue != LC_NO_GROUP) {
                                            a = bValue;
                                        } else {
                                            a = generateId();
                                        }
                                        __threadfence();
                                        foundAnyLink = true;
                                    }
                                    if (a > bValue) {
                                        a = bValue;
                                        __threadfence();
                                        foundAnyLink = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            __syncwarp();
            bool foundLink;
            do {
                foundLink = false;
                for (int pj = self.partRange.first; pj < self.partRange.second; pj++) {
                    auto& a = selfParts.group[pj];
                    LightconeGroup minGroup = LC_NO_GROUP;
                    int thisFoundLink = 0;
                    auto const maxpi =
                        roundUp(self.partRange.second - self.partRange.first, BLOCK_SIZE) +
                        self.partRange.first;
                    for (int pi = self.partRange.first + tid; pi < maxpi; pi += BLOCK_SIZE) {
                        LightconeGroup thisMinGroup = LC_NO_GROUP;
                        if (pi != pj && pi < self.partRange.second) {
                            auto const bValue = selfParts.group[pi];
                            float const dx =
                                distance(selfParts.pos[XDIM][pi], selfParts.pos[XDIM][pj]);
                            float const dy =
                                distance(selfParts.pos[YDIM][pi], selfParts.pos[YDIM][pj]);
                            float const dz =
                                distance(selfParts.pos[ZDIM][pi], selfParts.pos[ZDIM][pj]);
                            float const r2 = sqr(dx, dy, dz);
                            if (r2 < linkLen2) {
                                thisMinGroup = bValue;
                                thisFoundLink++;
                            }
                        }
                        sharedReduceMin(thisMinGroup);
                        minGroup = min(minGroup, thisMinGroup);
                    }
                    sharedReduceAdd(thisFoundLink);
                    if (thisFoundLink && (a > minGroup || a == LC_NO_GROUP)) {
                        foundLink = true;
                        foundAnyLink = true;
                        if (tid == 0) {
                            a = minGroup;
                            if (a == LC_NO_GROUP) {
                                a = generateId();
                            }
                            __threadfence();
                        }
                    }
                }
            } while (foundLink);
            sharedReduceAdd(foundAnyLink);
            if (foundAnyLink) {
                for (int li = tid; li < neighbors.size(); li += BLOCK_SIZE) {
                    auto const ni = neighbors[li];
                    auto& other = treeMap[ni.pix][ni.index];
                    other.active = 1;
                }
                if (tid == 0) {
                    atomicAdd(rcPtr, 1);
                }
            }
        }
        if (tid == 0) {
            index = atomicAdd(indexPtr, 1);
            //	PRINT( "%i %i\n", index, N);
        }
        __syncwarp();
    }
    //	PRINT( "! %lli\n", *rc_ptr);
}

size_t cudaLightcone(DeviceVector<LightconeTreeId> const& leaves,
    LightconeParticleMapType* partMapPtr, LightconeTreeMapType* treeMapPtr,
    LightconeGroup* nextIdValue) {
    int numberBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberBlocks, (void const*) cudaLightconeKernel, BLOCK_SIZE, 0));
    numberBlocks *= cudaSmpCount();
    int* indexPtr;
    LightconeGroup* nextIdPtr;
    int* rcPtr;
    nextIdPtr = (LightconeGroup*) cudaAllocate(sizeof(LightconeGroup));
    indexPtr = (int*) cudaAllocate(sizeof(int));
    rcPtr = (int*) cudaAllocate(sizeof(int));
    *rcPtr = 0;
    *indexPtr = 0;
    *nextIdPtr = *nextIdValue;
    auto const b = getOptions().lcB;
    auto const numberParticles = getOptions().numberParticles;
    double const linkLen = b * pow(numberParticles, -1.0 / NDIM);
    cudaLightconeKernel<<<numberBlocks, BLOCK_SIZE>>>(partMapPtr, treeMapPtr, rcPtr, leaves.data(),
        leaves.size(), indexPtr, nextIdPtr, linkLen, hpxSize(), hpxRank());
    CUDA_CHECK(cudaDeviceSynchronize());
    PRINT("lc blocks = %i next id = %lli\n", numberBlocks, *nextIdPtr);
    size_t rc = *rcPtr;
    *nextIdValue = *nextIdPtr;
    cudaRelease(rcPtr);
    cudaRelease(indexPtr);
    cudaRelease(nextIdPtr);
    return rc;
}
