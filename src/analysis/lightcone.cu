/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <memory>
#include <stdexcept>

#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>

constexpr int kLightconeBlockSize = 32;

__global__ void cudaLightconeKernel(LightconeParticleMapType* particleMapPtr,
    LightconeTreeMapType* treeMapPtr, int* rcPtr, LightconeTreeId const* leaves, int n,
    int* indexPtr, LightconeGroup* nextIdPtr, double linkingLength, int hpxSize, int hpxRank) {
    int const& tid = threadIdx.x;
    __shared__ int index;
    float const linkingLengthSquared = sqr(linkingLength);
    auto& particleMap = *particleMapPtr;
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
            Range<double> mybox = self.box.pad(1.01 * linkingLength);
            int foundAnyLink = false;
            auto const& neighbors = self.neighbors;
            auto& selfParticles = particleMap[self.pix];
            for (int li = 0; li < neighbors.size(); li++) {
                if (thisLeaf != neighbors[li]) {
                    auto const ni = neighbors[li];
                    auto const& other = treeMap[ni.pix][ni.index];
                    auto const& otherParticles = particleMap[other.pix];
                    for (int pi = other.particleRange.first; pi < other.particleRange.second;
                         pi++) {
                        auto const bValue = otherParticles.group[pi];
                        Array<double, kDimensions> b;
                        b[kXDimension] = otherParticles.pos[kXDimension][pi].toDouble();
                        b[kYDimension] = otherParticles.pos[kYDimension][pi].toDouble();
                        b[kZDimension] = otherParticles.pos[kZDimension][pi].toDouble();
                        if (mybox.contains(b)) {
                            for (int pj = self.particleRange.first + tid;
                                pj < self.particleRange.second; pj += kLightconeBlockSize) {
                                auto& a = selfParticles.group[pj];
                                float const dx = distance(selfParticles.pos[kXDimension][pj],
                                    otherParticles.pos[kXDimension][pi]);
                                float const dy = distance(selfParticles.pos[kYDimension][pj],
                                    otherParticles.pos[kYDimension][pi]);
                                float const dz = distance(selfParticles.pos[kZDimension][pj],
                                    otherParticles.pos[kZDimension][pi]);
                                float const r2 = sqr(dx, dy, dz);
                                if (r2 < linkingLengthSquared) {
                                    if (a == kLightconeNoGroup) {
                                        if (bValue != kLightconeNoGroup) {
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
                for (int pj = self.particleRange.first; pj < self.particleRange.second; pj++) {
                    auto& a = selfParticles.group[pj];
                    LightconeGroup minGroup = kLightconeNoGroup;
                    int thisFoundLink = 0;
                    auto const maxpi = roundUp(self.particleRange.second - self.particleRange.first,
                                           kLightconeBlockSize) +
                        self.particleRange.first;
                    for (int pi = self.particleRange.first + tid; pi < maxpi;
                        pi += kLightconeBlockSize) {
                        LightconeGroup thisMinGroup = kLightconeNoGroup;
                        if (pi != pj && pi < self.particleRange.second) {
                            auto const bValue = selfParticles.group[pi];
                            float const dx = distance(selfParticles.pos[kXDimension][pi],
                                selfParticles.pos[kXDimension][pj]);
                            float const dy = distance(selfParticles.pos[kYDimension][pi],
                                selfParticles.pos[kYDimension][pj]);
                            float const dz = distance(selfParticles.pos[kZDimension][pi],
                                selfParticles.pos[kZDimension][pj]);
                            float const r2 = sqr(dx, dy, dz);
                            if (r2 < linkingLengthSquared) {
                                thisMinGroup = bValue;
                                thisFoundLink++;
                            }
                        }
                        sharedReduceMin(thisMinGroup);
                        minGroup = min(minGroup, thisMinGroup);
                    }
                    sharedReduceAdd(thisFoundLink);
                    if (thisFoundLink && (a > minGroup || a == kLightconeNoGroup)) {
                        foundLink = true;
                        foundAnyLink = true;
                        if (tid == 0) {
                            a = minGroup;
                            if (a == kLightconeNoGroup) {
                                a = generateId();
                            }
                            __threadfence();
                        }
                    }
                }
            } while (foundLink);
            sharedReduceAdd(foundAnyLink);
            if (foundAnyLink) {
                for (int li = tid; li < neighbors.size(); li += kLightconeBlockSize) {
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
    LightconeParticleMapType* particleMapPtr, LightconeTreeMapType* treeMapPtr,
    LightconeGroup* nextIdValue, double linkingLength, int rank, int ranks, int maximumBlocks) {
    int numberOfBlocks;
    if (maximumBlocks > 0) {
        numberOfBlocks = maximumBlocks;
    } else {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numberOfBlocks, (void const*) cudaLightconeKernel, kLightconeBlockSize, 0));
        numberOfBlocks *= cudaSmpCount();
    }
    int* indexPtr;
    LightconeGroup* nextIdPtr;
    int* rcPtr;
    std::unique_ptr<LightconeGroup, decltype(&cudaRelease)> nextIdOwner(
        static_cast<LightconeGroup*>(cudaAllocate(sizeof(LightconeGroup))), cudaRelease);
    std::unique_ptr<int, decltype(&cudaRelease)> indexOwner(
        static_cast<int*>(cudaAllocate(sizeof(int))), cudaRelease);
    std::unique_ptr<int, decltype(&cudaRelease)> resultOwner(
        static_cast<int*>(cudaAllocate(sizeof(int))), cudaRelease);
    nextIdPtr = nextIdOwner.get();
    indexPtr = indexOwner.get();
    rcPtr = resultOwner.get();
    *rcPtr = 0;
    *indexPtr = 0;
    *nextIdPtr = *nextIdValue;
    int device;
    auto const status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    CudaStream stream(device);
    cudaLightconeKernel<<<numberOfBlocks, kLightconeBlockSize, 0, stream.nativeHandle()>>>(
        particleMapPtr, treeMapPtr, rcPtr, leaves.data(), leaves.size(), indexPtr, nextIdPtr,
        linkingLength, ranks, rank);
    stream.checkLaunch();
    stream.wait();
    PRINT("lc blocks = %i next id = %lli\n", numberOfBlocks, *nextIdPtr);
    size_t rc = *rcPtr;
    *nextIdValue = *nextIdPtr;
    return rc;
}
