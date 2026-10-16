#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/fixed_capacity_vector.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/stack_vector.hpp>

#define BLOCK_SIZE 32

__global__ void rockstarFindSubgroupsKernel(RockstarTree* nodes, int const* leaves,
    RockstarParticle* parts, float linkingLength, int* nextIndex, int* numberOfActiveLeaves);
__global__ void rockstarAssignLinklenKernel(RockstarTree const* nodes, int const* leaves,
    RockstarParticle* parts, float linkingLength, bool phase);

static std::atomic<int> blocksUsed(0);

bool rockstarCudaFree() {
    return blocksUsed < cudaSmpCount();
}

void rockstarAssignLinklenCuda(DeviceVector<RockstarTree> const& nodes,
    DeviceVector<int> const& leaves, DeviceVector<RockstarParticle>& parts, float linkingLength,
    bool phase) {
    auto stream = cudaGetStream();
    blocksUsed += leaves.size();
    rockstarAssignLinklenKernel<<<leaves.size(), BLOCK_SIZE, 0, stream>>>(
        nodes.data(), leaves.data(), parts.data(), linkingLength, phase);
    cudaEndStream(stream);
    blocksUsed -= leaves.size();
}

void rockstarFindSubgroupsCuda(DeviceVector<RockstarTree>& nodes, DeviceVector<int> const& inleaves,
    DeviceVector<RockstarParticle>& parts, float linkingLength, int& nextId) {
    int* nextIndex;
    int* numberOfActiveLeaves;
    nextIndex = (int*) cudaAllocate(sizeof(int));
    numberOfActiveLeaves = (int*) cudaAllocate(sizeof(int));
    *nextIndex = nextId;
    int cnt = 0;
    do {
        DeviceVector<int> leaves;
        *numberOfActiveLeaves = 0;
        for (int i = 0; i < inleaves.size(); i++) {
            int const li = inleaves[i];
            nodes[li].lastActive = nodes[li].active;
            if (nodes[li].active) {
                leaves.pushBack(li);
            }
        }
        auto stream = cudaGetStream();
        blocksUsed += leaves.size();
        rockstarFindSubgroupsKernel<<<leaves.size(), BLOCK_SIZE, 0, stream>>>(nodes.data(),
            leaves.data(), parts.data(), linkingLength, nextIndex, numberOfActiveLeaves);
        cudaEndStream(stream);
        blocksUsed -= leaves.size();
        cnt++;
    } while (*numberOfActiveLeaves);
    nextId = *nextIndex;
}

__global__ void rockstarFindSubgroupsKernel(RockstarTree* nodes, int const* leaves,
    RockstarParticle* parts, float linkingLength, int* nextIndex, int* numberOfActiveLeaves) {
    int const& tid = threadIdx.x;
    int const& index = blockIdx.x;
    __syncthreads();
    int const selfi = leaves[index];
    RockstarTree const& self = nodes[selfi];
    auto const selfbox = self.box.pad(1.0001 * linkingLength);
    if (self.lastActive) {
        auto const& neighbors = self.neighbors;
        float const linkingLengthSquared = sqr(linkingLength);
        int foundAnyLink = false;
        auto const generateId = [nextIndex]() { return atomicAdd(nextIndex, 1); };
        for (int ni = 0; ni < neighbors.size(); ni++) {
            int const otheri = neighbors[ni];
            if (otheri != selfi) {
                auto const& other = nodes[otheri];
                for (int i = other.particleBegin; i < other.particleEnd; i++) {
                    auto const& xb = parts[i].xValue;
                    if (selfbox.contains(xb)) {
                        for (int j = self.particleBegin + tid; j < self.particleEnd;
                             j += BLOCK_SIZE) {
                            auto const& xa = parts[j].xValue;
                            float dx2 = 0.0f;
                            for (int dim = 0; dim < 2 * NDIM; dim++) {
                                dx2 += sqr(xa[dim] - xb[dim]);
                            }
                            if (dx2 < linkingLengthSquared) {
                                auto& a = parts[j].subgroup;
                                auto const& b = parts[i].subgroup;
                                if (a == ROCKSTAR_NO_GROUP) {
                                    a = generateId();
                                    foundAnyLink = true;
                                }
                                if (a > b) {
                                    a = b;
                                    foundAnyLink = true;
                                }
                                __threadfence();
                            }
                        }
                    }
                }
            }
        }
        int foundLink;
        int thisFoundLink;
        do {
            foundLink = false;
            for (int i = self.particleBegin; i < self.particleEnd; i++) {
                auto const& xa = parts[i].xValue;
                int const jend =
                    roundUp(self.particleEnd - self.particleBegin, BLOCK_SIZE) + self.particleBegin;
                int bmin = ROCKSTAR_NO_GROUP;
                for (int j = self.particleBegin + tid; j < jend; j += BLOCK_SIZE) {
                    thisFoundLink = false;
                    auto const& xb = parts[j].xValue;
                    auto const& b = parts[j].subgroup;
                    int thisMin = ROCKSTAR_NO_GROUP;
                    if (j != i && j < self.particleEnd) {
                        float dx2 = 0.0f;
                        for (int dim = 0; dim < 2 * NDIM; dim++) {
                            dx2 += sqr(xa[dim] - xb[dim]);
                        }
                        if (dx2 < linkingLengthSquared) {
                            thisMin = min(thisMin, b);
                            thisFoundLink = true;
                        }
                        bmin = min(bmin, thisMin);
                    }
                    sharedReduceMin<BLOCK_SIZE>(thisMin);
                    sharedReduceAdd<int, BLOCK_SIZE>(thisFoundLink);
                }
                auto& a = parts[i].subgroup;
                if (tid == 0) {
                    if (thisFoundLink && a == ROCKSTAR_NO_GROUP && a == bmin) {
                        a = generateId();
                        foundLink = true;
                        foundAnyLink = true;
                    }
                    if (a > bmin) {
                        foundLink = true;
                        a = bmin;
                    }
                    __threadfence();
                }
            }
            sharedReduceAdd(foundLink);
        } while (foundLink);
        sharedReduceAdd(foundAnyLink);
        if (foundAnyLink) {
            for (int i = tid; i < neighbors.size(); i += BLOCK_SIZE) {
                auto& other = nodes[neighbors[i]];
                other.active = true;
            }
            atomicAdd(numberOfActiveLeaves, 1);
        }
    }
}

__global__ void rockstarAssignLinklenKernel(RockstarTree const* nodes, int const* leaves,
    RockstarParticle* parts, float linkingLength, bool phase) {
    int const& tid = threadIdx.x;
    int const& index = blockIdx.x;
    __syncthreads();
    int const selfi = leaves[index];
    RockstarTree const& self = nodes[selfi];
    auto const selfbox = self.box.pad(1.0001 * linkingLength);
    auto const& neighbors = self.neighbors;
    float const linkingLengthSquared = sqr(linkingLength);
    int const dimmax = phase ? 2 * NDIM : NDIM;
    for (int ni = 0; ni < neighbors.size(); ni++) {
        int const otheri = neighbors[ni];
        auto const& other = nodes[otheri];
        for (int i = other.particleBegin; i < other.particleEnd; i++) {
            auto const& xb = parts[i].xValue;
            if (selfbox.contains(xb)) {
                for (int j = self.particleBegin + tid; j < self.particleEnd; j += BLOCK_SIZE) {
                    auto const& xa = parts[j].xValue;
                    float dx2 = 0.0f;
                    for (int dim = 0; dim < dimmax; dim++) {
                        dx2 += sqr(xa[dim] - xb[dim]);
                    }
                    if (dx2 < linkingLengthSquared && dx2 > 0.0f) {
                        parts[j].minDist2 = min(dx2, parts[j].minDist2);
                    }
                }
            }
        }
    }
}
