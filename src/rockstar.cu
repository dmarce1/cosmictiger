#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/stack_vector.hpp>

#define BLOCK_SIZE 32

__global__ void rockstarFindSubgroupsKernel(RockstarTree* nodes, int const* leaves,
    RockstarParticle* parts, float linkLen, int* nextIndex, int* activeCnt);
__global__ void rockstarAssignLinklenKernel(RockstarTree const* nodes, int const* leaves,
    RockstarParticle* parts, float linkLen, bool phase);

static std::atomic<int> blocksUsed(0);

bool rockstarCudaFree() {
    return blocksUsed < cudaSmpCount();
}

void rockstarAssignLinklenCuda(DeviceVector<RockstarTree> const& nodes,
    DeviceVector<int> const& leaves, DeviceVector<RockstarParticle>& parts, float linkLen,
    bool phase) {
    auto stream = cudaGetStream();
    blocksUsed += leaves.size();
    rockstarAssignLinklenKernel<<<leaves.size(), BLOCK_SIZE, 0, stream>>>(
        nodes.data(), leaves.data(), parts.data(), linkLen, phase);
    cudaEndStream(stream);
    blocksUsed -= leaves.size();
}

void rockstarFindSubgroupsCuda(DeviceVector<RockstarTree>& nodes, DeviceVector<int> const& inleaves,
    DeviceVector<RockstarParticle>& parts, float linkLen, int& nextId) {
    int* nextIndex;
    int* activeCnt;
    nextIndex = (int*) cudaAllocate(sizeof(int));
    activeCnt = (int*) cudaAllocate(sizeof(int));
    *nextIndex = nextId;
    int cnt = 0;
    do {
        DeviceVector<int> leaves;
        *activeCnt = 0;
        for (int i = 0; i < inleaves.size(); i++) {
            int const li = inleaves[i];
            nodes[li].lastActive = nodes[li].active;
            if (nodes[li].active) {
                leaves.pushBack(li);
            }
        }
        auto stream = cudaGetStream();
        blocksUsed += leaves.size();
        rockstarFindSubgroupsKernel<<<leaves.size(), BLOCK_SIZE, 0, stream>>>(
            nodes.data(), leaves.data(), parts.data(), linkLen, nextIndex, activeCnt);
        cudaEndStream(stream);
        blocksUsed -= leaves.size();
        cnt++;
    } while (*activeCnt);
    nextId = *nextIndex;
}

__global__ void rockstarFindSubgroupsKernel(RockstarTree* nodes, int const* leaves,
    RockstarParticle* parts, float linkLen, int* nextIndex, int* activeCnt) {
    int const& tid = threadIdx.x;
    int const& index = blockIdx.x;
    __syncthreads();
    int const selfi = leaves[index];
    RockstarTree const& self = nodes[selfi];
    auto const selfbox = self.box.pad(1.0001 * linkLen);
    if (self.lastActive) {
        auto const& neighbors = self.neighbors;
        float const linkLen2 = sqr(linkLen);
        int foundAnyLink = false;
        auto const generateId = [nextIndex]() { return atomicAdd(nextIndex, 1); };
        for (int ni = 0; ni < neighbors.size(); ni++) {
            int const otheri = neighbors[ni];
            if (otheri != selfi) {
                auto const& other = nodes[otheri];
                for (int i = other.partBegin; i < other.partEnd; i++) {
                    auto const& xb = parts[i].xValue;
                    if (selfbox.contains(xb)) {
                        for (int j = self.partBegin + tid; j < self.partEnd; j += BLOCK_SIZE) {
                            auto const& xa = parts[j].xValue;
                            float dx2 = 0.0f;
                            for (int dim = 0; dim < 2 * NDIM; dim++) {
                                dx2 += sqr(xa[dim] - xb[dim]);
                            }
                            if (dx2 < linkLen2) {
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
            for (int i = self.partBegin; i < self.partEnd; i++) {
                auto const& xa = parts[i].xValue;
                int const jend =
                    roundUp(self.partEnd - self.partBegin, BLOCK_SIZE) + self.partBegin;
                int bmin = ROCKSTAR_NO_GROUP;
                for (int j = self.partBegin + tid; j < jend; j += BLOCK_SIZE) {
                    thisFoundLink = false;
                    auto const& xb = parts[j].xValue;
                    auto const& b = parts[j].subgroup;
                    int thisMin = ROCKSTAR_NO_GROUP;
                    if (j != i && j < self.partEnd) {
                        float dx2 = 0.0f;
                        for (int dim = 0; dim < 2 * NDIM; dim++) {
                            dx2 += sqr(xa[dim] - xb[dim]);
                        }
                        if (dx2 < linkLen2) {
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
            atomicAdd(activeCnt, 1);
        }
    }
}

__global__ void rockstarAssignLinklenKernel(RockstarTree const* nodes, int const* leaves,
    RockstarParticle* parts, float linkLen, bool phase) {
    int const& tid = threadIdx.x;
    int const& index = blockIdx.x;
    __syncthreads();
    int const selfi = leaves[index];
    RockstarTree const& self = nodes[selfi];
    auto const selfbox = self.box.pad(1.0001 * linkLen);
    auto const& neighbors = self.neighbors;
    float const linkLen2 = sqr(linkLen);
    int const dimmax = phase ? 2 * NDIM : NDIM;
    for (int ni = 0; ni < neighbors.size(); ni++) {
        int const otheri = neighbors[ni];
        auto const& other = nodes[otheri];
        for (int i = other.partBegin; i < other.partEnd; i++) {
            auto const& xb = parts[i].xValue;
            if (selfbox.contains(xb)) {
                for (int j = self.partBegin + tid; j < self.partEnd; j += BLOCK_SIZE) {
                    auto const& xa = parts[j].xValue;
                    float dx2 = 0.0f;
                    for (int dim = 0; dim < dimmax; dim++) {
                        dx2 += sqr(xa[dim] - xb[dim]);
                    }
                    if (dx2 < linkLen2 && dx2 > 0.0f) {
                        parts[j].minDist2 = min(dx2, parts[j].minDist2);
                    }
                }
            }
        }
    }
}
