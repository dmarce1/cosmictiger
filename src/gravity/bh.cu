#include <memory>

#include <cosmictiger/core/fixed_capacity_vector.hpp>
#include <cosmictiger/core/stack_vector.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/cuda/device_vector.hpp>
#include <cosmictiger/gravity/bh.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/math/math.hpp>

#define BH_LIST_SIZE 1024
#define BH_SOURCE_LIST_SIZE (16 * 1024)
#define BH_STACK_SIZE 16384
#define BH_MAX_DEPTH 32
#define BH_WORKSIZE 1024

struct BhWorkspace {
    DeviceVector<int> nextlist;
    DeviceVector<int> leaflist;
    DeviceVector<float> sourceX;
    DeviceVector<float> sourceY;
    DeviceVector<float> sourceZ;
    DeviceVector<float> srcM;
    StackVector<int> checklist;
};

struct BhShmem {
    int si;
    Array<float, BH_BUCKET_SIZE> sinkX;
    Array<float, BH_BUCKET_SIZE> sinkY;
    Array<float, BH_BUCKET_SIZE> sinkZ;
};

__global__ void bhTreeEvaluateKernel(BhTreeNode const* treeNodes, int const* sinkBuckets,
    float* phi, Array<float, NDIM> const* parts, float relativeError, float hsoft, float gm,
    int* nextSinkBucket, int numberOfSinks) {
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    __shared__ BhShmem shmem;
    int& si = shmem.si;
    auto& sinkX = shmem.sinkX;
    auto& sinkY = shmem.sinkY;
    auto& sinkZ = shmem.sinkZ;
    if (tid == 0) {
        si = atomicAdd(nextSinkBucket, 1);
    }
    __syncwarp();
    __shared__ BhWorkspace ws;
    new (&ws) BhWorkspace();
    while (si < numberOfSinks) {
        int sinkIndex = sinkBuckets[si];
        auto& checklist = ws.checklist;
        auto& nextlist = ws.nextlist;
        auto& sourceX = ws.sourceX;
        auto& sourceY = ws.sourceY;
        auto& sourceZ = ws.sourceZ;
        auto& srcM = ws.srcM;
        auto& leaflist = ws.leaflist;
        sourceX.resize(0);
        sourceY.resize(0);
        sourceZ.resize(0);
        srcM.resize(0);
        checklist.resize(1);
        checklist[0] = 0;
        auto const& self = treeNodes[sinkIndex];
        float const myRadius = self.radius;
        int sinkSize = self.parts.second - self.parts.first;
        for (int i = tid; i < sinkSize; i += WARP_SIZE) {
            sinkX[i] = parts[self.parts.first + i][XDIM];
            sinkY[i] = parts[self.parts.first + i][YDIM];
            sinkZ[i] = parts[self.parts.first + i][ZDIM];
        }
        while (checklist.size()) {
            int maxci = roundUp(checklist.size(), WARP_SIZE);
            for (int ci = tid; ci < maxci; ci += WARP_SIZE) {
                bool source = false;
                bool leaf = false;
                bool next = false;
                if (ci < checklist.size()) {
                    auto const& other = treeNodes[checklist[ci]];
                    float const dx = self.pos[XDIM] - other.pos[XDIM];
                    float const dy = self.pos[YDIM] - other.pos[YDIM];
                    float const dz = self.pos[ZDIM] - other.pos[ZDIM];
                    float const r2 = sqr(dx, dy, dz);
                    if (detail::bhAcceptMonopole(r2, other, myRadius, hsoft, relativeError)) {
                        source = true;
                    } else if (other.children[LEFT] == -1) {
                        leaf = true;
                    } else {
                        next = true;
                    }
                }
                int total;
                int index;
                int offset;

                index = source;
                computeIndices(index, total);
                offset = sourceX.size();
                __syncwarp();
                sourceX.resize(offset + total);
                sourceY.resize(offset + total);
                sourceZ.resize(offset + total);
                srcM.resize(offset + total);
                index += offset;
                if (source) {
                    auto const& other = treeNodes[checklist[ci]];
                    sourceX[index] = other.pos[XDIM];
                    sourceY[index] = other.pos[YDIM];
                    sourceZ[index] = other.pos[ZDIM];
                    srcM[index] = other.mass;
                }

                index = leaf;
                computeIndices(index, total);
                offset = leaflist.size();
                __syncwarp();
                leaflist.resize(offset + total);
                index += offset;
                if (leaf) {
                    leaflist[index] = checklist[ci];
                }

                index = next;
                computeIndices(index, total);
                offset = nextlist.size();
                __syncwarp();
                nextlist.resize(offset + 2 * total);
                index = 2 * index + offset;
                if (next) {
                    nextlist[index + LEFT] = treeNodes[checklist[ci]].children[LEFT];
                    nextlist[index + RIGHT] = treeNodes[checklist[ci]].children[RIGHT];
                }
            }
            checklist.resize(nextlist.size());
            for (int ci = tid; ci < nextlist.size(); ci += WARP_SIZE) {
                checklist[ci] = nextlist[ci];
            }

            nextlist.resize(0);
            for (int i = 0; i < leaflist.size(); i++) {
                auto const& other = treeNodes[leaflist[i]];
                int offset = sourceX.size();
                int newSize = offset + other.parts.second - other.parts.first;
                sourceX.resize(newSize);
                sourceY.resize(newSize);
                sourceZ.resize(newSize);
                srcM.resize(newSize);
                for (int j = tid + other.parts.first; j < other.parts.second; j += WARP_SIZE) {
                    int k = j + offset - other.parts.first;
                    sourceX[k] = parts[j][XDIM];
                    sourceY[k] = parts[j][YDIM];
                    sourceZ[k] = parts[j][ZDIM];
                    srcM[k] = 1.0f;
                }
                __syncwarp();
            }
            leaflist.resize(0);
        }

        float const h = hsoft;
        float const hinv = 1.0f / h;
        float const h2 = h * h;
        for (int i = 0; i < sinkSize; i++) {
            float thisPhi = 0.0;
            for (int j = tid; j < sourceX.size(); j += WARP_SIZE) {
                float rinv1, m;
                float const dx = sourceX[j] - sinkX[i];
                float const dy = sourceY[j] - sinkY[i];
                float const dz = sourceZ[j] - sinkZ[i];
                m = srcM[j];
                float const r2 = sqr(dx, dy, dz);
                float const r = sqrt(r2);    // 4
                if (r2 > h2) {
                    rinv1 = float(1) / r;    // 5
                } else {
                    rinv1 = radial_kernel::potential(selectedGravityKernel(), r * hinv) * hinv;
                }
                thisPhi += -m * rinv1;
            }
            sharedReduceAdd(thisPhi);
            if (tid == 0) {
                thisPhi += selfPhi() * hinv;
                phi[i + self.parts.first] = gm * thisPhi;
            }
        }
        if (tid == 0) {
            si = atomicAdd(nextSinkBucket, 1);
        }
        __syncwarp();
    }
    (&ws)->~BhWorkspace();
}

__global__ void bhTreeEvaluatePointsKernel(BhTreeNode const* treeNodes, float* phi,
    Array<float, NDIM> const* parts, Array<float, NDIM> const* sinks, float relativeError,
    float hsoft, float gm, int* nextSink, int numberOfSinks) {
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    __shared__ int si;
    if (tid == 0) {
        si = atomicAdd(nextSink, 1);
    }
    __syncwarp();
    __shared__ BhWorkspace ws;
    new (&ws) BhWorkspace();
    while (si < numberOfSinks) {
        Array<float, NDIM> const& sink = sinks[si];
        auto& checklist = ws.checklist;
        auto& nextlist = ws.nextlist;
        auto& sourceX = ws.sourceX;
        auto& sourceY = ws.sourceY;
        auto& sourceZ = ws.sourceZ;
        auto& srcM = ws.srcM;
        auto& leaflist = ws.leaflist;
        sourceX.resize(0);
        sourceY.resize(0);
        sourceZ.resize(0);
        srcM.resize(0);
        checklist.resize(1);
        checklist[0] = 0;
        while (checklist.size()) {
            int maxci = roundUp(checklist.size(), WARP_SIZE);
            for (int ci = tid; ci < maxci; ci += WARP_SIZE) {
                bool source = false;
                bool leaf = false;
                bool next = false;
                if (ci < checklist.size()) {
                    auto const& other = treeNodes[checklist[ci]];
                    float const dx = sink[XDIM] - other.pos[XDIM];
                    float const dy = sink[YDIM] - other.pos[YDIM];
                    float const dz = sink[ZDIM] - other.pos[ZDIM];
                    float const r2 = sqr(dx, dy, dz);
                    if (detail::bhAcceptMonopole(r2, other, 0.0f, hsoft, relativeError)) {
                        source = true;
                    } else if (other.children[LEFT] == -1) {
                        leaf = true;
                    } else {
                        next = true;
                    }
                }
                int total;
                int index;
                int offset;

                index = source;
                computeIndices(index, total);
                offset = sourceX.size();
                __syncwarp();
                sourceX.resize(offset + total);
                sourceY.resize(offset + total);
                sourceZ.resize(offset + total);
                srcM.resize(offset + total);
                index += offset;
                if (source) {
                    auto const& other = treeNodes[checklist[ci]];
                    sourceX[index] = other.pos[XDIM];
                    sourceY[index] = other.pos[YDIM];
                    sourceZ[index] = other.pos[ZDIM];
                    srcM[index] = other.mass;
                }

                index = leaf;
                computeIndices(index, total);
                offset = leaflist.size();
                __syncwarp();
                leaflist.resize(offset + total);
                index += offset;
                if (leaf) {
                    leaflist[index] = checklist[ci];
                }

                index = next;
                computeIndices(index, total);
                offset = nextlist.size();
                __syncwarp();
                nextlist.resize(offset + 2 * total);
                index = 2 * index + offset;
                if (next) {
                    nextlist[index + LEFT] = treeNodes[checklist[ci]].children[LEFT];
                    nextlist[index + RIGHT] = treeNodes[checklist[ci]].children[RIGHT];
                }
            }
            checklist.resize(nextlist.size());
            for (int ci = tid; ci < nextlist.size(); ci += WARP_SIZE) {
                checklist[ci] = nextlist[ci];
            }

            nextlist.resize(0);
            for (int i = 0; i < leaflist.size(); i++) {
                auto const& other = treeNodes[leaflist[i]];
                int offset = sourceX.size();
                int newSize = offset + other.parts.second - other.parts.first;
                sourceX.resize(newSize);
                sourceY.resize(newSize);
                sourceZ.resize(newSize);
                srcM.resize(newSize);
                for (int j = tid + other.parts.first; j < other.parts.second; j += WARP_SIZE) {
                    int k = j + offset - other.parts.first;
                    sourceX[k] = parts[j][XDIM];
                    sourceY[k] = parts[j][YDIM];
                    sourceZ[k] = parts[j][ZDIM];
                    srcM[k] = 1.0f;
                }
                __syncwarp();
            }
            leaflist.resize(0);
        }

        float const h = hsoft;
        float const hinv = 1.0f / h;
        float const h2 = h * h;
        if (tid == 0) {
            phi[si] = 0.f;
        }
        float thisPhi = 0.0f;
        for (int j = tid; j < sourceX.size(); j += WARP_SIZE) {
            float rinv1, m;
            float const dx = sourceX[j] - sink[XDIM];
            float const dy = sourceY[j] - sink[YDIM];
            float const dz = sourceZ[j] - sink[ZDIM];
            m = srcM[j];
            float const r2 = sqr(dx, dy, dz);
            float const r = sqrt(r2);    // 4
            if (r2 > h2) {
                rinv1 = float(1) / r;    // 5
            } else if (r2 > 0.f) {
                rinv1 = radial_kernel::potential(selectedGravityKernel(), r * hinv) * hinv;
            } else {
                rinv1 = 0.0f;
            }
            thisPhi -= m * rinv1;
        }
        sharedReduceAdd(thisPhi);
        if (tid == 0) {
            phi[si] += thisPhi;
            phi[si] *= gm;
            si = atomicAdd(nextSink, 1);
        }
        __syncwarp();
    }
    (&ws)->~BhWorkspace();
}

DeviceVector<float> bhEvaluatePotentialGpu(DeviceVector<BhTreeNode> const& devTreeNodes,
    DeviceVector<Array<float, NDIM>> const& devX, DeviceVector<int> const devSinkBuckets,
    float relativeError, float hsoft, float gm, int maximumBlockCount) {
    BhWorkspace* workspaces;
    int* nextSinkBucket;
    int zero = 0;
    int numberOfBlocks;
    Timer tm1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberOfBlocks, (void const*) bhTreeEvaluateKernel, WARP_SIZE, sizeof(BhShmem)));
    numberOfBlocks *= cudaSmpCount();
    numberOfBlocks = std::min(numberOfBlocks, devSinkBuckets.size());
    // A bounded launch also lets regression tests exercise multiple buckets
    // per workspace without constructing a device-sized halo.
    if (maximumBlockCount > 0) {
        numberOfBlocks = std::min(numberOfBlocks, maximumBlockCount);
    }
    //	numberOfBlocks *= cuda_smp_count();
    std::unique_ptr<int, decltype(&cudaRelease)> counterOwner(
        static_cast<int*>(cudaAllocate(sizeof(int))), cudaRelease);
    nextSinkBucket = counterOwner.get();
    *nextSinkBucket = 0;
    tm1.start();
    DeviceVector<float> devPhi(devX.size());
    CudaStream stream(cudaGetDevice());
    bhTreeEvaluateKernel<<<numberOfBlocks, WARP_SIZE, 0, stream.nativeHandle()>>>(
        devTreeNodes.data(), devSinkBuckets.data(), devPhi.data(), devX.data(), relativeError,
        hsoft, gm, nextSinkBucket, devSinkBuckets.size());
    stream.checkLaunch();
    stream.wait();
    tm1.stop();
    return devPhi;
}

DeviceVector<float> bhEvaluatePotentialPointsGpu(DeviceVector<BhTreeNode> const& devTreeNodes,
    DeviceVector<Array<float, NDIM>> const& devX, DeviceVector<Array<float, NDIM>> const& devY,
    float relativeError, float hsoft, float gm) {
    int* nextSink;
    int zero = 0;
    int numberOfBlocks;
    Timer tm1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberOfBlocks, (void const*) bhTreeEvaluatePointsKernel, WARP_SIZE, 0));
    numberOfBlocks *= cudaSmpCount();
    numberOfBlocks = std::max(1, std::min(numberOfBlocks, devY.size() / BH_BUCKET_SIZE));
    std::unique_ptr<int, decltype(&cudaRelease)> counterOwner(
        static_cast<int*>(cudaAllocate(sizeof(int))), cudaRelease);
    nextSink = counterOwner.get();
    *nextSink = 0;
    tm1.start();
    DeviceVector<float> devPhi(devY.size());
    CudaStream stream(cudaGetDevice());
    bhTreeEvaluatePointsKernel<<<numberOfBlocks, WARP_SIZE, 0, stream.nativeHandle()>>>(
        devTreeNodes.data(), devPhi.data(), devX.data(), devY.data(), relativeError, hsoft, gm,
        nextSink, devY.size());
    stream.checkLaunch();
    stream.wait();
    tm1.stop();
    ALWAYS_ASSERT(devY.size());
    return devPhi;
}
