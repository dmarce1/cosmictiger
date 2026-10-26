#include <memory>

#include <cosmictiger/core/fixed_capacity_vector.hpp>
#include <cosmictiger/core/stack_vector.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/cuda/device_vector.hpp>
#include <cosmictiger/analysis/bh.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/math/math.hpp>

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
    Array<float, kBhBucketSize> sinkX;
    Array<float, kBhBucketSize> sinkY;
    Array<float, kBhBucketSize> sinkZ;
};

__global__ void bhTreeEvaluateKernel(BhTreeNode const* treeNodes, int const* sinkBuckets,
    float* phi, Array<float, kDimensions> const* parts, float relativeError, float hsoft, float gm,
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
        for (int i = tid; i < sinkSize; i += kWarpSize) {
            sinkX[i] = parts[self.parts.first + i][kXDimension];
            sinkY[i] = parts[self.parts.first + i][kYDimension];
            sinkZ[i] = parts[self.parts.first + i][kZDimension];
        }
        while (checklist.size()) {
            int maxci = roundUp(checklist.size(), kWarpSize);
            for (int ci = tid; ci < maxci; ci += kWarpSize) {
                bool source = false;
                bool leaf = false;
                bool next = false;
                if (ci < checklist.size()) {
                    auto const& other = treeNodes[checklist[ci]];
                    float const dx = self.pos[kXDimension] - other.pos[kXDimension];
                    float const dy = self.pos[kYDimension] - other.pos[kYDimension];
                    float const dz = self.pos[kZDimension] - other.pos[kZDimension];
                    float const r2 = sqr(dx, dy, dz);
                    if (detail::bhAcceptMonopole(r2, other, myRadius, hsoft, relativeError)) {
                        source = true;
                    } else if (other.children[kLeftChild] == -1) {
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
                    sourceX[index] = other.pos[kXDimension];
                    sourceY[index] = other.pos[kYDimension];
                    sourceZ[index] = other.pos[kZDimension];
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
                    nextlist[index + kLeftChild] = treeNodes[checklist[ci]].children[kLeftChild];
                    nextlist[index + kRightChild] = treeNodes[checklist[ci]].children[kRightChild];
                }
            }
            checklist.resize(nextlist.size());
            for (int ci = tid; ci < nextlist.size(); ci += kWarpSize) {
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
                for (int j = tid + other.parts.first; j < other.parts.second; j += kWarpSize) {
                    int k = j + offset - other.parts.first;
                    sourceX[k] = parts[j][kXDimension];
                    sourceY[k] = parts[j][kYDimension];
                    sourceZ[k] = parts[j][kZDimension];
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
            for (int j = tid; j < sourceX.size(); j += kWarpSize) {
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
    Array<float, kDimensions> const* parts, Array<float, kDimensions> const* sinks,
    float relativeError, float hsoft, float gm, int* nextSink, int numberOfSinks) {
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
        Array<float, kDimensions> const& sink = sinks[si];
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
            int maxci = roundUp(checklist.size(), kWarpSize);
            for (int ci = tid; ci < maxci; ci += kWarpSize) {
                bool source = false;
                bool leaf = false;
                bool next = false;
                if (ci < checklist.size()) {
                    auto const& other = treeNodes[checklist[ci]];
                    float const dx = sink[kXDimension] - other.pos[kXDimension];
                    float const dy = sink[kYDimension] - other.pos[kYDimension];
                    float const dz = sink[kZDimension] - other.pos[kZDimension];
                    float const r2 = sqr(dx, dy, dz);
                    if (detail::bhAcceptMonopole(r2, other, 0.0f, hsoft, relativeError)) {
                        source = true;
                    } else if (other.children[kLeftChild] == -1) {
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
                    sourceX[index] = other.pos[kXDimension];
                    sourceY[index] = other.pos[kYDimension];
                    sourceZ[index] = other.pos[kZDimension];
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
                    nextlist[index + kLeftChild] = treeNodes[checklist[ci]].children[kLeftChild];
                    nextlist[index + kRightChild] = treeNodes[checklist[ci]].children[kRightChild];
                }
            }
            checklist.resize(nextlist.size());
            for (int ci = tid; ci < nextlist.size(); ci += kWarpSize) {
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
                for (int j = tid + other.parts.first; j < other.parts.second; j += kWarpSize) {
                    int k = j + offset - other.parts.first;
                    sourceX[k] = parts[j][kXDimension];
                    sourceY[k] = parts[j][kYDimension];
                    sourceZ[k] = parts[j][kZDimension];
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
        for (int j = tid; j < sourceX.size(); j += kWarpSize) {
            float rinv1, m;
            float const dx = sourceX[j] - sink[kXDimension];
            float const dy = sourceY[j] - sink[kYDimension];
            float const dz = sourceZ[j] - sink[kZDimension];
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
    DeviceVector<Array<float, kDimensions>> const& devX, DeviceVector<int> const devSinkBuckets,
    float relativeError, float hsoft, float gm, int maximumBlockCount) {
    BhWorkspace* workspaces;
    int* nextSinkBucket;
    int zero = 0;
    int numberOfBlocks;
    Timer tm1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberOfBlocks, (void const*) bhTreeEvaluateKernel, kWarpSize, sizeof(BhShmem)));
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
    bhTreeEvaluateKernel<<<numberOfBlocks, kWarpSize, 0, stream.nativeHandle()>>>(
        devTreeNodes.data(), devSinkBuckets.data(), devPhi.data(), devX.data(), relativeError,
        hsoft, gm, nextSinkBucket, devSinkBuckets.size());
    stream.checkLaunch();
    stream.wait();
    tm1.stop();
    return devPhi;
}

DeviceVector<float> bhEvaluatePotentialPointsGpu(DeviceVector<BhTreeNode> const& devTreeNodes,
    DeviceVector<Array<float, kDimensions>> const& devX,
    DeviceVector<Array<float, kDimensions>> const& devY, float relativeError, float hsoft,
    float gm) {
    int* nextSink;
    int zero = 0;
    int numberOfBlocks;
    Timer tm1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberOfBlocks, (void const*) bhTreeEvaluatePointsKernel, kWarpSize, 0));
    numberOfBlocks *= cudaSmpCount();
    numberOfBlocks = std::max(1, std::min(numberOfBlocks, devY.size() / kBhBucketSize));
    std::unique_ptr<int, decltype(&cudaRelease)> counterOwner(
        static_cast<int*>(cudaAllocate(sizeof(int))), cudaRelease);
    nextSink = counterOwner.get();
    *nextSink = 0;
    tm1.start();
    DeviceVector<float> devPhi(devY.size());
    CudaStream stream(cudaGetDevice());
    bhTreeEvaluatePointsKernel<<<numberOfBlocks, kWarpSize, 0, stream.nativeHandle()>>>(
        devTreeNodes.data(), devPhi.data(), devX.data(), devY.data(), relativeError, hsoft, gm,
        nextSink, devY.size());
    stream.checkLaunch();
    stream.wait();
    tm1.stop();
    ALWAYS_ASSERT(devY.size());
    return devPhi;
}
