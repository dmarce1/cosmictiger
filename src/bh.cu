#include <cosmictiger/bh.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/fixed_capacity_vector.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/timer.hpp>

#define BH_LIST_SIZE 1024
#define BH_SOURCE_LIST_SIZE (16 * 1024)
#define BH_STACK_SIZE 16384
#define BH_MAX_DEPTH 32
#define BH_WORKSIZE 1024

struct BhWorkspace {
    DeviceVector<int> nextlist;
    DeviceVector<int> leaflist;
    DeviceVector<float> srcX;
    DeviceVector<float> srcY;
    DeviceVector<float> srcZ;
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
    float* phi, Array<float, NDIM> const* parts, float theta, float hsoft, float gm,
    int* nextSinkBucket, int numberSinks) {
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
    while (si < numberSinks) {
        int sinkIndex = sinkBuckets[si];
        auto& checklist = ws.checklist;
        auto& nextlist = ws.nextlist;
        auto& srcX = ws.srcX;
        auto& srcY = ws.srcY;
        auto& srcZ = ws.srcZ;
        auto& srcM = ws.srcM;
        auto& leaflist = ws.leaflist;
        checklist.resize(1);
        checklist[0] = 0;
        auto const& self = treeNodes[sinkIndex];
        float const myRadius = self.radius;
        float const thetainv = 1.0 / theta;
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
                    float const otherRadius = other.radius;
                    float const dx = self.pos[XDIM] - other.pos[XDIM];
                    float const dy = self.pos[YDIM] - other.pos[YDIM];
                    float const dz = self.pos[ZDIM] - other.pos[ZDIM];
                    float const r2 = sqr(dx, dy, dz);
                    if (r2 > sqr(thetainv * (otherRadius + myRadius))) {
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
                offset = srcX.size();
                __syncwarp();
                srcX.resize(offset + total);
                srcY.resize(offset + total);
                srcZ.resize(offset + total);
                srcM.resize(offset + total);
                index += offset;
                if (source) {
                    auto const& other = treeNodes[checklist[ci]];
                    srcX[index] = other.pos[XDIM];
                    srcY[index] = other.pos[YDIM];
                    srcZ[index] = other.pos[ZDIM];
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
                int offset = srcX.size();
                int newSize = offset + other.parts.second - other.parts.first;
                srcX.resize(newSize);
                srcY.resize(newSize);
                srcZ.resize(newSize);
                srcM.resize(newSize);
                for (int j = tid + other.parts.first; j < other.parts.second; j += WARP_SIZE) {
                    int k = j + offset - other.parts.first;
                    srcX[k] = parts[j][XDIM];
                    srcY[k] = parts[j][YDIM];
                    srcZ[k] = parts[j][ZDIM];
                    srcM[k] = 1.0f;
                }
                __syncwarp();
            }
            leaflist.resize(0);
        }

        float const h = hsoft;
        float const hinv = 1.0 / (2.f * h);
        float const h2inv = 1.0 / (4.f * h * h);
        float const h2 = 4.f * h * h;
        for (int i = 0; i < sinkSize; i++) {
            float thisPhi = 0.0;
            for (int j = tid; j < srcX.size(); j += WARP_SIZE) {
                float rinv1, m;
                float const dx = srcX[j] - sinkX[i];
                float const dy = srcY[j] - sinkY[i];
                float const dz = srcZ[j] - sinkZ[i];
                m = srcM[j];
                float const r2 = sqr(dx, dy, dz);
                float const r = sqrt(r2);    // 4
                if (r2 > h2) {
                    rinv1 = float(1) / r;    // 5
                } else {
                    float const r1overh1 = r * hinv;            // 1
                    float const r2oh2 = r1overh1 * r1overh1;    // 1
                    rinv1 = 3.0f / 8.0f;
                    rinv1 = fmaf(rinv1, r2oh2, float(-5.0f / 4.0f));    // 2
                    rinv1 = fmaf(rinv1, r2oh2, float(15.0f / 8.0f));    // 2
                    rinv1 *= hinv;                                      // 1
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
    Array<float, NDIM> const* parts, Array<float, NDIM> const* sinks, float theta, float hsoft,
    float gm, int* nextSink, int numberSinks) {
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    __shared__ int si;
    if (tid == 0) {
        si = atomicAdd(nextSink, 1);
    }
    __syncwarp();
    __shared__ BhWorkspace ws;
    new (&ws) BhWorkspace();
    while (si < numberSinks) {
        Array<float, NDIM> const& sink = sinks[si];
        auto& checklist = ws.checklist;
        auto& nextlist = ws.nextlist;
        auto& srcX = ws.srcX;
        auto& srcY = ws.srcY;
        auto& srcZ = ws.srcZ;
        auto& srcM = ws.srcM;
        auto& leaflist = ws.leaflist;
        checklist.resize(1);
        checklist[0] = 0;
        float const thetainv = 1.0 / theta;
        while (checklist.size()) {
            int maxci = roundUp(checklist.size(), WARP_SIZE);
            for (int ci = tid; ci < maxci; ci += WARP_SIZE) {
                bool source = false;
                bool leaf = false;
                bool next = false;
                if (ci < checklist.size()) {
                    auto const& other = treeNodes[checklist[ci]];
                    float const otherRadius = other.radius;
                    float const dx = sink[XDIM] - other.pos[XDIM];
                    float const dy = sink[YDIM] - other.pos[YDIM];
                    float const dz = sink[ZDIM] - other.pos[ZDIM];
                    float const r2 = sqr(dx, dy, dz);
                    if (r2 > sqr(thetainv * (otherRadius))) {
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
                offset = srcX.size();
                __syncwarp();
                srcX.resize(offset + total);
                srcY.resize(offset + total);
                srcZ.resize(offset + total);
                srcM.resize(offset + total);
                index += offset;
                if (source) {
                    auto const& other = treeNodes[checklist[ci]];
                    srcX[index] = other.pos[XDIM];
                    srcY[index] = other.pos[YDIM];
                    srcZ[index] = other.pos[ZDIM];
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
                int offset = srcX.size();
                int newSize = offset + other.parts.second - other.parts.first;
                srcX.resize(newSize);
                srcY.resize(newSize);
                srcZ.resize(newSize);
                srcM.resize(newSize);
                for (int j = tid + other.parts.first; j < other.parts.second; j += WARP_SIZE) {
                    int k = j + offset - other.parts.first;
                    srcX[k] = parts[j][XDIM];
                    srcY[k] = parts[j][YDIM];
                    srcZ[k] = parts[j][ZDIM];
                    srcM[k] = 1.0f;
                }
                __syncwarp();
            }
            leaflist.resize(0);
        }

        float const h = hsoft;
        float const hinv = 1.0 / (2.f * h);
        float const h2inv = 1.0 / (4.f * h * h);
        float const h2 = 4.f * h * h;
        if (tid == 0) {
            phi[si] = 0.f;
        }
        float thisPhi = 0.0f;
        for (int j = tid; j < srcX.size(); j += WARP_SIZE) {
            float rinv1, m;
            float const dx = srcX[j] - sink[XDIM];
            float const dy = srcY[j] - sink[YDIM];
            float const dz = srcZ[j] - sink[ZDIM];
            m = srcM[j];
            float const r2 = sqr(dx, dy, dz);
            float const r = sqrt(r2);    // 4
            if (r2 > h2) {
                rinv1 = float(1) / r;    // 5
            } else if (r2 > 0.f) {
                float const r1overh1 = r * hinv;            // 1
                float const r2oh2 = r1overh1 * r1overh1;    // 1
                rinv1 = 3.0f / 8.0f;
                rinv1 = fmaf(rinv1, r2oh2, float(-5.0f / 4.0f));    // 2
                rinv1 = fmaf(rinv1, r2oh2, float(15.0f / 8.0f));    // 2
                rinv1 *= hinv;                                      // 1
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
    float theta, float hsoft, float gm) {
    auto stream = cudaGetStream();
    BhWorkspace* workspaces;
    int* nextSinkBucket;
    int zero = 0;
    int numberBlocks;
    Timer tm1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberBlocks, (void const*) bhTreeEvaluateKernel, WARP_SIZE, sizeof(BhShmem)));
    numberBlocks *= cudaSmpCount();
    numberBlocks = std::min(numberBlocks, devSinkBuckets.size());
    //	numberBlocks *= cuda_smp_count();
    nextSinkBucket = (int*) cudaAllocate(sizeof(int));
    *nextSinkBucket = 0;
    tm1.start();
    DeviceVector<float> devPhi(devX.size());
    bhTreeEvaluateKernel<<<numberBlocks, WARP_SIZE, 0, stream>>>(devTreeNodes.data(),
        devSinkBuckets.data(), devPhi.data(), devX.data(), theta, hsoft, gm, nextSinkBucket,
        devSinkBuckets.size());
    synchronizeCudaStream(stream);
    tm1.stop();
    cudaEndStream(stream);
    cudaRelease(nextSinkBucket);
    return devPhi;
}

DeviceVector<float> bhEvaluatePotentialPointsGpu(DeviceVector<BhTreeNode> const& devTreeNodes,
    DeviceVector<Array<float, NDIM>> const& devX, DeviceVector<Array<float, NDIM>> const& devY,
    float theta, float hsoft, float gm) {
    auto stream = cudaGetStream();
    int* nextSink;
    int zero = 0;
    int numberBlocks;
    Timer tm1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numberBlocks, (void const*) bhTreeEvaluateKernel, WARP_SIZE, sizeof(BhShmem)));
    numberBlocks *= cudaSmpCount();
    numberBlocks = std::max(1, std::min(numberBlocks, devY.size() / BH_BUCKET_SIZE));
    nextSink = (int*) cudaAllocate(sizeof(int));
    *nextSink = 0;
    tm1.start();
    DeviceVector<float> devPhi(devY.size());
    bhTreeEvaluatePointsKernel<<<numberBlocks, WARP_SIZE, 0, stream>>>(devTreeNodes.data(),
        devPhi.data(), devX.data(), devY.data(), theta, hsoft, gm, nextSink, devY.size());
    synchronizeCudaStream(stream);
    tm1.stop();
    ALWAYS_ASSERT(devY.size());
    cudaEndStream(stream);
    cudaRelease(nextSink);
    return devPhi;
}
