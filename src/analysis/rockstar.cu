#include <atomic>

#include <cosmictiger/analysis/rockstar.hpp>
#include <cosmictiger/analysis/rockstar_core.hpp>
#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/math/math.hpp>

constexpr int kRockstarBlockSize = 32;

__global__ void rockstarFindSubgroupsKernel(RockstarTree* nodes, int const* leaves,
    RockstarParticle* parts, float linkingLength, int* numberOfChangedLeaves);
__global__ void rockstarAssignLinklenKernel(RockstarTree const* nodes, int const* leaves,
    RockstarParticle* parts, float linkingLength, bool phase);

static std::atomic<int> blocksUsed(0);

namespace detail {

class RockstarBlocksLease {
public:
    explicit RockstarBlocksLease(int count)
      : count_(count) {
        blocksUsed += count_;
    }

    ~RockstarBlocksLease() {
        blocksUsed -= count_;
    }

    RockstarBlocksLease(RockstarBlocksLease const&) = delete;
    RockstarBlocksLease& operator=(RockstarBlocksLease const&) = delete;

private:
    int count_;
};

}    // namespace detail

bool rockstarCudaFree() {
    return blocksUsed < cudaSmpCount();
}

void rockstarAssignLinklenCuda(DeviceVector<RockstarTree> const& nodes,
    DeviceVector<int> const& leaves, DeviceVector<RockstarParticle>& parts, float linkingLength,
    bool phase) {
    if (leaves.size() == 0) {
        return;
    }
    detail::RockstarBlocksLease const blocks(leaves.size());
    CudaStream stream(cudaGetDevice());
    rockstarAssignLinklenKernel<<<leaves.size(), kRockstarBlockSize, 0, stream.nativeHandle()>>>(
        nodes.data(), leaves.data(), parts.data(), linkingLength, phase);
    stream.checkLaunch();
    stream.wait();
}

void rockstarFindSubgroupsCuda(DeviceVector<RockstarTree>& nodes, DeviceVector<int> const& leaves,
    DeviceVector<RockstarParticle>& parts, float linkingLength, int& nextId) {
    if (leaves.size() == 0) {
        return;
    }
    // CUDA connectivity for ROCKSTAR Sec. III.2's Eq. (1) neighbor graph.
    // Label propagation is our implementation of that FOF connectivity.
    // Unique initial labels also represent singletons; the common size filter
    // removes those after connectivity has converged.
    for (int i = 0; i < parts.size(); i++) {
        parts[i].subgroup = nextId++;
    }
    DeviceVector<int> numberOfChangedLeaves(1);
    do {
        numberOfChangedLeaves[0] = 0;
        detail::RockstarBlocksLease const blocks(leaves.size());
        CudaStream stream(cudaGetDevice());
        rockstarFindSubgroupsKernel<<<leaves.size(), kRockstarBlockSize, 0,
            stream.nativeHandle()>>>(
            nodes.data(), leaves.data(), parts.data(), linkingLength, numberOfChangedLeaves.data());
        stream.checkLaunch();
        stream.wait();
    } while (numberOfChangedLeaves[0] != 0);
}

struct AtomicRockstarLabelReader {
    __device__ int operator()(int const* label) const {
        return atomicAdd(const_cast<int*>(label), 0);
    }
};

__global__ void rockstarFindSubgroupsKernel(RockstarTree* nodes, int const* leaves,
    RockstarParticle* parts, float linkingLength, int* numberOfChangedLeaves) {
    int const tid = threadIdx.x;
    auto const& self = nodes[leaves[blockIdx.x]];
    float const linkingLengthSquared = sqr(linkingLength);
    int changed = 0;
    AtomicRockstarLabelReader const readLabel;
    for (int particle = self.particleBegin + tid; particle < self.particleEnd;
        particle += kRockstarBlockSize) {
        int minimum = readLabel(&parts[particle].subgroup);
        for (int ni = 0; ni < self.neighbors.size(); ni++) {
            auto const& other = nodes[self.neighbors[ni]];
            minimum = rockstar::detail::minimumLinkedLabel(parts, particle, other.particleBegin,
                other.particleEnd, linkingLengthSquared, minimum, readLabel);
        }
        int const previous = atomicMin(&parts[particle].subgroup, minimum);
        changed |= minimum < previous;
    }
    sharedReduceAdd<int, kRockstarBlockSize>(changed);
    if (tid == 0 && changed != 0) {
        atomicAdd(numberOfChangedLeaves, 1);
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
    int const dimmax = phase ? 2 * kDimensions : kDimensions;
    for (int ni = 0; ni < neighbors.size(); ni++) {
        int const otheri = neighbors[ni];
        auto const& other = nodes[otheri];
        for (int i = other.particleBegin; i < other.particleEnd; i++) {
            auto const& xb = parts[i].xValue;
            if (rockstar::detail::boxContains(selfbox, xb, dimmax)) {
                for (int j = self.particleBegin + tid; j < self.particleEnd;
                    j += kRockstarBlockSize) {
                    auto const& xa = parts[j].xValue;
                    float dx2 = 0.0f;
                    for (int dim = 0; dim < dimmax; dim++) {
                        dx2 += sqr(xa[dim] - xb[dim]);
                    }
                    if (i != j && dx2 <= linkingLengthSquared) {
                        parts[j].minDist2 = min(dx2, parts[j].minDist2);
                    }
                }
            }
        }
    }
}
