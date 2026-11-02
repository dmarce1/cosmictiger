/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/hpx.hpp>
#ifdef USE_CUDA
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_stream.hpp>
#include <cosmictiger/hydro/sph/sph_cuda.hpp>
#endif
#include <cosmictiger/hydro/sph/sph_cpu.hpp>
#include <cosmictiger/hydro/sph/staged_run.hpp>

namespace detail {

void validateSphStagedRanges(SphRunData const& data, size_t sourceCount, size_t sinkCount,
    size_t treeCount, size_t neighborCount) {
    if (data.numberOfTargetNodes < 0 || !data.treeNodes || !data.targetNodeIndices ||
        !data.sinkParticleIndices || !data.sinkParticleRungs ||
        (neighborCount != 0 && !data.neighborNodeIndices)) {
        throw std::invalid_argument("SPH staged run has missing arrays or a negative target count");
    }
    auto const validRange = [](auto const& range, size_t count) {
        return range.first >= 0 && range.second >= range.first && size_t(range.second) <= count;
    };
    for (size_t sink = 0; sink < sinkCount; ++sink) {
        if (data.sinkParticleIndices[sink] < 0 || size_t(data.sinkParticleIndices[sink]) != sink) {
            throw std::invalid_argument("SPH staged sink indices must be the packed identity map");
        }
    }
    for (int target = 0; target < data.numberOfTargetNodes; ++target) {
        int const nodeIndex = data.targetNodeIndices[target];
        if (nodeIndex < 0 || size_t(nodeIndex) >= treeCount) {
            throw std::out_of_range("SPH staged target node is outside the packed tree");
        }
        auto const& node = data.treeNodes[nodeIndex];
        if (!validRange(node.particleRange, sourceCount) ||
            !validRange(node.sinkParticleRange, sinkCount) ||
            !validRange(node.neighborRange, neighborCount) ||
            node.particleRange.second - node.particleRange.first !=
                node.sinkParticleRange.second - node.sinkParticleRange.first) {
            throw std::out_of_range("SPH staged target has inconsistent source or sink ranges");
        }
        for (int neighbor = node.neighborRange.first; neighbor < node.neighborRange.second;
            ++neighbor) {
            int const neighborIndex = data.neighborNodeIndices[neighbor];
            if (neighborIndex < 0 || size_t(neighborIndex) >= treeCount ||
                !validRange(data.treeNodes[neighborIndex].particleRange, sourceCount)) {
                throw std::out_of_range("SPH staged neighbor is outside the packed source arrays");
            }
        }
    }
}

}    // namespace detail

SphRunReturn sphRunStaged(SphRunParameters const& parameters, SphRunData data, size_t sourceCount,
    size_t sinkCount, size_t treeCount, size_t neighborCount, bool useCuda) {
    if (data.numberOfTargetNodes == 0) {
        return {};
    }
    detail::validateSphStagedRanges(data, sourceCount, sinkCount, treeCount, neighborCount);
    if (!useCuda) {
        int const workerCount =
            std::min(std::max(1, hpxHardwareConcurrency()), data.numberOfTargetNodes);
        if (workerCount == 1) {
            SphRunReturn result;
            for (int target = 0; target < data.numberOfTargetNodes; ++target) {
                result += sphRunCpuNode(parameters, data, target);
            }
            return result;
        }
        std::atomic<int> nextTarget(0);
        std::vector<hpx::future<SphRunReturn>> workers;
        workers.reserve(workerCount);
        try {
            for (int worker = 0; worker < workerCount; ++worker) {
                workers.push_back(hpx::async(hpx::annotated_function(
                    [&]() {
                        SphRunReturn result;
                        for (int target = nextTarget++; target < data.numberOfTargetNodes;
                            target = nextTarget++) {
                            result += sphRunCpuNode(parameters, data, target);
                        }
                        return result;
                    },
                    "sph.staged.cpu")));
            }
        } catch (...) {
            hpx::wait_all(workers.begin(), workers.end());
            throw;
        }
        // All referenced snapshots and stack objects outlive every worker, also
        // when get() propagates a worker's allocation or numerical exception.
        hpx::wait_all(workers.begin(), workers.end());
        SphRunReturn result;
        for (auto& worker : workers) {
            result += worker.get();
        }
        return result;
    }
#ifdef USE_CUDA
    struct SinkCopy {
        void* host;
        void const* device;
        size_t bytes;
    };
    using DeviceOwner = std::unique_ptr<void, decltype(&cudaFree)>;
    std::vector<DeviceOwner> buffers;
    std::vector<SinkCopy> sinkCopies;
    buffers.reserve(48);
    sinkCopies.reserve(16);
    // Last declared, first destroyed: pending H2D/kernel/D2H work is drained
    // before any device buffer is released, including exceptional exits.
    CudaStream stream(cudaGetDevice());
    auto const stageInput = [&](auto*& pointer, size_t count) {
        auto* const host = pointer;
        pointer = nullptr;
        if (!host || count == 0) {
            return;
        }
        if (count > std::numeric_limits<size_t>::max() / sizeof(*host)) {
            throw std::length_error("SPH staged CUDA array size overflows size_t");
        }
        size_t const bytes = sizeof(*host) * count;
        CUDA_CHECK(cudaMalloc(&pointer, bytes));
        // Own the allocation before vector insertion, which can itself fail.
        DeviceOwner owner(pointer, cudaFree);
        buffers.push_back(std::move(owner));
        stream.copyAsync(pointer, host, bytes, cudaMemcpyHostToDevice);
    };
    auto const stageSink = [&](auto*& pointer) {
        auto* const host = pointer;
        stageInput(pointer, sinkCount);
        if (pointer) {
            sinkCopies.push_back({host, pointer, sizeof(*host) * sinkCount});
        }
    };
    stageInput(data.sourcePositionX, sourceCount);
    stageInput(data.sourcePositionY, sourceCount);
    stageInput(data.sourcePositionZ, sourceCount);
    stageInput(data.sourceVelocityX, sourceCount);
    stageInput(data.sourceVelocityY, sourceCount);
    stageInput(data.sourceVelocityZ, sourceCount);
    stageInput(data.sourceInternalEnergy, sourceCount);
    stageInput(data.sourcePressureEnergyRatio, sourceCount);
    stageInput(data.sourceAdiabaticIndex, sourceCount);
    stageInput(data.sourceDensity, sourceCount);
    stageInput(data.sourceBalsaraFactor, sourceCount);
    stageInput(data.sourceViscosityAlpha, sourceCount);
    stageInput(data.sourceAccelerationX, sourceCount);
    stageInput(data.sourceAccelerationY, sourceCount);
    stageInput(data.sourceAccelerationZ, sourceCount);
    stageInput(data.sourceVelocityDivergence, sourceCount);
    stageInput(data.sourceSupportRadius, sourceCount);
    stageInput(data.sourceGradHCorrection, sourceCount);
    stageInput(data.sourceRungs, sourceCount);
    stageInput(data.treeNodes, treeCount);
    stageInput(data.targetNodeIndices, size_t(data.numberOfTargetNodes));
    stageInput(data.neighborNodeIndices, neighborCount);
    stageInput(data.sinkParticleIndices, sinkCount);
    stageInput(data.sinkGravityAccelerationX, sinkCount);
    stageInput(data.sinkGravityAccelerationY, sinkCount);
    stageInput(data.sinkGravityAccelerationZ, sinkCount);
    stageInput(data.sinkGravityTimestep, sinkCount);
    stageSink(data.sinkInternalEnergyRate);
    stageSink(data.sinkDensity);
    stageSink(data.sinkLogDensityRate);
    stageSink(data.sinkGradHCorrection);
    stageSink(data.sinkBalsaraFactor);
    stageSink(data.sinkViscosityStates);
    stageSink(data.sinkCullenDehnenStates);
    stageSink(data.sinkSemiactive);
    stageSink(data.sinkSmoothingConverged);
    stageSink(data.sinkThermodynamicStates);
    stageSink(data.sinkAccelerations);
    stageSink(data.sinkParticleRungs);
    SphRunReturn const result = sphRunCuda(parameters, data, stream);
    for (auto const& copy : sinkCopies) {
        stream.copyAsync(copy.host, copy.device, copy.bytes, cudaMemcpyDeviceToHost);
    }
    stream.wait();
    return result;
#else
    throw std::runtime_error("SPH staged CUDA execution requires a CUDA-enabled build");
#endif
}
