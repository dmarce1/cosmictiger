/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_stream.hpp>
#include <cosmictiger/gravity/nitadori/near_pp.hpp>
#include <cosmictiger/math/kernel.hpp>

namespace pmmm::detail {

constexpr int kNearBlockSize = 128;

__global__ void nearKernel(PmmmNearParticle const* sinks, size_t sinkCount,
    PmmmNearParticle const* sources, PmmmNearSourceRange const* sourceRanges, double cellWidth,
    double inverseSupportRadius, double inverseSupportRadiusCubed, bool doPotential,
    std::array<double, 4>* fields) {
    FlopCounter<> flops;
    for (size_t sink = size_t(blockIdx.x) * blockDim.x + threadIdx.x; sink < sinkCount;
        sink += size_t(blockDim.x) * gridDim.x) {
        std::array<double, 4> field{};
        auto const range = sourceRanges[sink];
        for (size_t source = range.begin; source < range.end; ++source) {
            flops +=
                accumulateNearPair(sinks[sink], sources[source], cellWidth, inverseSupportRadius,
                    inverseSupportRadiusCubed, doPotential, selectedGravityKernel(), field);
        }
        fields[sink] = field;
    }
    addGpuFlops(flops);
}

struct DeviceDeleter {
    int device;

    void operator()(void* pointer) const noexcept {
        int previousDevice = device;
        bool const restoreDevice = ::cudaGetDevice(&previousDevice) == cudaSuccess;
        ::cudaSetDevice(device);
        ::cudaFree(pointer);
        if (restoreDevice) {
            ::cudaSetDevice(previousDevice);
        }
    }
};

template <class Value>
std::unique_ptr<Value, DeviceDeleter> allocateNearBuffer(size_t count, int device) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(Value)) {
        throw std::overflow_error("PMMM CUDA allocation size exceeds the addressable range");
    }
    Value* pointer = nullptr;
    auto const error = cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(Value));
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("PMMM CUDA allocation failed: ") + cudaGetErrorString(error));
    }
    return {pointer, DeviceDeleter{device}};
}

template <class Value>
struct NearDeviceBuffer {
    std::unique_ptr<Value, DeviceDeleter> data;
    size_t capacity = 0;

    explicit NearDeviceBuffer(int device)
      : data(nullptr, DeviceDeleter{device}) {}

    void reserve(size_t count, int device) {
        if (count <= capacity) {
            return;
        }
        size_t const maximum = std::numeric_limits<size_t>::max() / sizeof(Value);
        size_t const nextCapacity =
            std::max(count, capacity <= maximum / 2 ? 2 * capacity : maximum);
        auto next = allocateNearBuffer<Value>(nextCapacity, device);
        data = std::move(next);
        capacity = nextCapacity;
    }
};

struct NearWorkspace {
    int device;
    NearDeviceBuffer<PmmmNearParticle> sinks;
    NearDeviceBuffer<PmmmNearParticle> sources;
    NearDeviceBuffer<PmmmNearSourceRange> sourceRanges;
    NearDeviceBuffer<std::array<double, 4>> fields;
    // Declared last so failed submissions drain before any device buffer dies.
    CudaStream stream;

    explicit NearWorkspace(int selectedDevice)
      : device(selectedDevice)
      , sinks(selectedDevice)
      , sources(selectedDevice)
      , sourceRanges(selectedDevice)
      , fields(selectedDevice)
      , stream(selectedDevice) {}

    void reserve(size_t sinkCount, size_t sourceCount) {
        sinks.reserve(sinkCount, device);
        sources.reserve(sourceCount, device);
        sourceRanges.reserve(sinkCount, device);
        fields.reserve(sinkCount, device);
    }
};

struct NearWorkspacePool {
    std::mutex mutex;
    std::unordered_map<int, std::vector<std::unique_ptr<NearWorkspace>>> available;
};

NearWorkspacePool& nearWorkspacePool() {
    // Explicit quiescent teardown owns the CUDA resources. Do not destroy CUDA
    // objects from process-static destructors after the runtime has shut down.
    static auto* pool = new NearWorkspacePool;
    return *pool;
}

std::unique_ptr<NearWorkspace> acquireNearWorkspace(int device) {
    auto& pool = nearWorkspacePool();
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        auto const found = pool.available.find(device);
        if (found != pool.available.end() && !found->second.empty()) {
            auto workspace = std::move(found->second.back());
            found->second.pop_back();
            return workspace;
        }
    }
    return std::make_unique<NearWorkspace>(device);
}

void releaseNearWorkspace(std::unique_ptr<NearWorkspace> workspace) {
    auto& pool = nearWorkspacePool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    pool.available[workspace->device].push_back(std::move(workspace));
}

void clearPmmmNearCudaWorkspaces() {
    auto& pool = nearWorkspacePool();
    decltype(pool.available) available;
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        available.swap(pool.available);
    }
    // Destruction, device switching, and frees all run outside the pool lock.
}

void pmmmNearFieldCuda(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources,
    std::vector<PmmmNearSourceRange> const& sourceRanges, double cellWidth, double supportRadius,
    bool doPotential, std::vector<std::array<double, 4>>& fields) {
    cudaSetDevice();
    int const device = cudaGetDevice();
    // A lease follows the HPX task, even if it resumes on another OS worker.
    // Only completed successful work returns to the pool; exceptions discard
    // the workspace, whose stream drains before the storage is released.
    auto workspace = acquireNearWorkspace(device);
    workspace->reserve(sinks.size(), sources.size());
    auto& stream = workspace->stream;
    stream.copyAsync(workspace->sinks.data.get(), sinks.data(),
        sinks.size() * sizeof(PmmmNearParticle), cudaMemcpyHostToDevice);
    stream.copyAsync(workspace->sources.data.get(), sources.data(),
        sources.size() * sizeof(PmmmNearParticle), cudaMemcpyHostToDevice);
    stream.copyAsync(workspace->sourceRanges.data.get(), sourceRanges.data(),
        sourceRanges.size() * sizeof(PmmmNearSourceRange), cudaMemcpyHostToDevice);
    double const inverseSupportRadius = 1 / supportRadius;
    double const inverseSupportRadiusCubed =
        inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
    addCpuFlops(3);
    int const blocks = static_cast<int>(std::min(
        size_t(65535), (sinks.size() + size_t(kNearBlockSize) - 1) / size_t(kNearBlockSize)));
    nearKernel<<<blocks, kNearBlockSize, 0, stream.nativeHandle()>>>(workspace->sinks.data.get(),
        sinks.size(), workspace->sources.data.get(), workspace->sourceRanges.data.get(), cellWidth,
        inverseSupportRadius, inverseSupportRadiusCubed, doPotential, workspace->fields.data.get());
    stream.checkLaunch();
    stream.copyAsync(fields.data(), workspace->fields.data.get(), fields.size() * sizeof(fields[0]),
        cudaMemcpyDeviceToHost);
    stream.wait();
    releaseNearWorkspace(std::move(workspace));
}

}    // namespace pmmm::detail
