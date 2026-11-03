/* CosmicTiger - P3M direct short-range pairs on CUDA; BSL-1.0. */

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
#include <cosmictiger/gravity/p3m/pp/short_pp.hpp>
#include <cosmictiger/math/kernel.hpp>

namespace cosmictiger::gravity::p3m::pp {

constexpr int kShortBlockSize = 128;

__global__ void shortKernel(ShortParticle const* sinks, size_t sinkCount,
    ShortParticle const* sources, ShortSourceRange const* sourceRanges,
    ShortPairParameters parameters, bool doPotential, std::array<double, 4>* fields,
    double* pairCurvature) {
    FlopCounter<> flops;
    for (size_t sink = size_t(blockIdx.x) * blockDim.x + threadIdx.x; sink < sinkCount;
        sink += size_t(blockDim.x) * gridDim.x) {
        std::array<double, 4> field{};
        double curvature = 0.0;
        auto const range = sourceRanges[sink];
        for (size_t source = range.begin; source < range.end; ++source) {
            flops += accumulateShortPair(sinks[sink], sources[source], parameters, doPotential,
                selectedGravityKernel(), field, &curvature);
        }
        fields[sink] = field;
        pairCurvature[sink] = curvature;
    }
    addGpuFlops(flops, FlopPrecision::Double);
}

namespace {

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
    std::unique_ptr<Value, DeviceDeleter> allocateBuffer(size_t count, int device) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(Value)) {
            throw std::overflow_error("P3M CUDA allocation size exceeds the addressable range");
        }
        Value* pointer = nullptr;
        auto const error = cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(Value));
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("P3M CUDA allocation failed: ") + cudaGetErrorString(error));
        }
        return {pointer, DeviceDeleter{device}};
    }

    template <class Value>
    struct DeviceBuffer {
        std::unique_ptr<Value, DeviceDeleter> data;
        size_t capacity = 0;

        explicit DeviceBuffer(int device)
          : data(nullptr, DeviceDeleter{device}) {}

        void reserve(size_t count, int device) {
            if (count <= capacity) {
                return;
            }
            size_t const maximum = std::numeric_limits<size_t>::max() / sizeof(Value);
            size_t const nextCapacity =
                std::max(count, capacity <= maximum / 2 ? 2 * capacity : maximum);
            auto next = allocateBuffer<Value>(nextCapacity, device);
            data = std::move(next);
            capacity = nextCapacity;
        }
    };

    struct Workspace {
        int device;
        DeviceBuffer<ShortParticle> sinks;
        DeviceBuffer<ShortParticle> sources;
        DeviceBuffer<ShortSourceRange> sourceRanges;
        DeviceBuffer<std::array<double, 4>> fields;
        DeviceBuffer<double> pairCurvature;
        // Declared last so failed submissions drain before any device buffer dies.
        CudaStream stream;

        explicit Workspace(int selectedDevice)
          : device(selectedDevice)
          , sinks(selectedDevice)
          , sources(selectedDevice)
          , sourceRanges(selectedDevice)
          , fields(selectedDevice)
          , pairCurvature(selectedDevice)
          , stream(selectedDevice) {}

        void reserve(size_t sinkCount, size_t sourceCount) {
            sinks.reserve(sinkCount, device);
            sources.reserve(sourceCount, device);
            sourceRanges.reserve(sinkCount, device);
            fields.reserve(sinkCount, device);
            pairCurvature.reserve(sinkCount, device);
        }
    };

    struct WorkspacePool {
        std::mutex mutex;
        std::unordered_map<int, std::vector<std::unique_ptr<Workspace>>> available;
    };

    WorkspacePool& workspacePool() {
        // Explicit quiescent teardown owns the CUDA resources; never destroy CUDA
        // objects from process-static destructors after the runtime has shut down.
        static auto* pool = new WorkspacePool;
        return *pool;
    }

    std::unique_ptr<Workspace> acquireWorkspace(int device) {
        auto& pool = workspacePool();
        {
            std::lock_guard<std::mutex> lock(pool.mutex);
            auto const found = pool.available.find(device);
            if (found != pool.available.end() && !found->second.empty()) {
                auto workspace = std::move(found->second.back());
                found->second.pop_back();
                return workspace;
            }
        }
        return std::make_unique<Workspace>(device);
    }

    void releaseWorkspace(std::unique_ptr<Workspace> workspace) {
        auto& pool = workspacePool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.available[workspace->device].push_back(std::move(workspace));
    }

}    // namespace

void clearShortCudaWorkspaces() {
    auto& pool = workspacePool();
    decltype(pool.available) available;
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        available.swap(pool.available);
    }
}

void shortFieldCuda(std::vector<ShortParticle> const& sinks,
    std::vector<ShortParticle> const& sources, std::vector<ShortSourceRange> const& sourceRanges,
    ShortPairParameters const& parameters, bool doPotential,
    std::vector<std::array<double, 4>>& fields, std::vector<double>& pairCurvature) {
    cudaSetDevice();
    int const device = cudaGetDevice();
    auto workspace = acquireWorkspace(device);
    workspace->reserve(sinks.size(), sources.size());
    auto& stream = workspace->stream;
    stream.copyAsync(workspace->sinks.data.get(), sinks.data(),
        sinks.size() * sizeof(ShortParticle), cudaMemcpyHostToDevice);
    stream.copyAsync(workspace->sources.data.get(), sources.data(),
        sources.size() * sizeof(ShortParticle), cudaMemcpyHostToDevice);
    stream.copyAsync(workspace->sourceRanges.data.get(), sourceRanges.data(),
        sourceRanges.size() * sizeof(ShortSourceRange), cudaMemcpyHostToDevice);
    int const blocks = static_cast<int>(std::min(
        size_t(65535), (sinks.size() + size_t(kShortBlockSize) - 1) / size_t(kShortBlockSize)));
    shortKernel<<<blocks, kShortBlockSize, 0, stream.nativeHandle()>>>(workspace->sinks.data.get(),
        sinks.size(), workspace->sources.data.get(), workspace->sourceRanges.data.get(), parameters,
        doPotential, workspace->fields.data.get(), workspace->pairCurvature.data.get());
    stream.checkLaunch();
    stream.copyAsync(fields.data(), workspace->fields.data.get(), fields.size() * sizeof(fields[0]),
        cudaMemcpyDeviceToHost);
    stream.copyAsync(pairCurvature.data(), workspace->pairCurvature.data.get(),
        pairCurvature.size() * sizeof(double), cudaMemcpyDeviceToHost);
    stream.wait();
    releaseWorkspace(std::move(workspace));
}

}    // namespace cosmictiger::gravity::p3m::pp
