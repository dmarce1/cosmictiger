/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_stream.hpp>
#include <cosmictiger/gravity/nitadori/near_pp.hpp>
#include <cosmictiger/math/kernel.hpp>

namespace pmmm::detail {

constexpr int kNearBlockSize = 128;

__global__ void nearKernel(PmmmNearParticle const* sinks, size_t sinkCount,
    PmmmNearParticle const* sources, size_t sourceCount, double cellWidth,
    double inverseSupportRadius, double inverseSupportRadiusCubed, bool doPotential,
    std::array<double, 4>* fields) {
    FlopCounter<> flops;
    for (size_t sink = size_t(blockIdx.x) * blockDim.x + threadIdx.x; sink < sinkCount;
        sink += size_t(blockDim.x) * gridDim.x) {
        std::array<double, 4> field{};
        for (size_t source = 0; source < sourceCount; ++source) {
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
    Value* pointer = nullptr;
    auto const error = cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(Value));
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("PMMM CUDA allocation failed: ") + cudaGetErrorString(error));
    }
    return {pointer, DeviceDeleter{device}};
}

void pmmmNearFieldCuda(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources, double cellWidth, double supportRadius,
    bool doPotential, std::vector<std::array<double, 4>>& fields) {
    cudaSetDevice();
    int const device = cudaGetDevice();
    auto deviceSinks = allocateNearBuffer<PmmmNearParticle>(sinks.size(), device);
    auto deviceSources = allocateNearBuffer<PmmmNearParticle>(sources.size(), device);
    auto deviceFields = allocateNearBuffer<std::array<double, 4>>(fields.size(), device);
    // The stream is destroyed before its storage owners, including on a failed
    // copy or launch, so no outstanding device operation observes freed memory.
    CudaStream stream(device);
    stream.copyAsync(deviceSinks.get(), sinks.data(), sinks.size() * sizeof(PmmmNearParticle),
        cudaMemcpyHostToDevice);
    stream.copyAsync(deviceSources.get(), sources.data(), sources.size() * sizeof(PmmmNearParticle),
        cudaMemcpyHostToDevice);
    double const inverseSupportRadius = 1 / supportRadius;
    double const inverseSupportRadiusCubed =
        inverseSupportRadius * inverseSupportRadius * inverseSupportRadius;
    addCpuFlops(3);
    int const blocks = static_cast<int>(std::min(
        size_t(65535), (sinks.size() + size_t(kNearBlockSize) - 1) / size_t(kNearBlockSize)));
    nearKernel<<<blocks, kNearBlockSize, 0, stream.nativeHandle()>>>(deviceSinks.get(),
        sinks.size(), deviceSources.get(), sources.size(), cellWidth, inverseSupportRadius,
        inverseSupportRadiusCubed, doPotential, deviceFields.get());
    stream.checkLaunch();
    stream.copyAsync(fields.data(), deviceFields.get(), fields.size() * sizeof(fields[0]),
        cudaMemcpyDeviceToHost);
    stream.wait();
}

}    // namespace pmmm::detail
