/* CosmicTiger - CUDA sampled gravity reference; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/cuda/cuda_stream.hpp>
#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/error_cuda.hpp>
#include <cosmictiger/gravity/reference_table.hpp>

namespace detail {

constexpr int kGravityErrorBlockSize = 128;
constexpr size_t kGravityErrorSourcesPerBlock = 1024;
constexpr size_t kGravityErrorSourceBatch = 65536;
constexpr size_t kGravityErrorSinkBatch = 128;

__global__ void gravityErrorReferenceKernel(GravityErrorPosition const* sinks,
    GravityErrorSource const* sources, size_t sourceCount, radial_kernel::KernelTable const* kernel,
    double supportRadius, GravityErrorForce const* correctionValues, GravityErrorForce* partials) {
    __shared__ double threadForces[3][kGravityErrorBlockSize];
    GravityErrorPosition const sink = sinks[blockIdx.y];
    size_t const sourceBegin = size_t(blockIdx.x) * kGravityErrorSourcesPerBlock;
    size_t const sourceEnd = sourceCount < sourceBegin + kGravityErrorSourcesPerBlock ?
        sourceCount :
        sourceBegin + kGravityErrorSourcesPerBlock;
    std::array<GravityReferenceSum, 3> sum;
    for (size_t source = sourceBegin + threadIdx.x; source < sourceEnd; source += blockDim.x) {
        GravityErrorSource const value = sources[source];
        GravityErrorForce separation;
        for (int dim = 0; dim < 3; ++dim) {
            separation[dim] = doubleDistance(sink[dim], value.position[dim]);
        }
        auto const force =
            tabulatedGravityReference(separation, *kernel, supportRadius, correctionValues);
        for (int dim = 0; dim < 3; ++dim) {
            sum[dim].add(value.mass * force[dim]);
        }
    }
    for (int dim = 0; dim < 3; ++dim) {
        threadForces[dim][threadIdx.x] = sum[dim].value();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // A fixed compensated reduction avoids launch-order-dependent atomics
        // and retains double accuracy when a nearly uniform load cancels.
        for (int dim = 0; dim < 3; ++dim) {
            GravityReferenceSum blockSum;
            for (int thread = 0; thread < kGravityErrorBlockSize; ++thread) {
                blockSum.add(threadForces[dim][thread]);
            }
            partials[size_t(blockIdx.y) * gridDim.x + blockIdx.x][dim] = blockSum.value();
        }
    }
}

struct GravityErrorDeviceDeleter {
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
std::unique_ptr<Value, GravityErrorDeviceDeleter> allocateGravityErrorBuffer(
    size_t count, int device) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(Value)) {
        throw std::overflow_error("Gravity error CUDA allocation exceeds the addressable range");
    }
    Value* pointer = nullptr;
    auto const error = cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(Value));
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Gravity error CUDA allocation failed: ") + cudaGetErrorString(error));
    }
    return {pointer, GravityErrorDeviceDeleter{device}};
}

std::vector<GravityErrorForce> gravityErrorDirectCuda(std::span<GravityErrorPosition const> sinks,
    size_t sourceCount, GravityErrorSourceLoader const& loadSources,
    radial_kernel::KernelTable const& kernel, double supportRadius,
    std::span<GravityErrorForce const> correctionValues) {
    std::vector<GravityErrorForce> result(sinks.size());
    if (sinks.empty() || sourceCount == 0) {
        return result;
    }
    if (!std::isfinite(supportRadius) || supportRadius <= 0) {
        throw std::invalid_argument(
            "Gravity error CUDA support radius must be finite and positive");
    }
    if (correctionValues.size() != kGravityReferenceTableSize) {
        throw std::invalid_argument("Gravity error CUDA correction table has an invalid size");
    }
    size_t const sourceCapacity = std::min(sourceCount, kGravityErrorSourceBatch);
    size_t const sinkCapacity = std::min(sinks.size(), kGravityErrorSinkBatch);
    size_t const blockCapacity =
        (sourceCapacity + kGravityErrorSourcesPerBlock - 1) / kGravityErrorSourcesPerBlock;
    size_t const partialCapacity = sinkCapacity * blockCapacity;
    std::vector<GravityErrorSource> hostSources(sourceCapacity);
    std::vector<GravityErrorForce> hostPartials(partialCapacity);
    std::vector<std::array<GravityReferenceSum, 3>> sums(sinks.size());
    cudaSetDevice();
    int const device = cudaGetDevice();
    auto deviceSinks = allocateGravityErrorBuffer<GravityErrorPosition>(sinkCapacity, device);
    auto deviceSources = allocateGravityErrorBuffer<GravityErrorSource>(sourceCapacity, device);
    auto devicePartials = allocateGravityErrorBuffer<GravityErrorForce>(partialCapacity, device);
    auto deviceKernel = allocateGravityErrorBuffer<radial_kernel::KernelTable>(1, device);
    auto deviceCorrections =
        allocateGravityErrorBuffer<GravityErrorForce>(correctionValues.size(), device);
    // Declared last so all pending transfers drain before host or device buffers
    // are released, including exceptions from source loading or CUDA submission.
    CudaStream stream(device);
    stream.copyAsync(
        deviceKernel.get(), &kernel, sizeof(radial_kernel::KernelTable), cudaMemcpyHostToDevice);
    stream.copyAsync(deviceCorrections.get(), correctionValues.data(),
        correctionValues.size_bytes(), cudaMemcpyHostToDevice);
    for (size_t sourceBegin = 0; sourceBegin < sourceCount;) {
        size_t const batchSources = std::min(sourceCount - sourceBegin, sourceCapacity);
        loadSources(sourceBegin, std::span<GravityErrorSource>(hostSources.data(), batchSources));
        stream.copyAsync(deviceSources.get(), hostSources.data(),
            batchSources * sizeof(GravityErrorSource), cudaMemcpyHostToDevice);
        size_t const blocks =
            (batchSources + kGravityErrorSourcesPerBlock - 1) / kGravityErrorSourcesPerBlock;
        for (size_t sinkBegin = 0; sinkBegin < sinks.size();) {
            size_t const batchSinks = std::min(sinks.size() - sinkBegin, sinkCapacity);
            size_t const partialCount = batchSinks * blocks;
            stream.copyAsync(deviceSinks.get(), sinks.data() + sinkBegin,
                batchSinks * sizeof(GravityErrorPosition), cudaMemcpyHostToDevice);
            dim3 const grid(static_cast<unsigned>(blocks), static_cast<unsigned>(batchSinks));
            gravityErrorReferenceKernel<<<grid, kGravityErrorBlockSize, 0, stream.nativeHandle()>>>(
                deviceSinks.get(), deviceSources.get(), batchSources, deviceKernel.get(),
                supportRadius, deviceCorrections.get(), devicePartials.get());
            stream.checkLaunch();
            stream.copyAsync(hostPartials.data(), devicePartials.get(),
                partialCount * sizeof(GravityErrorForce), cudaMemcpyDeviceToHost);
            stream.wait();
            for (size_t sink = 0; sink < batchSinks; ++sink) {
                for (size_t block = 0; block < blocks; ++block) {
                    auto const& partial = hostPartials[sink * blocks + block];
                    for (int dim = 0; dim < 3; ++dim) {
                        sums[sinkBegin + sink][dim].add(partial[dim]);
                    }
                }
            }
            sinkBegin += batchSinks;
        }
        sourceBegin += batchSources;
    }
    for (size_t sink = 0; sink < sinks.size(); ++sink) {
        for (int dim = 0; dim < 3; ++dim) {
            result[sink][dim] = sums[sink][dim].value();
        }
    }
    return result;
}

}    // namespace detail
