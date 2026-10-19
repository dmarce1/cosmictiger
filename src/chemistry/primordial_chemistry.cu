/* CosmicTiger - primordial CUDA batching; GPL-2.0-or-later. */
#include <algorithm>
#include <limits>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/chemistry/primordial_chemistry.hpp>

namespace primordial {
namespace detail {
    __global__ void primordialBatchKernel(
        BatchInput const* inputs, StepResult* results, size_t count, SolverOptions options) {
        size_t const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            index < count;) {
            results[index] = advance(
                inputs[index].state, inputs[index].environment, inputs[index].interval, options);
            if (count - index <= stride) {
                break;
            }
            index += stride;
        }
    }
}    // namespace detail

bool integrateBatchCuda(BatchInput const* inputs, StepResult* results, size_t count,
    SolverOptions const& options, std::string* error) {
    if (error != nullptr) {
        error->clear();
    }
    if (count == 0) {
        return true;
    }
    auto fail = [&](char const* message, Status status = Status::BackendFailure) {
        if (inputs != nullptr && results != nullptr) {
            for (size_t index = 0; index < count; ++index) {
                results[index] = StepResult{};
                results[index].state = inputs[index].state;
                results[index].status = status;
            }
        }
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (inputs == nullptr || results == nullptr ||
        count > std::numeric_limits<size_t>::max() / sizeof(BatchInput) ||
        count > std::numeric_limits<size_t>::max() / sizeof(StepResult)) {
        if (error != nullptr) {
            *error = "Invalid primordial CUDA batch buffers";
        }
        return false;
    }
    int device;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return fail(cudaGetErrorString(status), Status::BackendUnavailable);
    }
    BatchInput* deviceInputs = nullptr;
    StepResult* deviceResults = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&deviceInputs), count * sizeof(BatchInput));
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&deviceResults), count * sizeof(StepResult));
    }
    if (status == cudaSuccess) {
        status =
            cudaMemcpy(deviceInputs, inputs, count * sizeof(BatchInput), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        constexpr unsigned kBlockSize = 64;
        unsigned const blocks =
            static_cast<unsigned>(std::min<size_t>(65535, 1 + (count - 1) / kBlockSize));
        detail::primordialBatchKernel<<<blocks, kBlockSize>>>(
            deviceInputs, deviceResults, count, options);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        // Synchronous copy also checks completion; no host fallback is allowed.
        status =
            cudaMemcpy(results, deviceResults, count * sizeof(StepResult), cudaMemcpyDeviceToHost);
    }
    cudaError_t const releaseResults = cudaFree(deviceResults);
    cudaError_t const releaseInputs = cudaFree(deviceInputs);
    if (status == cudaSuccess) {
        status = releaseResults != cudaSuccess ? releaseResults : releaseInputs;
    }
    if (status != cudaSuccess) {
        return fail(cudaGetErrorString(status));
    }
    return true;
}
}    // namespace primordial
