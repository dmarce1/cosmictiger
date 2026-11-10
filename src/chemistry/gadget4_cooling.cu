/* CosmicTiger - GADGET-4 cooling CUDA batch; BSL-1.0. */
#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/chemistry/gadget4_cooling.hpp>

namespace gadget4 {
namespace detail {
    /// Advance independent gas cooling states using a grid-stride device loop.
    __global__ void coolingBatchKernel(
        Input const* inputs, Result* results, size_t count, Parameters parameters) {
        size_t const stride = size_t(blockDim.x) * gridDim.x;
        for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;) {
            results[index] = advance(inputs[index], parameters);
            if (count - index <= stride) {
                break;
            }
            index += stride;
        }
    }
}    // namespace detail

/// Return a cached rate-table allocation for the current device, reporting allocation or
/// transfer failures.
RateRow const* deviceRateTable(std::string* error, Status* resultStatus) {
    if (error != nullptr) {
        error->clear();
    }
    if (resultStatus != nullptr) {
        *resultStatus = Status::Success;
    }
    int device;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        if (resultStatus != nullptr) {
            *resultStatus = status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ?
                Status::BackendUnavailable :
                Status::BackendFailure;
        }
        if (error != nullptr) {
            *error = cudaGetErrorString(status);
        }
        return nullptr;
    }
    static std::mutex mutex;
    // A table lives for the CUDA context lifetime. Deliberately do not call
    // cudaFree from a static destructor after the runtime may have shut down.
    static std::map<int, RateRow*> tables;
    std::lock_guard<std::mutex> lock(mutex);
    auto const found = tables.find(device);
    if (found != tables.end()) {
        return found->second;
    }
    RateRow* table = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&table), sizeof(RateTable));
    if (status == cudaSuccess) {
        status =
            cudaMemcpy(table, hostRateTable().data(), sizeof(RateTable), cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        cudaFree(table);
        if (resultStatus != nullptr) {
            *resultStatus = Status::BackendFailure;
        }
        if (error != nullptr) {
            *error = cudaGetErrorString(status);
        }
        return nullptr;
    }
    try {
        tables.emplace(device, table);
    } catch (std::bad_alloc const&) {
        cudaFree(table);
        if (resultStatus != nullptr) {
            *resultStatus = Status::BackendFailure;
        }
        if (error != nullptr) {
            *error = "Cannot cache the GADGET-4 CUDA rate table";
        }
        return nullptr;
    }
    return table;
}

/// Transfer a cooling batch to the device, execute it, retrieve results, and release
/// temporary buffers.
bool integrateBatchCuda(Input const* inputs, Result* results, size_t count,
    Parameters const& parameters, std::string* error) {
    if (error != nullptr) {
        error->clear();
    }
    if (count == 0) {
        return true;
    }
    auto fail = [&](char const* message, Status failure = Status::BackendFailure) {
        if (inputs != nullptr && results != nullptr) {
            for (size_t index = 0; index < count; ++index) {
                results[index].electronFraction = inputs[index].electronFraction;
                results[index].specificEnergy = inputs[index].specificEnergy;
                results[index].status = failure;
            }
        }
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (inputs == nullptr || results == nullptr ||
        count > std::numeric_limits<size_t>::max() / sizeof(Input) ||
        count > std::numeric_limits<size_t>::max() / sizeof(Result)) {
        if (error != nullptr) {
            *error = "Invalid GADGET-4 CUDA batch buffers";
        }
        return false;
    }
    // The public batch uses the immutable official table on both backends.
    if (parameters.rates != hostRateTable().data()) {
        return fail("GADGET-4 CUDA batch requires the official host rate table");
    }
    Parameters deviceParameters = parameters;
    std::string tableError;
    Status tableStatus;
    deviceParameters.rates = deviceRateTable(&tableError, &tableStatus);
    if (deviceParameters.rates == nullptr) {
        return fail(tableError.c_str(), tableStatus);
    }
    Input* deviceInputs = nullptr;
    Result* deviceResults = nullptr;
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&deviceInputs), count * sizeof(Input));
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&deviceResults), count * sizeof(Result));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(deviceInputs, inputs, count * sizeof(Input), cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        constexpr unsigned kBlockSize = 64;
        unsigned const blocks = unsigned(std::min<size_t>(65535, 1 + (count - 1) / kBlockSize));
        detail::coolingBatchKernel<<<blocks, kBlockSize>>>(
            deviceInputs, deviceResults, count, deviceParameters);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(results, deviceResults, count * sizeof(Result), cudaMemcpyDeviceToHost);
    }
    cudaError_t const releaseResults = cudaFree(deviceResults);
    cudaError_t const releaseInputs = cudaFree(deviceInputs);
    if (status == cudaSuccess) {
        status = releaseResults != cudaSuccess ? releaseResults : releaseInputs;
    }
    return status == cudaSuccess ? true : fail(cudaGetErrorString(status));
}

}    // namespace gadget4
