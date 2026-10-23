/* CosmicTiger - CUDA managed-memory API compatibility; BSL-1.0. */

#pragma once

#ifdef USE_CUDA

#include <cstddef>

#include <cuda_runtime_api.h>

#if CUDART_VERSION >= 13000
namespace detail {

inline cudaMemLocation cudaMemoryLocation(int device) {
    // CUDA 13 represents host memory by its location type, not device ordinal -1.
    if (device == cudaCpuDeviceId) {
        return {cudaMemLocationTypeHost, 0};
    }
    return {cudaMemLocationTypeDevice, device};
}

}    // namespace detail
#endif

inline cudaError_t cudaAdviseMemory(
    void const* pointer, size_t bytes, cudaMemoryAdvise advice, int device) {
#if CUDART_VERSION >= 13000
    return cudaMemAdvise(pointer, bytes, advice, detail::cudaMemoryLocation(device));
#else
    return cudaMemAdvise(pointer, bytes, advice, device);
#endif
}

inline cudaError_t cudaPrefetchMemoryAsync(
    void const* pointer, size_t bytes, int device, cudaStream_t stream) {
#if CUDART_VERSION >= 13000
    return cudaMemPrefetchAsync(pointer, bytes, detail::cudaMemoryLocation(device), 0, stream);
#else
    return cudaMemPrefetchAsync(pointer, bytes, device, stream);
#endif
}

#endif
