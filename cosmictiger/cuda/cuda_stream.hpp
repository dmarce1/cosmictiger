#pragma once

#ifdef USE_CUDA

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

// HPX ownership stays behind the implementation so CUDA translation units can
// submit their existing kernels without including HPX headers.
class CudaStream {
public:
    explicit CudaStream(int device);
    ~CudaStream() noexcept;

    CudaStream(CudaStream const&) = delete;
    CudaStream& operator=(CudaStream const&) = delete;
    CudaStream(CudaStream&&) = delete;
    CudaStream& operator=(CudaStream&&) = delete;

    // Obtaining the handle marks a launch pending. Check it immediately after
    // launching, then wait before reading results or releasing their storage.
    cudaStream_t nativeHandle() const;
    void copyAsync(void* destination, void const* source, size_t bytes, cudaMemcpyKind kind);
    void prefetchAsync(void const* pointer, size_t bytes, int destinationDevice);
    void checkLaunch();
    void wait();

private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

namespace detail {

void initializeCudaStreamRuntime();
bool cudaStreamEventsInitialized() noexcept;

}    // namespace detail

#endif
