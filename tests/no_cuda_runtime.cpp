/* CosmicTiger - CPU execution regression; BSL-1.0. */
#include <cstdio>
#include <cstdlib>

// LD_PRELOAD guard for the shared CUDA runtime. Compiler registration entry
// points are deliberately permitted; context, device and allocation APIs are not.
#define FORBID_CUDA(name)                                                 \
    extern "C" int name(...) {                                            \
        std::fprintf(stderr, "Forbidden CUDA runtime call: %s\n", #name); \
        std::_Exit(99);                                                   \
    }
FORBID_CUDA(cudaGetDeviceCount)
FORBID_CUDA(cudaGetDevice)
FORBID_CUDA(cudaSetDevice)
FORBID_CUDA(cudaDeviceReset)
FORBID_CUDA(cudaDeviceSynchronize)
FORBID_CUDA(cudaMalloc)
FORBID_CUDA(cudaMallocManaged)
FORBID_CUDA(cudaHostAlloc)
FORBID_CUDA(cudaMallocHost)
FORBID_CUDA(cudaFree)
FORBID_CUDA(cudaFreeHost)
FORBID_CUDA(cudaMemAdvise)
FORBID_CUDA(cudaMemPrefetchAsync)
FORBID_CUDA(cudaMemcpyToSymbol)
FORBID_CUDA(cudaMemcpy)
FORBID_CUDA(cudaMemcpyAsync)
FORBID_CUDA(cudaMemset)
FORBID_CUDA(cudaStreamCreate)
FORBID_CUDA(cudaStreamCreateWithFlags)
FORBID_CUDA(cudaLaunchKernel)
FORBID_CUDA(cuInit)
#undef FORBID_CUDA
