/* CosmicTiger - optional resource guard for CUDA validation; BSL-1.0. */
// Build as a preload library with CUDA headers, -fPIC -shared -ldl -lcudart.
// Validation only: serialize launches and cap the two audited work queues.
// Unrecognized oversized grids and extended launches fail before execution.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <cuda_runtime_api.h>
#include <dlfcn.h>

namespace detail {
constexpr unsigned kMaximumBlocks = 16;
std::mutex launchMutex;
size_t launches = 0;
size_t capped = 0;
unsigned maximumBlocks = 0;

[[noreturn]] void reject(char const* reason) {
    // CUDA compiler stubs may discard a launch function's return value. Exit
    // explicitly so a rejected launch cannot look like a passing GPU test.
    std::fprintf(stderr, "CUDA validation guard: %s\n", reason);
    std::exit(86);
}

struct Report {
    ~Report() {
        std::fprintf(stderr,
            "CUDA validation guard: %zu serialized launches, %zu capped, "
            "maximum %u blocks; extended launches disabled\n",
            launches, capped, maximumBlocks);
    }
} report;
template <class Function>
cudaError_t guardedLaunch(char const* symbol, Function function, dim3 grid, dim3 block,
    void** arguments, size_t sharedBytes, cudaStream_t stream) {
    using Launch = cudaError_t (*)(Function, dim3, dim3, void**, size_t, cudaStream_t);
    auto const launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, symbol));
    std::lock_guard<std::mutex> lock(detail::launchMutex);
    if (launch == nullptr) {
        reject("launch entry point is unavailable");
    }
    int device = 0;
    cudaDeviceProp properties{};
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (properties.multiProcessorCount < 4 * detail::kMaximumBlocks) {
        reject("the fixed block bound exceeds one quarter of this device");
    }
    char const* name = nullptr;
    status = cudaFuncGetName(&name, reinterpret_cast<void const*>(function));
    if (status != cudaSuccess) {
        return status;
    }
    size_t const blocks = size_t(grid.x) * grid.y * grid.z;
    if (blocks > detail::kMaximumBlocks) {
        // cudaKickKernel pulls every item from an atomic queue. nearKernel has
        // a grid-stride sink loop. Fewer blocks preserve complete coverage.
        bool const queue = std::strstr(name, "cudaKickKernel") != nullptr ||
            std::strstr(name, "nearKernel") != nullptr;
        if (!queue || grid.y != 1 || grid.z != 1) {
            std::fprintf(
                stderr, "CUDA validation guard rejected %zu blocks for %s\n", blocks, name);
            reject("an oversized kernel has no audited grid-stride loop or work queue");
        }
        grid.x = detail::kMaximumBlocks;
        ++detail::capped;
    }
    detail::maximumBlocks = std::max(detail::maximumBlocks, grid.x * grid.y * grid.z);
    ++detail::launches;
    status = launch(function, grid, block, arguments, sharedBytes, stream);
    return status == cudaSuccess ? cudaDeviceSynchronize() : status;
}

}    // namespace detail

extern "C" cudaError_t cudaLaunchKernel(void const* function, dim3 grid, dim3 block,
    void** arguments, size_t sharedBytes, cudaStream_t stream) {
    return detail::guardedLaunch(
        "cudaLaunchKernel", function, grid, block, arguments, sharedBytes, stream);
}

extern "C" cudaError_t cudaLaunchKernel_ptsz(void const* function, dim3 grid, dim3 block,
    void** arguments, size_t sharedBytes, cudaStream_t stream) {
    return detail::guardedLaunch(
        "cudaLaunchKernel_ptsz", function, grid, block, arguments, sharedBytes, stream);
}

// CUDA 13 compiler-generated stubs use kernel handles through these entry points.
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t function, dim3 grid, dim3 block,
    void** arguments, size_t sharedBytes, cudaStream_t stream) {
    return detail::guardedLaunch(
        "__cudaLaunchKernel", function, grid, block, arguments, sharedBytes, stream);
}

extern "C" cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t function, dim3 grid, dim3 block,
    void** arguments, size_t sharedBytes, cudaStream_t stream) {
    return detail::guardedLaunch(
        "__cudaLaunchKernel_ptsz", function, grid, block, arguments, sharedBytes, stream);
}

extern "C" cudaError_t cudaLaunchKernelExC(cudaLaunchConfig_t const*, void const*, void**) {
    detail::reject("extended launches are disabled");
}

extern "C" cudaError_t cudaLaunchKernelExC_ptsz(cudaLaunchConfig_t const*, void const*, void**) {
    detail::reject("extended launches are disabled");
}
