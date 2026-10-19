/* CosmicTiger - HPX CUDA completion regressions; GPL-2.0-or-later. */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>

#include <cosmictiger/cuda/cuda_stream.hpp>

void cudaAsyncTestLaunch(int* values, int count, cudaStream_t stream, bool invalid);

namespace detail {

constexpr int kCudaAsyncCount = 64;
constexpr size_t kCudaAsyncBytes = kCudaAsyncCount * sizeof(int);

void cudaAsyncRequire(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void cudaAsyncCheck(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct CudaAsyncPinnedDeleter {
    void operator()(int* pointer) const noexcept {
        cudaFreeHost(pointer);
    }
};

struct CudaAsyncDeviceDeleter {
    void operator()(int* pointer) const noexcept {
        cudaFree(pointer);
    }
};

int* cudaAsyncAllocate(bool pinned) {
    void* pointer = nullptr;
    cudaAsyncCheck(
        pinned ? cudaMallocHost(&pointer, kCudaAsyncBytes) : cudaMalloc(&pointer, kCudaAsyncBytes),
        "allocating CUDA completion test storage");
    return static_cast<int*>(pointer);
}

struct CudaAsyncBuffers {
    std::unique_ptr<int, CudaAsyncPinnedDeleter> input{cudaAsyncAllocate(true)};
    std::unique_ptr<int, CudaAsyncPinnedDeleter> output{cudaAsyncAllocate(true)};
    std::unique_ptr<int, CudaAsyncDeviceDeleter> device{cudaAsyncAllocate(false)};

    void reset() {
        for (int index = 0; index < kCudaAsyncCount; ++index) {
            input.get()[index] = 3 * index + 5;
            output.get()[index] = -1;
        }
    }

    void enqueue(CudaStream& stream) {
        stream.copyAsync(device.get(), input.get(), kCudaAsyncBytes, cudaMemcpyHostToDevice);
        cudaAsyncTestLaunch(device.get(), kCudaAsyncCount, stream.nativeHandle(), false);
        stream.checkLaunch();
        stream.copyAsync(output.get(), device.get(), kCudaAsyncBytes, cudaMemcpyDeviceToHost);
    }

    void verify() const {
        for (int index = 0; index < kCudaAsyncCount; ++index) {
            cudaAsyncRequire(output.get()[index] == 6 * index + 17,
                "CUDA completion returned before the kernel and result copy completed");
        }
    }
};

struct CudaAsyncGate {
    std::atomic<bool> released{false};
    std::atomic<bool> timedOut{false};
    bool waiting = false;
};

void CUDART_CB cudaAsyncWaitForHpx(void* argument) {
    auto& gate = *static_cast<CudaAsyncGate*>(argument);
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!gate.released.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            gate.timedOut.store(true, std::memory_order_release);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void cudaAsyncCheckProgress(CudaAsyncBuffers& buffers, bool destroyOtherStream) {
    buffers.reset();
    CudaAsyncGate gate;
    CudaStream stream(0);
    if (destroyOtherStream) {
        // Destroying an unrelated stream must not unregister the polling needed
        // by the surviving stream. Do this before the gate so CUDA stream setup
        // cannot obscure whether the eventual completion wait suspends.
        CudaStream other(0);
        other.wait();
    }
    buffers.enqueue(stream);
    // All submissions that can trigger lazy CUDA loading precede the gate.
    // Only completion waits follow it; the event must include this callback.
    cudaAsyncCheck(cudaLaunchHostFunc(stream.nativeHandle(), cudaAsyncWaitForHpx, &gate),
        "queuing the HPX progress gate");
    auto release = hpx::async(hpx::launch::async, [&gate]() {
        cudaAsyncRequire(gate.waiting, "HPX release task ran before the tested wait");
        gate.released.store(true, std::memory_order_release);
    });
    gate.waiting = true;
    try {
        // With one HPX worker, the release task can only run if this wait yields.
        stream.wait();
    } catch (...) {
        gate.released.store(true, std::memory_order_release);
        release.wait();
        throw;
    }
    release.get();
    cudaAsyncRequire(!gate.timedOut.load(std::memory_order_acquire),
        "CUDA completion blocked the sole HPX worker instead of suspending");
    buffers.verify();
}

void cudaAsyncCheckLaunchRecovery(CudaAsyncBuffers& buffers) {
    buffers.reset();
    CudaStream stream(0);
    cudaAsyncTestLaunch(buffers.device.get(), kCudaAsyncCount, stream.nativeHandle(), true);
    bool rejected = false;
    try {
        stream.checkLaunch();
    } catch (std::exception const&) {
        rejected = true;
    }
    cudaAsyncRequire(rejected, "Invalid CUDA launch was not reported");
    buffers.enqueue(stream);
    stream.wait();
    buffers.verify();
}

struct CudaAsyncExpectedUnwind {};

void cudaAsyncCheckDestructor(CudaAsyncBuffers& buffers, bool unwind) {
    buffers.reset();
    CudaAsyncGate gate;
    hpx::future<void> release;
    try {
        CudaStream stream(0);
        buffers.enqueue(stream);
        cudaAsyncCheck(cudaLaunchHostFunc(stream.nativeHandle(), cudaAsyncWaitForHpx, &gate),
            "queuing the destructor progress gate");
        release = hpx::async(hpx::launch::async, [&gate]() {
            cudaAsyncRequire(gate.waiting, "HPX release task ran before stream destruction");
            gate.released.store(true, std::memory_order_release);
        });
        gate.waiting = true;
        if (unwind) {
            throw CudaAsyncExpectedUnwind{};
        }
        // Completion is deliberately left to the destructor, including while
        // unwinding. It must allow the sole HPX worker to run the release task.
    } catch (CudaAsyncExpectedUnwind const&) {
    }
    release.get();
    cudaAsyncRequire(!gate.timedOut.load(std::memory_order_acquire),
        "CUDA stream destruction blocked the sole HPX worker instead of suspending");
    buffers.verify();
}

void cudaAsyncWarmup(CudaAsyncBuffers& buffers) {
    buffers.reset();
    CudaStream stream(0);
    buffers.enqueue(stream);
    stream.wait();
    buffers.verify();
}

void cudaAsyncStage(char const* label) {
    std::printf("CUDA_ASYNC_CASE %s\n", label);
    std::fflush(stdout);
}

}    // namespace detail

int hpx_main(int, char*[]) {
    int status = 0;
    try {
        detail::cudaAsyncRequire(hpx::get_num_worker_threads() == 1,
            "CUDA completion regression requires exactly one HPX worker");
        detail::cudaAsyncCheck(cudaSetDevice(0), "selecting the CUDA test device");
        detail::CudaAsyncBuffers buffers;
        detail::cudaAsyncStage("warmup");
        detail::cudaAsyncWarmup(buffers);
        detail::cudaAsyncStage("wait_progress");
        detail::cudaAsyncCheckProgress(buffers, false);
        detail::cudaAsyncStage("independent_stream_lifetimes");
        detail::cudaAsyncCheckProgress(buffers, true);
        detail::cudaAsyncStage("launch_error_recovery");
        detail::cudaAsyncCheckLaunchRecovery(buffers);
        detail::cudaAsyncStage("destructor_progress");
        detail::cudaAsyncCheckDestructor(buffers, false);
        detail::cudaAsyncStage("exception_unwind_progress");
        detail::cudaAsyncCheckDestructor(buffers, true);
        std::puts("HPX CUDA completion regressions passed (one HPX worker, one CUDA block)");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "HPX CUDA completion regression failed: %s\n", error.what());
        status = 1;
    }
    hpx::finalize();
    return status;
}

int main(int argc, char* argv[]) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("No CUDA device available");
        return 77;
    }
    std::vector<std::string> configuration{"hpx.os_threads=1", "hpx.stacks.small_size=524288"};
#ifdef HPX_EARLY
    return hpx::init(argc, argv, configuration);
#else
    hpx::init_params parameters;
    parameters.cfg = std::move(configuration);
    return hpx::init(argc, argv, parameters);
#endif
}
