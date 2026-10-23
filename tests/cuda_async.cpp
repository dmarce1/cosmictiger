/* CosmicTiger - HPX CUDA completion regressions; BSL-1.0. */

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
#include <gtest/gtest.h>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>

#include <cosmictiger/cuda/cuda_memory_advice.hpp>
#include <cosmictiger/cuda/cuda_stream.hpp>
#include <tests/cuda_test.hpp>

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

int cudaAsyncMemoryAttribute(void const* pointer, cudaMemRangeAttribute attribute) {
    int value = cudaInvalidDeviceId;
    cudaAsyncCheck(
        cudaMemRangeGetAttribute(&value, sizeof(value), attribute, pointer, kCudaAsyncBytes),
        "querying managed-memory advice or prefetch destination");
    return value;
}

}    // namespace detail

namespace detail {

class CudaAsyncTest : public CudaTest {
protected:
    std::unique_ptr<CudaAsyncBuffers> buffers_;

    void SetUp() override {
        CudaTest::SetUp();
        if (HasFatalFailure() || IsSkipped()) {
            return;
        }
        ASSERT_EQ(hpx::get_num_worker_threads(), 1)
            << "CUDA completion regression requires exactly one HPX worker";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        buffers_ = std::make_unique<CudaAsyncBuffers>();
        cudaAsyncWarmup(*buffers_);
    }
};

TEST_F(CudaAsyncTest, WaitProgress) {
    cudaAsyncCheckProgress(*buffers_, false);
}

TEST_F(CudaAsyncTest, IndependentStreamLifetimes) {
    cudaAsyncCheckProgress(*buffers_, true);
}

TEST_F(CudaAsyncTest, LaunchErrorRecovery) {
    cudaAsyncCheckLaunchRecovery(*buffers_);
}

TEST_F(CudaAsyncTest, DestructorProgress) {
    cudaAsyncCheckDestructor(*buffers_, false);
}

TEST_F(CudaAsyncTest, ExceptionUnwindProgress) {
    cudaAsyncCheckDestructor(*buffers_, true);
}

TEST_F(CudaAsyncTest, ManagedMemoryAdviceAndPrefetch) {
    int concurrentManagedAccess = 0;
    ASSERT_EQ(
        cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, 0),
        cudaSuccess);
    if (!concurrentManagedAccess) {
        GTEST_SKIP() << "Device does not support managed-memory prefetch";
    }

    void* allocation = nullptr;
    ASSERT_EQ(cudaMallocManaged(&allocation, kCudaAsyncBytes), cudaSuccess);
    std::unique_ptr<int, CudaAsyncDeviceDeleter> values(static_cast<int*>(allocation));
    for (int index = 0; index < kCudaAsyncCount; ++index) {
        values.get()[index] = 3 * index + 5;
    }

    ASSERT_EQ(cudaAdviseMemory(values.get(), kCudaAsyncBytes, cudaMemAdviseSetPreferredLocation, 0),
        cudaSuccess);
    EXPECT_EQ(cudaAsyncMemoryAttribute(values.get(), cudaMemRangeAttributePreferredLocation), 0);
    ASSERT_EQ(cudaAdviseMemory(values.get(), kCudaAsyncBytes, cudaMemAdviseSetAccessedBy, 0),
        cudaSuccess);
    ASSERT_EQ(cudaAdviseMemory(values.get(), kCudaAsyncBytes, cudaMemAdviseUnsetAccessedBy, 0),
        cudaSuccess);

    CudaStream stream(0);
    stream.prefetchAsync(values.get(), kCudaAsyncBytes, 0);
    cudaAsyncTestLaunch(values.get(), kCudaAsyncCount, stream.nativeHandle(), false);
    stream.checkLaunch();
    stream.wait();
    EXPECT_EQ(cudaAsyncMemoryAttribute(values.get(), cudaMemRangeAttributeLastPrefetchLocation), 0);

    ASSERT_EQ(cudaAdviseMemory(values.get(), kCudaAsyncBytes, cudaMemAdviseSetPreferredLocation,
                  cudaCpuDeviceId),
        cudaSuccess);
    EXPECT_EQ(cudaAsyncMemoryAttribute(values.get(), cudaMemRangeAttributePreferredLocation),
        cudaCpuDeviceId);
    stream.prefetchAsync(values.get(), kCudaAsyncBytes, cudaCpuDeviceId);
    stream.wait();
    EXPECT_EQ(cudaAsyncMemoryAttribute(values.get(), cudaMemRangeAttributeLastPrefetchLocation),
        cudaCpuDeviceId);
    for (int index = 0; index < kCudaAsyncCount; ++index) {
        EXPECT_EQ(values.get()[index], 6 * index + 17);
    }

    ASSERT_EQ(cudaAdviseMemory(values.get(), kCudaAsyncBytes, cudaMemAdviseUnsetPreferredLocation,
                  cudaCpuDeviceId),
        cudaSuccess);
    EXPECT_EQ(cudaAsyncMemoryAttribute(values.get(), cudaMemRangeAttributePreferredLocation),
        cudaInvalidDeviceId);
}

}    // namespace detail

int hpx_main(int, char*[]) {
    int const status = RUN_ALL_TESTS();
    hpx::finalize();
    return status;
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    if (GTEST_FLAG_GET(list_tests)) {
        return RUN_ALL_TESTS();
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
