/* CosmicTiger - CUDA allocator host regressions; BSL-1.0. */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda/atomic>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include <cosmictiger/cuda/cuda_memory.hpp>

// Compile the production implementation and class declaration, with only its
// CUDA/dependency interfaces replaced through the target's include path.
#include "../src/cuda/cuda_memory.cu"

static_assert(CUDA_MEM_STACK_SIZE <= 64,
    "Build this host regression with a small CUDA_MEM_STACK_SIZE to exercise queue wraps");

namespace cuda_memory_test {
namespace detail {
    struct Allocation {
        void* pointer;
        size_t bytes;
        bool mapped;
    };

    std::vector<Allocation> allocations;
    size_t allocationCalls = 0;
    size_t failAllocationCall = 0;
    size_t releaseCalls = 0;
    bool invalidRelease = false;

    void require(bool condition, char const* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    size_t allocationBin(void const* pointer) {
        size_t bin = 0;
        std::memcpy(&bin, static_cast<char const*>(pointer) - sizeof(size_t), sizeof(bin));
        return bin;
    }

    // A stopped thread resumes even when an assertion in the controlling
    // thread fails, so deterministic interleaving failures cannot deadlock.
    class AtomicPause {
        std::mutex mutex_;
        std::condition_variable condition_;
        AtomicOperation operation_;
        bool paused_ = false;
        bool resumed_ = false;

    public:
        explicit AtomicPause(AtomicOperation operation)
          : operation_(operation) {}

        void observe(AtomicEvent const& event) {
            if (event.operation != operation_ || !event.success) {
                return;
            }
            std::unique_lock<std::mutex> lock(mutex_);
            if (paused_) {
                return;
            }
            paused_ = true;
            condition_.notify_all();
            condition_.wait(lock, [&] { return resumed_; });
        }

        bool wait() {
            std::unique_lock<std::mutex> lock(mutex_);
            return condition_.wait_for(lock, std::chrono::seconds(5), [&] { return paused_; });
        }

        void resume() {
            std::lock_guard<std::mutex> lock(mutex_);
            resumed_ = true;
            condition_.notify_all();
        }
    };

    thread_local AtomicPause* currentPause = nullptr;

    void pauseAtomic(AtomicEvent const& event) {
        currentPause->observe(event);
    }

    class PausedThread {
        AtomicPause pause_;
        std::exception_ptr exception_;
        std::thread thread_;

    public:
        template <class Function>
        PausedThread(AtomicOperation operation, Function function)
          : pause_(operation)
          , thread_([this, function] {
              currentPause = &pause_;
              atomicHook = pauseAtomic;
              try {
                  function();
              } catch (...) {
                  exception_ = std::current_exception();
              }
              atomicHook = nullptr;
              currentPause = nullptr;
          }) {}

        ~PausedThread() {
            pause_.resume();
            if (thread_.joinable()) {
                thread_.join();
            }
        }

        void wait() {
            require(pause_.wait(), "worker did not reach the expected atomic operation");
        }

        void finish() {
            pause_.resume();
            thread_.join();
            if (exception_) {
                std::rethrow_exception(exception_);
            }
        }
    };

    thread_local CudaMemory* observedPool = nullptr;
    thread_local size_t observedPublications = 0;

    void inspectPublishedHeader(AtomicEvent const& event) {
        if (event.operation != AtomicOperation::Store) {
            return;
        }
        // Interpose a consumer immediately after publication. Disable its own
        // observer so the recursive allocate/free operations are not intercepted.
        atomicHook = nullptr;
        if (void* pointer = observedPool->allocate(64)) {
            ++observedPublications;
            require(allocationBin(pointer) == 6,
                "consumer observed a free block before its header was initialized");
            observedPool->free(pointer);
        }
        atomicHook = inspectPublishedHeader;
    }

    class HeaderObserver {
    public:
        explicit HeaderObserver(CudaMemory& pool) {
            observedPool = &pool;
            observedPublications = 0;
            atomicHook = inspectPublishedHeader;
        }

        ~HeaderObserver() {
            atomicHook = nullptr;
            observedPool = nullptr;
        }
    };

    void headerBeforePublication() {
        CudaMemory pool(64 + sizeof(size_t));
        HeaderObserver observer(pool);
        // A fresh block may be returned directly, or published with a complete
        // header. Publishing first and writing the header later is caught by
        // the consumer above, without relying on probabilistic thread timing.
        void* pointer = pool.allocate(64);
        require(pointer != nullptr, "fresh allocation failed");
        require(allocationBin(pointer) == 6, "fresh allocation has an invalid header");
        pool.free(pointer);
        require(observedPublications != 0, "header observer did not see a recycled allocation");
        atomicHook = nullptr;
        require(pool.allocate(64) == pointer, "observed publication lost the allocation");
    }

    void exactFitAndFailedReservation() {
        CudaMemory pool(80);
        require(pool.allocate(128) == nullptr, "oversized reservation must fail");
        std::vector<void*> pointers;
        for (int i = 0; i < 5; ++i) {
            void* pointer = pool.allocate(8);
            require(pointer != nullptr, "failed reservation consumed heap capacity or exact fit");
            require(allocationBin(pointer) == 3, "fresh allocation has an uninitialized header");
            pointers.push_back(pointer);
        }
        require(pool.allocate(8) == nullptr, "allocation exceeded heap capacity");
        require(std::unordered_set<void*>(pointers.begin(), pointers.end()).size() == 5,
            "heap reservations overlap");
        for (void* pointer : pointers) {
            pool.free(pointer);
        }
        for (int i = 0; i < 5; ++i) {
            require(pool.allocate(8) != nullptr, "free block was lost after heap exhaustion");
        }
    }

    void largeRequestRounding() {
        size_t const largestBlock = size_t{1} << 31;
        // mmap reserves virtual address space without committing 2 GiB of
        // physical memory. The allocator only writes the small header.
        CudaMemory pool(largestBlock + sizeof(size_t));
        require(pool.allocate(largestBlock + 1) == nullptr, "unsupported bin was accepted");
        require(pool.allocate(std::numeric_limits<size_t>::max()) == nullptr,
            "maximum size_t request was accepted");
        void* pointer = pool.allocate((size_t{1} << 30) + 1);
        require(pointer != nullptr, "request above 1 GiB did not round safely to bin 31");
        require(allocationBin(pointer) == 31, "large request received the wrong bin header");
        require(pool.allocate(8) == nullptr, "2 GiB reservation did not consume its full size");
        pool.free(pointer);
        require(pool.allocate(largestBlock) == pointer, "largest supported block was not reused");
    }

    void concurrentFailedReservation() {
        CudaMemory pool(80);
        void* first = nullptr;
        PausedThread worker(AtomicOperation::CompareExchange, [&] { first = pool.allocate(8); });
        worker.wait();
        require(pool.allocate(64) == nullptr, "non-fitting concurrent reservation was accepted");
        std::vector<void*> others;
        for (int i = 0; i < 4; ++i) {
            void* pointer = pool.allocate(8);
            require(pointer != nullptr, "concurrent failed reservation consumed capacity");
            others.push_back(pointer);
        }
        worker.finish();
        require(first != nullptr, "another reservation invalidated a fitting allocation");
        require(std::find(others.begin(), others.end(), first) == others.end(),
            "concurrent heap reservations overlap");
        require(allocationBin(first) == 3, "delayed reservation returned an invalid header");
        require(pool.allocate(8) == nullptr, "concurrent reservation exceeded heap capacity");
    }

    void delayedProducerAndQueueWrap() {
        CudaMemory pool(2 * (8 + sizeof(size_t)));
        void* first = pool.allocate(8);
        void* second = pool.allocate(8);
        require(first != nullptr && second != nullptr, "test heap could not hold two blocks");
        for (int cycle = 0; cycle < 3 * CUDA_MEM_STACK_SIZE; ++cycle) {
            // The first producer keeps an old enqueue position while the
            // second producer and a consumer advance past that position.
            PausedThread worker(AtomicOperation::Load, [&] { pool.free(first); });
            worker.wait();
            pool.free(second);
            require(pool.allocate(8) == second, "independent producer could not make progress");
            worker.finish();
            require(pool.allocate(8) == first, "delayed producer stranded a free block");
            require(pool.allocate(8) == nullptr, "queue returned an allocated block twice");
        }
    }

    void headersAndConcurrentReuse() {
        constexpr int kThreads = 4;
        constexpr int kIterations = 1000;
        CudaMemory pool(kThreads * (64 + sizeof(size_t)));
        std::mutex mutex;
        std::unordered_set<void*> active;
        std::exception_ptr exception;
        std::vector<std::thread> workers;
        for (int worker = 0; worker < kThreads; ++worker) {
            workers.emplace_back([&] {
                try {
                    for (int iteration = 0; iteration < kIterations; ++iteration) {
                        void* pointer = nullptr;
                        for (int attempt = 0; pointer == nullptr && attempt < 100000; ++attempt) {
                            pointer = pool.allocate(64);
                            if (pointer == nullptr) {
                                std::this_thread::yield();
                            }
                        }
                        require(pointer != nullptr, "concurrent reuse lost available capacity");
                        require(allocationBin(pointer) == 6,
                            "allocation was published before initializing its header");
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            require(active.insert(pointer).second,
                                "same allocation was handed to concurrent callers");
                        }
                        std::this_thread::yield();
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            active.erase(pointer);
                        }
                        pool.free(pointer);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!exception) {
                        exception = std::current_exception();
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        if (exception) {
            std::rethrow_exception(exception);
        }
        std::unordered_set<void*> recovered;
        for (int i = 0; i < kThreads; ++i) {
            void* pointer = pool.allocate(64);
            require(pointer != nullptr && recovered.insert(pointer).second,
                "concurrent free did not return every block exactly once");
        }
        require(pool.allocate(64) == nullptr, "concurrent free duplicated a block");
    }

    class PublishedPoolCleanup {
    public:
        ~PublishedPoolCleanup() {
            failAllocationCall = 0;
            if (memory != nullptr) {
                memory->~CudaMemory();
                cudaFree(memory);
                memory = nullptr;
            }
        }
    };

    void initializationFailures() {
        PublishedPoolCleanup cleanup;
        size_t const originalCount = allocations.size();
        size_t const originalCalls = allocationCalls;
        failAllocationCall = allocationCalls + 1;
        bool threw = false;
        try {
            cudaMemoryInitialize(80);
        } catch (std::exception const&) {
            threw = true;
        }
        require(threw, "pool-object allocation failure was not propagated");
        require(allocationCalls == originalCalls + 1, "failed pool allocation was retried");
        require(memory == nullptr, "failed object allocation published the pool");
        require(allocations.size() == originalCount, "failed object allocation leaked memory");

        failAllocationCall = allocationCalls + 2;
        size_t const beforeHeapFailure = releaseCalls;
        threw = false;
        try {
            cudaMemoryInitialize(80);
        } catch (std::exception const&) {
            threw = true;
        }
        require(threw, "heap allocation failure was not propagated");
        require(memory == nullptr, "failed constructor published a partial pool");
        require(allocations.size() == originalCount, "failed constructor leaked the pool object");
        require(releaseCalls == beforeHeapFailure + 1,
            "failed constructor did not release its pool object exactly once");
        require(!invalidRelease, "failure cleanup released an uninitialized or duplicate pointer");

        failAllocationCall = 0;
        cudaMemoryInitialize(80);
        CudaMemory* const published = memory;
        require(published != nullptr, "successful initialization did not publish the pool");
        require(allocations.size() == originalCount + 2, "successful initialization lost storage");
        void* pointer = cudaAllocate(8);
        require(pointer != nullptr, "pool failed after recoverable initialization errors");
        cudaRelease(pointer);
        require(cudaAllocate(8) == pointer, "global allocator could not recycle storage");
        cudaRelease(pointer);

        for (size_t failure = 1; failure <= 2; ++failure) {
            size_t const beforeCalls = allocationCalls;
            size_t const beforeCount = allocations.size();
            failAllocationCall = allocationCalls + failure;
            threw = false;
            try {
                cudaMemoryInitialize(80);
            } catch (std::exception const&) {
                threw = true;
            }
            require(threw, "replacement pool allocation failure was not propagated");
            require(allocationCalls == beforeCalls + failure, "failed initialization was retried");
            require(memory == published, "failed initialization replaced the live global pool");
            require(allocations.size() == beforeCount, "failed replacement leaked storage");
            require(cudaAllocate(8) == pointer, "failed initialization damaged the live pool");
            cudaRelease(pointer);
        }
        require(!invalidRelease, "initialization cleanup released an invalid pointer");
    }
}    // namespace detail
}    // namespace cuda_memory_test

cudaError_t cudaMallocManaged(void** pointer, size_t bytes) {
    using namespace cuda_memory_test::detail;
    ++allocationCalls;
    if (allocationCalls == failAllocationCall) {
        // CUDA need not provide usable output storage on failure. Leaving this
        // untouched catches code that tries placement-new through that output.
        return cudaErrorMemoryAllocation;
    }
    bool const mapped = bytes > (size_t{1} << 26);
    void* result = mapped ?
        mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) :
        std::malloc(bytes);
    if (result == nullptr || result == MAP_FAILED) {
        return cudaErrorMemoryAllocation;
    }
    if (!mapped) {
        std::memset(result, 0xa5, bytes);
    }
    allocations.push_back({result, bytes, mapped});
    *pointer = result;
    return cudaSuccess;
}

cudaError_t cudaFree(void* pointer) {
    using namespace cuda_memory_test::detail;
    ++releaseCalls;
    auto const found = std::find_if(allocations.begin(), allocations.end(),
        [pointer](Allocation const& allocation) { return allocation.pointer == pointer; });
    if (found == allocations.end()) {
        invalidRelease = true;
        return cudaErrorInvalidValue;
    }
    if (found->mapped) {
        munmap(found->pointer, found->bytes);
    } else {
        std::free(found->pointer);
    }
    allocations.erase(found);
    return cudaSuccess;
}

char const* cudaGetErrorString(cudaError_t error) {
    return error == cudaErrorMemoryAllocation ? "injected allocation failure" : "invalid pointer";
}

bool cudaEnabled() {
    return true;
}

void* allocateManagedMemory(size_t bytes) {
    return std::malloc(bytes);
}

void freeManagedMemory(void* pointer) noexcept {
    std::free(pointer);
}

TEST(CudaMemory, ExactFitAndFailedReservation) {
    cuda_memory_test::detail::exactFitAndFailedReservation();
    EXPECT_TRUE(cuda_memory_test::detail::allocations.empty());
}

TEST(CudaMemory, HeaderBeforePublication) {
    cuda_memory_test::detail::headerBeforePublication();
    EXPECT_TRUE(cuda_memory_test::detail::allocations.empty());
}

TEST(CudaMemory, LargeRequestRounding) {
    cuda_memory_test::detail::largeRequestRounding();
    EXPECT_TRUE(cuda_memory_test::detail::allocations.empty());
}

TEST(CudaMemory, ConcurrentFailedReservation) {
    cuda_memory_test::detail::concurrentFailedReservation();
    EXPECT_TRUE(cuda_memory_test::detail::allocations.empty());
}

TEST(CudaMemory, DelayedProducerAndQueueWrap) {
    cuda_memory_test::detail::delayedProducerAndQueueWrap();
    EXPECT_TRUE(cuda_memory_test::detail::allocations.empty());
}

TEST(CudaMemory, InitializedHeadersAndConcurrentReuse) {
    cuda_memory_test::detail::headersAndConcurrentReuse();
    EXPECT_TRUE(cuda_memory_test::detail::allocations.empty());
}

TEST(CudaMemory, InitializationFailures) {
    cuda_memory_test::detail::initializationFailures();
    EXPECT_TRUE(cuda_memory_test::detail::allocations.empty());
}
