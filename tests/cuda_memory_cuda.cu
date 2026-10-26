/* CosmicTiger - CUDA allocator regressions; BSL-1.0. */

#include <cstddef>
#include <new>
#include <set>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/cuda/cuda_memory.hpp>
#include <tests/cuda_test.hpp>

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

namespace detail {

constexpr int kBlockCount = 2;
constexpr int kThreadsPerBlock = 32;
constexpr int kThreadCount = kBlockCount * kThreadsPerBlock;
constexpr int kInventory = 96;
constexpr int kRounds = 256;
constexpr size_t kRequestBytes = 24;
constexpr size_t kPayloadBytes = 32;
constexpr size_t kBlockBytes = kPayloadBytes + sizeof(size_t);
constexpr size_t kBin = 5;

static_assert(CUDA_MEM_STACK_SIZE >= kInventory);

void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class CudaMemoryOwner {
    CudaMemory* memory_ = nullptr;

public:
    explicit CudaMemoryOwner(size_t bytes) {
        checkCuda(cudaMallocManaged(&memory_, sizeof(CudaMemory)), "allocate pool object");
        try {
            new (memory_) CudaMemory(bytes);
        } catch (...) {
            cudaFree(memory_);
            throw;
        }
    }

    CudaMemoryOwner(CudaMemoryOwner const&) = delete;
    CudaMemoryOwner& operator=(CudaMemoryOwner const&) = delete;

    ~CudaMemoryOwner() {
        memory_->~CudaMemory();
        cudaFree(memory_);
    }

    CudaMemory* get() const {
        return memory_;
    }
};

template <class Value>
class ManagedValue {
    Value* value_ = nullptr;

public:
    ManagedValue() {
        checkCuda(cudaMallocManaged(&value_, sizeof(Value)), "allocate test state");
        new (value_) Value{};
    }

    ManagedValue(ManagedValue const&) = delete;
    ManagedValue& operator=(ManagedValue const&) = delete;

    ~ManagedValue() {
        value_->~Value();
        cudaFree(value_);
    }

    Value* get() const {
        return value_;
    }
};

struct AllocationResults {
    void* pointers[kThreadCount];
    int failures;
};

struct RecycleState {
    void* blocks[kInventory];
    int owners[kInventory];
    int successes[kThreadCount];
    int failures;
};

CUDA_EXPORT void setPayload(void* pointer, unsigned long long value) {
    auto* payload = static_cast<unsigned long long*>(pointer);
    payload[0] = value;
    payload[1] = ~value;
    payload[2] = value + 1;
}

CUDA_EXPORT bool validPayload(void const* pointer) {
    auto const* payload = static_cast<unsigned long long const*>(pointer);
    return payload[1] == ~payload[0] && payload[2] == payload[0] + 1;
}

__global__ void exerciseBoundaries(CudaMemory* memory, AllocationResults* results) {
    // Large requests must neither overflow bin rounding nor consume a small
    // heap. The final small allocation exactly reaches the end of this heap.
    size_t const rejected[] = {
        (size_t{1} << 30) + 1, size_t{1} << 31, (size_t{1} << 31) + 1, ~size_t{0}, 128};
    for (size_t bytes : rejected) {
        if (memory->allocate(bytes) != nullptr) {
            results->failures++;
        }
    }
    for (int index = 0; index < 5; index++) {
        void* pointer = memory->allocate(8);
        results->pointers[index] = pointer;
        if (pointer == nullptr || *(static_cast<size_t*>(pointer) - 1) != 3) {
            results->failures++;
        }
    }
    if (memory->allocate(8) != nullptr) {
        results->failures++;
    }
}

__global__ void allocateFresh(CudaMemory* memory, AllocationResults* results) {
    int const lane = blockIdx.x * blockDim.x + threadIdx.x;
    void* pointer = memory->allocate(kRequestBytes);
    results->pointers[lane] = pointer;
    if (pointer == nullptr) {
        atomicAdd(&results->failures, 1);
        return;
    }
    if (*(static_cast<size_t*>(pointer) - 1) != kBin) {
        atomicAdd(&results->failures, 1);
    }
    setPayload(pointer, lane + 1);
}

__global__ void releaseFresh(CudaMemory* memory, AllocationResults const* results) {
    int const lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (results->pointers[lane] != nullptr) {
        memory->free(results->pointers[lane]);
    }
}

__global__ void recycleConcurrently(CudaMemory* memory, RecycleState* state) {
    int const lane = blockIdx.x * blockDim.x + threadIdx.x;
    for (int round = 0; round < kRounds; round++) {
        void* pointer = memory->allocate(kRequestBytes);
        if (pointer == nullptr) {
            // A producer can have reserved the next queue slot without yet
            // publishing it. tryPop may fail transiently during that window.
            continue;
        }
        int index = 0;
        while (index < kInventory && state->blocks[index] != pointer) {
            index++;
        }
        if (index == kInventory || atomicCAS(&state->owners[index], -1, lane) != -1) {
            atomicAdd(&state->failures, 1);
            return;
        }
        if (*(static_cast<size_t*>(pointer) - 1) != kBin || !validPayload(pointer)) {
            atomicAdd(&state->failures, 1);
        }
        setPayload(pointer, kInventory + 1 + lane * kRounds + round);
        state->successes[lane]++;
        atomicExch(&state->owners[index], -1);
        memory->free(pointer);
    }
}

void finishKernel() {
    checkCuda(cudaGetLastError(), "launch allocator probe");
    checkCuda(cudaDeviceSynchronize(), "complete allocator probe");
}

}    // namespace detail

class CudaMemoryCuda : public detail::CudaTest {};

TEST_F(CudaMemoryCuda, FailedRequestsPreserveCapacityAndAcceptExactFit) {
    detail::CudaMemoryOwner owner(5 * (8 + sizeof(size_t)));
    detail::ManagedValue<detail::AllocationResults> results;
    detail::exerciseBoundaries<<<1, 1>>>(owner.get(), results.get());
    detail::finishKernel();

    EXPECT_EQ(results.get()->failures, 0);
    std::set<void*> allocations;
    for (int index = 0; index < 5; index++) {
        ASSERT_NE(results.get()->pointers[index], nullptr);
        EXPECT_TRUE(allocations.insert(results.get()->pointers[index]).second);
    }
}

TEST_F(CudaMemoryCuda, ConcurrentFreshAllocationsAreInitializedAndDoNotOverlap) {
    detail::CudaMemoryOwner owner(detail::kThreadCount * detail::kBlockBytes);
    detail::ManagedValue<detail::AllocationResults> results;
    detail::allocateFresh<<<detail::kBlockCount, detail::kThreadsPerBlock>>>(
        owner.get(), results.get());
    detail::finishKernel();

    ASSERT_EQ(results.get()->failures, 0);
    std::set<void*> original;
    for (int lane = 0; lane < detail::kThreadCount; lane++) {
        void* pointer = results.get()->pointers[lane];
        ASSERT_NE(pointer, nullptr);
        EXPECT_TRUE(original.insert(pointer).second);
        EXPECT_EQ(*static_cast<unsigned long long*>(pointer), lane + 1);
        EXPECT_TRUE(detail::validPayload(pointer));
    }

    detail::releaseFresh<<<detail::kBlockCount, detail::kThreadsPerBlock>>>(
        owner.get(), results.get());
    detail::finishKernel();
    std::set<void*> recycled;
    for (int index = 0; index < detail::kThreadCount; index++) {
        void* pointer = owner.get()->allocate(detail::kRequestBytes);
        ASSERT_NE(pointer, nullptr);
        EXPECT_TRUE(recycled.insert(pointer).second);
    }
    EXPECT_EQ(recycled, original);
    EXPECT_EQ(owner.get()->allocate(detail::kRequestBytes), nullptr);
}

TEST_F(CudaMemoryCuda, HostBlocksSurviveConcurrentDeviceReuseAndQueueWraparound) {
    detail::CudaMemoryOwner owner(detail::kInventory * detail::kBlockBytes);
    detail::ManagedValue<detail::RecycleState> state;
    std::set<void*> original;
    for (int index = 0; index < detail::kInventory; index++) {
        void* pointer = owner.get()->allocate(detail::kRequestBytes);
        ASSERT_NE(pointer, nullptr);
        EXPECT_TRUE(original.insert(pointer).second);
        state.get()->blocks[index] = pointer;
        state.get()->owners[index] = -1;
        detail::setPayload(pointer, index + 1);
    }
    ASSERT_EQ(owner.get()->allocate(detail::kRequestBytes), nullptr);
    for (int index = 0; index < detail::kInventory; index++) {
        owner.get()->free(state.get()->blocks[index]);
    }

    detail::recycleConcurrently<<<detail::kBlockCount, detail::kThreadsPerBlock>>>(
        owner.get(), state.get());
    detail::finishKernel();
    ASSERT_EQ(state.get()->failures, 0);
    int successes = 0;
    for (int lane = 0; lane < detail::kThreadCount; lane++) {
        EXPECT_GT(state.get()->successes[lane], 0) << "lane " << lane;
        successes += state.get()->successes[lane];
    }
    EXPECT_GE(successes, 4 * CUDA_MEM_STACK_SIZE);

    std::set<void*> recycled;
    for (int index = 0; index < detail::kInventory; index++) {
        EXPECT_EQ(state.get()->owners[index], -1);
        void* pointer = owner.get()->allocate(detail::kRequestBytes);
        ASSERT_NE(pointer, nullptr) << "lost pool block " << index;
        EXPECT_TRUE(recycled.insert(pointer).second);
        EXPECT_TRUE(detail::validPayload(pointer));
    }
    EXPECT_EQ(recycled, original);
    EXPECT_EQ(owner.get()->allocate(detail::kRequestBytes), nullptr);
}
