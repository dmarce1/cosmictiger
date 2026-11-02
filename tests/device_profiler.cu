// CosmicTiger - bounded CUDA device-profiler regression; BSL-1.0.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

#include <cosmictiger/cuda/device_profiler.hpp>

namespace detail {

constexpr int kWarpSize = 32;
constexpr int kRecordCount = 2;
constexpr std::uint64_t kSentinelCycles = 0x123456789abcdef0ULL;
constexpr std::uint64_t kSentinelCalls = 0xabcdef0123456789ULL;

#if !COSMICTIGER_PROFILE || !COSMICTIGER_DEVICE_PROFILE
static_assert(std::is_empty<profile::DeviceProfiler>::value,
    "The compiled-out device profiler must not retain per-thread state");
#endif

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        cudaFree(pointer_);
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    T* data() const {
        return pointer_;
    }

private:
    T* pointer_ = nullptr;
};

__host__ __device__ std::uint32_t advance(std::uint32_t value, unsigned visit) {
    for (int step = 0; step < 32; ++step) {
        value = value * 1664525U + 1013904223U + visit;
    }
    return value;
}

__global__ void exerciseProfiler(
    profile::DeviceBlockRecord* records, std::uint32_t* output, unsigned pass, bool enabled) {
    unsigned const block = blockIdx.x;
    unsigned const lane = threadIdx.x;
    profile::DeviceProfiler timer(
        enabled ? records + block : nullptr, profile::DevicePhase::KickPrepare);
    std::uint32_t value = lane + 1U + 17U * block + 23U * pass;
    unsigned const visits = block + pass + 1U;
    timer.next(profile::DevicePhase::KickTraversal);
    for (unsigned visit = 0; visit < visits; ++visit) {
        timer.next(profile::DevicePhase::KickP2P);
        value = advance(value, visit);
        timer.next(profile::DevicePhase::KickTraversal);
        value ^= visit + 0x9e3779b9U;
    }
    timer.next(profile::DevicePhase::KickCopyWait);
    // Deliberately empty: entered phases count even when they contain no work.
    timer.next(profile::DevicePhase::KickLeafKick);
    output[block * kWarpSize + lane] = value;
    timer.finish();
    timer.finish();
    timer.next(profile::DevicePhase::KickCleanup);
    timer.finish();
}

std::uint32_t expectedOutput(unsigned block, unsigned lane, unsigned pass) {
    std::uint32_t value = lane + 1U + 17U * block + 23U * pass;
    for (unsigned visit = 0; visit < block + pass + 1U; ++visit) {
        value = advance(value, visit);
        value ^= visit + 0x9e3779b9U;
    }
    return value;
}

void checkOutput(std::array<std::uint32_t, kRecordCount * kWarpSize> const& output, unsigned block,
    unsigned pass) {
    for (unsigned lane = 0; lane < kWarpSize; ++lane) {
        require(output[block * kWarpSize + lane] == expectedOutput(block, lane, pass),
            "Profiling changed the kernel's data output");
    }
}

void checkEqual(
    profile::DeviceBlockRecord const& actual, profile::DeviceBlockRecord const& expected) {
    for (size_t phase = 0; phase < profile::kDevicePhaseCount; ++phase) {
        require(actual.phases[phase].cycles == expected.phases[phase].cycles &&
                actual.phases[phase].calls == expected.phases[phase].calls,
            "An inactive profiler modified a block record");
    }
}

void checkCounters(profile::DeviceBlockRecord const& actual, unsigned block, unsigned pass,
    profile::DeviceBlockRecord const& sentinel) {
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    (void) sentinel;
    unsigned const visits = block + pass + 1U;
    for (size_t phase = 0; phase < profile::kDevicePhaseCount; ++phase) {
        std::uint64_t expectedCalls = 0;
        switch (static_cast<profile::DevicePhase>(phase)) {
        case profile::DevicePhase::KickPrepare:
        case profile::DevicePhase::KickCopyWait:
        case profile::DevicePhase::KickLeafKick:
            expectedCalls = 1;
            break;
        case profile::DevicePhase::KickTraversal:
            expectedCalls = visits + 1U;
            break;
        case profile::DevicePhase::KickP2P:
            expectedCalls = visits;
            break;
        default:
            break;
        }
        require(actual.phases[phase].calls == expectedCalls, "Incorrect phase invocation count");
        require(
            expectedCalls == 0 ? actual.phases[phase].cycles == 0 : actual.phases[phase].cycles > 0,
            "Incorrect empty or active phase duration");
    }
#else
    (void) block;
    (void) pass;
    checkEqual(actual, sentinel);
#endif
}

int run() {
    int deviceCount = 0;
    cudaError_t const status = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && deviceCount == 0)) {
        std::fprintf(stderr, "No usable CUDA device: %s\n", cudaGetErrorString(status));
        return 77;
    }
    checkCuda(status);
    checkCuda(cudaSetDevice(0));
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0));
    int const blockCount = std::min(kRecordCount, properties.multiProcessorCount / 6);
    if (blockCount < kRecordCount) {
        std::fprintf(stderr,
            "Device profiler test needs two blocks within the one-sixth GPU cap (%d SMs)\n",
            properties.multiProcessorCount);
        return 77;
    }

    std::array<profile::DeviceBlockRecord, kRecordCount> records{};
    for (auto& record : records) {
        for (size_t phase = 0; phase < profile::kDevicePhaseCount; ++phase) {
            record.phases[phase].cycles = kSentinelCycles + phase;
            record.phases[phase].calls = kSentinelCalls + phase;
        }
    }
    auto const sentinels = records;
    std::array<std::uint32_t, kRecordCount * kWarpSize> output{};
    DeviceBuffer<profile::DeviceBlockRecord> deviceRecords(kRecordCount);
    DeviceBuffer<std::uint32_t> deviceOutput(output.size());
    // Host copies avoid cudaMemset, which may launch an internal wide kernel.
    checkCuda(
        cudaMemcpy(deviceRecords.data(), records.data(), sizeof(records), cudaMemcpyHostToDevice));
    checkCuda(
        cudaMemcpy(deviceOutput.data(), output.data(), sizeof(output), cudaMemcpyHostToDevice));
    auto const readBack = [&]() {
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(
            records.data(), deviceRecords.data(), sizeof(records), cudaMemcpyDeviceToHost));
        checkCuda(
            cudaMemcpy(output.data(), deviceOutput.data(), sizeof(output), cudaMemcpyDeviceToHost));
    };

    // Exactly four blocks are launched across this test: two, then one, then one.
    // Every grid occupies at most one sixth of the device's SMs.
    exerciseProfiler<<<blockCount, kWarpSize>>>(deviceRecords.data(), deviceOutput.data(), 0, true);
    readBack();
    for (unsigned block = 0; block < unsigned(blockCount); ++block) {
        checkOutput(output, block, 0);
        checkCounters(records[block], block, 0, sentinels[block]);
    }
    auto const firstRecords = records;

    exerciseProfiler<<<1, kWarpSize>>>(deviceRecords.data(), deviceOutput.data(), 2, true);
    readBack();
    checkOutput(output, 0, 2);
    checkCounters(records[0], 0, 2, sentinels[0]);
    checkEqual(records[1], firstRecords[1]);
    checkOutput(output, 1, 0);
    auto const secondRecords = records;
    auto const profiledOutput = output;

    exerciseProfiler<<<1, kWarpSize>>>(deviceRecords.data(), deviceOutput.data(), 2, false);
    readBack();
    for (size_t block = 0; block < kRecordCount; ++block) {
        checkEqual(records[block], secondRecords[block]);
    }
    require(output == profiledOutput, "Enabled and null-disabled profiling outputs differ");
    std::puts("Device profiler counts, overwrite, inactive paths, and data output passed");
    return 0;
}

}    // namespace detail

int main() {
    try {
        return detail::run();
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Device profiler test failed: %s\n", error.what());
        return 1;
    }
}
