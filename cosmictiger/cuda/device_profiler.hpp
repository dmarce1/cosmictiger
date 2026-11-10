// CosmicTiger - collective CUDA block phase timing; BSL-1.0.

#pragma once

#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

#include <cosmictiger/core/device_profile.hpp>

namespace profile {

// Collective on exactly one complete warp per block. Every lane must pass the
// same record and phases and reach each boundary without divergence. The record
// belongs to this block; finish overwrites it once, including unused phases.
// Durations are elapsed SM cycles, including stalls and scheduling, not cycles
// spent issuing this warp's instructions. Profiling overhead is included.
class DeviceProfiler {
public:
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    /// Begin cooperative timing for a single-warp block when a result record is
    /// supplied.
    __device__ explicit DeviceProfiler(DeviceBlockRecord* perBlockRecord, DevicePhase initial)
      : record_(perBlockRecord)
      , phase_(initial) {
        static_assert(kDevicePhaseCount <= 32, "Each device phase needs its own warp lane");
        if (record_ != nullptr) {
            assert(blockDim.x == 32 && blockDim.y == 1 && blockDim.z == 1);
            assert(static_cast<unsigned>(initial) < kDevicePhaseCount);
            __syncwarp();
            if (threadIdx.x == 0) {
                previous_ = static_cast<std::uint64_t>(clock64());
            }
            // No lane may begin the initial phase before its timestamp exists.
            __syncwarp();
        }
    }

    /// Charge elapsed cycles to the previous phase and switch to the requested phase.
    __device__ void next(DevicePhase phase) {
        if (record_ != nullptr) {
            assert(static_cast<unsigned>(phase) < kDevicePhaseCount);
            accumulate();
            phase_ = phase;
        }
    }

    /// Flush the final phase counters to the block record and disable further
    /// recording.
    __device__ void finish() {
        if (record_ != nullptr) {
            accumulate();
            if (threadIdx.x < kDevicePhaseCount) {
                record_->phases[threadIdx.x].cycles = cycles_;
                record_->phases[threadIdx.x].calls = calls_;
            }
            record_ = nullptr;
        }
    }
#else
    /// Provide the no-op counterpart when device profiling is disabled.
    __device__ explicit DeviceProfiler(DeviceBlockRecord*, DevicePhase) {}
    /// Provide the no-op counterpart when device profiling is disabled.
    __device__ void next(DevicePhase) {}
    /// Provide the no-op counterpart when device profiling is disabled.
    __device__ void finish() {}
#endif

    DeviceProfiler(DeviceProfiler const&) = delete;
    DeviceProfiler& operator=(DeviceProfiler const&) = delete;

private:
#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
    /// Synchronize the warp and charge the leader's elapsed interval to the lane owning
    /// the current phase.
    __device__ void accumulate() {
        // Wait for all work in the preceding phase, then broadcast one leader
        // interval. Only the lane owning that phase updates its local counters.
        __syncwarp();
        unsigned long long elapsed = 0;
        if (threadIdx.x == 0) {
            auto const now = static_cast<std::uint64_t>(clock64());
            elapsed = static_cast<unsigned long long>(now - previous_);
            previous_ = now;
        }
        elapsed = __shfl_sync(0xffffffffU, elapsed, 0);
        if (threadIdx.x == static_cast<unsigned>(phase_)) {
            cycles_ += static_cast<std::uint64_t>(elapsed);
            ++calls_;
        }
    }

    DeviceBlockRecord* record_;
    DevicePhase phase_;
    std::uint64_t previous_ = 0;
    std::uint64_t cycles_ = 0;
    std::uint64_t calls_ = 0;
#endif
};

}    // namespace profile
