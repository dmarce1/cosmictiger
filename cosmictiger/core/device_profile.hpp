// CosmicTiger - shared CUDA block profiling records; BSL-1.0.

#pragma once

#include <cstddef>
#include <cstdint>

namespace profile {

enum class DevicePhase {
    KickPrepare,
    KickPeriodic,
    KickTraversal,
    KickM2L,
    KickP2L,
    KickM2P,
    KickP2P,
    KickLeafKick,
    KickCopyWait,
    KickSchedule,
    KickCleanup,
    Count
};

constexpr size_t kDevicePhaseCount = static_cast<size_t>(DevicePhase::Count);

struct DeviceCounter {
    std::uint64_t cycles = 0;
    std::uint64_t calls = 0;
};

struct DeviceBlockRecord {
    DeviceCounter phases[kDevicePhaseCount]{};
};

}    // namespace profile
