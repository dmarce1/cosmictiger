/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

#include <chrono>
#include <string>

#include <cosmictiger/core/device_profile.hpp>

// Manual phase profiler. COSMICTIGER_PROFILE (a CMake option, on by default)
// compiles the instrumentation; without it every call is an empty inline.
//
// profileBegin/profileEnd delimit phases on the coordinator's
// logical control flow (driver, scenario driver, initialization). Phases nest:
// entering a child pauses its parent, so each name accumulates exclusive wall
// time and the rows of the report sum to the instrumented total. Do not
// instrument kernels, per-particle loops, or work that runs concurrently on
// other HPX tasks; the single phase stack is not meant for parallel scopes.
namespace profile {

// Fixed identifiers keep concurrent work sampling free of allocation and name
// lookup. These intervals are independent of the coordinator phase stack.
enum class WorkPhase {
    KickCpuPrepare,
    KickCpuPeriodic,
    KickCpuTraversal,
    KickCpuM2L,
    KickCpuP2L,
    KickCpuM2P,
    KickCpuP2P,
    KickCpuLeafKick,
    KickCpuSchedule,
    KickGpuQueue,
    KickChildJoin,
    KickResultWait,
    KickGpuGroupSources,
    KickGpuCollectSources,
    KickGpuRemap,
    KickGpuSortFetchWait,
    KickGpuMemoryAdvice,
    KickGpuCleanup,
    KickGpuLaunchQueue,
    KickGpuPrepareLaunch,
    KickGpuSubmitWait,
    KickGpuHostToDevice,
    KickGpuKernel,
    KickGpuDeviceToHost,
    Count
};

#if COSMICTIGER_PROFILE
void profileBegin(char const* name);
void profileEnd();
// Exclusive coordinator phases plus separate concurrent work detail tables.
// Empty when no completed work intervals or coordinator phases were recorded.
std::string profileReport();
// Writes profileReport() to filename (overwrite). Safe while phases are open.
void profileOutput(std::string const& filename);
// Record externally timed intervals, such as CUDA event durations, in seconds.
// Rank filtering is the caller's responsibility for all work profiling APIs.
void recordWork(WorkPhase phase, double seconds) noexcept;
#else
inline void profileBegin(char const*) noexcept {}
inline void profileEnd() noexcept {}
inline std::string profileReport() {
    return {};
}
inline void profileOutput(std::string const&) noexcept {}
inline void recordWork(WorkPhase, double) noexcept {}
#endif

#if COSMICTIGER_PROFILE && COSMICTIGER_DEVICE_PROFILE
// Import completed per-block records after the launch's existing stream wait.
// Null/empty batches and counters with no calls are ignored. Rank filtering is
// the caller's responsibility, as with recordWork.
void recordDeviceBlocks(DeviceBlockRecord const* blocks, size_t count) noexcept;
#else
inline void recordDeviceBlocks(DeviceBlockRecord const*, size_t) noexcept {}
#endif

// Independent host elapsed intervals, safe in concurrent HPX tasks. They may
// include suspension/waits and overlap other work, so they are not CPU time or
// additive to the coordinator total. Only completed intervals are reported.
class ScopedWork {
public:
#if COSMICTIGER_PROFILE
    explicit ScopedWork(WorkPhase phase, bool enabled = true) noexcept;
    ~ScopedWork();
    // Finish the current interval and start another without reading the clock
    // twice. A stopped or disabled scope stays inactive.
    void next(WorkPhase phase) noexcept;
    void stop() noexcept;
#else
    explicit ScopedWork(WorkPhase, bool = true) noexcept {}
    ~ScopedWork() = default;
    void next(WorkPhase) noexcept {}
    void stop() noexcept {}
#endif

    ScopedWork(ScopedWork const&) = delete;
    ScopedWork& operator=(ScopedWork const&) = delete;
    ScopedWork(ScopedWork&&) = delete;
    ScopedWork& operator=(ScopedWork&&) = delete;

private:
#if COSMICTIGER_PROFILE
    WorkPhase phase_;
    std::chrono::steady_clock::time_point resumed_;
    bool enabled_;
#endif
};

// Balance a coordinator phase on every exit path, including exceptions. Use
// enabled only where the caller can identify a serialized coordinator entry;
// being on rank zero alone does not make a parallel task safe to instrument.
class ScopedPhase {
public:
    explicit ScopedPhase(char const* name, bool enabled = true)
      : enabled_(enabled) {
        if (enabled_) {
            profileBegin(name);
        }
    }

    ~ScopedPhase() {
        if (enabled_) {
            profileEnd();
        }
    }

    ScopedPhase(ScopedPhase const&) = delete;
    ScopedPhase& operator=(ScopedPhase const&) = delete;
    ScopedPhase(ScopedPhase&&) = delete;
    ScopedPhase& operator=(ScopedPhase&&) = delete;

private:
    bool enabled_;
};

}    // namespace profile
