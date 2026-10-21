/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>

namespace detail {

// Each OS worker owns one cache line. Registration is paid once; ordinary
// submissions never contend on the old process-wide spinlock. Atomics also
// make diagnostic reads/reset data-race-free (interval boundaries must still
// wait for all producers).
struct alignas(64) CpuFlopShard {
    std::atomic<FlopCount> count{0};
};

std::mutex flopShardsMutex;
std::vector<std::unique_ptr<CpuFlopShard>> flopShards;
std::chrono::steady_clock::time_point flopStart = std::chrono::steady_clock::now();

CpuFlopShard& cpuFlopShard() {
    thread_local CpuFlopShard* shard = []() {
        auto entry = std::make_unique<CpuFlopShard>();
        auto* pointer = entry.get();
        std::lock_guard<std::mutex> lock(flopShardsMutex);
        flopShards.push_back(std::move(entry));
        return pointer;
    }();
    return *shard;
}

}    // namespace detail

HPX_PLAIN_ACTION(resetFlops, ResetFlopsAction);
HPX_PLAIN_ACTION(readFlops, ReadFlopsAction);

void resetFlops() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ResetFlopsAction>(child));
    }
    resetGpuFlops();
    {
        std::lock_guard<std::mutex> lock(detail::flopShardsMutex);
        for (auto const& shard : detail::flopShards) {
            shard->count.store(0, std::memory_order_relaxed);
        }
    }
    for (auto& future : futures) {
        future.get();
    }
    detail::flopStart = std::chrono::steady_clock::now();
}

FlopCount readFlops() {
    Vector<hpx::future<FlopCount>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ReadFlopsAction>(child));
    }
    FlopCount result = getGpuFlops();
    {
        std::lock_guard<std::mutex> lock(detail::flopShardsMutex);
        for (auto const& shard : detail::flopShards) {
            result += shard->count.load(std::memory_order_relaxed);
        }
    }
    for (auto& future : futures) {
        result += future.get();
    }
    return result;
}

double flopsPerSecond() {
    // Freeze the elapsed time before collecting remote counters so reporting
    // latency does not lower the rate. A read does not restart the clock.
    double const elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - detail::flopStart).count();
    auto const count = readFlops();
    return elapsed > 0.0 ? double(count) / elapsed : 0.0;
}

void addCpuFlops(FlopCount count) {
#ifdef COUNT_FLOPS
    if (count != 0) {
        auto& total = detail::cpuFlopShard().count;
        // Only this OS thread writes its shard between quiescent resets.
        // Relaxed load/store avoids even an uncontended locked RMW operation.
        total.store(total.load(std::memory_order_relaxed) + count, std::memory_order_relaxed);
    }
#endif
}

#ifndef USE_CUDA
void resetGpuFlops() {}
FlopCount getGpuFlops() {
    return 0;
}
#endif
