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
    std::atomic<FlopCount> singlePrecision{0};
    std::atomic<FlopCount> doublePrecision{0};
    std::atomic<FlopCount> unclassified{0};
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
HPX_PLAIN_ACTION(readFlopCounts, ReadFlopCountsAction);

void resetFlops() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ResetFlopsAction>(child));
    }
    resetGpuFlops();
    {
        std::lock_guard<std::mutex> lock(detail::flopShardsMutex);
        for (auto const& shard : detail::flopShards) {
            shard->singlePrecision.store(0, std::memory_order_relaxed);
            shard->doublePrecision.store(0, std::memory_order_relaxed);
            shard->unclassified.store(0, std::memory_order_relaxed);
        }
    }
    for (auto& future : futures) {
        future.get();
    }
    detail::flopStart = std::chrono::steady_clock::now();
}

FlopCounts readFlopCounts() {
    Vector<hpx::future<FlopCounts>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<ReadFlopCountsAction>(child));
    }
    FlopCounts result = getGpuFlopCounts();
    {
        std::lock_guard<std::mutex> lock(detail::flopShardsMutex);
        for (auto const& shard : detail::flopShards) {
            result.singlePrecision += shard->singlePrecision.load(std::memory_order_relaxed);
            result.doublePrecision += shard->doublePrecision.load(std::memory_order_relaxed);
            result.unclassified += shard->unclassified.load(std::memory_order_relaxed);
        }
    }
    for (auto& future : futures) {
        result += future.get();
    }
    return result;
}

FlopCount readFlops() {
    return readFlopCounts().total();
}

FlopRates flopRatesPerSecond() {
    // Freeze the elapsed time before collecting remote counters so reporting
    // latency does not lower the rate. A read does not restart the clock.
    double const elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - detail::flopStart).count();
    auto const counts = readFlopCounts();
    if (elapsed <= 0.0) {
        return {};
    }
    return {double(counts.singlePrecision) / elapsed, double(counts.doublePrecision) / elapsed,
        double(counts.unclassified) / elapsed};
}

double flopsPerSecond() {
    return flopRatesPerSecond().total();
}

void addCpuFlops(FlopCount count, FlopPrecision precision) {
#ifdef COUNT_FLOPS
    if (count != 0) {
        auto& shard = detail::cpuFlopShard();
        auto& total = precision == FlopPrecision::Single ? shard.singlePrecision :
            precision == FlopPrecision::Double           ? shard.doublePrecision :
                                                           shard.unclassified;
        // Only this OS thread writes its shard between quiescent resets.
        // Relaxed load/store avoids even an uncontended locked RMW operation.
        total.store(total.load(std::memory_order_relaxed) + count, std::memory_order_relaxed);
    }
#endif
}

#ifndef USE_CUDA
void resetGpuFlops() {}
FlopCounts getGpuFlopCounts() {
    return {};
}
FlopCount getGpuFlops() {
    return getGpuFlopCounts().total();
}
#endif
