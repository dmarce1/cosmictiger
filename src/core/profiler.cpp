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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>

#if COSMICTIGER_PROFILE

namespace profile {
namespace detail {

    using Clock = std::chrono::steady_clock;

    constexpr size_t kWorkCount = static_cast<size_t>(WorkPhase::Count);
    constexpr size_t kDeviceWorkBegin = static_cast<size_t>(WorkPhase::KickGpuHostToDevice);
    constexpr std::array<char const*, kWorkCount> kWorkNames{
        {"kick.cpu.prepare", "kick.cpu.periodic", "kick.cpu.traversal", "kick.cpu.M2L",
            "kick.cpu.P2L", "kick.cpu.M2P", "kick.cpu.P2P", "kick.cpu.leafKick",
            "kick.cpu.schedule", "kick.gpu.queueWork", "kick.wait.children", "kick.wait.result",
            "kick.gpu.groupSources", "kick.gpu.collectSources", "kick.gpu.remap",
            "kick.gpu.sortFetchWait", "kick.gpu.memoryAdvice", "kick.gpu.cleanup",
            "kick.gpu.launchQueue", "kick.gpu.prepareLaunch", "kick.gpu.submitWait",
            "kick.gpu.hostToDevice", "kick.gpu.kernel", "kick.gpu.deviceToHost"}};

    // Keep unrelated phases on separate cache lines. Sampling touches only two
    // relaxed atomics; reporting does not block workers on the phase mutex.
    struct alignas(64) WorkCounter {
        std::atomic<std::uint64_t> nanoseconds{0};
        std::atomic<std::uint64_t> calls{0};
    };

    std::array<WorkCounter, kWorkCount> workCounters;

    struct WorkRow {
        char const* name;
        std::uint64_t nanoseconds;
        std::uint64_t calls;
    };

    void recordNanoseconds(WorkPhase phase, std::uint64_t nanoseconds) noexcept {
        size_t const index = static_cast<size_t>(phase);
        if (index >= kWorkCount) {
            return;
        }
        auto& counter = workCounters[index];
        counter.nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
        counter.calls.fetch_add(1, std::memory_order_relaxed);
    }

    void recordInterval(WorkPhase phase, Clock::duration elapsed) noexcept {
        auto const nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        recordNanoseconds(phase, static_cast<std::uint64_t>(nanoseconds));
    }

    void appendWorkReport(std::string& report, bool device) {
        std::vector<WorkRow> rows;
        size_t const begin = device ? kDeviceWorkBegin : 0;
        size_t const end = device ? kWorkCount : kDeviceWorkBegin;
        size_t width = 4;
        for (size_t index = begin; index < end; ++index) {
            auto const& counter = workCounters[index];
            auto const calls = counter.calls.load(std::memory_order_relaxed);
            if (calls != 0) {
                // A report taken during concurrent sampling is a best-effort
                // snapshot. Once workers finish, both counters are exact.
                auto const nanoseconds = counter.nanoseconds.load(std::memory_order_relaxed);
                rows.push_back({kWorkNames[index], nanoseconds, calls});
                width = std::max(width, std::char_traits<char>::length(kWorkNames[index]));
            }
        }
        if (rows.empty()) {
            return;
        }
        std::sort(rows.begin(), rows.end(), [](auto const& a, auto const& b) {
            return a.nanoseconds != b.nanoseconds ? a.nanoseconds > b.nanoseconds :
                                                    std::string(a.name) < b.name;
        });
        if (!report.empty()) {
            report += '\n';
        }
        if (device) {
            report += "Profile detail: rank-0 CUDA device elapsed intervals (overlap possible)\n"
                      "Device intervals are not additive to host or coordinator totals.\n";
        } else {
            report +=
                "Profile detail: rank-0 local host elapsed intervals (overlap; not CPU time)\n"
                "Intervals may overlap and are not additive to the coordinator total.\n";
        }
        char line[512];
        std::snprintf(line, sizeof(line), "%-*s %10s %14s %14s\n", int(width), "Work", "Calls",
            "Total", "Mean(us)");
        report += line;
        for (auto const& row : rows) {
            double const seconds = static_cast<double>(row.nanoseconds) * 1.0e-9;
            double const meanUs =
                static_cast<double>(row.nanoseconds) * 1.0e-3 / static_cast<double>(row.calls);
            std::snprintf(line, sizeof(line), "%-*s %10llu %14s %14.3f\n", int(width), row.name,
                static_cast<unsigned long long>(row.calls),
                terminal::formatDuration(seconds).c_str(), meanUs);
            report += line;
        }
    }

    struct Phase {
        Clock::duration elapsed{};
        std::uint64_t calls = 0;
    };

    struct Frame {
        std::string name;
        Clock::time_point resumed;
    };

    struct Registry {
        std::mutex mutex;
        std::unordered_map<std::string, Phase> phases;
        std::vector<Frame> stack;
    };

    Registry& registry() {
        static Registry value;
        return value;
    }

    // Charge the running top-of-stack interval to its phase and restart it at now.
    void settleTop(Registry& state, Clock::time_point now) {
        if (state.stack.empty()) {
            return;
        }
        auto& top = state.stack.back();
        state.phases[top.name].elapsed += now - top.resumed;
        top.resumed = now;
    }

}    // namespace detail

ScopedWork::ScopedWork(WorkPhase phase, bool enabled) noexcept
  : phase_(phase)
  , resumed_(enabled ? detail::Clock::now() : detail::Clock::time_point{})
  , enabled_(enabled) {}

ScopedWork::~ScopedWork() {
    stop();
}

void ScopedWork::next(WorkPhase phase) noexcept {
    if (enabled_) {
        auto const now = detail::Clock::now();
        detail::recordInterval(phase_, now - resumed_);
        phase_ = phase;
        resumed_ = now;
    }
}

void ScopedWork::stop() noexcept {
    if (enabled_) {
        detail::recordInterval(phase_, detail::Clock::now() - resumed_);
        enabled_ = false;
    }
}

void recordWork(WorkPhase phase, double seconds) noexcept {
    long double const nanoseconds = static_cast<long double>(seconds) * 1.0e9L;
    if (!std::isfinite(seconds) || nanoseconds < 0.0L ||
        nanoseconds >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return;
    }
    detail::recordNanoseconds(phase, static_cast<std::uint64_t>(nanoseconds));
}

void profileBegin(char const* name) {
    auto const now = detail::Clock::now();
    auto& state = detail::registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    detail::settleTop(state, now);
    state.stack.push_back({name, now});
    ++state.phases[name].calls;
}

void profileEnd() {
    auto const now = detail::Clock::now();
    auto& state = detail::registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.stack.empty()) {
        return;
    }
    detail::settleTop(state, now);
    state.stack.pop_back();
    // The parent resumes from this instant; its paused interval is not charged.
    if (!state.stack.empty()) {
        state.stack.back().resumed = now;
    }
}

std::string profileReport() {
    auto const now = detail::Clock::now();
    auto& state = detail::registry();
    std::vector<std::pair<std::string, detail::Phase>> rows;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        detail::settleTop(state, now);
        rows.assign(state.phases.begin(), state.phases.end());
    }
    if (rows.empty()) {
        std::string report;
        detail::appendWorkReport(report, false);
        detail::appendWorkReport(report, true);
        return report;
    }
    std::sort(rows.begin(), rows.end(), [](auto const& a, auto const& b) {
        return a.second.elapsed != b.second.elapsed ? a.second.elapsed > b.second.elapsed :
                                                      a.first < b.first;
    });
    double total = 0.0;
    size_t width = 5;
    for (auto const& row : rows) {
        total += std::chrono::duration<double>(row.second.elapsed).count();
        width = std::max(width, row.first.size());
    }
    double const share = total > 0.0 ? 100.0 / total : 0.0;
    std::string report = "Profile: exclusive coordinator wall time per phase (hh:mm:ss.ss)\n";
    char line[512];
    std::snprintf(line, sizeof(line), "%-*s %10s %14s %14s %8s\n", int(width), "Phase", "Calls",
        "Total", "Mean", "Share");
    report += line;
    for (auto const& row : rows) {
        double const seconds = std::chrono::duration<double>(row.second.elapsed).count();
        std::snprintf(line, sizeof(line), "%-*s %10llu %14s %14s %7.2f%%\n", int(width),
            row.first.c_str(), static_cast<unsigned long long>(row.second.calls),
            terminal::formatDuration(seconds).c_str(),
            terminal::formatDuration(seconds / static_cast<double>(row.second.calls)).c_str(),
            seconds * share);
        report += line;
    }
    std::snprintf(line, sizeof(line), "%-*s %10s %14s\n", int(width), "Total", "",
        terminal::formatDuration(total).c_str());
    report += line;
    detail::appendWorkReport(report, false);
    detail::appendWorkReport(report, true);
    return report;
}

void profileOutput(std::string const& filename) {
    std::string const report = profileReport();
    if (report.empty()) {
        return;
    }
    std::FILE* file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "Unable to write profile: %s\n", filename.c_str());
        return;
    }
    std::fputs(report.c_str(), file);
    if (std::fclose(file) != 0) {
        std::fprintf(stderr, "Unable to finish profile: %s\n", filename.c_str());
    }
}

}    // namespace profile

#endif
