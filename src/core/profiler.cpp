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

#include <cosmictiger/core/profiler.hpp>

#if COSMICTIGER_PROFILE

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cosmictiger/core/terminal.hpp>

namespace profile {
namespace detail {

    using Clock = std::chrono::steady_clock;

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
    std::lock_guard<std::mutex> lock(state.mutex);
    detail::settleTop(state, now);
    std::vector<std::pair<std::string, detail::Phase>> rows(
        state.phases.begin(), state.phases.end());
    if (rows.empty()) {
        return {};
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
