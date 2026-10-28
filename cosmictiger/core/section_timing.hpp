#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

namespace section_timing {

enum class Section {
    initialization,
    transferFunctions,
    domains,
    particleSort,
    treeBuild,
    gravity,
    hydrodynamics,
    drift,
    checkpointRead,
    checkpointWrite,
    snapshotOutput,
    analysis,
    count
};

namespace detail {

    using Clock = std::chrono::steady_clock;

    struct Entry {
        Clock::duration elapsed{};
        std::uint64_t calls = 0;
    };

    struct Registry {
        std::mutex mutex;
        std::atomic<bool> enabled{false};
        bool started = false;
        bool exitReportRegistered = false;
        bool finished = false;
        std::string reportPath;
        std::array<Entry, static_cast<size_t>(Section::count)> entries{};
    };

    inline Registry& registry() {
        static Registry value;
        return value;
    }

    inline void writeReport(std::FILE* stream, Registry const& state) noexcept {
        static constexpr std::array<char const*, static_cast<size_t>(Section::count)> labels{
            "Initial conditions", "Transfer functions", "Domains", "Particle sort", "Tree build",
            "Gravity", "Hydrodynamics", "Drift", "Checkpoint read", "Checkpoint write",
            "Snapshot output", "Analysis"};
        std::fprintf(stream, "Major section timings (wall seconds)\n");
        std::fprintf(stream, "Inclusive coordinator timings; nested sections overlap.\n");
        std::fprintf(stream, "%-24s %12s %16s %16s\n", "Section", "Calls", "Total (s)", "Mean (s)");
        for (size_t index = 0; index < state.entries.size(); ++index) {
            auto const& entry = state.entries[index];
            if (entry.calls == 0) {
                continue;
            }
            double const seconds = std::chrono::duration<double>(entry.elapsed).count();
            std::fprintf(stream, "%-24s %12llu %16.6f %16.6f\n", labels[index],
                static_cast<unsigned long long>(entry.calls), seconds,
                seconds / static_cast<double>(entry.calls));
        }
        std::fflush(stream);
    }

}    // namespace detail

// Enable once, on the coordinator process after its options are available.
inline bool start(std::string reportPath) {
    auto& state = detail::registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.started) {
        return false;
    }
    state.reportPath = std::move(reportPath);
    state.started = true;
    state.enabled.store(true, std::memory_order_release);
    return true;
}

class ScopedSection {
public:
    explicit ScopedSection(Section section) noexcept
      : index_(static_cast<size_t>(section)) {
        auto& state = detail::registry();
        enabled_ = index_ < state.entries.size() && state.enabled.load(std::memory_order_acquire);
        if (enabled_) {
            start_ = detail::Clock::now();
        }
    }

    ScopedSection(ScopedSection const&) = delete;
    ScopedSection& operator=(ScopedSection const&) = delete;
    ScopedSection(ScopedSection&&) = delete;
    ScopedSection& operator=(ScopedSection&&) = delete;

    ~ScopedSection() noexcept {
        if (!enabled_) {
            return;
        }
        auto const elapsed = detail::Clock::now() - start_;
        try {
            auto& state = detail::registry();
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!state.finished) {
                auto& entry = state.entries[index_];
                entry.elapsed += elapsed;
                ++entry.calls;
            }
        } catch (...) {
            // Reporting must never mask an exception from the measured work.
        }
    }

private:
    size_t index_;
    bool enabled_ = false;
    detail::Clock::time_point start_{};
};

template <typename Callable>
decltype(auto) measure(Section section, Callable&& callable) {
    ScopedSection scope(section);
    return std::invoke(std::forward<Callable>(callable));
}

// Call after measured scopes have unwound. Safe to call repeatedly.
inline void finish() noexcept {
    try {
        auto& state = detail::registry();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.started || state.finished) {
            return;
        }
        state.finished = true;
        state.enabled.store(false, std::memory_order_release);
        detail::writeReport(stdout, state);
        if (!state.reportPath.empty()) {
            std::FILE* file = std::fopen(state.reportPath.c_str(), "w");
            if (file == nullptr) {
                std::fprintf(
                    stderr, "Unable to write section timings: %s\n", state.reportPath.c_str());
                return;
            }
            detail::writeReport(file, state);
            if (std::fclose(file) != 0) {
                std::fprintf(stderr, "Unable to finish section timings file: %s\n",
                    state.reportPath.c_str());
            }
        }
    } catch (...) {
        std::fputs("Unable to finish section timings report.\n", stderr);
    }
}

// Register before other profiler callbacks so this concise report is last.
// Constructing the registry first makes its destructor run after finish().
inline bool registerExitReport() noexcept {
    try {
        auto& state = detail::registry();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.exitReportRegistered) {
            return true;
        }
        state.exitReportRegistered = std::atexit(finish) == 0;
        return state.exitReportRegistered;
    } catch (...) {
        return false;
    }
}

class ReportGuard {
public:
    ReportGuard() = default;
    ReportGuard(ReportGuard const&) = delete;
    ReportGuard& operator=(ReportGuard const&) = delete;
    ReportGuard(ReportGuard&&) = delete;
    ReportGuard& operator=(ReportGuard&&) = delete;
    ~ReportGuard() noexcept {
        finish();
    }
};

}    // namespace section_timing
