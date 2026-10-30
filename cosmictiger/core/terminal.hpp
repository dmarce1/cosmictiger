// Copyright (C) 2026 Dominic C. Marcello
// Distributed under the Boost Software License, Version 1.0.
// See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace terminal {

namespace detail {
    inline std::atomic<bool> verboseEnabled{false};
}    // namespace detail

// Each locality receives this runtime setting through setOptions. The memory
// monitor may print from another thread while scenario options are refreshed.
inline void setVerbose(bool enabled) {
    detail::verboseEnabled.store(enabled, std::memory_order_relaxed);
}

inline bool verbose() {
    return detail::verboseEnabled.load(std::memory_order_relaxed);
}

// Wall-clock durations use hundredths of a second and hours do not wrap at 24.
inline std::string formatDuration(double seconds) {
    constexpr std::uint64_t kExponentMask = 0x7ff0000000000000ULL;
    // Production fast-math builds may assume std::isfinite is always true.
    if ((std::bit_cast<std::uint64_t>(seconds) & kExponentMask) == kExponentMask) {
        return "--:--:--.--";
    }
    double const rounded = std::round((seconds > 0.0 ? seconds : 0.0) * 100.0);
    if (rounded >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        return "--:--:--.--";
    }
    auto const centiseconds = static_cast<std::uint64_t>(rounded);
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%02llu:%02llu:%02llu.%02llu",
        static_cast<unsigned long long>(centiseconds / 360000),
        static_cast<unsigned long long>((centiseconds / 6000) % 60),
        static_cast<unsigned long long>((centiseconds / 100) % 60),
        static_cast<unsigned long long>(centiseconds % 100));
    return text.data();
}

template <class... Args>
inline void detailPrint(char const* format, Args... args) {
    if (verbose()) {
        if constexpr (sizeof...(args) == 0) {
            std::fputs(format, stdout);
        } else {
            std::printf(format, args...);
        }
        std::fflush(stdout);
    }
}

inline void section(char const* title) {
    std::printf("\n  %s\n  %s\n", title,
        "----------------------------------------------------------------------------");
    std::fflush(stdout);
}

}    // namespace terminal
