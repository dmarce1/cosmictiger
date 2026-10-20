// Copyright (C) 2026 Dominic C. Marcello
// Distributed under the Boost Software License, Version 1.0.
// See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

#pragma once

#include <atomic>
#include <cstdio>

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
