/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace detail {

inline void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

inline void requireClose(char const* label, double actual, double expected, double tolerance) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

}    // namespace detail
