/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stddef.h>

namespace detail {

inline size_t evenlySpacedParticleIndex(size_t selected, size_t totalCount, size_t selectedCount) {
    size_t const stride = totalCount / selectedCount;
    size_t const remainder = totalCount % selectedCount;
    // Selection counts come from a positive int option, bounding both factors
    // of the remainder product even when the total particle count is larger.
    static_assert(std::numeric_limits<int>::digits <= 31);
    return selected * stride + size_t(uint64_t(selected) * remainder / selectedCount);
}

inline void setTracerFlags(
    char* flags, size_t localCount, size_t globalOffset, size_t totalCount, int requestedCount) {
    if (localCount == 0) {
        return;
    }
    std::fill_n(flags, localCount, char{0});
    if (totalCount == 0 || requestedCount <= 0) {
        return;
    }
    size_t const selectedCount = std::min(totalCount, size_t(requestedCount));
    auto const tracerIndex = [=](size_t tracer) {
        return evenlySpacedParticleIndex(tracer, totalCount, selectedCount);
    };
    size_t begin = 0;
    size_t end = selectedCount;
    while (begin < end) {
        size_t const middle = begin + (end - begin) / 2;
        if (tracerIndex(middle) < globalOffset) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    for (size_t tracer = begin; tracer < selectedCount; tracer++) {
        size_t const localIndex = tracerIndex(tracer) - globalOffset;
        if (localIndex >= localCount) {
            break;
        }
        flags[localIndex] = 1;
    }
}

}    // namespace detail
