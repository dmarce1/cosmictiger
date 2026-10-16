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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stddef.h>

namespace detail {

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
    size_t const stride = totalCount / selectedCount;
    size_t const remainder = totalCount % selectedCount;
    auto const tracerIndex = [=](size_t tracer) {
        // Both factors in the remainder product are bounded by the int request count.
        static_assert(std::numeric_limits<int>::digits <= 31);
        return tracer * stride + size_t(uint64_t(tracer) * remainder / selectedCount);
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
