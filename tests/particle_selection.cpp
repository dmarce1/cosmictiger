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
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cosmictiger/particles/particle_selection.hpp>

namespace detail {

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void checkSelection(size_t totalCount, int requestedCount) {
    std::vector<char> expected(totalCount, 0);
    size_t const selectedCount = std::min(totalCount, size_t(requestedCount));
    for (size_t i = 0; i < selectedCount; i++) {
        expected[i * totalCount / selectedCount] = 1;
    }
    std::vector<char> actual(totalCount, -1);
    setTracerFlags(actual.data(), actual.size(), 0, totalCount, requestedCount);
    require(actual == expected, "Tracer positions or requested count differ");
    for (size_t partitions : {2, 3, 7, 17}) {
        std::fill(actual.begin(), actual.end(), char{-1});
        for (size_t partition = 0; partition < partitions; partition++) {
            size_t const begin = partition * totalCount / partitions;
            size_t const end = (partition + 1) * totalCount / partitions;
            char* flags = begin == end ? nullptr : actual.data() + begin;
            setTracerFlags(flags, end - begin, begin, totalCount, requestedCount);
        }
        require(actual == expected, "Partitioning changed the selected tracer particles");
    }
}

void checkLargeCounts() {
    size_t const totalCount = std::numeric_limits<size_t>::max() - 1234567890;
    int const requestedCount =
        int(std::min(totalCount / 4, size_t(std::numeric_limits<int>::max())));
    size_t const lastIndex =
        totalCount - totalCount / requestedCount - size_t(totalCount % requestedCount != 0);
    char flags[3];
    setTracerFlags(flags, 3, lastIndex - 1, totalCount, requestedCount);
    require(flags[0] == 0 && flags[1] == 1 && flags[2] == 0, "Large tracer positions overflowed");
    setTracerFlags(flags, 3, totalCount - 3, totalCount, requestedCount);
    require(flags[0] == 0 && flags[1] == 0 && flags[2] == 0,
        "Selected a tracer beyond the last global tracer");
}

}    // namespace detail

int main() {
    for (size_t totalCount = 1; totalCount <= 64; totalCount++) {
        for (int requestedCount = 1; requestedCount <= 80; requestedCount++) {
            detail::checkSelection(totalCount, requestedCount);
        }
    }
    detail::checkSelection(262144, 1000000);
    detail::setTracerFlags(nullptr, 0, 0, 0, 1);
    detail::checkLargeCounts();
    std::puts("Tracer selection regression checks passed");
}
