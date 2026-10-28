/* CosmicTiger - hierarchical PMMM mesh selection; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pmmm::detail {

// Zero denotes an unrestricted prime factor. A finite limit is inclusive and
// need not itself be prime (for example, limits five and six are equivalent).
inline int fftMeshSize(int requestedSize, int maxPrimeFactor) {
    if (requestedSize <= 0 || maxPrimeFactor < 0 || maxPrimeFactor == 1) {
        throw std::invalid_argument("Invalid PMMM mesh size or maximum prime factor");
    }
    if (maxPrimeFactor == 0 || requestedSize <= maxPrimeFactor) {
        return requestedSize;
    }
    auto const allowed = [maxPrimeFactor](int64_t value) {
        for (int64_t divisor = 2; divisor <= maxPrimeFactor && divisor * divisor <= value;
            divisor += divisor == 2 ? 1 : 2) {
            while (value % divisor == 0) {
                value /= divisor;
            }
        }
        return value <= maxPrimeFactor;
    };
    if (allowed(requestedSize)) {
        return requestedSize;
    }
    constexpr int64_t kMaximum = std::numeric_limits<int>::max();
    if (maxPrimeFactor < 17) {
        // Small prime limits can leave very large gaps between valid lengths.
        // Enumerate products instead of scanning every integer in those gaps.
        constexpr std::array<int, 6> kPrimes{2, 3, 5, 7, 11, 13};
        int64_t best = kMaximum + 1;
        auto const search = [&](auto const& self, size_t index, int64_t value) -> void {
            if (value >= requestedSize) {
                best = std::min(best, value);
                return;
            }
            if (index == kPrimes.size() || kPrimes[index] > maxPrimeFactor) {
                return;
            }
            for (int64_t product = value; product < best && product <= kMaximum;
                product *= kPrimes[index]) {
                self(self, index + 1, product);
            }
        };
        search(search, 0, 1);
        if (best <= kMaximum) {
            return int(best);
        }
    } else {
        for (int64_t candidate = int64_t(requestedSize) + 1; candidate <= kMaximum; ++candidate) {
            if (allowed(candidate)) {
                return int(candidate);
            }
        }
    }
    throw std::overflow_error("No allowed PMMM FFT mesh length fits in an int");
}

// The base mesh represents all owned particles. Select the smallest allowed
// side whose volume is at least the requested base volume times active / total.
inline int hierarchicalMeshSize(int baseSize, int nearWidth, uint64_t activeCount,
    uint64_t totalCount, int maxPrimeFactor = 0) {
    if (nearWidth < 1 || nearWidth > 4 || baseSize < 2 * nearWidth + 1 ||
        activeCount > totalCount) {
        throw std::invalid_argument("Invalid hierarchical PMMM mesh or particle counts");
    }
    auto const cube = [](int side) {
        auto const value = static_cast<unsigned __int128>(side);
        return value * value * value;
    };
    auto const baseCells = cube(baseSize);
    if (baseCells > std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("Hierarchical PMMM mesh exceeds cell indexing limits");
    }
    int lower = 2 * nearWidth + 1;
    int upper = baseSize;
    if (activeCount == 0) {
        return fftMeshSize(lower, maxPrimeFactor);
    }
    // Exact integer comparisons avoid rounding an exact cube ratio upward or
    // losing a one-particle increment in large global populations.
    auto const target = baseCells * activeCount;
    while (lower < upper) {
        int const middle = lower + (upper - lower) / 2;
        if (cube(middle) * totalCount < target) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return fftMeshSize(lower, maxPrimeFactor);
}

}    // namespace pmmm::detail
