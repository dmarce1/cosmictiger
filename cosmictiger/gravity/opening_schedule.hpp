/* CosmicTiger - configurable global FMM opening angle; BSL-1.0. */

#pragma once

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace gravity {

struct OpeningSchedulePoint {
    double redshift;
    double theta;
};

namespace detail {

    inline std::string_view trimOpeningScheduleToken(std::string_view token) {
        size_t const first = token.find_first_not_of(" \t\r\n");
        if (first == std::string_view::npos) {
            return {};
        }
        size_t const last = token.find_last_not_of(" \t\r\n");
        return token.substr(first, last - first + 1);
    }

    inline double parseOpeningScheduleNumber(std::string_view token) {
        token = trimOpeningScheduleToken(token);
        if (!token.empty() && token.front() == '+') {
            token.remove_prefix(1);
            if (!token.empty() && token.front() == '-') {
                throw std::invalid_argument(
                    "gravity.openingSchedule numbers cannot contain two signs");
            }
        }
        if (token.empty() || token.front() == '+') {
            throw std::invalid_argument(
                "gravity.openingSchedule requires finite redshift:theta pairs");
        }
        double value{};
        auto const parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        // The driver permits fast-math; inspect the exponent instead of isfinite.
        bool const finite =
            (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || !finite) {
            throw std::invalid_argument(
                "gravity.openingSchedule requires finite redshift:theta pairs");
        }
        return value;
    }

}    // namespace detail

// Input order is immaterial. A CLI schedule replaces the constant opening angle.
inline std::vector<OpeningSchedulePoint> parseOpeningSchedule(std::string const& text) {
    std::vector<OpeningSchedulePoint> schedule;
    std::string_view remaining = detail::trimOpeningScheduleToken(text);
    if (remaining.empty() || remaining == "none") {
        return schedule;
    }
    for (;;) {
        size_t const comma = remaining.find(',');
        std::string_view const pair = remaining.substr(0, comma);
        size_t const colon = pair.find(':');
        if (colon == std::string_view::npos ||
            pair.find(':', colon + 1) != std::string_view::npos) {
            throw std::invalid_argument(
                "gravity.openingSchedule requires comma-separated redshift:theta pairs");
        }
        double const redshift = detail::parseOpeningScheduleNumber(pair.substr(0, colon));
        double const theta = detail::parseOpeningScheduleNumber(pair.substr(colon + 1));
        if (!(redshift > -1.0)) {
            throw std::invalid_argument("gravity.openingSchedule redshifts must exceed -1");
        }
        if (theta < std::numeric_limits<float>::min() ||
            theta > std::numeric_limits<float>::max()) {
            throw std::invalid_argument(
                "gravity.openingSchedule theta must be positive and within the normal float range");
        }
        schedule.push_back({redshift, theta});
        if (comma == std::string_view::npos) {
            break;
        }
        remaining.remove_prefix(comma + 1);
    }
    std::sort(schedule.begin(), schedule.end(),
        [](auto const& first, auto const& second) { return first.redshift < second.redshift; });
    for (size_t index = 1; index < schedule.size(); ++index) {
        if (schedule[index - 1].redshift == schedule[index].redshift) {
            throw std::invalid_argument("gravity.openingSchedule contains duplicate redshifts");
        }
    }
    return schedule;
}

inline double openingScheduleInitialTheta(
    std::vector<OpeningSchedulePoint> const& schedule, double startRedshift, double fallbackTheta) {
    long double distance = std::numeric_limits<long double>::max();
    double theta = fallbackTheta;
    for (auto const& point : schedule) {
        long double const nextDistance =
            std::abs(static_cast<long double>(point.redshift) - startRedshift);
        // Sorted ascending: replacing on a tie selects the higher redshift.
        if (nextDistance <= distance) {
            distance = nextDistance;
            theta = point.theta;
        }
    }
    return theta;
}

// Expansion applies each threshold at arrival. Contraction restores the value
// on the preceding expansion interval once redshift rises above that threshold.
// Reconstructing from the original start also handles restart and steps spanning
// a turnaround without retaining traversal history.
inline double openingScheduleTheta(std::vector<OpeningSchedulePoint> const& schedule,
    double startRedshift, double currentRedshift, double fallbackTheta) {
    double theta = openingScheduleInitialTheta(schedule, startRedshift, fallbackTheta);
    for (auto point = schedule.rbegin(); point != schedule.rend(); ++point) {
        if (point->redshift <= startRedshift && point->redshift >= currentRedshift) {
            theta = point->theta;
        }
    }
    return theta;
}

}    // namespace gravity
