#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <stdfloat>

#include <cosmictiger/fp16.hpp>

#ifndef __STDCPP_FLOAT16_T__
#error "ROCKSTAR catalog conversion requires C++23 std::float16_t support"
#endif

static_assert(sizeof(std::float16_t) == sizeof(Float16));
static_assert(std::numeric_limits<std::float16_t>::is_iec559 &&
    std::numeric_limits<std::float16_t>::radix == 2 &&
    std::numeric_limits<std::float16_t>::digits == 11 &&
    std::numeric_limits<std::float16_t>::min_exponent == -13 &&
    std::numeric_limits<std::float16_t>::max_exponent == 16);

Float16 double2float16(double value) {
    // Check the original value before rounding: even a finite value just above
    // the largest binary16 value must fail instead of rounding back into range.
    if (std::isfinite(value) &&
        std::abs(value) > static_cast<double>(std::numeric_limits<std::float16_t>::max())) {
        throw std::overflow_error("IEEE binary16 overflow: finite magnitude exceeds 65504");
    }
    std::float16_t const half = static_cast<std::float16_t>(value);
    return {std::bit_cast<std::uint16_t>(half)};
}

double float162double(Float16 value) {
    return static_cast<double>(std::bit_cast<std::float16_t>(value.bits));
}
