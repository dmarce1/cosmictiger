#include <algorithm>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cosmictiger/cosmology/recfast.hpp>

namespace recfast {
namespace detail {

    static_assert(
        std::numeric_limits<double>::is_iec559 && sizeof(double) == sizeof(std::uint64_t));
    constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
    constexpr std::uint64_t kSignMask = UINT64_C(0x8000000000000000);

    bool isFiniteNonnegative(double value) {
        // Keep validation effective under the main target's -ffast-math, including
        // negative subnormal values that floating-point comparisons may flush to zero.
        auto const bits = std::bit_cast<std::uint64_t>(value);
        return (bits & kExponentMask) != kExponentMask &&
            ((bits & kSignMask) == 0 || (bits & ~kSignMask) == 0);
    }

    bool isFinitePositive(double value) {
        auto const bits = std::bit_cast<std::uint64_t>(value);
        return (bits & kExponentMask) != kExponentMask && (bits & kSignMask) == 0 && bits != 0;
    }

}    // namespace detail

IonizationHistory::IonizationHistory(std::istream& input) {
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Unable to read RECFAST header");
    }
    std::istringstream header(line);
    std::string redshiftLabel;
    std::string electronLabel;
    std::string extra;
    if (!(header >> redshiftLabel >> electronLabel) || redshiftLabel != "z" ||
        electronLabel != "x_e" || (header >> extra)) {
        throw std::runtime_error("Expected RECFAST header: z x_e");
    }
    for (int row = 0; row < kSampleCount; ++row) {
        std::string const rowLabel = "RECFAST row " + std::to_string(row + 1);
        if (!std::getline(input, line)) {
            throw std::runtime_error("Unable to read " + rowLabel);
        }
        std::istringstream values(line);
        double redshift;
        double electronFraction;
        if (!(values >> redshift >> electronFraction) || (values >> extra)) {
            throw std::runtime_error("Expected two numeric columns in " + rowLabel);
        }
        int const index = kSampleCount - 1 - row;
        if (!detail::isFiniteNonnegative(redshift) || redshift != index * kRedshiftStep) {
            throw std::runtime_error("Unexpected redshift in " + rowLabel);
        }
        if (!detail::isFiniteNonnegative(electronFraction)) {
            throw std::runtime_error(
                "Expected finite nonnegative electron fraction in " + rowLabel);
        }
        electronFractions_[index] = electronFraction;
    }
    input >> std::ws;
    if (input.bad() || (input.fail() && !input.eof())) {
        throw std::runtime_error("Unable to finish reading RECFAST output");
    }
    if (!input.eof()) {
        throw std::runtime_error("Unexpected data after RECFAST table");
    }
}

double IonizationHistory::operator()(double scaleFactor) const {
    if (!detail::isFinitePositive(scaleFactor)) {
        throw std::invalid_argument("RECFAST scale factor must be finite and positive");
    }
    if (scaleFactor >= 1.0) {
        return electronFractions_.front();
    }
    if (scaleFactor <= 1.0 / (1.0 + kMaximumRedshift)) {
        return electronFractions_.back();
    }
    double const coordinate = (1.0 / scaleFactor - 1.0) / kRedshiftStep;
    int const index = std::min(static_cast<int>(coordinate), kSampleCount - 2);
    double const fraction = coordinate - index;
    double const left = electronFractions_[index];
    double const right = electronFractions_[index + 1];
    if (index == 0 || index == kSampleCount - 2) {
        // The four-point cubic has no outer neighbor in these intervals.
        // Linear interpolation reaches both endpoint samples exactly.
        return (1.0 - fraction) * left + fraction * right;
    }
    double const outerLeft = electronFractions_[index - 1];
    double const outerRight = electronFractions_[index + 2];
    double const product = fraction * (1.0 - fraction);
    double const outer = -0.5 * product * ((1.0 - fraction) * outerLeft + fraction * outerRight);
    double const innerLeft = (1.0 - fraction + product * (1.0 - 1.5 * fraction)) * left;
    double const innerRight = (fraction + product * (1.5 * fraction - 0.5)) * right;
    return outer + innerLeft + innerRight;
}

IonizationHistory loadIonizationHistory(std::string const& filename) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("Unable to open RECFAST output: " + filename);
    }
    try {
        return IonizationHistory(input);
    } catch (std::exception const& error) {
        throw std::runtime_error("Invalid RECFAST output " + filename + ": " + error.what());
    }
}

}    // namespace recfast
