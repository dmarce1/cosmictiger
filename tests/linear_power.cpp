#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/cosmology/linear_power.hpp>

namespace detail {

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void requireClose(double actual, double expected, double tolerance, char const* label) {
    if (!linear_power::detail::isFinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(label) + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected));
    }
}

template <class Function>
void requireFailure(Function function, char const* label) {
    try {
        function();
    } catch (std::exception const&) {
        return;
    }
    throw std::runtime_error(label);
}

void testInterpolation() {
    std::vector<double> const wavenumbers = {0.01, 0.025, 0.1, 0.7, 2.0, 5.0};
    std::vector<double> powers, signedValues;
    auto const polynomial = [](double coordinate) {
        return 1.0 + coordinate * (0.4 + coordinate * (-0.2 + 0.03 * coordinate));
    };
    for (double const wavenumber : wavenumbers) {
        powers.push_back(3.0 * std::pow(wavenumber, -1.7));
        signedValues.push_back(polynomial(std::log(wavenumber)));
    }
    linear_power::LogarithmicTable const power(wavenumbers, powers, true);
    linear_power::LogarithmicTable const transfer(wavenumbers, signedValues);
    for (double const wavenumber : {0.01, 0.012, 0.1, 0.3, 1.3, 4.9, 5.0}) {
        requireClose(power(wavenumber) / (3.0 * std::pow(wavenumber, -1.7)), 1.0, 2.0e-13,
            "Log-power interpolation preserves a power law including endpoints");
        requireClose(transfer(wavenumber), polynomial(std::log(wavenumber)), 2.0e-13,
            "Signed interpolation preserves a cubic in log(k)");
    }
    require(transfer(0.01) < 0.0, "Signed transfers retain their sign");
    requireFailure([&] { power(0.009); }, "Lower out-of-range query must fail");
    requireFailure([&] { power(5.1); }, "Upper out-of-range query must fail");
    requireFailure([&] { power(0.0); }, "Zero wavenumber must fail");
    requireFailure(
        [&] { power(std::numeric_limits<double>::quiet_NaN()); }, "NaN wavenumber must fail");
    requireFailure([] { linear_power::LogarithmicTable({}, {}); }, "Empty table must fail");
    requireFailure([] { linear_power::LogarithmicTable({1.0, 1.0}, {1.0, 2.0}); },
        "Duplicate wavenumber must fail");
    requireFailure([] { linear_power::LogarithmicTable({2.0, 1.0}, {1.0, 2.0}); },
        "Unordered wavenumbers must fail");
    requireFailure([] { linear_power::LogarithmicTable({1.0, 2.0}, {1.0, 0.0}, true); },
        "Nonpositive power must fail");
    requireFailure(
        [] {
            linear_power::LogarithmicTable(
                {1.0, 2.0}, {1.0, std::numeric_limits<double>::infinity()}, true);
        },
        "Infinite power must fail");
    linear_power::LogarithmicTable const shortTable({1.0, 2.0}, {2.0, 4.0}, true);
    requireClose(shortTable(1.5), 3.0, 1.0e-14, "Two-sample interpolation remains bounded");
}

void testNormalization() {
    requireClose(linear_power::topHatWindow(0.0), 1.0, 0.0, "Top-hat zero limit");
    requireClose(linear_power::topHatWindow(1.0e-7), 1.0 - 1.0e-15, 1.0e-15,
        "Top-hat small argument avoids cancellation");
    // P(k)=k^-2 has sigma_R^2=3/(10*pi*R). The finite lower limit
    // contributes less than 6e-8 relative error at R=8/0.7.
    std::vector<double> const wavenumbers = {1.0e-8, 1.0e-4, 1.0, 1000.0};
    std::vector<double> values;
    for (double const wavenumber : wavenumbers) {
        values.push_back(1.0 / (wavenumber * wavenumber));
    }
    linear_power::LogarithmicTable const power(wavenumbers, values, true);
    double const radius = 8.0 / 0.7;
    double const variance = linear_power::sigmaSquared(power, radius);
    double const expected = 3.0 / (10.0 * std::acos(-1.0) * radius);
    requireClose(variance / expected, 1.0, 2.0e-6, "Sigma integral against analytic spectrum");
    double const factor = linear_power::normalization(power, 0.8, 0.7);
    requireClose(factor * variance, 0.64, 1.0e-14, "Sigma8 normalization");
    requireFailure(
        [&] { linear_power::normalization(power, -0.8, 0.7); }, "Negative sigma8 must fail");
}

void testMatterTransfer() {
    double const combined = linear_power::matterTransfer(2.0, -1.0, 0.2);
    requireClose(combined, 1.4, 1.0e-15, "Combine signed transfers before squaring");
    requireClose(combined * combined, 1.96, 1.0e-15, "Matter power includes the cross term");
    requireClose(linear_power::matterDerivative(-4.0, 0.0, 3.0, 0.2), 2.0, 0.0,
        "Synchronous zero CDM velocity retains nonzero growing-mode displacement velocity");
    requireClose(linear_power::matterDerivative(-4.0, 0.5, 3.0, 0.2), 1.7, 1.0e-15,
        "Baryon continuity contributes to the matter displacement derivative");
}

}    // namespace detail

int main() {
    try {
        detail::testInterpolation();
        detail::testNormalization();
        detail::testMatterTransfer();
        std::puts("Linear power interpolation, normalization, and displacement tests passed");
        return EXIT_SUCCESS;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
}
