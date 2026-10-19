#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>

#include <cosmictiger/cosmology/recfast.hpp>
#include <cosmictiger/cosmology/zero_order.hpp>

namespace detail {

bool isFinite(double value) {
    static_assert(std::numeric_limits<double>::is_iec559);
    constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
    return (std::bit_cast<std::uint64_t>(value) & kExponentMask) != kExponentMask;
}

void require(bool condition, char const* label) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", label);
        std::exit(EXIT_FAILURE);
    }
}

void requireClose(double actual, double expected, double tolerance, char const* label) {
    require(isFinite(actual) && isFinite(expected), "Nonfinite background result");
    if (std::abs(actual - expected) > tolerance * std::max(std::abs(actual), std::abs(expected))) {
        std::fprintf(stderr, "%s: actual %.17g, expected %.17g\n", label, actual, expected);
        std::exit(EXIT_FAILURE);
    }
}

void testBackground(char const* historyPath) {
    using namespace constants;
    auto const history = recfast::loadIonizationHistory(historyPath);
    CosmicParameters parameters{};
    parameters.omegaB = 0.049389;
    parameters.omegaC = 0.26503;
    parameters.neff = 3.046;
    parameters.y = 0.2454006;
    parameters.theta = 2.7255 / kTcmb;
    parameters.hubble = 0.6732;
    double const hubbleToday = parameters.hubble * kH0;
    double const criticalDensity = 3.0 * hubbleToday * hubbleToday / (8.0 * std::acos(-1.0) * kG);
    double const radiationConstant = 4.0 * kSigma / kC;
    parameters.omegaGam =
        radiationConstant * std::pow(parameters.theta * kTcmb, 4) / (criticalDensity * kC * kC);
    parameters.omegaNu =
        parameters.omegaGam * parameters.neff * (7.0 / 8.0) * std::pow(4.0 / 11.0, 4.0 / 3.0);
    ZeroOrderUniverse universe;
    createZeroOrderUniverse(&universe, history, 1.1, parameters);

    auto const baryonDensity = [=](double scaleFactor) {
        return criticalDensity * parameters.omegaB / std::pow(scaleFactor, 3);
    };
    double const sampleScale = 1.0 / 51.0;
    double const sampleDensity = baryonDensity(sampleScale);
    double const hydrogenNumberDensity = (1.0 - parameters.y) * sampleDensity / kMh;
    double const heliumNumberDensity = parameters.y * sampleDensity / (4.0 * kMh);
    double const pressure = universe.k(sampleScale) * std::pow(sampleDensity, 5.0 / 3.0);
    double const temperature = pressure /
        (kB * (hydrogenNumberDensity * (1.0 + history(sampleScale)) + heliumNumberDensity));
    // Independent thermal solver reference: bundled RECFAST 1.5.2, fiducial
    // parameters above, with untruncated fixed-form source and its matter
    // temperature output enabled. A 2% allowance covers the separate background
    // integrator, approximate atomic masses/constants, and finite xe table.
    // https://www.astro.ubc.ca/people/scott/recfast.readme
    requireClose(temperature, 50.669107, 0.02, "Background temperature against RECFAST at z=50");

    auto const& thermalTable = universe.k;
    auto const& soundTable = universe.cs2;
    require(thermalTable.values.size() == soundTable.values.size(), "Thermal table size mismatch");
    size_t const last = thermalTable.values.size() - 1;
    require(last >= 2, "Background thermal table too short");
    for (double const soundSpeed : soundTable.values) {
        require(isFinite(soundSpeed) && soundSpeed > 0.0, "Invalid background sound speed");
    }
    auto const scaleAt = [&](size_t index) {
        return std::exp(thermalTable.minloga + index * thermalTable.dloga);
    };
    auto const pressureAt = [&](size_t index) {
        return thermalTable.values[index] * std::pow(baryonDensity(scaleAt(index)), 5.0 / 3.0);
    };
    auto const derivativeBetween = [&](size_t left, size_t right) {
        return (pressureAt(right) - pressureAt(left)) /
            (baryonDensity(scaleAt(right)) - baryonDensity(scaleAt(left))) / (kC * kC);
    };
    requireClose(soundTable.values[last], derivativeBetween(last - 1, last), 2.e-10,
        "Final sound speed uses one-sided derivative and c^2 normalization");
    requireClose(soundTable.values[last - 1], derivativeBetween(last - 2, last), 2.e-10,
        "Penultimate sound speed retains centered derivative");
    requireClose(soundTable.values[last], soundTable.values[last - 1], 0.03,
        "Sound speed is continuous at final endpoint");
    requireClose(soundTable(universe.amax), soundTable.values[last], 2.e-10,
        "Sound speed interpolation reaches final endpoint");
    std::printf("Production background: Tgas(z=50)=%.9g K, final cs2/c2=%.9g\n", temperature,
        soundTable.values[last]);
}

}    // namespace detail

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fputs("Usage: recombination_background_test FIDUCIAL_RECFAST_HISTORY\n", stderr);
        return EXIT_FAILURE;
    }
    try {
        detail::testBackground(argv[1]);
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Background regression failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
