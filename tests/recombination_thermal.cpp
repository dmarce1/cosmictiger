#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/cosmology/recombination_thermal.hpp>

namespace detail {

bool isFinite(double value) {
    static_assert(std::numeric_limits<double>::is_iec559);
    // Keep NaN/Inf checks effective under the production -ffast-math flags.
    constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
    return (std::bit_cast<std::uint64_t>(value) & kExponentMask) != kExponentMask;
}

void requireClose(double actual, double expected, double relativeTolerance, char const* label) {
    double const scale = std::max(std::abs(actual), std::abs(expected));
    if (!isFinite(actual) || !isFinite(expected) ||
        std::abs(actual - expected) > relativeTolerance * scale) {
        std::fprintf(stderr, "%s: actual %.17g, expected %.17g\n", label, actual, expected);
        throw std::runtime_error(label);
    }
}

void testComptonCoefficient() {
    using namespace constants;
    double const hubbleToday = 0.678 * kH0;
    double const criticalDensity = 3.0 * hubbleToday * hubbleToday / (8.0 * std::acos(-1.0) * kG);
    double const radiationConstant = 4.0 * kSigma / kC;
    double const photonDensityParameter =
        radiationConstant * std::pow(kTcmb, 4) / (criticalDensity * kC * kC);
    for (double const scaleFactor : {1.e-4, 1.e-3, 0.01, 0.1, 1.0}) {
        double const radiationTemperature = kTcmb / scaleFactor;
        for (double const heliumMassFraction : {0.0, 0.24, 0.4}) {
            double const heliumNumberFraction =
                heliumMassFraction / (4.0 * (1.0 - heliumMassFraction));
            for (double const electronFraction :
                {0.0, 2.e-4, 1.0, 1.0 + 2.0 * heliumNumberFraction}) {
                double const directPhysicalRate = 8.0 * kSigmaT * radiationConstant *
                    std::pow(radiationTemperature, 4) / (3.0 * kMe * kC) * electronFraction /
                    (1.0 + heliumNumberFraction + electronFraction);
                for (double const baryonDensityParameter : {0.02, 0.049, 0.1}) {
                    double const baryonDensity =
                        criticalDensity * baryonDensityParameter / std::pow(scaleFactor, 3);
                    double const hydrogenNumberDensity =
                        (1.0 - heliumMassFraction) * baryonDensity / kMh;
                    double const heliumNumberDensity =
                        heliumMassFraction * baryonDensity / (4.0 * kMh);
                    double const electronNumberDensity = electronFraction * hydrogenNumberDensity;
                    // CDM changes H, but must not dilute the physical gas coupling
                    // Gamma_C. Changing the baryon density at fixed composition
                    // also leaves Gamma_C unchanged.
                    for (double const coldMatterDensityParameter : {0.0, 0.26, 0.7}) {
                        double const matterDensityParameter =
                            baryonDensityParameter + coldMatterDensityParameter;
                        double const hubbleRate = hubbleToday *
                            std::sqrt(photonDensityParameter / std::pow(scaleFactor, 4) +
                                matterDensityParameter / std::pow(scaleFactor, 3) + 1.0 -
                                photonDensityParameter - matterDensityParameter);
                        double const coupling = recombination::comptonTemperatureCoupling(
                            scaleFactor, photonDensityParameter, baryonDensityParameter,
                            hydrogenNumberDensity, heliumNumberDensity, electronNumberDensity,
                            hubbleRate);
                        requireClose(coupling, directPhysicalRate / hubbleRate, 2.e-14,
                            "Compton coefficient against radiation T^4 formula");
                        requireClose(coupling * hubbleRate, directPhysicalRate, 2.e-14,
                            "Physical Compton rate independent of CDM and baryon density");
                    }
                }
            }
        }
    }
}

void testTemperatureEvolution() {
    double const initialTemperature = 300.0;
    requireClose(recombination::advanceGasTemperature(initialTemperature, 1000.0, 0.0, 1.e10),
        initialTemperature, 0.0, "Zero timestep preserves temperature");
    double const logScaleInterval = 1.0;
    int const stepCount = 1000;
    double temperature = initialTemperature;
    for (int step = 0; step < stepCount; ++step) {
        temperature = recombination::advanceGasTemperature(
            temperature, 1000.0, logScaleInterval / stepCount, 0.0);
    }
    requireClose(temperature, initialTemperature * std::exp(-2.0 * logScaleInterval), 4.e-7,
        "Expansion-only T proportional to a^-2");

    // Independent exact solution with constant radiation temperature and rate:
    // T = T_eq + (T_initial - T_eq) exp[-(2 + Gamma_C/H) Delta ln(a)].
    double const radiationTemperature = 100.0;
    double const coupling = 3.0;
    double const equilibriumTemperature = radiationTemperature * coupling / (2.0 + coupling);
    for (double const startingTemperature : {1.0, 300.0}) {
        temperature = startingTemperature;
        for (int step = 0; step < stepCount; ++step) {
            temperature = recombination::advanceGasTemperature(
                temperature, radiationTemperature, logScaleInterval / stepCount, coupling);
        }
        double const exactTemperature = equilibriumTemperature +
            (startingTemperature - equilibriumTemperature) *
                std::exp(-(2.0 + coupling) * logScaleInterval);
        requireClose(temperature, exactTemperature, 2.e-7,
            "Gas heating and cooling against exact constant-rate solution");
    }
    requireClose(recombination::advanceGasTemperature(
                     equilibriumTemperature, radiationTemperature, 0.05, coupling),
        equilibriumTemperature, 2.e-14, "Thermal equilibrium is stationary");

    double const strongCoupling = 1.e10;
    double const strongEquilibrium = radiationTemperature * strongCoupling / (2.0 + strongCoupling);
    for (double const startingTemperature : {1.0, 300.0}) {
        requireClose(recombination::advanceGasTemperature(
                         startingTemperature, radiationTemperature, 0.01, strongCoupling),
            strongEquilibrium, 2.e-7, "Stiff Compton coupling approaches equilibrium");
    }
}

}    // namespace detail

TEST(RecombinationThermal, FiniteValueValidation) {
    EXPECT_FALSE(detail::isFinite(std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(detail::isFinite(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_TRUE(detail::isFinite(1.0));
}

TEST(RecombinationThermal, ComptonCoefficient) {
    detail::testComptonCoefficient();
}

TEST(RecombinationThermal, TemperatureEvolution) {
    detail::testTemperatureEvolution();
}
