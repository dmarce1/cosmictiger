#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/adaptive_softening.hpp>
#if defined(ADAPTIVE_SOFTENING_TEST_SIMD)
#include <cosmictiger/simd.hpp>
#endif

namespace detail {

using radial_kernel::Family;
using radial_kernel::KernelSpec;
using radial_kernel::KernelTable;
using std::size_t;
using Position = std::array<double, 3>;

constexpr std::array<KernelSpec, 14> kSpecifications = {{{Family::Spline, 3}, {Family::Spline, 4},
    {Family::Spline, 5}, {Family::Spline, 6}, {Family::Spline, 7}, {Family::Spline, 8},
    {Family::Spline, 9}, {Family::Wendland, 2}, {Family::Wendland, 5}, {Family::Wendland, 8},
    {Family::Ferrers, 2}, {Family::Ferrers, 4}, {Family::Ferrers, 6}, {Family::Ferrers, 8}}};
constexpr double kUnitSphereVolume = 4 * std::numbers::pi / 3;

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void close(double actual, double expected, double tolerance, char const* label) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(
            std::string(label) + ": " + std::to_string(actual) + " != " + std::to_string(expected));
    }
}

double separation(Position const& first, Position const& second) {
    double squaredRadius = 0;
    for (size_t dimension = 0; dimension < first.size(); ++dimension) {
        double const displacement = first[dimension] - second[dimension];
        squaredRadius += displacement * displacement;
    }
    return std::sqrt(squaredRadius);
}

double numberConstraint(KernelTable const& table, std::vector<Position> const& positions,
    size_t particle, double supportRadius) {
    double value = 0;
    for (auto const& position : positions) {
        value += kUnitSphereVolume *
            radial_kernel::value(table, separation(positions[particle], position) / supportRadius);
    }
    return value;
}

struct SofteningState {
    std::vector<double> radii;
    std::vector<double> corrections;
    size_t lowerBoundCount = 0;
    size_t upperBoundCount = 0;
};

SofteningState solveRadii(KernelTable const& table, std::vector<Position> const& positions,
    std::vector<double> const& masses, double target, double minimumRadius, double maximumRadius) {
    SofteningState result;
    result.radii.resize(positions.size());
    result.corrections.resize(positions.size());
    for (size_t particle = 0; particle < positions.size(); ++particle) {
        double lower = minimumRadius;
        double upper = maximumRadius;
        bool radiusIsAdaptive = true;
        if (numberConstraint(table, positions, particle, lower) >= target) {
            upper = lower;
            radiusIsAdaptive = false;
            ++result.lowerBoundCount;
        } else if (numberConstraint(table, positions, particle, upper) <= target) {
            lower = upper;
            radiusIsAdaptive = false;
            ++result.upperBoundCount;
        } else {
            // Deliberately use bisection, independently of the analytic
            // constraint derivative being tested.
            for (int iteration = 0; iteration < 64; ++iteration) {
                double const middle = (lower + upper) / 2;
                if (numberConstraint(table, positions, particle, middle) < target) {
                    lower = middle;
                } else {
                    upper = middle;
                }
            }
        }
        double const supportRadius = (lower + upper) / 2;
        result.radii[particle] = supportRadius;
        double constraintDerivative = 0;
        double potentialDerivative = 0;
        for (size_t neighbor = 0; neighbor < positions.size(); ++neighbor) {
            double const radius = separation(positions[particle], positions[neighbor]);
            constraintDerivative +=
                adaptive_softening::numberConstraintPair(table, radius, supportRadius)
                    .radiusDerivative;
            if (neighbor != particle) {
                potentialDerivative += masses[neighbor] *
                    adaptive_softening::potentialRadiusDerivative(table, radius, supportRadius);
            }
        }
        if (radiusIsAdaptive) {
            require(constraintDerivative > 0, "positive unconstrained radius derivative");
            close(numberConstraint(table, positions, particle, supportRadius), target, 2e-14,
                "solved number constraint including self");
        }
        result.corrections[particle] = adaptive_softening::correctionCoefficient(
            potentialDerivative, constraintDerivative, radiusIsAdaptive);
    }
    return result;
}

double totalEnergy(KernelTable const& table, std::vector<Position> const& positions,
    std::vector<double> const& masses, double target, double minimumRadius, double maximumRadius) {
    auto const state = solveRadii(table, positions, masses, target, minimumRadius, maximumRadius);
    double energy = 0;
    for (size_t first = 0; first < positions.size(); ++first) {
        for (size_t second = first + 1; second < positions.size(); ++second) {
            double const radius = separation(positions[first], positions[second]);
            // Explicit pair sum excludes self energy and does not use the
            // force helper or its potential derivative.
            energy -= 0.5 * masses[first] * masses[second] *
                (radial_kernel::potential(table, radius / state.radii[first]) / state.radii[first] +
                    radial_kernel::potential(table, radius / state.radii[second]) /
                        state.radii[second]);
        }
    }
    return energy;
}

void testEnergyGradient(KernelSpec spec, bool useBounds, bool unequalClassMasses, double targetOverride = 0.0) {
    auto const table = radial_kernel::makeKernelTable(spec);
    std::vector<Position> positions;
    std::vector<double> masses;
    for (int particle = 0; particle < 16; ++particle) {
        positions.push_back({std::sin(0.71 * particle) + 0.03 * particle, std::cos(1.37 * particle),
            std::sin(0.43 * particle + 1)});
        // Each of the two interleaved particle classes has one constant mass.
        masses.push_back(unequalClassMasses ? (particle % 2 == 0 ? 0.15 : 0.85) : 1.0);
    }
    double const target = targetOverride > 0.0 ? targetOverride : 4 * kUnitSphereVolume * radial_kernel::value(table, 0.0);
    double minimumRadius = 1e-6;
    double maximumRadius = 64;
    if (useBounds) {
        auto const unconstrained =
            solveRadii(table, positions, masses, target, minimumRadius, maximumRadius);
        auto orderedRadii = unconstrained.radii;
        std::sort(orderedRadii.begin(), orderedRadii.end());
        minimumRadius = (orderedRadii[3] + orderedRadii[4]) / 2;
        maximumRadius = (orderedRadii[11] + orderedRadii[12]) / 2;
    }
    auto const state = solveRadii(table, positions, masses, target, minimumRadius, maximumRadius);
    if (useBounds) {
        require(state.lowerBoundCount == 4 && state.upperBoundCount == 4,
            "mixed lower bounds, interior solutions, and upper bounds");
    }
    std::vector<Position> accelerations(positions.size());
    std::vector<Position> uncorrectedAccelerations(positions.size());
    Position totalForce{};
    Position totalTorque{};
    for (size_t first = 0; first < positions.size(); ++first) {
        for (size_t second = 0; second < positions.size(); ++second) {
            if (first == second) {
                continue;
            }
            double const radius = separation(positions[first], positions[second]);
            auto const pair = adaptive_softening::pairInteraction(table, radius, state.radii[first],
                state.radii[second], state.corrections[first], state.corrections[second],
                masses[first], masses[second]);
            auto const uncorrected = adaptive_softening::pairInteraction(table, radius,
                state.radii[first], state.radii[second], 0.0, 0.0, masses[first], masses[second]);
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                double const displacement =
                    positions[first][dimension] - positions[second][dimension];
                accelerations[first][dimension] -=
                    masses[second] * pair.accelerationFactor * displacement;
                uncorrectedAccelerations[first][dimension] -=
                    masses[second] * uncorrected.accelerationFactor * displacement;
            }
        }
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            totalForce[dimension] += masses[first] * accelerations[first][dimension];
            size_t const next = (dimension + 1) % 3;
            size_t const last = (dimension + 2) % 3;
            totalTorque[dimension] += masses[first] *
                (positions[first][next] * accelerations[first][last] -
                    positions[first][last] * accelerations[first][next]);
        }
    }
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        close(totalForce[dimension], 0, 2e-13, "zero net force");
        close(totalTorque[dimension], 0, 2e-13, "zero net torque");
    }
    double largestUncorrectedError = 0;
    for (size_t particle = 0; particle < positions.size(); ++particle) {
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            constexpr double kStep = 2e-6;
            auto displaced = positions;
            displaced[particle][dimension] += kStep;
            double const upperEnergy =
                totalEnergy(table, displaced, masses, target, minimumRadius, maximumRadius);
            displaced[particle][dimension] -= 2 * kStep;
            double const lowerEnergy =
                totalEnergy(table, displaced, masses, target, minimumRadius, maximumRadius);
            double const expected = -(upperEnergy - lowerEnergy) / (2 * kStep * masses[particle]);
            close(accelerations[particle][dimension], expected, 1e-7,
                "force matches re-solved total energy gradient");
            largestUncorrectedError = std::max(largestUncorrectedError,
                std::abs(uncorrectedAccelerations[particle][dimension] - expected));
        }
    }
    require(largestUncorrectedError > 0.01, "energy test detects missing adaptive correction");
}

void testKernelDerivatives(KernelSpec spec) {
    auto const table = radial_kernel::makeKernelTable(spec);
    constexpr double kSupportRadius = 1.3;
    for (int sample = 0; sample <= 120; ++sample) {
        double const radius = 0.01 * sample * kSupportRadius;
        constexpr double kStep = 2e-6;
        double const upperRadius = kSupportRadius + kStep;
        double const lowerRadius = kSupportRadius - kStep;
        double const potentialDerivative =
            (radial_kernel::potential(table, radius / upperRadius) / upperRadius -
                radial_kernel::potential(table, radius / lowerRadius) / lowerRadius) /
            (2 * kStep);
        close(adaptive_softening::potentialRadiusDerivative(table, radius, kSupportRadius),
            potentialDerivative, 3e-9, "analytic potential support derivative");
        double const constraintDerivative = kUnitSphereVolume *
            (radial_kernel::value(table, radius / upperRadius) -
                radial_kernel::value(table, radius / lowerRadius)) /
            (2 * kStep);
        auto const constraint =
            adaptive_softening::numberConstraintPair(table, radius, kSupportRadius);
        if (sample == 100) {
            // The lowest-order kernels are only C1 at support. A centered
            // finite difference there has a first-order truncation error.
            close(constraint.radiusDerivative, 0, 2e-14,
                "zero number constraint derivative at support");
        } else {
            close(constraint.radiusDerivative, constraintDerivative, 2e-8,
                "analytic number constraint support derivative");
        }
        auto const pair = adaptive_softening::pairInteraction(
            table, radius, kSupportRadius, 0.9 * kSupportRadius, -0.1, -0.2, 0.15, 0.85);
        auto const reversed = adaptive_softening::pairInteraction(
            table, radius, 0.9 * kSupportRadius, kSupportRadius, -0.2, -0.1, 0.85, 0.15);
        close(pair.accelerationFactor, reversed.accelerationFactor, 2e-15, "symmetric pair force");
        close(pair.potential, reversed.potential, 2e-15, "symmetric pair potential");
        auto const singlePrecision = adaptive_softening::pairInteraction(table,
            static_cast<float>(radius), static_cast<float>(kSupportRadius),
            static_cast<float>(0.9 * kSupportRadius), -0.1f, -0.2f, 0.15f, 0.85f);
        close(
            singlePrecision.accelerationFactor, pair.accelerationFactor, 6e-6, "float pair force");
        close(singlePrecision.potential, pair.potential, 6e-6, "float pair potential");
    }
    for (double const radius : {1.3, 1.31, 2.0, 1e20}) {
        close(adaptive_softening::potentialRadiusDerivative(table, radius, kSupportRadius), 0, 0,
            "exactly zero exterior support derivative");
        auto const pair = adaptive_softening::pairInteraction(
            table, radius, kSupportRadius, 0.8, -0.3, -0.7, 0.15, 0.85);
        close(pair.potential, 1 / radius, 2e-15, "Newtonian exterior potential");
        close(pair.accelerationFactor, 1 / (radius * radius * radius), 2e-15,
            "Newtonian exterior acceleration");
    }
    double const interior = std::nextafter(kSupportRadius, 0.0);
    double const derivative =
        adaptive_softening::potentialRadiusDerivative(table, interior, kSupportRadius);
    require(std::isfinite(derivative) && derivative <= 0, "stable boundary support derivative");
#if defined(ADAPTIVE_SOFTENING_TEST_SIMD)
    for (int sample = 0; sample < 10; ++sample) {
        SimdFloat radii;
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; ++lane) {
            radii[lane] = 0.1f * ((sample * SIMD_FLOAT_SIZE + lane) % 19);
        }
        auto const pair =
            adaptive_softening::pairInteraction(table, radii, SimdFloat(1.3f), SimdFloat(0.9f),
                SimdFloat(-0.1f), SimdFloat(-0.2f), SimdFloat(0.15f), SimdFloat(0.85f));
        auto const derivatives =
            adaptive_softening::potentialRadiusDerivative(table, radii, SimdFloat(1.3f));
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; ++lane) {
            auto const scalar = adaptive_softening::pairInteraction(
                table, radii[lane], 1.3f, 0.9f, -0.1f, -0.2f, 0.15f, 0.85f);
            close(pair.potential[lane], scalar.potential, 6e-6, "SIMD pair potential");
            close(pair.accelerationFactor[lane], scalar.accelerationFactor, 6e-6,
                "SIMD pair acceleration");
            close(derivatives[lane],
                adaptive_softening::potentialRadiusDerivative(table, radii[lane], 1.3f), 6e-6,
                "SIMD support derivative");
        }
    }
#endif
}

void testFerrersAnalytic() {
    auto const table = radial_kernel::makeKernelTable({Family::Ferrers, 4});
    constexpr double kSupportRadius = 1.7;
    for (int sample = 0; sample <= 100; ++sample) {
        double const q = sample / 100.0;
        double const radius = q * kSupportRadius;
        double const remainder = 1 - q * q;
        close(adaptive_softening::potentialRadiusDerivative(table, radius, kSupportRadius),
            -35.0 / 16 * remainder * remainder * remainder / (kSupportRadius * kSupportRadius),
            5e-15, "Ferrers F2 closed form support derivative");
        auto const constraint =
            adaptive_softening::numberConstraintPair(table, radius, kSupportRadius);
        close(constraint.value, 35.0 / 8 * remainder * remainder, 5e-15,
            "Ferrers F2 closed form number constraint");
        close(constraint.radiusDerivative, 35.0 / 2 * q * q * remainder / kSupportRadius, 5e-15,
            "Ferrers F2 closed form constraint derivative");
    }
    close(adaptive_softening::correctionCoefficient(-2.0, 4.0), -0.5, 0,
        "adaptive correction coefficient");
    close(adaptive_softening::correctionCoefficient(-2.0, 4.0, false), 0, 0,
        "clamped radius has zero correction");
    close(adaptive_softening::correctionCoefficient(-2.0, 0.0), 0, 0,
        "degenerate radius derivative does not divide by zero");
}

}    // namespace detail

int main() {
    try {
        detail::testFerrersAnalytic();
        for (auto const spec : detail::kSpecifications) {
            detail::testKernelDerivatives(spec);
        }
        for (auto const spec : {radial_kernel::KernelSpec{radial_kernel::Family::Ferrers, 4},
                 radial_kernel::KernelSpec{radial_kernel::Family::Ferrers, 8},
                 radial_kernel::KernelSpec{radial_kernel::Family::Spline, 3},
                 radial_kernel::KernelSpec{radial_kernel::Family::Wendland, 5}}) {
            for (bool const unequalClassMasses : {false, true}) {
                detail::testEnergyGradient(spec, false, unequalClassMasses);
                detail::testEnergyGradient(spec, true, unequalClassMasses);
                if (spec.family == radial_kernel::Family::Ferrers && spec.degree == 8) {
                    detail::testEnergyGradient(spec, false, unequalClassMasses, 12.0);
                    detail::testEnergyGradient(spec, true, unequalClassMasses, 12.0);
                    std::printf("Ferrers degree 8 target 12: unconstrained and bounded energy gradients passed, unequal classes=%d\n", int(unequalClassMasses));
                }
            }
        }
        std::puts("Adaptive softening passed analytic derivatives, pair symmetry, re-solved "
                  "energy gradients, and active radius bounds.");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "adaptive softening test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
