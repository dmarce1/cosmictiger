/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/analysis/rockstar_properties.hpp>

namespace detail {

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireClose(double actual, double expected, double tolerance, char const* message) {
    std::uint64_t const exponent =
        std::bit_cast<std::uint64_t>(actual) & UINT64_C(0x7ff0000000000000);
    require(exponent != UINT64_C(0x7ff0000000000000) && std::abs(actual - expected) <= tolerance,
        message);
}

template <class Callable>
void requireInvalid(Callable const& callable) {
    bool rejected = false;
    try {
        callable();
    } catch (std::invalid_argument const&) {
        rejected = true;
    }
    require(rejected, "Invalid property input must be rejected");
}

void testSphericalOverdensity() {
    double const density = 3.0 / (4.0 * std::numbers::pi);
    auto const profile = rockstar::sphericalOverdensity({0.1, 0.2, 1.2, 4.0}, density);
    require(profile.count == 3, "SO mass must count particles inside the mean-density crossing");
    requireClose(
        profile.radius, std::cbrt(3.0), 1.e-14, "SO boundary must have the requested mean density");
    auto const empty = rockstar::sphericalOverdensity({}, density);
    require(empty.count == 0 && empty.radius == 0.0, "Empty profile must have no SO boundary");
    auto const diffuse = rockstar::sphericalOverdensity({3.0, 4.0}, density);
    require(diffuse.count == 0 && diffuse.radius == 0.0, "Diffuse profile must have no SO mass");
    auto const tied = rockstar::sphericalOverdensity({1.3, 1.3, 1.3}, density);
    require(tied.count == 3, "Equal-radius particles must form one complete radial shell");
    auto const nonmonotonic = rockstar::sphericalOverdensity({0.1, 1.4, 1.4, 1.4, 10.0}, density);
    require(nonmonotonic.count == 4, "SO search must find the outermost density crossing");
    auto const remotePair = rockstar::sphericalOverdensity({0.1, 100.0, 100.000001}, density);
    require(remotePair.count == 1, "A distant close pair must not inflate a spherical mass");
    auto const central = rockstar::sphericalOverdensity({0.0, 0.0, 4.0}, density);
    require(central.count == 2 && central.radius > 0.0,
        "Coincident central particles must retain their complete SO mass");
    std::vector<double> radii;
    for (int index = 0; index < 300; ++index) {
        radii.push_back(0.02 * index);
    }
    for (double const threshold : {0.1, 1.0, 10.0, 100.0, 1000.0}) {
        auto const result = rockstar::sphericalOverdensity(radii, threshold);
        auto const count =
            std::upper_bound(radii.begin(), radii.end(), result.radius) - radii.begin();
        require(size_t(count) == result.count, "Returned SO radius must contain exactly its mass");
        double const measured =
            result.count / (4.0 * std::numbers::pi / 3.0 * std::pow(result.radius, 3));
        requireClose(measured / threshold, 1.0, 2.e-14, "SO density and mass must be consistent");
    }
    requireInvalid([&]() { rockstar::sphericalOverdensity({1.0, 0.5}, density); });
    requireInvalid([&]() { rockstar::sphericalOverdensity({-1.0}, density); });
    requireInvalid([&]() { rockstar::sphericalOverdensity({1.0}, 0.0); });
    requireInvalid([&]() {
        rockstar::sphericalOverdensity({std::numeric_limits<double>::quiet_NaN()}, density);
    });
}

void testVirialOverdensity() {
    auto const eds = rockstar::virialOverdensity(0.1, 1.0);
    requireClose(eds.critical, 18.0 * std::numbers::pi * std::numbers::pi, 1.e-12,
        "Einstein-de Sitter overdensity must be 18 pi squared");
    requireClose(eds.meanMatter, eds.critical, 1.e-12,
        "Critical and matter density must coincide in Einstein-de Sitter");
    auto const today = rockstar::virialOverdensity(1.0, 0.3);
    requireClose(today.critical, 101.14287921960845, 1.e-11,
        "Flat LCDM virial overdensity must use Omega_m(a)-1");
    requireClose(today.meanMatter, 337.14293073202816, 1.e-11,
        "Virial density must convert from critical to mean matter density");
    auto const earlier = rockstar::virialOverdensity(0.5, 0.3);
    requireClose(earlier.matterFraction, 0.3 / (0.3 + 0.7 * 0.125), 1.e-15,
        "Matter fraction must evolve with scale factor");
    require(earlier.critical > today.critical && earlier.meanMatter < today.meanMatter,
        "Critical and background overdensities must evolve toward their early-time limits");
    auto const radiation = rockstar::virialOverdensity(0.5, 0.3, 0.01);
    requireClose(radiation.matterFraction, 0.3 / (0.3 + 0.01 / 0.5 + 0.69 * 0.125), 1.e-15,
        "Flat expansion must include the supplied radiation contribution");
    auto const open = rockstar::virialOverdensityFromMatterFraction(0.3, true);
    requireClose(open.critical, 18.0 * std::numbers::pi * std::numbers::pi - 42.0 - 15.68, 1.e-12,
        "Open matter cosmology must use the open-universe fit");
    requireInvalid([]() { rockstar::virialOverdensity(0.0, 0.3); });
    requireInvalid([]() { rockstar::virialOverdensityFromMatterFraction(0.0); });
}

void verifyEigenpairs(
    rockstar::Matrix3 const& matrix, std::array<double, 3> const& expected, double scale) {
    auto const result = rockstar::symmetricEigenpairs(matrix);
    for (size_t axis = 0; axis < 3; ++axis) {
        requireClose(result.values[axis] / scale, expected[axis] / scale, 2.e-13,
            "Shape eigenvalues must be sorted and independent of orientation");
        for (size_t row = 0; row < 3; ++row) {
            double product = 0.0;
            for (size_t column = 0; column < 3; ++column) {
                product += matrix[row][column] / scale * result.vectors[axis][column];
            }
            requireClose(product, result.values[axis] / scale * result.vectors[axis][row], 2.e-13,
                "Shape eigenvectors must satisfy the tensor eigenproblem");
        }
        for (size_t other = 0; other < 3; ++other) {
            double dot = 0.0;
            for (size_t component = 0; component < 3; ++component) {
                dot += result.vectors[axis][component] * result.vectors[other][component];
            }
            requireClose(dot, axis == other ? 1.0 : 0.0, 2.e-13,
                "Shape eigenvectors must be orthonormal even for repeated eigenvalues");
        }
    }
}

void testEigenpairs() {
    for (double const scale : {1.e-200, 1.e-6, 1.0, 1.e200}) {
        std::array<double, 3> const eigenvalues{9.0 * scale, 4.0 * scale, scale};
        rockstar::Matrix3 diagonal{};
        for (size_t axis = 0; axis < 3; ++axis) {
            diagonal[axis][axis] = eigenvalues[axis];
        }
        verifyEigenpairs(diagonal, eigenvalues, scale);
        auto const result = rockstar::symmetricEigenpairs(diagonal);
        requireClose(std::sqrt(result.values[1] / result.values[0]), 2.0 / 3.0, 1.e-14,
            "An axis-aligned ellipsoid must not be reported as spherical");
        for (double const angle : {0.01, 0.4, 1.2, 2.1}) {
            double const cosine = std::cos(angle);
            double const sine = std::sin(angle);
            // Rotate first about z, then y, so every component participates.
            rockstar::Matrix3 const rotation{{{cosine * cosine, -cosine * sine, sine},
                {sine, cosine, 0.0}, {-sine * cosine, sine * sine, cosine}}};
            rockstar::Matrix3 tensor{};
            for (size_t row = 0; row < 3; ++row) {
                for (size_t column = 0; column < 3; ++column) {
                    for (size_t axis = 0; axis < 3; ++axis) {
                        tensor[row][column] +=
                            rotation[row][axis] * eigenvalues[axis] * rotation[column][axis];
                    }
                }
            }
            verifyEigenpairs(tensor, eigenvalues, scale);
        }
    }
    rockstar::Matrix3 zero{};
    verifyEigenpairs(zero, {0.0, 0.0, 0.0}, 1.0);
    rockstar::Matrix3 repeated{};
    for (size_t axis = 0; axis < 3; ++axis) {
        repeated[axis][axis] = 2.0;
    }
    verifyEigenpairs(repeated, {2.0, 2.0, 2.0}, 1.0);
    rockstar::Matrix3 rankOne{{{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}}};
    verifyEigenpairs(rankOne, {14.0, 0.0, 0.0}, 1.0);
}

void testNfwFits() {
    for (double const concentration : {2.0, 5.0, 10.0, 20.0}) {
        double const normalization =
            std::log1p(concentration) - concentration / (1.0 + concentration);
        for (double const outerRadius : {0.001, 1.0}) {
            std::vector<double> radii;
            constexpr size_t kParticleCount = 5000;
            for (size_t index = 1; index <= kParticleCount; ++index) {
                // Exact deterministic NFW mass quantiles, independent of the fitter.
                double const targetMass = normalization * index / kParticleCount;
                double lower = 0.0;
                double upper = concentration;
                for (int iteration = 0; iteration < 64; ++iteration) {
                    double const middle = 0.5 * (lower + upper);
                    double const mass = std::log1p(middle) - middle / (1.0 + middle);
                    if (mass < targetMass) {
                        lower = middle;
                    } else {
                        upper = middle;
                    }
                }
                radii.push_back(outerRadius * 0.5 * (lower + upper) / concentration);
            }
            auto const fit = rockstar::nfwFit(radii);
            require(fit.scaleRadius > 0.0 && fit.density > 0.0,
                "Resolved NFW profiles must have finite positive fitted properties");
            requireClose(outerRadius / fit.scaleRadius / concentration, 1.0, 0.005,
                "Binned NFW likelihood must recover the input concentration");
            double const fittedConcentration = outerRadius / fit.scaleRadius;
            double const recoveredMass = 4.0 * std::numbers::pi * fit.density *
                std::pow(fit.scaleRadius, 3) *
                (std::log1p(fittedConcentration) -
                    fittedConcentration / (1.0 + fittedConcentration));
            requireClose(recoveredMass / kParticleCount, 1.0, 1.e-12,
                "NFW scale density must normalize to the fitted particle mass");
        }
    }
    require(rockstar::nfwFit({}).scaleRadius == 0.0, "Empty profile must have no NFW fit");
    require(rockstar::nfwFit(std::vector<double>(20, 1.0)).scaleRadius == 0.0,
        "A single radial bin cannot resolve an NFW concentration");
    require(rockstar::nfwFit(std::vector<double>(100, 0.0)).scaleRadius == 0.0,
        "Coincident central particles must not generate an infinite NFW density");
    require(rockstar::nfwFit(std::vector<double>(100, 1.0)).scaleRadius == 0.0,
        "A single occupied shell must not generate a fictitious resolved NFW fit");
    std::vector<double> uniformSphere;
    for (int index = 1; index <= 5000; ++index) {
        uniformSphere.push_back(std::cbrt(index / 5000.0));
    }
    require(rockstar::nfwFit(uniformSphere).scaleRadius == 0.0,
        "A boundary-limited NFW likelihood must report an unresolved fit");
}

void testKlypinFit() {
    constexpr double kMaximumRadiusRatio = 2.1625815870646098;
    double const peakIntegral =
        std::log1p(kMaximumRadiusRatio) - kMaximumRadiusRatio / (1.0 + kMaximumRadiusRatio);
    for (double const concentration : {3.0, 5.0, 20.0, 100.0}) {
        double const integral = std::log1p(concentration) - concentration / (1.0 + concentration);
        double const virialRadius = 0.005;
        double const gravitationalMass = 1.e-8;
        double const velocity = std::sqrt(gravitationalMass / virialRadius * concentration /
            integral * peakIntegral / kMaximumRadiusRatio);
        double const radius =
            rockstar::klypinScaleRadius(velocity, virialRadius, gravitationalMass);
        requireClose(radius / (virialRadius / concentration), 1.0, 2.e-11,
            "Klypin fit must invert the NFW circular-velocity relation");
    }
    requireClose(rockstar::klypinScaleRadius(1.0, 1.0, 1.0), 1.0 / kMaximumRadiusRatio, 1.e-14,
        "Vmax=Vvir must retain the limiting concentration");
    require(rockstar::klypinScaleRadius(0.5, 1.0, 1.0) == 0.0,
        "Inconsistent Vmax<Vvir must terminate without a fictitious scale radius");
    require(rockstar::klypinScaleRadius(0.0, 1.0, 1.0) == 0.0,
        "Zero circular velocity must not enter a root-finding loop");
    require(rockstar::klypinScaleRadius(std::numeric_limits<double>::infinity(), 1.0, 1.0) == 0.0,
        "Nonfinite circular velocity must not enter a root-finding loop");
}

}    // namespace detail

TEST(RockstarProperties, SphericalOverdensity) {
    detail::testSphericalOverdensity();
}

TEST(RockstarProperties, VirialOverdensity) {
    detail::testVirialOverdensity();
}

TEST(RockstarProperties, Eigenpairs) {
    detail::testEigenpairs();
}

TEST(RockstarProperties, NfwFits) {
    detail::testNfwFits();
}

TEST(RockstarProperties, KlypinFit) {
    detail::testKlypinFit();
}
