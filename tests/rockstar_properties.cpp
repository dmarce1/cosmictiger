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
#include <random>
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

// Radius enclosing the given NFW mass fraction, in units of the outer radius.
double nfwQuantile(double concentration, double massFraction) {
    double const targetMass =
        massFraction * (std::log1p(concentration) - concentration / (1.0 + concentration));
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
    return 0.5 * (lower + upper) / concentration;
}

// Exact NFW mass quantiles (index + 1/2) / count with the outer radius at one.
std::vector<double> nfwQuantileRadii(double concentration, size_t count) {
    std::vector<double> radii(count);
    for (size_t index = 0; index < count; ++index) {
        radii[index] = nfwQuantile(concentration, (index + 0.5) / count);
    }
    radii.back() = 1.0;
    return radii;
}

// A thirty-particle reproduction of the halo class that aborted the 128^3 LCDM
// production run at step 1367 (z = 1.47): exact rho ~ 1/r mass quantiles, in
// units of the outer radius, with the median pair moved so that the two
// equal-mass bins put the inner mass fraction barely above the rho ~ 1/r limit.
// The former ln c >= -16 window accepted c = 5e-4, a scale radius of 2007
// outer radii, which overflowed the binary16 record field after conversion to
// kpc/h, as the production halo's Rs = 134 Mpc/h did.
std::vector<double> productionFailureRadii() {
    return {0.129099, 0.223607, 0.288675, 0.341565, 0.387298, 0.428174, 0.465475, 0.5, 0.532291,
        0.562731, 0.591608, 0.619139, 0.645497, 0.67082, 0.706988, 0.707088, 0.74162, 0.763763,
        0.785281, 0.806226, 0.82664, 0.846562, 0.866025, 0.885061, 0.903696, 0.921954, 0.939858,
        0.957427, 0.974679, 1.0};
}

// Exact rho ~ 1/r mass quantiles, the c -> 0 limit of the NFW family.
std::vector<double> inverseRadiusQuantiles(size_t count) {
    std::vector<double> radii(count);
    for (size_t index = 0; index < count; ++index) {
        radii[index] = std::sqrt((index + 0.5) / count);
    }
    radii.back() = 1.0;
    return radii;
}

// Thirty radii whose single equal-mass boundary solves p(c) = 1/2 exactly at the
// given concentration, so a two-bin fit probes the window edges directly.
std::vector<double> twoBinRadii(double concentration) {
    double const boundary = nfwQuantile(concentration, 0.5);
    std::vector<double> radii;
    for (int index = 1; index <= 14; ++index) {
        radii.push_back(0.999 * boundary * std::sqrt(index / 15.0));
    }
    radii.push_back(boundary - 1.e-6);
    radii.push_back(boundary + 1.e-6);
    for (int index = 1; index <= 13; ++index) {
        radii.push_back(boundary + (1.0 - boundary) * index / 14.0);
    }
    radii.push_back(1.0);
    return radii;
}

// Half the particles inside 2e-5 outer radii drive the optimum to the upper
// window edge; the criterion is symmetric and reports that as unresolved too.
std::vector<double> centralCuspRadii() {
    std::vector<double> radii(30);
    for (size_t index = 0; index < 16; ++index) {
        radii[index] = 1.e-5 * (1.0 + (index + 0.5) / 16.0);
    }
    for (size_t index = 16; index < 30; ++index) {
        radii[index] = 0.1 + 0.9 * (index - 15.0) / 14.0;
    }
    return radii;
}

// Deterministic Poisson sample from the raw engine mantissa, so the sequence is
// identical on every standard library. A negative concentration selects a
// uniform sphere, zero selects rho ~ 1/r, and a positive value selects NFW.
std::vector<double> sampledRadii(double concentration, size_t count, std::mt19937_64& engine) {
    std::vector<double> radii(count);
    for (auto& radius : radii) {
        double const massFraction = double(engine() >> 11) * 0x1.0p-53;
        if (concentration > 0.0) {
            radius = nfwQuantile(concentration, massFraction);
        } else if (concentration == 0.0) {
            radius = std::sqrt(massFraction);
        } else {
            radius = std::cbrt(massFraction);
        }
    }
    std::sort(radii.begin(), radii.end());
    return radii;
}

void testNfwScaleRadiusWindow() {
    // Catalogue radii derive from binary32 positions; the synthetic plateau halo
    // must report an unresolved fit at every outer radius.
    for (double const outerRadius : {1.0, 0.12, 0.1, 1.e-3}) {
        std::vector<double> radii;
        for (double const radius : productionFailureRadii()) {
            radii.push_back(double(float(radius * outerRadius)));
        }
        std::sort(radii.begin(), radii.end());
        auto const fit = rockstar::nfwFit(radii);
        require(fit.scaleRadius == 0.0 && fit.density == 0.0,
            "The production two-bin likelihood plateau must report an unresolved fit");
    }
    // rho ~ 1/r is the c -> 0 limit itself and must stay unresolved at every count.
    for (size_t const count : {30, 45, 100, 300, 1000, 5000}) {
        require(rockstar::nfwFit(inverseRadiusQuantiles(count)).scaleRadius == 0.0,
            "The rho ~ 1/r limit must not resolve a scale radius at any particle count");
    }
    // Two equal-mass bins determine the concentration exactly. Optima at or below
    // the c = 1 edge, and interior optima within one sigma of it, are unresolved;
    // decisive optima are recovered exactly. The rises at c* = 2.25 (0.47) and
    // c* = 2.35 (0.53) pin kMinBoundaryLogLikelihoodRise = 0.5 from both sides.
    for (double const concentration : {0.5, 0.9, 0.999, 1.0000001, 1.5, 2.0, 2.25}) {
        require(rockstar::nfwFit(twoBinRadii(concentration)).scaleRadius == 0.0,
            "A two-bin optimum within one sigma of the c = 1 edge must be unresolved");
    }
    for (double const concentration : {2.35, 3.0, 5.0, 20.0, 9000.0}) {
        auto const fit = rockstar::nfwFit(twoBinRadii(concentration));
        require(fit.scaleRadius > 0.0 && fit.scaleRadius < 1.0,
            "A decisive two-bin median must resolve a scale radius inside the profile");
        requireClose(1.0 / fit.scaleRadius / concentration, 1.0, 1.e-4,
            "Two-bin fits inside the window must still be exactly determined");
    }
    require(rockstar::nfwFit(centralCuspRadii()).scaleRadius == 0.0,
        "An optimum pinned at the upper window edge must be unresolved");
    // Ideal halos: the likelihood rise at the c = 1 edge grows with N and c, so
    // concentrated small halos stay resolved and recover their concentration ...
    for (double const concentration : {3.0, 5.0, 10.0}) {
        auto const fit = rockstar::nfwFit(nfwQuantileRadii(concentration, 30));
        require(fit.scaleRadius > 0.0 && fit.density > 0.0,
            "Exact NFW quantiles of a concentrated small halo must remain resolved");
        requireClose(1.0 / fit.scaleRadius / concentration, 1.0, 0.005,
            "A resolved small halo must recover its concentration");
    }
    for (double const concentration : {2.0, 3.0, 5.0, 10.0}) {
        auto const fit = rockstar::nfwFit(nfwQuantileRadii(concentration, 150));
        require(fit.scaleRadius > 0.0 && fit.density > 0.0,
            "Exact NFW quantiles of a 150-particle halo must remain resolved");
        requireClose(1.0 / fit.scaleRadius / concentration, 1.0, 0.005,
            "A resolved 150-particle halo must recover its concentration");
    }
    auto const nearEdge = rockstar::nfwFit(nfwQuantileRadii(1.5, 5000));
    requireClose(1.0 / nearEdge.scaleRadius / 1.5, 1.0, 0.005,
        "A well-sampled profile close to the c = 1 edge must still be recovered");
    // ... while profiles consistent with c = 1 at one sigma are unresolved.
    require(rockstar::nfwFit(nfwQuantileRadii(2.0, 30)).scaleRadius == 0.0,
        "A c = 2 halo with thirty particles cannot exclude c = 1 at one sigma");
    require(rockstar::nfwFit(nfwQuantileRadii(1.5, 60)).scaleRadius == 0.0,
        "A c = 1.5 halo with sixty particles cannot exclude c = 1 at one sigma");
    require(rockstar::nfwFit(nfwQuantileRadii(1.0, 5000)).scaleRadius == 0.0,
        "A scale radius at the outer radius is unresolved by definition");
    // Structural invariants on deterministic Poisson samples of uniform, rho ~ 1/r
    // and NFW profiles: a resolved scale radius never leaves the fitted profile,
    // and the scale radius and density are resolved together.
    std::mt19937_64 engine(20261028);
    size_t resolved = 0;
    for (double const concentration : {-1.0, 0.0, 1.0, 3.0, 10.0}) {
        for (size_t const count : {30, 45, 100}) {
            for (int trial = 0; trial < 100; ++trial) {
                double const outerRadius = 1.e-3 + 0.5 * double(engine() >> 11) * 0x1.0p-53;
                auto radii = sampledRadii(concentration, count, engine);
                for (double& radius : radii) {
                    radius *= outerRadius;
                }
                auto const fit = rockstar::nfwFit(radii);
                require(fit.scaleRadius >= 0.0 && fit.scaleRadius <= radii.back(),
                    "A fitted scale radius must never exceed the outermost fitted radius");
                require((fit.scaleRadius > 0.0) == (fit.density > 0.0),
                    "Scale radius and density must be resolved together");
                resolved += fit.scaleRadius > 0.0;
            }
        }
    }
    require(resolved > 0, "Sampled profiles must include resolved fits");
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

TEST(RockstarProperties, NfwScaleRadiusWindow) {
    detail::testNfwScaleRadiusWindow();
}

TEST(RockstarProperties, KlypinFit) {
    detail::testKlypinFit();
}

TEST(RockstarProperties, HaloDensityConventionPreservesSupportedFamilies) {
    using rockstar::HaloDensityConvention;
    EXPECT_EQ(rockstar::haloDensityConvention(0.7, 0.0), HaloDensityConvention::bryanNormanFlat);
    EXPECT_EQ(rockstar::haloDensityConvention(0.0, 0.0), HaloDensityConvention::bryanNormanFlat);
    EXPECT_EQ(rockstar::haloDensityConvention(0.0, 0.7), HaloDensityConvention::bryanNormanOpen);
    EXPECT_EQ(rockstar::haloDensityConvention(0.0, -1.0), HaloDensityConvention::meanMatter);
    EXPECT_EQ(rockstar::haloDensityConvention(-1.0, 0.0), HaloDensityConvention::meanMatter);
    EXPECT_EQ(rockstar::haloDensityConvention(0.6, 0.1), HaloDensityConvention::meanMatter);
    EXPECT_EQ(rockstar::haloDensityConvention(0.8, -0.1), HaloDensityConvention::meanMatter);
}

TEST(RockstarProperties, SupportedHaloThresholdsRetainCriticalReferenceAndVirialFit) {
    constexpr double kMatter = 0.3;
    for (double const radiation : {0.0, 9.2e-5}) {
        for (bool const open : {false, true}) {
            double const lambda = open ? 0.0 : 1.0 - kMatter - radiation;
            double const curvature = open ? 1.0 - kMatter - radiation : 0.0;
            for (double const scale : {0.02, 0.5, 1.0, 10.0}) {
                double const criticalToMean = 1.0 + radiation / (kMatter * scale) +
                    curvature * scale / kMatter + lambda * scale * scale * scale / kMatter;
                double const offset = 1.0 / criticalToMean - 1.0;
                double const virialCritical = 18.0 * std::numbers::pi * std::numbers::pi +
                    (open ? 60.0 : 82.0) * offset - (open ? 32.0 : 39.0) * offset * offset;
                auto const thresholds =
                    rockstar::haloDensityThresholds(scale, kMatter, radiation, lambda, curvature);
                EXPECT_NEAR(thresholds.soReferenceMeanMatter / criticalToMean, 1.0, 2.e-14);
                EXPECT_NEAR(
                    thresholds.virialMeanMatter / (virialCritical * criticalToMean), 1.0, 2.e-14);
            }
        }
    }
}

TEST(RockstarProperties, MeanReferenceRemainsFiniteAtClosedAndNegativeLambdaTurnaround) {
    std::vector<double> const radii{0.0, 0.0001, 0.0002};
    constexpr double kMeanParticleDensity = 1.e6;
    for (double const lambda : {0.0, -1.0}) {
        double const curvature = lambda == 0.0 ? -1.0 : 0.0;
        double const turnaround = lambda == 0.0 ? 2.0 : std::cbrt(2.0);
        for (double const scale : {0.02, 0.99 * turnaround, turnaround,
                 std::nextafter(turnaround, std::numeric_limits<double>::infinity())}) {
            // The same scale occurs on expansion and contraction. This policy
            // uses a fixed catalogue definition, with no division by H(a)^2.
            auto const thresholds =
                rockstar::haloDensityThresholds(scale, 2.0, 0.0, lambda, curvature);
            EXPECT_DOUBLE_EQ(thresholds.virialMeanMatter, 200.0);
            EXPECT_DOUBLE_EQ(thresholds.soReferenceMeanMatter, 1.0);
            auto const primary = rockstar::sphericalOverdensity(
                radii, kMeanParticleDensity * thresholds.virialMeanMatter);
            EXPECT_EQ(primary.count, size_t(3));
            EXPECT_GT(primary.radius, 0.0);
            double previousRadius = primary.radius;
            for (double const multiplier : {200.0, 500.0, 2500.0}) {
                auto const secondary = rockstar::sphericalOverdensity(
                    radii, kMeanParticleDensity * multiplier * thresholds.soReferenceMeanMatter);
                double const expectedRadius = std::cbrt(
                    3.0 / (4.0 * std::numbers::pi / 3.0 * kMeanParticleDensity * multiplier));
                EXPECT_EQ(secondary.count, size_t(3));
                EXPECT_NEAR(secondary.radius / expectedRadius, 1.0, 2.e-14);
                EXPECT_LE(secondary.radius, previousRadius);
                previousRadius = secondary.radius;
            }
        }
    }
}

TEST(RockstarProperties, CurvedLambdaCataloguesKeepOneMeanDensityConventionAtEveryEpoch) {
    constexpr double kRadiation = 9.2e-5;
    constexpr double kMatter = 0.3;
    for (double const curvature : {-0.1, 0.1}) {
        double const lambda = 1.0 - kMatter - kRadiation - curvature;
        for (double const scale : {0.02, 1.0, 10.0}) {
            auto const thresholds =
                rockstar::haloDensityThresholds(scale, kMatter, kRadiation, lambda, curvature);
            EXPECT_DOUBLE_EQ(thresholds.virialMeanMatter, 200.0);
            EXPECT_DOUBLE_EQ(thresholds.soReferenceMeanMatter, 1.0);
        }
    }
}

TEST(RockstarProperties, InvalidHaloThresholdInputsAreRejectedBeforeFallback) {
    double const infinity = std::numeric_limits<double>::infinity();
    double const nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(rockstar::haloDensityThresholds(0.0, 2.0, 0.0, 0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(rockstar::haloDensityThresholds(1.0, 0.0, 0.0, 0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(rockstar::haloDensityThresholds(1.0, 2.0, -0.1, 0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(rockstar::haloDensityThresholds(nan, 2.0, 0.0, 0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(
        rockstar::haloDensityThresholds(1.0, infinity, 0.0, 0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(
        rockstar::haloDensityThresholds(1.0, 2.0, infinity, 0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(rockstar::haloDensityThresholds(1.0, 2.0, 0.0, nan, -1.0), std::invalid_argument);
    EXPECT_THROW(
        rockstar::haloDensityThresholds(1.0, 2.0, 0.0, -1.0, infinity), std::invalid_argument);
}
