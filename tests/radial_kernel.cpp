#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include <cosmictiger/math/radial_kernel.hpp>
#if defined(RADIAL_KERNEL_TEST_SIMD)
#include <cosmictiger/math/simd.hpp>
#endif

namespace detail {

using radial_kernel::Family;
using radial_kernel::KernelSpec;
using radial_kernel::KernelTable;

constexpr std::array<KernelSpec, 15> kSpecifications = {
    {{Family::Spline, 2}, {Family::Spline, 3}, {Family::Spline, 4}, {Family::Spline, 5},
        {Family::Spline, 6}, {Family::Spline, 7}, {Family::Spline, 8}, {Family::Spline, 9},
        {Family::Wendland, 2}, {Family::Wendland, 5}, {Family::Wendland, 8}, {Family::Ferrers, 2},
        {Family::Ferrers, 4}, {Family::Ferrers, 6}, {Family::Ferrers, 8}}};

constexpr auto kQuadratic = radial_kernel::makeKernelTable({Family::Spline, 2});
constexpr auto kCubic = radial_kernel::makeKernelTable({Family::Spline, 3});
constexpr auto kFerrers = radial_kernel::makeKernelTable({Family::Ferrers, 8});
static_assert(kQuadratic.degree == 2 && kQuadratic.pieceCount == 2);
static_assert(kQuadratic.changePoints[1] == 1.0 / 3 && kQuadratic.changePoints[2] == 1);
static_assert(kQuadratic.centralPotential == 39.0 / 16);
static_assert(kCubic.pieceCount == 2 && kCubic.changePoints[1] == 0.5);
static_assert(kFerrers.centralPotential == 693.0 / 256);
static_assert(std::is_trivially_copyable_v<KernelTable>);
static_assert(std::is_standard_layout_v<KernelTable>);
static_assert(!radial_kernel::isValid({Family::Spline, 1}));
static_assert(!radial_kernel::isValid({Family::Wendland, 9}));
static_assert(!radial_kernel::isValid({Family::Ferrers, 3}));
static_assert(radial_kernel::makeKernelTable({Family::Ferrers, 9}).pieceCount == 0);

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

template <typename Function>
double integrate(KernelTable const& table, double lower, double upper, Function function) {
    // Eight-point Gauss integration is exact for every polynomial moment used
    // here when split at the kernel's change points (degree at most thirteen).
    constexpr double kNodes[4] = {0.18343464249564980494, 0.52553240991632898582,
        0.79666647741362673959, 0.96028985649753623168};
    constexpr double kWeights[4] = {0.36268378337836198297, 0.31370664587788728734,
        0.22238103445337447054, 0.10122853629037625915};
    double result = 0;
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        double const a = std::max(lower, table.changePoints[piece]);
        double const b = std::min(upper, table.changePoints[piece + 1]);
        if (b <= a) {
            continue;
        }
        double const center = (a + b) / 2;
        double const radius = (b - a) / 2;
        for (int i = 0; i < 4; ++i) {
            result += radius * kWeights[i] *
                (function(center - radius * kNodes[i]) + function(center + radius * kNodes[i]));
        }
    }
    return result;
}

long double referenceShape(KernelSpec spec, long double q) {
    if (spec.family == Family::Ferrers) {
        return std::pow(1 - q * q, spec.degree / 2);
    }
    if (spec.family == Family::Wendland) {
        if (spec.degree == 2) {
            return (1 - q) * (1 - q);
        }
        if (spec.degree == 5) {
            return std::pow(1 - q, 4) * (1 + 4 * q);
        }
        return std::pow(1 - q, 6) * (1 + 6 * q + (35.0L / 3) * q * q);
    }
    // Independent truncated-power cardinal spline formula. Long double limits
    // cancellation at high order; this reference is only used away from q=1.
    int const order = spec.degree + 1;
    long double sum = 0;
    long double binomial = 1;
    for (int i = 0; i <= order; ++i) {
        long double const argument = order * (q + 1) / 2 - i;
        if (argument > 0) {
            sum += (i % 2 == 0 ? 1 : -1) * binomial * std::pow(argument, spec.degree);
        }
        binomial *= static_cast<long double>(order - i) / (i + 1);
    }
    return sum;
}

void testTable(KernelSpec spec) {
    auto const table = radial_kernel::makeKernelTable(spec);
    double const fourPi = 4 * std::numbers::pi;
    auto density = [&](double q) { return radial_kernel::value(table, q); };
    double const mass =
        integrate(table, 0, 1, [&](double q) { return fourPi * q * q * density(q); });
    double const secondMoment =
        integrate(table, 0, 1, [&](double q) { return fourPi * q * q * q * q * density(q); });
    close(mass, 1, 4e-14, "unit mass");
    close(secondMoment,
        3 / (4 * table.supportRadiusPerEffectiveRadius * table.supportRadiusPerEffectiveRadius),
        4e-14, "variance width");
    close(radial_kernel::forceFactor(table, 0.0), fourPi * density(0) / 3, 4e-14,
        "central harmonic force");
    close(radial_kernel::potential(table, 0.0),
        integrate(table, 0, 1, [&](double q) { return fourPi * q * density(q); }), 4e-14,
        "central potential");
    for (int i = 0; i <= 80; ++i) {
        double const q = static_cast<double>(i) / 100;
        double const expected =
            density(0) * static_cast<double>(referenceShape(spec, q) / referenceShape(spec, 0));
        close(density(q), expected, 2e-10, "independent polynomial shape");
    }
    for (int i = 1; i < 100; ++i) {
        double const q = static_cast<double>(i) / 100;
        double const enclosedMass =
            integrate(table, 0, q, [&](double r) { return fourPi * r * r * density(r); });
        double const shells =
            integrate(table, q, 1, [&](double r) { return fourPi * r * density(r); });
        close(radial_kernel::forceFactor(table, q), enclosedMass / (q * q * q), 8e-14,
            "enclosed mass force");
        close(radial_kernel::potential(table, q), enclosedMass / q + shells, 8e-14,
            "shell theorem potential");
        double const step = 1e-6;
        double const densityDerivative = (density(q + step) - density(q - step)) / (2 * step);
        close(radial_kernel::derivative(table, q), densityDerivative, 2e-8, "density derivative");
        double const potentialDerivative = (radial_kernel::potential(table, q + step) -
                                               radial_kernel::potential(table, q - step)) /
            (2 * step);
        close(potentialDerivative, -q * radial_kernel::forceFactor(table, q), 2e-8,
            "potential force derivative");
        double const forceDerivative = (radial_kernel::forceFactor(table, q + step) -
                                           radial_kernel::forceFactor(table, q - step)) /
            (2 * step);
        close(3 * radial_kernel::forceFactor(table, q) + q * forceDerivative, fourPi * density(q),
            2e-7, "Poisson equation");
    }
    for (int i = 0; i <= 2000; ++i) {
        float const q = static_cast<float>(i) / 1000;
        float const value = radial_kernel::value(table, q);
        require(value >= 0 && std::isfinite(value), "nonnegative float density");
        close(value, density(q), 4e-6, "float density accuracy");
        close(radial_kernel::derivative(table, q), radial_kernel::derivative(table, double(q)),
            5e-6, "float derivative accuracy");
        close(radial_kernel::forceFactor(table, q), radial_kernel::forceFactor(table, double(q)),
            5e-6, "float force accuracy");
        close(radial_kernel::potential(table, q), radial_kernel::potential(table, double(q)), 5e-6,
            "float potential accuracy");
    }
    for (int exponent = 0; exponent <= 40; ++exponent) {
        float const q = static_cast<float>(std::pow(10.0, -exponent));
        close(radial_kernel::forceFactor(table, q), radial_kernel::forceFactor(table, double(q)),
            5e-6, "near-origin float force");
        close(radial_kernel::potential(table, q), radial_kernel::potential(table, double(q)), 5e-6,
            "near-origin float potential");
    }
    for (int piece = 1; piece <= table.pieceCount; ++piece) {
        double const q = table.changePoints[piece];
        double const left = std::nextafter(q, 0.0);
        close(density(left), density(q), 2e-13, "density continuity");
        close(radial_kernel::forceFactor(table, left), radial_kernel::forceFactor(table, q), 2e-13,
            "force continuity");
        close(radial_kernel::potential(table, left), radial_kernel::potential(table, q), 2e-13,
            "potential continuity");
        if (piece < table.pieceCount) {
            close(radial_kernel::derivative(table, left), radial_kernel::derivative(table, q),
                2e-13, "interior derivative continuity");
        }
    }
    for (double const q : {1.0, 1.1, 2.0, 1e20}) {
        require(density(q) == 0 && radial_kernel::derivative(table, q) == 0, "compact support");
        close(radial_kernel::forceFactor(table, q), 1 / (q * q * q), 4e-15,
            "Newtonian exterior force");
        close(radial_kernel::potential(table, q), 1 / q, 4e-15, "Newtonian exterior potential");
    }
    if (spec.family == Family::Ferrers) {
        close(table.supportRadiusPerEffectiveRadius, std::sqrt(spec.degree + 5.0) / 2, 4e-15,
            "Ferrers width conversion");
    }
#if defined(RADIAL_KERNEL_TEST_SIMD)
    for (int sample = 0; sample < 32; ++sample) {
        SimdFloat radius{};
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; ++lane) {
            simdSetLane(
                radius, lane, static_cast<float>((sample * SIMD_FLOAT_SIZE + lane) % 31) / 15);
        }
        if (sample == 0) {
            simdSetLane(radius, 0, 0);
            simdSetLane(radius, 1, 1e-30f);
            simdSetLane(radius, 2, 1e30f);
        }
        auto const values = radial_kernel::value(table, radius);
        auto const derivatives = radial_kernel::derivative(table, radius);
        auto const forces = radial_kernel::forceFactor(table, radius);
        auto const potentials = radial_kernel::potential(table, radius);
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; ++lane) {
            float const scalarRadius = radius[lane];
            close(values[lane], radial_kernel::value(table, scalarRadius), 5e-6, "SIMD density");
            close(derivatives[lane], radial_kernel::derivative(table, scalarRadius), 5e-6,
                "SIMD derivative");
            close(
                forces[lane], radial_kernel::forceFactor(table, scalarRadius), 5e-6, "SIMD force");
            close(potentials[lane], radial_kernel::potential(table, scalarRadius), 5e-6,
                "SIMD potential");
        }
    }
#endif
}

// Closed forms for the normalized quadratic (M3) spline and its spherical
// gravity integrals, independent of the Bernstein table construction.
void testQuadraticSpline() {
    close(kQuadratic.supportRadiusPerEffectiveRadius, std::sqrt(135.0 / 52), 4e-15,
        "quadratic width conversion");
    double const normalization = 81.0 / (16 * std::numbers::pi);
    for (double const q : {0.0, 1e-30, 0.1, std::nextafter(1.0 / 3, 0.0), 1.0 / 3,
             std::nextafter(1.0 / 3, 1.0), 0.5, 0.9, std::nextafter(1.0, 0.0), 1.0, 2.0}) {
        double const x = q * q;
        double density = 0;
        double derivative = 0;
        double force;
        double potential;
        if (q < 1.0 / 3) {
            density = normalization * (1 - 3 * x);
            derivative = -6 * normalization * q;
            force = 27.0 / 4 - 243.0 / 20 * x;
            potential = 39.0 / 16 - 27.0 / 8 * x + 243.0 / 80 * x * x;
        } else if (q < 1) {
            density = 1.5 * normalization * (1 - q) * (1 - q);
            derivative = -3 * normalization * (1 - q);
            force = 81.0 / 8 - 243.0 / 16 * q + 243.0 / 40 * x - 1 / (80 * q * x);
            potential =
                81.0 / 32 - 81.0 / 16 * x + 81.0 / 16 * x * q - 243.0 / 160 * x * x - 1 / (80 * q);
        } else {
            force = 1 / (q * x);
            potential = 1 / q;
        }
        close(radial_kernel::value(kQuadratic, q), density, 2e-14, "quadratic density");
        close(radial_kernel::derivative(kQuadratic, q), derivative, 2e-14, "quadratic derivative");
        close(radial_kernel::forceFactor(kQuadratic, q), force, 2e-14, "quadratic force");
        close(radial_kernel::potential(kQuadratic, q), potential, 2e-14, "quadratic potential");
    }
}

void testLegacyKernels() {
    close(kCubic.supportRadiusPerEffectiveRadius, std::sqrt(10.0 / 3), 4e-15,
        "cubic width conversion");
    close(radial_kernel::makeKernelTable({Family::Wendland, 5}).supportRadiusPerEffectiveRadius,
        std::sqrt(15.0) / 2, 4e-15, "Wendland C2 width");
    close(radial_kernel::makeKernelTable({Family::Wendland, 8}).supportRadiusPerEffectiveRadius,
        std::sqrt(39.0 / 8), 4e-15, "Wendland C4 width");
    for (int i = 0; i <= 100; ++i) {
        double const q = static_cast<double>(i) / 100;
        double const cubic = q < 0.5 ? 8 / std::numbers::pi * (1 - 6 * q * q + 6 * q * q * q) :
                                       16 / std::numbers::pi * std::pow(1 - q, 3);
        close(radial_kernel::value(kCubic, q), cubic, 2e-14, "legacy cubic density");
        double const x = q * q;
        double const force = 1155.0 / 128 - 693.0 / 32 * x + 1485.0 / 64 * x * x -
            385.0 / 32 * x * x * x + 315.0 / 128 * x * x * x * x;
        double const potential = 693.0 / 256 - 1155.0 / 256 * x + 693.0 / 128 * x * x -
            495.0 / 128 * x * x * x + 385.0 / 256 * x * x * x * x - 63.0 / 256 * x * x * x * x * x;
        close(radial_kernel::forceFactor(kFerrers, q), force, 2e-14, "legacy Ferrers F4 force");
        close(
            radial_kernel::potential(kFerrers, q), potential, 2e-14, "legacy Ferrers F4 potential");
    }
}

}    // namespace detail

TEST(RadialKernel, QuadraticSpline) {
    detail::testQuadraticSpline();
}

TEST(RadialKernel, LegacyKernels) {
    detail::testLegacyKernels();
}

TEST(RadialKernel, AllTables) {
    for (auto const spec : detail::kSpecifications) {
        detail::testTable(spec);
    }
}
