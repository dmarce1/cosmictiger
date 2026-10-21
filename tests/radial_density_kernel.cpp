#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>

#include <gtest/gtest.h>

#include <cosmictiger/math/radial_density_kernel.hpp>
#if defined(RADIAL_KERNEL_TEST_SIMD)
#include <cosmictiger/math/simd.hpp>
#endif

namespace detail {

using radial_kernel::Family;
using radial_kernel::KernelSpec;
using std::size_t;

constexpr auto kDimensionOne = radial_kernel::makeDensityKernelTable<1>({Family::Wendland, 7});
constexpr auto kDimensionTwenty = radial_kernel::makeDensityKernelTable<20>({Family::Spline, 9});
static_assert(kDimensionOne.pieceCount == 1 && kDimensionOne.degree == 7);
static_assert(
    kDimensionTwenty.pieceCount == 5 && kDimensionTwenty.supportRadiusPerEffectiveRadius > 0);
static_assert(radial_kernel::isValidDensity<4>({Family::Wendland, 9}));
static_assert(!radial_kernel::isValidDensity<20>({Family::Wendland, 9}));
static_assert(radial_kernel::makeDensityKernelTable<20>({Family::Wendland, 9}).pieceCount == 0);

void close(double actual, double expected, double tolerance, char const* label) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        std::fprintf(stderr, "%s: %.17g != %.17g\n", label, actual, expected);
        throw std::runtime_error(label);
    }
}

template <size_t dimension, typename Function>
double integrate(radial_kernel::DensityKernelTable<dimension> const& table, Function function) {
    // Degree <= 9 with D <= 20 gives moment integrands of degree <= 30;
    // sixteen-point Gauss quadrature on each span integrates them exactly.
    constexpr double kNodes[8] = {0.09501250983763744019, 0.28160355077925891323,
        0.45801677765722738634, 0.61787624440264374845, 0.75540440835500303390,
        0.86563120238783174388, 0.94457502307323257608, 0.98940093499164993260};
    constexpr double kWeights[8] = {0.18945061045506849629, 0.18260341504492358887,
        0.16915651939500253819, 0.14959598881657673208, 0.12462897125553387205,
        0.09515851168249278481, 0.06225352393864789286, 0.02715245941175409485};
    double result = 0;
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        double const midpoint = (table.changePoints[piece] + table.changePoints[piece + 1]) / 2;
        double const radius = (table.changePoints[piece + 1] - table.changePoints[piece]) / 2;
        for (int i = 0; i < 8; ++i) {
            result += radius * kWeights[i] *
                (function(midpoint - radius * kNodes[i]) + function(midpoint + radius * kNodes[i]));
        }
    }
    return result;
}

template <size_t dimension>
long double wendlandReference(int degree, long double q) {
    int const k = (degree - int(dimension / 2 + 1)) / 3;
    int p = int(dimension / 2) + k + 1;
    long double coefficients[10]{};
    long double binomial = 1;
    for (int i = 0; i <= p; ++i) {
        coefficients[i] = (i % 2 == 0 ? 1 : -1) * binomial;
        binomial *= static_cast<long double>(p - i) / (i + 1);
    }
    // Independent power-basis integration is only a test reference. Production
    // generates positive Bernstein coefficients without alternating sums.
    for (int iteration = 0; iteration < k; ++iteration) {
        long double next[10]{};
        for (int i = 0; i <= p; ++i) {
            next[0] += coefficients[i] / (i + 2);
            next[i + 2] = -coefficients[i] / (i + 2);
        }
        p += 2;
        for (int i = 0; i <= p; ++i) {
            coefficients[i] = next[i];
        }
    }
    long double value = coefficients[p];
    for (int i = p - 1; i >= 0; --i) {
        value = value * q + coefficients[i];
    }
    return value;
}

template <size_t dimension>
void test(KernelSpec spec) {
    auto const table = radial_kernel::makeDensityKernelTable<dimension>(spec);
    if (table.pieceCount == 0) {
        throw std::runtime_error("valid specification produced an empty table");
    }
    double const area =
        2 * std::pow(std::numbers::pi, dimension / 2.0) / std::tgamma(dimension / 2.0);
    auto density = [&](double q) { return radial_kernel::value(table, q); };
    double const mass =
        integrate(table, [&](double q) { return area * std::pow(q, dimension - 1) * density(q); });
    double const variance =
        integrate(table, [&](double q) { return area * std::pow(q, dimension + 1) * density(q); });
    close(mass, 1, 8e-14, "D-dimensional normalization");
    close(variance,
        dimension /
            (4 * table.supportRadiusPerEffectiveRadius * table.supportRadiusPerEffectiveRadius),
        8e-14, "D-dimensional width");
    auto const base = radial_kernel::makeKernelTable(spec);
    for (int i = 0; i <= 100; ++i) {
        double const q = double(i) / 100;
        if (density(q) < 0) {
            throw std::runtime_error("negative density");
        }
        close(radial_kernel::value(table, float(q)), density(float(q)), 8e-6,
            "D-dimensional float density");
        close(radial_kernel::derivative(table, float(q)),
            radial_kernel::derivative(table, double(float(q))), 8e-6,
            "D-dimensional float derivative");
        if (spec.family == Family::Wendland && q <= 0.8) {
            double const expected = density(0) *
                double(wendlandReference<dimension>(spec.degree, q) /
                    wendlandReference<dimension>(spec.degree, 0));
            close(density(q), expected, 2e-9, "dimension-specific Wendland shape");
        }
        if (spec.family == Family::Spline) {
            close(density(q) / density(0),
                radial_kernel::value(base, q) / radial_kernel::value(base, 0.0), 2e-14,
                "dimension-independent spline shape");
        }
        if (spec.family == Family::Ferrers) {
            double const m = spec.degree / 2;
            double const normalizer = std::tgamma(m + dimension / 2.0 + 1) /
                (std::pow(std::numbers::pi, dimension / 2.0) * std::tgamma(m + 1));
            close(density(q), normalizer * std::pow(1 - q * q, m), 8e-14,
                "analytic Ferrers normalization");
            close(table.supportRadiusPerEffectiveRadius,
                std::sqrt(dimension + spec.degree + 2.0) / 2, 4e-15, "analytic Ferrers width");
        }
        if constexpr (dimension == 3) {
            close(density(q), radial_kernel::value(base, q), 3e-14, "production 3D density parity");
            close(radial_kernel::derivative(table, q), radial_kernel::derivative(base, q), 3e-14,
                "production 3D derivative parity");
            close(table.supportRadiusPerEffectiveRadius, base.supportRadiusPerEffectiveRadius,
                4e-15, "production 3D width parity");
        }
        if (i > 0 && i < 100) {
            double const step = 1e-6;
            close(radial_kernel::derivative(table, q),
                (density(q + step) - density(q - step)) / (2 * step), 2e-7,
                "D-dimensional derivative");
        }
    }
    // Near the cutoff, the quadratic derivative vanishes only linearly.
    // Computing both span weights directly preserves these small values even
    // when high-dimensional normalization makes absolute errors visible.
    if (spec.family == Family::Spline && spec.degree == 2) {
        float const q = std::nextafter(1.0f, 0.0f);
        double const centralDensity = density(0);
        double const expectedDensity = 1.5 * centralDensity * (1 - double(q)) * (1 - double(q));
        double const expectedDerivative = -3 * centralDensity * (1 - double(q));
        close(radial_kernel::value(table, q) / expectedDensity, 1, 8e-6,
            "quadratic float cutoff density");
        close(radial_kernel::derivative(table, q) / expectedDerivative, 1, 8e-6,
            "quadratic float cutoff derivative");
    }
    close(density(1.5), 0, 0, "compact support");
#if defined(RADIAL_KERNEL_TEST_SIMD)
    SimdFloat radii{};
    for (int lane = 0; lane < SIMD_FLOAT_SIZE; ++lane) {
        simdSetLane(radii, lane, 2.0f * lane / SIMD_FLOAT_SIZE);
    }
    simdSetLane(radii, 1, 1e-30f);
    simdSetLane(radii, 2, std::nextafter(1.0f, 0.0f));
    auto const values = radial_kernel::value(table, radii);
    auto const derivatives = radial_kernel::derivative(table, radii);
    for (int lane = 0; lane < SIMD_FLOAT_SIZE; ++lane) {
        close(values[lane], density(radii[lane]), 8e-6, "D-dimensional SIMD density");
        close(derivatives[lane], radial_kernel::derivative(table, double(radii[lane])), 8e-6,
            "D-dimensional SIMD derivative");
    }
#endif
}

template <size_t dimension>
void testDimension() {
    for (int degree = 2; degree <= 9; ++degree) {
        test<dimension>({Family::Spline, degree});
    }
    for (int degree = 2; degree <= 8; degree += 2) {
        test<dimension>({Family::Ferrers, degree});
    }
    for (int degree = 1; degree <= 9; ++degree) {
        if (radial_kernel::isValidDensity<dimension>({Family::Wendland, degree})) {
            test<dimension>({Family::Wendland, degree});
        }
    }
}

}    // namespace detail

TEST(RadialDensityKernel, Dimension1) {
    detail::testDimension<1>();
}

TEST(RadialDensityKernel, Dimension2) {
    detail::testDimension<2>();
}

TEST(RadialDensityKernel, Dimension3) {
    detail::testDimension<3>();
}

TEST(RadialDensityKernel, Dimension4) {
    detail::testDimension<4>();
}

TEST(RadialDensityKernel, Dimension5) {
    detail::testDimension<5>();
}

TEST(RadialDensityKernel, Dimension6) {
    detail::testDimension<6>();
}

TEST(RadialDensityKernel, Dimension7) {
    detail::testDimension<7>();
}

TEST(RadialDensityKernel, Dimension20) {
    detail::testDimension<20>();
}
