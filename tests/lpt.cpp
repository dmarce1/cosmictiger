#include <array>
#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include <cosmictiger/cosmology/lpt.hpp>

namespace detail {

using Hessian = std::array<std::array<double, 3>, 3>;

double secondOrderSource(Hessian const& hessian) {
    std::array<std::array<int, 2>, 9> const dimensions = {
        {{1, 1}, {2, 2}, {0, 0}, {2, 2}, {0, 0}, {1, 1}, {0, 1}, {0, 2}, {1, 2}}};
    double previous = 0.0;
    double source = 0.0;
    for (int phase = 0; phase < int(dimensions.size()); ++phase) {
        auto const& indices = dimensions[phase];
        lpt::accumulateSecondOrderSource(phase, hessian[indices[0]][indices[1]], previous, source);
    }
    return source;
}

double cloudCorrection(double normalizedWavenumber) {
    if (normalizedWavenumber == 0.0) {
        return 1.0;
    }
    double const argument = 0.5 * normalizedWavenumber;
    return std::pow(argument / std::sin(argument), 6);
}

Hessian modeHessian(std::array<double, 3> const& wavevector, double densityAmplitude) {
    double squaredWavenumber = 0.0;
    double correction = 1.0;
    for (double const component : wavevector) {
        squaredWavenumber += component * component;
        correction *= cloudCorrection(component);
    }
    Hessian hessian{};
    for (int first = 0; first < 3; ++first) {
        for (int second = 0; second < 3; ++second) {
            double const derivative =
                -densityAmplitude * wavevector[first] * wavevector[second] / squaredWavenumber;
            hessian[first][second] = lpt::interpolationCorrectedDerivative(derivative, first,
                second, [correction](double value) { return value * correction; });
        }
    }
    return hessian;
}

}    // namespace detail

TEST(Lpt, SinglePlaneWaveHasNoSecondOrderSource) {
    // A plane wave has a rank-one Hessian, so every principal 2x2 minor vanishes.
    // The three planar directions ensure that none of xy, xz or yz is omitted.
    std::array<std::array<double, 3>, 4> const directions = {
        {{1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, -2.0, 3.0}}};
    for (auto const& direction : directions) {
        for (double const amplitude : {-0.7, 0.0, 0.2, 1.0}) {
            auto const hessian = detail::modeHessian(direction, amplitude);
            EXPECT_NEAR(detail::secondOrderSource(hessian), 0.0, 1.0e-14);
        }
    }
}

TEST(Lpt, DiagonalHessianRetainsAllCrossProducts) {
    detail::Hessian const hessian = {{{2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 5.0}}};
    EXPECT_DOUBLE_EQ(detail::secondOrderSource(hessian), 31.0);
}

TEST(Lpt, PerpendicularModeSourceIsIndependentOfParticleInterpolation) {
    // For two perpendicular modes the analytic source is delta_x delta_y.
    // Large, unequal cloud corrections expose deconvolution applied before the
    // grid-space Hessian product, even though a final particle interpolation
    // would correctly cancel the correction on a displacement field.
    for (double const fraction : {0.2, 0.6, 0.9}) {
        double const firstWavenumber = fraction * std::numbers::pi;
        double const secondWavenumber = 0.8 * firstWavenumber;
        for (int point = 0; point < 11; ++point) {
            double const firstDensity = 0.13 * std::cos(0.37 * point);
            double const secondDensity = 0.21 * std::cos(0.59 * point + 0.2);
            auto hessian = detail::modeHessian({firstWavenumber, 0.0, 0.0}, firstDensity);
            auto const other = detail::modeHessian({0.0, secondWavenumber, 0.0}, secondDensity);
            for (int first = 0; first < 3; ++first) {
                for (int second = 0; second < 3; ++second) {
                    hessian[first][second] += other[first][second];
                }
            }
            EXPECT_NEAR(detail::secondOrderSource(hessian), firstDensity * secondDensity, 1.0e-14);
        }
    }
}

TEST(Lpt, ParticleDisplacementRetainsInterpolationCorrection) {
    double const correction = detail::cloudCorrection(0.9 * std::numbers::pi);
    double const derivative = 0.125;
    for (auto const dimensions : {std::array{0, 3}, std::array{3, 1}}) {
        double const corrected = lpt::interpolationCorrectedDerivative(derivative, dimensions[0],
            dimensions[1], [correction](double value) { return value * correction; });
        EXPECT_NEAR(corrected / correction, derivative, 1.0e-15);
        EXPECT_GT(corrected, derivative);
    }
}
