#include <array>
#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include <cosmictiger/cosmology/lpt.hpp>
#include <cosmictiger/math/vector3.hpp>

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

Hessian modeHessian(Vector3<double> const& wavevector, double densityAmplitude) {
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
    std::array<Vector3<double>, 4> const directions = {
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

TEST(Lpt, RandomBoundaryPlaneIsHermitianWithoutAxisFolding) {
    constexpr int kSize = 64;
    double crossPlane = 0.0, foldedAxis = 0.0, variance = 0.0;
    for (int x = 1; x < kSize / 2; ++x) {
        for (int y = 1; y < kSize / 2; ++y) {
            auto const value = lpt::gaussianMode(x, y, 0, kSize);
            EXPECT_EQ(value, std::conj(lpt::gaussianMode(kSize - x, kSize - y, 0, kSize)));
            crossPlane += std::real(value * std::conj(lpt::gaussianMode(x, y, 1, kSize)));
            foldedAxis += std::real(value * std::conj(lpt::gaussianMode(x, kSize - y, 0, kSize)));
            variance += std::norm(value);
        }
    }
    EXPECT_LT(std::abs(crossPlane / variance), 0.1);
    EXPECT_LT(std::abs(foldedAxis / variance), 0.1);
    EXPECT_EQ(lpt::gaussianMode(0, 0, 0, kSize), std::complex<float>{});
    for (int index = 0; index < kSize; ++index) {
        EXPECT_EQ(lpt::gaussianMode(kSize / 2, index, 1, kSize), std::complex<float>{});
        EXPECT_EQ(lpt::gaussianMode(index, kSize / 2, 1, kSize), std::complex<float>{});
        EXPECT_EQ(lpt::gaussianMode(index, 1, kSize / 2, kSize), std::complex<float>{});
    }
}

TEST(Lpt, ShiftedInverseModeIsExactAtEachLatticeOffset) {
    constexpr int kSize = 32;
    // Check a near-Nyquist mode, where M6 interpolation/deconvolution produced
    // opposing dark-matter and gas amplitude errors.
    constexpr int kMode = 14;
    for (double const offset : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        auto const shift = lpt::latticeShift(kMode, 0, 0, kSize, offset);
        auto const partner = lpt::latticeShift(kSize - kMode, 0, 0, kSize, offset);
        EXPECT_NEAR(std::abs(shift), 1.0, 1.0e-7);
        EXPECT_EQ(shift, std::conj(partner));
        for (int node = 0; node < kSize; ++node) {
            double const angle = 2.0 * std::numbers::pi * kMode * node / kSize;
            std::complex<double> const inverse(std::cos(angle), std::sin(angle));
            double const actual = std::real(std::complex<double>(shift) * inverse);
            double const expected =
                std::cos(2.0 * std::numbers::pi * kMode * (node + offset) / kSize);
            EXPECT_NEAR(actual, expected, 1.0e-7);
        }
    }
}

TEST(Lpt, TruncatedLatticeCoordinatesRecoverTheirOwnedNodes) {
    for (int const size : {32, 768, 1536}) {
        for (bool const closePack : {false, true}) {
            for (int phase = 0; phase < (closePack ? 4 : 2); ++phase) {
                double const offset = lpt::latticeOffset(phase, closePack);
                for (int node = 0; node < size; ++node) {
                    double position = (node + offset) / size;
                    position -= std::floor(position);
                    position = std::floor(position * 0x1p32) * 0x1p-32;
                    EXPECT_EQ(lpt::latticeNode(position, size, offset), node);
                }
            }
        }
    }
}
