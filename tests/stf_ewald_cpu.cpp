/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>

#include <cosmictiger/ewald_kernels.hpp>
#include <cosmictiger/hpx.hpp>

// This standalone test exercises the actual CUDA-disabled initializer without
// starting a distributed runtime. No child locality needs initialization.
Vector<hpx::id_type> const& hpxChildren() {
    static Vector<hpx::id_type> const children;
    return children;
}

namespace detail {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kAlpha = 2.;
using Position = std::array<double, 3>;

void requireNear(double actual, double expected, double tolerance, char const* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::fprintf(
            stderr, "%s: %.17g != %.17g (tolerance %.3g)\n", label, actual, expected, tolerance);
        throw std::runtime_error(label);
    }
}

double referenceFactorial(int value) {
    double result = 1;
    for (int i = 2; i <= value; ++i) {
        result *= i;
    }
    return result;
}

void checkImageIndices() {
    std::set<std::array<int, 3>> expected;
    for (int x = -4; x <= 4; ++x) {
        for (int y = -4; y <= 4; ++y) {
            for (int z = -4; z <= 4; ++z) {
                double radiusSquared = 0;
                for (int coordinate : {x, y, z}) {
                    double const nearest = std::max(std::abs(coordinate) - .5, 0.);
                    radiusSquared += nearest * nearest;
                }
                if ((x || y || z) && radiusSquared < EWALD_REAL_CUTOFF2) {
                    expected.insert({x, y, z});
                }
            }
        }
    }
    std::set<std::array<int, 3>> actual;
    for (int i = 0; i < EwaldConst::nreal(); ++i) {
        auto const& image = EwaldConst::realIndex(i);
        actual.insert({int(image[0]), int(image[1]), int(image[2])});
    }
    if (actual != expected || actual.size() != NREAL) {
        throw std::runtime_error("CPU Ewald real-image list has omitted or duplicate images");
    }
}

void checkSelfDerivatives() {
    auto const derivatives = EwaldConst::d0();
    for (int x = 0; x < LORDER; ++x) {
        for (int y = 0; y < LORDER - x; ++y) {
            for (int z = 0; z < LORDER - x - y; ++z) {
                int const rank = x + y + z;
                double expected = 0;
                if ((x % 2 == 0) && (y % 2 == 0) && (z % 2 == 0)) {
                    int const radialPower = rank / 2;
                    // Taylor coefficients of -erf(alpha*r)/r, expanded as a
                    // Cartesian polynomial in (x*x+y*y+z*z)^radialPower.
                    expected = -2 / std::sqrt(kPi) * (radialPower % 2 ? -1 : 1) *
                        std::pow(kAlpha, rank + 1) / (rank + 1) * referenceFactorial(x) *
                        referenceFactorial(y) * referenceFactorial(z) /
                        (referenceFactorial(x / 2) * referenceFactorial(y / 2) *
                            referenceFactorial(z / 2));
                }
                double const physical = derivatives(x, y, z) * std::pow(SCALE_FACTOR, rank + 1);
                requireNear(physical, expected, 2e-6 * std::max(1., std::abs(expected)),
                    "CPU Ewald self derivative erf-series reference");
            }
        }
    }
}

void checkFourierScaling() {
    for (int image = 0; image < EwaldConst::nfour(); ++image) {
        auto const& wave = EwaldConst::fourIndex(image);
        auto const& derivative = EwaldConst::fourExpansion(image);
        double const h2 = wave[0] * wave[0] + wave[1] * wave[1] + wave[2] * wave[2];
        double const amplitude = -std::exp(-kPi * kPi * h2 / (kAlpha * kAlpha)) / (kPi * h2);
        // Ranks zero through two retain the complete Fourier derivatives; STF
        // trace subtraction starts at rank three for this intermediate tensor.
        for (int rank = 0; rank <= 2; ++rank) {
            for (int x = 0; x <= rank; ++x) {
                for (int y = 0; y <= rank - x; ++y) {
                    int const z = rank - x - y;
                    double const sign = rank == 0 ? 1. : -1.;
                    double const expected = amplitude * sign * std::pow(2 * kPi, rank) *
                        std::pow(wave[0], x) * std::pow(wave[1], y) * std::pow(wave[2], z);
                    double const physical = derivative(x, y, z) * std::pow(SCALE_FACTOR, rank + 1);
                    requireNear(physical, expected, 2e-6 * std::max(1e-20, std::abs(expected)),
                        "CPU Ewald Fourier normalization");
                }
            }
        }
    }
}

std::array<double, 4> directEwaldCorrection(Position const& position) {
    std::array<double, 4> result{};
    double const inverseSqrtPi = 1 / std::sqrt(kPi);
    double const radiusSquared =
        position[0] * position[0] + position[1] * position[1] + position[2] * position[2];
    result[0] = kPi / (kAlpha * kAlpha);
    if (radiusSquared == 0) {
        result[0] += 2 * kAlpha * inverseSqrtPi;
    } else {
        double const radius = std::sqrt(radiusSquared);
        result[0] += std::erf(kAlpha * radius) / radius;
        double const radial =
            (2 * kAlpha * inverseSqrtPi * std::exp(-kAlpha * kAlpha * radiusSquared) -
                std::erf(kAlpha * radius) / radius) /
            radiusSquared;
        for (int d = 0; d < 3; ++d) {
            result[d + 1] += radial * position[d];
        }
    }
    // Larger independent real and reciprocal cubes exceed production cutoffs.
    for (int x = -4; x <= 4; ++x) {
        for (int y = -4; y <= 4; ++y) {
            for (int z = -4; z <= 4; ++z) {
                if (!(x || y || z)) {
                    continue;
                }
                Position const image{position[0] + x, position[1] + y, position[2] + z};
                double const imageRadiusSquared =
                    image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
                double const radius = std::sqrt(imageRadiusSquared);
                double const screened = std::erfc(kAlpha * radius) / radius;
                result[0] -= screened;
                double const radial = (screened +
                                          2 * kAlpha * inverseSqrtPi *
                                              std::exp(-kAlpha * kAlpha * imageRadiusSquared)) /
                    imageRadiusSquared;
                Position const wave{2 * kPi * x, 2 * kPi * y, 2 * kPi * z};
                double const waveSquared = x * x + y * y + z * z;
                double const coefficient =
                    std::exp(-kPi * kPi * waveSquared / (kAlpha * kAlpha)) / (kPi * waveSquared);
                double const phase =
                    wave[0] * position[0] + wave[1] * position[1] + wave[2] * position[2];
                result[0] -= coefficient * std::cos(phase);
                for (int d = 0; d < 3; ++d) {
                    result[d + 1] += radial * image[d] + coefficient * std::sin(phase) * wave[d];
                }
            }
        }
    }
    return result;
}

void checkPeriodicField() {
    constexpr std::array<Position, 7> kPositions{{{0., 0., 0.}, {.01, -.02, .03}, {.1, 0., 0.},
        {.23, -.17, .31}, {-.23, .17, -.31}, {.499, -.497, .11}, {-.49, .48, .47}}};
    for (auto const& position : kPositions) {
        auto const expected = directEwaldCorrection(position);
        Array<double, NDIM> separation;
        for (int d = 0; d < 3; ++d) {
            separation[d] = position[d];
        }
        SymmetricTracelessTensor<double, LORDER> actual;
        ewaldCorrectionDerivatives(actual, separation);
        requireNear(actual(0, 0, 0) * SCALE_FACTOR, expected[0], 2e-5,
            "CPU Ewald independent periodic potential");
        for (int d = 0; d < 3; ++d) {
            requireNear(actual(d == 0, d == 1, d == 2) * SCALE_FACTOR2, expected[d + 1], 2e-5,
                "CPU Ewald independent periodic gradient");
        }
    }
}

}    // namespace detail

int main() {
    try {
        EwaldConst::init();
        detail::checkImageIndices();
        detail::checkSelfDerivatives();
        detail::checkFourierScaling();
        detail::checkPeriodicField();
        std::puts("CPU Ewald initialization and independent periodic potential/gradient passed.");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "CPU Ewald test failed: %s\n", error.what());
        return 1;
    }
}
