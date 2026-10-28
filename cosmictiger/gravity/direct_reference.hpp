/* CosmicTiger - independent direct gravity accuracy reference; BSL-1.0. */
#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <cosmictiger/math/radial_kernel.hpp>

namespace detail {

// Compensated double accumulation keeps the diagnostic's roundoff below the
// single-precision production forces, including nearly homogeneous loads.
class GravityReferenceSum {
    double sum_ = 0.0;
    double correction_ = 0.0;

public:
    void add(double value) {
        double const adjusted = value - correction_;
        double const next = sum_ + adjusted;
        correction_ = (next - sum_) - adjusted;
        sum_ = next;
    }
    double value() const {
        return sum_;
    }
};

struct GravityReferenceWave {
    std::array<double, 3> wave;
    double coefficient;
};

inline std::vector<std::array<double, 3>> const& gravityReferenceImages() {
    static auto const images = [] {
        std::vector<std::array<double, 3>> result;
        for (int x = -4; x <= 4; ++x) {
            for (int y = -4; y <= 4; ++y) {
                for (int z = -4; z <= 4; ++z) {
                    if (x != 0 || y != 0 || z != 0) {
                        result.push_back({double(x), double(y), double(z)});
                    }
                }
            }
        }
        return result;
    }();
    return images;
}

inline std::vector<GravityReferenceWave> const& gravityReferenceWaves() {
    static auto const waves = [] {
        std::vector<GravityReferenceWave> result;
        constexpr double kPi = std::numbers::pi;
        for (int x = -4; x <= 4; ++x) {
            for (int y = -4; y <= 4; ++y) {
                for (int z = -4; z <= 4; ++z) {
                    int const squared = x * x + y * y + z * z;
                    if (squared != 0) {
                        result.push_back({{2 * kPi * x, 2 * kPi * y, 2 * kPi * z},
                            std::exp(-kPi * kPi * squared / 4) / (kPi * squared)});
                    }
                }
            }
        }
        return result;
    }();
    return waves;
}

// Unit-source acceleration, in the same unit box and fixed support convention
// as the production gravity solvers. displacement is sink minus source; the
// caller supplies its nearest periodic image. The periodic term independently
// sums the zero-mean Ewald Green function (alpha=2, larger +/-4 image cubes),
// without any FMM expansion, mesh approximation or production Ewald tables.
inline std::array<double, 3> directGravityReference(std::array<double, 3> const& displacement,
    radial_kernel::KernelTable const& kernel, double supportRadius, bool periodic = true) {
    double const radiusSquared = displacement[0] * displacement[0] +
        displacement[1] * displacement[1] + displacement[2] * displacement[2];
    if (radiusSquared == 0.0) {
        // Both a source's own force and the force of a distinct coincident
        // softened source vanish, including all periodic images.
        return {};
    }
    double const radius = std::sqrt(radiusSquared);
    double const inverseSupport = 1.0 / supportRadius;
    double const softened = radius < supportRadius ?
        radial_kernel::forceFactor(kernel, radius * inverseSupport) * inverseSupport *
            inverseSupport * inverseSupport :
        1.0 / (radiusSquared * radius);
    std::array<GravityReferenceSum, 3> result;
    for (int dim = 0; dim < 3; ++dim) {
        result[dim].add(-softened * displacement[dim]);
    }
    if (periodic) {
        constexpr double kAlpha = 2.0;
        double const normalization = 2 * kAlpha / std::sqrt(std::numbers::pi);
        double const scaledSquared = kAlpha * kAlpha * radiusSquared;
        double radial;
        if (scaledSquared < 1.0e-4) {
            // (erf(alpha*r)/r - 2*alpha/sqrt(pi)*exp(-alpha^2*r^2))/r^2,
            // with its cancellation removed analytically at tiny separation.
            radial = normalization * kAlpha * kAlpha *
                (2.0 / 3 +
                    scaledSquared *
                        (-2.0 / 5 +
                            scaledSquared *
                                (1.0 / 7 + scaledSquared * (-1.0 / 27 + scaledSquared / 132))));
        } else {
            radial =
                (std::erf(kAlpha * radius) / radius - normalization * std::exp(-scaledSquared)) /
                radiusSquared;
        }
        for (int dim = 0; dim < 3; ++dim) {
            result[dim].add(radial * displacement[dim]);
        }
        for (auto const& shift : gravityReferenceImages()) {
            std::array<double, 3> const image{
                displacement[0] + shift[0], displacement[1] + shift[1], displacement[2] + shift[2]};
            double const squared = image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
            double const distance = std::sqrt(squared);
            double const screened = std::erfc(kAlpha * distance) / distance;
            double const imageRadial =
                (screened + normalization * std::exp(-kAlpha * kAlpha * squared)) / squared;
            for (int dim = 0; dim < 3; ++dim) {
                result[dim].add(-imageRadial * image[dim]);
            }
        }
        for (auto const& wave : gravityReferenceWaves()) {
            double phase = 0;
            for (int dim = 0; dim < 3; ++dim) {
                phase += wave.wave[dim] * displacement[dim];
            }
            double const amplitude = -wave.coefficient * std::sin(phase);
            for (int dim = 0; dim < 3; ++dim) {
                result[dim].add(amplitude * wave.wave[dim]);
            }
        }
    }
    return {result[0].value(), result[1].value(), result[2].value()};
}

}    // namespace detail
