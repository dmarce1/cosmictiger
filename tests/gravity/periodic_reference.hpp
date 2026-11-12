/* CosmicTiger - independent periodic correction reference; BSL-1.0. */
#pragma once

#include <array>
#include <cmath>
#include <numbers>

#include <cosmictiger/math/vector3.hpp>

namespace pmmm_test {
// Independent double lattice sum, with the primary singularity removed.
// It uses a full cubic image sum, without the generated tables or recurrences.
inline std::array<double, 4> periodicCorrectionReference(Vector3<double> const& point) {
    constexpr double kPi = std::numbers::pi;
    constexpr double kA = 4 * std::numbers::inv_sqrtpi;
    double const r2 = point.dot(point);
    double const r = std::sqrt(r2);
    std::array<double, 4> result{};
    result[0] = kPi / 4 + (r == 0 ? kA : std::erf(2 * r) / r);
    double const radial = r2 < 1e-10 ? -32 * std::numbers::inv_sqrtpi / 3 :
                                       (kA * std::exp(-4 * r2) - std::erf(2 * r) / r) / r2;
    for (int dim = 0; dim < 3; ++dim) {
        result[dim + 1] = -radial * point[dim];
    }
    for (int x = -4; x <= 4; ++x) {
        for (int y = -4; y <= 4; ++y) {
            for (int z = -4; z <= 4; ++z) {
                if (x == 0 && y == 0 && z == 0) {
                    continue;
                }
                Vector3<double> const image{point[0] + x, point[1] + y, point[2] + z};
                double const image2 = image.dot(image);
                double const radius = std::sqrt(image2);
                double const screened = std::erfc(2 * radius) / radius;
                double const gradient = (screened + kA * std::exp(-4 * image2)) / image2;
                double const wave2 = x * x + y * y + z * z;
                double const coefficient = std::exp(-kPi * kPi * wave2 / 4) / (kPi * wave2);
                Vector3<double> const wave{2 * kPi * x, 2 * kPi * y, 2 * kPi * z};
                double const phase = wave.dot(point);
                result[0] -= screened + coefficient * std::cos(phase);
                for (int dim = 0; dim < 3; ++dim) {
                    result[dim + 1] -=
                        gradient * image[dim] + coefficient * std::sin(phase) * wave[dim];
                }
            }
        }
    }
    return result;
}

}    // namespace pmmm_test
