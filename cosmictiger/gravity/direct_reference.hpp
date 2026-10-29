/* CosmicTiger - independent direct gravity accuracy reference; BSL-1.0. */
#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <numbers>
#include <type_traits>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/radial_kernel.hpp>

namespace detail {

// Compensated double accumulation keeps the diagnostic's roundoff below the
// single-precision production forces, including nearly homogeneous loads.
class GravityReferenceSum {
    double sum_ = 0.0;
    double correction_ = 0.0;

public:
    CUDA_EXPORT GravityReferenceSum() = default;

    CUDA_EXPORT void add(double value) {
        double const adjusted = value - correction_;
        double const next = sum_ + adjusted;
        correction_ = (next - sum_) - adjusted;
        sum_ = next;
    }
    CUDA_EXPORT double value() const {
        return sum_;
    }
};

struct GravityReferenceWave {
    double wave[3];
    double coefficient;
};

inline constexpr int kGravityReferenceImageCount = 364;
inline constexpr int kGravityReferenceWaveCount = 128;
inline constexpr double kGravityReferenceRealRadius = 3.6;
inline constexpr double kGravityReferenceRealRadiusSquared =
    kGravityReferenceRealRadius * kGravityReferenceRealRadius;
inline constexpr int kGravityReferenceWaveRadiusSquared = 16;

// Trivially copyable storage lets CPU and CUDA evaluate the same independent
// reference, with coefficients prepared once on the host rather than per pair.
struct GravityReferenceTables {
    double images[kGravityReferenceImageCount][3];
    GravityReferenceWave waves[kGravityReferenceWaveCount];
};

static_assert(std::is_trivially_copyable_v<GravityReferenceTables>);

inline GravityReferenceTables const& gravityReferenceTables() {
    static auto const tables = [] {
        GravityReferenceTables result{};
        int imageCount = 0;
        int waveCount = 0;
        constexpr double kPi = std::numbers::pi;
        for (int x = -4; x <= 4; ++x) {
            for (int y = -4; y <= 4; ++y) {
                for (int z = -4; z <= 4; ++z) {
                    int const squared = x * x + y * y + z * z;
                    if (squared == 0) {
                        continue;
                    }
                    // The wrapped displacement lies in [-1/2,1/2]^3. Keep
                    // exactly the shifts whose image can enter radius 3.6.
                    double const nearestX = x == 0 ? 0 : std::abs(x) - 0.5;
                    double const nearestY = y == 0 ? 0 : std::abs(y) - 0.5;
                    double const nearestZ = z == 0 ? 0 : std::abs(z) - 0.5;
                    double const minimumSquared =
                        nearestX * nearestX + nearestY * nearestY + nearestZ * nearestZ;
                    if (minimumSquared < kGravityReferenceRealRadiusSquared) {
                        assert(imageCount < kGravityReferenceImageCount);
                        result.images[imageCount][0] = x;
                        result.images[imageCount][1] = y;
                        result.images[imageCount][2] = z;
                        ++imageCount;
                    }
                    // The +h and -h force terms are identical: sin changes
                    // sign along with h. Store one representative and double
                    // its coefficient, including modes exactly on radius 4.
                    bool const positive = x > 0 || (x == 0 && y > 0) || (x == 0 && y == 0 && z > 0);
                    if (positive && squared <= kGravityReferenceWaveRadiusSquared) {
                        assert(waveCount < kGravityReferenceWaveCount);
                        result.waves[waveCount] = {{2 * kPi * x, 2 * kPi * y, 2 * kPi * z},
                            2 * std::exp(-kPi * kPi * squared / 4) / (kPi * squared)};
                        ++waveCount;
                    }
                }
            }
        }
        assert(imageCount == kGravityReferenceImageCount);
        assert(waveCount == kGravityReferenceWaveCount);
        return result;
    }();
    return tables;
}

// Smooth unit-source periodic acceleration minus the primary Newtonian force.
// displacement is sink minus source, normally wrapped to [-1/2,1/2]^3. The
// image list also covers the table-builder halo [-67/128,67/128]^3. This
// independent Ewald sum uses alpha=2, real radius 3.6 and reciprocal radius 4. The cutoffs
// exceed production FMM's radius 2.6 and reciprocal squared radius 8, without
// using its tables or approximated erfc. No singular primary force is formed,
// so this can also build a correction table accurately arbitrarily near zero.
CUDA_EXPORT inline std::array<double, 3> gravityReferenceCorrection(
    std::array<double, 3> const& displacement, GravityReferenceTables const& tables) {
    double const radiusSquared = displacement[0] * displacement[0] +
        displacement[1] * displacement[1] + displacement[2] * displacement[2];
    if (radiusSquared == 0.0) {
        return {};
    }
    double const radius = std::sqrt(radiusSquared);
    std::array<GravityReferenceSum, 3> result;
    constexpr double kAlpha = 2.0;
    constexpr double kNormalization = 2 * kAlpha * std::numbers::inv_sqrtpi;
    double const scaledSquared = kAlpha * kAlpha * radiusSquared;
    double radial;
    if (scaledSquared < 1.0e-4) {
        // (erf(alpha*r)/r - 2*alpha/sqrt(pi)*exp(-alpha^2*r^2))/r^2,
        // with its cancellation removed analytically at tiny separation.
        radial = kNormalization * kAlpha * kAlpha *
            (2.0 / 3 +
                scaledSquared *
                    (-2.0 / 5 +
                        scaledSquared *
                            (1.0 / 7 + scaledSquared * (-1.0 / 27 + scaledSquared / 132))));
    } else {
        radial = (std::erf(kAlpha * radius) / radius - kNormalization * std::exp(-scaledSquared)) /
            radiusSquared;
    }
    for (int dim = 0; dim < 3; ++dim) {
        result[dim].add(radial * displacement[dim]);
    }
    for (auto const& shift : tables.images) {
        std::array<double, 3> const image{
            displacement[0] + shift[0], displacement[1] + shift[1], displacement[2] + shift[2]};
        double const squared = image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
        if (squared >= kGravityReferenceRealRadiusSquared) {
            continue;
        }
        double const distance = std::sqrt(squared);
        double const screened = std::erfc(kAlpha * distance) / distance;
        double const imageRadial =
            (screened + kNormalization * std::exp(-kAlpha * kAlpha * squared)) / squared;
        for (int dim = 0; dim < 3; ++dim) {
            result[dim].add(-imageRadial * image[dim]);
        }
    }
    for (auto const& wave : tables.waves) {
        double phase = 0;
        for (int dim = 0; dim < 3; ++dim) {
            phase += wave.wave[dim] * displacement[dim];
        }
        double const amplitude = -wave.coefficient * std::sin(phase);
        for (int dim = 0; dim < 3; ++dim) {
            result[dim].add(amplitude * wave.wave[dim]);
        }
    }
    return {result[0].value(), result[1].value(), result[2].value()};
}

// Unit-source acceleration with the production compact-support convention.
// Only the nearest image is softened, even when support exceeds half a box.
CUDA_EXPORT inline std::array<double, 3> directGravityReference(
    std::array<double, 3> const& displacement, radial_kernel::KernelTable const& kernel,
    double supportRadius, bool periodic, GravityReferenceTables const& tables) {
    double const radiusSquared = displacement[0] * displacement[0] +
        displacement[1] * displacement[1] + displacement[2] * displacement[2];
    if (radiusSquared == 0.0) {
        // Self and distinct coincident softened sources both exert zero force.
        return {};
    }
    double const radius = std::sqrt(radiusSquared);
    double const inverseSupport = 1.0 / supportRadius;
    // Reuse the CPU/CUDA radial evaluator used by both production PP solvers.
    double const softened = radius < supportRadius ?
        radial_kernel::forceFactor(kernel, radius * inverseSupport) * inverseSupport *
            inverseSupport * inverseSupport :
        1.0 / (radiusSquared * radius);
    std::array<double, 3> result =
        periodic ? gravityReferenceCorrection(displacement, tables) : std::array<double, 3>{};
    for (int dim = 0; dim < 3; ++dim) {
        result[dim] -= softened * displacement[dim];
    }
    return result;
}

inline std::array<double, 3> directGravityReference(std::array<double, 3> const& displacement,
    radial_kernel::KernelTable const& kernel, double supportRadius, bool periodic = true) {
    return directGravityReference(
        displacement, kernel, supportRadius, periodic, gravityReferenceTables());
}

}    // namespace detail
