/* CosmicTiger - symmetry-compressed periodic gravity reference; BSL-1.0. */
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/radial_kernel.hpp>

namespace detail {

inline constexpr int kGravityReferenceTableIntervals = 64;
inline constexpr int kGravityReferenceTableHalo = 3;
inline constexpr int kGravityReferenceTableMaximumIndex =
    kGravityReferenceTableIntervals + kGravityReferenceTableHalo;
inline constexpr int kGravityReferenceTableStencilSize = 8;
inline constexpr size_t kGravityReferenceTableSize = 54740;

// Dense packing of the integer wedge 0 <= k <= j <= i, including its faces.
CUDA_EXPORT constexpr size_t gravityReferenceTableIndex(int i, int j, int k) {
    return size_t(i) * (i + 1) * (i + 2) / 6 + size_t(j) * (j + 1) / 2 + k;
}

static_assert(gravityReferenceTableIndex(kGravityReferenceTableMaximumIndex + 1, 0, 0) ==
    kGravityReferenceTableSize);

// Built only on first use, independently from production FMM/PMMM tables. The
// positive halo is evaluated analytically beyond 1/2; it is never wrapped.
std::vector<std::array<double, 3>> const& gravityReferenceCorrectionTable();

template <class Value>
CUDA_EXPORT inline void gravityReferenceTableSort(Value* coordinates, int* permutation) {
    for (int pass = 0; pass < 3; ++pass) {
        int const first = pass == 1 ? 1 : 0;
        int const second = first + 1;
        if (coordinates[first] < coordinates[second]) {
            Value const coordinate = coordinates[first];
            coordinates[first] = coordinates[second];
            coordinates[second] = coordinate;
            int const dimension = permutation[first];
            permutation[first] = permutation[second];
            permutation[second] = dimension;
        }
    }
}

CUDA_EXPORT inline void gravityReferenceTableEqualComponents(
    double const* coordinates, std::array<double, 3>& force) {
    if (coordinates[0] == coordinates[2]) {
        double const mean = (force[0] + force[1] + force[2]) / 3;
        force = {mean, mean, mean};
    } else if (coordinates[0] == coordinates[1]) {
        double const mean = (force[0] + force[1]) / 2;
        force[0] = force[1] = mean;
    } else if (coordinates[1] == coordinates[2]) {
        double const mean = (force[1] + force[2]) / 2;
        force[1] = force[2] = mean;
    }
}

CUDA_EXPORT inline std::array<double, 3> gravityReferenceTableNode(
    int i, int j, int k, std::array<double, 3> const* values) {
    int coordinates[3] = {i < 0 ? -i : i, j < 0 ? -j : j, k < 0 ? -k : k};
    double const signs[3] = {i < 0 ? -1.0 : 1.0, j < 0 ? -1.0 : 1.0, k < 0 ? -1.0 : 1.0};
    int permutation[3] = {0, 1, 2};
    gravityReferenceTableSort(coordinates, permutation);
    auto const& value =
        values[gravityReferenceTableIndex(coordinates[0], coordinates[1], coordinates[2])];
    std::array<double, 3> result{};
    for (int dim = 0; dim < 3; ++dim) {
        result[permutation[dim]] = signs[permutation[dim]] * value[dim];
    }
    return result;
}

// Smooth periodic correction for a nearest-image displacement in [-1/2,1/2]^3.
// The vector is odd in its own coordinate and even in the other two. Sorting
// every stencil node also restores components when a stencil crosses a wedge
// face. Centered degree-seven interpolation uses the analytic positive halo
// and parity at zero, without imposing false periodicity on the correction.
CUDA_EXPORT inline std::array<double, 3> gravityReferenceTableCorrection(
    std::array<double, 3> const& displacement, std::array<double, 3> const* values) {
    double const radiusSquared = displacement[0] * displacement[0] +
        displacement[1] * displacement[1] + displacement[2] * displacement[2];
    if (radiusSquared < 1.0e-12) {
        // Cubic symmetry and Poisson's equation fix the linear response.
        // This avoids losing tiny fixed64 separations when the interpolation
        // offset rounds to an integer. Omitted cubic terms are about 1e-17
        // at the branch boundary, below the table's absolute error floor.
        constexpr double kOriginSlope = 4 * std::numbers::pi / 3;
        return {kOriginSlope * displacement[0], kOriginSlope * displacement[1],
            kOriginSlope * displacement[2]};
    }
    double coordinates[3] = {
        std::abs(displacement[0]), std::abs(displacement[1]), std::abs(displacement[2])};
    int permutation[3] = {0, 1, 2};
    gravityReferenceTableSort(coordinates, permutation);
    int starts[3];
    double weights[3][kGravityReferenceTableStencilSize];
    constexpr double kInverseDenominators[kGravityReferenceTableStencilSize] = {-1.0 / 5040,
        1.0 / 720, -1.0 / 240, 1.0 / 144, -1.0 / 144, 1.0 / 240, -1.0 / 720, 1.0 / 5040};
    for (int dim = 0; dim < 3; ++dim) {
        double const coordinate = coordinates[dim] * (2 * kGravityReferenceTableIntervals);
        int const cell = int(coordinate) < kGravityReferenceTableIntervals ?
            int(coordinate) :
            kGravityReferenceTableIntervals - 1;
        starts[dim] = cell - kGravityReferenceTableHalo;
        double const offset = coordinate - starts[dim];
        double prefixes[kGravityReferenceTableStencilSize + 1];
        prefixes[0] = 1;
        for (int i = 0; i < kGravityReferenceTableStencilSize; ++i) {
            prefixes[i + 1] = prefixes[i] * (offset - i);
        }
        double suffix = 1;
        for (int i = kGravityReferenceTableStencilSize - 1; i >= 0; --i) {
            weights[dim][i] = kInverseDenominators[i] * prefixes[i] * suffix;
            suffix *= offset - i;
        }
    }
    std::array<double, 3> force{};
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int i = 0; i < kGravityReferenceTableStencilSize; ++i) {
        std::array<double, 3> plane{};
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int j = 0; j < kGravityReferenceTableStencilSize; ++j) {
            std::array<double, 3> row{};
#ifdef __CUDACC__
#pragma unroll 1
#endif
            for (int k = 0; k < kGravityReferenceTableStencilSize; ++k) {
                auto const value =
                    gravityReferenceTableNode(starts[0] + i, starts[1] + j, starts[2] + k, values);
                for (int dim = 0; dim < 3; ++dim) {
                    row[dim] += weights[2][k] * value[dim];
                }
            }
            for (int dim = 0; dim < 3; ++dim) {
                plane[dim] += weights[1][j] * row[dim];
            }
        }
        for (int dim = 0; dim < 3; ++dim) {
            force[dim] += weights[0][i] * plane[dim];
        }
    }
    gravityReferenceTableEqualComponents(coordinates, force);
    std::array<double, 3> result{};
    for (int dim = 0; dim < 3; ++dim) {
        int const original = permutation[dim];
        result[original] = displacement[original] == 0 ?
            0 :
            (displacement[original] < 0 ? -force[dim] : force[dim]);
    }
    return result;
}

// Unit-source acceleration. The primary force uses the selected production
// softening kernel; periodic images remain Newtonian, including when the
// support radius exceeds half the box. Mass and G belong to the caller.
CUDA_EXPORT inline std::array<double, 3> tabulatedGravityReference(
    std::array<double, 3> const& displacement, radial_kernel::KernelTable const& kernel,
    double supportRadius, std::array<double, 3> const* values) {
    double const radiusSquared = displacement[0] * displacement[0] +
        displacement[1] * displacement[1] + displacement[2] * displacement[2];
    if (radiusSquared == 0) {
        return {};
    }
    double const radius = std::sqrt(radiusSquared);
    double const inverseSupport = 1 / supportRadius;
    double const primary = radius < supportRadius ?
        radial_kernel::forceFactor(kernel, radius * inverseSupport) * inverseSupport *
            inverseSupport * inverseSupport :
        1 / (radiusSquared * radius);
    auto force = gravityReferenceTableCorrection(displacement, values);
    for (int dim = 0; dim < 3; ++dim) {
        force[dim] -= primary * displacement[dim];
        // Reflection makes the complete Newtonian periodic force normal to a
        // half-box face exactly zero. Softening only the primary image breaks
        // that cancellation when the support reaches this separation.
        if (radius >= supportRadius && std::abs(displacement[dim]) == 0.5) {
            force[dim] = 0;
        }
    }
    return force;
}

}    // namespace detail
