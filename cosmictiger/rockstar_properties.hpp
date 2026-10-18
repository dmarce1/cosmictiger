/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Licensed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace rockstar {

namespace detail {

    // Production builds enable -ffast-math, which may discard std::isfinite checks.
    inline bool isFinite(double value) {
        static_assert(
            sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
        constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
        return (std::bit_cast<std::uint64_t>(value) & kExponentMask) != kExponentMask;
    }

}    // namespace detail

struct SphericalOverdensity {
    size_t count = 0;
    double radius = 0.0;
};

// Use the outermost complete radial shell above the requested mean number density.
// Between particles, M(<r) is constant, so this boundary gives both an integer
// particle mass and exactly the requested mean density. The next shell, if any,
// lies outside this radius. The caller must provide all candidate particles that
// could lie inside the SO boundary; a truncated candidate list cannot establish
// that no further outer crossing exists.
inline SphericalOverdensity sphericalOverdensity(
    std::vector<double> const& sortedRadii, double numberDensity) {
    if (!(numberDensity > 0.0) || !detail::isFinite(numberDensity)) {
        throw std::invalid_argument("SO number density must be finite and positive");
    }
    SphericalOverdensity result;
    double previousRadius = 0.0;
    size_t shellEnd = 0;
    while (shellEnd < sortedRadii.size()) {
        double const shellRadius = sortedRadii[shellEnd];
        if (!(shellRadius >= previousRadius) || !detail::isFinite(shellRadius)) {
            throw std::invalid_argument("SO radii must be finite, nonnegative, and sorted");
        }
        do {
            ++shellEnd;
        } while (shellEnd < sortedRadii.size() && sortedRadii[shellEnd] == shellRadius);
        double const densityRadius =
            std::cbrt((double(shellEnd) / numberDensity) * (3.0 / (4.0 * std::numbers::pi)));
        if (shellRadius <= densityRadius) {
            result = {shellEnd, densityRadius};
        }
        previousRadius = shellRadius;
    }
    return result;
}

struct VirialOverdensity {
    double matterFraction;
    double critical;
    double meanMatter;
};

// Bryan & Norman's fits are calibrated for flat matter+Lambda cosmologies and,
// with openMatterOnly=true, open matter-only cosmologies. Including radiation
// in Omega_m(a) is an extrapolation, not a new spherical-collapse fit.
inline VirialOverdensity virialOverdensityFromMatterFraction(
    double matterFraction, bool openMatterOnly = false) {
    if (!(matterFraction > 0.0) || !detail::isFinite(matterFraction)) {
        throw std::invalid_argument("Virial matter fraction must be finite and positive");
    }
    double const offset = matterFraction - 1.0;
    double const linear = openMatterOnly ? 60.0 : 82.0;
    double const quadratic = openMatterOnly ? 32.0 : 39.0;
    double const critical =
        18.0 * std::numbers::pi * std::numbers::pi + linear * offset - quadratic * offset * offset;
    if (!(critical > 0.0) || !detail::isFinite(critical)) {
        throw std::invalid_argument("Virial overdensity is outside the fitted domain");
    }
    return {matterFraction, critical, critical / matterFraction};
}

// Convenience for flat cosmologies. Call the matter-fraction helper directly
// when the simulation supplies independent curvature or Lambda parameters.
inline VirialOverdensity virialOverdensity(
    double scaleFactor, double omegaMatter, double omegaRadiation = 0.0) {
    if (!(scaleFactor > 0.0) || !detail::isFinite(scaleFactor) || !(omegaMatter > 0.0) ||
        !detail::isFinite(omegaMatter) || !(omegaRadiation >= 0.0) ||
        !detail::isFinite(omegaRadiation) || omegaMatter + omegaRadiation > 1.0) {
        throw std::invalid_argument("Invalid flat cosmology for virial overdensity");
    }
    double const omegaLambda = 1.0 - omegaMatter - omegaRadiation;
    // Multiplying H(a)^2/H0^2 by a^3 avoids inverse powers at early epochs.
    double const denominator = omegaMatter + omegaRadiation / scaleFactor +
        omegaLambda * scaleFactor * scaleFactor * scaleFactor;
    return virialOverdensityFromMatterFraction(omegaMatter / denominator);
}

namespace detail {

    inline double nfwMassIntegral(double radiusRatio) {
        if (radiusRatio < 1.e-3) {
            return radiusRatio * radiusRatio *
                (0.5 +
                    radiusRatio *
                        (-2.0 / 3.0 +
                            radiusRatio *
                                (0.75 +
                                    radiusRatio *
                                        (-0.8 +
                                            radiusRatio * (5.0 / 6.0 - radiusRatio * 6.0 / 7.0)))));
        }
        return std::log1p(radiusRatio) - radiusRatio / (1.0 + radiusRatio);
    }

}    // namespace detail

struct NfwFit {
    // Zero denotes an unresolved or boundary-limited fit.
    double scaleRadius = 0.0;
    double density = 0.0;
};

inline NfwFit nfwFit(std::vector<double> const& sortedRadii) {
    size_t const particleCount = sortedRadii.size();
    size_t const targetBins = std::min(size_t(50), particleCount / 15);
    if (targetBins < 2) {
        return {};
    }
    double previousRadius = 0.0;
    for (double const radius : sortedRadii) {
        if (!(radius >= previousRadius) || !detail::isFinite(radius)) {
            throw std::invalid_argument("NFW radii must be finite, nonnegative, and sorted");
        }
        previousRadius = radius;
    }
    double const outerRadius = sortedRadii.back();
    if (!(outerRadius > 0.0)) {
        return {};
    }
    std::vector<double> boundaries{0.0};
    std::vector<size_t> counts;
    size_t previousCount = 0;
    for (size_t bin = 1; bin < targetBins; ++bin) {
        size_t const count = bin * particleCount / targetBins;
        // Never split coincident particles or create a zero-width radial bin.
        if (sortedRadii[count - 1] == sortedRadii[count]) {
            continue;
        }
        double const boundary =
            (0.5 * sortedRadii[count - 1] + 0.5 * sortedRadii[count]) / outerRadius;
        if (boundary <= boundaries.back()) {
            continue;
        }
        boundaries.push_back(boundary);
        counts.push_back(count - previousCount);
        previousCount = count;
    }
    boundaries.push_back(1.0);
    counts.push_back(particleCount - previousCount);
    if (counts.size() < 2) {
        return {};
    }
    auto const negativeLogLikelihood = [&](double logConcentration) {
        double const concentration = std::exp(logConcentration);
        double const normalization = detail::nfwMassIntegral(concentration);
        double result = 0.0;
        double innerMass = 0.0;
        for (size_t bin = 0; bin < counts.size(); ++bin) {
            double const outerMass = detail::nfwMassIntegral(concentration * boundaries[bin + 1]);
            double const probability = (outerMass - innerMass) / normalization;
            if (!(probability > 0.0)) {
                return std::numeric_limits<double>::infinity();
            }
            result -= double(counts[bin]) / particleCount * std::log(probability);
            innerMass = outerMass;
        }
        return result;
    };
    // Fit the multinomial radial-bin likelihood, including the shell-integrated
    // NFW mass. Sampling density at bin midpoints biases resolved scale radii.
    // The wide finite interval also bounds work on unresolved/non-NFW profiles.
    constexpr double kMinLogConcentration = -16.0;
    constexpr double kMaxLogConcentration = 16.0;
    constexpr double kGoldenRatio = 0.6180339887498948482;
    double lower = kMinLogConcentration;
    double upper = kMaxLogConcentration;
    double left = upper - kGoldenRatio * (upper - lower);
    double right = lower + kGoldenRatio * (upper - lower);
    double leftValue = negativeLogLikelihood(left);
    double rightValue = negativeLogLikelihood(right);
    for (int iteration = 0; iteration < 96 && upper - lower > 1.e-9; ++iteration) {
        if (leftValue < rightValue) {
            upper = right;
            right = left;
            rightValue = leftValue;
            left = upper - kGoldenRatio * (upper - lower);
            leftValue = negativeLogLikelihood(left);
        } else {
            lower = left;
            left = right;
            leftValue = rightValue;
            right = lower + kGoldenRatio * (upper - lower);
            rightValue = negativeLogLikelihood(right);
        }
    }
    double const logConcentration = 0.5 * (lower + upper);
    double const likelihood = negativeLogLikelihood(logConcentration);
    double const boundaryLikelihood = std::min(
        negativeLogLikelihood(kMinLogConcentration), negativeLogLikelihood(kMaxLogConcentration));
    double const likelihoodResolution = 256.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(boundaryLikelihood));
    if (!detail::isFinite(likelihood) || boundaryLikelihood - likelihood <= likelihoodResolution) {
        return {};
    }
    double const concentration = std::exp(logConcentration);
    double const scaleRadius = outerRadius / concentration;
    double const density = particleCount /
        (4.0 * std::numbers::pi * scaleRadius * scaleRadius * scaleRadius *
            detail::nfwMassIntegral(concentration));
    if (!(scaleRadius > 0.0) || !detail::isFinite(scaleRadius) || !detail::isFinite(density)) {
        return {};
    }
    return {scaleRadius, density};
}

inline double klypinScaleRadius(
    double maximumCircularVelocity, double virialRadius, double gravitationalMass) {
    if (!(maximumCircularVelocity > 0.0) || !detail::isFinite(maximumCircularVelocity) ||
        !(virialRadius > 0.0) || !detail::isFinite(virialRadius) || !(gravitationalMass > 0.0) ||
        !detail::isFinite(gravitationalMass)) {
        return 0.0;
    }
    constexpr double kMaximumRadiusRatio = 2.1625815870646098;
    double const velocityRatio =
        maximumCircularVelocity * maximumCircularVelocity * virialRadius / gravitationalMass;
    if (!detail::isFinite(velocityRatio) || velocityRatio < 1.0 - 1.e-10) {
        return 0.0;
    }
    // For c below Rmax/Rs the maximum inside Rvir lies at the halo boundary;
    // Vmax=Vvir cannot distinguish these low concentrations. Retain that limit.
    if (velocityRatio <= 1.0 + 1.e-10) {
        return virialRadius / kMaximumRadiusRatio;
    }
    double const target =
        velocityRatio * kMaximumRadiusRatio / detail::nfwMassIntegral(kMaximumRadiusRatio);
    if (!detail::isFinite(target)) {
        return 0.0;
    }
    auto const ratio = [](double logConcentration) {
        double const concentration = std::exp(logConcentration);
        return concentration / detail::nfwMassIntegral(concentration);
    };
    double lower = std::log(kMaximumRadiusRatio);
    double upper = lower + 1.0;
    while (upper < 709.0 && ratio(upper) < target) {
        upper = std::min(709.0, upper + 1.0);
    }
    if (ratio(upper) < target) {
        return 0.0;
    }
    for (int iteration = 0; iteration < 96 && upper - lower > 1.e-12; ++iteration) {
        double const middle = 0.5 * (lower + upper);
        if (ratio(middle) < target) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    return virialRadius / std::exp(0.5 * (lower + upper));
}

using Matrix3 = std::array<std::array<double, 3>, 3>;

struct SymmetricEigenpairs {
    // Sorted largest to smallest; vectors[axis][component] is a unit eigenvector.
    std::array<double, 3> values{};
    Matrix3 vectors{};
};

inline SymmetricEigenpairs symmetricEigenpairs(Matrix3 const& matrix) {
    double scale = 0.0;
    for (auto const& row : matrix) {
        for (double const value : row) {
            if (!detail::isFinite(value)) {
                throw std::invalid_argument("Shape tensor must be finite");
            }
            scale = std::max(scale, std::abs(value));
        }
    }
    Matrix3 vectors{};
    Matrix3 diagonalized{};
    for (size_t row = 0; row < 3; ++row) {
        vectors[row][row] = 1.0;
        for (size_t column = 0; column < 3; ++column) {
            if (scale > 0.0) {
                diagonalized[row][column] =
                    0.5 * (matrix[row][column] / scale + matrix[column][row] / scale);
            }
        }
    }
    // Jacobi rotations obtain all three orthogonal eigenvectors, including when
    // a coordinate axis is already an eigenvector or eigenvalues are repeated.
    for (int iteration = 0; iteration < 64; ++iteration) {
        size_t first = 0;
        size_t second = 1;
        double largestOffDiagonal = std::abs(diagonalized[first][second]);
        double largestDiagonal = 0.0;
        for (size_t row = 0; row < 3; ++row) {
            largestDiagonal = std::max(largestDiagonal, std::abs(diagonalized[row][row]));
            for (size_t column = row + 1; column < 3; ++column) {
                double const offDiagonal = std::abs(diagonalized[row][column]);
                if (offDiagonal > largestOffDiagonal) {
                    first = row;
                    second = column;
                    largestOffDiagonal = offDiagonal;
                }
            }
        }
        if (largestOffDiagonal <= 8.0 * std::numeric_limits<double>::epsilon() * largestDiagonal) {
            break;
        }
        double const offDiagonal = diagonalized[first][second];
        double const tangentTwice =
            (diagonalized[second][second] - diagonalized[first][first]) / (2.0 * offDiagonal);
        double const tangent = std::copysign(1.0, tangentTwice) /
            (std::abs(tangentTwice) + std::hypot(1.0, tangentTwice));
        double const cosine = 1.0 / std::hypot(1.0, tangent);
        double const sine = tangent * cosine;
        diagonalized[first][first] -= tangent * offDiagonal;
        diagonalized[second][second] += tangent * offDiagonal;
        diagonalized[first][second] = diagonalized[second][first] = 0.0;
        for (size_t axis = 0; axis < 3; ++axis) {
            if (axis != first && axis != second) {
                double const firstValue = diagonalized[axis][first];
                double const secondValue = diagonalized[axis][second];
                diagonalized[axis][first] = diagonalized[first][axis] =
                    cosine * firstValue - sine * secondValue;
                diagonalized[axis][second] = diagonalized[second][axis] =
                    sine * firstValue + cosine * secondValue;
            }
            double const firstVector = vectors[axis][first];
            double const secondVector = vectors[axis][second];
            vectors[axis][first] = cosine * firstVector - sine * secondVector;
            vectors[axis][second] = sine * firstVector + cosine * secondVector;
        }
    }
    std::array<size_t, 3> order{0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        return diagonalized[left][left] > diagonalized[right][right];
    });
    SymmetricEigenpairs result;
    for (size_t axis = 0; axis < 3; ++axis) {
        result.values[axis] = diagonalized[order[axis]][order[axis]] * scale;
        for (size_t component = 0; component < 3; ++component) {
            result.vectors[axis][component] = vectors[component][order[axis]];
        }
    }
    return result;
}

}    // namespace rockstar
