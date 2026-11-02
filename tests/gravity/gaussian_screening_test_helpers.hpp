/* CosmicTiger - independent Gaussian-screening derivative oracle; BSL-1.0. */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include <cosmictiger/gravity/kernels/cartesian_operators.hpp>
#include <cosmictiger/gravity/kernels/gaussian_screening.hpp>
#include <cosmictiger/gravity/p3m/fmm/kernels.hpp>
#include <cosmictiger/gravity/p3m/pp/kernels.hpp>

namespace detail {

using GaussianPosition = std::array<long double, 3>;
inline constexpr int kGaussianFixtureCount = 60;

inline double gaussianDerivativeUnderflowUnits(int rank) {
    // A subnormal normalized radial coefficient rounds before Cartesian
    // reconstruction multiplies it by a pairing count. For example, the
    // rank-seven term r*A_4 has multiplicity 105 in the pure x derivative.
    // Keep ordinary relative tolerances unchanged and account only for this
    // absolute underflow amplification in the smallest float coefficients.
    double multiplicity = 1;
    double maximum = 32;
    for (int traces = 1; traces <= rank / 2; ++traces) {
        multiplicity *= double((rank - 2 * traces + 2) * (rank - 2 * traces + 1)) / (2 * traces);
        maximum = std::max(maximum, multiplicity);
    }
    return maximum;
}

struct GaussianQuadraturePoint {
    long double position;
    long double weight;
};

inline std::vector<GaussianQuadraturePoint> const& gaussianQuadrature() {
    static auto const points = [] {
        constexpr int kCount = 128;
        std::vector<GaussianQuadraturePoint> result(kCount);
        for (int i = 0; i < kCount / 2; ++i) {
            long double root =
                std::cos(std::numbers::pi_v<long double> * (i + 0.75L) / (kCount + 0.5L));
            long double derivative = 0;
            for (int iteration = 0; iteration < 32; ++iteration) {
                long double previous = 1;
                long double current = root;
                for (int n = 2; n <= kCount; ++n) {
                    long double const next =
                        ((2 * n - 1) * root * current - (n - 1) * previous) / n;
                    previous = current;
                    current = next;
                }
                derivative = kCount * (root * current - previous) / (root * root - 1);
                long double const correction = current / derivative;
                root -= correction;
                if (std::abs(correction) < 2.0e-20L) {
                    break;
                }
            }
            long double const weight = 2 / ((1 - root * root) * derivative * derivative);
            result[i] = {-root, weight};
            result[kCount - i - 1] = {root, weight};
        }
        return result;
    }();
    return points;
}

// Independent Cartesian oracle: differentiate exp(-s^2 |x|^2) as a product
// of three one-dimensional Hermite polynomials, then numerically integrate
// -2/sqrt(pi) ds over [0, alpha] (long) or [alpha, infinity) (short).
// This never uses the implementation's radial derivative recurrence. The
// short integral uses u=s*r and ends at alpha*r+12; its omitted Gaussian tail
// is negligible compared with double precision throughout these fixtures.
template <int order>
std::array<long double, cosmictiger::gravity::cartesian::coefficientCount(order - 1)>
gaussianDerivativeOracle(GaussianPosition const& position, long double alpha, bool longRange) {
    constexpr int kDegree = order - 1;
    constexpr int kCount = cosmictiger::gravity::cartesian::coefficientCount(kDegree);
    std::array<long double, kCount> result{};
    long double const radius = std::sqrt(
        position[0] * position[0] + position[1] * position[1] + position[2] * position[2]);
    long double const lower = longRange ? 0 : alpha * radius;
    long double const upper = longRange ? 1 : lower + 12;
    long double const halfWidth = (upper - lower) / 2;
    long double const center = (upper + lower) / 2;
    for (auto const& point : gaussianQuadrature()) {
        long double const coordinate = center + halfWidth * point.position;
        long double const s = longRange ? alpha * coordinate : coordinate / radius;
        long double const measure = longRange ? alpha : 1 / radius;
        long double const gaussian = std::exp(-s * s * radius * radius);
        long double const weight = -2 / std::sqrt(std::numbers::pi_v<long double>) * point.weight *
            halfWidth * measure * gaussian;
        std::array<std::array<long double, kDegree + 1>, 3> hermite{};
        for (int axis = 0; axis < 3; ++axis) {
            long double const argument = s * position[axis];
            hermite[axis][0] = 1;
            hermite[axis][1] = 2 * argument;
            for (int degree = 2; degree <= kDegree; ++degree) {
                hermite[axis][degree] = 2 * argument * hermite[axis][degree - 1] -
                    2 * (degree - 1) * hermite[axis][degree - 2];
            }
        }
        long double signedPower = 1;
        int index = 0;
        for (int degree = 0; degree <= kDegree; ++degree) {
            for (int x = 0; x <= degree; ++x) {
                for (int y = 0; y <= degree - x; ++y) {
                    int const z = degree - x - y;
                    result[index++] +=
                        weight * signedPower * hermite[0][x] * hermite[1][y] * hermite[2][z];
                }
            }
            signedPower *= -s;
        }
    }
    return result;
}

inline std::array<long double, 4> gaussianPairOracle(
    GaussianPosition const& position, long double alpha) {
    long double const radiusSquared =
        position[0] * position[0] + position[1] * position[1] + position[2] * position[2];
    long double const radius = std::sqrt(radiusSquared);
    long double const z = alpha * radius;
    long double const complement = std::erfc(z);
    long double const forceFactor =
        (complement + 2 * z * std::exp(-z * z) / std::sqrt(std::numbers::pi_v<long double>)) /
        (radiusSquared * radius);
    return {-complement / radius, -forceFactor * position[0], -forceFactor * position[1],
        -forceFactor * position[2]};
}

inline std::array<double, 3> gaussianFixturePosition(double alpha, int sample) {
    constexpr std::array<double, 12> kRadii{
        0.03, 0.1, 0.4, 1.2, 2.7, 2.99, 3.0, 3.01, 3.2, 4.7, 6.25, 8.0};
    double const radius = kRadii[sample % kRadii.size()] / alpha;
    if (sample % 3 == 0) {
        return {radius, 0, 0};
    }
    return {0.36 * radius, -0.48 * radius, 0.8 * radius};
}

template <class Real>
struct GaussianInput {
    Real alpha;
    Vector3<Real> position;
};

template <class Real, int order>
struct GaussianResult {
    cosmictiger::gravity::cartesian::Green<Real, order> longRange;
    cosmictiger::gravity::cartesian::Green<Real, order> shortRange;
    cosmictiger::gravity::Field<Real> shortPair;
};

template <class Real, int order>
CUDA_EXPORT void evaluateGaussianFixture(
    GaussianInput<Real> const& input, GaussianResult<Real, order>& result) {
    using cosmictiger::gravity::GaussianComponent;
    cosmictiger::gravity::GaussianScreening<SimdScalar<Real>> const screening(input.alpha);
    cosmictiger::gravity::cartesian::evaluateGreen<GaussianComponent::Long>(
        result.longRange, input.position, screening);
    cosmictiger::gravity::p3m::fmm::evaluateGreen(result.shortRange, input.position, screening);
    result.shortPair = cosmictiger::gravity::p3m::pp::evaluateField(input.position, screening);
}

template <class Real>
GaussianInput<Real> gaussianFixture(int sample) {
    constexpr std::array<double, 5> kAlphas{0.002, 0.125, 0.7, 2.75, 6.0};
    double const alpha = kAlphas[(sample / 12) % kAlphas.size()];
    auto const position = gaussianFixturePosition(alpha, sample % 12);
    return {Real(alpha), {Real(position[0]), Real(position[1]), Real(position[2])}};
}

}    // namespace detail
