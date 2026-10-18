/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <vector>

#include <cosmictiger/simd_math.hpp>

#include "stf_kernel_test_helpers.hpp"

namespace detail {

using Index = std::array<int, 3>;
using Position = std::array<double, 3>;
using Polynomial = std::map<Index, double>;
using Multipole = Coefficients<double, 36>;
using Expansion = Coefficients<double, 50>;
using Field = Coefficients<double, 4>;

struct Particle {
    double mass;
    Position position;
};

int degree(Index const& index) {
    return index[0] + index[1] + index[2];
}

int compactIndex(Index const& index) {
    int const rank = degree(index);
    return rank * rank + (index[2] == 0 ? index[0] : rank + 1 + index[0]);
}

Index sum(Index left, Index const& right) {
    for (int d = 0; d < 3; ++d) {
        left[d] += right[d];
    }
    return left;
}

double factorial(int n) {
    double result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

double factorial(Index const& index) {
    return factorial(index[0]) * factorial(index[1]) * factorial(index[2]);
}

double monomial(Position const& position, Index const& index) {
    return std::pow(position[0], index[0]) * std::pow(position[1], index[1]) *
        std::pow(position[2], index[2]);
}

std::vector<Index> indices(int rank) {
    std::vector<Index> result;
    for (int x = 0; x <= rank; ++x) {
        for (int y = 0; y <= rank - x; ++y) {
            result.push_back({x, y, rank - x - y});
        }
    }
    return result;
}

Polynomial multiply(Polynomial const& left, Polynomial const& right, int maximum) {
    Polynomial result;
    for (auto const& [a, av] : left) {
        for (auto const& [b, bv] : right) {
            auto const index = sum(a, b);
            if (degree(index) <= maximum) {
                result[index] += av * bv;
            }
        }
    }
    return result;
}

Polynomial laplacian(Polynomial const& polynomial) {
    Polynomial result;
    for (auto const& [index, value] : polynomial) {
        for (int d = 0; d < 3; ++d) {
            if (index[d] >= 2) {
                auto reduced = index;
                reduced[d] -= 2;
                result[reduced] += value * index[d] * (index[d] - 1);
            }
        }
    }
    return result;
}

// Repeated polynomial Laplacians provide an independent STF projection.
Polynomial harmonicProjection(Polynomial const& raw, int rank) {
    Polynomial const radius{{{2, 0, 0}, 1}, {{0, 2, 0}, 1}, {{0, 0, 2}, 1}};
    Polynomial result, trace = raw, radial{{{0, 0, 0}, 1}};
    double coefficient = 1;
    for (int k = 0; k <= rank / 2; ++k) {
        for (auto const& [index, value] : multiply(radial, trace, rank)) {
            result[index] += coefficient * value;
        }
        trace = laplacian(trace);
        radial = multiply(radial, radius, rank);
        coefficient /= -double(2 * (k + 1) * (2 * rank - 2 * k - 1));
    }
    return result;
}

Multipole projectedMoments(std::vector<Particle> const& particles, double scale) {
    Multipole result{};
    for (auto const& particle : particles) {
        Polynomial const linear{{{1, 0, 0}, particle.position[0]},
            {{0, 1, 0}, particle.position[1]}, {{0, 0, 1}, particle.position[2]}};
        Polynomial raw{{{0, 0, 0}, 1}};
        for (int rank = 0; rank <= 5; ++rank) {
            auto const harmonic = harmonicProjection(raw, rank);
            for (auto const& [index, value] : harmonic) {
                if (index[2] <= 1) {
                    result[compactIndex(index)] += particle.mass * value * factorial(index) /
                        factorial(rank) / std::pow(scale, rank);
                }
            }
            raw = multiply(raw, linear, 5);
        }
    }
    return result;
}

// Taylor-expand -1/|R+h| directly with the scalar binomial series. This oracle
// uses full Cartesian coefficients and no generated contractions or recurrences.
Polynomial greenDerivatives(Position const& separation, int order) {
    double const r2 = separation[0] * separation[0] + separation[1] * separation[1] +
        separation[2] * separation[2];
    Polynomial const q{{{1, 0, 0}, 2 * separation[0] / r2}, {{0, 1, 0}, 2 * separation[1] / r2},
        {{0, 0, 1}, 2 * separation[2] / r2}, {{2, 0, 0}, 1 / r2}, {{0, 2, 0}, 1 / r2},
        {{0, 0, 2}, 1 / r2}};
    Polynomial power{{{0, 0, 0}, 1}}, result;
    double binomial = 1;
    for (int k = 0; k <= order; ++k) {
        for (auto const& [index, value] : power) {
            result[index] -= binomial * value / std::sqrt(r2);
        }
        power = multiply(power, q, order);
        binomial *= (-.5 - k) / (k + 1);
    }
    for (auto& [index, value] : result) {
        value *= factorial(index);
    }
    return result;
}

void near(
    double actual, double expected, double scale, char const* label, double tolerance = 3e-11) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(scale, 1e-24)) {
        std::fprintf(stderr, "%s: %.17g != %.17g; difference %.3g, scale %.3g\n", label, actual,
            expected, actual - expected, scale);
        throw std::runtime_error(label);
    }
}

template <int order, typename Actual, typename Expected>
void compareRanks(Actual const& actual, Expected const& expected, char const* label) {
    for (int rank = 0; rank <= order; ++rank) {
        double scale = 0;
        for (int i = rank * rank; i < (rank + 1) * (rank + 1); ++i) {
            scale = std::max(scale, std::abs(expected[i]));
        }
        for (int i = rank * rank; i < (rank + 1) * (rank + 1); ++i) {
            near(actual[i], expected[i], scale, label);
        }
    }
}

std::vector<Particle> particleCloud() {
    return {{.7, {.11, -.09, .17}}, {1.3, {-.21, .13, -.02}}, {.4, {.09, .22, -.18}},
        {.8, {.17, -.16, -.11}}, {1.1, {.03, .05, .12}}};
}

void testParticleMomentsAndTranslation() {
    auto const particles = particleCloud();
    for (auto const scales : {std::array<double, 2>{.6, 1.3}, {1.4, .7}}) {
        auto const expected = projectedMoments(particles, scales[0]);
        Multipole actual{};
        for (auto const& particle : particles) {
            kernels::P2mKernel<5>::add(actual, particle.mass, particle.position[0] / scales[0],
                particle.position[1] / scales[0], particle.position[2] / scales[0]);
        }
        compareRanks<5>(actual, expected, "P2M independent harmonic projection");
        Position const displacement{.13, -.17, .09};
        auto shifted = particles;
        for (auto& particle : shifted) {
            for (int d = 0; d < 3; ++d) {
                particle.position[d] += displacement[d];
            }
        }
        auto const translatedExpected = projectedMoments(shifted, scales[1]);
        Multipole translated{};
        kernels::M2mKernel<5>::add(translated, actual, displacement[0] / scales[1],
            displacement[1] / scales[1], displacement[2] / scales[1], scales[0] / scales[1]);
        compareRanks<5>(translated, translatedExpected, "M2M independent particle re-expansion");
    }
}

template <bool centered>
void testInteractions() {
    // Both modes intentionally receive a nonzero dipole so that omission is tested.
    auto const particles = particleCloud();
    constexpr std::array<Position, 7> kDirections{{{1.3, -.9, 1.7}, {-1.3, .9, -1.7}, {2.1, 0, 0},
        {0, -2.1, 0}, {0, 0, 2.1}, {.001, 2., -.03}, {-1.7, -1.1, .7}}};
    for (int test = 0; test < int(kDirections.size()); ++test) {
        auto const& separation = kDirections[test];
        double const sourceScale = .47 + .13 * test;
        double const targetScale = .63 + .11 * test;
        double const inverseRadius = 1 /
            std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                separation[2] * separation[2]);
        auto const source = projectedMoments(particles, sourceScale);
        auto const derivatives = greenDerivatives(separation, 6);
        Polynomial raw;
        for (int rank = 0; rank <= 5; ++rank) {
            if (centered && rank == 1) {
                continue;
            }
            for (auto const& index : indices(rank)) {
                for (auto const& particle : particles) {
                    raw[index] += particle.mass * monomial(particle.position, index);
                }
            }
        }
        Expansion expected{}, actual{}, pointExpected{}, pointActual{};
        for (int rank = 0; rank <= 6; ++rank) {
            for (auto const& a : indices(rank)) {
                if (a[2] > 1) {
                    continue;
                }
                double value = 0;
                for (auto const& [b, moment] : raw) {
                    if (rank + degree(b) <= 6) {
                        value += (degree(b) % 2 ? -1 : 1) * moment * derivatives.at(sum(a, b)) /
                            factorial(b);
                    }
                }
                expected[compactIndex(a)] = value * std::pow(targetScale, rank);
                pointExpected[compactIndex(a)] =
                    1.7 * derivatives.at(a) * std::pow(targetScale, rank);
            }
        }
        kernels::M2lKernel<6, centered>::template add<double, Expansion, Multipole, 5>(actual,
            source, separation[0] * inverseRadius, separation[1] * inverseRadius,
            separation[2] * inverseRadius, inverseRadius, sourceScale * inverseRadius,
            targetScale * inverseRadius);
        compareRanks<6>(actual, expected, "M2L full Cartesian derivative oracle");
        Expansion ewald{}, storedDerivatives{}, unscaledExpected{};
        auto const unscaledSource = projectedMoments(particles, 1.);
        for (int rank = 0; rank <= 6; ++rank) {
            for (auto const& index : indices(rank)) {
                if (index[2] <= 1) {
                    int const i = compactIndex(index);
                    storedDerivatives[i] = derivatives.at(index);
                    unscaledExpected[i] = expected[i] / std::pow(targetScale, rank);
                }
            }
        }
        kernels::EwaldM2lKernel<6, centered>::template add<double>(
            ewald, unscaledSource, storedDerivatives);
        compareRanks<6>(ewald, unscaledExpected, "Ewald M2L full Cartesian contraction oracle");
        kernels::P2lKernel<6>::add(pointActual, 1.7, separation[0] * inverseRadius,
            separation[1] * inverseRadius, separation[2] * inverseRadius, inverseRadius,
            targetScale * inverseRadius);
        compareRanks<6>(pointActual, pointExpected, "P2L binomial derivative oracle");
        Field field{}, fieldExpected{};
        for (auto const& [b, moment] : raw) {
            double const coefficient = (degree(b) % 2 ? -1 : 1) * moment / factorial(b);
            fieldExpected[0] += coefficient * derivatives.at(b);
            for (int d = 0; d < 3; ++d) {
                auto a = b;
                ++a[d];
                fieldExpected[d + 1] -= coefficient * derivatives.at(a);
            }
        }
        kernels::M2pKernel<5, centered>::add(field, source, separation[0] * inverseRadius,
            separation[1] * inverseRadius, separation[2] * inverseRadius, inverseRadius,
            sourceScale * inverseRadius);
        for (int i = 0; i < 4; ++i) {
            near(field[i], fieldExpected[i], std::abs(fieldExpected[0]),
                "M2P source-truncated Cartesian potential/force oracle");
        }
    }
}

Polynomial potentialPolynomial(double background) {
    Polynomial result, raw{{{0, 0, 0}, 1}};
    Polynomial const linear{{{1, 0, 0}, .31}, {{0, 1, 0}, -.27}, {{0, 0, 1}, .43}};
    for (int rank = 0; rank <= 6; ++rank) {
        for (auto const& [index, value] : harmonicProjection(raw, rank)) {
            result[index] += (rank % 2 ? -.4 : .4) * value / factorial(rank);
        }
        raw = multiply(raw, linear, 6);
    }
    for (int d = 0; d < 3; ++d) {
        Index index{};
        index[d] = 2;
        result[index] += background / 6;
    }
    return result;
}

double polynomialDerivative(
    Polynomial const& polynomial, Index const& derivative, Position const& position) {
    double result = 0;
    for (auto const& [power, value] : polynomial) {
        Index remainder{};
        bool contains = true;
        for (int d = 0; d < 3; ++d) {
            remainder[d] = power[d] - derivative[d];
            contains = contains && remainder[d] >= 0;
        }
        if (contains) {
            result +=
                value * factorial(power) / factorial(remainder) * monomial(position, remainder);
        }
    }
    return result;
}

Expansion localAt(
    Polynomial const& polynomial, Position const& position, double scale, double background) {
    Expansion result{};
    for (int rank = 0; rank <= 6; ++rank) {
        for (auto const& index : indices(rank)) {
            if (index[2] <= 1) {
                result[compactIndex(index)] =
                    polynomialDerivative(polynomial, index, position) * std::pow(scale, rank);
            }
        }
    }
    result[49] = background * scale * scale;
    return result;
}

template <bool withTrace>
void testLocalTranslation() {
    double const background = withTrace ? -1.7 : 0;
    auto const polynomial = potentialPolynomial(background);
    Position const displacement{.13, -.17, .09};
    for (auto const scales : {std::array<double, 2>{.6, 1.3}, {1.4, .7}}) {
        auto const source = localAt(polynomial, {}, scales[0], background);
        auto const expected = localAt(polynomial, displacement, scales[1], background);
        Expansion actual{};
        kernels::L2lKernel<6, withTrace>::add(actual, source, displacement[0] / scales[0],
            displacement[1] / scales[0], displacement[2] / scales[0], scales[1] / scales[0]);
        compareRanks<6>(actual, expected, "L2L direct Cartesian polynomial translation");
        if constexpr (withTrace) {
            near(actual[49], expected[49], std::abs(expected[49]), "L2L background rescaling");
        }
        Field field{};
        kernels::L2pKernel<6, withTrace>::add(field, source, displacement[0] / scales[0],
            displacement[1] / scales[0], displacement[2] / scales[0], 1 / scales[0]);
        near(field[0], polynomialDerivative(polynomial, {}, displacement), 1,
            "L2P direct polynomial potential");
        for (int d = 0; d < 3; ++d) {
            Index derivative{};
            derivative[d] = 1;
            near(field[d + 1], -polynomialDerivative(polynomial, derivative, displacement), 1,
                "L2P direct polynomial acceleration");
        }
    }
}

void testDirectParticleRemainder() {
    for (double q : {.3, .15, .075}) {
        Multipole source{};
        kernels::P2mKernel<5>::add(source, 1., 2 * q, 0., 0.);
        Field field{};
        kernels::M2pKernel<5, false>::add(field, source, 1., 0., 0., .5, .5);
        double const potential = -1 / (2 * (1 - q));
        double const acceleration = -1 / (4 * (1 - q) * (1 - q));
        near(potential - field[0], -std::pow(q, 6) / (2 * (1 - q)), 1,
            "Direct particle exact potential remainder", 3e-14);
        near(acceleration - field[1], -std::pow(q, 6) * (7 - 6 * q) / (4 * (1 - q) * (1 - q)), 1,
            "Direct particle exact acceleration remainder", 3e-14);
        near(field[2], 0, 1, "Direct particle transverse y force");
        near(field[3], 0, 1, "Direct particle transverse z force");
    }
}

template <typename Real, int lanes>
void testSimd() {
    using Pack = SimdVector<Real, lanes>;
    constexpr double kTolerance = sizeof(Real) == sizeof(float) ? 2e-5 : 3e-12;
    for (int batch = 0; batch < 3; ++batch) {
        std::array<KernelInput<Real>, lanes> input{};
        for (int lane = 0; lane < lanes; ++lane) {
            input[lane] = makeKernelInput<Real>(batch * lanes + lane);
        }
        KernelInput<Pack> packed{};
        for (int i = 0; i < 36; ++i) {
            packed.multipole[i] = Pack([&](auto lane) { return input[lane].multipole[i]; });
        }
        for (int i = 0; i < 50; ++i) {
            packed.local[i] = Pack([&](auto lane) { return input[lane].local[i]; });
        }
        for (int i = 0; i < 12; ++i) {
            packed.geometry[i] = Pack([&](auto lane) { return input[lane].geometry[i]; });
        }
        KernelResult<Pack> actual{};
        evaluateKernels(packed, actual);
        for (int lane = 0; lane < lanes; ++lane) {
            KernelResult<Real> expected{};
            evaluateKernels(input[lane], expected);
            for (int i = 0; i < kResultCount; ++i) {
                near(actual[i][lane], expected[i], std::max(1., std::abs(double(expected[i]))),
                    "SIMD/scalar kernel coefficient parity", kTolerance);
            }
        }
    }
}

}    // namespace detail

int main() {
    try {
        detail::testParticleMomentsAndTranslation();
        detail::testInteractions<false>();
        detail::testInteractions<true>();
        detail::testLocalTranslation<false>();
        detail::testLocalTranslation<true>();
        detail::testDirectParticleRemainder();
        detail::testSimd<float, 8>();
        detail::testSimd<double, 2>();
        std::puts("STF scalar, Cartesian reference, background, dipole and SIMD checks passed.");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "STF kernel test failed: %s\n", error.what());
        return 1;
    }
}
