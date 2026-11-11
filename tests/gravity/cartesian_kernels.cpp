/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include "cartesian_kernel_test_helpers.hpp"

namespace detail {

using Index = std::array<int, 3>;
using Position = Vector3<double>;
using Polynomial = std::map<Index, double>;
using Field = cosmictiger::gravity::Field<double>;
template <int order>
using Multipole = cosmictiger::gravity::cartesian::Multipole<double>;
template <int order>
using Expansion = cosmictiger::gravity::cartesian::Expansion<double>;
template <int order>
using Green = cosmictiger::gravity::cartesian::Green<double>;

// The Cartesian oracles below carry full (non-compact) coefficients, so their
// rounding grows with the rank through the harmonic projection's alternating
// trace sums. The budget is relative to the largest coefficient of each rank.
// Measured worst normalized discrepancies per FMM order (multipole degree p,
// local degree p + 1) over the Cartesian, binomial, polynomial and remainder
// oracles with this file's fixtures (GCC 13, -O3 -ffast-math -march=native):
//   order  3: 6.9e-16    order  4: 1.0e-15    order  5: 1.3e-15    order  6: 3.3e-15
//   order  7: 4.6e-15    order  8: 1.2e-14    order  9: 2.5e-14
// (order 10, since dropped from the supported range, measured 5.7e-14.)
// The SIMD/scalar parity worst cases were 5.1e-7 (float, order 9; budget 2e-5)
// and 2.5e-16 (double, order 9; budget 3e-12). The oracle discrepancy roughly
// doubles per order, so one budget covers every supported order.
constexpr double kOracleTolerance = 3e-11;

// Worst normalized discrepancy seen per check label, reset for every order and
// reported with that order's pass line.
std::map<std::string, double> worstDiscrepancy;

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

// Repeated polynomial Laplacians provide an independent Cartesian projection.
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

template <int order, int sourceDegree = KernelLayout<order>::kMultipoleDegree_>
Multipole<order> projectedMoments(std::vector<Particle> const& particles) {
    constexpr int kMultipoleDegree = sourceDegree;
    Multipole<order> result{};
    for (auto const& particle : particles) {
        Polynomial const linear{{{1, 0, 0}, particle.position[0]},
            {{0, 1, 0}, particle.position[1]}, {{0, 0, 1}, particle.position[2]}};
        Polynomial raw{{{0, 0, 0}, 1}};
        for (int rank = 0; rank <= kMultipoleDegree; ++rank) {
            auto const harmonic = harmonicProjection(raw, rank);
            for (auto const& [index, value] : harmonic) {
                if (index[2] <= 1) {
                    result[compactIndex(index)] +=
                        particle.mass * value * factorial(index) / factorial(rank);
                }
            }
            raw = multiply(raw, linear, kMultipoleDegree);
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

void near(double actual, double expected, double scale, char const* label,
    double tolerance = kOracleTolerance) {
    double const normalized = std::abs(actual - expected) / std::max(scale, 1e-24);
    auto& worst = worstDiscrepancy[label];
    worst = std::isfinite(actual) ? std::max(worst, normalized) : INFINITY;
    if (!std::isfinite(actual) || normalized > tolerance) {
        std::fprintf(stderr, "%s: %.17g != %.17g; difference %.3g, scale %.3g\n", label, actual,
            expected, actual - expected, scale);
        throw std::runtime_error(label);
    }
}

template <int degree, typename Actual, typename Expected>
void compareRanks(Actual const& actual, Expected const& expected, char const* label) {
    for (int rank = 0; rank <= degree; ++rank) {
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

template <int order>
void testParticleMomentsAndTranslation() {
    constexpr int kMultipoleDegree = KernelLayout<order>::kMultipoleDegree_;
    auto const particles = particleCloud();
    auto const expected = projectedMoments<order>(particles);
    Multipole<order> Mi{};
    for (auto const& particle : particles) {
        kernels::P2M(particle.mass, particle.position, Mi);
    }
    compareRanks<kMultipoleDegree>(Mi, expected, "P2M independent harmonic projection");
    Position const displacement{.13, -.17, .09};
    auto shifted = particles;
    for (auto& particle : shifted) {
        for (int d = 0; d < 3; ++d) {
            particle.position[d] += displacement[d];
        }
    }
    auto const translatedExpected = projectedMoments<order>(shifted);
    Multipole<order> Mo{};
    kernels::M2M(Mi, displacement, 1.0, Mo);
    compareRanks<kMultipoleDegree>(Mo, translatedExpected, "M2M independent particle re-expansion");
}

template <int order>
void testInteractions() {
    constexpr int kMultipoleDegree = KernelLayout<order>::kMultipoleDegree_;
    constexpr int kLocalDegree = KernelLayout<order>::kLocalDegree_;
    // The source has a nonzero dipole, which the general kernels must retain.
    auto const particles = particleCloud();
    constexpr std::array<Position, 7> kDirections{{{1.3, -.9, 1.7}, {-1.3, .9, -1.7}, {2.1, 0, 0},
        {0, -2.1, 0}, {0, 0, 2.1}, {.001, 2., -.03}, {-1.7, -1.1, .7}}};
    for (int test = 0; test < int(kDirections.size()); ++test) {
        auto const& separation = kDirections[test];
        double const iR = 1 /
            std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                separation[2] * separation[2]);
        auto const source = projectedMoments<order, kMultipoleDegree>(particles);
        auto const derivatives = greenDerivatives(separation, kLocalDegree);
        Polynomial raw;
        for (int rank = 0; rank <= kMultipoleDegree; ++rank) {
            for (auto const& index : indices(rank)) {
                for (auto const& particle : particles) {
                    raw[index] += particle.mass * monomial(particle.position, index);
                }
            }
        }
        // Source rank n and local rank k interact only while n + k stays within
        // the local degree.
        Expansion<order> expected{}, actual{}, pointExpected{}, pointActual{};
        for (int rank = 0; rank <= kLocalDegree; ++rank) {
            for (auto const& a : indices(rank)) {
                if (a[2] > 1) {
                    continue;
                }
                double value = 0;
                for (auto const& [b, moment] : raw) {
                    if (rank + degree(b) <= kLocalDegree) {
                        value += (degree(b) % 2 ? -1 : 1) * moment * derivatives.at(sum(a, b)) /
                            factorial(b);
                    }
                }
                expected[compactIndex(a)] = value;
                pointExpected[compactIndex(a)] = 1.7 * derivatives.at(a);
            }
        }
        if constexpr (cosmictiger::gravity::cartesian::kIndependentM2lRatios) {
            kernels::M2L(source, separation, iR, iR, actual);
        } else {
            kernels::M2L(source, separation, actual);
        }
        compareRanks<kLocalDegree>(actual, expected, "M2L unit Cartesian derivative oracle");
        {
            auto const unscaledSource = projectedMoments<order>(particles);
            kernels::P2L(1.7, separation, pointActual);
            compareRanks<kLocalDegree>(
                pointActual, pointExpected, "P2L binomial derivative oracle");
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
            kernels::M2P(unscaledSource, separation, field);
            for (int i = 0; i < kFieldCount; ++i) {
                near(field[i], fieldExpected[i], std::abs(fieldExpected[0]),
                    "M2P source-truncated Cartesian potential/force oracle");
            }
        }
    }
}

Polynomial potentialPolynomial(int localDegree, double background) {
    Polynomial result, raw{{{0, 0, 0}, 1}};
    Polynomial const linear{{{1, 0, 0}, .31}, {{0, 1, 0}, -.27}, {{0, 0, 1}, .43}};
    for (int rank = 0; rank <= localDegree; ++rank) {
        for (auto const& [index, value] : harmonicProjection(raw, rank)) {
            result[index] += (rank % 2 ? -.4 : .4) * value / factorial(rank);
        }
        raw = multiply(raw, linear, localDegree);
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

template <int order>
Expansion<order> localAt(Polynomial const& polynomial, Position const& position) {
    constexpr int kLocalDegree = KernelLayout<order>::kLocalDegree_;
    Expansion<order> result{};
    for (int rank = 0; rank <= kLocalDegree; ++rank) {
        for (auto const& index : indices(rank)) {
            if (index[2] <= 1) {
                result[compactIndex(index)] = polynomialDerivative(polynomial, index, position);
            }
        }
    }
    return result;
}

template <int order>
void testLocalTranslation() {
    constexpr int kLocalDegree = KernelLayout<order>::kLocalDegree_;
    double const background = 0;
    auto const polynomial = potentialPolynomial(kLocalDegree, background);
    Position const displacement{.13, -.17, .09};
    auto const Li = localAt<order>(polynomial, {});
    auto const expected = localAt<order>(polynomial, displacement);
    Expansion<order> Lo{};
    kernels::L2L(Li, displacement, 1.0, Lo);
    compareRanks<kLocalDegree>(Lo, expected, "L2L direct Cartesian polynomial translation");
    Field field{};
    kernels::L2P(Li, displacement, field);
    near(field[0], polynomialDerivative(polynomial, {}, displacement), 1,
        "L2P direct polynomial potential");
    for (int d = 0; d < 3; ++d) {
        Index derivative{};
        derivative[d] = 1;
        near(field[d + 1], -polynomialDerivative(polynomial, derivative, displacement), 1,
            "L2P direct polynomial acceleration");
    }
}

// A unit mass at x = 2q on the axis, expanded about the origin and evaluated at
// x = 2: the axial rank-n multipole term contributes -(1/2) q^n to the potential
// and -(n + 1) q^n / 4 to the axial acceleration. Truncating the source at the
// multipole degree p therefore leaves the tails of both series beyond rank p.
struct AxialRemainder {
    double potential;
    double acceleration;
};

AxialRemainder closedFormRemainder(int multipoleDegree, double q) {
    double const tail = std::pow(q, multipoleDegree + 1);
    AxialRemainder remainder;
    remainder.potential = -tail / (2 * (1 - q));
    remainder.acceleration =
        -tail * ((multipoleDegree + 2) - (multipoleDegree + 1) * q) / (4 * (1 - q) * (1 - q));
    return remainder;
}

AxialRemainder bruteForceRemainder(int multipoleDegree, double q) {
    AxialRemainder remainder{};
    double power = std::pow(q, multipoleDegree + 1);
    for (int n = multipoleDegree + 1; power > 1e-40; ++n) {
        remainder.potential -= power / 2;
        remainder.acceleration -= (n + 1) * power / 4;
        power *= q;
    }
    return remainder;
}

template <int order>
void testDirectParticleRemainder() {
    constexpr int kMultipoleDegree = KernelLayout<order>::kMultipoleDegree_;
    for (double q : {.3, .15, .075}) {
        auto const closedForm = closedFormRemainder(kMultipoleDegree, q);
        auto const series = bruteForceRemainder(kMultipoleDegree, q);
        near(closedForm.potential, series.potential, std::abs(series.potential),
            "Direct particle potential remainder closed form", 1e-12);
        near(closedForm.acceleration, series.acceleration, std::abs(series.acceleration),
            "Direct particle acceleration remainder closed form", 1e-12);
        Multipole<order> source{};
        kernels::P2M(1., Position{2 * q, 0., 0.}, source);
        Field field{};
        kernels::M2P(source, Position{2., 0., 0.}, field);
        double const potential = -1 / (2 * (1 - q));
        double const acceleration = -1 / (4 * (1 - q) * (1 - q));
        near(potential - field[0], series.potential, 1, "Direct particle exact potential remainder",
            3e-14);
        near(acceleration - field[1], series.acceleration, 1,
            "Direct particle exact acceleration remainder", 3e-14);
        near(field[2], 0, 1, "Direct particle transverse y force");
        near(field[3], 0, 1, "Direct particle transverse z force");
    }
}

template <class Coefficient, typename Real, int lanes>
void testSimdCoefficientCopies() {
    using Pack = SimdVector<Real, lanes>;
    Coefficient source{};
    for (int i = 0; i < source.size(); ++i) {
        source[i] = Pack([&](auto lane) { return Real(i * lanes + int(lane) + 1); });
    }
    Coefficient copied(source);
    Coefficient assigned{};
    assigned = source;
    Coefficient moved(std::move(copied));
    // The gravity gather grows vectors of SIMD multipoles; force a relocation.
    std::vector<Coefficient> gathered;
    gathered.reserve(1);
    gathered.push_back(source);
    gathered.push_back(std::move(assigned));
    gathered[0] = source;
    for (int i = 0; i < source.size(); ++i) {
        for (int lane = 0; lane < lanes; ++lane) {
            EXPECT_EQ(Real(moved[i][lane]), Real(source[i][lane]));
            EXPECT_EQ(Real(gathered[0][i][lane]), Real(source[i][lane]));
            EXPECT_EQ(Real(gathered[1][i][lane]), Real(source[i][lane]));
        }
    }
}

template <int order, typename Real, int lanes>
void testSimd() {
    using Pack = SimdVector<Real, lanes>;
    using Layout = KernelLayout<order>;
    namespace cartesian = cosmictiger::gravity::cartesian;
    static_assert(std::is_trivially_copyable_v<cartesian::Multipole<Real>>);
    static_assert(std::is_trivially_copyable_v<cartesian::Expansion<Real>>);
    static_assert(std::is_trivially_copyable_v<cartesian::Green<Real>>);
    testSimdCoefficientCopies<cartesian::Multipole<Pack>, Real, lanes>();
    testSimdCoefficientCopies<cartesian::Expansion<Pack>, Real, lanes>();
    testSimdCoefficientCopies<cartesian::Green<Pack>, Real, lanes>();
    constexpr double kTolerance = sizeof(Real) == sizeof(float) ? 2e-5 : 3e-12;
    for (int batch = 0; batch < 3; ++batch) {
        std::array<KernelInput<Real, order>, lanes> input{};
        for (int lane = 0; lane < lanes; ++lane) {
            input[lane] = makeKernelInput<Real, order>(batch * lanes + lane);
        }
        KernelInput<Pack, order> packed{};
        for (int i = 0; i < Layout::kMultipoleCount_; ++i) {
            packed.multipole[i] = Pack([&](auto lane) { return input[lane].multipole[i]; });
        }
        for (int i = 0; i < Layout::kLocalCount_; ++i) {
            packed.local[i] = Pack([&](auto lane) { return input[lane].local[i]; });
        }
        for (int i = 0; i < kGeometryCount; ++i) {
            packed.geometry[i] = Pack([&](auto lane) { return input[lane].geometry[i]; });
        }
        KernelResult<Pack, order> actual{};
        evaluateKernels(packed, actual);
        for (int lane = 0; lane < lanes; ++lane) {
            KernelResult<Real, order> expected{};
            evaluateKernels(input[lane], expected);
            for (int i = 0; i < Layout::kResultCount_; ++i) {
                near(actual[i][lane], expected[i], std::max(1., std::abs(double(expected[i]))),
                    sizeof(Real) == sizeof(float) ? "SIMD/scalar float kernel coefficient parity" :
                                                    "SIMD/scalar double kernel coefficient parity",
                    kTolerance);
            }
        }
    }
}

template <int order>
void testOrder() {
    worstDiscrepancy.clear();
    testParticleMomentsAndTranslation<order>();
    testInteractions<order>();
    testLocalTranslation<order>();
    testDirectParticleRemainder<order>();
    testSimd<order, float, 8>();
    testSimd<order, double, 2>();
    double worstOracle = 0;
    double worstSimdFloat = 0;
    double worstSimdDouble = 0;
    for (auto const& [label, value] : worstDiscrepancy) {
        if (label.rfind("SIMD/scalar float", 0) == 0) {
            worstSimdFloat = std::max(worstSimdFloat, value);
        } else if (label.rfind("SIMD/scalar double", 0) == 0) {
            worstSimdDouble = std::max(worstSimdDouble, value);
        } else {
            worstOracle = std::max(worstOracle, value);
        }
    }
    std::printf("Cartesian order %d (degree %d/%d) scalar, Cartesian reference, "
                "dipole and "
                "SIMD checks passed; worst normalized discrepancy: oracles %.1e, SIMD float %.1e, "
                "SIMD double %.1e.\n",
        order, KernelLayout<order>::kMultipoleDegree_, KernelLayout<order>::kLocalDegree_,
        worstOracle, worstSimdFloat, worstSimdDouble);
}

}    // namespace detail

class CartesianKernels : public ::testing::TestWithParam<int> {};

TEST_P(CartesianKernels, SupportedOrder) {
    detail::forEachFmmOrder([&](auto order) {
        if (order.value == GetParam()) {
            detail::testOrder<order.value>();
        }
    });
}

INSTANTIATE_TEST_SUITE_P(AllOrders, CartesianKernels, ::testing::Range(kFmmOrder, kFmmOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
