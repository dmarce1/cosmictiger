/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/kernels/cartesian_operators.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include "cartesian_symmetric_test_helpers.hpp"

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace detail {
namespace cs = cosmictiger::gravity::cartesian;
using Index = std::array<int, 3>;
using Position = Vector3<double>;
using Polynomial = std::map<Index, double>;
using Field = cosmictiger::gravity::Field<double>;

template <class Target, class Source, class Kernel>
concept AcceptsM2L = requires(
    Target& target, Source const& source, Kernel const& kernel) { M2L(target, source, kernel); };

using FullMultipole = cs::Multipole<double, 5>;
using FullExpansion = cs::Expansion<double, 5>;
using FullGreen = cs::Green<double, 5>;
using TracefreeMultipole = cs::tracefree::Multipole<double, 5>;
using TracefreeExpansion = cs::tracefree::Expansion<double, 5>;
using TracefreeGreen = cs::tracefree::Green<double, 5>;

template <class Argument>
concept AcceptsP2LCoordinateArgument = requires(FullExpansion& target, Position const& displacement,
    Argument argument) { P2L(target, 1.0, displacement, argument); };

template <class Argument>
concept AcceptsM2LCoordinateArgument =
    requires(FullExpansion& target, FullMultipole const& source, Position const& displacement,
        Argument argument) { M2L(target, source, displacement, argument); };

template <class Argument>
concept AcceptsM2PCoordinateArgument =
    requires(Field& target, FullMultipole const& source, Position const& displacement,
        Argument argument) { M2P(target, source, displacement, argument); };

static_assert(AcceptsP2LCoordinateArgument<double> && AcceptsP2LCoordinateArgument<bool>);
static_assert(AcceptsM2LCoordinateArgument<double> && AcceptsM2LCoordinateArgument<bool>);
static_assert(AcceptsM2PCoordinateArgument<double> && AcceptsM2PCoordinateArgument<bool>);
static_assert(!AcceptsP2LCoordinateArgument<float> && !AcceptsP2LCoordinateArgument<int>);
static_assert(!AcceptsM2LCoordinateArgument<float> && !AcceptsM2LCoordinateArgument<int>);
static_assert(!AcceptsM2PCoordinateArgument<float> && !AcceptsM2PCoordinateArgument<int>);
static_assert(requires(
    FullExpansion& local, Field& field, FullMultipole const& source, Position const& displacement) {
    P2L(local, 1.0, displacement);
    M2L(local, source, displacement);
    M2P(field, source, displacement);
});

static_assert(!std::is_same_v<FullMultipole, FullExpansion>);
static_assert(!std::is_same_v<FullExpansion, FullGreen>);
static_assert(!std::is_convertible_v<FullGreen, FullExpansion>);
static_assert(!std::is_convertible_v<FullExpansion, FullGreen>);
static_assert(!std::is_assignable_v<FullExpansion&, FullGreen const&>);
static_assert(!std::is_assignable_v<FullGreen&, FullExpansion const&>);
static_assert(!std::is_convertible_v<FullMultipole, TracefreeMultipole>);
static_assert(!std::is_convertible_v<TracefreeMultipole, FullMultipole>);
static_assert(!std::is_convertible_v<FullExpansion, TracefreeExpansion>);
static_assert(!std::is_convertible_v<TracefreeExpansion, FullExpansion>);
static_assert(!std::is_convertible_v<FullGreen, TracefreeGreen>);
static_assert(!std::is_convertible_v<TracefreeGreen, FullGreen>);
static_assert(AcceptsM2L<FullExpansion, FullMultipole, FullGreen>);
static_assert(AcceptsM2L<TracefreeExpansion, TracefreeMultipole, TracefreeGreen>);
static_assert(!AcceptsM2L<FullExpansion, TracefreeMultipole, FullGreen>);
static_assert(!AcceptsM2L<TracefreeExpansion, FullMultipole, TracefreeGreen>);
static_assert(!AcceptsM2L<FullExpansion, FullMultipole, TracefreeGreen>);
static_assert(!AcceptsM2L<TracefreeExpansion, TracefreeMultipole, FullGreen>);
static_assert(!AcceptsM2L<FullExpansion, FullMultipole, FullExpansion>);
static_assert(!AcceptsM2L<FullGreen, FullMultipole, FullGreen>);
static_assert(!AcceptsM2L<FullExpansion, cs::Multipole<double, 4>, FullGreen>);

double factorial(int n) {
    double result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

double factorial(Index const& a) {
    return factorial(a[0]) * factorial(a[1]) * factorial(a[2]);
}

int degree(Index const& a) {
    return a[0] + a[1] + a[2];
}

std::vector<Index> indices(int maximum) {
    std::vector<Index> result;
    for (int n = 0; n <= maximum; ++n) {
        for (int x = 0; x <= n; ++x) {
            for (int y = 0; y <= n - x; ++y) {
                result.push_back({x, y, n - x - y});
            }
        }
    }
    return result;
}

Index sum(Index a, Index const& b) {
    for (int i = 0; i < 3; ++i) {
        a[i] += b[i];
    }
    return a;
}

Position sum(Position a, Position const& b) {
    for (int i = 0; i < 3; ++i) {
        a[i] += b[i];
    }
    return a;
}

double monomial(Position const& p, Index const& a) {
    return std::pow(p[0], a[0]) * std::pow(p[1], a[1]) * std::pow(p[2], a[2]);
}

void check(double actual, double expected, std::string const& label, double scale = 1,
    double tolerance = 3e-11) {
    bool const finite =
        (std::bit_cast<uint64_t>(actual) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    if (!finite || std::abs(actual - expected) > tolerance * std::max(scale, std::abs(expected))) {
        std::fprintf(
            stderr, "%s: %.17g != %.17g (scale %.17g)\n", label.c_str(), actual, expected, scale);
        throw std::runtime_error(label);
    }
}

template <class Target, class Operation>
void checkAccumulation(Target const& increment, Operation&& operation, std::string const& label,
    bool doPotential = true) {
    constexpr int kCount = [] {
        if constexpr (requires { Target::kSize_; }) {
            return Target::kSize_;
        } else {
            return 4;
        }
    }();
    Target target{};
    for (int i = 0; i < kCount; ++i) {
        target[i] = 0.73 + 0.013 * i;
    }
    operation(target);
    for (int i = 0; i < kCount; ++i) {
        double const seed = 0.73 + 0.013 * i;
        check(target[i], seed + (i != 0 || doPotential ? increment[i] : 0.0), label);
    }
}

Polynomial multiply(Polynomial const& left, Polynomial const& right, int maximum) {
    Polynomial result;
    for (auto const& [a, av] : left) {
        for (auto const& [b, bv] : right) {
            Index const c = sum(a, b);
            if (degree(c) <= maximum) {
                result[c] += av * bv;
            }
        }
    }
    return result;
}

// Independent multivariate binomial series around r. This never differentiates
// the radial analytic formula used by the generator.
Polynomial plummerTaylor(Position const& r, double epsilonSquared, int maximum) {
    double const q = epsilonSquared + r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    Polynomial perturbation;
    for (int axis = 0; axis < 3; ++axis) {
        Index a{};
        a[axis] = 1;
        perturbation[a] = 2 * r[axis] / q;
        a[axis] = 2;
        perturbation[a] = 1 / q;
    }
    Polynomial power{{Index{}, 1}};
    Polynomial result;
    double coefficient = -1 / std::sqrt(q);
    for (int n = 0; n <= maximum; ++n) {
        for (auto const& [a, value] : power) {
            result[a] += coefficient * value;
        }
        power = multiply(power, perturbation, maximum);
        coefficient *= (-0.5 - n) / (n + 1);
    }
    return result;
}

template <int localDegree>
void testOrder() {
    constexpr int kSourceDegree = localDegree - 1;
    using Source = cs::Multipole<double, localDegree + 1>;
    using Local = cs::Expansion<double, localDegree + 1>;
    using Green = cs::Green<double, localDegree + 1>;
    std::vector<Index> const all = indices(localDegree);
    for (int i = 0; i < int(all.size()); ++i) {
        auto const& a = all[i];
        check(Local::index(a[0], a[1], a[2]), i, "packed index");
    }
    check(all.size(), Local::kCount_, "coefficient count");

    std::array<Position, 4> const particles{
        {{0.012, -0.014, 0.018}, {-0.009, 0.015, -0.008}, {0.017, 0.011, -0.013}, {0, 0, 0}}};
    std::array<double, 4> const masses{{0.4, 0.7, 0.2, 0.11}};
    Position const shift{0.08, -0.06, 0.03};
    Source moments{};
    for (int p = 0; p < int(particles.size()); ++p) {
        P2M(moments, masses[p], particles[p]);
    }
    Source translated{};
    M2M(translated, moments, shift);
    for (auto const& a : indices(kSourceDegree)) {
        double moment = 0;
        double shifted = 0;
        for (int p = 0; p < int(particles.size()); ++p) {
            moment += masses[p] * monomial(particles[p], a);
            shifted += masses[p] * monomial(sum(particles[p], shift), a);
        }
        check(moments(a[0], a[1], a[2]), moment, "P2M raw moments", 1.0e-20);
        check(translated(a[0], a[1], a[2]), shifted, "M2M translated point moments", 1.0e-20);
    }
    Source roundTrip{};
    M2M(roundTrip, translated, Position{-shift[0], -shift[1], -shift[2]});
    for (int i = 0; i < Source::kCount_; ++i) {
        check(roundTrip[i], moments[i], "M2M inverse shift");
    }
    checkAccumulation(
        moments,
        [&](auto& target) {
            for (size_t p = 0; p < particles.size(); ++p) {
                P2M(target, masses[p], particles[p]);
            }
        },
        "additive P2M");
    checkAccumulation(
        translated, [&](auto& target) { M2M(target, moments, shift); }, "additive M2M");

    Local polynomial{};
    for (int i = 0; i < Local::kCount_; ++i) {
        polynomial[i] = std::sin(double(i + 1)) / (i + 1);
    }
    Local moved{};
    L2L(moved, polynomial, shift);
    for (auto const& a : all) {
        double expected = 0;
        for (auto const& b : indices(localDegree - degree(a))) {
            Index const c = sum(a, b);
            expected += polynomial(c[0], c[1], c[2]) * monomial(shift, b) / factorial(b);
        }
        check(moved(a[0], a[1], a[2]), expected, "L2L arbitrary nonharmonic polynomial");
    }
    Position const evaluation{0.05, 0.02, -0.04};
    Position const oldEvaluation = sum(evaluation, shift);
    Field field{};
    L2P(field, moved, evaluation);
    for (int component = 0; component < 4; ++component) {
        Index a{};
        if (component != 0) {
            a[component - 1] = 1;
        }
        double expected = 0;
        for (auto const& b : indices(localDegree - degree(a))) {
            Index const c = sum(a, b);
            expected += polynomial(c[0], c[1], c[2]) * monomial(oldEvaluation, b) / factorial(b);
        }
        check(field[component], (component == 0 ? 1 : -1) * expected, "L2P translated polynomial");
    }
    for (bool doPotential : {false, true}) {
        checkAccumulation(
            moved, [&](auto& target) { L2L(target, polynomial, shift, doPotential); },
            "additive L2L and potential preservation", doPotential);
        checkAccumulation(
            field, [&](auto& target) { L2P(target, moved, evaluation, doPotential); },
            "additive L2P and potential preservation", doPotential);
    }

    std::array<Position, 5> const locations{{{0.73, -0.37, 0.91}, {0, 0, 0}, {0, 0, -0.8},
        {1e-12, -2e-12, 3e-12}, {-0.71, 0.38, -0.67}}};
    for (int location = 0; location < int(locations.size()); ++location) {
        Position const r = locations[location];
        double const epsilonSquared = location == 4 ? 0.0 : 0.31;
        Green derivatives{};
        cs::evaluateGreen(derivatives, r, epsilonSquared);
        Polynomial const reference = plummerTaylor(r, epsilonSquared, localDegree);
        std::array<double, localDegree + 1> scales{};
        for (auto const& a : all) {
            scales[degree(a)] =
                std::max(scales[degree(a)], std::abs(reference.at(a) * factorial(a)));
        }
        for (auto const& a : all) {
            check(derivatives(a[0], a[1], a[2]), reference.at(a) * factorial(a),
                "Plummer derivatives independent Taylor series", scales[degree(a)]);
        }
        double const q = epsilonSquared + r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        check(derivatives(2, 0, 0) + derivatives(0, 2, 0) + derivatives(0, 0, 2),
            3 * epsilonSquared / std::pow(q, 2.5), "nonzero Plummer Laplacian", scales[2]);

        Local local{};
        M2L(local, moments, derivatives);
        for (auto const& a : all) {
            double expected = 0;
            for (int p = 0; p < int(particles.size()); ++p) {
                for (auto const& b : indices(std::min(kSourceDegree, localDegree - degree(a)))) {
                    Index const c = sum(a, b);
                    expected += masses[p] * (degree(b) % 2 == 0 ? 1 : -1) *
                        monomial(particles[p], b) * reference.at(c) * factorial(c) / factorial(b);
                }
            }
            check(local(a[0], a[1], a[2]), expected, "M2L independent particle Taylor contraction",
                scales[degree(a)]);
        }
        Local pointLocal{};
        P2L(pointLocal, 1.73, derivatives);
        for (int i = 0; i < Local::kCount_; ++i) {
            check(pointLocal[i], 1.73 * derivatives[i], "P2L", 2 * scales[degree(all[i])]);
        }
        Field pointField{};
        M2P(pointField, moments, derivatives);
        check(pointField[0], local[0], "M2P potential matches M2L");
        for (int axis = 0; axis < 3; ++axis) {
            Index a{};
            a[axis] = 1;
            check(pointField[axis + 1], -local(a[0], a[1], a[2]), "M2P acceleration matches M2L");
        }
        Source noDipole = moments;
        for (int axis = 0; axis < 3; ++axis) {
            Index a{};
            a[axis] = 1;
            noDipole(a[0], a[1], a[2]) = 0.0;
        }
        Local centered{}, zeroDipole{};
        Field centeredField{}, zeroDipoleField{};
        M2L<true>(centered, moments, derivatives);
        M2L(zeroDipole, noDipole, derivatives);
        M2P<true>(centeredField, moments, derivatives);
        M2P(zeroDipoleField, noDipole, derivatives);
        for (auto const& a : all) {
            double expected = 0.0;
            for (auto const& b : indices(std::min(kSourceDegree, localDegree - degree(a)))) {
                if (degree(b) != 1) {
                    Index const c = sum(a, b);
                    expected += moments(b[0], b[1], b[2]) * (degree(b) % 2 == 0 ? 1.0 : -1.0) *
                        reference.at(c) * factorial(c) / factorial(b);
                }
            }
            check(centered(a[0], a[1], a[2]), expected, "centered M2L skips only dipoles",
                scales[degree(a)]);
            check(centered(a[0], a[1], a[2]), zeroDipole(a[0], a[1], a[2]),
                "centered and explicitly zero dipole M2L", scales[degree(a)]);
        }
        for (int component = 0; component < 4; ++component) {
            check(centeredField[component], zeroDipoleField[component],
                "centered and explicitly zero dipole M2P");
        }
        // Physical tiny particle offsets suppress high moments. Independently
        // seed every source rank at O(1), so dropping high traces or odd ranks
        // cannot hide below an absolute tolerance in the particle fixture.
        Source arbitrary{};
        for (int i = 0; i < Source::kCount_; ++i) {
            arbitrary[i] = 0.2 * std::sin(double(i + 1));
        }
        Local arbitraryLocal{}, arbitraryCentered{};
        Field arbitraryField{}, arbitraryCenteredField{};
        M2L(arbitraryLocal, arbitrary, derivatives);
        M2L<true>(arbitraryCentered, arbitrary, derivatives);
        M2P(arbitraryField, arbitrary, derivatives);
        M2P<true>(arbitraryCenteredField, arbitrary, derivatives);
        for (auto const& a : all) {
            double expected = 0.0;
            double centeredExpected = 0.0;
            double magnitude = 1.0;
            for (auto const& b : indices(std::min(kSourceDegree, localDegree - degree(a)))) {
                Index const c = sum(a, b);
                double const term = arbitrary(b[0], b[1], b[2]) *
                    (degree(b) % 2 == 0 ? 1.0 : -1.0) * reference.at(c) * factorial(c) /
                    factorial(b);
                expected += term;
                magnitude += std::abs(term);
                if (degree(b) != 1) {
                    centeredExpected += term;
                }
            }
            check(arbitraryLocal(a[0], a[1], a[2]), expected,
                "all source ranks independent M2L contraction", magnitude);
            check(arbitraryCentered(a[0], a[1], a[2]), centeredExpected,
                "centered M2L retains all higher odd ranks and traces", magnitude);
            if (degree(a) <= 1) {
                int const component = degree(a) == 0 ? 0 : a[0] + 2 * a[1] + 3 * a[2];
                double const sign = component == 0 ? 1.0 : -1.0;
                check(arbitraryField[component], sign * expected,
                    "all source ranks independent M2P contraction", magnitude);
                check(arbitraryCenteredField[component], sign * centeredExpected,
                    "centered M2P retains all higher odd ranks and traces", magnitude);
            }
        }
        if (location == 0) {
            EXPECT_GT(std::abs(centeredField.potential() - pointField.potential()), 1.0e-5);
        }
        for (bool doPotential : {false, true}) {
            checkAccumulation(
                local, [&](auto& target) { M2L(target, moments, derivatives, doPotential); },
                "additive M2L and potential preservation", doPotential);
            checkAccumulation(
                pointLocal, [&](auto& target) { P2L(target, 1.73, derivatives, doPotential); },
                "additive P2L and potential preservation", doPotential);
            checkAccumulation(
                pointField, [&](auto& target) { M2P(target, moments, derivatives, doPotential); },
                "additive M2P and potential preservation", doPotential);
            checkAccumulation(
                centered,
                [&](auto& target) { M2L<true>(target, moments, derivatives, doPotential); },
                "additive centered M2L and potential preservation", doPotential);
            checkAccumulation(
                centeredField,
                [&](auto& target) { M2P<true>(target, moments, derivatives, doPotential); },
                "additive centered M2P and potential preservation", doPotential);
        }
        Local coordinateLocal{}, coordinatePointLocal{};
        Field coordinateField{};
        M2L(coordinateLocal, moments, r, epsilonSquared);
        P2L(coordinatePointLocal, 1.73, r, epsilonSquared);
        M2P(coordinateField, moments, r, epsilonSquared);
        for (int i = 0; i < Local::kCount_; ++i) {
            check(coordinateLocal[i], local[i], "coordinate Plummer M2L", scales[degree(all[i])]);
            check(coordinatePointLocal[i], pointLocal[i], "coordinate Plummer P2L",
                2 * scales[degree(all[i])]);
        }
        for (int component = 0; component < 4; ++component) {
            check(coordinateField[component], pointField[component], "coordinate Plummer M2P");
        }
        Source pointMoment{};
        Field analyticField{};
        P2M(pointMoment, 1.73, Position{});
        M2P(analyticField, pointMoment, derivatives);
        check(analyticField.potential(), -1.73 / std::sqrt(q), "negative Plummer potential");
        for (int axis = 0; axis < 3; ++axis) {
            check(analyticField[axis + 1], -1.73 * r[axis] / std::pow(q, 1.5),
                "Plummer acceleration is negative potential gradient");
        }
        if (epsilonSquared == 0.0) {
            Local newtonianLocal{}, newtonianPointLocal{};
            Field newtonianField{};
            M2L(newtonianLocal, moments, r);
            P2L(newtonianPointLocal, 1.73, r);
            M2P(newtonianField, moments, r);
            for (int i = 0; i < Local::kCount_; ++i) {
                check(newtonianLocal[i], local[i], "Newtonian M2L coordinate overload",
                    scales[degree(all[i])]);
                check(newtonianPointLocal[i], pointLocal[i], "Newtonian P2L coordinate overload",
                    2 * scales[degree(all[i])]);
            }
            for (int component = 0; component < 4; ++component) {
                check(newtonianField[component], pointField[component],
                    "Newtonian M2P coordinate overload");
            }
        }
        // Additive operators must preserve nonzero target values.
        Field twice = pointField;
        M2P(twice, moments, derivatives);
        for (int component = 0; component < 4; ++component) {
            check(twice[component], 2 * pointField[component], "additive M2P");
        }
    }

    // The quadratic kernel has a trace and terminates exactly. All source
    // moments through rank two must contribute to its potential.
    if constexpr (kSourceDegree >= 2) {
        Green quadratic{};
        Position const r{0.3, -0.2, 0.6};
        quadratic[0] = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        for (int axis = 0; axis < 3; ++axis) {
            Index a{};
            a[axis] = 1;
            quadratic(a[0], a[1], a[2]) = 2 * r[axis];
            a[axis] = 2;
            quadratic(a[0], a[1], a[2]) = 2;
        }
        Field result{};
        M2P(result, moments, quadratic);
        Field expected{};
        for (int p = 0; p < int(particles.size()); ++p) {
            for (int axis = 0; axis < 3; ++axis) {
                double const displacement = r[axis] - particles[p][axis];
                expected[0] += masses[p] * displacement * displacement;
                expected[axis + 1] -= 2 * masses[p] * displacement;
            }
        }
        for (int component = 0; component < 4; ++component) {
            check(result[component], expected[component],
                "nonharmonic quadratic exact particle field");
        }
    }
    std::printf("CartesianSymmetric FMM order %d passed\n", localDegree + 1);
}

template <class Real, int order>
void testScalarOrSimd() {
    constexpr double kTolerance = std::is_same_v<SimdScalar<Real>, float> ? 2.0e-5 : 3.0e-12;
    if constexpr (std::is_arithmetic_v<Real>) {
        for (int sample = 0; sample < 4; ++sample) {
            SymmetricResult<Real, order> actual{};
            SymmetricResult<double, order> reference{};
            evaluateSymmetricOperators(Real(sample), actual);
            evaluateSymmetricOperators(double(sample), reference);
            compareSymmetricResults(
                actual, reference, [&](Real value, double expected, double scale) {
                    check(double(value), expected, "scalar precision parity", scale, kTolerance);
                });
        }
    } else {
        Real sample = simdBroadcast<Real>(0);
        for (size_t lane = 0; lane < Real::size(); ++lane) {
            simdSetLane(sample, lane, lane);
        }
        SymmetricResult<Real, order> actual{};
        evaluateSymmetricOperators(sample, actual);
        for (size_t lane = 0; lane < Real::size(); ++lane) {
            SymmetricResult<double, order> reference{};
            evaluateSymmetricOperators(double(lane), reference);
            compareSymmetricResults(
                actual, reference, [&](Real const& value, double expected, double scale) {
                    check(double(value[lane]), expected, "distinct SIMD lane parity", scale,
                        kTolerance);
                });
        }
    }
}

}    // namespace detail

class CartesianSymmetric : public ::testing::TestWithParam<int> {};

TEST_P(CartesianSymmetric, IndependentPolynomialAndParticleOracles) {
    fmmDispatch(GetParam(), [](auto order) { detail::testOrder<order.value - 1>(); });
}

TEST_P(CartesianSymmetric, FloatDoubleAndDistinctSimdLanes) {
    fmmDispatch(GetParam(), [](auto order) {
        detail::testScalarOrSimd<float, order.value>();
        detail::testScalarOrSimd<double, order.value>();
        detail::testScalarOrSimd<SimdFloat4, order.value>();
        detail::testScalarOrSimd<SimdFloat, order.value>();
        detail::testScalarOrSimd<SimdDouble, order.value>();
    });
}

INSTANTIATE_TEST_SUITE_P(AllOrders, CartesianSymmetric,
    ::testing::Range(kFmmMinOrder, kFmmMaxOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
