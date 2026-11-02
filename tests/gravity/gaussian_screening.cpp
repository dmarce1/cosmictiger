/* CosmicTiger - Gaussian-screening numerical oracle tests; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "gaussian_screening_test_helpers.hpp"

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace detail {

namespace cartesian = cosmictiger::gravity::cartesian;
using cosmictiger::gravity::GaussianComponent;
using cosmictiger::gravity::GaussianScreening;

template <class Real, int order>
void expectGaussianOracle(cartesian::Green<Real, order> const& actual,
    Vector3<Real> const& position, Real alpha, bool longRange) {
    auto const reference =
        gaussianDerivativeOracle<order>({position[0], position[1], position[2]}, alpha, longRange);
    constexpr double kTolerance = std::is_same_v<Real, float> ? 5.0e-5 : 3.0e-11;
    int index = 0;
    for (int rank = 0; rank < order; ++rank) {
        int const count = (rank + 1) * (rank + 2) / 2;
        long double scale = 0;
        for (int component = 0; component < count; ++component) {
            scale = std::max(scale, std::abs(reference[index + component]));
        }
        for (int component = 0; component < count; ++component) {
            double const budget = kTolerance * double(scale) +
                gaussianDerivativeUnderflowUnits(rank) *
                    double(std::numeric_limits<Real>::denorm_min());
            EXPECT_NEAR(double(actual[index]), double(reference[index]), budget)
                << "rank=" << rank << " coefficient=" << index << " alpha=" << alpha
                << " long=" << longRange;
            ++index;
        }
    }
}

template <class Real>
void expectGaussianPairOracle(
    cosmictiger::gravity::Field<Real> const& actual, Vector3<Real> const& position, Real alpha) {
    auto const expected = gaussianPairOracle({position[0], position[1], position[2]}, alpha);
    constexpr double kTolerance = std::is_same_v<Real, float> ? 5.0e-5 : 3.0e-11;
    for (int component = 0; component < 4; ++component) {
        EXPECT_NEAR(double(actual[component]), double(expected[component]),
            kTolerance * std::abs(double(expected[component])) +
                32 * double(std::numeric_limits<Real>::denorm_min()))
            << "PP adapter component=" << component << " alpha=" << alpha;
    }
}

template <class Real, int order>
void checkScalarGaussian() {
    for (int sample = 0; sample < kGaussianFixtureCount; ++sample) {
        SCOPED_TRACE("sample=" + std::to_string(sample));
        auto const input = gaussianFixture<Real>(sample);
        GaussianResult<Real, order> result{};
        evaluateGaussianFixture(input, result);
        expectGaussianOracle(result.longRange, input.position, input.alpha, true);
        expectGaussianOracle(result.shortRange, input.position, input.alpha, false);
        expectGaussianPairOracle(result.shortPair, input.position, input.alpha);
        // Integrating the Gaussian from zero to infinity gives -1/r. Keep
        // the reference in long double: the existing float Newtonian radial
        // powers can underflow before a high-rank Cartesian derivative does.
        auto const newtonian = gaussianDerivativeOracle<order>(
            {input.position[0], input.position[1], input.position[2]}, 0, false);
        constexpr double kTolerance = std::is_same_v<Real, float> ? 6.0e-5 : 4.0e-11;
        int index = 0;
        for (int rank = 0; rank < order; ++rank) {
            int const count = (rank + 1) * (rank + 2) / 2;
            double scale = 0;
            for (int component = 0; component < count; ++component) {
                scale = std::max(scale, std::abs(double(newtonian[index + component])));
            }
            for (int component = 0; component < count; ++component) {
                EXPECT_NEAR(double(result.longRange[index]) + result.shortRange[index],
                    double(newtonian[index]), kTolerance * scale);
                ++index;
            }
        }
        long double const radiusSquared =
            static_cast<long double>(input.position[0]) * input.position[0] +
            static_cast<long double>(input.position[1]) * input.position[1] +
            static_cast<long double>(input.position[2]) * input.position[2];
        double const laplacian = double(4 * std::pow(static_cast<long double>(input.alpha), 3) /
            std::sqrt(std::numbers::pi_v<long double>) *
            std::exp(-input.alpha * static_cast<long double>(input.alpha) * radiusSquared));
        for (bool longRange : {false, true}) {
            auto const& green = longRange ? result.longRange : result.shortRange;
            double const trace = double(green(2, 0, 0)) + green(0, 2, 0) + green(0, 0, 2);
            double const scale = std::max({std::abs(double(green(2, 0, 0))),
                std::abs(double(green(0, 2, 0))), std::abs(double(green(0, 0, 2))), laplacian});
            EXPECT_NEAR(trace, longRange ? laplacian : -laplacian, 3 * kTolerance * scale);
        }
    }
    // The stable long-range recurrence can change branches at y=rank+1,
    // where y=(alpha*r)^2. Probe both sides of every relevant threshold.
    for (int threshold = 1; threshold <= order; ++threshold) {
        for (int side : {-1, 1}) {
            Real const alpha = Real(1.3);
            double const separation = std::sqrt(double(threshold)) / double(alpha) *
                (1 + side * (std::is_same_v<Real, float> ? 1.0e-4 : 1.0e-8));
            GaussianInput<Real> const input{
                alpha, {Real(0.36 * separation), Real(-0.48 * separation), Real(0.8 * separation)}};
            GaussianResult<Real, order> result{};
            evaluateGaussianFixture(input, result);
            expectGaussianOracle(result.longRange, input.position, alpha, true);
            expectGaussianOracle(result.shortRange, input.position, alpha, false);
        }
    }
}

template <class Real, int order>
void checkOriginGaussian() {
    for (Real alpha : {Real(0.002), Real(0.125), Real(2), Real(6)}) {
        GaussianScreening<Real> const screening(alpha);
        for (Real scale : {Real(0), Real(1.0e-20), Real(1.0e-10), Real(1.0e-5)}) {
            Vector3<Real> const position{scale, Real(-0.5) * scale, Real(0.25) * scale};
            cartesian::Green<Real, order> green{};
            cartesian::evaluateGreen<GaussianComponent::Long>(green, position, screening);
            expectGaussianOracle(green, position, alpha, true);
            if (scale == 0) {
                double const rootPi = std::sqrt(std::numbers::pi);
                EXPECT_NEAR(double(green[0]), -2 * double(alpha) / rootPi,
                    (std::is_same_v<Real, float> ? 1.0e-6 : 1.0e-13) * double(alpha));
                EXPECT_EQ(green(1, 0, 0), Real(0));
                EXPECT_EQ(green(0, 1, 0), Real(0));
                EXPECT_EQ(green(0, 0, 1), Real(0));
                double const second = 4 * std::pow(double(alpha), 3) / (3 * rootPi);
                EXPECT_NEAR(double(green(2, 0, 0)), second,
                    (std::is_same_v<Real, float> ? 1.0e-6 : 1.0e-13) * second);
            }
        }
    }
}

template <class Real, int order>
void checkSimdGaussian() {
    using Scalar = SimdScalar<Real>;
    for (Scalar alpha : {Scalar(0.002), Scalar(0.7), Scalar(2.75)}) {
        GaussianScreening<Scalar> const screening(alpha);
        for (int first = 0; first < 16; first += Real::size()) {
            Vector3<Real> position{
                simdBroadcast<Real>(0), simdBroadcast<Real>(0), simdBroadcast<Real>(0)};
            for (size_t lane = 0; lane < Real::size(); ++lane) {
                auto point = gaussianFixturePosition(alpha, (first + lane) % 12);
                for (int axis = 0; axis < 3; ++axis) {
                    simdSetLane(position[axis], lane, point[axis]);
                }
            }
            for (bool longRange : {false, true}) {
                cartesian::Green<Real, order> green{};
                cosmictiger::gravity::Field<Real> shortPair{};
                if (longRange) {
                    // Mix exact origin, tiny radius and far-field lanes in the
                    // same packet, exercising branch masks independently.
                    simdSetLane(position[0], 0, first == 0 ? 0.0 : 1.0e-20);
                    simdSetLane(position[1], 0, 0.0);
                    simdSetLane(position[2], 0, 0.0);
                    cartesian::evaluateGreen<GaussianComponent::Long>(green, position, screening);
                } else {
                    cosmictiger::gravity::p3m::fmm::evaluateGreen(green, position, screening);
                    shortPair = cosmictiger::gravity::p3m::pp::evaluateField(position, screening);
                }
                for (size_t lane = 0; lane < Real::size(); ++lane) {
                    cartesian::Green<Scalar, order> scalar{};
                    Vector3<Scalar> const point{Scalar(position[0][lane]),
                        Scalar(position[1][lane]), Scalar(position[2][lane])};
                    for (int i = 0; i < scalar.kSize_; ++i) {
                        scalar[i] = green[i][lane];
                    }
                    expectGaussianOracle(scalar, point, alpha, longRange);
                    if (!longRange) {
                        cosmictiger::gravity::Field<Scalar> scalarPair{};
                        for (int component = 0; component < 4; ++component) {
                            scalarPair[component] = shortPair[component][lane];
                        }
                        expectGaussianPairOracle(scalarPair, point, alpha);
                    }
                }
            }
        }
    }
}

template <class Real, int order>
void checkGaussianScaling() {
    Vector3<Real> const point{Real(0.1), Real(-0.2), Real(0.8)};
    GaussianResult<Real, order> original{};
    evaluateGaussianFixture(GaussianInput<Real>{Real(2), point}, original);
    for (Real scale : {Real(0.03125), Real(32), Real(1000)}) {
        Vector3<Real> const scaledPoint{scale * point[0], scale * point[1], scale * point[2]};
        GaussianResult<Real, order> scaled{};
        evaluateGaussianFixture(GaussianInput<Real>{Real(2) / scale, scaledPoint}, scaled);
        for (bool longRange : {false, true}) {
            auto const& before = longRange ? original.longRange : original.shortRange;
            auto const& after = longRange ? scaled.longRange : scaled.shortRange;
            double factor = 1.0 / scale;
            int index = 0;
            for (int rank = 0; rank < order; ++rank) {
                int const count = (rank + 1) * (rank + 2) / 2;
                double rankScale = 0;
                for (int component = 0; component < count; ++component) {
                    rankScale = std::max(rankScale, std::abs(double(before[index + component])));
                }
                for (int component = 0; component < count; ++component) {
                    EXPECT_NEAR(double(after[index]), double(before[index]) * factor,
                        (std::is_same_v<Real, float> ? 8.0e-5 : 4.0e-11) * rankScale * factor);
                    ++index;
                }
                factor /= scale;
            }
        }
    }
}

template <int order>
void checkScreenedMultipoleConvergence() {
    std::array<Vector3<double>, 3> const positions{
        {{0.13, -0.07, 0.09}, {-0.11, 0.15, -0.04}, {0.08, 0.03, -0.16}}};
    std::array<double, 3> const masses{0.4, 0.7, 0.2};
    Vector3<double> const center{0.8, -0.4, 1.1};
    GaussianScreening<double> const screening(1.3);
    std::array<double, 2> errors{};
    for (int refinement = 0; refinement < 2; ++refinement) {
        // At this separation the screened field varies faster than 1/r.
        // Start with compact cells so even the lowest retained order is in
        // its asymptotic convergence regime, then halve both source and
        // evaluation offsets without changing the center or screening.
        double const scale = std::ldexp(1.0, -refinement - 2);
        Vector3<double> const evaluation{0.06 * scale, -0.04 * scale, 0.03 * scale};
        cartesian::Multipole<double, order> moments{};
        for (size_t particle = 0; particle < masses.size(); ++particle) {
            Vector3<double> const offset{positions[particle][0] * scale,
                positions[particle][1] * scale, positions[particle][2] * scale};
            P2M(moments, masses[particle], offset);
        }
        cartesian::Green<double, order> green{};
        cosmictiger::gravity::p3m::fmm::evaluateGreen(green, center, screening);
        cartesian::Expansion<double, order> local{};
        cosmictiger::gravity::Field<double> actual{};
        M2L(local, moments, green);
        L2P(actual, local, evaluation);
        std::array<long double, 4> expected{};
        for (size_t particle = 0; particle < masses.size(); ++particle) {
            GaussianPosition displacement{};
            long double radiusSquared = 0;
            for (int axis = 0; axis < 3; ++axis) {
                displacement[axis] = static_cast<long double>(center[axis]) + evaluation[axis] -
                    scale * positions[particle][axis];
                radiusSquared += displacement[axis] * displacement[axis];
            }
            long double const radius = std::sqrt(radiusSquared);
            long double const y = screening.alpha() * radius;
            long double const function = std::erfc(y);
            long double const gaussian =
                2 * y * std::exp(-y * y) / std::sqrt(std::numbers::pi_v<long double>);
            expected[0] -= masses[particle] * function / radius;
            for (int axis = 0; axis < 3; ++axis) {
                expected[axis + 1] -= masses[particle] * displacement[axis] *
                    (function + gaussian) / (radiusSquared * radius);
            }
        }
        double norm = 0;
        for (int field = 0; field < 4; ++field) {
            errors[refinement] =
                std::max(errors[refinement], std::abs(actual[field] - double(expected[field])));
            norm = std::max(norm, std::abs(double(expected[field])));
        }
        EXPECT_LT(errors[refinement], 0.1 * norm);
    }
    EXPECT_LT(errors[1], 0.7 * errors[0] + 3.0e-14)
        << "supplied Gaussian Green M2L/L2P must converge toward analytic particle force";
}

}    // namespace detail

TEST(GaussianScreeningParameters, ExplicitAlphaAndSplitRadiusConvention) {
    using cosmictiger::gravity::GaussianScreening;
    static_assert(!std::is_convertible_v<double, GaussianScreening<double>>);
    constexpr GaussianScreening<double> kDirect(2);
    constexpr auto kFromRadius = GaussianScreening<double>::fromSplitRadius(0.25);
    static_assert(kDirect.alpha() == kFromRadius.alpha());
    static_assert(kDirect.alphaSquared() == 4);
    EXPECT_DOUBLE_EQ(kFromRadius.alpha(), 2);
}

class GaussianScreening : public ::testing::TestWithParam<int> {};

TEST_P(GaussianScreening, IndependentScalarDerivativeOraclesAndComplementarity) {
    fmmDispatch(GetParam(), [](auto order) {
        detail::checkScalarGaussian<float, order.value>();
        detail::checkScalarGaussian<double, order.value>();
    });
}

TEST_P(GaussianScreening, FiniteOriginAndSmallRadiusLimits) {
    fmmDispatch(GetParam(), [](auto order) {
        detail::checkOriginGaussian<float, order.value>();
        detail::checkOriginGaussian<double, order.value>();
    });
}

TEST_P(GaussianScreening, MixedSimdLanesAndCoordinateScaling) {
    fmmDispatch(GetParam(), [](auto order) {
        detail::checkSimdGaussian<SimdFloat4, order.value>();
        detail::checkSimdGaussian<SimdFloat, order.value>();
        detail::checkSimdGaussian<SimdDouble, order.value>();
        detail::checkGaussianScaling<float, order.value>();
        detail::checkGaussianScaling<double, order.value>();
    });
}

TEST_P(GaussianScreening, ShortGreenMultipolesConvergeToAnalyticParticleFields) {
    fmmDispatch(
        GetParam(), [](auto order) { detail::checkScreenedMultipoleConvergence<order.value>(); });
}

INSTANTIATE_TEST_SUITE_P(AllOrders, GaussianScreening,
    ::testing::Range(kFmmMinOrder, kFmmMaxOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
