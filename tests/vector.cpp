/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cmath>
#include <concepts>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <cosmictiger/math/vector.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace detail {

template <class Spatial>
constexpr bool kPreservesArithmeticTypes = [] {
    using Scalar = std::remove_cvref_t<decltype(std::declval<Spatial&>()[0])>;
    using Mutable = Spatial&;
    using ReadOnly = Spatial const&;
    return std::same_as<decltype(+std::declval<ReadOnly>()), Spatial> &&
        std::same_as<decltype(-std::declval<ReadOnly>()), Spatial> &&
        std::same_as<decltype(std::declval<ReadOnly>() + std::declval<ReadOnly>()), Spatial> &&
        std::same_as<decltype(std::declval<ReadOnly>() - std::declval<ReadOnly>()), Spatial> &&
        std::same_as<decltype(std::declval<ReadOnly>() * std::declval<Scalar const&>()), Spatial> &&
        std::same_as<decltype(std::declval<Scalar const&>() * std::declval<ReadOnly>()), Spatial> &&
        std::same_as<decltype(std::declval<ReadOnly>() / std::declval<Scalar const&>()), Spatial> &&
        std::same_as<decltype(std::declval<Mutable>() += std::declval<ReadOnly>()), Mutable> &&
        std::same_as<decltype(std::declval<Mutable>() -= std::declval<ReadOnly>()), Mutable> &&
        std::same_as<decltype(std::declval<Mutable>() *= std::declval<Scalar const&>()), Mutable> &&
        std::same_as<decltype(std::declval<Mutable>() /= std::declval<Scalar const&>()), Mutable> &&
        std::same_as<decltype(std::declval<ReadOnly>().dot(std::declval<ReadOnly>())), Scalar>;
}();

template <class Spatial>
concept HasCross = requires(Spatial const& value) { value.cross(value); };

struct FourVector : math::Vector<double, 4, FourVector> {
    using Base = math::Vector<double, 4, FourVector>;
    using Base::Base;
};

// Products promote to int; converting the final dot product back must happen once.
// The deleted default constructor also protects explicit component construction.
struct PromotingScalar {
    int value;
    PromotingScalar() = delete;
    constexpr PromotingScalar(int input)
      : value(input < -100  ? -100 :
                input > 100 ? 100 :
                              input) {}
    constexpr operator int() const {
        return value;
    }
};

using Spatial = Vector3<double>;
static_assert(kPreservesArithmeticTypes<Spatial>);
static_assert(std::same_as<decltype(Spatial::size()), int>);
static_assert(std::is_trivially_default_constructible_v<Spatial>);
static_assert(kPreservesArithmeticTypes<FourVector>);
static_assert(std::same_as<decltype(std::declval<Spatial&>()[0]), double&>);
static_assert(std::same_as<decltype(std::declval<Spatial const&>()[0]), double const&>);
static_assert(std::same_as<decltype(std::declval<Spatial&>().data()), double*>);
static_assert(std::same_as<decltype(std::declval<Spatial const&>().data()), double const*>);
static_assert(std::same_as<decltype(std::declval<Spatial&>().begin()), double*>);
static_assert(std::same_as<decltype(std::declval<Spatial const&>().begin()), double const*>);
static_assert(std::same_as<decltype(std::declval<Spatial&>().end()), double*>);
static_assert(std::same_as<decltype(std::declval<Spatial const&>().end()), double const*>);
static_assert(std::same_as<decltype(std::declval<Spatial const&>().squaredNorm()), double>);
static_assert(std::same_as<decltype(std::declval<Spatial const&>().norm()), double>);
static_assert(
    std::same_as<decltype(std::declval<Spatial const&>().cross(std::declval<Spatial const&>())),
        Spatial>);
static_assert(std::same_as<decltype(math::select(true, std::declval<Spatial const&>(),
                               std::declval<Spatial const&>())),
    Spatial>);
static_assert(std::same_as<decltype(std::declval<Spatial>() == std::declval<Spatial>()), bool>);
static_assert(std::same_as<decltype(std::declval<Spatial>() != std::declval<Spatial>()), bool>);
static_assert(std::is_constructible_v<Spatial, double>);
static_assert(!std::is_convertible_v<double, Spatial>);
static_assert(std::is_constructible_v<Spatial, double, double, double>);
static_assert(!std::is_constructible_v<Spatial, double, double>);
static_assert(!std::is_constructible_v<Spatial, double, double, double, double>);
static_assert(std::is_constructible_v<Spatial, Vector3<float> const&>);
static_assert(!std::is_convertible_v<Vector3<float>, Spatial>);
static_assert(HasCross<Spatial>);
static_assert(!HasCross<math::Vector<double, 3>>);
static_assert(!HasCross<FourVector>);
static_assert(std::is_standard_layout_v<Vector3<float>>);
static_assert(std::is_standard_layout_v<Vector3<double>>);
static_assert(std::is_trivially_copyable_v<Vector3<float>>);
static_assert(std::is_trivially_copyable_v<Vector3<double>>);
static_assert(sizeof(Vector3<float>) == 3 * sizeof(float));
static_assert(sizeof(Vector3<double>) == 3 * sizeof(double));
static_assert(alignof(Vector3<float>) == alignof(float));
static_assert(alignof(Vector3<double>) == alignof(double));

template <int dimensions, size_t... indices>
constexpr math::Vector<double, dimensions> increasingVector(std::index_sequence<indices...>) {
    return math::Vector<double, dimensions>(double(indices + 1)...);
}

template <int dimensions>
void verifyDimension() {
    using Vector = math::Vector<double, dimensions>;
    static_assert(kPreservesArithmeticTypes<Vector>);
    static_assert(!HasCross<Vector>);
    static_assert(!std::is_convertible_v<double, Vector>);
    constexpr Vector kZero{};
    constexpr Vector kValues = increasingVector<dimensions>(std::make_index_sequence<dimensions>{});
    constexpr Vector kBroadcast(2.0);
    constexpr double kSquaredNorm = dimensions * (dimensions + 1) * (2 * dimensions + 1) / 6.0;
    static_assert(Vector::size() == dimensions);
    static_assert(kZero == Vector(0.0));
    static_assert(kValues.dot(kValues) == kSquaredNorm);
    static_assert(((kValues + kBroadcast) * 2.0) / 2.0 - kBroadcast == kValues);
    constexpr math::Vector<float, dimensions> kConverted(kValues);
    static_assert(kConverted[dimensions - 1] == float(dimensions));
    Vector actual{};
    std::copy(kValues.begin(), kValues.end(), actual.data());
    EXPECT_EQ(actual, kValues);
    EXPECT_EQ(actual.end() - actual.begin(), dimensions);
    for (int dim = 0; dim < dimensions; ++dim) {
        EXPECT_EQ(&actual[dim], actual.data() + dim);
        EXPECT_EQ((-kValues)[dim], -double(dim + 1));
        EXPECT_EQ((+kValues)[dim], double(dim + 1));
        EXPECT_EQ(kBroadcast[dim], 2.0);
    }
    actual[dimensions - 1] += 1.0;
    EXPECT_NE(actual, kValues);
    EXPECT_EQ(math::select(true, kValues, actual), kValues);
    EXPECT_EQ(math::select(false, kValues, actual), actual);
    EXPECT_DOUBLE_EQ(kValues.norm(), std::sqrt(kSquaredNorm));
}

}    // namespace detail

TEST(Vector, GeneralDimensions) {
    detail::verifyDimension<1>();
    detail::verifyDimension<2>();
    detail::verifyDimension<4>();
    detail::verifyDimension<6>();
}

TEST(Vector, DerivedResultsRetainTheirTypeAndObject) {
    detail::FourVector value(1.0, 2.0, 3.0, 4.0);
    detail::FourVector const increment(2.0);
    auto* const original = &value;
    EXPECT_EQ(&(value += increment), original);
    EXPECT_EQ(&(value -= increment), original);
    EXPECT_EQ(&(value *= 2.0), original);
    EXPECT_EQ(&(value /= 2.0), original);
    auto const sum = value + increment;
    EXPECT_EQ(sum[0], 3.0);
    EXPECT_EQ(sum[3], 6.0);
    EXPECT_EQ(value.dot(value), 30.0);
}

TEST(Vector, ScalarOperandMayAliasAComponent) {
    Vector3<double> multiply(2.0, 3.0, 5.0);
    multiply *= multiply[0];
    EXPECT_EQ(multiply, Vector3<double>(4.0, 6.0, 10.0));
    Vector3<double> divide(2.0, 4.0, 6.0);
    divide /= divide[0];
    EXPECT_EQ(divide, Vector3<double>(1.0, 2.0, 3.0));
}

TEST(Vector, DotPreservesEvaluationAndPromotion) {
    Vector3<double> const cancellation(1.0e16, -1.0e16, 1.0);
    EXPECT_EQ(cancellation.dot(Vector3<double>(1.0)), 1.0);
    Vector3<double> const negativeZero(-0.0);
    EXPECT_TRUE(std::signbit(negativeZero.dot(Vector3<double>(1.0))));
    math::Vector<double, 6> const negativeZeroSix(-0.0);
    EXPECT_TRUE(std::signbit(negativeZeroSix.dot(math::Vector<double, 6>(1.0))));
    using Scalar = detail::PromotingScalar;
    constexpr Vector3<Scalar> kFirst(100, 100, -100);
    constexpr Vector3<Scalar> kSecond(2, 2, 3);
    static_assert(kFirst.dot(kSecond).value == 100);
    constexpr Vector3<Scalar> kBroadcast(Scalar(3));
    constexpr Vector3<int> kIntegers(1, 2, 3);
    constexpr Vector3<Scalar> kConverted(kIntegers);
    static_assert(kBroadcast[2].value == 3);
    static_assert(kConverted[2].value == 3);
    constexpr math::Vector<Scalar, 4> kFour(100, 100, -100, 1);
    constexpr math::Vector<Scalar, 4> kWeights(2, 2, 3, -1);
    static_assert(kFour.dot(kWeights).value == 99);
    EXPECT_EQ(kFirst.dot(kSecond).value, 100);
    EXPECT_EQ(kFour.dot(kWeights).value, 99);
}

TEST(Vector, SimdArithmeticAndMasksCoverEveryComponent) {
    using Pack = SimdVector<float, 4>;
    using Spatial = math::Vector<Pack, 4>;
    using Mask = decltype(std::declval<Pack>() == std::declval<Pack>());
    static_assert(detail::kPreservesArithmeticTypes<Spatial>);
    static_assert(std::same_as<decltype(std::declval<Spatial>() == std::declval<Spatial>()), Mask>);
    static_assert(
        std::same_as<decltype(std::declval<Vector3<Pack>>() == std::declval<Vector3<Pack>>()),
            Mask>);
    Spatial left(Pack(0.0f));
    for (int dim = 0; dim < 4; ++dim) {
        left[dim] = Pack([dim](auto lane) { return float((dim + 1) * (int(lane) + 1)); });
    }
    Spatial right = left;
    simdSetLane(right[3], 2, 13.0f);
    auto const equal = left == right;
    auto const unequal = left != right;
    auto const selected = math::select(equal, left + Spatial(Pack(10.0f)), right);
    auto const combined = Pack(2.0f) * (left + right) - left;
    auto const dot = left.dot(right);
    for (int lane = 0; lane < 4; ++lane) {
        EXPECT_EQ(bool(equal[lane]), lane != 2);
        EXPECT_EQ(bool(unequal[lane]), lane == 2);
        float expectedDot = 0.0f;
        for (int dim = 0; dim < 4; ++dim) {
            float const first = float((dim + 1) * (lane + 1));
            float const second = dim == 3 && lane == 2 ? 13.0f : first;
            EXPECT_EQ(float(selected[dim][lane]), lane == 2 ? second : first + 10.0f);
            EXPECT_EQ(float(combined[dim][lane]), 2.0f * (first + second) - first);
            expectedDot += first * second;
        }
        EXPECT_EQ(float(dot[lane]), expectedDot);
    }
    Vector3<Pack> const axis(Pack(1.0f), Pack(0.0f), Pack(0.0f));
    Vector3<Pack> const spatial(left[0], left[1], left[3]);
    auto const perpendicular = axis.cross(spatial);
    auto const chosen = math::select(equal, perpendicular, spatial);
    static_assert(std::same_as<std::remove_cv_t<decltype(chosen)>, Vector3<Pack>>);
    for (int lane = 0; lane < 4; ++lane) {
        EXPECT_EQ(float(perpendicular[0][lane]), 0.0f);
        EXPECT_EQ(float(perpendicular[1][lane]), -4.0f * (lane + 1));
        EXPECT_EQ(float(perpendicular[2][lane]), 2.0f * (lane + 1));
        EXPECT_EQ(float(chosen[0][lane]), lane == 2 ? 3.0f : 0.0f);
    }
}
