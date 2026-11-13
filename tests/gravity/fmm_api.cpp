#include <array>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <initializer_list>
#include <utility>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/kernels/cartesian_tracefree/ewald_m2l.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/math/vector3.hpp>

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace {

using CartesianMultipole = cosmictiger::gravity::cartesian::Multipole<double>;
using CartesianExpansion = cosmictiger::gravity::cartesian::Expansion<double>;
using CartesianGreen = cosmictiger::gravity::cartesian::Green<double>;
using Field = cosmictiger::gravity::Field<double>;
// Production overloads accept only the compiled scalar types and put the
// accumulator last. An unconstrained declaration would incorrectly accept long double.
template <class Real>
concept HasCompiledM2P = requires(cosmictiger::gravity::cartesian::Multipole<Real> const& source,
    Vector3<Real> const& displacement, cosmictiger::gravity::Field<Real>& field) {
    { cosmictiger::gravity::cartesian::M2P(source, displacement, field) } -> std::same_as<void>;
};
static_assert(HasCompiledM2P<float>);
static_assert(HasCompiledM2P<double>);
static_assert(HasCompiledM2P<SimdVector<float, 8>>);
static_assert(!HasCompiledM2P<long double>);

template <class Real>
concept HasCompiledM2Lewald =
    requires(cosmictiger::gravity::cartesian::Multipole<Real> const& source,
        cosmictiger::gravity::cartesian::Green<Real> const& green,
        cosmictiger::gravity::cartesian::Expansion<Real>& local) {
        { cosmictiger::gravity::cartesian::M2Lewald(source, green, local) } -> std::same_as<void>;
    };
static_assert(HasCompiledM2Lewald<float>);
static_assert(HasCompiledM2Lewald<double>);
static_assert(HasCompiledM2Lewald<SimdVector<float, 8>>);
static_assert(!HasCompiledM2Lewald<long double>);

template <class Coefficient>
concept HasQuadraticZz = requires(Coefficient value) { value.zz; };

template <class Coefficient>
void checkQuadraticStorage() {
    constexpr bool kStored =
        cosmictiger::gravity::cartesian::kQuadrupoleTrace && Coefficient::kDegree_ >= 2;
    constexpr int kHarmonicCount = (Coefficient::kDegree_ + 1) * (Coefficient::kDegree_ + 1);
    static_assert(HasQuadraticZz<Coefficient> == kStored);
    Coefficient value{};
    EXPECT_EQ(value.size(), kHarmonicCount + int(kStored));
    if constexpr (kStored) {
        value.xx = 2.0;
        value.yy = 3.0;
        value.zz = 7.0;
        EXPECT_DOUBLE_EQ(value[kHarmonicCount], 7.0);
        EXPECT_DOUBLE_EQ(std::as_const(value)(0, 0, 2), 7.0);
        value(0, 0, 2) = 11.0;
        EXPECT_DOUBLE_EQ(value.zz, 11.0);
        EXPECT_DOUBLE_EQ(value.xx, 2.0);
        EXPECT_DOUBLE_EQ(value.yy, 3.0);
    }
}

template <class Multipole, class Expansion>
void checkPointFields() {
    std::array<Vector3<double>, 4> const sourceCenters{
        {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {.7, -.4, 1.2}}};
    Vector3<double> const zero{};
    for (auto const& center : sourceCenters) {
        Vector3<double> displacement{};
        double radiusSquared = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            displacement[dim] = -center[dim];
            radiusSquared += center[dim] * center[dim];
        }
        double const iR = 1.0 / std::sqrt(radiusSquared);
        Multipole moments{};
        Expansion local{};
        Field multipoleField{};
        Field localField{};
        P2M(2.0, zero, moments);
        P2L(2.0, displacement, local);
        M2P(moments, displacement, multipoleField);
        L2P(local, zero, localField);
        for (Field const& field : {multipoleField, localField}) {
            EXPECT_NEAR(field.potential(), -2.0 * iR, 1e-12);
            for (int dim = 0; dim < kDimensions; ++dim) {
                EXPECT_NEAR(field[dim + 1], -2.0 * displacement[dim] * iR / radiusSquared, 1e-12);
            }
        }
        Field accumulated{};
        accumulated.potential() = 3.0;
        M2P(moments, displacement, false, accumulated);
        L2P(local, zero, false, accumulated);
        EXPECT_DOUBLE_EQ(accumulated.potential(), 3.0);
        for (int dim = 0; dim < kDimensions; ++dim) {
            EXPECT_NEAR(accumulated[dim + 1], 2.0 * multipoleField[dim + 1], 1e-12);
        }
    }
}

TEST(FmmApi, CartesianPointFields) {
    checkPointFields<CartesianMultipole, CartesianExpansion>();
}

TEST(FmmApi, CartesianQuadraticStorage) {
    checkQuadraticStorage<CartesianMultipole>();
    checkQuadraticStorage<CartesianExpansion>();
    checkQuadraticStorage<CartesianGreen>();
}

}    // namespace
