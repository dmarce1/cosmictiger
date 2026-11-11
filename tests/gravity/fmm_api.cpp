#include <array>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <initializer_list>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/kernels/spherical/p2m.hpp>
#include <cosmictiger/math/vector3.hpp>

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace {

using CartesianMultipole = cosmictiger::gravity::cartesian::Multipole<double>;
using CartesianExpansion = cosmictiger::gravity::cartesian::Expansion<double>;
using SphericalMultipole = cosmictiger::gravity::spherical::Multipole<double>;
using Field = cosmictiger::gravity::Field<double>;
static_assert(cosmictiger::gravity::spherical::Multipole<double>::kCount_ ==
    (kFmmOrder - 1) * (kFmmOrder - 1));

template <class Target, class Source>
concept HasM2L = requires(Target& target, Source const& source,
    Vector3<double> const& displacement) { M2L(source, displacement, target); };

static_assert(!HasM2L<CartesianExpansion, SphericalMultipole>);

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

TEST(FmmApi, SphericalDipolePacking) {
    SphericalMultipole moments{};
    Vector3<double> const offset{0.4, -0.3, 0.2};
    cosmictiger::gravity::spherical::detail::P2M(2.0, offset[0], offset[1], offset[2], moments);
    EXPECT_DOUBLE_EQ(moments.real(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(moments.real(1, 0), -0.4);
    EXPECT_DOUBLE_EQ(moments.real(1, 1), -0.4);
    EXPECT_DOUBLE_EQ(moments.imaginary(1, 1), 0.3);
    EXPECT_EQ(&moments.imaginary(1, 1), &moments[1]);
    EXPECT_EQ(&moments.real(1, 0), &moments[2]);
    EXPECT_EQ(&moments.real(1, 1), &moments[3]);
}

}    // namespace
