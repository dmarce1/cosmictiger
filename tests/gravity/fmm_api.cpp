#include <array>
#include <cmath>
#include <cstdlib>
#include <initializer_list>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/kernels/spherical_operators.hpp>
#include <cosmictiger/math/vector3.hpp>

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace {

using CartesianMultipole =
    cosmictiger::gravity::cartesian::tracefree::Multipole<double, kFmmDefaultOrder>;
using CartesianExpansion =
    cosmictiger::gravity::cartesian::tracefree::Expansion<double, kFmmDefaultOrder>;
using SphericalMultipole = cosmictiger::gravity::spherical::Multipole<double, kFmmDefaultOrder>;
using SphericalExpansion = cosmictiger::gravity::spherical::Expansion<double, kFmmDefaultOrder>;
using Field = cosmictiger::gravity::Field<double>;

template <class Target, class Source>
concept HasM2L = requires(Target& target, Source const& source,
    Vector3<double> const& displacement) { M2L(target, source, displacement); };

static_assert(!HasM2L<CartesianExpansion, SphericalMultipole>);
static_assert(!HasM2L<SphericalExpansion, CartesianMultipole>);
constexpr int kDifferentOrder = kFmmDefaultOrder == 9 ? 8 : kFmmDefaultOrder + 1;
static_assert(
    !HasM2L<cosmictiger::gravity::cartesian::tracefree::Expansion<double, kDifferentOrder>,
        CartesianMultipole>);

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
        double const inverseRadius = 1.0 / std::sqrt(radiusSquared);
        Multipole moments{};
        Expansion local{};
        Field multipoleField{};
        Field localField{};
        P2M(moments, 2.0, zero);
        P2L(local, 2.0, displacement);
        M2P(multipoleField, moments, displacement);
        L2P(localField, local, zero);
        for (Field const& field : {multipoleField, localField}) {
            EXPECT_NEAR(field.potential(), -2.0 * inverseRadius, 1e-12);
            for (int dim = 0; dim < kDimensions; ++dim) {
                EXPECT_NEAR(field[dim + 1],
                    -2.0 * displacement[dim] * inverseRadius / radiusSquared, 1e-12);
            }
        }
        Field accumulated{};
        accumulated.potential() = 3.0;
        M2P(accumulated, moments, displacement, false);
        L2P(accumulated, local, zero, false);
        EXPECT_DOUBLE_EQ(accumulated.potential(), 3.0);
        for (int dim = 0; dim < kDimensions; ++dim) {
            EXPECT_NEAR(accumulated[dim + 1], 2.0 * multipoleField[dim + 1], 1e-12);
        }
    }
}

TEST(FmmApi, CartesianPointFields) {
    checkPointFields<CartesianMultipole, CartesianExpansion>();
}

TEST(FmmApi, SphericalPointFields) {
    checkPointFields<SphericalMultipole, SphericalExpansion>();
}

}    // namespace
