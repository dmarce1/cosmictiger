#include <array>
#include <cmath>
#include <cstdlib>
#include <initializer_list>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp>
#include <cosmictiger/gravity/fmm_kernels/spherical_operators.hpp>

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace {

using CartesianMultipole = cosmictiger::gravity::cartesian::Multipole<double, 7>;
using CartesianExpansion = cosmictiger::gravity::cartesian::Expansion<double, 7>;
using SphericalMultipole = cosmictiger::gravity::spherical::Multipole<double, 7>;
using SphericalExpansion = cosmictiger::gravity::spherical::Expansion<double, 7>;
using Field = cosmictiger::gravity::Field<double>;

template <class Target, class Source>
concept HasM2L = requires(Target& target, Source const& source,
    Array<double, kDimensions> const& displacement) { M2L(target, source, displacement); };

static_assert(!HasM2L<CartesianExpansion, SphericalMultipole>);
static_assert(!HasM2L<SphericalExpansion, CartesianMultipole>);
static_assert(!HasM2L<cosmictiger::gravity::cartesian::Expansion<double, 6>, CartesianMultipole>);

template <class Multipole, class Expansion>
void checkPointFields() {
    std::array<std::array<double, kDimensions>, 4> const sourceCenters{
        {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {.7, -.4, 1.2}}};
    Array<double, kDimensions> const zero{};
    for (auto const& center : sourceCenters) {
        Array<double, kDimensions> displacement{};
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
