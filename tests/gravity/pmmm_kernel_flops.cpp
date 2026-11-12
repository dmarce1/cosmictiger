/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <array>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#define COSMICTIGER_FMM_INSTRUMENTATION

#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kernels/spherical_ewald_kernels.hpp>
#include <cosmictiger/gravity/pmmm_kernels.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <tests/flop_counting_real.hpp>
#include <tests/gravity/kernels/instrumentation/spherical_ewald_green.cpp>
#include <tests/gravity/kernels/instrumentation/spherical_green.cpp>
#include <tests/gravity/kernels/instrumentation/spherical_p2m.cpp>

#include <tests/gravity/kernels/instrumentation/cartesian_tracefree_ewald_green.cpp>
#include <tests/gravity/kernels/instrumentation/cartesian_tracefree_ewald_m2l.cpp>

#include "pmmm_reference.hpp"

// The generated scalar/SIMD adapters use std math internally. Specialize only
// those adapters so the execution oracle also observes each elementary result.

namespace cosmictiger::gravity::spherical::detail {

template <>
inline ::detail::CountingReal ewaldExp(::detail::CountingReal const& value) {
    ::detail::CountingReal::add(1);
    return std::exp(value.value());
}

template <>
inline ::detail::CountingReal ewaldErfc(::detail::CountingReal const& value) {
    ::detail::CountingReal::add(1);
    return std::erfc(value.value());
}

template <>
inline ::detail::CountingReal ewaldCos(::detail::CountingReal const& value) {
    ::detail::CountingReal::add(1);
    return std::cos(value.value());
}

template <>
inline ::detail::CountingReal ewaldSin(::detail::CountingReal const& value) {
    ::detail::CountingReal::add(1);
    return std::sin(value.value());
}

}    // namespace cosmictiger::gravity::spherical::detail

namespace pmmm_test {
namespace detail {

    template <int order>
    void checkKernelCounts() {
        using Real = ::detail::CountingReal;
        namespace spherical = cosmictiger::gravity::spherical;
        constexpr int kDegree = order - 1;
        constexpr double kCellWidth = 0.125;
        Vector3<double> const point{0.13, -0.21, 0.38};
        std::array<double, (order - 1) * (order - 1)> moments{};
        std::array<Real, moments.size()> countedMoments{};
        FlopCount const momentCount = pmmm::addMoments(order, moments.data(), 0.7, point);
        Real::reset();
        spherical::detail::P2M(
            Real(0.7), Real(point[0]), Real(point[1]), Real(point[2]), countedMoments);
        EXPECT_EQ(momentCount, Real::count());

        // The regular central-image series has the same source operation count
        // at zero as elsewhere; near cells now bypass the entire Green calculation.
        for (auto const& sample : std::array<Vector3<double>, 2>{{{0.0, 0.0, 0.0}, point}}) {
            for (bool near : {true, false}) {
                if (!near && sample[0] == 0.0 && sample[1] == 0.0 && sample[2] == 0.0) {
                    continue;
                }
                SCOPED_TRACE(near ? "near Green" : "far Green");
                std::array<double, order * order> coefficients{};
                FlopCount const greenCount =
                    pmmm::green(order, coefficients.data(), sample, kCellWidth, near);
                if (near) {
                    EXPECT_EQ(greenCount, 0);
                    continue;
                }
                std::array<Real, order * order> countedCoefficients{};
                Real trace{};
                Real::reset();
                spherical::evaluateEwaldGreen(
                    trace, Real(sample[0]), Real(sample[1]), Real(sample[2]), countedCoefficients);
                EXPECT_EQ(spherical::kEwaldGreenConventionalFlops<kDegree>, Real::count());
                if (!near) {
                    std::array<Real, order * order> primary{};
                    spherical::detail::evaluateGreen(
                        Real(sample[0]), Real(sample[1]), Real(sample[2]), primary);
                    for (size_t slot = 0; slot < primary.size(); ++slot) {
                        countedCoefficients[slot] += primary[slot];
                    }
                }
                Real scale = 1.0;
                for (int rank = 0; rank <= kDegree; ++rank) {
                    for (int slot = rank * rank; slot < (rank + 1) * (rank + 1); ++slot) {
                        countedCoefficients[slot] *= scale;
                    }
                    scale *= Real(kCellWidth);
                }
                EXPECT_EQ(greenCount, Real::count());
            }
        }
    }

}    // namespace detail
}    // namespace pmmm_test

class PmmmKernelFlops : public ::testing::TestWithParam<int> {};

TEST_P(PmmmKernelFlops, SourceOperationsMatchExecutedArithmetic) {
    {
        pmmm_test::detail::checkKernelCounts<kFmmOrder>();
    };
}

INSTANTIATE_TEST_SUITE_P(AllOrders, PmmmKernelFlops, ::testing::Range(kFmmOrder, kFmmOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });

namespace cosmictiger::gravity::cartesian::detail {
template <>
inline ::detail::CountingReal ewaldExp(::detail::CountingReal value) {
    ::detail::CountingReal::add(1);
    return std::exp(value.value());
}
template <>
inline ::detail::CountingReal ewaldErfc(::detail::CountingReal value) {
    ::detail::CountingReal::add(1);
    return std::erfc(value.value());
}
template <>
inline ::detail::CountingReal ewaldSin(::detail::CountingReal value) {
    ::detail::CountingReal::add(1);
    return std::sin(value.value());
}
template <>
inline ::detail::CountingReal ewaldCos(::detail::CountingReal value) {
    ::detail::CountingReal::add(1);
    return std::cos(value.value());
}
}    // namespace cosmictiger::gravity::cartesian::detail

TEST(CartesianEwaldFlops, GeneratedKernelsMatchExecutedArithmetic) {
    using Real = ::detail::CountingReal;
    namespace cartesian = cosmictiger::gravity::cartesian;
    Vector3<Real> const position{.13, -.21, .38};
    cartesian::Green<Real> green{};
    Real::reset();
    cartesian::evaluateEwaldGreen(position, green);
    // Three scalar float sign/image products are outside the counted Real.
    constexpr size_t kImagePairs =
        sizeof(cartesian::detail::kEwaldImages) / sizeof(cartesian::detail::kEwaldImages[0]);
    EXPECT_EQ(cartesian::kEwaldGreenFlops, Real::count() + 6 * kImagePairs);
    cartesian::Multipole<Real> source{};
    cartesian::Expansion<Real> local{};
    source[0] = 1;
    Real::reset();
    cartesian::M2L(source, green, local);
    EXPECT_EQ(cartesian::kEwaldM2LFlops, Real::count());
}
