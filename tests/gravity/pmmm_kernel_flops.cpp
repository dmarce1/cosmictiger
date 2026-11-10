/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <array>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kernels/spherical_ewald_kernels.hpp>
#include <cosmictiger/gravity/pmmm_kernels.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <tests/flop_counting_real.hpp>

// The generated scalar/SIMD adapters use std math internally. Specialize only
// those adapters so the execution oracle also observes each elementary result.
#include "pmmm_reference.hpp"

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
        spherical::detail::P2M<order - 2, Real>(
            countedMoments, Real(0.7), Real(point[0]), Real(point[1]), Real(point[2]));
        EXPECT_EQ(momentCount, Real::count());

        // The regular central-image series has the same source operation count
        // at zero as elsewhere; near controls only addition of the primary Green.
        for (auto const& sample : std::array<Vector3<double>, 2>{{{0.0, 0.0, 0.0}, point}}) {
            for (bool near : {true, false}) {
                if (!near && sample[0] == 0.0 && sample[1] == 0.0 && sample[2] == 0.0) {
                    continue;
                }
                SCOPED_TRACE(near ? "near Green" : "far Green");
                std::array<double, order * order> coefficients{};
                FlopCount const greenCount =
                    pmmm::green(order, coefficients.data(), sample, kCellWidth, near);
                std::array<Real, order * order> countedCoefficients{};
                Real trace{};
                Real::reset();
                spherical::evaluateEwaldGreen<kDegree, Real>(
                    countedCoefficients, trace, Real(sample[0]), Real(sample[1]), Real(sample[2]));
                EXPECT_EQ(spherical::kEwaldGreenConventionalFlops<kDegree>, Real::count());
                if (!near) {
                    std::array<Real, order * order> primary{};
                    spherical::detail::evaluateGreen<kDegree, Real>(
                        primary, Real(sample[0]), Real(sample[1]), Real(sample[2]));
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
    fmmDispatch(GetParam(), [](auto selectedOrder) {
        pmmm_test::detail::checkKernelCounts<decltype(selectedOrder)::value>();
    });
}

INSTANTIATE_TEST_SUITE_P(AllOrders, PmmmKernelFlops,
    ::testing::Range(kFmmMinOrder, kFmmMaxOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
