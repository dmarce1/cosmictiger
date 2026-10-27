/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <array>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/fmm_kernels/spherical_ewald_kernels.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/nitadori/kernels.hpp>
#include <tests/flop_counting_real.hpp>

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
        std::array<double, 3> const point{0.13, -0.21, 0.38};
        std::array<double, (order - 1) * (order - 1)> moments{};
        std::array<Real, moments.size()> countedMoments{};
        FlopCount const momentCount = pmmm::addMoments(order, moments.data(), 0.7, point);
        Real::reset();
        spherical::detail::P2M<order - 2>::template add<Real>(
            countedMoments, Real(0.7), Real(point[0]), Real(point[1]), Real(point[2]));
        EXPECT_EQ(momentCount, Real::count());

        // The regular central-image series has the same source operation count
        // at zero as elsewhere; near controls only addition of the primary Green.
        for (auto const& sample : std::array<std::array<double, 3>, 2>{{{0.0, 0.0, 0.0}, point}}) {
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
                spherical::EwaldGreen<kDegree>::template evaluate<Real>(
                    countedCoefficients, trace, Real(sample[0]), Real(sample[1]), Real(sample[2]));
                EXPECT_EQ(spherical::EwaldGreen<kDegree>::kConventionalFlops_, Real::count());
                if (!near) {
                    std::array<Real, order * order> primary{};
                    spherical::detail::Green<kDegree>::template evaluate<Real>(
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

        std::array<double, order * order> local{};
        std::array<Real, order * order> countedLocal{};
        for (size_t slot = 0; slot < local.size(); ++slot) {
            local[slot] = 0.01 * (slot + 1);
            countedLocal[slot] = local[slot];
        }
        std::array<double, 4> field{};
        FlopCount const evaluationCount =
            pmmm::evaluate(order, field, local.data(), 0.2, point, kCellWidth);
        std::array<Real, 4> countedField{};
        Real::reset();
        spherical::EwaldL2P<kDegree>::template add<Real>(
            countedField, countedLocal, Real(0.2), Real(point[0]), Real(point[1]), Real(point[2]));
        EXPECT_EQ(spherical::EwaldL2P<kDegree>::kConventionalFlops_, Real::count());
        for (int component = 1; component < 4; ++component) {
            countedField[component] = countedField[component] / Real(kCellWidth);
        }
        EXPECT_EQ(evaluationCount, Real::count());

        std::array<Real, order * order> translated{};
        Real translatedTrace{};
        Real::reset();
        spherical::EwaldL2L<kDegree>::template add<Real>(translated, translatedTrace, countedLocal,
            Real(0.2), Real(point[0]), Real(point[1]), Real(point[2]));
        EXPECT_EQ(spherical::EwaldL2L<kDegree>::kConventionalFlops_, Real::count());

        Real::reset();
        spherical::EwaldM2L<kDegree>::template add<Real>(
            translated, translatedTrace, countedMoments, Real(0.1), countedLocal, Real(0.2));
        EXPECT_EQ(spherical::EwaldM2L<kDegree>::kConventionalFlops_, Real::count());
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
