/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/fmm_acceptance.hpp>
#include <cosmictiger/gravity/fmm_kernels/cartesian_ewald_operators.hpp>
#include <cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp>
#include <tests/flop_counting_real.hpp>

// Exercise the real tables without a distributed runtime, as in
// cartesian_ewald_cpu.
Vector<hpx::id_type> const& hpxChildren() {
    static Vector<hpx::id_type> const children;
    return children;
}

void cosmictigerAssert(char const* condition, bool satisfied, char const* file, int line) {
    if (!satisfied) {
        std::fprintf(stderr, "%s:%d: assertion \"%s\" failed\n", file, line, condition);
        throw std::runtime_error("Regression check failed; see diagnostic above");
    }
}

namespace detail {

template <class Function>
void checkCount(char const* label, Function&& operation) {
    CountingReal::reset();
    int const reported = operation();
    uint64_t const executed = CountingReal::count();
    if (reported < 0 || uint64_t(reported) != executed) {
        throw std::runtime_error(std::string(label) + " reported " + std::to_string(reported) +
            " operations, executed " + std::to_string(executed));
    }
}

template <int order>
void checkOrder() {
    using Real = CountingReal;
    cosmictiger::gravity::cartesian::Multipole<Real, order> source{};
    cosmictiger::gravity::cartesian::Expansion<Real, order> local{};
    cosmictiger::gravity::cartesian::Green<Real, order> derivatives{};
    cosmictiger::gravity::Field<Real> field{};
    Array<Real, kDimensions> position;
    position[0] = .13;
    position[1] = -.21;
    position[2] = .38;
    for (int i = 0; i < source.kSize_; ++i) {
        source[i] = .01 * (i + 1);
    }
    for (int i = 0; i < local.kSize_; ++i) {
        local[i] = .02 * (i + 1);
        derivatives[i] = .03 * (i + 1);
    }
    checkCount("P2M", [&] {
        cosmictiger::gravity::cartesian::Multipole<Real, order> result{};
        return cosmictiger::gravity::cartesian::P2M(
                   result, Real(1), fmmScaleDisplacement(position)) +
            3;
    });
    checkCount("M2M", [&] {
        cosmictiger::gravity::cartesian::Multipole<Real, order> result{};
        return cosmictiger::gravity::cartesian::M2M(
                   result, source, fmmScaleDisplacement(position)) +
            3;
    });
    for (bool enabled : {false, true}) {
        checkCount("P2L", [&] {
            return cosmictiger::gravity::cartesian::P2L(
                       local, Real(1), fmmScaleDisplacement(position), enabled) +
                3;
        });
        checkCount("M2L", [&] {
            return cosmictiger::gravity::cartesian::M2L(
                       local, source, fmmScaleDisplacement(position), enabled) +
                3;
        });
        checkCount("M2P", [&] {
            return cosmictiger::gravity::cartesian::M2P(
                       field, source, fmmScaleDisplacement(position), enabled) +
                3;
        });
        checkCount("Ewald M2L", [&] {
            return cosmictiger::gravity::cartesian::M2L(local, source, derivatives, enabled);
        });
        checkCount("Centered M2L", [&] {
            return cosmictiger::gravity::cartesian::M2L<true>(
                       local, source, fmmScaleDisplacement(position), enabled) +
                3;
        });
        checkCount("Centered M2P", [&] {
            return cosmictiger::gravity::cartesian::M2P<true>(
                       field, source, fmmScaleDisplacement(position), enabled) +
                3;
        });
        checkCount("Centered Ewald M2L", [&] {
            return cosmictiger::gravity::cartesian::M2L<true>(local, source, derivatives, enabled);
        });
        checkCount("L2L", [&] {
            cosmictiger::gravity::cartesian::Expansion<Real, order> result{};
            return cosmictiger::gravity::cartesian::L2L(
                       result, local, fmmScaleDisplacement(position), enabled) +
                3;
        });
        checkCount("L2P", [&] {
            cosmictiger::gravity::Field<Real> result{};
            return cosmictiger::gravity::cartesian::L2P(
                       result, local, fmmScaleDisplacement(position), enabled) +
                3;
        });
    }

    setFmmOrder(order);
    EwaldConst::init();
    // Different points exercise both outcomes of the real-image cutoff and
    // different accepted-image counts, including the self-origin selection.
    for (auto const& sample :
        std::array<std::array<double, 3>, 3>{{{0., 0., 0.}, {.13, -.21, .38}, {.49, -.49, .49}}}) {
        Array<Real, kDimensions> point;
        for (int dim = 0; dim < kDimensions; ++dim) {
            point[dim] = sample[dim];
        }
        SymmetricTracelessTensor<Real, order> result{};
        checkCount("Ewald derivatives", [&] { return ewaldCorrectionDerivatives(result, point); });
        checkCount("Ewald Green function", [&] {
            return cosmictiger::gravity::cartesian::evaluateEwaldGreen(derivatives, point);
        });
    }
    std::printf("FMM order %d source-operation counts match executed arithmetic.\n", order);
}

}    // namespace detail

TEST(FmmFlopAcceptance, ScalarShortCircuitCounts) {
    using Real = detail::CountingReal;
    std::array<bool, 3> observedCounts{};
    for (double sinkRadius : {0.08, 0.19}) {
        double const sourceRadius = 0.27 - sinkRadius;
        for (double distanceSquared : {0.01, 0.14, 4.0}) {
            for (double boxDistanceSquared : {0.0, 4.0}) {
                FlopCount reported = 0;
                auto const expected = computeFmmAcceptance(distanceSquared, sinkRadius,
                    sourceRadius, boxDistanceSquared, boxDistanceSquared, 0.05, 2.0, &reported);
                // Run the actual expressions with overloaded arithmetic. CountingReal
                // comparisons return bool, so the same scalar short circuits execute.
                Real::reset();
                auto const measured = computeFmmAcceptance(Real(distanceSquared), Real(sinkRadius),
                    Real(sourceRadius), Real(boxDistanceSquared), Real(boxDistanceSquared),
                    Real(0.05), Real(2.0));
                EXPECT_EQ(reported, Real::count());
                EXPECT_EQ(expected.cellCell, measured.cellCell);
                EXPECT_EQ(expected.particleCell, measured.particleCell);
                EXPECT_EQ(expected.cellParticle, measured.cellParticle);
                EXPECT_EQ(expected.supportSeparated, measured.supportSeparated);
                ASSERT_GE(reported, 17);
                ASSERT_LE(reported, 19);
                observedCounts[reported - 17] = true;
            }
        }
    }
    EXPECT_TRUE(std::all_of(
        observedCounts.begin(), observedCounts.end(), [](bool observed) { return observed; }));
}

class FmmFlops : public ::testing::TestWithParam<int> {};

TEST_P(FmmFlops, SupportedOrder) {
    fmmDispatch(GetParam(),
        [](auto selectedOrder) { detail::checkOrder<decltype(selectedOrder)::value>(); });
}

INSTANTIATE_TEST_SUITE_P(AllOrders, FmmFlops, ::testing::Range(kFmmMinOrder, kFmmMaxOrder + 1),
    [](::testing::TestParamInfo<int> const& info) { return "Order" + std::to_string(info.param); });
