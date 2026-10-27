/* CosmicTiger - PMMM direct-pair arithmetic-count regression; BSL-1.0. */

#include <array>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/nitadori/near_pp.hpp>
#include <cosmictiger/math/radial_kernel.hpp>
#include <tests/flop_counting_real.hpp>

namespace detail {

// CountingReal is deliberately not a built-in arithmetic type. Select the
// scalar radial branch here, then execute the actual production expressions
// with overloaded arithmetic instead of the eager SIMD path.
struct CountingNearKernel : radial_kernel::KernelTable {
    mutable uint64_t widthOperations = 0;

    explicit CountingNearKernel(radial_kernel::KernelTable const& table)
      : radial_kernel::KernelTable(table) {}
};

CountingReal forceFactor(CountingNearKernel const& kernel, CountingReal q) {
    for (int piece = 0; piece < kernel.pieceCount; ++piece) {
        if (q < CountingReal(kernel.changePoints[piece + 1])) {
            ++kernel.widthOperations;
            return radial_kernel::detail::forceOnPiece(kernel, piece, q);
        }
    }
    auto const inverseRadius = CountingReal(1) / q;
    return inverseRadius * inverseRadius * inverseRadius;
}

CountingReal potential(CountingNearKernel const& kernel, CountingReal q) {
    for (int piece = 0; piece < kernel.pieceCount; ++piece) {
        if (q < CountingReal(kernel.changePoints[piece + 1])) {
            ++kernel.widthOperations;
            return radial_kernel::detail::potentialOnPiece(kernel, piece, q);
        }
    }
    return CountingReal(1) / q;
}

void checkNearPairCounts(radial_kernel::KernelTable const& table, double radius) {
    using Real = CountingReal;
    PmmmNearParticle const sink{{0, 0, 0}, 2, 0, 1};
    PmmmNearParticle const source{{radius, 0, 0}, 1.75, 0, 2};
    for (bool doPotential : {false, true}) {
        std::array<double, 4> field{0.5, 0.75, 1, 1.25};
        auto const reported = pmmm::detail::accumulateNearPair<true>(
            sink, source, 0.125, 8.0, 512.0, doPotential, table, field);
        CountingNearKernel kernel(table);
        std::array<Real, 4> measuredField{0.5, 0.75, 1, 1.25};
        Real::reset();
        auto const measuredReport = pmmm::detail::accumulateNearPair<true>(
            sink, source, Real(0.125), Real(8), Real(512), doPotential, kernel, measuredField);
        // The span-width subtraction remains double in the radial table code;
        // every other floating operation executes through CountingReal.
        EXPECT_EQ(reported, Real::count() + kernel.widthOperations);
        EXPECT_EQ(measuredReport, reported);
        for (size_t component = 0; component < field.size(); ++component) {
            EXPECT_DOUBLE_EQ(field[component], measuredField[component].value());
        }
        if (!doPotential) {
            EXPECT_DOUBLE_EQ(field[0], 0.5);
        }
    }
}

}    // namespace detail

TEST(PmmmNearFlops, EveryRadialFamilySpanBoundaryAndOutsideBranch) {
    for (auto family : {radial_kernel::Family::Spline, radial_kernel::Family::Wendland,
             radial_kernel::Family::Ferrers}) {
        for (int degree = 2; degree <= radial_kernel::kMaximumDegree; ++degree) {
            radial_kernel::KernelSpec const spec{family, degree};
            if (!radial_kernel::isValid(spec)) {
                continue;
            }
            auto const table = radial_kernel::makeKernelTable(spec);
            for (int piece = 0; piece < table.pieceCount; ++piece) {
                detail::checkNearPairCounts(table, table.changePoints[piece]);
                detail::checkNearPairCounts(
                    table, 0.5 * (table.changePoints[piece] + table.changePoints[piece + 1]));
            }
            detail::checkNearPairCounts(table, 1.0);
            detail::checkNearPairCounts(table, 1.5);
        }
    }
}

TEST(PmmmNearFlops, IdentityExclusionAndDisabledCountingPreserveFields) {
    auto const table = radial_kernel::makeKernelTable({radial_kernel::Family::Spline, 3});
    PmmmNearParticle const sink{{0, 0, 0}, 1, 3, 5};
    std::array<double, 4> const initial{1, 2, 3, 4};
    for (int rank : {3, 4}) {
        for (int64_t index : {5, 6}) {
            PmmmNearParticle const source{{0.25, -0.5, 0.75}, 2, rank, index};
            auto field = initial;
            auto disabledField = initial;
            auto const reported = pmmm::detail::accumulateNearPair<true>(
                sink, source, 0.125, 8.0, 512.0, true, table, field);
            EXPECT_EQ(pmmm::detail::accumulateNearPair<false>(
                          sink, source, 0.125, 8.0, 512.0, true, table, disabledField),
                0);
            EXPECT_EQ(field, disabledField);
            if (rank == sink.rank && index == sink.index) {
                EXPECT_EQ(reported, 0);
                EXPECT_EQ(field, initial);
            } else {
                EXPECT_GT(reported, 0);
                EXPECT_NE(field, initial);
            }
        }
    }
}

TEST(PmmmNearFlops, PairCountsAccumulateBeyond32Bits) {
    auto const table = radial_kernel::makeKernelTable({radial_kernel::Family::Ferrers, 2});
    PmmmNearParticle const sink{{0, 0, 0}, 1, 0, 1};
    PmmmNearParticle const source{{2, 0, 0}, 1, 0, 2};
    std::array<double, 4> field{};
    auto const reported =
        pmmm::detail::accumulateNearPair<true>(sink, source, 1.0, 1.0, 1.0, true, table, field);
    FlopCounter<FlopCount, true> total = std::numeric_limits<uint32_t>::max();
    total += reported;
    EXPECT_EQ(FlopCount(total), FlopCount(std::numeric_limits<uint32_t>::max()) + reported);
    EXPECT_GT(FlopCount(total), FlopCount(std::numeric_limits<uint32_t>::max()));
}
