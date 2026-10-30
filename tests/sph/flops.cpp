/* CosmicTiger - SPH arithmetic-count regression; BSL-1.0. */

#include <array>

#include <gtest/gtest.h>

#include <cosmictiger/sph/monaghan/viscosity.hpp>
#include <cosmictiger/sph/sph_flops.hpp>

#include <tests/flop_counting_real.hpp>

namespace detail {

void checkSphRadialCounts(radial_kernel::KernelTable const& table, double radius) {
    using Real = CountingReal;
    Real const q(radius);
    for (bool derivative : {false, true}) {
        Real::reset();
        if (derivative) {
            radial_kernel::derivative(table, q);
        } else {
            radial_kernel::value(table, q);
        }
        // Width subtraction uses double, outside the intercepted Real arithmetic.
        EXPECT_EQ(
            sph_flops::densityKernelFlops(table, q, derivative), Real::count() + table.pieceCount);
        Real::reset();
        int selected = table.pieceCount;
        for (int piece = 0; piece < table.pieceCount; ++piece) {
            if (radius < table.changePoints[piece + 1]) {
                selected = piece;
                auto const* coefficients =
                    derivative ? table.derivativeCoefficients[piece] : table.coefficients[piece];
                radial_kernel::detail::evaluateDensitySpan(coefficients,
                    table.degree - int(derivative), q, table.changePoints[piece],
                    table.changePoints[piece + 1]);
                break;
            }
        }
        EXPECT_EQ(sph_flops::densityKernelFlops(table, radius, derivative),
            Real::count() + (selected < table.pieceCount ? 1 : 0));
    }
    Real::reset();
    radial_kernel::forceFactor(table, q);
    EXPECT_EQ(sph_flops::forceKernelFlops(table, q), Real::count() + table.pieceCount);
    Real::reset();
    bool selected = false;
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        if (radius < table.changePoints[piece + 1]) {
            radial_kernel::detail::forceOnPiece(table, piece, q);
            selected = true;
            break;
        }
    }
    if (!selected) {
        auto const inverse = Real(1) / q;
        auto const force = inverse * inverse * inverse;
        (void) force;
    }
    EXPECT_EQ(sph_flops::forceKernelFlops(table, radius), Real::count() + int(selected));
}

}    // namespace detail

TEST(SphFlops, EveryRadialFamilySpanAndOutsideBranch) {
    for (auto family : {radial_kernel::Family::Spline, radial_kernel::Family::Wendland,
             radial_kernel::Family::Ferrers}) {
        for (int degree = 2; degree <= radial_kernel::kMaximumDegree; ++degree) {
            radial_kernel::KernelSpec const spec{family, degree};
            if (!radial_kernel::isValid(spec)) {
                continue;
            }
            auto const table = radial_kernel::makeKernelTable(spec);
            for (int piece = 0; piece < table.pieceCount; ++piece) {
                detail::checkSphRadialCounts(table, table.changePoints[piece]);
                detail::checkSphRadialCounts(
                    table, 0.5 * (table.changePoints[piece] + table.changePoints[piece + 1]));
            }
            detail::checkSphRadialCounts(table, 1.0);
            detail::checkSphRadialCounts(table, 1.5);
        }
    }
}

TEST(SphFlops, MonaghanExpressionsMatchExecutedArithmetic) {
    using Real = detail::CountingReal;
    for (double velocity : {-0.3, 0.0, 0.3}) {
        Real::reset();
        auto const mu = monaghan::viscosityMu(
            Real(0.1), Real(velocity), Real(0.4), Real(0.5), Real(0.01), Real(2), Real(1));
        EXPECT_EQ(Real::count(), sph_flops::kViscosityMu);
        Real::reset();
        monaghan::artificialViscosity(Real(1), Real(2), Real(3), Real(4), mu, Real(1), Real(2));
        EXPECT_EQ(Real::count(), sph_flops::kArtificialViscosity);
    }
}
