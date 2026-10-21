/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cstdio>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/gravity.hpp>
#include "flop_counting_real.hpp"

namespace detail {

void requireCount(char const* label, FlopCount reported, FlopCount executed) {
    if (reported != executed) {
        throw std::runtime_error(std::string(label) + " reported " + std::to_string(reported) +
            " operations, executed " + std::to_string(executed));
    }
}

void checkSoftening(radial_kernel::KernelTable const& kernel, double radius, bool doPhi) {
    using Real = CountingReal;
    gravityKernel = kernel;
    Real force;
    Real potential;
    Real const radiusSquared(radius * radius);
    Real const unit(1);
    // CountingReal follows the same eager, all-spans path as a SIMD packet.
    // The width is one so scalar table-width subtractions remain distinguishable
    // from the intercepted Real arithmetic. Each span evaluates b-a once per
    // force/potential call using double, outside the overloaded scalar oracle.
    CountingReal::reset();
    FlopCount const packetReported =
        gravitySofteningKernel(force, potential, radiusSquared, unit, unit, unit, doPhi);
    FlopCount const packetExecuted = CountingReal::count() + kernel.pieceCount * (1 + int(doPhi));
    requireCount("SIMD softening", packetReported, packetExecuted);

    int selected = kernel.pieceCount;
    for (int piece = 0; piece < kernel.pieceCount; ++piece) {
        if (radius < kernel.changePoints[piece + 1]) {
            selected = piece;
            break;
        }
    }
    // Execute the actual scalar-selected arithmetic routines with CountingReal.
    // This avoids specializing std::is_arithmetic for the instrumentation type.
    CountingReal::reset();
    Real const q = simdSqrt(radiusSquared * unit);
    if (selected < kernel.pieceCount) {
        force = radial_kernel::detail::forceOnPiece(kernel, selected, q) * unit;
        if (doPhi) {
            potential = radial_kernel::detail::potentialOnPiece(kernel, selected, q) * unit;
        }
    } else {
        Real const inverseRadius = unit / q;
        force = inverseRadius * inverseRadius * inverseRadius * unit;
        if (doPhi) {
            potential = (unit / q) * unit;
        }
    }
    FlopCount const scalarExecuted =
        CountingReal::count() + (selected < kernel.pieceCount ? 1 + int(doPhi) : 0);
    requireCount("Scalar softening", gravitySofteningFlops(kernel, radius, doPhi), scalarExecuted);
}

}    // namespace detail

TEST(SofteningFlops, ScalarAndSimdCounts) {
    for (auto family : {radial_kernel::Family::Spline, radial_kernel::Family::Wendland,
             radial_kernel::Family::Ferrers}) {
        for (int degree = 2; degree <= radial_kernel::kMaximumDegree; ++degree) {
            radial_kernel::KernelSpec const spec{family, degree};
            if (!radial_kernel::isValid(spec)) {
                continue;
            }
            auto const kernel = radial_kernel::makeKernelTable(spec);
            for (bool doPhi : {false, true}) {
                for (int piece = 0; piece < kernel.pieceCount; ++piece) {
                    detail::checkSoftening(kernel, kernel.changePoints[piece], doPhi);
                    detail::checkSoftening(kernel,
                        .5 * (kernel.changePoints[piece] + kernel.changePoints[piece + 1]), doPhi);
                }
                detail::checkSoftening(kernel, 1., doPhi);
                detail::checkSoftening(kernel, 1.5, doPhi);
            }
        }
    }
}
