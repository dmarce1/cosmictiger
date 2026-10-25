/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/sph/kernels.hpp>
#include <tests/sph/test_helpers.hpp>

namespace detail {

constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename Real>
void testKernel() {
    double const roundoff = 32.0 * std::numeric_limits<Real>::epsilon();
    requireClose(
        "central kernel normalization", sph_kernels::kernelValue(Real(0)), 8.0 / kPi, roundoff);
    requireClose(
        "kernel at the spline join", sph_kernels::kernelValue(Real(0.5)), 2.0 / kPi, roundoff);
    requireClose("gradient at the spline join", sph_kernels::kernelDerivative(Real(0.5)),
        -12.0 / kPi, roundoff);
    require(sph_kernels::kernelDerivative(Real(0)) == Real(0), "finite central gradient");
    for (Real const q : {Real(1), Real(1.25), Real(2)}) {
        require(sph_kernels::kernelValue(q) == Real(0), "zero kernel outside support");
        require(sph_kernels::kernelDerivative(q) == Real(0), "zero gradient outside support");
    }

    // A radial volume integral checks the actual three-dimensional normalization.
    constexpr int kIntervals = 4096;
    double integral = 0.0;
    for (int i = 0; i <= kIntervals; i++) {
        double const q = double(i) / kIntervals;
        double const weight = (i == 0 || i == kIntervals) ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
        double const value = sph_kernels::kernelValue(Real(q));
        require(value >= 0.0 && std::isfinite(value), "nonnegative finite kernel");
        integral += weight * 4.0 * kPi * q * q * value;
    }
    integral /= 3.0 * kIntervals;
    requireClose(
        "unit three-dimensional kernel integral", integral, 1.0, std::max(roundoff, 2.0e-12));

    Real const step = sizeof(Real) == sizeof(float) ? Real(5.0e-4) : Real(1.0e-6);
    double const derivativeTolerance = sizeof(Real) == sizeof(float) ? 8.0e-4 : 2.0e-9;
    for (Real const q :
        {Real(0.07), Real(0.25), Real(0.49), Real(0.5), Real(0.51), Real(0.79), Real(0.99)}) {
        Real const finiteDifference =
            (sph_kernels::kernelValue(q + step) - sph_kernels::kernelValue(q - step)) /
            (Real(2) * step);
        requireClose("kernel gradient finite difference", sph_kernels::kernelDerivative(q),
            finiteDifference, derivativeTolerance);
    }
    Real const joinStep = sizeof(Real) == sizeof(float) ? Real(1.0e-5) : Real(1.0e-7);
    for (Real const join : {Real(0.5), Real(1)}) {
        requireClose("continuous kernel", sph_kernels::kernelValue(join - joinStep),
            sph_kernels::kernelValue(join + joinStep), 20.0 * joinStep);
        requireClose("continuous kernel gradient", sph_kernels::kernelDerivative(join - joinStep),
            sph_kernels::kernelDerivative(join + joinStep), 80.0 * joinStep);
    }

    // Compare against the paper's h-based polynomial, where support H is 2h.
    double const smoothingLength = 0.37;
    double const support = 2.0 * smoothingLength;
    for (double const qPaper : {0.0, 0.2, 0.75, 1.0, 1.25, 1.75, 2.0}) {
        double const radius = smoothingLength * qPaper;
        double const shape = qPaper <= 1.0 ?
            1.0 - 1.5 * qPaper * qPaper + 0.75 * qPaper * qPaper * qPaper :
            0.25 * std::pow(2.0 - qPaper, 3);
        double const expected = shape / (kPi * std::pow(smoothingLength, 3));
        double const actual =
            sph_kernels::kernelValue(Real(radius / support)) / std::pow(support, 3);
        requireClose("support radius equals twice paper h", actual, expected, roundoff);
    }
}

}    // namespace detail

TEST(SphKernels, KernelDouble) {
    detail::testKernel<double>();
}

TEST(SphKernels, KernelFloat) {
    detail::testKernel<float>();
}
