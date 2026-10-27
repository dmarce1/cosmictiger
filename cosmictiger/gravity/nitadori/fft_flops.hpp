/* CosmicTiger - PMMM FFT operation accounting; BSL-1.0. */
#pragma once

#include <cmath>
#include <stdexcept>

#include <fftw3.h>

#include <cosmictiger/core/flops.hpp>

namespace pmmm::detail {

// FFTW reports the arithmetic in the actual plan, including its batched
// transforms. For MPI plans this query is collective and returns a GLOBAL sum
// on every rank. Cache it under the planner lock and submit on one rank only.
// Transposes and communication are copies, not arithmetic operations.
inline FlopCount fftPlanFlops(fftwf_plan plan) {
    double additions = 0.0;
    double multiplications = 0.0;
    double fusedMultiplyAdds = 0.0;
    fftwf_flops(plan, &additions, &multiplications, &fusedMultiplyAdds);
    double const total = additions + multiplications + 2 * fusedMultiplyAdds;
    if (!(total >= 0.0) || !std::isfinite(total) || total >= 0x1p64 || std::floor(total) != total) {
        throw std::overflow_error("PMMM FFT operation count is not a 64-bit integer");
    }
    return static_cast<FlopCount>(total);
}

template <class Real>
inline void accumulateFourierTerm(Real& real, Real& imaginary, Real weight, Real sourceReal,
    Real sourceImaginary, Real kernelReal, Real kernelImaginary) {
    real += weight * (sourceReal * kernelReal - sourceImaginary * kernelImaginary);
    imaginary += weight * (sourceReal * kernelImaginary + sourceImaginary * kernelReal);
}

// One complex product, two real weights, and accumulation into both components.
inline constexpr FlopCount kFourierTermFlops = 10;

}    // namespace pmmm::detail
