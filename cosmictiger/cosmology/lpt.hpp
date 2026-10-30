/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#pragma once

namespace lpt {

inline constexpr int kIdentityDerivative = 3;
inline constexpr int kSecondOrderProductPhases = 6;

// Indices 0, 1 and 2 differentiate the potential; index 3 leaves it unchanged.
// Only a first derivative is interpolated directly to particles. Hessians are
// multiplied on the FFT grid, where interpolation deconvolution would amplify
// the second-order source. Its final displacement is deconvolved separately.
template <typename Value, typename Filter>
inline Value interpolationCorrectedDerivative(
    Value value, int firstDimension, int secondDimension, Filter const& filter) {
    if ((firstDimension == kIdentityDerivative) != (secondDimension == kIdentityDerivative)) {
        return filter(value);
    }
    return value;
}

// The production sequence is yy, zz, xx, zz, xx, yy, xy, xz, yz. It forms
// S = phi_xx phi_yy + phi_xx phi_zz + phi_yy phi_zz
//     - phi_xy^2 - phi_xz^2 - phi_yz^2.
template <typename Real>
inline void accumulateSecondOrderSource(
    int phase, Real derivative, Real& previousDerivative, Real& source) {
    if (phase >= kSecondOrderProductPhases) {
        source -= derivative * derivative;
    } else if (phase % 2 == 0) {
        previousDerivative = derivative;
    } else {
        source += previousDerivative * derivative;
    }
}

}    // namespace lpt
