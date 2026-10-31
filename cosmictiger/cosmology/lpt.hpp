/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>

namespace lpt {

inline constexpr int kIdentityDerivative = 3;
inline constexpr int kSecondOrderProductPhases = 6;

// A counter-based stream gives each full Fourier mode independent draws. Only
// the kz=0 conjugate pair shares a key. Nyquist modes are discarded so arbitrary
// lattice shifts and odd derivatives remain Hermitian on the real FFT grid.
inline uint64_t mixModeBits(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

inline bool isNyquist(int64_t index, int64_t size) {
    return size % 2 == 0 && index == size / 2;
}

inline std::complex<float> gaussianMode(int64_t x, int64_t y, int64_t z, int64_t size) {
    if ((x == 0 && y == 0 && z == 0) || isNyquist(x, size) || isNyquist(y, size) ||
        isNyquist(z, size)) {
        return {};
    }
    bool conjugate = false;
    if (z == 0) {
        int64_t const partnerX = (size - x) % size;
        int64_t const partnerY = (size - y) % size;
        if (partnerX < x || (partnerX == x && partnerY < y)) {
            x = partnerX;
            y = partnerY;
            conjugate = true;
        }
    }
    uint64_t const key =
        (uint64_t(x) * uint64_t(size) + uint64_t(y)) * uint64_t(size) + uint64_t(z);
    auto const uniform = [](uint64_t bits) { return (double(bits >> 12) + 0.5) * 0x1p-52; };
    double const radius = std::sqrt(-std::log(uniform(mixModeBits(2 * key + 42))));
    double const phase = 2.0 * std::numbers::pi * uniform(mixModeBits(2 * key + 43));
    std::complex<float> const result(radius * std::cos(phase), radius * std::sin(phase));
    return conjugate ? std::conj(result) : result;
}

inline double latticeOffset(int phase, bool closePack) {
    if (!closePack) {
        return 0.5 * phase;
    }
    return 0.25 + 0.5 * (phase % 2) + 0.25 * (phase / 2);
}

inline int64_t latticeNode(double position, int64_t size, double offset) {
    int64_t const node = std::llround(position * size - offset);
    return (node + size) % size;
}

inline std::complex<float> latticeShift(
    int64_t x, int64_t y, int64_t z, int64_t size, double offset) {
    auto const signedMode = [size](
                                int64_t index) { return index <= size / 2 ? index : index - size; };
    double const phase =
        2.0 * std::numbers::pi * offset / size * (signedMode(x) + signedMode(y) + signedMode(z));
    return {float(std::cos(phase)), float(std::sin(phase))};
}

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
