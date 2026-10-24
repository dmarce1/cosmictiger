#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace particle_mesh {

struct CloudWeights {
    int first = 0;
    int count = 0;
    std::array<double, 4> weights{};
};

inline int cloudOrder(std::string const& cloud) {
    if (cloud == "ngp") {
        return 1;
    }
    if (cloud == "cic") {
        return 2;
    }
    if (cloud == "tsc") {
        return 3;
    }
    if (cloud == "pcs") {
        return 4;
    }
    throw std::invalid_argument("gravity.cloud must be ngp, cic, tsc, or pcs");
}

// Centered cardinal B-splines, with unit knot spacing. The order is the number
// of grid points in one dimension, one greater than the polynomial degree.
inline CloudWeights cloudWeightsInCell(int cell, double fraction, int order) {
    CloudWeights result;
    result.count = order;
    if (order == 1) {
        result.first = cell + (fraction >= 0.5);
        result.weights[0] = 1;
    } else if (order == 2) {
        result.first = cell;
        result.weights[0] = 1 - fraction;
        result.weights[1] = fraction;
    } else if (order == 3) {
        int const nearest = cell + (fraction >= 0.5);
        double const offset = fraction - (fraction >= 0.5);
        result.first = nearest - 1;
        result.weights[0] = 0.5 * (0.5 - offset) * (0.5 - offset);
        result.weights[1] = 0.75 - offset * offset;
        result.weights[2] = 0.5 * (0.5 + offset) * (0.5 + offset);
    } else if (order == 4) {
        double const complement = 1 - fraction;
        result.first = cell - 1;
        result.weights[0] = complement * complement * complement / 6;
        result.weights[1] = (4 - 6 * fraction * fraction + 3 * fraction * fraction * fraction) / 6;
        result.weights[2] =
            (4 - 6 * complement * complement + 3 * complement * complement * complement) / 6;
        result.weights[3] = fraction * fraction * fraction / 6;
    } else {
        throw std::invalid_argument("particle-mesh cloud order must be 1..4");
    }
    return result;
}

inline CloudWeights cloudWeights(double coordinate, int order) {
    int const cell = static_cast<int>(std::floor(coordinate));
    return cloudWeightsInCell(cell, coordinate - cell, order);
}

template <class Integer>
inline CloudWeights cloudWeightsFromFixed(Integer rawPosition, int meshSize, int order) {
    static_assert(std::is_unsigned_v<Integer>);
    constexpr int kBits = std::numeric_limits<Integer>::digits;
    static_assert(kBits == 32 || kBits == 64);
    unsigned __int128 const scaled = static_cast<unsigned __int128>(rawPosition) * meshSize;
    int const cell = static_cast<int>(scaled >> kBits);
    Integer const remainder = static_cast<Integer>(scaled);
    double fraction = std::ldexp(static_cast<double>(remainder), -kBits);
    // The exact quotient must choose the stencil: reconstructing cell+fraction
    // as a double can round up across a cell (and the containing tile) boundary.
    if (fraction >= 1) {
        fraction = std::nextafter(1.0, 0.0);
    }
    Integer const half = Integer(1) << (kBits - 1);
    if (remainder < half && fraction >= 0.5) {
        fraction = std::nextafter(0.5, 0.0);
    }
    return cloudWeightsInCell(cell, fraction, order);
}

inline int periodicIndex(int index, int meshSize) {
    index %= meshSize;
    return index < 0 ? index + meshSize : index;
}

inline double effectiveRadius(int meshSize, int order) {
    return std::sqrt(order / 3.0) / meshSize;
}

inline int signedMode(int index, int meshSize) {
    return index <= meshSize / 2 ? index : index - meshSize;
}

inline double derivativeWavenumber(int index, int meshSize) {
    // An even mesh's Nyquist mode is its own negative. Its odd derivative must
    // vanish, including when another component of the wave vector is nonzero.
    if (meshSize % 2 == 0 && index == meshSize / 2) {
        return 0;
    }
    return 2 * std::numbers::pi * signedMode(index, meshSize);
}

inline double assignmentWindow(int mode, int meshSize, int order) {
    if (mode == 0) {
        return 1;
    }
    double const argument = std::numbers::pi * mode / meshSize;
    return std::pow(std::sin(argument) / argument, order);
}

// Unit box, excluding gm and inverse-FFT normalization. Both mass deposition
// and force interpolation contribute a window, hence the squared denominator.
inline double spectralPotentialFactor(int meshSize, int order, int x, int y, int z) {
    int const modes[3] = {
        signedMode(x, meshSize), signedMode(y, meshSize), signedMode(z, meshSize)};
    double norm = 0;
    double window = 1;
    for (int axis = 0; axis < 3; ++axis) {
        norm += double(modes[axis]) * modes[axis];
        window *= assignmentWindow(modes[axis], meshSize, order);
    }
    if (norm == 0) {
        return 0;
    }
    return -1 / (std::numbers::pi * norm * window * window);
}

}    // namespace particle_mesh
