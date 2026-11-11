/* CosmicTiger - direct spherical test references; BSL-1.0. */
#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kernels/spherical_ewald_kernels.hpp>

namespace pmmm {

/// Accumulate one particle's spherical source moments and return the generated arithmetic
/// count.
inline FlopCount addMoments(
    int order, double* moments, double mass, Vector3<double> const& offset) {
    if (order < kFmmOrder || order > kFmmOrder) {
        throw std::invalid_argument("Unsupported PMMM reference order");
    }
    FlopCount count = 0;
    {
        constexpr int kDegree = kFmmOrder - 2;
        cosmictiger::gravity::spherical::detail::P2M(
            mass, offset[0], offset[1], offset[2], moments);
        count = cosmictiger::gravity::spherical::detail::kP2MConventionalFlops<kDegree>;
    };
    return count;
}

namespace detail {

    template <int degree, class Field, class Source>
    void addSphericalLocalReference(
        Field& field, Source const& source, double trace, double x, double y, double z) {
        using Complex = std::complex<double>;
        using Jet = std::array<Complex, 4>;
        auto product = [](Jet const& a, Jet const& b) {
            Jet result{};
            result[0] = a[0] * b[0];
            for (int axis = 1; axis < 4; ++axis) {
                result[axis] = a[axis] * b[0] + a[0] * b[axis];
            }
            return result;
        };
        auto slot = [](int rank, int mode) { return rank * (rank + 1) + mode; };
        std::array<Jet, (degree + 1) * (degree + 1)> regular{};
        regular[0][0] = 1;
        Jet const xy{Complex(x, y), 1., Complex(0, 1), 0.};
        Jet const zz{z, 0., 0., 1.};
        Jet const radiusSquared{x * x + y * y + z * z, 2 * x, 2 * y, 2 * z};
        for (int mode = 0; mode <= degree; ++mode) {
            if (mode) {
                auto const next = product(regular[slot(mode - 1, mode - 1)], xy);
                for (int component = 0; component < 4; ++component) {
                    regular[slot(mode, mode)][component] = next[component] / double(2 * mode);
                }
            }
            if (mode < degree) {
                regular[slot(mode + 1, mode)] = product(zz, regular[slot(mode, mode)]);
            }
            for (int rank = mode + 2; rank <= degree; ++rank) {
                auto const first = product(zz, regular[slot(rank - 1, mode)]);
                auto const second = product(radiusSquared, regular[slot(rank - 2, mode)]);
                for (int component = 0; component < 4; ++component) {
                    regular[slot(rank, mode)][component] =
                        (double(2 * rank - 1) * first[component] - second[component]) /
                        double(rank * rank - mode * mode);
                }
            }
        }
        for (int component = 0; component < 4; ++component) {
            double value = -.5 * trace * radiusSquared[component].real();
            for (int rank = 0; rank <= degree; ++rank) {
                for (int mode = 0; mode <= rank; ++mode) {
                    Complex const coefficient(
                        source[slot(rank, mode)], mode ? source[slot(rank, -mode)] : 0.);
                    value += (mode ? 2. : 1.) *
                        (std::conj(coefficient) * regular[slot(rank, mode)][component]).real();
                }
            }
            field[component] += component ? -value : value;
        }
    }

}    // namespace detail

inline void evaluate(int order, std::array<double, 4>& field, double const* coefficients,
    double localTrace, Vector3<double> const& cellOffset, double cellWidth) {
    if (!(cellWidth > 0) || !std::isfinite(cellWidth)) {
        throw std::invalid_argument("PMMM reference cell width must be positive");
    }
    field = {};
    {
        detail::addSphericalLocalReference<kFmmOrder - 1>(
            field, coefficients, localTrace, cellOffset[0], cellOffset[1], cellOffset[2]);
    };
    for (int component = 1; component < 4; ++component) {
        field[component] /= cellWidth;
    }
}

}    // namespace pmmm
