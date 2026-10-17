#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <cosmictiger/radial_kernel.hpp>

#if defined(__CUDACC__)
#define COSMICTIGER_DENSITY_HD __host__ __device__
#else
#define COSMICTIGER_DENSITY_HD
#endif

namespace radial_kernel {

// Density-only tables keep arbitrary-dimensional kernels distinct from the
// three-dimensional mass, force, and potential contract of KernelTable.
template <size_t dimension>
struct DensityKernelTable {
    static_assert(dimension > 0);
    static constexpr size_t kDimension = dimension;
    KernelSpec spec{};
    int degree = 0;
    int pieceCount = 0;
    double changePoints[kMaximumPieces + 1]{};
    double coefficients[kMaximumPieces][kMaximumDegree + 1]{};
    double derivativeCoefficients[kMaximumPieces][kMaximumDegree]{};
    double supportPerSmoothing = 0;
};

template <size_t dimension>
constexpr bool isValidDensity(KernelSpec spec) {
    static_assert(dimension > 0);
    if (spec.family != Family::Wendland) {
        return isValid(spec);
    }
    size_t const baseDegree = dimension / 2 + 1;
    return spec.degree >= 1 && spec.degree <= kMaximumDegree &&
        static_cast<size_t>(spec.degree) >= baseDegree &&
        (static_cast<size_t>(spec.degree) - baseDegree) % 3 == 0;
}

namespace detail {

    template <size_t power>
    constexpr long double densityMoment(
        long double const* coefficients, int degree, long double a, long double b) {
        std::array<long double, kMaximumDegree + power + 1> weighted{};
        for (int i = 0; i <= degree; ++i) {
            weighted[i] = coefficients[i];
        }
        if constexpr (power > 0) {
            for (size_t i = 0; i < power; ++i) {
                size_t const nextDegree = static_cast<size_t>(degree) + i + 1;
                for (size_t j = nextDegree; j > 0; --j) {
                    weighted[j] = a * (nextDegree - j) / nextDegree * weighted[j] +
                        b * j / nextDegree * weighted[j - 1];
                }
                weighted[0] *= a;
            }
        }
        long double integral = 0;
        for (size_t i = 0; i <= static_cast<size_t>(degree) + power; ++i) {
            integral += weighted[i];
        }
        return (b - a) * integral / (degree + power + 1);
    }

    template <size_t dimension>
    constexpr long double sphereArea() {
        long double area = dimension % 2 == 0 ? kFourPi / 2 : 2;
        for (size_t d = dimension % 2 == 0 ? 4 : 3; d <= dimension; d += 2) {
            area *= (kFourPi / 2) / (d - 2);
        }
        return area;
    }

    template <size_t dimension, typename Real>
    COSMICTIGER_DENSITY_HD inline Real densityCoordinate(
        DensityKernelTable<dimension> const& table, int piece, Real q) {
        double const a = table.changePoints[piece];
        double const b = table.changePoints[piece + 1];
        return (clamp(q, a, b) - Real(a)) / Real(b - a);
    }

    template <size_t dimension, typename Real>
    COSMICTIGER_DENSITY_HD inline Real densityMask(
        DensityKernelTable<dimension> const& table, int piece, Real q) {
        Real const upper = q < Real(table.changePoints[piece + 1]);
        return piece == 0 ? upper : upper * Real(q >= Real(table.changePoints[piece]));
    }

}    // namespace detail

// W(r,H) = H^-dimension * value(table,r/H). Returns pieceCount == 0 for an
// invalid family/degree or a normalization unrepresentable by this double table.
// Extremely large dimensions can also exceed compiler constexpr/storage limits.
template <size_t dimension>
constexpr DensityKernelTable<dimension> makeDensityKernelTable(KernelSpec spec) {
    static_assert(dimension > 0);
    DensityKernelTable<dimension> result{};
    result.spec = spec;
    if (!isValidDensity<dimension>(spec)) {
        return result;
    }
    int const degree = spec.degree;
    long double raw[kMaximumPieces][kMaximumDegree + 1]{};
    if (spec.family != Family::Wendland) {
        // These radial shapes are dimension-independent. Reuse their existing
        // construction; the three-dimensional normalization cancels below.
        auto const base = makeKernelTable(spec);
        result.pieceCount = base.pieceCount;
        for (int piece = 0; piece < base.pieceCount; ++piece) {
            result.changePoints[piece] = base.changePoints[piece];
            result.changePoints[piece + 1] = base.changePoints[piece + 1];
            for (int i = 0; i <= degree; ++i) {
                raw[piece][i] = base.coefficients[piece][i];
            }
        }
    } else {
        result.pieceCount = 1;
        result.changePoints[1] = 1;
        int const k = (degree - static_cast<int>(dimension / 2 + 1)) / 3;
        int currentDegree = static_cast<int>(dimension / 2) + k + 1;
        // Start with (1-q)^ell. Applying I[f](q)=integral_q^1 s f(s) ds
        // k times yields the dimension-appropriate Wendland polynomial.
        raw[0][0] = 1;
        for (int iteration = 0; iteration < k; ++iteration) {
            detail::multiplyRadius(raw[0], currentDegree, 0, 1);
            ++currentDegree;
            long double primitive[kMaximumDegree + 1]{};
            for (int i = currentDegree; i >= 0; --i) {
                primitive[i] = primitive[i + 1] + raw[0][i] / (currentDegree + 1);
            }
            ++currentDegree;
            for (int i = 0; i <= currentDegree; ++i) {
                raw[0][i] = primitive[i];
            }
        }
    }
    long double integral = 0;
    long double secondMoment = 0;
    for (int piece = 0; piece < result.pieceCount; ++piece) {
        double const a = result.changePoints[piece];
        double const b = result.changePoints[piece + 1];
        integral += detail::densityMoment<dimension - 1>(raw[piece], degree, a, b);
        secondMoment += detail::densityMoment<dimension + 1>(raw[piece], degree, a, b);
    }
    long double const denominator = detail::sphereArea<dimension>() * integral;
    if (!(denominator > 0 && denominator <= std::numeric_limits<long double>::max() &&
            secondMoment > 0)) {
        return DensityKernelTable<dimension>{};
    }
    long double const normalization = 1 / denominator;
    long double const support = detail::squareRoot(dimension * integral / (4 * secondMoment));
    if (!(support > 0 && support <= std::numeric_limits<double>::max())) {
        return DensityKernelTable<dimension>{};
    }
    result.degree = degree;
    result.supportPerSmoothing = static_cast<double>(support);
    for (int piece = 0; piece < result.pieceCount; ++piece) {
        double const width = result.changePoints[piece + 1] - result.changePoints[piece];
        for (int i = 0; i <= degree; ++i) {
            long double const coefficient = normalization * raw[piece][i];
            if (!(coefficient >= 0 && coefficient <= std::numeric_limits<double>::max())) {
                return DensityKernelTable<dimension>{};
            }
            result.coefficients[piece][i] = static_cast<double>(coefficient);
        }
        for (int i = 0; i < degree; ++i) {
            long double const coefficient =
                degree * normalization * (raw[piece][i + 1] - raw[piece][i]) / width;
            if (!(coefficient >= -std::numeric_limits<double>::max() &&
                    coefficient <= std::numeric_limits<double>::max())) {
                return DensityKernelTable<dimension>{};
            }
            result.derivativeCoefficients[piece][i] = static_cast<double>(coefficient);
        }
    }
    return result;
}

template <size_t dimension, typename Real>
COSMICTIGER_DENSITY_HD inline Real value(DensityKernelTable<dimension> const& table, Real q) {
    Real result = Real(0);
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        if constexpr (std::is_arithmetic_v<Real>) {
            if (q < Real(table.changePoints[piece + 1])) {
                return detail::evaluate(table.coefficients[piece], table.degree,
                    detail::densityCoordinate(table, piece, q));
            }
        } else {
            result += detail::densityMask(table, piece, q) *
                detail::evaluate(table.coefficients[piece], table.degree,
                    detail::densityCoordinate(table, piece, q));
        }
    }
    return result;
}

template <size_t dimension, typename Real>
COSMICTIGER_DENSITY_HD inline Real derivative(DensityKernelTable<dimension> const& table, Real q) {
    Real result = Real(0);
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        if constexpr (std::is_arithmetic_v<Real>) {
            if (q < Real(table.changePoints[piece + 1])) {
                return detail::evaluate(table.derivativeCoefficients[piece], table.degree - 1,
                    detail::densityCoordinate(table, piece, q));
            }
        } else {
            result += detail::densityMask(table, piece, q) *
                detail::evaluate(table.derivativeCoefficients[piece], table.degree - 1,
                    detail::densityCoordinate(table, piece, q));
        }
    }
    return result;
}

}    // namespace radial_kernel

#undef COSMICTIGER_DENSITY_HD
