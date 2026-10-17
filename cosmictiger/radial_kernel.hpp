#pragma once

#include <cstddef>
#include <numbers>
#include <type_traits>

#if defined(__CUDACC__)
#define COSMICTIGER_RADIAL_HD __host__ __device__
#else
#define COSMICTIGER_RADIAL_HD
#endif

// Normalized spherical density kernels in three spatial dimensions. All radial
// arguments are q = r/H, where H is the compact support radius. The common
// smoothing length is h = 2 sigma, with sigma the one-coordinate deviation.
namespace radial_kernel {

using std::size_t;

inline constexpr int kMaximumDegree = 9;
inline constexpr int kMaximumPieces = 5;

enum class Family { Spline, Wendland, Ferrers };

struct KernelSpec {
    Family family = Family::Spline;
    int degree = 3;
};

struct KernelTable {
    KernelSpec spec{};
    int degree = 0;
    int pieceCount = 0;
    double changePoints[kMaximumPieces + 1]{};
    // Bernstein coefficients on the local coordinate of each radial span.
    double coefficients[kMaximumPieces][kMaximumDegree + 1]{};
    double derivativeCoefficients[kMaximumPieces][kMaximumDegree]{};
    double massCoefficients[kMaximumPieces][kMaximumDegree + 4]{};
    double shellCoefficients[kMaximumPieces][kMaximumDegree + 3]{};
    // Factoring the first span analytically avoids division by q near zero.
    double centralForceCoefficients[kMaximumDegree + 1]{};
    double centralPotentialDropCoefficients[kMaximumDegree + 1]{};
    double supportPerSmoothing = 0;
    double centralPotential = 0;
};

constexpr bool isValid(KernelSpec spec) {
    switch (spec.family) {
    case Family::Spline:
        return spec.degree >= 3 && spec.degree <= kMaximumDegree;
    case Family::Wendland:
        return spec.degree == 2 || spec.degree == 5 || spec.degree == 8;
    case Family::Ferrers:
        return spec.degree >= 2 && spec.degree <= 8 && spec.degree % 2 == 0;
    }
    return false;
}

namespace detail {

    inline constexpr long double kFourPi = 4 * std::numbers::pi_v<long double>;

    constexpr long double binomial(int n, int k) {
        long double result = 1;
        for (int i = 1; i <= k; ++i) {
            result *= static_cast<long double>(n - i + 1) / i;
        }
        return result;
    }

    constexpr long double squareRoot(long double value) {
        long double result = value > 1 ? value : 1;
        for (int i = 0; i < 64; ++i) {
            result = (result + value / result) / 2;
        }
        return result;
    }

    // Multiply a Bernstein polynomial by q = a(1-t) + bt in place.
    constexpr void multiplyRadius(
        long double* coefficients, int degree, long double a, long double b) {
        for (int i = degree + 1; i > 0; --i) {
            coefficients[i] = a * (degree + 1 - i) / (degree + 1) * coefficients[i] +
                b * i / (degree + 1) * coefficients[i - 1];
        }
        coefficients[0] *= a;
    }

    constexpr long double moment(
        long double const* coefficients, int degree, int power, long double a, long double b) {
        long double weighted[kMaximumDegree + 5]{};
        for (int i = 0; i <= degree; ++i) {
            weighted[i] = coefficients[i];
        }
        for (int i = 0; i < power; ++i) {
            multiplyRadius(weighted, degree + i, a, b);
        }
        long double result = 0;
        for (int i = 0; i <= degree + power; ++i) {
            result += weighted[i];
        }
        return result * (b - a) / (degree + power + 1);
    }

    constexpr void multiplyPolynomials(
        long double* left, int leftDegree, long double const* right, int rightDegree) {
        long double product[kMaximumDegree + 1]{};
        for (int i = 0; i <= leftDegree; ++i) {
            for (int j = 0; j <= rightDegree; ++j) {
                product[i + j] += left[i] * right[j] * binomial(leftDegree, i) *
                    binomial(rightDegree, j) / binomial(leftDegree + rightDegree, i + j);
            }
        }
        for (int i = 0; i <= leftDegree + rightDegree; ++i) {
            left[i] = product[i];
        }
    }

    template <typename Real>
    COSMICTIGER_RADIAL_HD inline Real clamp(Real value, double lower, double upper) {
        if constexpr (std::is_arithmetic_v<Real>) {
            return value < Real(lower) ? Real(lower) : (value > Real(upper) ? Real(upper) : value);
        } else {
            Real const below = value < Real(lower);
            Real const above = value > Real(upper);
            return below * Real(lower) + above * Real(upper) + (Real(1) - below - above) * value;
        }
    }

    template <typename Real>
    COSMICTIGER_RADIAL_HD inline Real evaluate(double const* coefficients, int degree, Real t) {
        Real work[kMaximumDegree + 4];
        for (int i = 0; i <= degree; ++i) {
            work[i] = Real(coefficients[i]);
        }
        for (int remaining = degree; remaining > 0; --remaining) {
            for (int i = 0; i < remaining; ++i) {
                work[i] = (Real(1) - t) * work[i] + t * work[i + 1];
            }
        }
        return work[0];
    }

    template <typename Real>
    COSMICTIGER_RADIAL_HD inline Real localCoordinate(KernelTable const& table, int piece, Real q) {
        double const a = table.changePoints[piece];
        double const b = table.changePoints[piece + 1];
        return (clamp(q, a, b) - Real(a)) / Real(b - a);
    }

    template <typename Real>
    COSMICTIGER_RADIAL_HD inline Real pieceMask(KernelTable const& table, int piece, Real q) {
        Real const upper = q < Real(table.changePoints[piece + 1]);
        if (piece == 0) {
            return upper;
        }
        return upper * Real(q >= Real(table.changePoints[piece]));
    }

}    // namespace detail

// Invalid specifications return a table with pieceCount == 0. Call isValid()
// before accepting a user selection. Generation is suitable for constexpr use;
// runtime evaluation has no allocation and is shared by CPU, SIMD, and CUDA.
constexpr KernelTable makeKernelTable(KernelSpec spec) {
    KernelTable result{};
    result.spec = spec;
    if (!isValid(spec)) {
        return result;
    }
    int const degree = spec.degree;
    result.degree = degree;
    result.pieceCount = spec.family == Family::Spline ? (degree + 2) / 2 : 1;
    long double raw[kMaximumPieces][kMaximumDegree + 1]{};
    long double points[kMaximumPieces + 1]{};
    if (spec.family == Family::Spline) {
        int const order = degree + 1;
        for (int piece = 0; piece < result.pieceCount; ++piece) {
            int const knot = order / 2 + piece;
            long double const x0 = piece == 0 ? order / 2.0L : knot;
            long double const x1 = knot + 1;
            points[piece] = (2 * x0 - order) / order;
            points[piece + 1] = (2 * x1 - order) / order;
            long double work[kMaximumDegree + 1][kMaximumDegree + 1]{};
            work[knot][0] = 1;
            for (int p = 1; p <= degree; ++p) {
                long double next[kMaximumDegree + 1][kMaximumDegree + 1]{};
                for (int i = 0; i + p < order; ++i) {
                    for (int k = 0; k <= p; ++k) {
                        if (k < p) {
                            next[i][k] += static_cast<long double>(p - k) / (p * p) *
                                ((x0 - i) * work[i][k] + (i + p + 1 - x0) * work[i + 1][k]);
                        }
                        if (k > 0) {
                            next[i][k] += static_cast<long double>(k) / (p * p) *
                                ((x1 - i) * work[i][k - 1] + (i + p + 1 - x1) * work[i + 1][k - 1]);
                        }
                    }
                }
                for (int i = 0; i < order; ++i) {
                    for (int k = 0; k < order; ++k) {
                        work[i][k] = next[i][k];
                    }
                }
            }
            for (int i = 0; i <= degree; ++i) {
                raw[piece][i] = work[0][i];
            }
        }
    } else {
        points[1] = 1;
        if (spec.family == Family::Ferrers) {
            raw[0][0] = 1;
            long double const factor[3] = {1, 1, 0};    // 1 - q^2
            for (int p = 0; p < degree; p += 2) {
                detail::multiplyPolynomials(raw[0], p, factor, 2);
            }
        } else {
            int const smoothness = (degree - 2) / 3;
            int const exponent = 2 + 2 * smoothness;
            // (1-q)^exponent times the positive Wendland factor.
            raw[0][0] = 1;
            long double factor[3] = {1, 0, 0};
            if (smoothness == 1) {
                factor[1] = 5;    // 1 + 4q in degree-one Bernstein form.
            } else if (smoothness == 2) {
                factor[1] = 4;
                factor[2] = 56.0L / 3;    // 1 + 6q + (35/3)q^2.
            }
            detail::multiplyPolynomials(raw[0], exponent, factor, smoothness);
        }
    }

    long double massMoment = 0;
    long double varianceMoment = 0;
    for (int piece = 0; piece < result.pieceCount; ++piece) {
        massMoment += detail::moment(raw[piece], degree, 2, points[piece], points[piece + 1]);
        varianceMoment += detail::moment(raw[piece], degree, 4, points[piece], points[piece + 1]);
    }
    long double const normalization = 1 / (detail::kFourPi * massMoment);
    result.supportPerSmoothing =
        static_cast<double>(detail::squareRoot(3 * massMoment / (4 * varianceMoment)));
    long double massPrefix = 0;
    for (int piece = 0; piece < result.pieceCount; ++piece) {
        long double const a = points[piece];
        long double const b = points[piece + 1];
        result.changePoints[piece] = static_cast<double>(a);
        result.changePoints[piece + 1] = static_cast<double>(b);
        long double weighted[kMaximumDegree + 4]{};
        for (int i = 0; i <= degree; ++i) {
            raw[piece][i] *= normalization;
            result.coefficients[piece][i] = static_cast<double>(raw[piece][i]);
            weighted[i] = detail::kFourPi * raw[piece][i];
        }
        for (int i = 0; i < degree; ++i) {
            result.derivativeCoefficients[piece][i] =
                static_cast<double>(degree * (raw[piece][i + 1] - raw[piece][i]) / (b - a));
        }
        detail::multiplyRadius(weighted, degree, a, b);
        detail::multiplyRadius(weighted, degree + 1, a, b);
        long double mass[kMaximumDegree + 4]{};
        mass[0] = massPrefix;
        result.massCoefficients[piece][0] = static_cast<double>(massPrefix);
        for (int i = 0; i <= degree + 2; ++i) {
            mass[i + 1] = mass[i] + (b - a) * weighted[i] / (degree + 3);
            result.massCoefficients[piece][i + 1] = static_cast<double>(mass[i + 1]);
        }
        massPrefix = mass[degree + 3];
        if (piece == 0) {
            long double force[kMaximumDegree + 3]{};
            for (int i = 0; i <= degree; ++i) {
                force[i] = mass[i + 3] * detail::binomial(degree + 3, i + 3) /
                    detail::binomial(degree, i) / (b * b * b);
                result.centralForceCoefficients[i] = static_cast<double>(force[i]);
            }
            detail::multiplyRadius(force, degree, 0, 1);
            long double primitive[kMaximumDegree + 3]{};
            for (int i = 0; i <= degree + 1; ++i) {
                primitive[i + 1] = primitive[i] + force[i] / (degree + 2);
            }
            for (int i = 0; i <= degree; ++i) {
                result.centralPotentialDropCoefficients[i] = static_cast<double>(primitive[i + 2] *
                    detail::binomial(degree + 2, i + 2) / detail::binomial(degree, i));
            }
        }
    }
    long double shellTail = 0;
    for (int piece = result.pieceCount - 1; piece >= 0; --piece) {
        long double weighted[kMaximumDegree + 3]{};
        for (int i = 0; i <= degree; ++i) {
            weighted[i] = detail::kFourPi * raw[piece][i];
        }
        detail::multiplyRadius(weighted, degree, points[piece], points[piece + 1]);
        result.shellCoefficients[piece][degree + 2] = static_cast<double>(shellTail);
        for (int i = degree + 1; i >= 0; --i) {
            shellTail += (points[piece + 1] - points[piece]) * weighted[i] / (degree + 2);
            result.shellCoefficients[piece][i] = static_cast<double>(shellTail);
        }
    }
    result.centralPotential = static_cast<double>(shellTail);
    return result;
}

template <typename Real>
COSMICTIGER_RADIAL_HD inline Real value(KernelTable const& table, Real q) {
    Real result = Real(0);
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        if constexpr (std::is_arithmetic_v<Real>) {
            if (q < Real(table.changePoints[piece + 1])) {
                return detail::evaluate(table.coefficients[piece], table.degree,
                    detail::localCoordinate(table, piece, q));
            }
        } else {
            result += detail::pieceMask(table, piece, q) *
                detail::evaluate(table.coefficients[piece], table.degree,
                    detail::localCoordinate(table, piece, q));
        }
    }
    return result;
}

template <typename Real>
COSMICTIGER_RADIAL_HD inline Real derivative(KernelTable const& table, Real q) {
    Real result = Real(0);
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        if constexpr (std::is_arithmetic_v<Real>) {
            if (q < Real(table.changePoints[piece + 1])) {
                return detail::evaluate(table.derivativeCoefficients[piece], table.degree - 1,
                    detail::localCoordinate(table, piece, q));
            }
        } else {
            result += detail::pieceMask(table, piece, q) *
                detail::evaluate(table.derivativeCoefficients[piece], table.degree - 1,
                    detail::localCoordinate(table, piece, q));
        }
    }
    return result;
}

namespace detail {

    template <typename Real>
    COSMICTIGER_RADIAL_HD inline Real forceOnPiece(KernelTable const& table, int piece, Real q) {
        Real const t = localCoordinate(table, piece, q);
        if (piece == 0) {
            return evaluate(table.centralForceCoefficients, table.degree, t);
        }
        Real const inverseRadius =
            Real(1) / clamp(q, table.changePoints[piece], table.changePoints[piece + 1]);
        return evaluate(table.massCoefficients[piece], table.degree + 3, t) * inverseRadius *
            inverseRadius * inverseRadius;
    }

    template <typename Real>
    COSMICTIGER_RADIAL_HD inline Real potentialOnPiece(
        KernelTable const& table, int piece, Real q) {
        Real const t = localCoordinate(table, piece, q);
        if (piece == 0) {
            Real const radius = clamp(q, 0, table.changePoints[1]);
            return Real(table.centralPotential) -
                radius * radius * evaluate(table.centralPotentialDropCoefficients, table.degree, t);
        }
        Real const inverseRadius =
            Real(1) / clamp(q, table.changePoints[piece], table.changePoints[piece + 1]);
        return evaluate(table.massCoefficients[piece], table.degree + 3, t) * inverseRadius +
            evaluate(table.shellCoefficients[piece], table.degree + 2, t);
    }

}    // namespace detail

// Acceleration = -G m/H^3 * forceFactor(q) * displacement. The finite origin
// value is 4 pi W(0)/3; the actual central acceleration is zero.
template <typename Real>
COSMICTIGER_RADIAL_HD inline Real forceFactor(KernelTable const& table, Real q) {
    Real result = Real(0);
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        if constexpr (std::is_arithmetic_v<Real>) {
            if (q < Real(table.changePoints[piece + 1])) {
                return detail::forceOnPiece(table, piece, q);
            }
        } else {
            result += detail::pieceMask(table, piece, q) * detail::forceOnPiece(table, piece, q);
        }
    }
    if constexpr (std::is_arithmetic_v<Real>) {
        Real const inverseRadius = Real(1) / q;
        return inverseRadius * inverseRadius * inverseRadius;
    } else {
        Real const outside = q >= Real(1);
        Real const inverseRadius = Real(1) / (outside * q + (Real(1) - outside));
        return result + outside * inverseRadius * inverseRadius * inverseRadius;
    }
}

// Positive potential magnitude: physical gravitational potential is -Gm/H
// times this value, with zero at infinity and exactly 1/q outside support.
template <typename Real>
COSMICTIGER_RADIAL_HD inline Real potential(KernelTable const& table, Real q) {
    Real result = Real(0);
    for (int piece = 0; piece < table.pieceCount; ++piece) {
        if constexpr (std::is_arithmetic_v<Real>) {
            if (q < Real(table.changePoints[piece + 1])) {
                return detail::potentialOnPiece(table, piece, q);
            }
        } else {
            result +=
                detail::pieceMask(table, piece, q) * detail::potentialOnPiece(table, piece, q);
        }
    }
    if constexpr (std::is_arithmetic_v<Real>) {
        return Real(1) / q;
    } else {
        Real const outside = q >= Real(1);
        return result + outside / (outside * q + (Real(1) - outside));
    }
}

}    // namespace radial_kernel

#undef COSMICTIGER_RADIAL_HD
