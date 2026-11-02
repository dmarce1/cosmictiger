/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/vector.hpp>

#ifndef COSMICTIGER_MATRIX_GENERAL_INVERSE_THRESHOLD
#define COSMICTIGER_MATRIX_GENERAL_INVERSE_THRESHOLD 4
#endif
#ifndef COSMICTIGER_MATRIX_SYMMETRIC_INVERSE_THRESHOLD
#define COSMICTIGER_MATRIX_SYMMETRIC_INVERSE_THRESHOLD 4
#endif
#ifndef COSMICTIGER_MATRIX_GENERAL_DETERMINANT_THRESHOLD
#define COSMICTIGER_MATRIX_GENERAL_DETERMINANT_THRESHOLD 4
#endif
#ifndef COSMICTIGER_MATRIX_SYMMETRIC_DETERMINANT_THRESHOLD
#define COSMICTIGER_MATRIX_SYMMETRIC_DETERMINANT_THRESHOLD 4
#endif
#ifndef COSMICTIGER_MATRIX_GENERAL_SOLVE_THRESHOLD
#define COSMICTIGER_MATRIX_GENERAL_SOLVE_THRESHOLD 3
#endif
#ifndef COSMICTIGER_MATRIX_SYMMETRIC_SOLVE_THRESHOLD
#define COSMICTIGER_MATRIX_SYMMETRIC_SOLVE_THRESHOLD 3
#endif

namespace math {

template <class T, int rowCount, int columnCount = rowCount>
class Matrix;
template <class T, int dimensions>
class SymmetricMatrix;
template <class T, int dimensions>
class DiagonalMatrix;

namespace detail {
    template <class T>
    struct MatrixTraits;
    template <class T, int rows, int cols>
    struct MatrixTraits<Matrix<T, rows, cols>> {
        using ValueType = T;
        static constexpr int kRows = rows;
        static constexpr int kCols = cols;
        static constexpr bool kSymmetric = false;
        static constexpr bool kDiagonal = false;
    };
    template <class T, int dimensions>
    struct MatrixTraits<SymmetricMatrix<T, dimensions>> {
        using ValueType = T;
        static constexpr int kRows = dimensions;
        static constexpr int kCols = dimensions;
        static constexpr bool kSymmetric = true;
        static constexpr bool kDiagonal = false;
    };
    template <class T, int dimensions>
    struct MatrixTraits<DiagonalMatrix<T, dimensions>> {
        using ValueType = T;
        static constexpr int kRows = dimensions;
        static constexpr int kCols = dimensions;
        static constexpr bool kSymmetric = true;
        static constexpr bool kDiagonal = true;
    };
    template <class T>
    using Traits = MatrixTraits<std::remove_cvref_t<T>>;
}    // namespace detail

template <class T>
concept FixedMatrix = requires { typename detail::Traits<T>::ValueType; };

template <class T, int rowCount, int columnCount>
class Matrix {
    static_assert(rowCount > 0 && columnCount > 0);
    Vector<T, columnCount> rows_[rowCount]{};

public:
    using ValueType = T;
    CUDA_EXPORT constexpr Matrix()
      : Matrix(T(0)) {}
    CUDA_EXPORT constexpr explicit Matrix(T const& value) {
        for (int i = 0; i < rowCount; ++i) {
            for (int j = 0; j < columnCount; ++j) {
                rows_[i][j] = value;
            }
        }
    }
    CUDA_EXPORT constexpr Matrix(std::initializer_list<std::initializer_list<T>> values)
      : Matrix() {
        int i = 0;
        for (auto const& rowValues : values) {
            if (i == rowCount) {
                break;
            }
            int j = 0;
            for (auto const& value : rowValues) {
                if (j == columnCount) {
                    break;
                }
                rows_[i][j++] = value;
            }
            ++i;
        }
    }
    template <FixedMatrix Other>
        requires(
            detail::Traits<Other>::kRows == rowCount && detail::Traits<Other>::kCols == columnCount)
    CUDA_EXPORT constexpr explicit Matrix(Other const& other) {
        for (int i = 0; i < rowCount; ++i) {
            for (int j = 0; j < columnCount; ++j) {
                rows_[i][j] = T(other(i, j));
            }
        }
    }
    CUDA_EXPORT static constexpr int rows() {
        return rowCount;
    }
    CUDA_EXPORT static constexpr int cols() {
        return columnCount;
    }
    CUDA_EXPORT static constexpr auto dims() {
        return std::pair{rowCount, columnCount};
    }
    CUDA_EXPORT static constexpr int size() {
        return rowCount * columnCount;
    }
    CUDA_EXPORT constexpr T& operator()(int i, int j) {
        return rows_[i][j];
    }
    CUDA_EXPORT constexpr T const& operator()(int i, int j) const {
        return rows_[i][j];
    }
    CUDA_EXPORT constexpr Vector<T, columnCount>& operator[](int i) {
        return rows_[i];
    }
    CUDA_EXPORT constexpr Vector<T, columnCount> const& operator[](int i) const {
        return rows_[i];
    }
    CUDA_EXPORT constexpr auto& row(int i) {
        return rows_[i];
    }
    CUDA_EXPORT constexpr auto const& row(int i) const {
        return rows_[i];
    }
    CUDA_EXPORT constexpr Vector<T, rowCount> col(int j) const {
        Vector<T, rowCount> result{};
        for (int i = 0; i < rowCount; ++i) {
            result[i] = rows_[i][j];
        }
        return result;
    }
    template <FixedMatrix Other>
        requires(
            detail::Traits<Other>::kRows == rowCount && detail::Traits<Other>::kCols == columnCount)
    CUDA_EXPORT constexpr Matrix& operator+=(Other const& other) {
        for (int i = 0; i < rowCount; ++i) {
            for (int j = 0; j < columnCount; ++j) {
                rows_[i][j] += other(i, j);
            }
        }
        return *this;
    }
    template <FixedMatrix Other>
        requires(
            detail::Traits<Other>::kRows == rowCount && detail::Traits<Other>::kCols == columnCount)
    CUDA_EXPORT constexpr Matrix& operator-=(Other const& other) {
        for (int i = 0; i < rowCount; ++i) {
            for (int j = 0; j < columnCount; ++j) {
                rows_[i][j] -= other(i, j);
            }
        }
        return *this;
    }
    CUDA_EXPORT constexpr Matrix& operator*=(T const& scalar) {
        T const value = scalar;
        for (int i = 0; i < rowCount; ++i) {
            rows_[i] *= value;
        }
        return *this;
    }
    CUDA_EXPORT constexpr Matrix& operator/=(T const& scalar) {
        T const value = scalar;
        for (int i = 0; i < rowCount; ++i) {
            for (int j = 0; j < columnCount; ++j) {
                rows_[i][j] /= value;
            }
        }
        return *this;
    }
};

template <class T, int dimensions>
class SymmetricMatrix {
    static_assert(dimensions > 0);
    T values_[(dimensions * (dimensions + 1)) / 2]{};
    CUDA_EXPORT static constexpr int index(int i, int j) {
        return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
    }

public:
    using ValueType = T;
    CUDA_EXPORT constexpr SymmetricMatrix()
      : SymmetricMatrix(T(0)) {}
    CUDA_EXPORT constexpr explicit SymmetricMatrix(T const& value) {
        for (auto& entry : values_) {
            entry = value;
        }
    }
    // The caller asserts symmetry; copy only the lower triangle.
    template <FixedMatrix Other>
        requires(detail::Traits<Other>::kRows == dimensions &&
            detail::Traits<Other>::kCols == dimensions)
    CUDA_EXPORT constexpr explicit SymmetricMatrix(Other const& other) {
        for (int i = 0; i < dimensions; ++i) {
            for (int j = 0; j <= i; ++j) {
                (*this)(i, j) = T(other(i, j));
            }
        }
    }
    CUDA_EXPORT static constexpr int rows() {
        return dimensions;
    }
    CUDA_EXPORT static constexpr int cols() {
        return dimensions;
    }
    CUDA_EXPORT static constexpr auto dims() {
        return std::pair{dimensions, dimensions};
    }
    CUDA_EXPORT static constexpr int size() {
        return dimensions * (dimensions + 1) / 2;
    }
    CUDA_EXPORT constexpr T& operator()(int i, int j) {
        return values_[index(i, j)];
    }
    CUDA_EXPORT constexpr T const& operator()(int i, int j) const {
        return values_[index(i, j)];
    }
    CUDA_EXPORT constexpr Vector<T, dimensions> row(int i) const {
        Vector<T, dimensions> result{};
        for (int j = 0; j < dimensions; ++j) {
            result[j] = (*this)(i, j);
        }
        return result;
    }
    CUDA_EXPORT constexpr auto col(int i) const {
        return row(i);
    }
    CUDA_EXPORT constexpr SymmetricMatrix& operator+=(SymmetricMatrix const& other) {
        for (int i = 0; i < size(); ++i) {
            values_[i] += other.values_[i];
        }
        return *this;
    }
    CUDA_EXPORT constexpr SymmetricMatrix& operator-=(SymmetricMatrix const& other) {
        for (int i = 0; i < size(); ++i) {
            values_[i] -= other.values_[i];
        }
        return *this;
    }
    CUDA_EXPORT constexpr SymmetricMatrix& operator*=(T const& scalar) {
        T const value = scalar;
        for (auto& entry : values_) {
            entry *= value;
        }
        return *this;
    }
    CUDA_EXPORT constexpr SymmetricMatrix& operator/=(T const& scalar) {
        T const value = scalar;
        for (auto& entry : values_) {
            entry /= value;
        }
        return *this;
    }
};

template <class T, int dimensions>
class DiagonalMatrix {
    static_assert(dimensions > 0);
    T values_[dimensions]{};

public:
    using ValueType = T;
    CUDA_EXPORT constexpr DiagonalMatrix()
      : DiagonalMatrix(T(0)) {}
    CUDA_EXPORT constexpr explicit DiagonalMatrix(T const& value) {
        for (auto& entry : values_) {
            entry = value;
        }
    }
    CUDA_EXPORT constexpr DiagonalMatrix(std::initializer_list<T> values)
      : DiagonalMatrix() {
        int i = 0;
        for (auto const& value : values) {
            if (i == dimensions) {
                break;
            }
            values_[i++] = value;
        }
    }
    template <class U, class Derived>
    CUDA_EXPORT constexpr explicit DiagonalMatrix(Vector<U, dimensions, Derived> const& values) {
        for (int i = 0; i < dimensions; ++i) {
            values_[i] = T(values[i]);
        }
    }
    CUDA_EXPORT static constexpr int rows() {
        return dimensions;
    }
    CUDA_EXPORT static constexpr int cols() {
        return dimensions;
    }
    CUDA_EXPORT static constexpr auto dims() {
        return std::pair{dimensions, dimensions};
    }
    CUDA_EXPORT static constexpr int size() {
        return dimensions;
    }
    CUDA_EXPORT constexpr T operator()(int i, int j) const {
        return i == j ? values_[i] : T(0);
    }
    CUDA_EXPORT constexpr T& diagonal(int i) {
        return values_[i];
    }
    CUDA_EXPORT constexpr T const& diagonal(int i) const {
        return values_[i];
    }
    CUDA_EXPORT constexpr T& operator[](int i) {
        return values_[i];
    }
    CUDA_EXPORT constexpr T const& operator[](int i) const {
        return values_[i];
    }
    CUDA_EXPORT constexpr DiagonalMatrix& operator+=(DiagonalMatrix const& other) {
        for (int i = 0; i < dimensions; ++i) {
            values_[i] += other.values_[i];
        }
        return *this;
    }
    CUDA_EXPORT constexpr DiagonalMatrix& operator-=(DiagonalMatrix const& other) {
        for (int i = 0; i < dimensions; ++i) {
            values_[i] -= other.values_[i];
        }
        return *this;
    }
    CUDA_EXPORT constexpr DiagonalMatrix& operator*=(T const& scalar) {
        T const value = scalar;
        for (auto& entry : values_) {
            entry *= value;
        }
        return *this;
    }
    CUDA_EXPORT constexpr DiagonalMatrix& operator/=(T const& scalar) {
        T const value = scalar;
        for (auto& entry : values_) {
            entry /= value;
        }
        return *this;
    }
};

template <class T, int dimensions>
CUDA_EXPORT constexpr auto identityMatrix() {
    return DiagonalMatrix<T, dimensions>(T(1));
}

template <FixedMatrix A>
CUDA_EXPORT constexpr auto operator+(A const& a) {
    return a;
}
template <FixedMatrix A>
CUDA_EXPORT constexpr auto operator-(A a) {
    return a *= typename A::ValueType(-1);
}

template <FixedMatrix A, FixedMatrix B>
    requires(detail::Traits<A>::kRows == detail::Traits<B>::kRows &&
        detail::Traits<A>::kCols == detail::Traits<B>::kCols)
CUDA_EXPORT constexpr auto operator+(A const& a, B const& b) {
    using T = decltype(a(0, 0) + b(0, 0));
    constexpr int kRows = A::rows();
    constexpr int kCols = A::cols();
    if constexpr (detail::Traits<A>::kDiagonal && detail::Traits<B>::kDiagonal) {
        DiagonalMatrix<T, kRows> result;
        for (int i = 0; i < kRows; ++i) {
            result.diagonal(i) = a(i, i) + b(i, i);
        }
        return result;
    } else if constexpr (detail::Traits<A>::kSymmetric && detail::Traits<B>::kSymmetric) {
        SymmetricMatrix<T, kRows> result;
        for (int i = 0; i < kRows; ++i) {
            for (int j = 0; j <= i; ++j) {
                result(i, j) = a(i, j) + b(i, j);
            }
        }
        return result;
    } else {
        Matrix<T, kRows, kCols> result;
        for (int i = 0; i < kRows; ++i) {
            for (int j = 0; j < kCols; ++j) {
                result(i, j) = a(i, j) + b(i, j);
            }
        }
        return result;
    }
}
template <FixedMatrix A, FixedMatrix B>
    requires(detail::Traits<A>::kRows == detail::Traits<B>::kRows &&
        detail::Traits<A>::kCols == detail::Traits<B>::kCols)
CUDA_EXPORT constexpr auto operator-(A const& a, B const& b) {
    return a + (-b);
}
template <FixedMatrix A>
CUDA_EXPORT constexpr A operator*(A a, typename A::ValueType const& value) {
    return a *= value;
}
template <FixedMatrix A>
CUDA_EXPORT constexpr A operator*(typename A::ValueType const& value, A a) {
    return a *= value;
}
template <FixedMatrix A>
CUDA_EXPORT constexpr A operator/(A a, typename A::ValueType const& value) {
    return a /= value;
}

template <FixedMatrix A, FixedMatrix B>
    requires(detail::Traits<A>::kCols == detail::Traits<B>::kRows)
CUDA_EXPORT constexpr auto operator*(A const& a, B const& b) {
    using T = decltype(a(0, 0) * b(0, 0));
    constexpr int kRows = A::rows();
    constexpr int kCols = B::cols();
    if constexpr (detail::Traits<A>::kDiagonal && detail::Traits<B>::kDiagonal) {
        DiagonalMatrix<T, kRows> result;
        for (int i = 0; i < kRows; ++i) {
            result.diagonal(i) = a(i, i) * b(i, i);
        }
        return result;
    } else {
        Matrix<T, kRows, kCols> result;
        for (int i = 0; i < kRows; ++i) {
            for (int j = 0; j < kCols; ++j) {
                if constexpr (detail::Traits<A>::kDiagonal) {
                    result(i, j) = a(i, i) * b(i, j);
                } else if constexpr (detail::Traits<B>::kDiagonal) {
                    result(i, j) = a(i, j) * b(j, j);
                } else {
                    T value = a(i, 0) * b(0, j);
                    for (int k = 1; k < A::cols(); ++k) {
                        value += a(i, k) * b(k, j);
                    }
                    result(i, j) = value;
                }
            }
        }
        return result;
    }
}
template <FixedMatrix A, class T, int dimensions, class Derived>
    requires(A::cols() == dimensions)
CUDA_EXPORT constexpr auto operator*(A const& a, Vector<T, dimensions, Derived> const& vector) {
    using R = decltype(a(0, 0) * vector[0]);
    Vector<R, A::rows()> result{};
    for (int i = 0; i < A::rows(); ++i) {
        if constexpr (detail::Traits<A>::kDiagonal) {
            result[i] = a(i, i) * vector[i];
        } else {
            R value = a(i, 0) * vector[0];
            for (int j = 1; j < dimensions; ++j) {
                value += a(i, j) * vector[j];
            }
            result[i] = value;
        }
    }
    return result;
}
template <class T, int dimensions, class Derived, FixedMatrix A>
    requires(A::rows() == dimensions)
CUDA_EXPORT constexpr auto operator*(Vector<T, dimensions, Derived> const& vector, A const& a) {
    using R = decltype(vector[0] * a(0, 0));
    Vector<R, A::cols()> result{};
    for (int j = 0; j < A::cols(); ++j) {
        if constexpr (detail::Traits<A>::kDiagonal) {
            result[j] = vector[j] * a(j, j);
        } else {
            R value = vector[0] * a(0, j);
            for (int i = 1; i < dimensions; ++i) {
                value += vector[i] * a(i, j);
            }
            result[j] = value;
        }
    }
    return result;
}

template <FixedMatrix A>
CUDA_EXPORT constexpr auto transpose(A const& a) {
    if constexpr (detail::Traits<A>::kSymmetric) {
        return a;
    } else {
        Matrix<typename A::ValueType, A::cols(), A::rows()> result;
        for (int i = 0; i < A::rows(); ++i) {
            for (int j = 0; j < A::cols(); ++j) {
                result(j, i) = a(i, j);
            }
        }
        return result;
    }
}
template <class T, int rows, class D, class U, int cols, class E>
CUDA_EXPORT constexpr auto outerProduct(Vector<T, rows, D> const& a, Vector<U, cols, E> const& b) {
    Matrix<decltype(a[0] * b[0]), rows, cols> result;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = a[i] * b[j];
        }
    }
    return result;
}
template <class T, int dimensions, class Derived>
CUDA_EXPORT constexpr auto symmetricOuterProduct(Vector<T, dimensions, Derived> const& a) {
    SymmetricMatrix<T, dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
        for (int j = 0; j <= i; ++j) {
            result(i, j) = a[i] * a[j];
        }
    }
    return result;
}
template <FixedMatrix A>
    requires(A::rows() == A::cols())
CUDA_EXPORT constexpr auto trace(A const& a) {
    auto result = a(0, 0);
    for (int i = 1; i < A::rows(); ++i) {
        result += a(i, i);
    }
    return result;
}
template <FixedMatrix A>
CUDA_EXPORT constexpr auto squaredNorm(A const& a) {
    using T = typename A::ValueType;
    T result = T(0);
    for (int i = 0; i < A::rows(); ++i) {
        for (int j = 0; j < A::cols(); ++j) {
            result += a(i, j) * a(i, j);
        }
    }
    return result;
}
template <FixedMatrix A>
CUDA_EXPORT auto frobeniusNorm(A const& a) {
    using T = typename A::ValueType;
    if constexpr (std::is_floating_point_v<T>) {
        // hypot rescales each pair: the norm can be representable even when
        // squaring an entry would overflow or underflow.
        T result = T(0);
        for (int i = 0; i < A::rows(); ++i) {
            for (int j = 0; j < A::cols(); ++j) {
                result = std::hypot(result, a(i, j));
            }
        }
        return result;
    } else {
        return simdSqrt(squaredNorm(a));
    }
}

enum class MatrixStatus { Success, Singular, NonFinite };

template <class T>
struct MatrixOptions {
    // Negative selects dimensions * machine epsilon. Zero permits exact-zero checks.
    T relativePivotTolerance = T(-1);
    T absolutePivotTolerance = T(0);
    // Cofactor-only guard on det(A / maxAbs(A)). Automatic dispatch raises this
    // independently of the pivot criterion; the original options are used by LU.
    T relativeDeterminantTolerance = T(0);
};

struct DefaultMatrixPolicy {
    static constexpr int kGeneralInverseThreshold = COSMICTIGER_MATRIX_GENERAL_INVERSE_THRESHOLD;
    static constexpr int kSymmetricInverseThreshold =
        COSMICTIGER_MATRIX_SYMMETRIC_INVERSE_THRESHOLD;
    static constexpr int kGeneralDeterminantThreshold =
        COSMICTIGER_MATRIX_GENERAL_DETERMINANT_THRESHOLD;
    static constexpr int kSymmetricDeterminantThreshold =
        COSMICTIGER_MATRIX_SYMMETRIC_DETERMINANT_THRESHOLD;
    static constexpr int kGeneralSolveThreshold = COSMICTIGER_MATRIX_GENERAL_SOLVE_THRESHOLD;
    static constexpr int kSymmetricSolveThreshold = COSMICTIGER_MATRIX_SYMMETRIC_SOLVE_THRESHOLD;
    static_assert(kGeneralInverseThreshold >= 0 && kGeneralInverseThreshold <= 6);
    static_assert(kSymmetricInverseThreshold >= 0 && kSymmetricInverseThreshold <= 6);
    static_assert(kGeneralDeterminantThreshold >= 0 && kGeneralDeterminantThreshold <= 6);
    static_assert(kSymmetricDeterminantThreshold >= 0 && kSymmetricDeterminantThreshold <= 6);
    static_assert(kGeneralSolveThreshold >= 0 && kGeneralSolveThreshold <= 6);
    static_assert(kSymmetricSolveThreshold >= 0 && kSymmetricSolveThreshold <= 6);
};

namespace detail {

    template <class T>
    CUDA_EXPORT constexpr bool matrixFinite(T value) {
        // Bit classification remains valid under -ffinite-math-only/-ffast-math.
        if constexpr (std::is_same_v<T, float>) {
            return (std::bit_cast<uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
        } else if constexpr (std::is_same_v<T, double>) {
            return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) !=
                0x7ff0000000000000ULL;
        } else {
            return std::isfinite(value);
        }
    }
    template <class T>
    CUDA_EXPORT constexpr T matrixAbs(T value) {
        return value < T(0) ? -value : value;
    }
    template <class T>
    CUDA_EXPORT constexpr T matrixMax(T a, T b) {
        return a > b ? a : b;
    }
    template <class T, int dimensions>
    CUDA_EXPORT constexpr T relativeTolerance(MatrixOptions<T> const& options) {
        return options.relativePivotTolerance < T(0) ?
            T(dimensions) * std::numeric_limits<T>::epsilon() :
            options.relativePivotTolerance;
    }
    template <FixedMatrix A>
    CUDA_EXPORT auto scaledMatrix(A const& a, Matrix<typename A::ValueType, A::rows()>& scaled,
        typename A::ValueType& scale, size_t* arithmeticOperations = nullptr) {
        using T = typename A::ValueType;
        scale = T(0);
        for (int i = 0; i < A::rows(); ++i) {
            for (int j = 0; j < A::cols(); ++j) {
                T const value = a(i, j);
                if (!matrixFinite(value)) {
                    return MatrixStatus::NonFinite;
                }
                scale = matrixMax(scale, matrixAbs(value));
            }
        }
        if (scale == T(0)) {
            return MatrixStatus::Singular;
        }
        if (arithmeticOperations != nullptr) {
            *arithmeticOperations += 1 + A::rows() * A::cols();
        }
        T const inverseScale = T(1) / scale;
        bool const multiply = matrixFinite(inverseScale) && inverseScale != T(0);
        for (int i = 0; i < A::rows(); ++i) {
            for (int j = 0; j < A::cols(); ++j) {
                // Keep division when the reciprocal cannot represent the scale.
                scaled(i, j) = multiply ? a(i, j) * inverseScale : a(i, j) / scale;
            }
        }
        return MatrixStatus::Success;
    }
    template <class T, int dimensions>
    CUDA_EXPORT constexpr T minorDeterminant(Matrix<T, dimensions> const& a) {
        if constexpr (dimensions == 1) {
            return a(0, 0);
        } else if constexpr (dimensions == 2) {
            return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
        } else if constexpr (dimensions == 3) {
            return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) +
                a(0, 1) * (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) +
                a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
        } else {
            T result = T(0);
            for (int column = 0; column < dimensions; ++column) {
                Matrix<T, dimensions - 1> minor;
                for (int i = 1; i < dimensions; ++i) {
                    int target = 0;
                    for (int j = 0; j < dimensions; ++j) {
                        if (j != column) {
                            minor(i - 1, target++) = a(i, j);
                        }
                    }
                }
                T const term = a(0, column) * minorDeterminant(minor);
                result += column % 2 == 0 ? term : -term;
            }
            return result;
        }
    }
    template <class T, int dimensions>
    CUDA_EXPORT constexpr T cofactor(Matrix<T, dimensions> const& a, int row, int column) {
        if constexpr (dimensions == 1) {
            return T(1);
        } else {
            Matrix<T, dimensions - 1> minor;
            int targetRow = 0;
            for (int i = 0; i < dimensions; ++i) {
                if (i == row) {
                    continue;
                }
                int targetColumn = 0;
                for (int j = 0; j < dimensions; ++j) {
                    if (j != column) {
                        minor(targetRow, targetColumn++) = a(i, j);
                    }
                }
                ++targetRow;
            }
            T const determinant = minorDeterminant(minor);
            return (row + column) % 2 == 0 ? determinant : -determinant;
        }
    }
    template <bool symmetric, class T, int dimensions>
    CUDA_EXPORT constexpr auto adjugate(Matrix<T, dimensions> const& a) {
        using Result =
            std::conditional_t<symmetric, SymmetricMatrix<T, dimensions>, Matrix<T, dimensions>>;
        Result result;
        if constexpr (dimensions == 3) {
            result(0, 0) = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
            result(1, 0) = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
            result(2, 0) = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
            result(1, 1) = a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0);
            result(2, 1) = a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1);
            result(2, 2) = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
            if constexpr (!symmetric) {
                result(0, 1) = a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2);
                result(0, 2) = a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1);
                result(1, 2) = a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2);
            }
        } else {
            for (int i = 0; i < dimensions; ++i) {
                for (int j = 0; j < (symmetric ? i + 1 : dimensions); ++j) {
                    result(i, j) = cofactor(a, j, i);
                }
            }
        }
        return result;
    }
    template <class T, int dimensions, FixedMatrix Adj>
    CUDA_EXPORT MatrixStatus explicitFactors(Matrix<T, dimensions> const& a, T scale,
        Adj const& adj, T& determinant, MatrixOptions<T> const& options) {
        if (!matrixFinite(options.relativePivotTolerance) ||
            !matrixFinite(options.absolutePivotTolerance) ||
            !matrixFinite(options.relativeDeterminantTolerance)) {
            return MatrixStatus::NonFinite;
        }
        determinant = T(0);
        T adjScale = T(0);
        for (int i = 0; i < dimensions; ++i) {
            determinant += a(0, i) * adj(i, 0);
            for (int j = 0; j < dimensions; ++j) {
                adjScale = matrixMax(adjScale, matrixAbs(adj(i, j)));
            }
        }
        T const tolerance = matrixMax(
            relativeTolerance<T, dimensions>(options), options.absolutePivotTolerance / scale);
        if (!matrixFinite(determinant) || !matrixFinite(adjScale)) {
            return MatrixStatus::NonFinite;
        }
        // The inverse-scale check alone misses cofactor cancellation when
        // several eigenvalues are small. The independent normalized-determinant
        // floor lets automatic dispatch use LU before that cancellation dominates.
        T const determinantTolerance =
            matrixMax(tolerance * adjScale, options.relativeDeterminantTolerance);
        return matrixAbs(determinant) <= determinantTolerance ? MatrixStatus::Singular :
                                                                MatrixStatus::Success;
    }
    template <FixedMatrix Out, FixedMatrix In>
    CUDA_EXPORT void assignMatrix(Out& out, In const& in) {
        for (int i = 0; i < Out::rows(); ++i) {
            for (int j = 0; j < (Traits<Out>::kSymmetric ? i + 1 : Out::cols()); ++j) {
                if constexpr (Traits<Out>::kDiagonal) {
                    if (i == j) {
                        out.diagonal(i) = in(i, i);
                    }
                } else {
                    out(i, j) = in(i, j);
                }
            }
        }
    }
    template <FixedMatrix A>
    CUDA_EXPORT bool allFinite(A const& a) {
        for (int i = 0; i < A::rows(); ++i) {
            for (int j = 0; j < A::cols(); ++j) {
                if (!matrixFinite(a(i, j))) {
                    return false;
                }
            }
        }
        return true;
    }
    template <class T, int dimensions>
    CUDA_EXPORT MatrixOptions<T> automaticExplicitOptions(MatrixOptions<T> options) {
        // Cofactor cancellation grows before singularity. Let LU handle this
        // region with the caller's original admissibility thresholds.
        T const cutoff = T(dimensions) * std::sqrt(std::numeric_limits<T>::epsilon());
        if (matrixFinite(options.relativePivotTolerance)) {
            options.relativePivotTolerance =
                matrixMax(relativeTolerance<T, dimensions>(options), cutoff);
        }
        if (matrixFinite(options.relativeDeterminantTolerance)) {
            options.relativeDeterminantTolerance =
                matrixMax(options.relativeDeterminantTolerance, cutoff);
        }
        return options;
    }
}    // namespace detail

// Float and double nonfinite classification remains reliable with fast math.
// Long double uses std::isfinite; compile its callers without finite-math-only
// when the NonFinite status is required for long-double input or intermediates.
template <class A>
concept ScalarSquareMatrix =
    FixedMatrix<A> && (A::rows() == A::cols()) && std::is_floating_point_v<typename A::ValueType>;

// Unscaled algebraic adjugate. The caller owns any conditioning and scaling
// policy; this also allows lane-wise SIMD arithmetic without scalar pivoting.
template <FixedMatrix A>
    requires(A::rows() == A::cols() && A::rows() <= 6)
CUDA_EXPORT constexpr auto adjugate(A const& a) {
    Matrix<typename A::ValueType, A::rows()> const dense(a);
    return detail::adjugate<detail::Traits<A>::kSymmetric>(dense);
}

namespace detail {
    template <class T, int dimensions>
    CUDA_EXPORT MatrixStatus checkDiagonal(
        DiagonalMatrix<T, dimensions> const& a, MatrixOptions<T> const& options) {
        if (!matrixFinite(options.relativePivotTolerance) ||
            !matrixFinite(options.absolutePivotTolerance) ||
            !matrixFinite(options.relativeDeterminantTolerance)) {
            return MatrixStatus::NonFinite;
        }
        for (int i = 0; i < dimensions; ++i) {
            T const value = a.diagonal(i);
            if (!matrixFinite(value)) {
                return MatrixStatus::NonFinite;
            }
            // Independent scalar equations have no cancellation or pivot growth.
            // Scale the relative threshold by each diagonal entry, not the largest.
            T const magnitude = matrixAbs(value);
            T const tolerance = matrixMax(options.absolutePivotTolerance,
                relativeTolerance<T, dimensions>(options) * magnitude);
            if (magnitude <= tolerance) {
                return MatrixStatus::Singular;
            }
        }
        return MatrixStatus::Success;
    }
}    // namespace detail

template <class T, int dimensions, FixedMatrix Out>
    requires(std::is_floating_point_v<T> && Out::rows() == dimensions &&
        Out::cols() == dimensions && std::is_same_v<T, typename Out::ValueType>)
CUDA_EXPORT MatrixStatus tryInverseExplicit(DiagonalMatrix<T, dimensions> const& a, Out& out,
    MatrixOptions<T> const& options = {}, size_t* arithmeticOperations = nullptr) {
    if (arithmeticOperations != nullptr) {
        *arithmeticOperations = 0;
    }
    auto const status = detail::checkDiagonal(a, options);
    if (status != MatrixStatus::Success) {
        return status;
    }
    DiagonalMatrix<T, dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
        if (arithmeticOperations != nullptr) {
            ++*arithmeticOperations;
        }
        result.diagonal(i) = T(1) / a.diagonal(i);
        if (!detail::matrixFinite(result.diagonal(i))) {
            return MatrixStatus::NonFinite;
        }
    }
    detail::assignMatrix(out, result);
    return MatrixStatus::Success;
}
template <class T, int dimensions>
    requires(std::is_floating_point_v<T>)
CUDA_EXPORT MatrixStatus tryDeterminantExplicit(
    DiagonalMatrix<T, dimensions> const& a, T& determinant, MatrixOptions<T> const& options = {}) {
    auto const status = detail::checkDiagonal(a, options);
    if (status != MatrixStatus::Success) {
        return status;
    }
    if constexpr (dimensions == 1) {
        determinant = a.diagonal(0);
        return MatrixStatus::Success;
    }
    T mantissa = T(1);
    int exponent = 0;
    for (int i = 0; i < dimensions; ++i) {
        int entryExponent;
        mantissa *= std::frexp(a.diagonal(i), &entryExponent);
        exponent += entryExponent;
        int adjustment;
        mantissa = std::frexp(mantissa, &adjustment);
        exponent += adjustment;
    }
    T const result = std::ldexp(mantissa, exponent);
    if (!detail::matrixFinite(result)) {
        return MatrixStatus::NonFinite;
    }
    determinant = result;
    return MatrixStatus::Success;
}
template <class T, int dimensions, int rhsCount>
    requires(std::is_floating_point_v<T>)
CUDA_EXPORT MatrixStatus trySolveExplicit(DiagonalMatrix<T, dimensions> const& a,
    Matrix<T, dimensions, rhsCount> const& rhs, Matrix<T, dimensions, rhsCount>& out,
    MatrixOptions<T> const& options = {}) {
    auto const status = detail::checkDiagonal(a, options);
    if (status != MatrixStatus::Success) {
        return status;
    }
    if (!detail::allFinite(rhs)) {
        return MatrixStatus::NonFinite;
    }
    Matrix<T, dimensions, rhsCount> result;
    for (int i = 0; i < dimensions; ++i) {
        for (int j = 0; j < rhsCount; ++j) {
            result(i, j) = rhs(i, j) / a.diagonal(i);
        }
    }
    if (!detail::allFinite(result)) {
        return MatrixStatus::NonFinite;
    }
    out = result;
    return MatrixStatus::Success;
}

// The optional inverse arithmetic counter is reset on entry and counts
// additions, subtractions, multiplications, and divisions, including failed
// reciprocal attempts. Automatic inversion sums explicit and fallback work.
// Validation arithmetic, swaps, assignments, and sign flips are excluded.
// On failure the output is unchanged. These are algebraic operations; SPD,
// conditioning, and application-specific admissibility remain caller policies.
template <ScalarSquareMatrix A, FixedMatrix Out>
    requires(A::rows() <= 6 && Out::rows() == A::rows() && Out::cols() == A::cols() &&
        std::is_same_v<typename A::ValueType, typename Out::ValueType> &&
        (!detail::Traits<Out>::kSymmetric || detail::Traits<A>::kSymmetric) &&
        (!detail::Traits<Out>::kDiagonal || detail::Traits<A>::kDiagonal))
CUDA_EXPORT MatrixStatus tryInverseExplicit(A const& a, Out& out,
    MatrixOptions<typename A::ValueType> const& options = {},
    size_t* arithmeticOperations = nullptr) {
    if (arithmeticOperations != nullptr) {
        *arithmeticOperations = 0;
    }
    using T = typename A::ValueType;
    if (!detail::matrixFinite(options.relativePivotTolerance) ||
        !detail::matrixFinite(options.absolutePivotTolerance) ||
        !detail::matrixFinite(options.relativeDeterminantTolerance)) {
        return MatrixStatus::NonFinite;
    }
    if constexpr (A::rows() == 1) {
        DiagonalMatrix<T, 1> const scalar(a(0, 0));
        return tryInverseExplicit(scalar, out, options, arithmeticOperations);
    }
    constexpr int kSize = A::rows();
    Matrix<T, kSize> scaled;
    T scale;
    auto status = detail::scaledMatrix(a, scaled, scale, arithmeticOperations);
    if (status != MatrixStatus::Success) {
        return status;
    }
    if (arithmeticOperations != nullptr) {
        constexpr size_t kEntries =
            detail::Traits<A>::kSymmetric ? kSize * (kSize + 1) / 2 : kSize * kSize;
        constexpr size_t kMinorOperations = kSize <= 2 ? 0 :
            kSize == 3                                 ? 3 :
            kSize == 4                                 ? 14 :
            kSize == 5                                 ? 64 :
                                                         330;
        *arithmeticOperations += kEntries * kMinorOperations + 2 * kSize;
    }
    auto result = detail::adjugate<detail::Traits<A>::kSymmetric>(scaled);
    T determinant;
    status = detail::explicitFactors(scaled, scale, result, determinant, options);
    if (status != MatrixStatus::Success) {
        return status;
    }
    T const factor = (T(1) / determinant) / scale;
    bool const multiply = detail::matrixFinite(factor) && factor != T(0);
    if (arithmeticOperations != nullptr) {
        constexpr size_t kEntries =
            detail::Traits<A>::kSymmetric ? kSize * (kSize + 1) / 2 : kSize * kSize;
        *arithmeticOperations += 2 + kEntries * (multiply ? 1 : 2);
    }
    for (int i = 0; i < kSize; ++i) {
        for (int j = 0; j < (detail::Traits<A>::kSymmetric ? i + 1 : kSize); ++j) {
            result(i, j) = multiply ? result(i, j) * factor : (result(i, j) / determinant) / scale;
        }
    }
    if (!detail::allFinite(result)) {
        return MatrixStatus::NonFinite;
    }
    detail::assignMatrix(out, result);
    return MatrixStatus::Success;
}

template <ScalarSquareMatrix A>
    requires(A::rows() <= 6)
CUDA_EXPORT MatrixStatus tryDeterminantExplicit(A const& a, typename A::ValueType& determinant,
    MatrixOptions<typename A::ValueType> const& options = {}) {
    using T = typename A::ValueType;
    if constexpr (A::rows() == 1) {
        DiagonalMatrix<T, 1> const scalar(a(0, 0));
        return tryDeterminantExplicit(scalar, determinant, options);
    }
    Matrix<T, A::rows()> scaled;
    T scale;
    auto status = detail::scaledMatrix(a, scaled, scale);
    if (status != MatrixStatus::Success) {
        return status;
    }
    auto const adj = detail::adjugate<detail::Traits<A>::kSymmetric>(scaled);
    T result;
    status = detail::explicitFactors(scaled, scale, adj, result, options);
    if (status != MatrixStatus::Success) {
        return status;
    }
    for (int i = 0; i < A::rows(); ++i) {
        result *= scale;
    }
    if (!detail::matrixFinite(result)) {
        return MatrixStatus::NonFinite;
    }
    determinant = result;
    return MatrixStatus::Success;
}

namespace detail {
    template <class T, int dimensions>
    struct PivotFactors {
        Matrix<T, dimensions> lu;
        int swaps[dimensions]{};
        int parity = 1;
    };
    template <ScalarSquareMatrix A>
    CUDA_EXPORT MatrixStatus factorPivoted(A const& a,
        PivotFactors<typename A::ValueType, A::rows()>& factors,
        MatrixOptions<typename A::ValueType> const& options,
        size_t* arithmeticOperations = nullptr) {
        using T = typename A::ValueType;
        constexpr int kSize = A::rows();
        T scale = T(0);
        for (int i = 0; i < kSize; ++i) {
            for (int j = 0; j < kSize; ++j) {
                T const value = a(i, j);
                if (!matrixFinite(value)) {
                    return MatrixStatus::NonFinite;
                }
                factors.lu(i, j) = value;
                scale = matrixMax(scale, matrixAbs(value));
            }
        }
        if (!matrixFinite(options.relativePivotTolerance) ||
            !matrixFinite(options.absolutePivotTolerance) ||
            !matrixFinite(options.relativeDeterminantTolerance)) {
            return MatrixStatus::NonFinite;
        }
        T const tolerance =
            matrixMax(relativeTolerance<T, kSize>(options) * scale, options.absolutePivotTolerance);
        auto& lu = factors.lu;
        for (int column = 0; column < kSize; ++column) {
            int pivot = column;
            T biggest = matrixAbs(lu(column, column));
            for (int row = column + 1; row < kSize; ++row) {
                T const value = matrixAbs(lu(row, column));
                if (value > biggest) {
                    biggest = value;
                    pivot = row;
                }
            }
            if (!matrixFinite(biggest)) {
                return MatrixStatus::NonFinite;
            }
            if (biggest <= tolerance) {
                return MatrixStatus::Singular;
            }
            factors.swaps[column] = pivot;
            if (pivot != column) {
                for (int j = 0; j < kSize; ++j) {
                    T const value = lu(column, j);
                    lu(column, j) = lu(pivot, j);
                    lu(pivot, j) = value;
                }
                factors.parity = -factors.parity;
            }
            if (arithmeticOperations != nullptr) {
                size_t const remaining = kSize - column - 1;
                *arithmeticOperations += remaining * (1 + 2 * remaining);
            }
            for (int row = column + 1; row < kSize; ++row) {
                lu(row, column) /= lu(column, column);
                for (int j = column + 1; j < kSize; ++j) {
                    lu(row, j) -= lu(row, column) * lu(column, j);
                }
            }
        }
        return allFinite(lu) ? MatrixStatus::Success : MatrixStatus::NonFinite;
    }
    template <class T, int dimensions, int rhsCount>
    CUDA_EXPORT MatrixStatus solveFactors(PivotFactors<T, dimensions> const& factors,
        Matrix<T, dimensions, rhsCount> const& rhs, Matrix<T, dimensions, rhsCount>& result,
        size_t* arithmeticOperations = nullptr) {
        if (!allFinite(rhs)) {
            return MatrixStatus::NonFinite;
        }
        if (arithmeticOperations != nullptr) {
            // Both triangular passes count each multiply/subtract pair, even
            // when the identity right-hand sides contain mathematical zeros.
            *arithmeticOperations += size_t(rhsCount) * dimensions * (2 * dimensions - 1);
        }
        result = rhs;
        for (int i = 0; i < dimensions; ++i) {
            if (factors.swaps[i] != i) {
                auto const row = result[i];
                result[i] = result[factors.swaps[i]];
                result[factors.swaps[i]] = row;
            }
        }
        for (int i = 0; i < dimensions; ++i) {
            for (int j = 0; j < i; ++j) {
                for (int k = 0; k < rhsCount; ++k) {
                    result(i, k) -= factors.lu(i, j) * result(j, k);
                }
            }
        }
        for (int i = dimensions - 1; i >= 0; --i) {
            for (int j = i + 1; j < dimensions; ++j) {
                for (int k = 0; k < rhsCount; ++k) {
                    result(i, k) -= factors.lu(i, j) * result(j, k);
                }
            }
            for (int k = 0; k < rhsCount; ++k) {
                result(i, k) /= factors.lu(i, i);
            }
        }
        return allFinite(result) ? MatrixStatus::Success : MatrixStatus::NonFinite;
    }
}    // namespace detail

template <ScalarSquareMatrix A, int rhsCount>
CUDA_EXPORT MatrixStatus trySolvePivoted(A const& a,
    Matrix<typename A::ValueType, A::rows(), rhsCount> const& rhs,
    Matrix<typename A::ValueType, A::rows(), rhsCount>& out,
    MatrixOptions<typename A::ValueType> const& options = {}) {
    if constexpr (detail::Traits<A>::kDiagonal) {
        return trySolveExplicit(a, rhs, out, options);
    }
    using T = typename A::ValueType;
    detail::PivotFactors<T, A::rows()> factors;
    auto status = detail::factorPivoted(a, factors, options);
    if (status != MatrixStatus::Success) {
        return status;
    }
    Matrix<T, A::rows(), rhsCount> result;
    status = detail::solveFactors(factors, rhs, result);
    if (status == MatrixStatus::Success) {
        out = result;
    }
    return status;
}
template <ScalarSquareMatrix A, FixedMatrix Out>
    requires(Out::rows() == A::rows() && Out::cols() == A::cols() &&
        std::is_same_v<typename A::ValueType, typename Out::ValueType> &&
        (!detail::Traits<Out>::kSymmetric || detail::Traits<A>::kSymmetric) &&
        (!detail::Traits<Out>::kDiagonal || detail::Traits<A>::kDiagonal))
CUDA_EXPORT MatrixStatus tryInversePivoted(A const& a, Out& out,
    MatrixOptions<typename A::ValueType> const& options = {},
    size_t* arithmeticOperations = nullptr) {
    if (arithmeticOperations != nullptr) {
        *arithmeticOperations = 0;
    }
    if constexpr (detail::Traits<A>::kDiagonal) {
        return tryInverseExplicit(a, out, options, arithmeticOperations);
    }
    using T = typename A::ValueType;
    detail::PivotFactors<T, A::rows()> factors;
    auto status = detail::factorPivoted(a, factors, options, arithmeticOperations);
    if (status != MatrixStatus::Success) {
        return status;
    }
    Matrix<T, A::rows()> const identity(identityMatrix<T, A::rows()>());
    Matrix<T, A::rows()> result;
    status = detail::solveFactors(factors, identity, result, arithmeticOperations);
    if (status == MatrixStatus::Success) {
        if constexpr (detail::Traits<Out>::kSymmetric && !detail::Traits<Out>::kDiagonal) {
            // Each independently solved column has its own rounding error.
            // Copying just one triangle can destroy cancellation in A*inverse
            // for ill-conditioned matrices; project both triangles together.
            for (int i = 0; i < A::rows(); ++i) {
                out(i, i) = result(i, i);
                for (int j = 0; j < i; ++j) {
                    out(i, j) = T(0.5) * result(i, j) + T(0.5) * result(j, i);
                }
            }
            if (arithmeticOperations != nullptr) {
                *arithmeticOperations += size_t(3) * A::rows() * (A::rows() - 1) / 2;
            }
        } else {
            detail::assignMatrix(out, result);
        }
    }
    return status;
}
template <ScalarSquareMatrix A>
CUDA_EXPORT MatrixStatus tryDeterminantPivoted(A const& a, typename A::ValueType& determinant,
    MatrixOptions<typename A::ValueType> const& options = {}) {
    if constexpr (detail::Traits<A>::kDiagonal) {
        return tryDeterminantExplicit(a, determinant, options);
    }
    using T = typename A::ValueType;
    detail::PivotFactors<T, A::rows()> factors;
    auto const status = detail::factorPivoted(a, factors, options);
    if (status != MatrixStatus::Success) {
        return status;
    }
    // Track the product's exponent so large and small pivots can cancel
    // without an intermediate overflow or underflow changing the determinant.
    T mantissa = T(factors.parity);
    int exponent = 0;
    for (int i = 0; i < A::rows(); ++i) {
        int entryExponent;
        mantissa *= std::frexp(factors.lu(i, i), &entryExponent);
        exponent += entryExponent;
        int adjustment;
        mantissa = std::frexp(mantissa, &adjustment);
        exponent += adjustment;
    }
    T const result = std::ldexp(mantissa, exponent);
    if (!detail::matrixFinite(result)) {
        return MatrixStatus::NonFinite;
    }
    determinant = result;
    return MatrixStatus::Success;
}

template <ScalarSquareMatrix A, int rhsCount>
    requires(A::rows() <= 6)
CUDA_EXPORT MatrixStatus trySolveExplicit(A const& a,
    Matrix<typename A::ValueType, A::rows(), rhsCount> const& rhs,
    Matrix<typename A::ValueType, A::rows(), rhsCount>& out,
    MatrixOptions<typename A::ValueType> const& options = {}) {
    using T = typename A::ValueType;
    if constexpr (A::rows() == 1) {
        DiagonalMatrix<T, 1> const scalar(a(0, 0));
        return trySolveExplicit(scalar, rhs, out, options);
    }
    Matrix<T, A::rows()> scaled;
    T scale;
    auto status = detail::scaledMatrix(a, scaled, scale);
    if (status != MatrixStatus::Success) {
        return status;
    }
    if (!detail::allFinite(rhs)) {
        return MatrixStatus::NonFinite;
    }
    auto const adj = detail::adjugate<detail::Traits<A>::kSymmetric>(scaled);
    T determinant;
    status = detail::explicitFactors(scaled, scale, adj, determinant, options);
    if (status != MatrixStatus::Success) {
        return status;
    }
    auto result = adj * rhs;
    T const factor = (T(1) / determinant) / scale;
    bool const multiply = detail::matrixFinite(factor) && factor != T(0);
    for (int i = 0; i < A::rows(); ++i) {
        for (int j = 0; j < rhsCount; ++j) {
            result(i, j) = multiply ? result(i, j) * factor : (result(i, j) / determinant) / scale;
        }
    }
    if (!detail::allFinite(result)) {
        return MatrixStatus::NonFinite;
    }
    out = result;
    return MatrixStatus::Success;
}

// Any failed cofactor attempt is retried with pivoting and the original caller
// tolerances. The pivoted path checks inputs again, so this can recover finite
// results from overflowing cofactor intermediates without accepting NaN inputs.
template <class Policy = DefaultMatrixPolicy, ScalarSquareMatrix A, FixedMatrix Out>
    requires(Out::rows() == A::rows() && Out::cols() == A::cols() &&
        std::is_same_v<typename A::ValueType, typename Out::ValueType> &&
        (!detail::Traits<Out>::kSymmetric || detail::Traits<A>::kSymmetric) &&
        (!detail::Traits<Out>::kDiagonal || detail::Traits<A>::kDiagonal))
CUDA_EXPORT MatrixStatus tryInverse(A const& a, Out& out,
    MatrixOptions<typename A::ValueType> const& options = {},
    size_t* arithmeticOperations = nullptr) {
    if (arithmeticOperations != nullptr) {
        *arithmeticOperations = 0;
    }
    constexpr int kThreshold = detail::Traits<A>::kSymmetric ? Policy::kSymmetricInverseThreshold :
                                                               Policy::kGeneralInverseThreshold;
    if constexpr (detail::Traits<A>::kDiagonal || (A::rows() <= kThreshold && A::rows() <= 6)) {
        auto const explicitOptions = detail::Traits<A>::kDiagonal ?
            options :
            detail::automaticExplicitOptions<typename A::ValueType, A::rows()>(options);
        size_t explicitOperations = 0;
        auto const status = tryInverseExplicit(a, out, explicitOptions,
            arithmeticOperations != nullptr ? &explicitOperations : nullptr);
        if (arithmeticOperations != nullptr) {
            *arithmeticOperations = explicitOperations;
        }
        if (status == MatrixStatus::Success) {
            return status;
        }
    }
    size_t pivotOperations = 0;
    auto const status = tryInversePivoted(
        a, out, options, arithmeticOperations != nullptr ? &pivotOperations : nullptr);
    if (arithmeticOperations != nullptr) {
        *arithmeticOperations += pivotOperations;
    }
    return status;
}
template <class Policy = DefaultMatrixPolicy, ScalarSquareMatrix A>
CUDA_EXPORT MatrixStatus tryDeterminant(A const& a, typename A::ValueType& determinant,
    MatrixOptions<typename A::ValueType> const& options = {}) {
    constexpr int kThreshold = detail::Traits<A>::kSymmetric ?
        Policy::kSymmetricDeterminantThreshold :
        Policy::kGeneralDeterminantThreshold;
    if constexpr (detail::Traits<A>::kDiagonal || (A::rows() <= kThreshold && A::rows() <= 6)) {
        auto const explicitOptions = detail::Traits<A>::kDiagonal ?
            options :
            detail::automaticExplicitOptions<typename A::ValueType, A::rows()>(options);
        auto const status = tryDeterminantExplicit(a, determinant, explicitOptions);
        if (status == MatrixStatus::Success) {
            return status;
        }
    }
    return tryDeterminantPivoted(a, determinant, options);
}
template <class Policy = DefaultMatrixPolicy, ScalarSquareMatrix A, int rhsCount>
CUDA_EXPORT MatrixStatus trySolve(A const& a,
    Matrix<typename A::ValueType, A::rows(), rhsCount> const& rhs,
    Matrix<typename A::ValueType, A::rows(), rhsCount>& out,
    MatrixOptions<typename A::ValueType> const& options = {}) {
    constexpr int kThreshold = detail::Traits<A>::kSymmetric ? Policy::kSymmetricSolveThreshold :
                                                               Policy::kGeneralSolveThreshold;
    if constexpr (detail::Traits<A>::kDiagonal || (A::rows() <= kThreshold && A::rows() <= 6)) {
        auto const explicitOptions = detail::Traits<A>::kDiagonal ?
            options :
            detail::automaticExplicitOptions<typename A::ValueType, A::rows()>(options);
        auto const status = trySolveExplicit(a, rhs, out, explicitOptions);
        if (status == MatrixStatus::Success) {
            return status;
        }
    }
    return trySolvePivoted(a, rhs, out, options);
}

#define COSMICTIGER_MATRIX_VECTOR_SOLVE(functionName)                 \
    template <ScalarSquareMatrix A, class Derived, class OutDerived>  \
    CUDA_EXPORT MatrixStatus functionName(A const& a,                 \
        Vector<typename A::ValueType, A::rows(), Derived> const& rhs, \
        Vector<typename A::ValueType, A::rows(), OutDerived>& out,    \
        MatrixOptions<typename A::ValueType> const& options = {}) {   \
        Matrix<typename A::ValueType, A::rows(), 1> b;                \
        Matrix<typename A::ValueType, A::rows(), 1> x;                \
        for (int i = 0; i < A::rows(); ++i) {                         \
            b(i, 0) = rhs[i];                                         \
        }                                                             \
        auto const status = functionName(a, b, x, options);           \
        if (status == MatrixStatus::Success) {                        \
            for (int i = 0; i < A::rows(); ++i) {                     \
                out[i] = x(i, 0);                                     \
            }                                                         \
        }                                                             \
        return status;                                                \
    }
COSMICTIGER_MATRIX_VECTOR_SOLVE(trySolveExplicit)
COSMICTIGER_MATRIX_VECTOR_SOLVE(trySolvePivoted)
#undef COSMICTIGER_MATRIX_VECTOR_SOLVE

template <class Policy = DefaultMatrixPolicy, ScalarSquareMatrix A, class Derived, class OutDerived>
CUDA_EXPORT MatrixStatus trySolve(A const& a,
    Vector<typename A::ValueType, A::rows(), Derived> const& rhs,
    Vector<typename A::ValueType, A::rows(), OutDerived>& out,
    MatrixOptions<typename A::ValueType> const& options = {}) {
    Matrix<typename A::ValueType, A::rows(), 1> b;
    Matrix<typename A::ValueType, A::rows(), 1> x;
    for (int i = 0; i < A::rows(); ++i) {
        b(i, 0) = rhs[i];
    }
    auto const status = trySolve<Policy>(a, b, x, options);
    if (status == MatrixStatus::Success) {
        for (int i = 0; i < A::rows(); ++i) {
            out[i] = x(i, 0);
        }
    }
    return status;
}
}    // namespace math
