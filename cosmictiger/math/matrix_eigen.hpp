/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cmath>
#include <limits>

#include <cosmictiger/math/matrix.hpp>

namespace math {

enum class JacobiPivot { Cyclic, LargestOffDiagonal };

template <class T>
struct SymmetricEigenOptions {
    JacobiPivot pivot = JacobiPivot::LargestOffDiagonal;
    // Cyclic: complete sweeps. LargestOffDiagonal: individual rotations.
    int maxIterations = 64;
    T absoluteOffDiagonalTolerance = T(0);
    T relativeOffDiagonalTolerance = T(8) * std::numeric_limits<T>::epsilon();
    // hypot avoids overflow in the rotation parameters. The normalized SPH
    // reconstruction keeps its original sqrt arithmetic and cyclic policy.
    bool useHypot = true;
};

template <class T, int dimensions>
struct SymmetricEigenpairs {
    Vector<T, dimensions> values{};
    // Eigenvectors are columns; eigenvalues are sorted largest first.
    Matrix<T, dimensions> vectors{};
    bool converged = false;
};

namespace detail {

    struct EigenNoCounter {
        CUDA_EXPORT constexpr void operator+=(int) {}
    };

    template <class T, int dimensions, class Counter>
    CUDA_EXPORT inline void jacobiRotate(SymmetricMatrix<T, dimensions>& matrix,
        Matrix<T, dimensions>& vectors, int first, int second, bool useHypot, Counter& operations) {
        T const offDiagonal = matrix(first, second);
        T const tangentTwice =
            (matrix(second, second) - matrix(first, first)) / (T(2) * offDiagonal);
        T const tangent = useHypot ? std::copysign(T(1), tangentTwice) /
                (std::abs(tangentTwice) + std::hypot(T(1), tangentTwice)) :
                                     (tangentTwice < T(0) ? T(-1) : T(1)) /
                (std::abs(tangentTwice) + std::sqrt(T(1) + tangentTwice * tangentTwice));
        T const cosine = useHypot ? T(1) / std::hypot(T(1), tangent) :
                                    T(1) / std::sqrt(T(1) + tangent * tangent);
        T const sine = tangent * cosine;
        matrix(first, first) -= tangent * offDiagonal;
        matrix(second, second) += tangent * offDiagonal;
        matrix(first, second) = T(0);
        for (int axis = 0; axis < dimensions; ++axis) {
            if (axis != first && axis != second) {
                T const firstValue = matrix(axis, first);
                T const secondValue = matrix(axis, second);
                matrix(axis, first) = cosine * firstValue - sine * secondValue;
                matrix(axis, second) = sine * firstValue + cosine * secondValue;
            }
            T const firstVector = vectors(axis, first);
            T const secondVector = vectors(axis, second);
            vectors(axis, first) = cosine * firstVector - sine * secondVector;
            vectors(axis, second) = sine * firstVector + cosine * secondVector;
        }
        // sqrt path: parameters (14), diagonal updates (4), remaining matrix
        // pairs (6 each), and eigenvector pairs (6 each). hypot costs one.
        operations += (useHypot ? 14 : 18) + 6 * (dimensions - 2) + 6 * dimensions;
    }

}    // namespace detail

// Low-level Jacobi iteration for an already normalized symmetric matrix. It
// retains the supplied iteration policy, leaves eigenvalues unsorted on the
// diagonal, and puts their eigenvectors in columns. No scaling is hidden here.
template <class T, int dimensions, class Counter>
CUDA_EXPORT inline void jacobiDiagonalize(SymmetricMatrix<T, dimensions>& matrix,
    Matrix<T, dimensions>& vectors, SymmetricEigenOptions<T> const& options, Counter& operations) {
    vectors = Matrix<T, dimensions>{};
    for (int axis = 0; axis < dimensions; ++axis) {
        vectors(axis, axis) = T(1);
    }
    if constexpr (dimensions > 1) {
        for (int iteration = 0; iteration < options.maxIterations; ++iteration) {
            T tolerance = options.absoluteOffDiagonalTolerance;
            if (options.relativeOffDiagonalTolerance != T(0)) {
                T largestDiagonal = T(0);
                for (int axis = 0; axis < dimensions; ++axis) {
                    largestDiagonal =
                        detail::matrixMax(largestDiagonal, std::abs(matrix(axis, axis)));
                }
                tolerance = detail::matrixMax(
                    tolerance, options.relativeOffDiagonalTolerance * largestDiagonal);
                operations += dimensions + 1;
            }
            if (options.pivot == JacobiPivot::Cyclic) {
                for (int first = 0; first < dimensions - 1; ++first) {
                    for (int second = first + 1; second < dimensions; ++second) {
                        operations += 1;
                        if (std::abs(matrix(first, second)) > tolerance) {
                            detail::jacobiRotate(
                                matrix, vectors, first, second, options.useHypot, operations);
                        }
                    }
                }
            } else {
                int first = 0;
                int second = 1;
                T largestOffDiagonal = std::abs(matrix(first, second));
                operations += 1;
                for (int row = 0; row < dimensions; ++row) {
                    for (int column = row + 1; column < dimensions; ++column) {
                        T const magnitude = std::abs(matrix(row, column));
                        operations += 1;
                        if (magnitude > largestOffDiagonal) {
                            first = row;
                            second = column;
                            largestOffDiagonal = magnitude;
                        }
                    }
                }
                if (largestOffDiagonal <= tolerance) {
                    break;
                }
                detail::jacobiRotate(matrix, vectors, first, second, options.useHypot, operations);
            }
        }
    }
}

// Finite input is symmetrized and scaled before iteration. Failure leaves out
// unchanged. Success reports finite results; out.converged separately reports
// whether the requested off-diagonal tolerance was reached within the budget.
template <ScalarSquareMatrix A>
CUDA_EXPORT inline MatrixStatus trySymmetricEigenpairs(A const& input,
    SymmetricEigenpairs<typename A::ValueType, A::rows()>& out,
    SymmetricEigenOptions<typename A::ValueType> const& options = {}) {
    using T = typename A::ValueType;
    constexpr int kSize = A::rows();
    T scale = T(0);
    for (int row = 0; row < kSize; ++row) {
        for (int column = 0; column < kSize; ++column) {
            T const value = input(row, column);
            if (!detail::matrixFinite(value)) {
                return MatrixStatus::NonFinite;
            }
            scale = detail::matrixMax(scale, std::abs(value));
        }
    }
    SymmetricMatrix<T, kSize> diagonalized;
    if (scale > T(0)) {
        for (int row = 0; row < kSize; ++row) {
            for (int column = 0; column <= row; ++column) {
                diagonalized(row, column) =
                    T(0.5) * (input(row, column) / scale + input(column, row) / scale);
            }
        }
    }
    Matrix<T, kSize> vectors;
    detail::EigenNoCounter counter;
    jacobiDiagonalize(diagonalized, vectors, options, counter);
    T largestDiagonal = T(0);
    T largestOffDiagonal = T(0);
    int order[kSize];
    for (int row = 0; row < kSize; ++row) {
        order[row] = row;
        largestDiagonal = detail::matrixMax(largestDiagonal, std::abs(diagonalized(row, row)));
        for (int column = row + 1; column < kSize; ++column) {
            largestOffDiagonal =
                detail::matrixMax(largestOffDiagonal, std::abs(diagonalized(row, column)));
        }
    }
    // Stable insertion sort avoids allocation and works in device code.
    for (int axis = 1; axis < kSize; ++axis) {
        int const selected = order[axis];
        int position = axis;
        while (position > 0 &&
            diagonalized(selected, selected) >
                diagonalized(order[position - 1], order[position - 1])) {
            order[position] = order[position - 1];
            --position;
        }
        order[position] = selected;
    }
    SymmetricEigenpairs<T, kSize> result;
    result.converged =
        largestOffDiagonal <= detail::matrixMax(options.absoluteOffDiagonalTolerance,
                                  options.relativeOffDiagonalTolerance * largestDiagonal);
    for (int axis = 0; axis < kSize; ++axis) {
        result.values[axis] = diagonalized(order[axis], order[axis]) * scale;
        if (!detail::matrixFinite(result.values[axis])) {
            return MatrixStatus::NonFinite;
        }
        for (int component = 0; component < kSize; ++component) {
            result.vectors(component, axis) = vectors(component, order[axis]);
        }
    }
    if (!detail::allFinite(result.vectors)) {
        return MatrixStatus::NonFinite;
    }
    out = result;
    return MatrixStatus::Success;
}

}    // namespace math
