/* CosmicTiger - fixed matrix CUDA checks; BSL-1.0. */

#include <cmath>
#include <limits>
#include <type_traits>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cosmictiger/math/matrix.hpp>

namespace detail {
template <class T, int dimensions>
__device__ int checkDeviceSystem() {
    math::Matrix<T, dimensions> matrix{};
    math::Vector<T, dimensions> rhs{}, solution{};
    for (int i = 0; i < dimensions; ++i) {
        rhs[i] = T(2 * (i + 1) + dimensions * (dimensions + 1) / 2);
        for (int j = 0; j < dimensions; ++j) {
            matrix(i, j) = T(i == j ? 3 : 1);
        }
    }
    math::Matrix<T, dimensions> inverse{};
    int failures = 0;
    T const tolerance = T(128 * dimensions) * std::numeric_limits<T>::epsilon();
    for (int algorithm = 0; algorithm < 3; ++algorithm) {
        math::MatrixStatus status;
        if (algorithm == 0) {
            status = math::tryInverse(matrix, inverse);
        } else if (algorithm == 1) {
            status = math::tryInversePivoted(matrix, inverse);
        } else {
            if constexpr (dimensions <= 6) {
                status = math::tryInverseExplicit(matrix, inverse);
            } else {
                continue;
            }
        }
        if (status != math::MatrixStatus::Success) {
            ++failures;
        }
        for (int i = 0; i < dimensions; ++i) {
            for (int j = 0; j < dimensions; ++j) {
                T const expected = T(0.5) * (T(i == j ? 1 : 0) - T(1) / T(dimensions + 2));
                if (::fabs(inverse(i, j) - expected) > tolerance) {
                    ++failures;
                }
            }
        }
    }
    if (math::trySolve(matrix, rhs, solution) != math::MatrixStatus::Success) {
        ++failures;
    }
    for (int i = 0; i < dimensions; ++i) {
        if (::fabs(solution[i] - T(i + 1)) > tolerance) {
            ++failures;
        }
    }
    T determinant{};
    if (math::tryDeterminant(matrix, determinant) != math::MatrixStatus::Success ||
        ::fabs(determinant / T((1 << (dimensions - 1)) * (dimensions + 2)) - T(1)) > tolerance) {
        ++failures;
    }
    return failures;
}

template <class T>
__global__ void matrixKernel(int* failures) {
    int count = checkDeviceSystem<T, 3>() + checkDeviceSystem<T, 16>();
    math::SymmetricMatrix<T, 3> symmetric{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            symmetric(i, j) = T(i == j ? 3 : 1);
        }
    }
    math::SymmetricMatrix<T, 3> inverse{};
    if (math::tryInverse(symmetric, inverse) != math::MatrixStatus::Success ||
        ::fabs(inverse(0, 0) - T(0.4)) > T(128) * std::numeric_limits<T>::epsilon()) {
        ++count;
    }
    // Two nearly unresolved eigenvalues exercise the cofactor safeguard and
    // packed pivot averaging, independently of a host/device agreement check.
    T const delta = std::is_same_v<T, float> ? T(1e-2) : T(1e-6);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            symmetric(i, j) = T((i + 1) * (j + 1)) + (i == j ? delta : T(0));
        }
    }
    if (math::tryInverse(symmetric, inverse) != math::MatrixStatus::Success) {
        ++count;
    }
    T const residualTolerance = T(32) * std::numeric_limits<T>::epsilon() * (T(14) + delta) / delta;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T value = T(0);
            for (int k = 0; k < 3; ++k) {
                value += symmetric(i, k) * inverse(k, j);
            }
            if (::fabs(value - T(i == j ? 1 : 0)) > residualTolerance) {
                ++count;
            }
        }
    }
    math::DiagonalMatrix<T, 9> diagonal{}, diagonalInverse{};
    for (int i = 0; i < 9; ++i) {
        diagonal.diagonal(i) = T(i + 1);
    }
    if (math::tryInverse(diagonal, diagonalInverse) != math::MatrixStatus::Success ||
        diagonalInverse.diagonal(7) != T(0.125)) {
        ++count;
    }
    symmetric(0, 0) = std::numeric_limits<T>::quiet_NaN();
    if (math::tryInverse(symmetric, inverse) != math::MatrixStatus::NonFinite) {
        ++count;
    }
    *failures = count;
}

template <class T>
void checkDevice() {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    if (count == 0) {
        GTEST_SKIP() << "No CUDA device";
    }
    int* deviceFailures = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceFailures, sizeof(int)), cudaSuccess);
    // Exactly one block bounds occupancy even while the workstation is shared.
    matrixKernel<T><<<1, 1>>>(deviceFailures);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    int failures = -1;
    EXPECT_EQ(
        cudaMemcpy(&failures, deviceFailures, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaFree(deviceFailures), cudaSuccess);
    EXPECT_EQ(failures, 0);
}
}    // namespace detail

TEST(MatrixCuda, FloatAlgorithms) {
    detail::checkDevice<float>();
}
TEST(MatrixCuda, DoubleAlgorithms) {
    detail::checkDevice<double>();
}
