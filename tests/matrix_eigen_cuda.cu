#include <algorithm>
#include <cmath>
#include <limits>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cosmictiger/math/matrix_eigen.hpp>

namespace detail {

template <class Real>
struct EigenCudaResult {
    Real eigenvalues[3]{};
    Real eigenvectors[3][3]{};
    math::MatrixStatus eigenStatus = math::MatrixStatus::NonFinite;
    bool converged = false;
};

template <class Real>
CUDA_EXPORT EigenCudaResult<Real> evaluateEigenpairs(math::Matrix<Real, 3> const& matrix) {
    EigenCudaResult<Real> result;
    math::SymmetricEigenpairs<Real, 3> eigenpairs;
    result.eigenStatus = math::trySymmetricEigenpairs(matrix, eigenpairs);
    result.converged = eigenpairs.converged;
    for (int row = 0; row < 3; ++row) {
        result.eigenvalues[row] = eigenpairs.values[row];
        for (int column = 0; column < 3; ++column) {
            result.eigenvectors[row][column] = eigenpairs.vectors(row, column);
        }
    }
    return result;
}

template <class Real>
__global__ void eigenKernel(math::Matrix<Real, 3> const* moments, EigenCudaResult<Real>* results) {
    int const sample = threadIdx.x;
    if (sample < 3) {
        results[sample] = evaluateEigenpairs(moments[sample]);
    }
}

template <class Real>
void testEigenDeviceParity() {
    int devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
    if (devices == 0) {
        GTEST_SKIP() << "No CUDA device";
    }
    // Three orthogonal oblique directions give rank1/rank2/full SPD cases.
    Real const directions[3][3] = {
        {Real(1), Real(2), Real(-1)}, {Real(2), Real(-1), Real(0)}, {Real(-1), Real(-2), Real(-5)}};
    math::Matrix<Real, 3> moments[3]{};
    for (int sample = 0; sample < 3; ++sample) {
        for (int direction = 0; direction <= sample; ++direction) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    moments[sample][row][column] +=
                        directions[direction][row] * directions[direction][column];
                }
            }
        }
    }
    math::Matrix<Real, 3>* deviceMoments = nullptr;
    EigenCudaResult<Real>* deviceResults = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceMoments, sizeof(moments)), cudaSuccess);
    cudaError_t const resultAllocation =
        cudaMalloc(&deviceResults, 3 * sizeof(EigenCudaResult<Real>));
    if (resultAllocation != cudaSuccess) {
        cudaFree(deviceMoments);
        FAIL() << cudaGetErrorString(resultAllocation);
    }
    cudaError_t status =
        cudaMemcpy(deviceMoments, moments, sizeof(moments), cudaMemcpyHostToDevice);
    EigenCudaResult<Real> results[3];
    if (status == cudaSuccess) {
        eigenKernel<<<1, 3>>>(deviceMoments, deviceResults);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(results, deviceResults, sizeof(results), cudaMemcpyDeviceToHost);
    }
    EXPECT_EQ(cudaFree(deviceResults), cudaSuccess);
    EXPECT_EQ(cudaFree(deviceMoments), cudaSuccess);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    Real const tolerance = Real(1024) * std::numeric_limits<Real>::epsilon();
    for (int sample = 0; sample < 3; ++sample) {
        SCOPED_TRACE(sample + 1);
        auto const expected = evaluateEigenpairs(moments[sample]);
        EXPECT_EQ(results[sample].eigenStatus, math::MatrixStatus::Success);
        EXPECT_TRUE(results[sample].converged);
        for (int row = 0; row < 3; ++row) {
            EXPECT_NEAR(
                results[sample].eigenvalues[row], expected.eigenvalues[row], tolerance * Real(30));
            // Eigenvectors can change sign (or basis within the repeated zero
            // eigenspace). Verify the device eigenproblem and orthogonality.
            for (int axis = 0; axis < 3; ++axis) {
                Real product = Real(0);
                Real dot = Real(0);
                for (int component = 0; component < 3; ++component) {
                    product += moments[sample][row][component] *
                        results[sample].eigenvectors[component][axis];
                    dot += results[sample].eigenvectors[component][row] *
                        results[sample].eigenvectors[component][axis];
                }
                EXPECT_NEAR(product,
                    results[sample].eigenvalues[axis] * results[sample].eigenvectors[row][axis],
                    tolerance * Real(30));
                EXPECT_NEAR(dot, row == axis ? Real(1) : Real(0), tolerance);
            }
        }
    }
}

}    // namespace detail

TEST(MatrixEigenCuda, FloatEigenpairs) {
    detail::testEigenDeviceParity<float>();
}

TEST(MatrixEigenCuda, DoubleEigenpairs) {
    detail::testEigenDeviceParity<double>();
}
