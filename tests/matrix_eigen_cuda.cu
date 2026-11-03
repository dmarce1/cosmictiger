#include <algorithm>
#include <cmath>
#include <limits>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cosmictiger/math/matrix_eigen.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>

namespace detail {

template <class Real>
struct EigenCudaResult {
    Real velocity[3][3]{};
    Real acceleration[3][3]{};
    Real eigenvalues[3]{};
    Real eigenvectors[3][3]{};
    int rank = 0;
    math::MatrixStatus eigenStatus = math::MatrixStatus::NonFinite;
    bool converged = false;
};

template <class Real>
CUDA_EXPORT EigenCudaResult<Real> evaluateEigenAndGradients(
    cullen_and_dehnen::GradientSums<Real> const& sums) {
    EigenCudaResult<Real> result;
    result.rank =
        cullen_and_dehnen::reconstructGradients(sums, result.velocity, result.acceleration);
    math::Matrix<Real, 3> matrix;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            matrix(row, column) = sums.moment[row][column];
        }
    }
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
__global__ void eigenGradientKernel(
    cullen_and_dehnen::GradientSums<Real> const* sums, EigenCudaResult<Real>* results) {
    int const sample = threadIdx.x;
    if (sample < 3) {
        results[sample] = evaluateEigenAndGradients(sums[sample]);
    }
}

template <class Real>
void testEigenAndGradientDeviceParity() {
    int devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
    if (devices == 0) {
        GTEST_SKIP() << "No CUDA device";
    }
    // Three orthogonal oblique directions give rank1/rank2/full SPD cases.
    Real const directions[3][3] = {
        {Real(1), Real(2), Real(-1)}, {Real(2), Real(-1), Real(0)}, {Real(-1), Real(-2), Real(-5)}};
    Real const velocityGradient[3][3] = {{Real(-0.4), Real(0.7), Real(-0.2)},
        {Real(0.3), Real(-0.8), Real(0.5)}, {Real(-0.1), Real(0.2), Real(-0.6)}};
    cullen_and_dehnen::GradientSums<Real> sums[3];
    for (int sample = 0; sample < 3; ++sample) {
        for (int direction = 0; direction <= sample; ++direction) {
            Real velocity[3]{};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    velocity[row] += velocityGradient[row][column] * directions[direction][column];
                }
                for (int column = 0; column < 3; ++column) {
                    sums[sample].moment[row][column] +=
                        directions[direction][row] * directions[direction][column];
                    sums[sample].velocity[row][column] +=
                        velocity[row] * directions[direction][column];
                    sums[sample].acceleration[row][column] +=
                        Real(2) * velocity[row] * directions[direction][column];
                }
            }
        }
    }
    cullen_and_dehnen::GradientSums<Real>* deviceSums = nullptr;
    EigenCudaResult<Real>* deviceResults = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceSums, sizeof(sums)), cudaSuccess);
    cudaError_t const resultAllocation =
        cudaMalloc(&deviceResults, 3 * sizeof(EigenCudaResult<Real>));
    if (resultAllocation != cudaSuccess) {
        cudaFree(deviceSums);
        FAIL() << cudaGetErrorString(resultAllocation);
    }
    cudaError_t status = cudaMemcpy(deviceSums, sums, sizeof(sums), cudaMemcpyHostToDevice);
    EigenCudaResult<Real> results[3];
    if (status == cudaSuccess) {
        eigenGradientKernel<<<1, 3>>>(deviceSums, deviceResults);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(results, deviceResults, sizeof(results), cudaMemcpyDeviceToHost);
    }
    EXPECT_EQ(cudaFree(deviceResults), cudaSuccess);
    EXPECT_EQ(cudaFree(deviceSums), cudaSuccess);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    Real const tolerance = Real(1024) * std::numeric_limits<Real>::epsilon();
    for (int sample = 0; sample < 3; ++sample) {
        SCOPED_TRACE(sample + 1);
        auto const expected = evaluateEigenAndGradients(sums[sample]);
        EXPECT_EQ(expected.rank, sample + 1);
        EXPECT_EQ(results[sample].rank, sample + 1);
        EXPECT_EQ(results[sample].eigenStatus, math::MatrixStatus::Success);
        EXPECT_TRUE(results[sample].converged);
        for (int row = 0; row < 3; ++row) {
            EXPECT_NEAR(
                results[sample].eigenvalues[row], expected.eigenvalues[row], tolerance * Real(30));
            for (int column = 0; column < 3; ++column) {
                EXPECT_NEAR(results[sample].velocity[row][column], expected.velocity[row][column],
                    tolerance);
                EXPECT_NEAR(results[sample].acceleration[row][column],
                    expected.acceleration[row][column], tolerance);
            }
            // Independently check the recovered affine velocity in every
            // resolved direction; this is more than CPU/device agreement.
            for (int direction = 0; direction <= sample; ++direction) {
                Real projected = Real(0);
                Real affine = Real(0);
                for (int column = 0; column < 3; ++column) {
                    projected +=
                        results[sample].velocity[row][column] * directions[direction][column];
                    affine += velocityGradient[row][column] * directions[direction][column];
                }
                EXPECT_NEAR(projected, affine, tolerance * Real(10));
            }
            // Eigenvectors can change sign (or basis within the repeated zero
            // eigenspace). Verify the device eigenproblem and orthogonality.
            for (int axis = 0; axis < 3; ++axis) {
                Real product = Real(0);
                Real dot = Real(0);
                for (int component = 0; component < 3; ++component) {
                    product += sums[sample].moment[row][component] *
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

TEST(MatrixEigenCuda, FloatGradientsAndEigenpairs) {
    detail::testEigenAndGradientDeviceParity<float>();
}

TEST(MatrixEigenCuda, DoubleGradientsAndEigenpairs) {
    detail::testEigenAndGradientDeviceParity<double>();
}
