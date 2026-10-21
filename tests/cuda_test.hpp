/* CosmicTiger - shared CUDA GoogleTest fixture; BSL-1.0. */

#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace detail {

class CudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        int deviceCount = 0;
        cudaError_t const status = cudaGetDeviceCount(&deviceCount);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && deviceCount == 0)) {
            GTEST_SKIP() << "No usable CUDA device: " << cudaGetErrorString(status);
        }
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    }
};

}    // namespace detail
