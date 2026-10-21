/* CosmicTiger - CUDA FLOP aggregation regression; BSL-1.0. */

#include <cstdio>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cosmictiger/core/flops.hpp>

bool cudaEnabled() {
    return true;
}

__global__ void contributeFlops(bool divergent) {
    unsigned const index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    if (!divergent || index % 3 == 1) {
        FlopCounter<> count = FlopCount(1) << 54;
        (count += index) += 1;
        addGpuFlops(count);
    }
}

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

using FlopsCuda = detail::CudaTest;

TEST_F(FlopsCuda, ExactCountsForMultiplePartialAndDivergentWarps) {
    // Tiny grids exercise multiple, partial, divergent and multidimensional
    // warps; no performance benchmark or sustained GPU load is needed.
    for (auto const block : {dim3(1), dim3(17), dim3(32), dim3(65), dim3(7, 3, 2)}) {
        for (bool divergent : {false, true}) {
            resetGpuFlops();
            contributeFlops<<<2, block>>>(divergent);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            FlopCount expected = 0;
            for (unsigned index = 0; index < block.x * block.y * block.z; ++index) {
                if (!divergent || index % 3 == 1) {
                    expected += 2 * ((FlopCount(1) << 54) + index + 1);
                }
            }
            ASSERT_EQ(getGpuFlops(), expected);
        }
    }
    resetGpuFlops();
    EXPECT_EQ(getGpuFlops(), 0);
    std::puts("CUDA exact FLOP counts passed: multiple, partial and divergent warps");
}
