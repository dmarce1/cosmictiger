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

__global__ void contributePrecisionFlops(bool divergent, int precision) {
    unsigned const index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    if (!divergent || index % 3 == 1) {
        FlopPrecision const selected =
            FlopPrecision(precision < 0 ? (index / 3) % 3 : unsigned(precision));
        addGpuFlops((FlopCount(1) << 54) + index + 3, selected);
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
            auto const counts = getGpuFlopCounts();
            EXPECT_EQ(counts.singlePrecision, 0);
            EXPECT_EQ(counts.doublePrecision, 0);
            EXPECT_EQ(counts.unclassified, expected);
        }
    }
    resetGpuFlops();
    EXPECT_EQ(getGpuFlops(), 0);
    std::puts("CUDA exact FLOP counts passed: multiple, partial and divergent warps");
}

TEST_F(FlopsCuda, PrecisionCountsForUniformMixedAndDivergentWarps) {
    for (auto const block : {dim3(1), dim3(17), dim3(32), dim3(65), dim3(7, 3, 2)}) {
        for (bool divergent : {false, true}) {
            for (int precision : {-1, 0, 1, 2}) {
                resetGpuFlops();
                contributePrecisionFlops<<<2, block>>>(divergent, precision);
                ASSERT_EQ(cudaGetLastError(), cudaSuccess);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                FlopCounts expected;
                for (unsigned index = 0; index < block.x * block.y * block.z; ++index) {
                    if (!divergent || index % 3 == 1) {
                        FlopCount const count = 2 * ((FlopCount(1) << 54) + index + 3);
                        unsigned const selected =
                            precision < 0 ? (index / 3) % 3 : unsigned(precision);
                        if (selected == 0) {
                            expected.singlePrecision += count;
                        } else if (selected == 1) {
                            expected.doublePrecision += count;
                        } else {
                            expected.unclassified += count;
                        }
                    }
                }
                auto const actual = getGpuFlopCounts();
                EXPECT_EQ(actual.singlePrecision, expected.singlePrecision);
                EXPECT_EQ(actual.doublePrecision, expected.doublePrecision);
                EXPECT_EQ(actual.unclassified, expected.unclassified);
                EXPECT_EQ(getGpuFlops(), expected.total());
            }
        }
    }
    resetGpuFlops();
    auto const counts = getGpuFlopCounts();
    EXPECT_EQ(counts.singlePrecision, 0);
    EXPECT_EQ(counts.doublePrecision, 0);
    EXPECT_EQ(counts.unclassified, 0);
}
