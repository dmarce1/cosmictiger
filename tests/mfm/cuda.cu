/* CosmicTiger - CUDA MFM parity, support, and neighbor-overflow regression; BSL-1.0. */

#include <stdexcept>
#include <string>

#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/mfm/cuda.hpp>
#include <tests/cuda_test.hpp>
#include <tests/mfm/prehydro_cases.hpp>

Options const& getOptions() {
    static Options const options{};
    return options;
}

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

int cudaSmpCount() {
    int device;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.multiProcessorCount;
}

namespace detail {

class MfmCudaTest : public CudaTest {
protected:
    void SetUp() override {
        CudaTest::SetUp();
        if (HasFatalFailure() || IsSkipped()) {
            return;
        }
        static bool initialized = false;
        if (!initialized) {
            constexpr size_t kPoolBytes = 64 * 1024 * 1024;
            cudaMemoryInitialize(kPoolBytes);
            initialized = true;
        }
    }
};

template <typename Real, int dimensions>
void compareMfmCuda(int count = 37) {
    MfmFixture<Real, dimensions> cpu(count);
    MfmFixture<Real, dimensions> gpu(count);
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CudaStream stream(device);
    auto const cpuResult = cpu.densityCpu();
    auto const gpuResult = mfm::densityAndGeometryCuda(gpu.parameters_, gpu.data_, stream);
    EXPECT_EQ(gpuResult.geometryFailures, cpuResult.geometryFailures);
    EXPECT_EQ(gpuResult.needsAnotherPass, cpuResult.needsAnotherPass);
    if (count > 1) {
        checkMfmReference(gpu);
    }
    double const tolerance = std::is_same_v<Real, float> ? 2e-5 : 2e-12;
    for (int i = 0; i < count; ++i) {
        int const sink = cpu.sink(i);
        EXPECT_EQ(gpu.data_.sinkGeometryValid[sink], cpu.data_.sinkGeometryValid[sink]);
        EXPECT_EQ(gpu.data_.sinkFlags[sink].bits, cpu.data_.sinkFlags[sink].bits);
        requireMfmClose(gpu.data_.sinkDensity[sink], cpu.data_.sinkDensity[sink], tolerance);
        for (int row = 0; row < dimensions; ++row) {
            for (int column = 0; column <= row; ++column) {
                requireMfmClose(gpu.data_.sinkInverseMoment[sink](row, column),
                    cpu.data_.sinkInverseMoment[sink](row, column), tolerance);
            }
        }
    }
    // Equal support bounds exercise the initial density path without depending on
    // reduction-order differences in the stopping iteration of the free solve.
    cpu.parameters_.minimumSmoothingLength = gpu.parameters_.minimumSmoothingLength = Real(0.125);
    cpu.parameters_.maximumSmoothingLength = gpu.parameters_.maximumSmoothingLength = Real(0.125);
    cpu.parameters_.initializeDensity = gpu.parameters_.initializeDensity = true;
    for (int i = 0; i < count; ++i) {
        cpu.data_.sinkFlags[cpu.sink(i)].setConverged(false);
        gpu.data_.sinkFlags[gpu.sink(i)].setConverged(false);
    }
    auto const cpuSmoothing = cpu.smoothingCpu();
    auto const gpuSmoothing = mfm::smoothingLengthsCuda(gpu.parameters_, gpu.data_, stream);
    EXPECT_EQ(cpuSmoothing.needsAnotherPass, gpuSmoothing.needsAnotherPass);
    for (int i = 0; i < count; ++i) {
        int const sink = cpu.sink(i);
        EXPECT_EQ(gpu.data_.sinkGeometryValid[sink], 0);
        EXPECT_EQ(gpu.data_.sinkFlags[sink].bits, cpu.data_.sinkFlags[sink].bits);
        EXPECT_EQ(gpu.data_.sinkSmoothingLength[sink], cpu.data_.sinkSmoothingLength[sink]);
        requireMfmClose(gpu.data_.sinkDensity[sink], cpu.data_.sinkDensity[sink], tolerance);
    }
}

TEST_F(MfmCudaTest, FloatOneDimension) {
    compareMfmCuda<float, 1>();
}
TEST_F(MfmCudaTest, FloatTwoDimensions) {
    compareMfmCuda<float, 2>();
}
TEST_F(MfmCudaTest, FloatThreeDimensions) {
    compareMfmCuda<float, 3>();
}
TEST_F(MfmCudaTest, DoubleOneDimension) {
    compareMfmCuda<double, 1>();
}
TEST_F(MfmCudaTest, DoubleTwoDimensions) {
    compareMfmCuda<double, 2>();
}
TEST_F(MfmCudaTest, DoubleThreeDimensions) {
    compareMfmCuda<double, 3>();
}
TEST_F(MfmCudaTest, SingularNeighborhoodInvalidatesGeometry) {
    compareMfmCuda<double, 3>(1);
}
TEST_F(MfmCudaTest, SharedNeighborOverflowMatchesCpu) {
    compareMfmCuda<double, 3>(1401);
}

}    // namespace detail
