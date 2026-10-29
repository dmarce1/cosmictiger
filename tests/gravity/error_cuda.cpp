/* CosmicTiger - sampled gravity CUDA batching regressions; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/error_cuda.hpp>
#include <cosmictiger/gravity/reference_table.hpp>

namespace detail {

void checkGravityErrorTestCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

GravityErrorPosition gravityErrorTestPosition(double x, double y, double z) {
    return {Position(x), Position(y), Position(z)};
}

std::vector<GravityErrorPosition> gravityErrorTestSinks(size_t count) {
    std::vector<GravityErrorPosition> sinks(count);
    for (size_t sink = 0; sink < count; ++sink) {
        sinks[sink] = gravityErrorTestPosition(double((sink * 13 + 1) % 251) / 251,
            double((sink * 31 + 7) % 241) / 241, double((sink * 19 + 11) % 239) / 239);
    }
    if (!sinks.empty()) {
        sinks.front() = gravityErrorTestPosition(.99, .03, .875);
    }
    return sinks;
}

std::vector<GravityErrorSource> gravityErrorTestSources(size_t count) {
    std::vector<GravityErrorSource> sources(count);
    for (size_t source = 0; source < count; ++source) {
        sources[source].position = gravityErrorTestPosition(double((source * 17 + 3) % 251) / 251,
            double((source * 41 + 13) % 241) / 241, double((source * 23 + 5) % 239) / 239);
        sources[source].mass = source % 11 == 0 ? 0 : double(1 + source % 7) / (4 * count);
    }
    if (count > 0) {
        // A distinct coincident source must contribute exactly zero, even
        // though this interface carries positions and masses rather than IDs.
        sources[0] = {gravityErrorTestPosition(.99, .03, .875), 1.0 / count};
    }
    if (count > 1) {
        sources[1].position = gravityErrorTestPosition(.01, .98, .125);
    }
    if (count > 2) {
        sources[2].position = gravityErrorTestPosition(.995, .032, .872);
    }
    return sources;
}

using GravityErrorLoadCalls = std::vector<std::pair<size_t, size_t>>;

std::vector<GravityErrorForce> evaluateGravityErrorCuda(
    std::vector<GravityErrorPosition> const& sinks, std::vector<GravityErrorSource> const& sources,
    radial_kernel::KernelTable const& kernel, double supportRadius,
    GravityErrorLoadCalls* calls = nullptr) {
    return gravityErrorDirectCuda(
        sinks, sources.size(),
        [&](size_t begin, std::span<GravityErrorSource> destination) {
            if (begin > sources.size() || destination.size() > sources.size() - begin) {
                throw std::out_of_range("Gravity error source loader received an invalid range");
            }
            if (calls != nullptr) {
                calls->emplace_back(begin, destination.size());
            }
            std::copy_n(sources.data() + begin, destination.size(), destination.data());
        },
        kernel, supportRadius, gravityReferenceCorrectionTable());
}

std::vector<GravityErrorForce> evaluateGravityErrorCpu(
    std::vector<GravityErrorPosition> const& sinks, std::vector<GravityErrorSource> const& sources,
    radial_kernel::KernelTable const& kernel, double supportRadius) {
    auto const& corrections = gravityReferenceCorrectionTable();
    std::vector<GravityErrorForce> result(sinks.size());
    for (size_t sink = 0; sink < sinks.size(); ++sink) {
        std::array<GravityReferenceSum, 3> sum;
        for (auto const& source : sources) {
            GravityErrorForce separation;
            for (int dim = 0; dim < 3; ++dim) {
                separation[dim] = doubleDistance(sinks[sink][dim], source.position[dim]);
            }
            auto const force =
                tabulatedGravityReference(separation, kernel, supportRadius, corrections.data());
            for (int dim = 0; dim < 3; ++dim) {
                sum[dim].add(source.mass * force[dim]);
            }
        }
        for (int dim = 0; dim < 3; ++dim) {
            result[sink][dim] = sum[dim].value();
        }
    }
    return result;
}

void compareGravityErrorForces(
    std::vector<GravityErrorForce> const& actual, std::vector<GravityErrorForce> const& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t sink = 0; sink < expected.size(); ++sink) {
        SCOPED_TRACE(sink);
        for (int dim = 0; dim < 3; ++dim) {
            SCOPED_TRACE(dim);
            ASSERT_TRUE(std::isfinite(actual[sink][dim]));
            double const tolerance = 2e-12 * std::max(1.0, std::abs(expected[sink][dim]));
            EXPECT_NEAR(actual[sink][dim], expected[sink][dim], tolerance);
        }
    }
}

auto const gravityErrorTestKernel =
    radial_kernel::makeKernelTable({radial_kernel::Family::Ferrers, 2});

}    // namespace detail

// The production device selector depends on application Options. These
// standalone tests use device zero and exercise CudaStream outside HPX.
void cudaSetDevice() {
    detail::checkGravityErrorTestCuda(::cudaSetDevice(0), "Select gravity error test device");
}

int cudaGetDevice() {
    int device = -1;
    detail::checkGravityErrorTestCuda(::cudaGetDevice(&device), "Read gravity error test device");
    return device;
}

TEST(GravityErrorCuda, EmptyInputsDoNotLoadSources) {
    auto const sinks = detail::gravityErrorTestSinks(7);
    size_t calls = 0;
    auto const loader = [&](size_t, std::span<detail::GravityErrorSource>) { ++calls; };
    auto const empty =
        detail::gravityErrorDirectCuda({}, 100, loader, detail::gravityErrorTestKernel, .1, {});
    EXPECT_TRUE(empty.empty());
    auto const zero =
        detail::gravityErrorDirectCuda(sinks, 0, loader, detail::gravityErrorTestKernel, .1, {});
    EXPECT_EQ(zero, std::vector<detail::GravityErrorForce>(sinks.size()));
    EXPECT_EQ(calls, 0);
}

TEST(GravityErrorCuda, PartialBlocksMultipleTilesAndUnequalMasses) {
    for (size_t const sourceCount :
        {size_t(1), size_t(127), size_t(129), size_t(1025), size_t(2057)}) {
        for (size_t const sinkCount : {size_t(1), size_t(7)}) {
            SCOPED_TRACE(sourceCount);
            SCOPED_TRACE(sinkCount);
            auto const sources = detail::gravityErrorTestSources(sourceCount);
            auto const sinks = detail::gravityErrorTestSinks(sinkCount);
            auto const actual = detail::evaluateGravityErrorCuda(
                sinks, sources, detail::gravityErrorTestKernel, .1);
            auto const expected =
                detail::evaluateGravityErrorCpu(sinks, sources, detail::gravityErrorTestKernel, .1);
            detail::compareGravityErrorForces(actual, expected);
        }
    }
}

TEST(GravityErrorCuda, ExplicitKernelCopiesAndDeterministicReduction) {
    auto const sources = detail::gravityErrorTestSources(2057);
    auto const sinks = detail::gravityErrorTestSinks(7);
    for (auto const spec : {radial_kernel::KernelSpec{radial_kernel::Family::Ferrers, 2},
             {radial_kernel::Family::Spline, 3}, {radial_kernel::Family::Wendland, 5}}) {
        SCOPED_TRACE(static_cast<int>(spec.family));
        auto const kernel = radial_kernel::makeKernelTable(spec);
        auto const actual = detail::evaluateGravityErrorCuda(sinks, sources, kernel, .125);
        auto const expected = detail::evaluateGravityErrorCpu(sinks, sources, kernel, .125);
        detail::compareGravityErrorForces(actual, expected);
        EXPECT_EQ(actual, detail::evaluateGravityErrorCuda(sinks, sources, kernel, .125));
    }
}

TEST(GravityErrorCuda, BoundedSourceStagingCrossesChunkBoundary) {
    auto const sources = detail::gravityErrorTestSources(65537);
    auto const sinks = detail::gravityErrorTestSinks(7);
    detail::GravityErrorLoadCalls calls;
    auto const actual = detail::evaluateGravityErrorCuda(
        sinks, sources, detail::gravityErrorTestKernel, .1, &calls);
    auto const expected =
        detail::evaluateGravityErrorCpu(sinks, sources, detail::gravityErrorTestKernel, .1);
    detail::compareGravityErrorForces(actual, expected);
    detail::GravityErrorLoadCalls const expectedCalls{{0, 65536}, {65536, 1}};
    EXPECT_EQ(calls, expectedCalls);
}

TEST(GravityErrorCuda, BoundedSinkBatchesReuseLoadedSources) {
    auto const sources = detail::gravityErrorTestSources(129);
    auto const sinks = detail::gravityErrorTestSinks(129);
    detail::GravityErrorLoadCalls calls;
    auto const actual = detail::evaluateGravityErrorCuda(
        sinks, sources, detail::gravityErrorTestKernel, .1, &calls);
    auto const expected =
        detail::evaluateGravityErrorCpu(sinks, sources, detail::gravityErrorTestKernel, .1);
    detail::compareGravityErrorForces(actual, expected);
    detail::GravityErrorLoadCalls const expectedCalls{{0, 129}};
    EXPECT_EQ(calls, expectedCalls);
}

TEST(GravityErrorCuda, FixedPositionLeastSignificantBitAndPeriodicWrap) {
    Position middle(.5);
    Position adjacent = middle;
    adjacent.raw() += 1;
    if constexpr (kPositionBits == 64) {
        // An accidental conversion of both coordinates to double before
        // subtraction would turn this finite softened interaction into zero.
        ASSERT_EQ(middle.toDouble(), adjacent.toDouble());
    }
    Position zero(0);
    Position last;
    last.raw() = std::numeric_limits<PositionInteger>::max();
    for (auto const coordinates : {std::array<Position, 2>{middle, adjacent}, {zero, last}}) {
        detail::GravityErrorPosition const sink{coordinates[0], middle, middle};
        detail::GravityErrorSource const source{{coordinates[1], middle, middle}, 3.7};
        std::vector<detail::GravityErrorPosition> const sinks{sink};
        std::vector<detail::GravityErrorSource> const sources{source};
        auto const actual =
            detail::evaluateGravityErrorCuda(sinks, sources, detail::gravityErrorTestKernel, 1e-6);
        auto const expected =
            detail::evaluateGravityErrorCpu(sinks, sources, detail::gravityErrorTestKernel, 1e-6);
        ASSERT_GT(std::abs(expected[0][0]), .1);
        detail::compareGravityErrorForces(actual, expected);
    }
}

TEST(GravityErrorCuda, InvalidArgumentsAndThrowingLoaderLeaveCudaUsable) {
    auto const sinks = detail::gravityErrorTestSinks(1);
    auto const sources = detail::gravityErrorTestSources(3);
    auto const& corrections = detail::gravityReferenceCorrectionTable();
    auto const unusedLoader = [](size_t, std::span<detail::GravityErrorSource>) {
        FAIL() << "Invalid arguments must be rejected before loading sources";
    };
    for (double const support : {0.0, -.1, std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(detail::gravityErrorDirectCuda(sinks, 1, unusedLoader,
                         detail::gravityErrorTestKernel, support, corrections),
            std::invalid_argument);
    }
    EXPECT_THROW(detail::gravityErrorDirectCuda(sinks, 1, unusedLoader,
                     detail::gravityErrorTestKernel, .1, std::span(corrections).first(1)),
        std::invalid_argument);
    auto const before =
        detail::evaluateGravityErrorCuda(sinks, sources, detail::gravityErrorTestKernel, .1);
    for (size_t const failureBegin : {size_t(0), size_t(65536)}) {
        size_t calls = 0;
        // The first case fails with initial kernel/table transfers submitted;
        // the second fails after a completed source chunk and kernel launch.
        EXPECT_THROW(detail::gravityErrorDirectCuda(
                         sinks, 65537,
                         [&](size_t begin, std::span<detail::GravityErrorSource> destination) {
                             ++calls;
                             if (begin == failureBegin) {
                                 throw std::runtime_error("Injected gravity source loader failure");
                             }
                             std::fill(destination.begin(), destination.end(), sources.front());
                         },
                         detail::gravityErrorTestKernel, .1, corrections),
            std::runtime_error);
        EXPECT_EQ(calls, failureBegin == 0 ? 1 : 2);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(before,
            detail::evaluateGravityErrorCuda(sinks, sources, detail::gravityErrorTestKernel, .1));
    }
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    if (GTEST_FLAG_GET(list_tests)) {
        return RUN_ALL_TESTS();
    }
    int deviceCount = 0;
    auto const status = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && deviceCount == 0)) {
        std::fprintf(stderr, "No usable CUDA device: %s\n", cudaGetErrorString(status));
        return 77;
    }
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA device discovery failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    return RUN_ALL_TESTS();
}
