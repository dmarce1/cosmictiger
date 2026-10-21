/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: BSL-1.0
 */

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/analysis/power_cuda.hpp>
#include "cuda_test.hpp"

#include "power_test_helpers.hpp"

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

namespace detail {

void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <class Value>
class CudaBuffer {
    Value* data_ = nullptr;

public:
    explicit CudaBuffer(size_t count) {
        checkCuda(cudaMallocManaged(&data_, std::max(size_t(1), count) * sizeof(Value)),
            "allocate density test buffer");
    }

    ~CudaBuffer() {
        cudaFree(data_);
    }

    CudaBuffer(CudaBuffer const&) = delete;
    CudaBuffer& operator=(CudaBuffer const&) = delete;

    Value* data() const {
        return data_;
    }
};

void testDensity(size_t count) {
    constexpr int kGridSize = 8;
    PowerFixture const fixture(count);
    CudaBuffer<Fixed32> positionX(count);
    CudaBuffer<Fixed32> positionY(count);
    CudaBuffer<Fixed32> positionZ(count);
    CudaBuffer<float> masses(count);
    std::copy(fixture.positions[XDIM].begin(), fixture.positions[XDIM].end(), positionX.data());
    std::copy(fixture.positions[YDIM].begin(), fixture.positions[YDIM].end(), positionY.data());
    std::copy(fixture.positions[ZDIM].begin(), fixture.positions[ZDIM].end(), positionZ.data());
    std::copy(fixture.masses.begin(), fixture.masses.end(), masses.data());
    for (bool weighted : {false, true}) {
        for (int fold : {1, 3}) {
            auto const reference = referenceDensity(fixture, fold, kGridSize, weighted);
            for (int tileSize : {3, kGridSize}) {
                auto const gpuDensity = assignTiledDensity(
                    kGridSize, tileSize, [&](Range<int64_t> const& box, float* output) {
                        size_t const cells = box.volume();
                        CudaBuffer<float> density(cells + 2);
                        constexpr float kGuard = 123456.0f;
                        std::fill_n(density.data(), cells + 2, 0.0f);
                        density.data()[0] = density.data()[cells + 1] = kGuard;
                        // Cap the entire launch at 4 * 256 = 1024 threads,
                        // including while War Thunder is on.
                        accumulateDensityKernel<<<4, 256>>>(positionX.data(), positionY.data(),
                            positionZ.data(), density.data() + 1,
                            weighted ? masses.data() : nullptr, fold, count, kGridSize, box);
                        checkCuda(cudaGetLastError(), "launch production density kernel");
                        checkCuda(cudaDeviceSynchronize(), "complete density kernel");
                        require(density.data()[0] == kGuard && density.data()[cells + 1] == kGuard,
                            "CUDA density assignment preserves guards around padded tiles");
                        std::copy_n(density.data() + 1, cells, output);
                    });
                auto const cpuDensity = assignTiledDensity(
                    kGridSize, tileSize, [&](Range<int64_t> const& box, float* output) {
                        accumulateDensityCpuKernel(fixture.positions[XDIM].data(),
                            fixture.positions[YDIM].data(), fixture.positions[ZDIM].data(),
                            weighted ? fixture.masses.data() : nullptr, count, fold, kGridSize, box,
                            output);
                    });
                checkDensity(gpuDensity, reference, totalMass(fixture, weighted));
                checkDensity(gpuDensity, cpuDensity, totalMass(fixture, weighted));
            }
        }
    }
}

}    // namespace detail

using PowerCuda = detail::CudaTest;

TEST_F(PowerCuda, EmptyDensity) {
    detail::testDensity(0);
}

TEST_F(PowerCuda, PartialBlockDensity) {
    detail::testDensity(79);
}

TEST_F(PowerCuda, MultipleBlockDensity) {
    detail::testDensity(1031);
}
