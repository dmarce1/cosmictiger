/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 This program is free software; you can redistribute it and/or modify it under
 the terms of the GNU General Public License, version 2 or any later version.
 This program is distributed without any warranty. See COPYING for details.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/kernel.hpp>

namespace detail {

constexpr float kKernelProbeRadius = 0.25f;

struct KernelSelectionProbe {
    int family;
    int degree;
    int pieceCount;
    double supportPerSmoothing;
    double centralPotential;
    float density;
    float force;
    float potential;
};

__device__ void probeKernelTable(
    radial_kernel::KernelTable const& table, KernelSelectionProbe& result) {
    result.family = static_cast<int>(table.spec.family);
    result.degree = table.spec.degree;
    result.pieceCount = table.pieceCount;
    result.supportPerSmoothing = table.supportPerSmoothing;
    result.centralPotential = table.centralPotential;
    result.density = radial_kernel::value(table, kKernelProbeRadius);
    result.force = radial_kernel::forceFactor(table, kKernelProbeRadius);
    result.potential = radial_kernel::potential(table, kKernelProbeRadius);
}

__global__ void probeSelectedKernelTables(KernelSelectionProbe* results) {
    probeKernelTable(selectedGravityKernel(), results[0]);
    probeKernelTable(selectedHydroKernel(), results[1]);
}

void checkKernelProbeCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void verifyKernelProbe(char const* label, KernelSelectionProbe const& actual,
    radial_kernel::KernelTable const& expected) {
    if (actual.family != static_cast<int>(expected.spec.family) ||
        actual.degree != expected.spec.degree || actual.pieceCount != expected.pieceCount ||
        actual.supportPerSmoothing != expected.supportPerSmoothing ||
        actual.centralPotential != expected.centralPotential) {
        throw std::runtime_error(
            std::string(label) + " startup CUDA kernel table differs from the selected host table");
    }
    auto const close = [&](float value, float reference) {
        if (!std::isfinite(value) ||
            std::abs(value - reference) > 8.0e-6f * std::max(1.0f, std::abs(reference))) {
            throw std::runtime_error(std::string(label) +
                " startup CUDA kernel evaluation differs from the selected host table");
        }
    };
    close(actual.density, radial_kernel::value(expected, kKernelProbeRadius));
    close(actual.force, radial_kernel::forceFactor(expected, kKernelProbeRadius));
    close(actual.potential, radial_kernel::potential(expected, kKernelProbeRadius));
}

}    // namespace detail

// Call before a regression changes options: a later setOptions would otherwise
// hide CUDA startup accidentally clearing the already uploaded constant tables.
void verifySelectedKernelsCuda() {
    cudaSetDevice();
    detail::KernelSelectionProbe results[2]{};
    detail::KernelSelectionProbe* deviceResults = nullptr;
    detail::checkKernelProbeCuda(
        cudaMalloc(&deviceResults, sizeof(results)), "Allocate startup kernel probe");
    try {
        detail::probeSelectedKernelTables<<<1, 1>>>(deviceResults);
        detail::checkKernelProbeCuda(cudaGetLastError(), "Launch startup kernel probe");
        detail::checkKernelProbeCuda(
            cudaMemcpy(results, deviceResults, sizeof(results), cudaMemcpyDeviceToHost),
            "Read startup kernel probe");
    } catch (...) {
        cudaFree(deviceResults);
        throw;
    }
    detail::checkKernelProbeCuda(cudaFree(deviceResults), "Free startup kernel probe");
    detail::verifyKernelProbe("Gravity", results[0], selectedGravityKernel());
    detail::verifyKernelProbe("Hydro", results[1], selectedHydroKernel());
}
