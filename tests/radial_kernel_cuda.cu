#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

#include <cosmictiger/math/radial_density_kernel.hpp>
#include <cosmictiger/math/radial_kernel.hpp>

namespace detail {

using radial_kernel::Family;
using radial_kernel::KernelSpec;
using radial_kernel::KernelTable;

constexpr std::array<KernelSpec, 15> kSpecifications = {
    {{Family::Spline, 2}, {Family::Spline, 3}, {Family::Spline, 4}, {Family::Spline, 5},
        {Family::Spline, 6}, {Family::Spline, 7}, {Family::Spline, 8}, {Family::Spline, 9},
        {Family::Wendland, 2}, {Family::Wendland, 5}, {Family::Wendland, 8}, {Family::Ferrers, 2},
        {Family::Ferrers, 4}, {Family::Ferrers, 6}, {Family::Ferrers, 8}}};
constexpr int kSamplesPerKernel = 128;
constexpr int kKernelCount = static_cast<int>(kSpecifications.size());
constexpr int kSampleCount = kKernelCount * kSamplesPerKernel;

struct Result {
    float value;
    float derivative;
    float force;
    float potential;
    double doubleValue;
    double doubleDerivative;
    double doubleForce;
    double doublePotential;
};

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void close(double actual, double expected, double tolerance, char const* label) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        std::fprintf(stderr, "%s: %.17g != %.17g (difference %.17g)\n", label, actual, expected,
            actual - expected);
        throw std::runtime_error(label);
    }
}

__global__ void evaluate(KernelTable const* tables, float const* radii, Result* results) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kSampleCount) {
        return;
    }
    auto const& table = tables[index / kSamplesPerKernel];
    float const q = radii[index];
    auto& result = results[index];
    result.value = radial_kernel::value(table, q);
    result.derivative = radial_kernel::derivative(table, q);
    result.force = radial_kernel::forceFactor(table, q);
    result.potential = radial_kernel::potential(table, q);
    result.doubleValue = radial_kernel::value(table, double(q));
    result.doubleDerivative = radial_kernel::derivative(table, double(q));
    result.doubleForce = radial_kernel::forceFactor(table, double(q));
    result.doublePotential = radial_kernel::potential(table, double(q));
}

void run() {
    std::array<KernelTable, kKernelCount> tables{};
    std::array<float, kSampleCount> radii{};
    std::array<Result, kSampleCount> results{};
    for (int kernel = 0; kernel < kKernelCount; ++kernel) {
        tables[kernel] = radial_kernel::makeKernelTable(kSpecifications[kernel]);
        int const base = kernel * kSamplesPerKernel;
        for (int sample = 0; sample < kSamplesPerKernel; ++sample) {
            radii[base + sample] = static_cast<float>(sample) / 100;
        }
        radii[base + 1] = 1e-30f;
        radii[base + 2] = 1e-10f;
        radii[base + 3] = 1e30f;
        for (int piece = 1; piece <= tables[kernel].pieceCount; ++piece) {
            float const join = static_cast<float>(tables[kernel].changePoints[piece]);
            radii[base + 4 + 3 * piece] = std::nextafter(join, 0.0f);
            radii[base + 5 + 3 * piece] = join;
            radii[base + 6 + 3 * piece] = std::nextafter(join, 2.0f);
        }
    }
    KernelTable* deviceTables = nullptr;
    float* deviceRadii = nullptr;
    Result* deviceResults = nullptr;
    check(cudaMalloc(&deviceTables, sizeof(tables)));
    try {
        check(cudaMalloc(&deviceRadii, sizeof(radii)));
        check(cudaMalloc(&deviceResults, sizeof(results)));
        check(cudaMemcpy(deviceTables, tables.data(), sizeof(tables), cudaMemcpyHostToDevice));
        check(cudaMemcpy(deviceRadii, radii.data(), sizeof(radii), cudaMemcpyHostToDevice));
        evaluate<<<(kSampleCount + 127) / 128, 128>>>(deviceTables, deviceRadii, deviceResults);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(results.data(), deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
        for (int index = 0; index < kSampleCount; ++index) {
            auto const& table = tables[index / kSamplesPerKernel];
            double const q = radii[index];
            auto const& result = results[index];
            close(result.value, radial_kernel::value(table, q), 8e-6, "CUDA float density");
            close(result.derivative, radial_kernel::derivative(table, q), 8e-6,
                "CUDA float derivative");
            close(result.force, radial_kernel::forceFactor(table, q), 8e-6, "CUDA float force");
            close(
                result.potential, radial_kernel::potential(table, q), 8e-6, "CUDA float potential");
            close(result.doubleValue, radial_kernel::value(table, q), 8e-14, "CUDA double density");
            close(result.doubleDerivative, radial_kernel::derivative(table, q), 8e-14,
                "CUDA double derivative");
            close(result.doubleForce, radial_kernel::forceFactor(table, q), 8e-14,
                "CUDA double force");
            close(result.doublePotential, radial_kernel::potential(table, q), 8e-14,
                "CUDA double potential");
        }
    } catch (...) {
        cudaFree(deviceResults);
        cudaFree(deviceRadii);
        cudaFree(deviceTables);
        throw;
    }
    check(cudaFree(deviceResults));
    check(cudaFree(deviceRadii));
    check(cudaFree(deviceTables));
}

template <size_t dimension>
__global__ void evaluateDensity(
    radial_kernel::DensityKernelTable<dimension> table, float const* radii, Result* results) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kSamplesPerKernel) {
        return;
    }
    float const q = radii[index];
    results[index].value = radial_kernel::value(table, q);
    results[index].derivative = radial_kernel::derivative(table, q);
    results[index].doubleValue = radial_kernel::value(table, double(q));
    results[index].doubleDerivative = radial_kernel::derivative(table, double(q));
}

template <size_t dimension>
void runDensityDimension() {
    Result* deviceResults = nullptr;
    float* deviceRadii = nullptr;
    std::array<Result, kSamplesPerKernel> results{};
    std::array<float, kSamplesPerKernel> radii{};
    check(cudaMalloc(&deviceResults, sizeof(results)));
    try {
        check(cudaMalloc(&deviceRadii, sizeof(radii)));
        for (auto const family : {Family::Spline, Family::Wendland, Family::Ferrers}) {
            for (int degree = 1; degree <= 9; ++degree) {
                KernelSpec const spec{family, degree};
                if (!radial_kernel::isValidDensity<dimension>(spec)) {
                    continue;
                }
                auto const table = radial_kernel::makeDensityKernelTable<dimension>(spec);
                for (int sample = 0; sample < kSamplesPerKernel; ++sample) {
                    radii[sample] = float(sample) / 100;
                }
                radii[1] = 1e-30f;
                radii[2] = 1e30f;
                for (int piece = 1; piece <= table.pieceCount; ++piece) {
                    float const join = static_cast<float>(table.changePoints[piece]);
                    radii[4 + 3 * piece] = std::nextafter(join, 0.0f);
                    radii[5 + 3 * piece] = join;
                    radii[6 + 3 * piece] = std::nextafter(join, 2.0f);
                }
                // Use identical binary inputs: CUDA fast-math division may
                // otherwise round a separately computed sampling grid differently.
                check(cudaMemcpy(deviceRadii, radii.data(), sizeof(radii), cudaMemcpyHostToDevice));
                evaluateDensity<dimension>
                    <<<1, kSamplesPerKernel>>>(table, deviceRadii, deviceResults);
                check(cudaGetLastError());
                check(cudaDeviceSynchronize());
                check(cudaMemcpy(
                    results.data(), deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
                for (int index = 0; index < kSamplesPerKernel; ++index) {
                    double const q = radii[index];
                    close(results[index].value, radial_kernel::value(table, q), 8e-6,
                        "CUDA D-dimensional float density");
                    close(results[index].derivative, radial_kernel::derivative(table, q), 8e-6,
                        "CUDA D-dimensional float derivative");
                    close(results[index].doubleValue, radial_kernel::value(table, q), 8e-14,
                        "CUDA D-dimensional double density");
                    close(results[index].doubleDerivative, radial_kernel::derivative(table, q),
                        8e-14, "CUDA D-dimensional double derivative");
                }
            }
        }
    } catch (...) {
        cudaFree(deviceRadii);
        cudaFree(deviceResults);
        throw;
    }
    check(cudaFree(deviceRadii));
    check(cudaFree(deviceResults));
}

}    // namespace detail

int main() {
    int deviceCount = 0;
    cudaError_t const status = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && deviceCount == 0)) {
        std::puts("No CUDA device is available; radial kernel device checks skipped.");
        return 77;
    }
    try {
        detail::check(status);
        detail::run();
        detail::runDensityDimension<1>();
        detail::runDensityDimension<2>();
        detail::runDensityDimension<3>();
        detail::runDensityDimension<4>();
        detail::runDensityDimension<7>();
        detail::runDensityDimension<20>();
        std::puts("All 15 radial kernels and arbitrary-dimensional densities passed CUDA "
                  "float/double parity checks.");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "CUDA radial kernel test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
