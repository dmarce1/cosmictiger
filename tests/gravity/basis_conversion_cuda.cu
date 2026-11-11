/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#ifdef TREE_LOCAL_CUDA_RUN
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>
#endif

#include <cosmictiger/gravity/fmm/tree_local.hpp>

// Compile only: nvcc -dc -std=c++20 -arch=sm_86 -I. \
//   -DORDER=7 -DCOSMICTIGER_DIMENSION_COUNT=3 \
//   tests/gravity/basis_conversion_cuda.cu -o /tmp/basis_conversion_cuda.o
namespace detail {
template <class Real, int order>
__device__ void convert(Real const* input, Real* output) {
    namespace gravity = cosmictiger::gravity;
    gravity::cartesian::Multipole<Real> source;
    gravity::spherical::Expansion<Real> local;
    for (int i = 0; i < source.size(); ++i) {
        source[i] = input[i];
    }
    for (int i = 0; i < local.kCount_; ++i) {
        local[i] = input[i];
    }
    auto const sphericalSource = static_cast<gravity::spherical::Multipole<Real>>(source);
    auto const cartesianLocal = static_cast<gravity::cartesian::Expansion<Real>>(local);
    for (int i = 0; i < sphericalSource.kCount_; ++i) {
        output[i] = sphericalSource[i];
    }
    for (int i = 0; i < cartesianLocal.size(); ++i) {
        output[(order - 1) * (order - 1) + i] = cartesianLocal[i];
    }
    if constexpr (order < ORDER) {
        convert<Real, order + 1>(input, output + 2 * order * order);
    }
}
template <int order, class PositionType, class Real = double>
CUDA_EXPORT void treeLocal(double const* input, double* output, bool doPotential, bool tiny) {
    namespace gravity = cosmictiger::gravity;
    namespace local = gravity::tree_local;
    gravity::cartesian::Expansion<double> initial{}, rootDouble{};
    gravity::cartesian::Expansion<Real> atRoot{}, shifted{};
    Vector3<PositionType> first, second;
    for (int axis = 0; axis < kDimensions; ++axis) {
        first[axis] = input[axis];
        second[axis] = input[axis + 3];
        if (tiny) {
            first[axis].raw() = PositionType::max().raw() - 64;
            second[axis].raw() = first[axis].raw() - 7 - axis;
        }
    }
    local::fromMesh(initial, input, .01, 8);
    auto const displacement = local::displacement(first, second);
    double const curvature = local::backgroundCurvature(.01, 8);
    local::translate(rootDouble, initial, displacement, curvature, doPotential);
    for (int i = 0; i < atRoot.size(); ++i) {
        atRoot[i] = Real(rootDouble[i]);
    }
    shifted[0] = doPotential ? 0 : 73;
    auto const treeOffset = local::displacement<Real>(first, second);
    local::translate(shifted, atRoot, treeOffset, Real(curvature), doPotential);
    gravity::Field<Real> field{};
    field[0] = 37;
    local::evaluate(field, shifted, treeOffset, Real(curvature), doPotential);
    for (int component = 0; component < 4; ++component) {
        output[component] = field[component];
    }
    if constexpr (order < ORDER) {
        treeLocal<order + 1, PositionType, Real>(input, output + 4, doPotential, tiny);
    }
}
}    // namespace detail

__global__ void basisConversionFloat(float const* input, float* output) {
    detail::convert<float, ORDER>(input, output);
}

__global__ void basisConversionDouble(double const* input, double* output) {
    detail::convert<double, ORDER>(input, output);
}

__global__ void treeLocalFixed32(
    double const* input, double* output, bool doPotential, bool tiny, bool single) {
    if (single) {
        detail::treeLocal<ORDER, Fixed32, float>(input, output, doPotential, tiny);
    } else {
        detail::treeLocal<ORDER, Fixed32>(input, output, doPotential, tiny);
    }
}

__global__ void treeLocalFixed64(
    double const* input, double* output, bool doPotential, bool tiny, bool single) {
    if (single) {
        detail::treeLocal<ORDER, Fixed64, float>(input, output, doPotential, tiny);
    } else {
        detail::treeLocal<ORDER, Fixed64>(input, output, doPotential, tiny);
    }
}

#ifdef TREE_LOCAL_CUDA_RUN
// Optional bounded parity executable: compile without -dc and with
// -DTREE_LOCAL_CUDA_RUN. Every launch is one block with one thread, serialized.
// The explicit SM guard keeps this check below a quarter-device compute budget.
void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        throw std::runtime_error("host assertion failed");
    }
}
__device__ void cosmictigerCudaAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        __trap();
    }
}
void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}
int main() {
    try {
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, 0));
        if (properties.multiProcessorCount < 4) {
            throw std::runtime_error("one SM exceeds the quarter-device test budget");
        }
        std::array<double, ORDER * ORDER> input;
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = .1 + .01 * i;
        }
        constexpr size_t kOutputs = 4 * (ORDER - ORDER + 1);
        std::array<double, kOutputs> output{}, expected{};
        double* deviceInput = nullptr;
        double* deviceOutput = nullptr;
        checkCuda(cudaMalloc(&deviceInput, sizeof(input)));
        checkCuda(cudaMalloc(&deviceOutput, sizeof(output)));
        checkCuda(cudaMemcpy(deviceInput, input.data(), sizeof(input), cudaMemcpyHostToDevice));
        double worst = 0;
        size_t checks = 0;
        for (bool single : {false, true}) {
            for (bool wide : {false, true}) {
                for (bool tiny : {false, true}) {
                    for (bool doPotential : {false, true}) {
                        if (wide) {
                            if (single) {
                                detail::treeLocal<ORDER, Fixed64, float>(
                                    input.data(), expected.data(), doPotential, tiny);
                            } else {
                                detail::treeLocal<ORDER, Fixed64>(
                                    input.data(), expected.data(), doPotential, tiny);
                            }
                            treeLocalFixed64<<<1, 1>>>(
                                deviceInput, deviceOutput, doPotential, tiny, single);
                        } else {
                            if (single) {
                                detail::treeLocal<ORDER, Fixed32, float>(
                                    input.data(), expected.data(), doPotential, tiny);
                            } else {
                                detail::treeLocal<ORDER, Fixed32>(
                                    input.data(), expected.data(), doPotential, tiny);
                            }
                            treeLocalFixed32<<<1, 1>>>(
                                deviceInput, deviceOutput, doPotential, tiny, single);
                        }
                        checkCuda(cudaGetLastError());
                        checkCuda(cudaDeviceSynchronize());
                        checkCuda(cudaMemcpy(
                            output.data(), deviceOutput, sizeof(output), cudaMemcpyDeviceToHost));
                        for (size_t i = 0; i < output.size(); ++i) {
                            double const error = std::abs(output[i] - expected[i]) /
                                std::max(1., std::abs(expected[i]));
                            worst = std::max(worst, error);
                            ++checks;
                            if (!std::isfinite(error) || error > (single ? 2e-6 : 3e-11)) {
                                throw std::runtime_error("CPU/CUDA tree-local field mismatch");
                            }
                        }
                    }
                }
            }
        }
        checkCuda(cudaFree(deviceOutput));
        checkCuda(cudaFree(deviceInput));
        std::printf("Tree local CUDA parity q=%d..%d: %zu checks, max error %.3e; %d SMs, "
                    "16 serialized launches of 1 block x 1 thread\n",
            ORDER, ORDER, checks, worst, properties.multiProcessorCount);
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
#endif
