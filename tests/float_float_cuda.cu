/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <cuda_runtime.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <cosmictiger/float_float.hpp>

namespace detail {

using Reference = boost::multiprecision::cpp_bin_float_quad;

constexpr int kSampleCount = 32;

struct Input {
    float leftHigh;
    float leftLow;
    float rightHigh;
    float rightLow;
    float addendHigh;
    float addendLow;
};

struct Result {
    FloatFloat sum;
    FloatFloat difference;
    FloatFloat product;
    FloatFloat quotient;
    FloatFloat multiplyAdd;
    FloatFloat accumulated;
    FloatFloat normalized;
    FloatFloat converted;
    FloatFloat lowProduct;
    FloatFloat subnormal;
    FloatFloat subnormalSum;
    FloatFloat subnormalProduct;
    FloatFloat subnormalDifference;
    FloatFloat subnormalNegation;
    FloatFloat squareRoot;
    FloatFloat smallSquareRoot;
    FloatFloat largeSquareRoot;
    int comparison;
    int specialValues;
};

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void close(FloatFloat actual, Reference const& expected, char const* label,
    Reference const& absoluteTolerance = 0) {
    Reference const represented = Reference(actual.high()) + Reference(actual.low());
    Reference const tolerance = abs(expected) * ldexp(Reference(12), -48) + absoluteTolerance;
    if (!actual.isFinite() || abs(represented - expected) > tolerance) {
        std::fprintf(stderr, "%s: (%.9g, %.9g), expected %.21Lg\n", label, actual.high(),
            actual.low(), expected.convert_to<long double>());
        throw std::runtime_error(label);
    }
}

__global__ void evaluate(Input const* inputs, Result* results) {
    int const index = threadIdx.x;
    Input const input = inputs[index];
    FloatFloat const left = FloatFloat::fromComponents(input.leftHigh, input.leftLow);
    FloatFloat const right = FloatFloat::fromComponents(input.rightHigh, input.rightLow);
    FloatFloat const addend = FloatFloat::fromComponents(input.addendHigh, input.addendLow);
    Result& result = results[index];
    result.sum = left + right;
    result.difference = left - right;
    result.product = left * right;
    result.quotient = left / right;
    result.multiplyAdd = fma(left, right, addend);
    result.accumulated = FloatFloat(1.0f);
    for (int increment = 0; increment < 1024; ++increment) {
        result.accumulated += FloatFloat(0x1p-30f);
    }
    result.normalized = FloatFloat::fromComponents(input.leftLow, input.leftHigh);
    result.converted = FloatFloat(1.0 + 0x1p-40);
    result.lowProduct = FloatFloat::fromComponents(0x1.5970bep0f, -0x1.1e636p-25f) *
        FloatFloat::fromComponents(0x1.8e7a42p0f, -0x1.c8d56p-25f);
    result.subnormal = FloatFloat(__int_as_float(0x00800000)) / FloatFloat(2.0f);
    result.subnormalSum = FloatFloat(__int_as_float(0x00800000)) + FloatFloat(__int_as_float(1));
    result.subnormalProduct = FloatFloat(__int_as_float(0x00800000)) * FloatFloat(0.5f);
    result.subnormalDifference =
        FloatFloat(__int_as_float(0x00800000)) - FloatFloat(__int_as_float(0x007fffff));
    result.subnormalNegation = -FloatFloat(__int_as_float(1));
    result.squareRoot = sqrt(left < FloatFloat(0) ? -left : left);
    result.smallSquareRoot = sqrt(FloatFloat(__int_as_float(0x00800000)));
    result.largeSquareRoot = sqrt(FloatFloat(__int_as_float(0x7f7fffff)));
    result.comparison = left < right ? -1 : (left > right ? 1 : 0);
    FloatFloat const infinity(__int_as_float(0x7f800000));
    FloatFloat const nan(__int_as_float(0x7fc00000));
    FloatFloat const zero(0.0f);
    FloatFloat const negativeZero(__int_as_float(0x80000000));
    result.specialValues =
        (!infinity.isFinite() && !infinity.isNan() && nan.isNan() && !nan.isFinite() &&
            (infinity - infinity).isNan() && (zero / zero).isNan() && (infinity * zero).isNan() &&
            !(nan < left) && !(nan == left) && nan != left && FloatFloat(1.0f) / infinity == zero &&
            sqrt(infinity) == infinity && sqrt(-infinity).isNan() && sqrt(nan).isNan() &&
            sqrt(FloatFloat(-1)).isNan() && __float_as_uint(sqrt(zero).high()) == 0 &&
            __float_as_uint(sqrt(negativeZero).high()) == 0x80000000U &&
            __float_as_uint(negativeZero.toFloat()) == 0x80000000U &&
            __double_as_longlong(negativeZero.toDouble()) == 0x8000000000000000ULL &&
            __float_as_uint(FloatFloat(__longlong_as_double(0x8000000000000000ULL)).high()) ==
                0x80000000U) ?
        1 :
        0;
}

void run() {
    std::array<Input, kSampleCount> inputs{};
    std::array<Result, kSampleCount> results{};
    for (int index = 0; index < kSampleCount; ++index) {
        int const exponent = index - 16;
        float const sign = (index & 1) ? -1.0f : 1.0f;
        inputs[index] = {std::ldexp(sign * (1.0f + index * 0x1p-5f), exponent),
            std::ldexp(sign * float(index + 1), exponent - 32),
            std::ldexp(1.25f + index * 0x1p-6f, -exponent),
            std::ldexp(-float(index + 3), -exponent - 32),
            std::ldexp(-sign * (1.5f + index * 0x1p-5f), exponent),
            std::ldexp(sign * float(index + 5), exponent - 32)};
    }
    // Include cancellation and inputs differing only in their low components.
    inputs[0] = {1.0f, 0x1p-40f, -1.0f, 0x1p-40f, 1.0f, 0.0f};
    inputs[1] = {1.0f, 0x1p-40f, 1.0f, -0x1p-40f, -1.0f, 0.0f};
    Input* deviceInputs = nullptr;
    Result* deviceResults = nullptr;
    check(cudaMalloc(&deviceInputs, sizeof(inputs)));
    try {
        check(cudaMalloc(&deviceResults, sizeof(results)));
        check(cudaMemcpy(deviceInputs, inputs.data(), sizeof(inputs), cudaMemcpyHostToDevice));
        // A single warp keeps this regression well below the shared GPU resource limit.
        evaluate<<<1, kSampleCount>>>(deviceInputs, deviceResults);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(results.data(), deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
        for (int index = 0; index < kSampleCount; ++index) {
            Input const& input = inputs[index];
            Result const& result = results[index];
            Reference const left = Reference(input.leftHigh) + Reference(input.leftLow);
            Reference const right = Reference(input.rightHigh) + Reference(input.rightLow);
            Reference const addend = Reference(input.addendHigh) + Reference(input.addendLow);
            close(result.sum, left + right, "CUDA addition");
            close(result.difference, left - right, "CUDA subtraction");
            close(result.product, left * right, "CUDA multiplication");
            close(result.quotient, left / right, "CUDA division");
            Reference const inputTolerance =
                (abs(left * right) + abs(addend)) * ldexp(Reference(12), -48);
            close(result.multiplyAdd, left * right + addend, "CUDA multiply-add", inputTolerance);
            if (index == 1 && result.multiplyAdd != FloatFloat(0.0f)) {
                throw std::runtime_error("CUDA multiply-add must preserve separate multiplication "
                                         "and addition rounding");
            }
            close(result.accumulated, Reference(1) + ldexp(Reference(1), -20), "CUDA accumulation");
            close(result.normalized, left, "CUDA normalization");
            close(result.converted, Reference(1) + ldexp(Reference(1), -40), "CUDA conversion");
            Reference const exactProduct = (Reference(0x1.5970bep0f) - Reference(0x1.1e636p-25f)) *
                (Reference(0x1.8e7a42p0f) - Reference(0x1.c8d56p-25f));
            close(result.lowProduct, exactProduct, "CUDA low product");
            if (abs(Reference(result.lowProduct.high()) + Reference(result.lowProduct.low()) -
                    exactProduct) >= ldexp(Reference(1), -47)) {
                throw std::runtime_error(
                    "CUDA multiplication must include the low-times-low contribution");
            }
            close(result.subnormal, ldexp(Reference(1), -127), "CUDA subnormal quotient");
            close(result.subnormalSum, ldexp(Reference(1), -126) + ldexp(Reference(1), -149),
                "CUDA subnormal addition");
            close(result.subnormalProduct, ldexp(Reference(1), -127), "CUDA subnormal product");
            close(
                result.subnormalDifference, ldexp(Reference(1), -149), "CUDA subnormal difference");
            close(result.subnormalNegation, -ldexp(Reference(1), -149), "CUDA subnormal negation");
            close(result.squareRoot, sqrt(abs(left)), "CUDA square root");
            close(result.smallSquareRoot, ldexp(Reference(1), -63), "CUDA small square root");
            close(result.largeSquareRoot, sqrt(Reference(std::numeric_limits<float>::max())),
                "CUDA large square root");
            int const comparison = left < right ? -1 : (left > right ? 1 : 0);
            if (result.comparison != comparison || result.specialValues != 1) {
                throw std::runtime_error("CUDA comparison or special-value behavior");
            }
        }
    } catch (...) {
        cudaFree(deviceResults);
        cudaFree(deviceInputs);
        throw;
    }
    check(cudaFree(deviceResults));
    check(cudaFree(deviceInputs));
}

}    // namespace detail

int main() {
#if defined(__SSE__)
    // Permit the independent host reference to read device subnormal results.
    // This does not change the CUDA kernel's arithmetic mode.
    _mm_setcsr(_mm_getcsr() & ~unsigned(0x8040));
#endif
    int deviceCount = 0;
    cudaError_t const status = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && deviceCount == 0)) {
        std::puts("Skipping FloatFloat CUDA regression: no usable CUDA device");
        return 77;
    }
    try {
        detail::check(status);
        detail::run();
        std::puts("FloatFloat CUDA arithmetic matches the independent multiprecision reference");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "FloatFloat CUDA test failed: %s\n", error.what());
        return 1;
    }
}
