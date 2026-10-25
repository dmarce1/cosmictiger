/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cuda_runtime.h>

#define COSMICTIGER_CARTESIAN_BENCHMARK_CUDA
#include "cartesian_kernels.cpp"

namespace detail {

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <BenchmarkOperator operation, bool doPhi, class Real>
__global__ void benchmarkKernel(
    BenchmarkInput<Real> const* input, BenchmarkResult<operation, Real>* output, int count) {
    int const stride = gridDim.x * blockDim.x;
    for (int sample = blockIdx.x * blockDim.x + threadIdx.x; sample < count; sample += stride) {
        evaluateBenchmark<operation, doPhi>(input[sample], output[sample]);
    }
}

template <BenchmarkOperator operation, bool doPhi, class Real>
void runCudaCase(BenchmarkInput<Real> const* input, char const* operationName, int samples,
    int rounds, int repetitions, int threads) {
    BenchmarkResult<operation, Real>* output = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    checkCuda(cudaMalloc(&output, size_t(samples) * sizeof(*output)));
    try {
        checkCuda(cudaEventCreate(&begin));
        checkCuda(cudaEventCreate(&end));
        // At most 32 one-warp blocks, below half the workstation's 82 SMs.
        int const blocks = std::min(32, (samples + threads - 1) / threads);
        auto const launch = [&]() {
            benchmarkKernel<operation, doPhi><<<blocks, threads>>>(input, output, samples);
            checkCuda(cudaGetLastError());
        };
        launch();
        launch();
        checkCuda(cudaDeviceSynchronize());
        std::vector<double> times;
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            checkCuda(cudaEventRecord(begin));
            for (int round = 0; round < rounds; ++round) {
                launch();
            }
            checkCuda(cudaEventRecord(end));
            checkCuda(cudaEventSynchronize(end));
            float elapsed = 0;
            checkCuda(cudaEventElapsedTime(&elapsed, begin, end));
            times.push_back(double(elapsed) * 1e-3);
        }
        std::vector<BenchmarkResult<operation, Real>> result(samples);
        checkCuda(cudaMemcpy(
            result.data(), output, size_t(samples) * sizeof(*output), cudaMemcpyDeviceToHost));
        std::sort(times.begin(), times.end());
        double const seconds = times[times.size() / 2];
        std::printf("%s,cuda,float,%s,%d,1,%d,%d,%d,%.9g,%.9g,%.17g\n", kImplementation,
            operationName, int(doPhi), samples, rounds, repetitions, seconds,
            seconds * 1e9 / (double(samples) * rounds), checksum<operation, Real>(result));
    } catch (...) {
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
        cudaFree(output);
        throw;
    }
    checkCuda(cudaEventDestroy(end));
    checkCuda(cudaEventDestroy(begin));
    checkCuda(cudaFree(output));
}

void runCuda(int samples, int rounds, int repetitions, int threads) {
    std::vector<BenchmarkInput<float>> input;
    input.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        input.push_back(makeBenchmarkInput<float>(sample));
    }
    BenchmarkInput<float>* deviceInput = nullptr;
    checkCuda(cudaMalloc(&deviceInput, input.size() * sizeof(input[0])));
    try {
        checkCuda(cudaMemcpy(
            deviceInput, input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice));
        runCudaCase<BenchmarkOperator::P2m, false>(
            deviceInput, "p2m", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::M2m, false>(
            deviceInput, "m2m", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::CcDirect, false>(
            deviceInput, "cc_direct", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::CcDirect, true>(
            deviceInput, "cc_direct", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::PcDirect, false>(
            deviceInput, "pc_direct", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::PcDirect, true>(
            deviceInput, "pc_direct", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::L2l, false>(
            deviceInput, "l2l", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::L2l, true>(
            deviceInput, "l2l", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::L2p, false>(
            deviceInput, "l2p", samples, rounds, repetitions, threads);
        runCudaCase<BenchmarkOperator::L2p, true>(
            deviceInput, "l2p", samples, rounds, repetitions, threads);
    } catch (...) {
        cudaFree(deviceInput);
        throw;
    }
    checkCuda(cudaFree(deviceInput));
}

}    // namespace detail

int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            throw std::invalid_argument(
                "Usage: cartesian_kernels_cuda [samples=32768] [rounds=100] "
                "[repetitions=7]");
        }
        int const samples =
            argc > 1 ? detail::positiveArgument(argv[1], "samples", 10000000) : 32768;
        int const rounds = argc > 2 ? detail::positiveArgument(argv[2], "rounds", 1000000) : 100;
        int const repetitions =
            argc > 3 ? detail::positiveArgument(argv[3], "repetitions", 1000) : 7;
        constexpr int kThreads = 32;
        cudaDeviceProp properties{};
        detail::checkCuda(cudaGetDeviceProperties(&properties, 0));
        std::fprintf(stderr,
            "CUDA device: %s; block size: %d. L2L uses one independent expansion "
            "per thread, not the production cooperative warp path.\n",
            properties.name, kThreads);
        std::puts("implementation,device,type,operator,do_phi,lanes,samples,rounds,repetitions,"
                  "median_seconds,ns_per_interaction,checksum");
        detail::runCuda(samples, rounds, repetitions, kThreads);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "Cartesian CUDA benchmark failed: %s\n", error.what());
        return 1;
    }
}
