/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <memory>

#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/sph/sph_cuda.hpp>

SphRunReturn sphRunCuda(SphRunParameters parameters, SphRunCudaData data, CudaStream& stream) {
    SphRunReturn result;
    SphReduction* reduction;
    CUDA_CHECK(cudaMallocManaged(&reduction, sizeof(SphReduction)));
    auto const releaseReduction = [&stream](SphReduction* pointer) noexcept {
        // This reduction belongs to the callee while the stream belongs to its
        // caller; finish outstanding work before releasing it on every exit.
        try {
            stream.wait();
        } catch (...) {
        }
        cudaFree(pointer);
    };
    std::unique_ptr<SphReduction, decltype(releaseReduction)> reductionOwner(
        reduction, releaseReduction);
    reduction->minimumTimestep = std::numeric_limits<float>::max();
    reduction->nextNodeIndex = reduction->pendingChanges = 0;
    reduction->minimumSupportRadius = std::numeric_limits<float>::max();
    reduction->maximumSupportRadius = 0.0f;
    reduction->floatingPointOperations = 0;
    reduction->maximumSignalSpeed = 0.0;
    reduction->maximumGravityRung = 0;
    reduction->maximumHydroRung = 0;
    reduction->maximumRung = 0;
    reduction->inverseAccelerationTimestep = 0.f;
    reduction->inverseSoundTimestep = 0.f;
    reduction->inverseViscosityTimestep = 0.f;
    static int smoothingBlockCount;
    static int densityBlockCount;
    static int hydroBlockCount;
    static int rungBlockCount;
    static bool first = true;
    Timer timer;
    if (first) {
        first = false;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &smoothingBlockCount, (void const*) sphCudaPrehydro1, kPrehydro1BlockSize, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &densityBlockCount, (void const*) sphCudaPrehydro2, kPrehydro2BlockSize, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &hydroBlockCount, (void const*) sphCudaHydro, kHydroBlockSize, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &rungBlockCount, (void const*) sphCudaRungs, kRungsBlockSize, 0));
        smoothingBlockCount *= cudaSmpCount();
        densityBlockCount *= cudaSmpCount();
        hydroBlockCount *= cudaSmpCount();
        rungBlockCount *= cudaSmpCount();
    }
    timer.start();
    switch (parameters.runType) {
    case kSphRunPrehydro1: {
        sphCudaPrehydro1<<<smoothingBlockCount, kPrehydro1BlockSize, 0, stream.nativeHandle()>>>(
            parameters, data, reduction);
        stream.checkLaunch();
        stream.wait();
        result.needsAnotherPass = reduction->pendingChanges;
        result.minimumSupportRadius = reduction->minimumSupportRadius;
        result.maximumSupportRadius = reduction->maximumSupportRadius;
    } break;
    case kSphRunPrehydro2: {
        sphCudaPrehydro2<<<densityBlockCount, kPrehydro2BlockSize, 0, stream.nativeHandle()>>>(
            parameters, data, reduction);
        stream.checkLaunch();
        stream.wait();
        result.needsAnotherPass = reduction->pendingChanges;
        result.minimumSupportRadius = reduction->minimumSupportRadius;
        result.maximumSupportRadius = reduction->maximumSupportRadius;
    } break;
    case kSphRunRungs: {
        sphCudaRungs<<<rungBlockCount, kRungsBlockSize, 0, stream.nativeHandle()>>>(
            parameters, data, reduction);
        stream.checkLaunch();
        stream.wait();
        result.maximumSignalSpeed = reduction->maximumSignalSpeed;
        result.maximumGravityRung = reduction->maximumGravityRung;
        result.maximumHydroRung = reduction->maximumHydroRung;
        result.maximumRung = reduction->maximumRung;
        result.needsAnotherPass = reduction->pendingChanges;
    } break;
    case kSphRunHydro: {
        sphCudaHydro<<<hydroBlockCount, kHydroBlockSize, 0, stream.nativeHandle()>>>(
            parameters, data, reduction);
        stream.checkLaunch();
        stream.wait();
        result.maximumSignalSpeed = reduction->maximumSignalSpeed;
        result.maximumGravityRung = reduction->maximumGravityRung;
        result.maximumHydroRung = reduction->maximumHydroRung;
        result.maximumRung = reduction->maximumRung;
        result.inverseAccelerationTimestep = reduction->inverseAccelerationTimestep;
        result.inverseSoundTimestep = reduction->inverseSoundTimestep;
        result.inverseViscosityTimestep = reduction->inverseViscosityTimestep;
    } break;
    }
    timer.stop();
    result.minimumTimestep = reduction->minimumTimestep;
    result.floatingPointOperations = reduction->floatingPointOperations;
    terminal::detailPrint(
        "SPH GFLOP/s = %e\n", double(reduction->floatingPointOperations) * 1.0e-9 / timer.read());

    return result;
}
