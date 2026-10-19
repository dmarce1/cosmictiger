/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <memory>

#include <cosmictiger/sph_cuda.hpp>

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
    reduction->nextNodeIndex = reduction->pendingChanges = 0;
    reduction->minimumSupportRadius = std::numeric_limits<float>::max();
    reduction->maximumSupportRadius = 0.0f;
    reduction->floatingPointOperations = 0.0;
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
            &smoothingBlockCount, (void const*) sphCudaPrehydro1, PREHYDRO1_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &densityBlockCount, (void const*) sphCudaPrehydro2, PREHYDRO2_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &hydroBlockCount, (void const*) sphCudaHydro, HYDRO_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &rungBlockCount, (void const*) sphCudaRungs, RUNGS_BLOCK_SIZE, 0));
        smoothingBlockCount *= cudaSmpCount();
        densityBlockCount *= cudaSmpCount();
        hydroBlockCount *= cudaSmpCount();
        rungBlockCount *= cudaSmpCount();
    }
    timer.start();
    switch (parameters.runType) {
    case SPH_RUN_PREHYDRO1: {
        sphCudaPrehydro1<<<smoothingBlockCount, PREHYDRO1_BLOCK_SIZE, 0, stream.nativeHandle()>>>(
            parameters, data, reduction);
        stream.checkLaunch();
        stream.wait();
        result.needsAnotherPass = reduction->pendingChanges;
        result.minimumSupportRadius = reduction->minimumSupportRadius;
        result.maximumSupportRadius = reduction->maximumSupportRadius;
    } break;
    case SPH_RUN_PREHYDRO2: {
        sphCudaPrehydro2<<<densityBlockCount, PREHYDRO2_BLOCK_SIZE, 0, stream.nativeHandle()>>>(
            parameters, data, reduction);
        stream.checkLaunch();
        stream.wait();
        result.needsAnotherPass = reduction->pendingChanges;
        result.minimumSupportRadius = reduction->minimumSupportRadius;
        result.maximumSupportRadius = reduction->maximumSupportRadius;
    } break;
    case SPH_RUN_RUNGS: {
        sphCudaRungs<<<rungBlockCount, RUNGS_BLOCK_SIZE, 0, stream.nativeHandle()>>>(
            parameters, data, reduction);
        stream.checkLaunch();
        stream.wait();
        result.maximumSignalSpeed = reduction->maximumSignalSpeed;
        result.maximumGravityRung = reduction->maximumGravityRung;
        result.maximumHydroRung = reduction->maximumHydroRung;
        result.maximumRung = reduction->maximumRung;
        result.needsAnotherPass = reduction->pendingChanges;
    } break;
    case SPH_RUN_HYDRO: {
        sphCudaHydro<<<hydroBlockCount, HYDRO_BLOCK_SIZE, 0, stream.nativeHandle()>>>(
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
    result.floatingPointOperations = reduction->floatingPointOperations;
    PRINT("GFLOPS = %e\n",
        reduction->floatingPointOperations / (1024.0 * 1024.0 * 1024.0) / timer.read());

    return result;
}
