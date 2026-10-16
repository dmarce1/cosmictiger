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

#include <cosmictiger/sph_cuda.hpp>

SphRunReturn sphRunCuda(SphRunParams params, SphRunCudaData data, cudaStream_t stream) {
    SphRunReturn rc;
    SphReduction* reduce;
    CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(SphReduction)));
    reduce->counter = reduce->flag = 0;
    reduce->hmin = std::numeric_limits<float>::max();
    reduce->hmax = 0.0f;
    reduce->flops = 0.0;
    reduce->vsigMax = 0.0;
    reduce->maxRungGrav = 0;
    reduce->maxRungHydro = 0;
    reduce->maxRung = 0;
    reduce->dtinvAcc = 0.f;
    reduce->dtinvCond = 0.f;
    reduce->dtinvDiff = 0.f;
    reduce->dtinvDivv = 0.f;
    reduce->dtinvCfl = 0.f;
    reduce->dtinvVisc = 0.f;
    reduce->dtinvOmega = 0.f;
    static int prehydro1Nblocks;
    static int prehydro2Nblocks;
    static int auxNblocks;
    static int hydroNblocks;
    static int rungsNblocks;
    static int condInitNblocks;
    static int conductionNblocks;
    static bool first = true;
    Timer tm;
    if (first) {
        first = false;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &auxNblocks, (void const*) sphCudaAux, AUX_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &prehydro1Nblocks, (void const*) sphCudaPrehydro1, PREHYDRO1_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &prehydro2Nblocks, (void const*) sphCudaPrehydro2, PREHYDRO2_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &hydroNblocks, (void const*) sphCudaHydro, HYDRO_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &conductionNblocks, (void const*) sphCudaConduction, CONDUCTION_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &condInitNblocks, (void const*) sphCudaCondInit, COND_INIT_BLOCK_SIZE, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &rungsNblocks, (void const*) sphCudaRungs, RUNGS_BLOCK_SIZE, 0));
        PRINT("%i %i %i %i %i %i\n", prehydro1Nblocks, prehydro2Nblocks, hydroNblocks, auxNblocks,
            rungsNblocks, condInitNblocks, conductionNblocks);
        auxNblocks *= cudaSmpCount();
        prehydro1Nblocks *= cudaSmpCount();
        prehydro2Nblocks *= cudaSmpCount();
        hydroNblocks *= cudaSmpCount();
        conductionNblocks *= cudaSmpCount();
        condInitNblocks *= cudaSmpCount();
        rungsNblocks *= cudaSmpCount();
    }
    tm.start();
    switch (params.runType) {
    case SPH_RUN_PREHYDRO1: {
        sphCudaPrehydro1<<<prehydro1Nblocks, PREHYDRO1_BLOCK_SIZE, 0, stream>>>(
            params, data, reduce);
        synchronizeCudaStream(stream);
        rc.rc = reduce->flag;
        rc.hmin = reduce->hmin;
        rc.hmax = reduce->hmax;
    } break;
    case SPH_RUN_PREHYDRO2: {
        sphCudaPrehydro2<<<prehydro2Nblocks, PREHYDRO2_BLOCK_SIZE, 0, stream>>>(
            params, data, reduce);
        synchronizeCudaStream(stream);
        rc.rc = reduce->flag;
        rc.hmin = reduce->hmin;
        rc.hmax = reduce->hmax;
    } break;
    case SPH_RUN_AUX: {
        sphCudaAux<<<auxNblocks, AUX_BLOCK_SIZE, 0, stream>>>(params, data, reduce);
        synchronizeCudaStream(stream);
        rc.maxRung = reduce->maxRung;
        rc.dtinvDivv = reduce->dtinvDivv;
        rc.dtinvOmega = reduce->dtinvOmega;
    } break;
    case SPH_RUN_COND_INIT: {
        sphCudaCondInit<<<condInitNblocks, COND_INIT_BLOCK_SIZE, 0, stream>>>(params, data, reduce);
        synchronizeCudaStream(stream);
    } break;
    case SPH_RUN_CONDUCTION: {
        sphCudaConduction<<<conductionNblocks, CONDUCTION_BLOCK_SIZE, 0, stream>>>(
            params, data, reduce);
        synchronizeCudaStream(stream);
    } break;
    case SPH_RUN_RUNGS: {
        sphCudaRungs<<<hydroNblocks, RUNGS_BLOCK_SIZE, 0, stream>>>(params, data, reduce);
        synchronizeCudaStream(stream);
        auto gflops = reduce->flops / tm.read() / (1024.0 * 1024 * 1024);
        rc.maxVsig = reduce->vsigMax;
        rc.maxRungGrav = reduce->maxRungGrav;
        rc.maxRungHydro = reduce->maxRungHydro;
        rc.maxRung = reduce->maxRung;
        rc.rc = reduce->flag;
    } break;
    case SPH_RUN_HYDRO: {
        sphCudaHydro<<<hydroNblocks, HYDRO_BLOCK_SIZE, 0, stream>>>(params, data, reduce);
        synchronizeCudaStream(stream);
        rc.maxVsig = reduce->vsigMax;
        rc.maxRungGrav = reduce->maxRungGrav;
        rc.maxRungHydro = reduce->maxRungHydro;
        rc.maxRung = reduce->maxRung;
        rc.dtinvAcc = reduce->dtinvAcc;
        rc.dtinvCfl = reduce->dtinvCfl;
        rc.dtinvVisc = reduce->dtinvVisc;
        rc.dtinvDiff = reduce->dtinvDiff;
        rc.dtinvCond = reduce->dtinvCond;
    } break;
    }
    tm.stop();
    rc.flops = reduce->flops;
    PRINT("GFLOPS = %e\n", reduce->flops / (1024.0 * 1024.0 * 1024.0) / tm.read());
    (cudaFree(reduce));

    return rc;
}
