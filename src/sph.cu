/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distribufted in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cosmictiger/sph_cuda.hpp>

sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream) {
	sph_run_return rc;
	sph_reduction* reduce;
	CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(sph_reduction)));
	reduce->counter = reduce->flag = 0;
	reduce->hmin = std::numeric_limits<float>::max();
	reduce->hmax = 0.0f;
	reduce->flops = 0.0;
	reduce->vsig_max = 0.0;
	reduce->max_rung_grav = 0;
	reduce->max_rung_hydro = 0;
	reduce->max_rung = 0;
	reduce->dtinv_acc = 0.f;
	reduce->dtinv_cond = 0.f;
	reduce->dtinv_diff = 0.f;
	reduce->dtinv_divv = 0.f;
	reduce->dtinv_cfl = 0.f;
	reduce->dtinv_visc = 0.f;
	reduce->dtinv_omega = 0.f;
	static int prehydro_nblocks;
	static int aux_nblocks;
	static int hydro_nblocks;
	static int rungs_nblocks;
	static int cond_init_nblocks;
	static int conduction_nblocks;
	static bool first = true;
	timer tm;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&aux_nblocks, (const void*) sph_cuda_aux, AUX_BLOCK_SIZE, 0));
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&prehydro_nblocks, (const void*) sph_cuda_prehydro, PREHYDRO_BLOCK_SIZE, 0));
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&hydro_nblocks, (const void*) sph_cuda_hydro, HYDRO_BLOCK_SIZE, 0));
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&conduction_nblocks, (const void*) sph_cuda_conduction, CONDUCTION_BLOCK_SIZE, 0));
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cond_init_nblocks, (const void*) sph_cuda_cond_init, COND_INIT_BLOCK_SIZE, 0));
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&rungs_nblocks, (const void*) sph_cuda_rungs, RUNGS_BLOCK_SIZE, 0));
		PRINT("%i %i %i %i %i\n", prehydro_nblocks, hydro_nblocks,aux_nblocks,  rungs_nblocks, cond_init_nblocks, conduction_nblocks);
		aux_nblocks *= cuda_smp_count();
		prehydro_nblocks *= cuda_smp_count();
		hydro_nblocks *= cuda_smp_count();
		conduction_nblocks *= cuda_smp_count();
		cond_init_nblocks *= cuda_smp_count();
		rungs_nblocks *= cuda_smp_count();
	}
	tm.start();
	switch (params.run_type) {
	case SPH_RUN_PREHYDRO: {
		sph_cuda_prehydro<<<prehydro_nblocks, PREHYDRO_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		rc.rc = reduce->flag;
		rc.hmin = reduce->hmin;
		rc.hmax = reduce->hmax;
	}
	break;
	case SPH_RUN_AUX: {
		sph_cuda_aux<<<aux_nblocks, AUX_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		rc.max_rung = reduce->max_rung;
		rc.dtinv_divv = reduce->dtinv_divv;
		rc.dtinv_omega = reduce->dtinv_omega;
	}
	break;
	case SPH_RUN_COND_INIT: {
		sph_cuda_cond_init<<<cond_init_nblocks, COND_INIT_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
	}
	break;
	case SPH_RUN_CONDUCTION: {
		sph_cuda_conduction<<<conduction_nblocks, CONDUCTION_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
	}
	break;
	case SPH_RUN_RUNGS: {
		sph_cuda_rungs<<<hydro_nblocks, RUNGS_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		auto gflops = reduce->flops / tm.read() / (1024.0*1024*1024);
		rc.max_vsig = reduce->vsig_max;
		rc.max_rung_grav = reduce->max_rung_grav;
		rc.max_rung_hydro = reduce->max_rung_hydro;
		rc.max_rung = reduce->max_rung;
		rc.rc = reduce->flag;
	}
	break;
	case SPH_RUN_HYDRO: {
		sph_cuda_hydro<<<hydro_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		rc.max_vsig = reduce->vsig_max;
		rc.max_rung_grav = reduce->max_rung_grav;
		rc.max_rung_hydro = reduce->max_rung_hydro;
		rc.max_rung = reduce->max_rung;
		rc.dtinv_acc = reduce->dtinv_acc;
		rc.dtinv_cfl = reduce->dtinv_cfl;
		rc.dtinv_visc = reduce->dtinv_visc;
		rc.dtinv_diff = reduce->dtinv_diff;
		rc.dtinv_cond = reduce->dtinv_cond;
	}
	break;
}
	tm.stop();
	rc.flops = reduce->flops;
	PRINT("GFLOPS = %e\n", reduce->flops / (1024.0 * 1024.0 * 1024.0) / tm.read());
	(cudaFree(reduce));

	return rc;
}
