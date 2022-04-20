/*

 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

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

#ifndef SPH_CUDA_HPP_
#define SPH_CUDA_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>


struct sph_run_cuda_data {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* dentr_snk;
	float* entr0_snk;
	float* entr;
	float* kappa;
	float* cold_frac;
	float gsoft;
	float* dalpha;
	array<float, NCHEMFRACS>* chem;
	float* divv;
	float* mmw;
	float* gradT;
	float* shearv;
	float* dcold_mass;
	char* sa_snk;
	char* oldrung_snk;
	float def_gamma;
	bool conduction;
	bool gravity;
	bool chemistry;
	char* converged_snk;
	float gcentral;
	float hcentral;
	float G;
	float t0;
	float kappa0;
	float rho0_c;
	float* alpha;
	float* alpha0;
	float rho0_b;
//	float* Z;
//	float* Y;
	float Y0;
	float* gx;
	float* gy;
	float* gz;
	float* vx;
	float* vy;
	float* vz;
	float* fpre1;
	float* fpre2;
	float* pre;
	float* h;
	char* rungs;
	char* rungs_snk;
	float* fpre1_snk;
	float* fpre2_snk;
	float* pre_snk;
	float* shear_snk;
	sph_record1* rec1_snk;
	sph_record3* rec3_snk;
	sph_record2* rec2_snk;
	part_int* dm_index_snk;
	float* gx_snk;
	float* gy_snk;
	float* gz_snk;
	float* kap_snk;
	sph_record5* rec5_snk;
	sph_record6* rec6_snk;
	float code_dif_to_cgs;
	float N;
	int nselfs;
	sph_tree_node* trees;
	int* selfs;
	int* neighbors;
	float m;
	float eta;
};


#define SPH_DIFFUSION_C 0.03f
#define COND_INIT_BLOCK_SIZE 32
#define CONDUCTION_BLOCK_SIZE 32
#define RUNGS_BLOCK_SIZE 256
#define AUX_BLOCK_SIZE 96
#define MAX_RUNG_DIF 1
#define SPH_SMOOTHLEN_TOLER float(5.0e-5)
#define SMOOTHLEN_BLOCK_SIZE 160
#define PREHYDRO_BLOCK_SIZE 64
#define HYDRO_BLOCK_SIZE 32

struct sph_reduction {
	int counter;
	int flag;
	float hmin;
	float hmax;
	float vsig_max;
	double flops;
	int max_rung_hydro;
	int max_rung_grav;
	int max_rung;
};

sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream);
__global__ void sph_cuda_smoothlen(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_prehydro(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_aux(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_cond_init(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_hydro(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_conduction(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_rungs(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);



static __constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6),
		1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
				/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24),
		1.0 / (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };


#endif /* SPH_CUDA_HPP_ */
