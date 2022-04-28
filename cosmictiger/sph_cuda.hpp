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


#define ETA 0.01f


struct sph_run_cuda_data {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	char* stars;
	float* dentr1_snk;
	float* dentr2_snk;
	float* entr0_snk;
	float* entr;
	float* kappa;
	float* cold_frac;
	float gsoft;
	array<float, NCHEMFRACS>* chem;
	float* divv_snk;
	float* rho_snk;
	float* mmw;
	float* gradT;
	float* shearv;
	float* cold_mass_snk;
	float* dcold_mass;
	char* sa_snk;
	float def_gamma;
	bool conduction;
	float* rho;
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
	float* omega;
	float* h;
	char* rungs;
	char* rungs_snk;
	float* omega_snk;
	float* shear_snk;
	sph_record1* rec1_snk;
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

#define PREHYDRO1_BLOCK_SIZE 160
#define PREHYDRO2_BLOCK_SIZE 64
#define HYDRO_BLOCK_SIZE 96
#define AUX_BLOCK_SIZE 64
#define RUNGS_BLOCK_SIZE 256

#define COND_INIT_BLOCK_SIZE 32
#define CONDUCTION_BLOCK_SIZE 32


#define SPH_DIFFUSION_C 0.03f
#define MAX_RUNG_DIF 2
#define SPH_SMOOTHLEN_TOLER float(1.0e-4)

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
	float dtinv_cfl;
	float dtinv_visc;
	float dtinv_diff;
	float dtinv_cond;
	float dtinv_acc;
	float dtinv_divv;
	float dtinv_omega;
};

sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream);
__global__ void sph_cuda_prehydro1(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_prehydro2(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_aux(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_cond_init(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_hydro(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_conduction(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_rungs(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);



#endif /* SPH_CUDA_HPP_ */
