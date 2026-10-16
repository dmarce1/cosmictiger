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

#ifndef SPH_CUDA123_HPP_
#define SPH_CUDA123_HPP_

#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/timer.hpp>

#define ETA 0.01f

struct sph_run_cuda_data {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	char* stars;
#ifdef ENTROPY
	float* dentr1_snk;
	float* dentr2_snk;
	float* entr0_snk;
	float* entr;
#else
	float* deint1_snk;
	float* deint2_snk;
	float* eint0_snk;
	float* eint;
#endif
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
	float* omegaP;
	float* pre;
	float* h;
	char* rungs;
	char* rungs_snk;
	float* omega_snk;
	float* omegaP_snk;
	float* pre_snk;
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
#define MAX_RUNG_DIF 1
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

#ifdef __CUDACC__

__global__ void sph_cuda_prehydro1(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_prehydro2(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_aux(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_cond_init(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_hydro(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_conduction(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
__global__ void sph_cuda_rungs(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce);
#endif

struct softlens_record {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	char type;
};

#ifdef __CUDACC__
template <int BLOCK_SIZE>
inline __device__ bool compute_softlens(float& h, float hmin, float hmax, float N, device_vector<softlens_record> const& rec, array<fixed32, NDIM> const& x,
										fixed32_range const& obox, int type_i, float& wcount) {
	int const tid = threadIdx.x;
	int const block_size = blockDim.x;
	if (h < hmin) {
		if (tid == 0) {
			h = hmin * 1.1f;
		}
		__syncthreads();
	}
	if (h > hmax) {
		if (tid == 0) {
			//			h = hmax * 0.5f;
		}
		__syncthreads();
	}
	float const hinv = 1.f / h;
	__syncthreads();
	int count;
	float f;
	float dfdh;
	int box_xceeded = false;
	int iter = 0;
	float dh;
	float error;
	float sgn;
	float last_sgn = 0.f;
	float w = 1.0f;
	do {
		float max_dh = h * 0.1f;
		float const hinv = 1.f / h; // 4
		float const h2 = sqr(h);	// 1
		count = 0;
		f = 0.f;
		dfdh = 0.f;
		auto const& x_i = x[XDIM];
		auto const& y_i = x[YDIM];
		auto const& z_i = x[ZDIM];
		for (int j = tid; j < rec.size(); j += block_size) {
			auto const& r = rec[j];
			auto const& type_j = r.type;
			if (type_j == type_i) {
				auto const& x_j = r.x;
				auto const& y_j = r.y;
				auto const& z_j = r.z;
				float const x_ij = distance(x_i, x_j);
				float const y_ij = distance(y_i, y_j);
				float const z_ij = distance(z_i, z_j);
				float const r2 = sqr(x_ij, y_ij, z_ij);			   // 2
				float const r = sqrt(r2);						   // 4
				float const q = r * hinv;						   // 1
				if (q < 1.f) {									   // 1
					float const w = kernelW(q);					   // 4
					float const dwdh = -q * dkernelW_dq(q) * hinv; // 3
					f += w;										   // 1
					dfdh += dwdh;								   // 1
					count++;
				}
			}
		}
		shared_reduce_add<float, BLOCK_SIZE>(f);
		shared_reduce_add<int, BLOCK_SIZE>(count);
		wcount = float(4 * M_PI / 3) * f;
		shared_reduce_add<float, BLOCK_SIZE>(dfdh);
		float X, dXdh;
		dsmoothX_dh(h, hmin, hmax, X, dXdh);
		dh = 0.2f * h;
		if (count > 1) {
			dfdh = dfdh + dXdh * f;
			f *= X;
			f -= N * float(3.0 / (4.0 * M_PI));
			dh = -f / dfdh;
			sgn = copysignf(1.f, dh);
			if (sgn * last_sgn < 0.f) {
				w = fmaxf(0.5f * w, 0.001f);
			} else {
				w = fminf(1.f, w * 1.1f);
			}
			last_sgn = sgn;
			dh = fminf(fmaxf(dh, -max_dh), max_dh);
		}
		error = fabsf(dh / h);
		__syncthreads();
		if (tid == 0) {
			h += w * dh;
			if (iter >= 100) {
				PRINT("over iteration on h solve - %i %e %e %e %e %i\n", iter, h, dh, w, error, count);
			}
		}
		__syncthreads();
		for (int dim = 0; dim < NDIM; dim++) {
			if (obox.end[dim] < range_fixed(x[dim] + fixed32(h)) + range_fixed::min()) {
				box_xceeded = true;
				break;
			}
			if (range_fixed(x[dim]) < obox.begin[dim] + range_fixed(h) + range_fixed::min()) {
				box_xceeded = true;
				break;
			}
		}
		iter++;
		shared_reduce_add<int, BLOCK_SIZE>(box_xceeded);
	} while (error > 1e-4f && !box_xceeded);
	if (tid == 0 && h <= 0.f) {
		PRINT("Less than ZERO H! sph.cu %e\n", h);
		__trap();
	}
	return !box_xceeded;
}
#endif

#endif /* SPH_CUDA_HPP_ */
