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

#define SPH_DIFFUSION_C 0.03f
#define SMOOTHLEN_BLOCK_SIZE 128
#define COND_INIT_BLOCK_SIZE 32
#define CONDUCTION_BLOCK_SIZE 128
#define RUNGS_BLOCK_SIZE 256
#define HYDRO_BLOCK_SIZE 32
#define AUX_BLOCK_SIZE 128
#define MAX_RUNG_DIF 2

#define SPH_SMOOTHLEN_TOLER float(1.0e-5)

#include <cosmictiger/sph_cuda.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/kernel.hpp>

#include <cosmictiger/math.hpp>

static __constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6),
		1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
				/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24),
		1.0 / (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

struct smoothlen_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
};

struct smoothlen_record2 {
	float entr;
	float cfrac;
};

struct smoothlen_workspace {
	device_vector<smoothlen_record1> x;
	device_vector<smoothlen_record2> v;
	device_vector<smoothlen_record1> xc;
	device_vector<smoothlen_record2> vc;
};

struct aux_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
};

struct aux_record2 {
	float vx;
	float vy;
	float vz;
};

struct aux_workspace {
	device_vector<aux_record1> x;
	device_vector<aux_record2> v;
	device_vector<aux_record1> xc;
	device_vector<aux_record2> vc;
};

struct rungs_workspace {
	device_vector<fixed32> x;
	device_vector<fixed32> y;
	device_vector<fixed32> z;
	device_vector<float> h;
	device_vector<char> rungs;
};

struct hydro_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};

struct hydro_record2 {
	array<float, NCHEMFRACS> chem;
	float vx;
	float vy;
	float vz;
	float entr;
	float alpha;
	float fpre1;
	float fpre2;
	float pre;
	float shearv;
	float cold_frac;
};

struct hydro_workspace {
	device_vector<hydro_record1> rec1_main;
	device_vector<hydro_record2> rec2_main;
	device_vector<hydro_record1> rec1;
	device_vector<hydro_record2> rec2;
};

struct sph_reduction {
	int counter;
	int flag;
	float hmin;
	float hmax;
	float vsig_max;
	float flops;
	int max_rung_hydro;
	int max_rung_grav;
	int max_rung;
};

__device__ void minverse(float& a00, float& a01, float& a02, float& a10, float& a11, float& a12, float &a20, float&a21, float&a22) {
	const float b00 = -a12 * a21 + a11 * a22;
	const float b01 = a02 * a21 - a01 * a22;
	const float b02 = -a02 * a11 + a01 * a12;
	const float b10 = a12 * a20 - a10 * a22;
	const float b11 = -a02 * a20 + a00 * a22;
	const float b12 = a02 * a10 - a00 * a12;
	const float b20 = -a11 * a20 + a10 * a21;
	const float b21 = a01 * a20 - a00 * a21;
	const float b22 = -a01 * a10 + a00 * a11;
	const float detinv = 1.f / (-(a02 * a11 * a20) + a01 * a12 * a20 + a02 * a10 * a21 - a00 * a12 * a21 - a01 * a10 * a22 + a00 * a11 * a22);
	a00 = b00 * detinv;
	a01 = b01 * detinv;
	a02 = b02 * detinv;
	a10 = b10 * detinv;
	a11 = b11 * detinv;
	a12 = b12 * detinv;
	a20 = b20 * detinv;
	a21 = b21 * detinv;
	a22 = b22 * detinv;
}

__device__ void mmult(float& a00, float& a01, float& a02, float& a10, float& a11, float& a12, float &a20, float&a21, float&a22, float b00, float b01, float b02,
		float b10, float b11, float b12, float b20, float b21, float b22) {
	const float c00 = a00 * b00 + a01 * b10 + a02 * b20;
	const float c01 = a00 * b01 + a01 * b11 + a02 * b21;
	const float c02 = a00 * b02 + a01 * b12 + a02 * b22;
	const float c10 = a10 * b00 + a11 * b10 + a12 * b20;
	const float c11 = a10 * b01 + a11 * b11 + a12 * b21;
	const float c12 = a10 * b02 + a11 * b12 + a12 * b22;
	const float c20 = a20 * b00 + a21 * b10 + a22 * b20;
	const float c21 = a20 * b01 + a21 * b11 + a22 * b21;
	const float c22 = a20 * b02 + a21 * b12 + a22 * b22;
	a00 = c00;
	a01 = c01;
	a02 = c02;
	a10 = c10;
	a11 = c11;
	a12 = c12;
	a20 = c20;
	a21 = c21;
	a22 = c22;
}

__global__ void sph_cuda_smoothlen(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ smoothlen_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) smoothlen_workspace();
	array<fixed32, NDIM> x;
	float error;
	while (index < data.nselfs) {

		int flops = 0;
		__syncthreads();
		ws.x.resize(0);
		ws.v.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		bool found_self = false;
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			if (data.neighbors[ni] == data.selfs[index]) {
				found_self = true;
			}
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					if (self.outer_box.contains(x)) {
						contains = true;
					}
				}
				j = contains;
				compute_indices<SMOOTHLEN_BLOCK_SIZE>(j, total);
				const int offset = ws.x.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				ws.v.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x[k].x = x[XDIM];
					ws.x[k].y = x[YDIM];
					ws.x[k].z = x[ZDIM];
					ws.v[k].entr = data.entr[pi];
					if (params.stars) {
						ws.v[k].cfrac = data.cold_frac[pi];
					} else {
						ws.v[k].cfrac = 0.f;
					}
				}
			}
		}
		ALWAYS_ASSERT(found_self);
		ALWAYS_ASSERT(ws.x.size());
		const float gamma0 = data.def_gamma;
		float hmin = 1e+20;
		float hmax = 0.0;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs_snk[data.dm_index_snk[snki]] >= params.min_rung;
			const bool converged = data.converged_snk[snki];
			const bool use = active && !converged;
			const float w0 = kernelW(0.f);
			if (use) {
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				int box_xceeded = false;
				int iter = 0;
				float& h = data.rec1_snk[snki].h;
				float drho_dh;
				float rhoh3;
				float last_dh = 0.0f;
				float w1 = 1.0f;
				do {
					const float hinv = 1.f / h; // 4
					const float h2 = sqr(h);    // 1
					drho_dh = 0.f;
					rhoh3 = 0.f;
					float rhoh30 = (3.0f * data.N) / (4.0f * float(M_PI));
					for (int j = tid; j < ws.x.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.x[j].x); // 2
						const float dy = distance(x[YDIM], ws.x[j].y); // 2
						const float dz = distance(x[ZDIM], ws.x[j].z); // 2
						const float r2 = sqr(dx, dy, dz);            // 2
						const float r = sqrt(r2);                    // 4
						const float q = r * hinv;                    // 1
						if (q < 1.f) {                               // 1
							const float w = kernelW(q); // 4
							const float dwdq = dkernelW_dq(q);
							const float dwdh = -q * dwdq * hinv; // 3
							drho_dh -= dwdq;
							rhoh3 += w;
						}

					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(drho_dh);
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(rhoh3);
					float dlogh;
					__syncthreads();
					if (rhoh3 <= 1.01f * w0) {
						PRINT("ZERO neighbors %i\n", ws.x.size());
						if (tid == 0) {
							h *= 1.1f;
						}
						iter--;
						error = 1.0;
					} else {
						drho_dh *= 0.33333333333f / rhoh30;
						const float fpre = fminf(fmaxf(1.0f / (drho_dh), 0.5f), 2.0f);
						dlogh = fminf(fmaxf(powf(rhoh30 / rhoh3, fpre * 0.3333333333333333f) - 1.f, -.1f), .1f);
						error = fabs(1.0f - rhoh3 / rhoh30);
						if (tid == 0) {
							h *= (1.f + w1 * dlogh);
						}
						if (last_dh * dlogh < 0.f) {
							w1 *= 0.5;
						} else {
							w1 = 1.f;
						}
						last_dh = dlogh;
					}
					__syncthreads();
					if (tid == 0) {
						if (iter > 100000) {
							PRINT("Solver failed to converge - %i %e %e %e\n", iter, h, dlogh, error);
							if (iter > 100010) {
								__trap();
							}
						}
					}
					for (int dim = 0; dim < NDIM; dim++) {
						if (self.outer_box.end[dim] < range_fixed(x[dim] + fixed32(h)) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
						if (range_fixed(x[dim]) < self.outer_box.begin[dim] + range_fixed(h) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
					}
					__syncthreads();
					iter++;
					shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(box_xceeded);
				} while (error > SPH_SMOOTHLEN_TOLER && !box_xceeded);
				if (!box_xceeded) {
					const float hinv = 1.f / h; // 4
					const float h2 = sqr(h);    // 1
					drho_dh = 0.f;
					float rhoh30 = (3.0f * data.N) / (4.0f * float(M_PI));
					const fixed32& x_i = x[XDIM];
					const fixed32& y_i = x[YDIM];
					const fixed32& z_i = x[ZDIM];
					float pre = 0.f;
					float dpdh = 0.f;
					ws.xc.resize(0);
					ws.vc.resize(0);
					__syncthreads();
					const int jmax = round_up(ws.x.size(), SMOOTHLEN_BLOCK_SIZE);
					for (int j = tid; j < jmax; j += block_size) {
						bool contains = false;
						if (j < ws.x.size()) {
							const fixed32 x_j = ws.x[j].x;
							const fixed32 y_j = ws.x[j].y;
							const fixed32 z_j = ws.x[j].z;
							const float x_ij = distance(x_i, x_j); // 2
							const float y_ij = distance(y_i, y_j); // 2
							const float z_ij = distance(z_i, z_j); // 2
							const float r2 = sqr(x_ij, y_ij, z_ij);
							const float r = sqrt(r2);                    // 4
							const float q = r * hinv;                    // 1
							if (q < 1.f) {                               // 1
								const float w = kernelW(q); // 4
								const float dwdq = dkernelW_dq(q);
								drho_dh -= q * dwdq;
								contains = true;
							}
						}
						int k = contains;
						int total;
						compute_indices<SMOOTHLEN_BLOCK_SIZE>(k, total);
						const int offset = ws.xc.size();
						__syncthreads();
						const int next_size = offset + total;
						ws.xc.resize(next_size);
						ws.vc.resize(next_size);
						if (contains) {
							const int l = offset + k;
							ws.xc[l] = ws.x[j];
							ws.vc[l] = ws.v[j];
						}
					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(drho_dh);
					const float m = data.m;
					drho_dh *= 0.33333333333f / rhoh30;
					const float fpre = 1.0f / drho_dh;
					__syncthreads();
					const float hinv_i = 1.0f / h;
					const float h3inv_i = hinv_i * sqr(hinv_i);
					const float h4inv_i = h3inv_i * hinv_i;
					for (int j = tid; j < ws.xc.size(); j += block_size) {
						const fixed32 x_j = ws.xc[j].x;
						const fixed32 y_j = ws.xc[j].y;
						const fixed32 z_j = ws.xc[j].z;
						const float A_j = ws.vc[j].entr;
						const float fc_j = ws.vc[j].cfrac;
						const float fh_j = 1.f - fc_j;
						const float x_ij = distance(x_i, x_j); // 2
						const float y_ij = distance(y_i, y_j); // 2
						const float z_ij = distance(z_i, z_j); // 2
						const float r2 = sqr(x_ij, y_ij, z_ij);
						const float r = sqrt(r2);                    // 4
						const float q = r * hinv;                    // 1
						const float rinv = 1.0f / (1.0e-30f + r);
						const float w = kernelW(q); // 4
						const float dwdq = dkernelW_dq(q);
						const float dWdr_i = dwdq * h4inv_i;
						const float A0_j = fh_j * powf(A_j, 1.0f / gamma0);
						pre += m * A0_j * kernelW(q) * h3inv_i;
						dpdh -= A0_j * (3.f * w + q * dwdq);
					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dpdh);
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(pre);
					pre = powf(pre, gamma0);
					dpdh *= 0.33333333333f / rhoh30;
					if (tid == 0) {
						data.rec1_snk[snki].fpre1 = fpre;
						data.rec1_snk[snki].fpre2 = dpdh;
						data.rec1_snk[snki].pre = pre;
						data.converged_snk[snki] = true;
					}
				} else {
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
					}
				}
				hmin = fminf(hmin, h);
				hmax = fmaxf(hmax, h);
			}
		}
		shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			atomicMax(&reduce->hmax, hmax);
			atomicMin(&reduce->hmin, hmin);
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	(&ws)->~smoothlen_workspace();
}

__global__ void sph_cuda_aux(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ aux_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) aux_workspace();
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {

		int flops = 0;
		__syncthreads();
		ws.x.resize(0);
		ws.v.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		bool found_self = false;
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			if (data.neighbors[ni] == data.selfs[index]) {
				found_self = true;
			}
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					if (self.outer_box.contains(x)) {
						contains = true;
					}
				}
				j = contains;
				compute_indices<SMOOTHLEN_BLOCK_SIZE>(j, total);
				const int offset = ws.x.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				ws.v.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x[k].x = x[XDIM];
					ws.x[k].y = x[YDIM];
					ws.x[k].z = x[ZDIM];
					ws.v[k].vx = data.vx[pi];
					ws.v[k].vy = data.vy[pi];
					ws.v[k].vz = data.vz[pi];
				}
			}
		}
		ALWAYS_ASSERT(found_self);
		ALWAYS_ASSERT(ws.x.size());
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs_snk[data.dm_index_snk[snki]] >= params.min_rung;
			const bool use = active;
			if (use) {
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				float& h = data.rec1_snk[snki].h;
				const float vx_i = data.vx[i];
				const float vy_i = data.vy[i];
				const float vz_i = data.vz[i];
				const float hinv = 1.f / h; // 4
				const float h2 = sqr(h);    // 1
				float rhoh30 = (3.0f * data.N) / (4.0f * float(M_PI));
				const fixed32& x_i = x[XDIM];
				const fixed32& y_i = x[YDIM];
				const fixed32& z_i = x[ZDIM];
				float dx_dx = 0.f;
				float dx_dy = 0.f;
				float dx_dz = 0.f;
				float dy_dx = 0.f;
				float dy_dy = 0.f;
				float dy_dz = 0.f;
				float dz_dx = 0.f;
				float dz_dy = 0.f;
				float dz_dz = 0.f;
				float dvx_dx = 0.f;
				float dvx_dy = 0.f;
				float dvx_dz = 0.f;
				float dvy_dx = 0.f;
				float dvy_dy = 0.f;
				float dvy_dz = 0.f;
				float dvz_dx = 0.f;
				float dvz_dy = 0.f;
				float dvz_dz = 0.f;
				float pre = 0.f;
				ws.xc.resize(0);
				ws.vc.resize(0);
				__syncthreads();
				const int jmax = round_up(ws.x.size(), SMOOTHLEN_BLOCK_SIZE);
				for (int j = tid; j < jmax; j += block_size) {
					bool contains = false;
					if (j < ws.x.size()) {
						const fixed32 x_j = ws.x[j].x;
						const fixed32 y_j = ws.x[j].y;
						const fixed32 z_j = ws.x[j].z;
						const float x_ij = distance(x_i, x_j); // 2
						const float y_ij = distance(y_i, y_j); // 2
						const float z_ij = distance(z_i, z_j); // 2
						const float r2 = sqr(x_ij, y_ij, z_ij);
						const float r = sqrt(r2);                    // 4
						const float q = r * hinv;                    // 1
						if (q < 1.f) {                               // 1
							contains = true;
						}
					}
					int k = contains;
					int total;
					compute_indices<SMOOTHLEN_BLOCK_SIZE>(k, total);
					const int offset = ws.xc.size();
					__syncthreads();
					const int next_size = offset + total;
					ws.xc.resize(next_size);
					ws.vc.resize(next_size);
					if (contains) {
						const int l = offset + k;
						ws.xc[l] = ws.x[j];
						ws.vc[l] = ws.v[j];
					}
				}
				const float m = data.m;
				__syncthreads();
				const float hinv_i = 1.0f / h;
				const float h3inv_i = hinv_i * sqr(hinv_i);
				const float h4inv_i = h3inv_i * hinv_i;
				for (int j = tid; j < ws.xc.size(); j += block_size) {
					const fixed32 x_j = ws.xc[j].x;
					const fixed32 y_j = ws.xc[j].y;
					const fixed32 z_j = ws.xc[j].z;
					const float x_ij = distance(x_i, x_j); // 2
					const float y_ij = distance(y_i, y_j); // 2
					const float z_ij = distance(z_i, z_j); // 2
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);                    // 4
					const float q = r * hinv;                    // 1
					const float vx_j = ws.vc[j].vx;
					const float vy_j = ws.vc[j].vy;
					const float vz_j = ws.vc[j].vz;
					const float vx0_ij = vx_i - vx_j;
					const float vy0_ij = vy_i - vy_j;
					const float vz0_ij = vz_i - vz_j;
					const float vx_ij = vx0_ij + x_ij * params.adot;
					const float vy_ij = vy0_ij + y_ij * params.adot;
					const float vz_ij = vz0_ij + z_ij * params.adot;
					const float rinv = 1.0f / (1.0e-30f + r);
					const float w = kernelW(q); // 4
					const float dwdq = dkernelW_dq(q);
					const float dWdr_i = dwdq * h4inv_i;
					dx_dx += x_ij * x_ij * dWdr_i * rinv;
					dx_dy += x_ij * y_ij * dWdr_i * rinv;
					dx_dz += x_ij * z_ij * dWdr_i * rinv;
					dy_dy += y_ij * y_ij * dWdr_i * rinv;
					dy_dz += y_ij * z_ij * dWdr_i * rinv;
					dz_dz += z_ij * z_ij * dWdr_i * rinv;
					dvx_dx += vx_ij * x_ij * dWdr_i * rinv;
					dvy_dx += vy_ij * x_ij * dWdr_i * rinv;
					dvz_dx += vz_ij * x_ij * dWdr_i * rinv;
					dvx_dy += vx_ij * y_ij * dWdr_i * rinv;
					dvy_dy += vy_ij * y_ij * dWdr_i * rinv;
					dvz_dy += vz_ij * y_ij * dWdr_i * rinv;
					dvx_dz += vx_ij * z_ij * dWdr_i * rinv;
					dvy_dz += vy_ij * z_ij * dWdr_i * rinv;
					dvz_dz += vz_ij * z_ij * dWdr_i * rinv;
				}
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dx_dx);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dx_dy);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dx_dz);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dy_dy);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dy_dz);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dz_dz);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvx_dx);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvx_dy);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvx_dz);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvy_dx);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvy_dy);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvy_dz);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvz_dx);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvz_dy);
				shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dvz_dz);
				dy_dx = dx_dy;
				dz_dx = dx_dz;
				dz_dy = dy_dz;
				minverse(dx_dx, dx_dy, dx_dz, dy_dx, dy_dy, dy_dz, dz_dx, dz_dy, dz_dz);
				mmult(dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz, dx_dx, dx_dy, dx_dz, dy_dx, dy_dy, dy_dz, dz_dx, dz_dy, dz_dz);
				float shear_xx, shear_xy, shear_xz, shear_yy, shear_yz, shear_zz;
				float div_v, curl_vx, curl_vy, curl_vz;
				div_v = dvx_dx + dvy_dy + dvz_dz;
				curl_vx = dvz_dy - dvy_dz;
				curl_vy = -dvz_dx + dvx_dz;
				curl_vz = dvy_dx - dvx_dy;
				shear_xx = dvx_dx - (1.f / 3.f) * div_v;
				shear_yy = dvy_dy - (1.f / 3.f) * div_v;
				shear_zz = dvz_dz - (1.f / 3.f) * div_v;
				shear_xy = 0.5f * (dvx_dy + dvy_dx);
				shear_xz = 0.5f * (dvx_dz + dvz_dx);
				shear_yz = 0.5f * (dvy_dz + dvz_dy);
				if (tid == 0) {
					const float curlv = sqrtf(sqr(curl_vx, curl_vy, curl_vz));
					const float shearv = sqrtf(sqr(shear_xx) + sqr(shear_yy) + sqr(shear_zz) + 2.0f * (sqr(shear_xy) + sqr(shear_xz) + sqr(shear_yz)));
					data.rec1_snk[snki].shearv = shearv;
					data.rec3_snk[snki].divv0 = data.rec3_snk[snki].divv;
					data.rec3_snk[snki].divv = div_v;
					data.curlv_snk[snki] = curlv;
				}
			}
		}
		shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	(&ws)->~aux_workspace();
}

__global__ void sph_cuda_hydro(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ hydro_workspace ws;
	new (&ws) hydro_workspace();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	float total_vsig_max = 0.;
	int max_rung_hydro = 0;
	int max_rung_grav = 0;
	int max_rung = 0;
	int flops = 0;
	const float ainv = 1.0f / params.a;
	while (index < data.nselfs) {
		__syncthreads();
		const sph_tree_node& self = data.trees[data.selfs[index]];
		ws.rec1_main.resize(0);
		ws.rec2_main.resize(0);
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					if (self.outer_box.contains(x)) {
						contains = true;
					}
					if (!contains) {
						contains = true;
						const float& h = data.h[pi];
						for (int dim = 0; dim < NDIM; dim++) {
							if (distance(x[dim], self.inner_box.begin[dim]) + h < 0.f) {
								contains = false;
								break;
							}
							if (distance(self.inner_box.end[dim], x[dim]) + h < 0.f) {
								contains = false;
								break;
							}
						}
					}
				}
				j = contains;
				compute_indices<HYDRO_BLOCK_SIZE>(j, total);
				const int offset = ws.rec1_main.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec1_main.resize(next_size);
				ws.rec2_main.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1_main[k].x = x[XDIM];
					ws.rec1_main[k].y = x[YDIM];
					ws.rec1_main[k].z = x[ZDIM];
					ws.rec1_main[k].h = data.h[pi];
					ws.rec2_main[k].vx = data.vx[pi];
					ws.rec2_main[k].vy = data.vy[pi];
					ws.rec2_main[k].vz = data.vz[pi];
					ws.rec2_main[k].entr = data.entr[pi];
					ws.rec2_main[k].alpha = data.alpha[pi];
					ws.rec2_main[k].fpre1 = data.fpre1[pi];
					ws.rec2_main[k].fpre2 = data.fpre2[pi];
					ws.rec2_main[k].pre = data.pre[pi];
					if (params.diffusion) {
						ws.rec2_main[k].shearv = data.shearv[pi];
					}
					if (params.stars) {
						ws.rec2_main[k].cold_frac = data.cold_frac[pi];
					} else {
						ws.rec2_main[k].cold_frac = 0.f;
					}
					if (params.diffusion) {
						if (data.chemistry) {
							ws.rec2_main[k].chem = data.chem[pi];
						}
					}
				}
			}
		}
		const float colog0 = log(1.5 * pow(constants::kb, 1.5) * pow(constants::e, -3) * pow(M_PI, -0.5));
		double kappa00 = 20.0 * pow(2.0 / M_PI, 1.5) * pow(constants::kb, 2.5) * pow(constants::me, -0.5) * pow(constants::e, -4.0);
		kappa00 *= (double) params.code_to_s * params.code_to_cm;
		const float lambda0 = pow(3, 1.5) * pow((double) constants::kb, 2.0) / (4.0 * sqrt(M_PI) * sqr(sqr((double) constants::e)));
		kappa00 /= constants::avo;
		kappa00 /= params.code_to_g;
		float kappa0 = kappa00;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			int rung_i = data.rungs_snk[data.dm_index_snk[snki]];
			bool use = rung_i >= params.min_rung;
			const float m = data.m;
			const float minv = 1.f / m;
			const float c0 = float(3.0f / 4.0f / M_PI * data.N);
			const float c0inv = 1.0f / c0;
			if (use) {
				const auto x_i = data.x[i];
				const auto y_i = data.y[i];
				const auto z_i = data.z[i];
				const auto vx_i = data.vx[i];
				const auto vy_i = data.vy[i];
				const auto vz_i = data.vz[i];
				const float h_i = data.h[i];
				const float h2_i = sqr(h_i);
				const float hinv_i = 1.f / h_i;
				const float h3inv_i = (sqr(hinv_i) * hinv_i);
				const float rho_i = m * c0 * h3inv_i;
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;
				const float A_i = data.entr[i];
				const float alpha_i = data.alpha[i];
				const float gamma0 = data.def_gamma;
				float cfrac_i;
				if (params.stars) {
					cfrac_i = data.cold_frac[i];
				} else {
					cfrac_i = 0.f;
				}
				const float hfrac_i = 1.f - cfrac_i;
				array<float, NCHEMFRACS> frac_i;
				float shearv_i;
				if ((params.diffusion) && data.chemistry) {
					frac_i = data.chem[i];
				}
				if (params.diffusion) {
					shearv_i = data.shearv[i];
				}
				const float de_dt0 = (gamma0 - 1.f) * powf(hfrac_i * rho_i, 1.f - gamma0);
				const float pre_i = data.pre[i];
				const float c_i = sqrtf(gamma0 * powf(pre_i, 1.0f - 1.f / gamma0) * powf(A_i, 1.0f / gamma0));
				const float fpre1_i = data.fpre1[i];
				const float fpre2_i = data.fpre2[i];
				float ax = 0.f;
				float ay = 0.f;
				float az = 0.f;
				float de_dt = 0.f;
				float dcm_dt = 0.f;
				array<float, NCHEMFRACS> dfrac_dt;
				if (data.chemistry) {
					for (int fi = 0; fi < NCHEMFRACS; fi++) {
						dfrac_dt[fi] = 0.f;
					}
				}
				float dtinv_cfl = 0.f;
				float one = 0.0f;
				float vsig = 0.f;
				constexpr float tiny = 1e-30f;
				const float& adot = params.adot;
				float D = 0.f;
				ws.rec1.resize(0);
				ws.rec2.resize(0);
				const float jmax = round_up(ws.rec1_main.size(), block_size);
				for (int j = tid; j < jmax; j += block_size) {
					int k;
					int total;
					bool use = false;
					if (j < ws.rec1_main.size()) {
						const auto rec1 = ws.rec1_main[j];
						const fixed32 x_j = rec1.x;
						const fixed32 y_j = rec1.y;
						const fixed32 z_j = rec1.z;
						const float h_j = rec1.h;
						const float x_ij = distance(x_i, x_j);				// 2
						const float y_ij = distance(y_i, y_j);				// 2
						const float z_ij = distance(z_i, z_j);				// 2
						const float r2 = sqr(x_ij, y_ij, z_ij);
						if (r2 < fmaxf(sqr(h_i), sqr(h_j))) {
							use = true;
						}
					}
					k = use;
					compute_indices<HYDRO_BLOCK_SIZE>(k, total);
					const int offset = ws.rec1.size();
					__syncthreads();
					const int next_size = offset + total;
					ws.rec1.resize(next_size);
					ws.rec2.resize(next_size);
					if (use) {
						const int l = offset + k;
						ws.rec1[l] = ws.rec1_main[j];
						ws.rec2[l] = ws.rec2_main[j];
					}
				}
				__syncthreads();
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					const auto rec1 = ws.rec1[j];
					const auto rec2 = ws.rec2[j];
					const float h_j = rec1.h;
					const float hinv_j = 1.f / h_j;															// 4
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);
					const float q_i = r * hinv_i;								// 1
					const float q_j = r * hinv_j;
					if (q_i < 1.f || q_j < 1.f) {
						const float cfrac_j = rec2.cold_frac;
						const float hfrac_j = 1.f - cfrac_j;
						const float shearv_j = rec2.shearv;
						const auto& frac_j = rec2.chem;
						const float vx_j = rec2.vx;
						const float vy_j = rec2.vy;
						const float vz_j = rec2.vz;
						const float pre_j = rec2.pre;
						const float fpre1_j = rec2.fpre1;
						const float fpre2_j = rec2.fpre2;
						const float h2_j = sqr(h_j);
						const float h3inv_j = sqr(hinv_j) * hinv_j;
						const float rho_j = m * c0 * h3inv_j;													// 2
						const float rhoinv_j = minv * c0inv * sqr(h_j) * h_j;								// 5
						const float A_j = rec2.entr;
						const float c_j = sqrtf(gamma0 * powf(pre_j, 1.0f - 1.f / gamma0) * powf(A_j, 1.0f / gamma0));
						const float alpha_j = rec2.alpha;
						const float vx0_ij = vx_i - vx_j;
						const float vy0_ij = vy_i - vy_j;
						const float vz0_ij = vz_i - vz_j;
						const float vx_ij = vx0_ij + x_ij * adot;
						const float vy_ij = vy0_ij + y_ij * adot;
						const float vz_ij = vz0_ij + z_ij * adot;
						const float rinv = 1.0f / (r > 0.f ? r : 1e37f);
						const float vdotx_ij = x_ij * vx_ij + y_ij * vy_ij + z_ij * vz_ij;
						const float h_ij = 0.5f * (h_i + h_j);
						const float w_ij = fminf(vdotx_ij * rinv, 0.f);
						const float mu_ij = fminf(vdotx_ij * h_ij / (r2 + sqr(h_ij) * 0.01f), 0.f);
//						const float mu_ij = w_ij * h_ij * rinv;
						const float rho_ij = 0.5f * (rho_i + rho_j);
						const float c_ij = 0.5f * (c_i + c_j);
						const float alpha_ij = 0.5f * (alpha_i + alpha_j);									// * (balsara_i + balsara_j);
						const float beta_ij = alpha_ij * 2.0f;
						const float hfrac_ij = 2.0f * (hfrac_i * hfrac_j) / (hfrac_i + hfrac_j + 1e-30f);
						const float vsig_ij = hfrac_ij * (alpha_ij * c_ij - beta_ij * mu_ij);
						const float pi_ij = -mu_ij * vsig_ij / rho_ij;
						const float dWdr_i = dkernelW_dq(q_i) * hinv_i * h3inv_i;
						const float dWdr_j = dkernelW_dq(q_j) * hinv_j * h3inv_j;
						const float dWdr_ij = 0.5f * (fpre1_i * dWdr_i + fpre1_j * dWdr_j);
						const float dWdr_x_ij = x_ij * rinv * dWdr_ij;
						const float dWdr_y_ij = y_ij * rinv * dWdr_ij;
						const float dWdr_z_ij = z_ij * rinv * dWdr_ij;
						const float acor_i = 1.f - fpre2_i * fpre1_i * powf(A_j, -1.0f / gamma0);
						const float acor_j = 1.f - fpre2_j * fpre1_j * powf(A_i, -1.0f / gamma0);
						const float dWdr_x_i = acor_i * x_ij * rinv * dWdr_i;
						const float dWdr_y_i = acor_i * y_ij * rinv * dWdr_i;
						const float dWdr_z_i = acor_i * z_ij * rinv * dWdr_i;
						const float dWdr_x_j = acor_j * x_ij * rinv * dWdr_j;
						const float dWdr_y_j = acor_j * y_ij * rinv * dWdr_j;
						const float dWdr_z_j = acor_j * z_ij * rinv * dWdr_j;
						const float aco = powf(A_i * A_j, 1.0f / gamma0);
						const float dp_i = aco * powf(pre_i, 1.0f - 2.0f / gamma0);
						const float dp_j = aco * powf(pre_j, 1.0f - 2.0f / gamma0);
						one += m / rho_i * kernelW(q_i) * h3inv_i;
						ax -= m * ainv * (dp_i * dWdr_x_i + dp_j * dWdr_x_j);
						ay -= m * ainv * (dp_i * dWdr_y_i + dp_j * dWdr_y_j);
						az -= m * ainv * (dp_i * dWdr_z_i + dp_j * dWdr_z_j);
						ax -= m * ainv * (pi_ij * dWdr_x_ij);
						ay -= m * ainv * (pi_ij * dWdr_y_ij);
						az -= m * ainv * (pi_ij * dWdr_z_ij);
						const float dW_ij = (vx_ij * dWdr_x_ij + vy_ij * dWdr_y_ij + vz_ij * dWdr_z_ij);
						de_dt += de_dt0 * 0.5f * m * ainv * pi_ij * dW_ij;
						const float h = fminf(h_i, h_j);
						float dtinv = (c_ij + 0.6f * vsig_ij) / h;
						dtinv_cfl = fmaxf(dtinv_cfl, dtinv);
						vsig = fmaxf(vsig, c_ij - w_ij);
						if (params.diffusion) {
							const float difco_i = SPH_DIFFUSION_C * sqr(h_i) * shearv_i;
							const float difco_j = SPH_DIFFUSION_C * sqr(h_j) * shearv_j;
							const float difco_ij = 0.5f * (difco_i + difco_j);
							const float D_ij = -2.f * m / rho_ij * difco_ij * dWdr_ij * rinv * ainv * (r2 / (r2 + 0.01f * sqr(h_ij)));
							D += D_ij;
							de_dt -= D_ij * (A_i - A_j * powf(hfrac_j * rho_j / (hfrac_i * rho_i), gamma0 - 1.f));
							if (params.stars) {
								dcm_dt -= D_ij * (cfrac_i - cfrac_j);
							}
							if (data.chemistry) {
								for (int fi = 0; fi < NCHEMFRACS; fi++) {
									dfrac_dt[fi] -= D_ij * (frac_i[fi] - frac_j[fi] * hfrac_j / hfrac_i);
								}
							}
						}
					}
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(de_dt);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ax);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ay);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(az);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(one);
				shared_reduce_max<HYDRO_BLOCK_SIZE>(dtinv_cfl);
				shared_reduce_max<HYDRO_BLOCK_SIZE>(vsig);
				if (params.diffusion) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(D);
					if (params.stars) {
						shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dcm_dt);
					}
					if (data.chemistry) {
						for (int fi = 0; fi < NCHEMFRACS; fi++) {
							shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dfrac_dt[fi]);
						}
					}
				}
				if (fabs(1. - one) > 1.0e-4 && tid == 0) {
					PRINT("one is off %e %i\n", one, data.converged_snk[snki]);
					__trap();
				}
				if (tid == 0) {
					float gx_i;
					float gy_i;
					float gz_i;
					if (data.gravity) {
						gx_i = data.gx_snk[snki];
						gy_i = data.gy_snk[snki];
						gz_i = data.gz_snk[snki];
					} else {
						gx_i = 0.f;
						gy_i = 0.f;
						gz_i = 0.f;
					}
					ax += gx_i;
					ay += gy_i;
					az += gz_i;
					data.rec6_snk[snki].dvel[XDIM] = ax;
					data.rec6_snk[snki].dvel[YDIM] = ay;
					data.rec6_snk[snki].dvel[ZDIM] = az;
					data.rec5_snk[snki].dA = de_dt;
					if (params.stars) {
						data.rec5_snk[snki].dfcold = dcm_dt;
					}
					if (data.chemistry) {
						data.rec5_snk[snki].dfrac = dfrac_dt;
					}
					const float div_v = data.rec3_snk[snki].divv;
					const float dtinv_divv = params.a * fabsf(div_v - 3.f * params.adot * ainv) * (1.f / 3.f);
					float dtinv_hydro1 = 1.0e-30f;
					dtinv_hydro1 = fmaxf(dtinv_hydro1, dtinv_divv);
					dtinv_hydro1 = fmaxf(dtinv_hydro1, dtinv_cfl);
					if (params.diffusion) {
						dtinv_hydro1 = fmaxf(dtinv_hydro1, params.a * D * 3.f);
					}
					const float a2 = sqr(ax, ay, az);
					const float dtinv_acc = sqrtf(sqrtf(a2) * hinv_i);
					float dthydro = params.cfl * params.a / (dtinv_hydro1 + 1e-30f);
					dthydro = fminf(data.eta * sqrtf(params.a) / (dtinv_acc + 1e-30f), dthydro);
					const float g2 = sqr(gx_i, gy_i, gz_i);
					const float dtinv_grav = sqrtf(sqrtf(g2));
					float dtgrav = data.eta * sqrtf(params.a * data.gsoft) / (dtinv_grav + 1e-30f);
					dthydro = fminf(dthydro, params.max_dt);
					ALWAYS_ASSERT(dcm_dt * dthydro + cfrac_i > -1e-7f);
					dtgrav = fminf(dtgrav, params.max_dt);
					total_vsig_max = fmaxf(total_vsig_max, dtinv_hydro1 * h_i);
					char& rung = data.rungs_snk[data.dm_index_snk[snki]];
					const float last_dt = rung_dt[rung] * params.t0;
					data.oldrung_snk[snki] = rung;
					const int rung_hydro = ceilf(log2f(params.t0) - log2f(dthydro));
					const int rung_grav = ceilf(log2f(params.t0) - log2f(dtgrav));
					max_rung_hydro = max(max_rung_hydro, rung_hydro);
					max_rung_grav = max(max_rung_grav, rung_grav);
					rung = max(max((int) max(rung_hydro, rung_grav), max(params.min_rung, (int) rung - 1)), 1);
					max_rung = max(max_rung, rung);
					if (rung < 0 || rung >= MAX_RUNG) {
						if (tid == 0) {
							PRINT("Rung out of range \n");
							__trap();
						}
					}
					const float alpha = alpha_i;
					const float curlv = data.curlv_snk[snki];
					const float divv0 = params.tau > 0.f ? data.rec3_snk[snki].divv0 : div_v;
					const float ddivv_dt = (div_v - divv0) / last_dt - 0.5f * params.adot * ainv * (div_v + divv0);
					const float S = sqr(h_i) * fmaxf(0.f, -ddivv_dt) * sqr(params.a);
					const float limiter = sqr(div_v) / (sqr(div_v) + sqr(curlv) + 1.0e-4f * sqr(c_i / h_i * ainv));
					const float alpha_targ = fmaxf(params.alpha1 * S / (S + sqr(c_i)), params.alpha0);
					const float lambda0 = params.alpha_decay * vsig * hinv_i * ainv;
					const float lambda1 = 1.f / dthydro;
					float dalpha_dt;
					if (alpha < limiter * alpha_targ) {
						dalpha_dt = (limiter * alpha_targ - alpha) * lambda1;
					} else {
						dalpha_dt = (limiter * (alpha_targ + (alpha - alpha_targ) * expf(-lambda0 * last_dt)) - alpha) * lambda1;
					}
					data.dalpha[snki] = dalpha_dt;
				}
			}
		}
		shared_reduce_add<int, HYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (float) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	if (tid == 0) {
		atomicMax(&reduce->vsig_max, total_vsig_max);
		atomicMax(&reduce->max_rung, max_rung);
		atomicMax(&reduce->max_rung_hydro, max_rung_hydro);
		atomicMax(&reduce->max_rung_grav, max_rung_grav);
	}
	(&ws)->~hydro_workspace();
}

__global__ void sph_cuda_rungs(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ rungs_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) rungs_workspace();

	array<fixed32, NDIM> x;
	int changes = 0;
	while (index < data.nselfs) {
		__syncthreads();
		int flops = 0;
		ws.x.resize(0);
		ws.y.resize(0);
		ws.z.resize(0);
		ws.h.resize(0);
		ws.rungs.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				float h;
				int rung;
				if (pi < other.part_range.second) {
					h = data.h[pi];
					rung = data.rungs[pi];
					if (rung >= params.min_rung) {
						x[XDIM] = data.x[pi];
						x[YDIM] = data.y[pi];
						x[ZDIM] = data.z[pi];
						if (self.outer_box.contains(x)) {
							contains = true;
						}
						if (!contains) {
							contains = true;
							for (int dim = 0; dim < NDIM; dim++) {
								if (distance(x[dim], self.inner_box.begin[dim]) + h < 0.f) {
									contains = false;
									break;
								}
								if (distance(self.inner_box.end[dim], x[dim]) + h < 0.f) {
									contains = false;
									break;
								}
							}
						}
					}
				}
				j = contains;
				compute_indices<RUNGS_BLOCK_SIZE>(j, total);
				const int offset = ws.x.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				ws.y.resize(next_size);
				ws.z.resize(next_size);
				ws.h.resize(next_size);
				ws.rungs.resize(next_size);
				if (contains) {
					ASSERT(j < total);
					const int k = offset + j;
					ws.x[k] = x[XDIM];
					ws.y[k] = x[YDIM];
					ws.z[k] = x[ZDIM];
					ws.h[k] = h;
					ws.rungs[k] = rung;
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			if (data.rungs[i] >= params.min_rung) {
				const auto x_i = data.x[i];
				const auto y_i = data.y[i];
				const auto z_i = data.z[i];
				const auto h_i = data.h[i];
				char& rung_i = data.rungs[i];
				const auto h2_i = sqr(h_i);
				const int jmax = round_up(ws.x.size(), block_size);
				int max_rung_j = 0;
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.x.size()) {
						const auto x_j = ws.x[j];
						const auto y_j = ws.y[j];
						const auto z_j = ws.z[j];
						const auto h_j = ws.h[j];
						const int rung_j = ws.rungs[j];
						const auto h2_j = sqr(h_j);
						const float x_ij = distance(x_i, x_j);
						const float y_ij = distance(y_i, y_j);
						const float z_ij = distance(z_i, z_j);
						const float r2 = sqr(x_ij, y_ij, z_ij);
						if (r2 < fmaxf(h2_i, h2_j)) {
							max_rung_j = max(max_rung_j, rung_j);
						}
					}
				}
				shared_reduce_max<RUNGS_BLOCK_SIZE>(max_rung_j);
				if (tid == 0) {
					if (rung_i < max_rung_j - MAX_RUNG_DIF) {
						changes++;
						rung_i = max_rung_j - MAX_RUNG_DIF;
					}
				}
			}
		}
		shared_reduce_add<int, RUNGS_BLOCK_SIZE>(flops);
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(&reduce->flag, changes);
	}
	(&ws)->~rungs_workspace();

}

struct cond_init_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};
struct cond_init_record2 {
	float entr;
	float cfrac;
};

struct cond_init_workspace {
	device_vector<cond_init_record1> rec1;
	device_vector<cond_init_record2> rec2;
};

struct conduction_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
	char rung;
};
struct conduction_record2 {
	float entr;
	float cfrac;
	float kappa;
	float fpre;
};

struct conduction_workspace {
	device_vector<conduction_record1> rec1;
	device_vector<conduction_record2> rec2;
};

__global__ void sph_cuda_conduction(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ conduction_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) conduction_workspace();
	const float gamma0 = data.def_gamma;
	const float m = data.m;
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {
		__syncthreads();
		int flops = 0;
		ws.rec1.resize(0);
		ws.rec2.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				float h;
				int rung;
				if (pi < other.part_range.second) {
					h = data.h[pi];
					ALWAYS_ASSERT(h > 0.f);
					rung = data.rungs[pi];
					if (rung >= params.min_rung) {
						x[XDIM] = data.x[pi];
						x[YDIM] = data.y[pi];
						x[ZDIM] = data.z[pi];
						contains = (self.outer_box.contains(x));
						if (!contains) {
							contains = true;
							for (int dim = 0; dim < NDIM; dim++) {
								if (distance(x[dim], self.inner_box.begin[dim]) + h < 0.f) {
									contains = false;
									break;
								}
								if (distance(self.inner_box.end[dim], x[dim]) + h < 0.f) {
									contains = false;
									break;
								}
							}
						}
					}
				}
				j = contains;
				compute_indices<CONDUCTION_BLOCK_SIZE>(j, total);
				const int offset = ws.rec1.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec1.resize(next_size);
				ws.rec2.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1[k].x = x[XDIM];
					ws.rec1[k].y = x[YDIM];
					ws.rec1[k].z = x[ZDIM];
					ALWAYS_ASSERT(h > 0.f);
					ws.rec1[k].h = h;
					ws.rec1[k].rung = data.rungs[pi];
					ws.rec2[k].entr = data.entr[pi];
					ws.rec2[k].kappa = data.kappa[pi];
					ws.rec2[k].fpre = data.fpre1[pi];
					if (params.stars) {
						ws.rec2[k].cfrac = data.cold_frac[pi];
						ALWAYS_ASSERT(ws.rec2[k].cfrac <= 1.f);
					} else {
						ws.rec2[k].cfrac = 0.f;
					}
				}
			}
		}
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs[i] >= params.min_rung;
			const bool semiactive = !active && data.sa_snk[snki];
			const bool converged = data.converged_snk[snki];
			if ((active || semiactive) && !converged) {
				const auto x_i = data.x[i];
				const auto y_i = data.y[i];
				const auto z_i = data.z[i];
				const float h_i = data.h[i];
				const float A_i = data.entr[i];
				float cfrac_i;
				if (params.stars) {
					cfrac_i = data.cold_frac[i];
				} else {
					cfrac_i = 0.f;
				}
				const float hfrac_i = 1.f - cfrac_i;
				const float fpre_i = data.fpre1[i];
				const float h2_i = sqr(h_i);
				const auto rung_i = data.rungs[i];
				const float kappa_i = data.kappa[i];
				const float c0 = float(3.0f / 4.0f / M_PI * data.N);
				const float hinv_i = 1.f / h_i;
				const float h3inv_i = (sqr(hinv_i) * hinv_i);
				const float rho_i = m * c0 * h3inv_i;
				const float ene_i = A_i * powf(rho_i * hfrac_i, gamma0 - 1.0f);
				const float dt_i = rung_dt[rung_i] * params.t0;
				float den = 0.f;
				float num = 0.f;
				float Aavg = 0.0f;
				ALWAYS_ASSERT(block_size==CONDUCTION_BLOCK_SIZE);
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					const auto& rec1 = ws.rec1[j];
					const auto& rec2 = ws.rec2[j];
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float h_j = rec1.h;
					ALWAYS_ASSERT(h_j > 0.f);
					const auto rung_j = rec1.rung;
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float h2_j = sqr(h_j);
					if (r2 > 0.f && (active || rung_j >= params.min_rung) && (r2 < fmaxf(h2_j, h2_i))) {
						const float A_j = rec2.entr;
						const float fpre_j = rec2.fpre;
						const float kappa_j = rec2.kappa;
						const float cfrac_j = rec2.cfrac;
						if (cfrac_j > 1.0f) {
							PRINT("%e\n", cfrac_j);
							__trap();
						}
						ALWAYS_ASSERT(cfrac_j <= 1.0f);
						const float hfrac_j = 1.f - cfrac_j;
						const float r = sqrtf(r2);
						const float rinv = 1.f / r;
						const float hinv_j = 1.f / h_j;
						const float h3inv_j = sqr(hinv_j) * hinv_j;
						const float rho_j = m * c0 * h3inv_j;													// 2
						const float q_i = r * hinv_i;
						const float q_j = r * hinv_j;
						const float dWdr_i = fpre_i * dkernelW_dq(q_i) * h3inv_i * hinv_i;
						const float dWdr_j = fpre_j * dkernelW_dq(q_j) * h3inv_j * hinv_j;
						const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);
						const float h_ij = 0.5f * (h_i + h_j);
						const float dWdr_rinv = r * dWdr_ij / (r2 + 0.01f * sqr(h_ij));
						const float kappa_ij = 2.f * kappa_i * kappa_j / (kappa_i + kappa_j + 1.0e-35f);
						const float dt_j = rung_dt[rung_j] * params.t0;
						const float dt_ij = fminf(dt_i, dt_j);
						const float D_ij = -2.f * m * kappa_ij * dWdr_rinv / (rho_i * rho_j) * dt_ij;
						num += D_ij * A_j * powf((rho_j * hfrac_j) / (rho_i * hfrac_i), gamma0 - 1.f);
						den += D_ij;
						Aavg += A_j * kernelW(q_i);
						if (!isfinite(num)) {
							PRINT("%e %e %e %e %e %e\n", A_j, rho_j, hfrac_j, kappa_i, kappa_j, kappa_ij);
						}
						ALWAYS_ASSERT(isfinite(den));
						ALWAYS_ASSERT(isfinite(num));
						ALWAYS_ASSERT(h_j > 0.0f);
						ALWAYS_ASSERT(isfinite(D_ij));
					}
				}
				shared_reduce_add<float, CONDUCTION_BLOCK_SIZE>(Aavg);
				shared_reduce_add<float, CONDUCTION_BLOCK_SIZE>(num);
				shared_reduce_add<float, CONDUCTION_BLOCK_SIZE>(den);
				if (tid == 0) {
					Aavg *= m / rho_i * h3inv_i;
					const float A0 = data.entr0_snk[snki];
					const float A1 = (A0 + num) / (1.f + den);
					const float dA = A1 - A_i;
					data.dentr_con_snk[snki] = dA;
					//		data.entr_avg_snk[snki] = Aavg;
//					ALWAYS_ASSERT(A1-A0==0.0);
					ALWAYS_ASSERT(isfinite(dA));
				}
			}
		}

		shared_reduce_add<int, CONDUCTION_BLOCK_SIZE>(flops);
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	(&ws)->~conduction_workspace();

}

__global__ void sph_cuda_cond_init(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ cond_init_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) cond_init_workspace();

	array<fixed32, NDIM> x;
	const float code_to_energy = sqr((double) params.code_to_cm) / sqr((double) params.code_to_s);
	const float code_to_density = (double) params.code_to_g / pow((double) params.code_to_cm, 3.);
	const float colog0 = log(1.5 * pow(constants::kb, 1.5) * pow(constants::e, -3) * pow(M_PI, -0.5));
	const float kappa0 = 20.0 * pow(2.0 / M_PI, 1.5) * pow(constants::kb, 2.5) * pow(constants::me, -0.5) * pow(constants::e, -4.0) * params.code_to_s
			* params.code_to_cm / (params.code_to_g * constants::avo);
	const float gamma0 = data.def_gamma;
	const float propc0 = 0.4 * (gamma0 - 1.0) * sqrtf(2.0 * constants::kb / M_PI / constants::me) / constants::c;
	while (index < data.nselfs) {
		__syncthreads();
		int flops = 0;
		ws.rec1.resize(0);
		ws.rec2.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				float h;
				int rung;
				if (pi < other.part_range.second) {
					h = data.h[pi];
					rung = data.rungs[pi];
					if (rung >= params.min_rung) {
						x[XDIM] = data.x[pi];
						x[YDIM] = data.y[pi];
						x[ZDIM] = data.z[pi];
						contains = (self.outer_box.contains(x));
						if (!contains) {
							contains = true;
							for (int dim = 0; dim < NDIM; dim++) {
								if (distance(x[dim], self.inner_box.begin[dim]) + h < 0.f) {
									contains = false;
									break;
								}
								if (distance(self.inner_box.end[dim], x[dim]) + h < 0.f) {
									contains = false;
									break;
								}
							}
						}
					}
				}
				j = contains;
				compute_indices<COND_INIT_BLOCK_SIZE>(j, total);
				const int offset = ws.rec1.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec1.resize(next_size);
				ws.rec2.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1[k].x = x[XDIM];
					ws.rec1[k].y = x[YDIM];
					ws.rec1[k].z = x[ZDIM];
					ws.rec1[k].h = h;
					ws.rec2[k].entr = data.entr[pi];
					if (params.stars) {
						ws.rec2[k].cfrac = data.cold_frac[pi];
					} else {
						ws.rec2[k].cfrac = 0.f;
					}
				}
			}
		}
		const float& m = data.m;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs[i] >= params.min_rung;
			int semiactive = 0;
			const float h_i = data.h[i];
			const float A_i = data.entr[i];
			float cfrac_i;
			if (params.stars) {
				cfrac_i = data.cold_frac[i];
			} else {
				cfrac_i = 0.f;
			}
			const float hfrac_i = 1.f - cfrac_i;
			const auto x_i = data.x[i];
			const auto y_i = data.y[i];
			const auto z_i = data.z[i];
			const float h2_i = sqr(h_i);
			if (active) {
				if (tid == 0) {
					data.sa_snk[snki] = true;
				}
			} else {
				const int jmax = round_up(ws.rec1.size(), block_size);
				if (tid == 0) {
					data.sa_snk[snki] = false;
				}
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.rec1.size()) {
						const auto x_j = ws.rec1[j].x;
						const auto y_j = ws.rec1[j].y;
						const auto z_j = ws.rec1[j].z;
						const auto h_j = ws.rec1[j].h;
						const auto h2_j = sqr(h_j);
						const float x_ij = distance(x_i, x_j);
						const float y_ij = distance(y_i, y_j);
						const float z_ij = distance(z_i, z_j);
						const float r2 = sqr(x_ij, y_ij, z_ij);
						if (r2 < fmaxf(h2_i, h2_j)) {
							semiactive++;
						}
					}
					shared_reduce_add<int, COND_INIT_BLOCK_SIZE>(semiactive);
					if (semiactive) {
						if (tid == 0) {
							data.sa_snk[snki] = true;
						}
						break;
					}
				}
			}
			if (semiactive || active) {
				const float c0 = float(3.0f / 4.0f / M_PI * data.N);
				const float hinv_i = 1.f / h_i;
				const float h3inv_i = (sqr(hinv_i) * hinv_i);
				const float rho_i = m * c0 * h3inv_i;
				const float ene_i = A_i * powf(rho_i * hfrac_i, gamma0 - 1.0f);
				float gradx = 0.0f;
				float grady = 0.0f;
				float gradz = 0.0f;
				const float fpre_i = data.rec1_snk[snki].fpre1;
				for (int j = tid; j < ws.rec1.size(); j += COND_INIT_BLOCK_SIZE) {
					const auto& rec1 = ws.rec1[j];
					const auto& rec2 = ws.rec2[j];
					const float A_j = rec2.entr;
					const float cfrac_j = rec2.cfrac;
					ALWAYS_ASSERT(cfrac_j <= 1.0f);
					const float hfrac_j = 1.f - cfrac_j;
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float r2 = sqr(x_ij, y_ij, z_ij);
					if (r2 < h2_i && r2 != 0.0f) {
						const float r = sqrtf(r2);
						const float rinv = 1.f / r;
						const float q = r * hinv_i;
						const float h_j = rec1.h;
						const float hinv_j = 1.f / h_j;
						const float h3inv_j = sqr(hinv_j) * hinv_j;
						const float rho_j = m * c0 * h3inv_j;													// 2
						const float ene_j = A_j * powf(rho_j * hfrac_j, gamma0 - 1.0f);
						const float dwdq = dkernelW_dq(q);
						const float dWdr_i = fpre_i * dwdq * h3inv_i * hinv_i;
						gradx += logf(ene_j / ene_i) * dWdr_i * x_ij * rinv;
						grady += logf(ene_j / ene_i) * dWdr_i * y_ij * rinv;
						gradz += logf(ene_j / ene_i) * dWdr_i * z_ij * rinv;
					}

				}

				gradx *= m / (params.a * rho_i);
				grady *= m / (params.a * rho_i);
				gradz *= m / (params.a * rho_i);
				shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(gradx);
				shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(grady);
				shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(gradz);
				if (tid == 0) {
					const float grad2 = sqr(gradx, grady, gradz);
					const float gradToT = sqrtf(grad2);
					const auto& frac_i = data.rec4_snk[snki].frac;
					const float& cfrac_i = data.rec2_snk[snki].fcold;
					const float& A_i = data.rec2_snk[snki].A;
					const float& H = frac_i[CHEM_H];
					const float& Hp = frac_i[CHEM_HP];
					const float& Hn = frac_i[CHEM_HN];
					const float& H2 = frac_i[CHEM_H2];
					const float& He = frac_i[CHEM_HE];
					const float& Hep = frac_i[CHEM_HEP];
					const float& Hepp = frac_i[CHEM_HEPP];
					const float hfrac_i = 1.f - cfrac_i;
					const float rho0 = rho_i * hfrac_i / (sqr(params.a) * params.a);
					float n0 = (H + 2.f * Hp + .5f * H2 + .25f * He + .5f * Hep + .75f * Hepp);
					const float mmw_i = 1.0f / n0;
					n0 *= constants::avo * rho0;
					const float ne_i = (Hp - Hn + 0.25f * Hep + 0.5f * Hepp) * rho0 * (constants::avo * code_to_density);
					const float cv0 = constants::kb / (gamma0 - 1.0);															// 4
					const float eint = code_to_energy * A_i * powf(rho0 * hfrac_i, gamma0 - 1.0) / (gamma0 - 1.0);
					const float T_i = rho0 * eint / (n0 * cv0);
					const float colog_i = colog0 + 1.5f * logf(T_i) - 0.5f * logf(ne_i);
					float kappa_i = (gamma0 - 1.f) * kappa0 * powf(T_i, 2.5f) / colog_i;
					const float sigmax_i = propc0 * sqrtf(T_i);
					const float R = mmw_i * kappa_i * gradToT / (rho_i * sigmax_i);
					const float phi = (2.f + 3.f * R) / (2.f + 3.f * R + 3.f * sqr(R));
					kappa_i *= phi;
					data.kap_snk[snki] = kappa_i;
					data.entr0_snk[snki] = A_i;
				}
			}

		}
		shared_reduce_add<int, COND_INIT_BLOCK_SIZE>(flops);
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	(&ws)->~cond_init_workspace();

}

sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream) {
	timer tm;
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
	static int smoothlen_nblocks;
	static int aux_nblocks;
	static int hydro_nblocks;
	static int rungs_nblocks;
	static int cond_init_nblocks;
	static int conduction_nblocks;
	static bool first = true;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&aux_nblocks, (const void*) sph_cuda_aux, SMOOTHLEN_BLOCK_SIZE, 0));
		aux_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&smoothlen_nblocks, (const void*) sph_cuda_smoothlen, SMOOTHLEN_BLOCK_SIZE, 0));
		smoothlen_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&hydro_nblocks, (const void*) sph_cuda_hydro, HYDRO_BLOCK_SIZE, 0));
		hydro_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&conduction_nblocks, (const void*) sph_cuda_conduction, CONDUCTION_BLOCK_SIZE, 0));
		conduction_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cond_init_nblocks, (const void*) sph_cuda_cond_init, COND_INIT_BLOCK_SIZE, 0));
		cond_init_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&rungs_nblocks, (const void*) sph_cuda_rungs, RUNGS_BLOCK_SIZE, 0));
		rungs_nblocks *= cuda_smp_count();
	}
	switch (params.run_type) {
	case SPH_RUN_SMOOTHLEN: {
		sph_cuda_smoothlen<<<smoothlen_nblocks, SMOOTHLEN_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		rc.rc = reduce->flag;
		rc.hmin = reduce->hmin;
		rc.hmax = reduce->hmax;
	}
	break;
	case SPH_RUN_AUX: {
		sph_cuda_aux<<<aux_nblocks, AUX_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
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
		timer tm;
		tm.start();
		sph_cuda_rungs<<<hydro_nblocks, RUNGS_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		tm.stop();
		auto gflops = reduce->flops / tm.read() / (1024.0*1024*1024);
		rc.max_vsig = reduce->vsig_max;
		rc.max_rung_grav = reduce->max_rung_grav;
		rc.max_rung_hydro = reduce->max_rung_hydro;
		rc.max_rung = reduce->max_rung;
		rc.rc = reduce->flag;
	}
	break;
	case SPH_RUN_HYDRO: {
		timer tm;
		tm.start();
		sph_cuda_hydro<<<hydro_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		tm.stop();
		rc.max_vsig = reduce->vsig_max;
		rc.max_rung_grav = reduce->max_rung_grav;
		rc.max_rung_hydro = reduce->max_rung_hydro;
		rc.max_rung = reduce->max_rung;
	}
	break;
}
	(cudaFree(reduce));
	return rc;
}
