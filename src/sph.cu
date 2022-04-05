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
#define SMOOTHLEN_BLOCK_SIZE 256
#define MARK_SEMIACTIVE_BLOCK_SIZE 256
#define RUNGS_BLOCK_SIZE 256
#define XSPH_BLOCK_SIZE 256
#define HYDRO_BLOCK_SIZE 128
#define AUX_BLOCK_SIZE 128

#define SPH_SMOOTHLEN_TOLER float(1.0e-5)

struct smoothlen_shmem {
	int index;
};

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

struct smoothlen_workspace {
	device_vector<fixed32> x;
	device_vector<fixed32> y;
	device_vector<fixed32> z;
};

struct mark_semiactive_workspace {
	device_vector<fixed32> x;
	device_vector<fixed32> y;
	device_vector<fixed32> z;
	device_vector<float> h;
	device_vector<char> rungs;
};

struct rungs_workspace {
	device_vector<fixed32> x;
	device_vector<fixed32> y;
	device_vector<fixed32> z;
	device_vector<float> h;
	device_vector<char> rungs;
};

struct xsph_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
};

struct xsph_record2 {
	float vx;
	float vy;
	float vz;
};

struct hydro_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};

struct hydro_record2 {
	array<float, NCHEMFRACS> chem;
	float shearv;
	float vx;
	float vy;
	float vz;
	float entr;
	float alpha;
	float fpre;
	float cold_frac;
	char rung;
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
	float entr;
	float gamma;
	float h;
};

struct hydro_workspace {
	device_vector<hydro_record1> rec1;
	device_vector<hydro_record2> rec2;
};

struct xsph_workspace {
	device_vector<xsph_record1> rec1;
	device_vector<xsph_record2> rec2;
};

struct aux_workspace {
	device_vector<aux_record1> rec1;
	device_vector<aux_record2> rec2;
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

__global__ void sph_cuda_xsph(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ xsph_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) xsph_workspace();
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {

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
				if (pi < other.part_range.second) {
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					contains = (self.outer_box.contains(x));
				}
				j = contains;
				compute_indices<XSPH_BLOCK_SIZE>(j, total);
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
					ws.rec2[k].vx = data.vx[pi];
					ws.rec2[k].vy = data.vy[pi];
					ws.rec2[k].vz = data.vz[pi];
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs[i] >= params.min_rung;
			const bool use = active;
			if (use) {
				const auto x_i = data.x[i];
				const auto y_i = data.y[i];
				const auto z_i = data.z[i];
				const float h_i = data.h_snk[snki];
				const float h2_i = sqr(h_i);
				const auto vx_i = data.vx[i];
				const auto vy_i = data.vy[i];
				const auto vz_i = data.vz[i];
				const float hinv_i = 1.f / h_i;
				float xvx = 0.0f;
				float xvy = 0.0f;
				float xvz = 0.0f;
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					const auto x_j = ws.rec1[j].x;
					const auto y_j = ws.rec1[j].y;
					const auto z_j = ws.rec1[j].z;
					const float x_ij = distance(x_i, x_j); // 2
					const float y_ij = distance(y_i, y_j); // 2
					const float z_ij = distance(z_i, z_j); // 2
					const float r2 = sqr(x_ij, y_ij, z_ij);            // 2
					if (r2 < h2_i) {
						const float r = sqrt(r2);                    // 4
						const auto& vx_j = ws.rec2[j].vx;
						const auto& vy_j = ws.rec2[j].vy;
						const auto& vz_j = ws.rec2[j].vz;
						const float q_i = r * hinv_i;
						const float W_i = kernelW(q_i);
						xvx = fmaf(W_i, vx_j - vx_i, xvx);
						xvy = fmaf(W_i, vy_j - vy_i, xvy);
						xvz = fmaf(W_i, vz_j - vz_i, xvz);
					}
				}
				shared_reduce_add<float, XSPH_BLOCK_SIZE>(xvx);
				shared_reduce_add<float, XSPH_BLOCK_SIZE>(xvy);
				shared_reduce_add<float, XSPH_BLOCK_SIZE>(xvz);
				if (tid == 0) {
					const float c0 = float(4.0f * M_PI / 3.0f) / data.N;
					xvx *= c0;
					xvy *= c0;
					xvz *= c0;
					data.xvx_snk[snki] = xvx;
					data.xvy_snk[snki] = xvy;
					data.xvz_snk[snki] = xvz;
				}
			}
		}
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	(&ws)->~xsph_workspace();
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
		ws.x.resize(0);
		ws.y.resize(0);
		ws.z.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
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
				}
				j = contains;
				compute_indices<SMOOTHLEN_BLOCK_SIZE>(j, total);
				const int offset = ws.x.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				ws.y.resize(next_size);
				ws.z.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x[k] = x[XDIM];
					ws.y[k] = x[YDIM];
					ws.z[k] = x[ZDIM];
				}
			}
		}
		float hmin = 1e+20;
		float hmax = 0.0;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs[i] >= params.min_rung;
			const bool converged = data.converged_snk[snki];
			const bool use = active && !converged;
			const float w0 = kernelW(0.f);
			if (use) {
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				int box_xceeded = false;
				int iter = 0;
				float& h = data.h_snk[snki];
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
						const float dx = distance(x[XDIM], ws.x[j]); // 2
						const float dy = distance(x[YDIM], ws.y[j]); // 2
						const float dz = distance(x[ZDIM], ws.z[j]); // 2
						const float r2 = sqr(dx, dy, dz);            // 2
						const float r = sqrt(r2);                    // 4
						const float q = r * hinv;                    // 1
						if (q < 1.f) {                               // 1
							const float w = kernelW(q); // 4
							const float dwdq = dkernelW_dq(q);
							const float dwdh = -q * dwdq * hinv; // 3
							drho_dh -= (3.f * w + q * dwdq);
							rhoh3 += w;
						}

					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(drho_dh);
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(rhoh3);
					float dlogh;
					__syncthreads();
					if (rhoh3 <= w0) {
						if (tid == 0) {
							h *= 1.1f;
						}
						iter--;
						error = 1.0;
					} else {
						drho_dh *= 0.33333333333f / rhoh30;
						const float fpre = fminf(fmaxf(1.0f / (1.0f + drho_dh), 0.5f), 2.0f);
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
					for (int j = tid; j < ws.x.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.x[j]); // 2
						const float dy = distance(x[YDIM], ws.y[j]); // 2
						const float dz = distance(x[ZDIM], ws.z[j]); // 2
						const float r2 = sqr(dx, dy, dz);            // 2
						const float r = sqrt(r2);                    // 4
						const float q = r * hinv;                    // 1
						if (q < 1.f) {                               // 1
							const float w = kernelW(q); // 4
							const float dwdq = dkernelW_dq(q);
							drho_dh -= (3.f * w + q * dwdq);
						}
					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(drho_dh);
					drho_dh *= 0.33333333333f / rhoh30;
					const float fpre = 1.0f / (1.0f + drho_dh);
					if (tid == 0) {
						data.fpre_snk[snki] = fpre;
						//					data.converged_snk[snki] = true;
					}
					hmin = fminf(hmin, h);
					hmax = fmaxf(hmax, h);
				} else {
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
					}
				}
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

__global__ void sph_cuda_mark_semiactive(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ mark_semiactive_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) mark_semiactive_workspace();

	array<fixed32, NDIM> x;
	while (index < data.nselfs) {
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
				compute_indices<MARK_SEMIACTIVE_BLOCK_SIZE>(j, total);
				const int offset = ws.x.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				ws.y.resize(next_size);
				ws.z.resize(next_size);
				ws.h.resize(next_size);
				ws.rungs.resize(next_size);
				if (contains) {
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
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			if (data.rungs[i] >= params.min_rung) {
				if (tid == 0) {
					data.sa_snk[snki] = true;
				}
			} else {
				const auto x0 = data.x[i];
				const auto y0 = data.y[i];
				const auto z0 = data.z[i];
				const auto h0 = data.h[i];
				const auto h02 = sqr(h0);
				int semiactive = 0;
				const int jmax = round_up(ws.x.size(), block_size);
				if (tid == 0) {
					data.sa_snk[snki] = false;
				}
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.x.size()) {
						const auto x1 = ws.x[j];
						const auto y1 = ws.y[j];
						const auto z1 = ws.z[j];
						const auto h1 = ws.h[j];
						const auto h12 = sqr(h1);
						const float dx = distance(x0, x1);
						const float dy = distance(y0, y1);
						const float dz = distance(z0, z1);
						const float r2 = sqr(dx, dy, dz);
						if (r2 < fmaxf(h02, h12)) {
							semiactive++;
						}
					}
					shared_reduce_add<int, MARK_SEMIACTIVE_BLOCK_SIZE>(semiactive);
					if (semiactive) {
						if (tid == 0) {
							data.sa_snk[snki] = true;
						}
						break;
					}
				}
			}
		}
		shared_reduce_add<int, MARK_SEMIACTIVE_BLOCK_SIZE>(flops);
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	(&ws)->~mark_semiactive_workspace();

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
		const sph_tree_node& self = data.trees[data.selfs[index]];
		ws.rec1.resize(0);
		ws.rec2.resize(0);
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
					ws.rec1[k].h = data.h[pi];
					ws.rec2[k].vx = data.vx[pi];
					ws.rec2[k].vy = data.vy[pi];
					ws.rec2[k].vz = data.vz[pi];
					ws.rec2[k].entr = data.entr[pi];
					ws.rec2[k].alpha = data.alpha[pi];
					ws.rec2[k].fpre = data.fpre[pi];
					ws.rec2[k].rung = data.rungs[pi];
					if (params.stars) {
						ws.rec2[k].cold_frac = data.cold_frac[pi];
					} else {
						ws.rec2[k].cold_frac = 0.f;
					}
					if (params.diffusion) {
						ws.rec2[k].shearv = data.shearv[pi];
						if (data.chemistry) {
							ws.rec2[k].chem = data.chem[pi];
						}
					}
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			int rung_i = data.rungs[i];
			bool use = rung_i >= params.min_rung;
			const int snki = self.sink_part_range.first - self.part_range.first + i;
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
				const float K_i = data.entr[i];
				const float alpha_i = data.alpha[i];
				const float gamma0 = data.def_gamma;
				const float p_i = K_i * powf(rho_i, gamma0);
				float cfrac_i;
				if (params.stars) {
					cfrac_i = data.cold_frac[i];
				} else {
					cfrac_i = 0.f;
				}
				const float hfrac_i = 1.f - cfrac_i;
				const float c_i = sqrtf(gamma0 * p_i * rhoinv_i);
				const float fpre_i = data.fpre[i];
				//	float shearv_i;
				//	array<float, NCHEMFRACS> frac_i;
				/*	if (params.diffusion) {
				 shearv_i = data.shearv[i];
				 if (data.chemistry) {
				 frac_i = data.chem[i];
				 }
				 }*/
				float ax = 0.f;
				float ay = 0.f;
				float az = 0.f;
				float de_dt = 0.f;
				float dcm_dt = 0.f;
				//	array<float, NCHEMFRACS> dfrac_dt;
				/*		if (params.diffusion && data.chemistry) {
				 for (int fi = 0; fi < NCHEMFRACS; fi++) {
				 dfrac_dt[fi] = 0.f;
				 }
				 }*/
				float dtinv_cfl = 0.f;
				float one = 0.0f;
				constexpr float tiny = 1e-30f;
				float vsig = 0.0f;
				const float& adot = params.adot;
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
						const float vx_j = rec2.vx;
						const float vy_j = rec2.vy;
						const float vz_j = rec2.vz;
						const float fpre_j = rec2.fpre;
						const float h2_j = sqr(h_j);
						const float h3inv_j = sqr(hinv_j) * hinv_j;
						const float rho_j = m * c0 * h3inv_j;													// 2
						const float rhoinv_j = minv * c0inv * sqr(h_j) * h_j;								// 5
						const float K_j = rec2.entr;
						const float p_j = K_j * powf(rho_j, gamma0);
						const float c_j = sqrtf(gamma0 * p_j * rhoinv_j);									// 6
						const float alpha_j = rec2.alpha;
						const float vx0_ij = vx_i - vx_j;
						const float vy0_ij = vy_i - vy_j;
						const float vz0_ij = vz_i - vz_j;
						const float vx_ij = vx0_ij + x_ij * adot;
						const float vy_ij = vy0_ij + y_ij * adot;
						const float vz_ij = vz0_ij + z_ij * adot;
						const float rinv = 1.0f / (r + tiny);
						const float vdotx_ij = fminf(0.0f, x_ij * vx_ij + y_ij * vy_ij + z_ij * vz_ij);
						const float h_ij = 0.5f * (h_i + h_j);
						const float mu_ij = vdotx_ij * h_ij / (r2 + 0.01f * sqr(h_ij));
						const float rho_ij = 0.5f * (rho_i + rho_j);
						const float c_ij = 0.5f * (c_i + c_j);
						const float alpha_ij = 0.5f * (alpha_i + alpha_j);
						const float vsig_ij = alpha_ij * (c_ij - params.beta * mu_ij);
						const float pi_ij = -mu_ij * vsig_ij / rho_ij;
						const float dWdr_i = fpre_i * dkernelW_dq(q_i) * hinv_i * h3inv_i;
						const float dWdr_j = fpre_j * dkernelW_dq(q_j) * hinv_j * h3inv_j;
						const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);
						const float dWdr_x_ij = x_ij * rinv * dWdr_ij;
						const float dWdr_y_ij = y_ij * rinv * dWdr_ij;
						const float dWdr_z_ij = z_ij * rinv * dWdr_ij;
						const float dWdr_x_i = x_ij * rinv * dWdr_i;
						const float dWdr_y_i = y_ij * rinv * dWdr_i;
						const float dWdr_z_i = z_ij * rinv * dWdr_i;
						const float dWdr_x_j = x_ij * rinv * dWdr_j;
						const float dWdr_y_j = y_ij * rinv * dWdr_j;
						const float dWdr_z_j = z_ij * rinv * dWdr_j;
						const float dp_i = p_i * sqr(rhoinv_i);
						const float dp_j = p_j * sqr(rhoinv_j);
						one += m / rho_i * kernelW(q_i) * h3inv_i;
						ax -= m * ainv * (dp_i * dWdr_x_i + dp_j * dWdr_x_j);
						ay -= m * ainv * (dp_i * dWdr_y_i + dp_j * dWdr_y_j);
						az -= m * ainv * (dp_i * dWdr_z_i + dp_j * dWdr_z_j);
						ax -= m * ainv * (pi_ij * dWdr_x_ij);
						ay -= m * ainv * (pi_ij * dWdr_y_ij);
						az -= m * ainv * (pi_ij * dWdr_z_ij);
						de_dt += (gamma0 - 1.f) * powf(rho_i, 1.f - gamma0) * 0.5f * m * ainv * pi_ij * (vx_ij * dWdr_x_ij + vy_ij * dWdr_y_ij + vz_ij * dWdr_z_ij);
						if (params.phase == 1 || params.damping > 0.f) {
							vsig = fmaxf(vsig, vsig_ij);
						}
						if (params.phase == 1) {
							float dtinv = (c_ij + 0.6f * vsig_ij) / fminf(h_i, 2.f * h_j);
							dtinv_cfl = fmaxf(dtinv_cfl, dtinv);
						}
					}
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(de_dt);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ax);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ay);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(az);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(one);
				if (params.phase == 1) {
					shared_reduce_max<HYDRO_BLOCK_SIZE>(dtinv_cfl);
					shared_reduce_max<HYDRO_BLOCK_SIZE>(vsig);
				}
				if (params.stars) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dcm_dt);
				}

				//		ALWAYS_ASSERT(data.converged_snk[snki] == 0);
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
					data.dvx_con[snki] = ax;
					data.dvy_con[snki] = ay;
					data.dvz_con[snki] = az;
					data.dentr_con[snki] = de_dt;
					if (params.diffusion && data.chemistry) {
						for (int fi = 0; fi < NCHEMFRACS; fi++) {
							data.dchem_con[snki][fi] = 0.f;
						}
					}
					if (params.phase == 1) {
						const float divv = data.divv_snk[snki];
						//				const float dtinv_divv = params.a * fabsf(divv - 3.f * params.adot * ainv) * (1.f / 3.f);
						float dtinv_hydro1 = 1.0e-30f;
						//			dtinv_hydro1 = fmaxf(dtinv_hydro1, dtinv_divv);
						dtinv_hydro1 = fmaxf(dtinv_hydro1, dtinv_cfl);
						const float a2_1 = sqr(ax, ay, az);
						const float a2_2 = sqr(ax - gx_i, ay - gy_i, az - gz_i);
						const float a2 = fminf(a2_1, a2_2);
						const float dtinv_acc = sqrtf(sqrtf(a2) * hinv_i);
						const float dtinv_hydro2 = dtinv_acc;
						float dthydro = params.cfl * params.a / (dtinv_hydro1 + 1e-30f);
						dthydro = fminf(data.eta * sqrtf(params.a) / (dtinv_hydro2 + 1e-30f), dthydro);
						const float g2 = sqr(gx_i, gy_i, gz_i);
						const float dtinv_grav = sqrtf(sqrtf(g2) * hinv_i);
						float dtgrav = data.eta * sqrtf(params.a) / (dtinv_grav + 1e-30f);
						dthydro = fminf(dthydro, params.max_dt);
						dtgrav = fminf(dtgrav, params.max_dt);
						total_vsig_max = fmaxf(total_vsig_max, dtinv_hydro1 * h_i);
						char& rung = data.rungs[i];
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
					}
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
	if (tid == 0 && params.phase == 1) {
		atomicMax(&reduce->vsig_max, total_vsig_max);
		atomicMax(&reduce->max_rung, max_rung);
		atomicMax(&reduce->max_rung_hydro, max_rung_hydro);
		atomicMax(&reduce->max_rung_grav, max_rung_grav);
	}
	(&ws)->~hydro_workspace();
}

__global__ void sph_cuda_aux(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ aux_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	int flops = 0;
	new (&ws) aux_workspace();

	const double kb = (double) constants::kb * sqr((double) params.code_to_s) / ((double) params.code_to_g * sqr((double) params.code_to_cm));
	const double mh = (double) constants::mh / (double) params.code_to_g;
	const float cv0 = kb / mh;
	const float ainv = 1.0f / params.a;
	while (index < data.nselfs) {
		const sph_tree_node& self = data.trees[data.selfs[index]];
		ws.rec1.resize(0);
		ws.rec2.resize(0);
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
					contains = self.outer_box.contains(x);
				}
				j = contains;
				compute_indices<AUX_BLOCK_SIZE>(j, total);
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
					ws.rec2[k].vx = data.vx[pi];
					ws.rec2[k].vy = data.vy[pi];
					ws.rec2[k].vz = data.vz[pi];
					if (data.chemistry) {
						ws.rec2[k].gamma = data.gamma[pi];
					} else {
						ws.rec2[k].gamma = data.def_gamma;
					}
					ws.rec2[k].h = data.h[pi];
					ws.rec2[k].entr = data.entr[pi];
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			int rung_i = data.rungs[i];
			const bool active = rung_i >= params.min_rung;
			const bool use = active;
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
				float h_i;
				h_i = data.h[i];
				const float h2_i = sqr(h_i);
				const float hinv_i = 1.f / h_i;
				const float h3inv_i = (sqr(hinv_i) * hinv_i);
				const float rho_i = m * c0 * h3inv_i;
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;
				const float fpre_i = data.fpre_snk[snki];
				float K_i, p_i, c_i;
				const float gamma0 = data.def_gamma;
				K_i = data.entr[i];
				p_i = K_i * powf(rho_i, gamma0);
				c_i = sqrtf(gamma0 * p_i * rhoinv_i);
				float dvx_dx = 0.0f;
				float dvx_dy = 0.0f;
				float dvx_dz = 0.0f;
				float dvy_dx = 0.0f;
				float dvy_dy = 0.0f;
				float dvy_dz = 0.0f;
				float dvz_dx = 0.0f;
				float dvz_dy = 0.0f;
				float dvz_dz = 0.0f;
				float vsig = 0.f;
				const float adot = params.adot;
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					const auto rec1 = ws.rec1[j];
					const auto rec2 = ws.rec2[j];
					const float vx_j = rec2.vx;
					const float vy_j = rec2.vy;
					const float vz_j = rec2.vz;
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float vx0_ij = vx_i - vx_j;
					const float vy0_ij = vy_i - vy_j;
					const float vz0_ij = vz_i - vz_j;
					const float vx_ij = vx0_ij + x_ij * adot;
					const float vy_ij = vy0_ij + y_ij * adot;
					const float vz_ij = vz0_ij + z_ij * adot;
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);
					const float rinv = 1.0f / (1.0e-30f + r);
					const float q_i = r * hinv_i;
					const float dWdr_i = fpre_i * dkernelW_dq(q_i) * hinv_i * h3inv_i;
					const float dWdr_x_i = dWdr_i * rinv * x_ij;
					const float dWdr_y_i = dWdr_i * rinv * y_ij;
					const float dWdr_z_i = dWdr_i * rinv * z_ij;
					const float vdotx_ij = fminf(vx_ij * x_ij + vy_ij * y_ij + vz_ij * z_ij, 0.f);
					const float h_j = rec2.h;
					const float hinv_j = 1.f / h_j;
					const float h3inv_j = (sqr(hinv_j) * hinv_j);
					const float rho_j = m * c0 * h3inv_j;
					const float gamma_j = rec2.gamma;
					const float K_j = rec2.entr;
					const float p_j = K_j * powf(rho_j, gamma0);
					const float c_j = sqrtf(gamma_j * p_j / rho_j);
					const float w_ij = vdotx_ij * rinv;
					const float vsig_i = 0.5f * (c_i + c_j) - w_ij;
					dvx_dx -= vx_ij * dWdr_x_i;
					dvy_dx -= vy_ij * dWdr_x_i;
					dvz_dx -= vz_ij * dWdr_x_i;
					dvx_dy -= vx_ij * dWdr_y_i;
					dvy_dy -= vy_ij * dWdr_y_i;
					dvz_dy -= vz_ij * dWdr_y_i;
					dvx_dz -= vx_ij * dWdr_z_i;
					dvy_dz -= vy_ij * dWdr_z_i;
					dvz_dz -= vz_ij * dWdr_z_i;
					vsig = fmaxf(vsig, vsig_i);
				}
				const float mrhoinvainv_i = m * rhoinv_i * ainv;
				dvx_dx *= mrhoinvainv_i;
				dvx_dy *= mrhoinvainv_i;
				dvx_dz *= mrhoinvainv_i;
				dvy_dx *= mrhoinvainv_i;
				dvy_dy *= mrhoinvainv_i;
				dvy_dz *= mrhoinvainv_i;
				dvz_dx *= mrhoinvainv_i;
				dvz_dy *= mrhoinvainv_i;
				dvz_dz *= mrhoinvainv_i;
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
				shared_reduce_add<float, AUX_BLOCK_SIZE>(shear_xx);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(shear_xy);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(shear_xz);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(shear_yy);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(shear_yz);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(shear_zz);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(div_v);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(curl_vx);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(curl_vy);
				shared_reduce_add<float, AUX_BLOCK_SIZE>(curl_vz);
				shared_reduce_max<AUX_BLOCK_SIZE>(vsig);
				if (tid == 0) {
					const float shearv = sqrtf(sqr(shear_xx) + sqr(shear_yy) + sqr(shear_zz) + 2.0f * (sqr(shear_xy) + sqr(shear_yz) + sqr(shear_xz)));
					const float curlv = sqrtf(sqr(curl_vx, curl_vy, curl_vz));
					data.divv_snk[snki] = div_v;
					data.shearv_snk[snki] = shearv;
					/* float& alpha = data.alpha_snk[snki];
					 const float dt = params.t0 * rung_dt[rung_i];
					 const float limiter = fabs(div_v) / (fabs(div_v) + curlv + 1e-30f);
					 const float S = fmaxf(0.f, -div_v) * limiter;
					 const float tauinv = params.alpha_decay * c_i * hinv_i * ainv;
					 alpha = (alpha + (params.alpha1 * S + params.alpha0 * tauinv) * dt) / (1.f + dt * (params.alpha0 * S + tauinv));*/
					float& alpha = data.alpha_snk[snki];
					const float divv0 = params.tau > 0.f ? data.divv0_snk[snki] : div_v;
					data.divv0_snk[snki] = div_v;
					if (params.tau > 0.f) {
						const float dt = params.t0 * rung_dt[rung_i];
						const float ddivv_dt = (div_v - divv0) / dt - 0.5f * params.adot * ainv * (div_v + divv0);
						const float S = sqr(h_i) * fmaxf(0.f, -ddivv_dt) * sqr(params.a);
						const float limiter = sqr(div_v) / (sqr(div_v) + sqr(curlv) + 1.0e-4f * sqr(c_i / h_i * ainv));
						const float alpha_targ = S / (S + sqr(c_i));
						const float lambda0 = params.alpha_decay * vsig * hinv_i * ainv * dt;
						if (alpha < limiter * alpha_targ) {
							alpha = limiter * alpha_targ;
						} else {
							alpha = (alpha + lambda0 * limiter * alpha_targ) / (1.f + lambda0);
						}
					} else {
						alpha = 0.f;
					}

					//				float& alpha = data.alpha_snk[snki];
					//				alpha = 0.75f;
				}
			}
		}
		shared_reduce_add<int, AUX_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (float) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~aux_workspace();
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
					if (rung_i < max_rung_j - 1) {
						changes++;
						rung_i = max_rung_j - 1;
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
	static int semiactive_nblocks;
	static int hydro_nblocks;
	static int aux_nblocks;
	static int xsph_nblocks;
	static int rungs_nblocks;
	static bool first = true;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&smoothlen_nblocks, (const void*) sph_cuda_smoothlen, SMOOTHLEN_BLOCK_SIZE, 0));
		smoothlen_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&semiactive_nblocks, (const void*) sph_cuda_mark_semiactive, MARK_SEMIACTIVE_BLOCK_SIZE, 0));
		semiactive_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&hydro_nblocks, (const void*) sph_cuda_hydro, HYDRO_BLOCK_SIZE, 0));
		hydro_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&aux_nblocks, (const void*) sph_cuda_aux, AUX_BLOCK_SIZE, 0));
		aux_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&xsph_nblocks, (const void*) sph_cuda_xsph, XSPH_BLOCK_SIZE, 0));
		xsph_nblocks *= cuda_smp_count();
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
	case SPH_RUN_MARK_SEMIACTIVE: {
		sph_cuda_mark_semiactive<<<semiactive_nblocks, MARK_SEMIACTIVE_BLOCK_SIZE,0,stream>>>(params,data,reduce);
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
		sph_cuda_hydro<<<hydro_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		rc.max_vsig = reduce->vsig_max;
		rc.max_rung_grav = reduce->max_rung_grav;
		rc.max_rung_hydro = reduce->max_rung_hydro;
		rc.max_rung = reduce->max_rung;
	}
	break;
	case SPH_RUN_AUX: {
		sph_cuda_aux<<<aux_nblocks, AUX_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
	}
	break;
	case SPH_RUN_XSPH: {
		timer tm;
		tm.start();
		sph_cuda_xsph<<<xsph_nblocks, XSPH_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
		tm.stop();
		PRINT( "XSPH time = %e\n", tm.read());
	}
	break;
}
	(cudaFree(reduce));
	return rc;
}
