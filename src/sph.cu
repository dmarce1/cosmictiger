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

struct smoothlen_shmem {
	int index;
};

#include <cosmictiger/sph_cuda.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/kernel.hpp>

static __constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6),
		1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
				/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24),
		1.0 / (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

#define WORKSPACE_SIZE (160*1024)
#define HYDRO_SIZE (16*1024)

struct smoothlen_workspace {
	fixedcapvec<fixed32, WORKSPACE_SIZE> x;
	fixedcapvec<fixed32, WORKSPACE_SIZE> y;
	fixedcapvec<fixed32, WORKSPACE_SIZE> z;
};

struct mark_semiactive_workspace {
	fixedcapvec<fixed32, WORKSPACE_SIZE + 1> x;
	fixedcapvec<fixed32, WORKSPACE_SIZE + 1> y;
	fixedcapvec<fixed32, WORKSPACE_SIZE + 1> z;
	fixedcapvec<float, WORKSPACE_SIZE + 1> h;
	fixedcapvec<char, WORKSPACE_SIZE + 1> rungs;
};

struct hydro_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
	char rung;
};

struct hydro_record2 {
	float gamma;
	float vx;
	float vy;
	float vz;
	float ent;
	float alpha;
	float f0;
	float fvel;
	float gx;
	float gy;
	float gz;
};

struct dif_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
	char rung;
};

struct dif_record2 {
	dif_vector vec;
	float gamma;
	float difco;
	float kappa;
	float mmw;
	char oldrung;
};

struct hydro_workspace {
	fixedcapvec<hydro_record1, WORKSPACE_SIZE + 2> rec1_main;
	fixedcapvec<hydro_record2, WORKSPACE_SIZE + 2> rec2_main;
	fixedcapvec<hydro_record1, HYDRO_SIZE + 2> rec1;
	fixedcapvec<hydro_record2, HYDRO_SIZE + 2> rec2;
};

struct dif_workspace {
	fixedcapvec<dif_record1, WORKSPACE_SIZE + 2> rec1_main;
	fixedcapvec<dif_record2, WORKSPACE_SIZE + 2> rec2_main;
	fixedcapvec<dif_record1, HYDRO_SIZE + 2> rec1;
	fixedcapvec<dif_record2, HYDRO_SIZE + 2> rec2;
};

struct deposit_workspace {
	fixedcapvec<float, WORKSPACE_SIZE + 2> sn;
	fixedcapvec<fixed32, WORKSPACE_SIZE + 2> x;
	fixedcapvec<fixed32, WORKSPACE_SIZE + 2> y;
	fixedcapvec<fixed32, WORKSPACE_SIZE + 2> z;
	fixedcapvec<float, WORKSPACE_SIZE + 2> vx;
	fixedcapvec<float, WORKSPACE_SIZE + 2> vy;
	fixedcapvec<float, WORKSPACE_SIZE + 2> vz;
	fixedcapvec<float, WORKSPACE_SIZE + 2> h;
};

struct courant_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};

struct courant_record2 {
//	float Y;
//	float Z;
	float gamma;
	float vx;
	float vy;
	float vz;
	float gx;
	float gy;
	float gz;
	float ent;
	float T;
	float lambda_e;
	float mmw;
	float alpha;
};

struct courant_workspace {
	fixedcapvec<courant_record1, WORKSPACE_SIZE + 3> rec1_main;
	fixedcapvec<courant_record2, WORKSPACE_SIZE + 3> rec2_main;
	fixedcapvec<courant_record1, HYDRO_SIZE + 3> rec1;
	fixedcapvec<courant_record2, HYDRO_SIZE + 3> rec2;
};

#define SMOOTHLEN_BLOCK_SIZE 512
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

__global__ void sph_cuda_smoothlen(sph_run_params params, sph_run_cuda_data data, smoothlen_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__
	float error;
	smoothlen_workspace& ws = workspaces[bid];
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {

		int flops = 0;
		ws.x.resize(0);
		ws.y.resize(0);
		ws.z.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		//	PRINT( "%i\n", self.neighbor_range.second - self.neighbor_range.first);
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			//PRINT( "%i\n", -self.neighbor_range.first+self.neighbor_range.second);
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
			if (data.rungs[i] >= params.min_rung) {
				const int snki = self.sink_part_range.first - self.part_range.first + i;
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				int count;
				float f;
				float dfdh;
				int box_xceeded = false;
				int iter = 0;
				float dh;
				float& h = data.h_snk[snki];
				do {
					float max_dh = h / sqrtf(iter + 100);
					const float hinv = 1.f / h; // 4
					const float h2 = sqr(h);    // 1
					count = 0;
					f = 0.f;
					dfdh = 0.f;
					for (int j = tid; j < ws.x.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.x[j]); // 2
						const float dy = distance(x[YDIM], ws.y[j]); // 2
						const float dz = distance(x[ZDIM], ws.z[j]); // 2
						const float r2 = sqr(dx, dy, dz);            // 2
						const float r = sqrt(r2);                    // 4
						const float q = r * hinv;                    // 1
						flops += 15;
						if (q < 1.f) {                               // 1
							const float w = kernelW(q); // 4
							const float dwdh = -q * dkernelW_dq(q) * hinv; // 3
							f += w;                                   // 1
							dfdh += dwdh;                             // 1
							flops += 15;
							count++;
						}
					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(f);
					shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(count);
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dfdh);
					dh = 0.2f * h;
					if (count > 1) {
						f -= data.N * float(3.0 / (4.0 * M_PI));
						dh = -f / dfdh;
						dh = fminf(fmaxf(dh, -max_dh), max_dh);
					}
					error = fabsf(f) / (data.N * float(3.0 / (4.0 * M_PI)));
					__syncthreads();
					if (tid == 0) {
						h += dh;
						if (iter > 30) {
							PRINT("over iteration on h solve - %i %e %e %e %e %i\n", iter, h, dh, max_dh, error, count);
						}
					}
					__syncthreads();
					for (int dim = 0; dim < NDIM; dim++) {
						if (distance(self.outer_box.end[dim], x[dim]) - h < 0.0f) {
							box_xceeded = true;
							break;
						}
						if (distance(x[dim], self.outer_box.begin[dim]) - h < 0.0f) {
							box_xceeded = true;
							break;
						}
					}
					iter++;
					if (max_dh / h < SPH_SMOOTHLEN_TOLER) {
						if (tid == 0) {
							PRINT("density solver failed to converge %i\n", ws.x.size());
							__trap();
						}
					}
					shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(box_xceeded);
				} while (error > SPH_SMOOTHLEN_TOLER && !box_xceeded);
				if (tid == 0 && h <= 0.f) {
					PRINT("Less than ZERO H! sph.cu %e\n", h);
					__trap();
				}
				//	if (tid == 0)
				//	PRINT("%i %e\n", count, data.N);
				//		PRINT( "%e\n", h);
				hmin = fminf(hmin, h);
				hmax = fmaxf(hmax, h);
				if (tid == 0) {
					if (box_xceeded) {
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

}

__global__ void sph_cuda_mark_semiactive(sph_run_params params, sph_run_cuda_data data, mark_semiactive_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	mark_semiactive_workspace& ws = workspaces[bid];
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {
		int flops = 0;
		ws.x.resize(0);
		ws.y.resize(0);
		ws.z.resize(0);
		ws.h.resize(0);
		ws.rungs.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		//	PRINT( "%i\n", self.neighbor_range.second - self.neighbor_range.first);
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			//PRINT( "%i\n", -self.neighbor_range.first+self.neighbor_range.second);
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
				compute_indices<SMOOTHLEN_BLOCK_SIZE>(j, total);
				const int offset = ws.x.size();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				ws.y.resize(next_size);
				ws.z.resize(next_size);
				ws.h.resize(next_size);
				ws.rungs.resize(next_size);
				if (contains) {
					if (j >= total) {
						PRINT("%i %i\n", j, total);
					}
					ASSERT(j < total);
					const int k = offset + j;
					ASSERT(k < next_size);
					ASSERT(k < ws.x.size());
					ASSERT(k < ws.y.size());
					ASSERT(k < ws.z.size());
					ASSERT(k < ws.h.size());
					ws.x[k] = x[XDIM];
					ws.y[k] = x[YDIM];
					ws.z[k] = x[ZDIM];
					ws.h[k] = h;
					ws.rungs[k] = rung;
				}
			}
		}

		for (int i = self.part_range.first; i < self.part_range.second; i++) {
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
							//			PRINT( "SEMIACTIVE\n");
							semiactive++;
						}
					}
					shared_reduce_add<int>(semiactive);
					if (semiactive) {
						if (tid == 0) {
							data.sa_snk[snki] = true;
						}
						break;
					}
				}
			}
		}
		shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(flops);
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}

}

__global__ void sph_cuda_diffusion(sph_run_params params, sph_run_cuda_data data, dif_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	dif_workspace& ws = workspaces[bid];
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	int flops = 0;
	while (index < data.nselfs) {
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
					if (self.outer_box.contains(x)) {											// 24
						contains = true;
					}
					flops += 24;
					if (!contains) {
						contains = true;
						const float& h = data.h[pi];
						for (int dim = 0; dim < NDIM; dim++) {
							if (distance(x[dim], self.inner_box.begin[dim]) + h < 0.f) { // 4
								contains = false;
								break;
							}
							if (distance(self.inner_box.end[dim], x[dim]) + h < 0.f) {   // 4
								contains = false;
								break;
							}
						}
						flops += 24;
					}
				}
				j = contains;
				compute_indices<HYDRO_BLOCK_SIZE>(j, total);
				const int offset = ws.rec1_main.size();
				const int next_size = offset + total;
				ws.rec1_main.resize(next_size);
				ws.rec2_main.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1_main[k].x = x[XDIM];
					ws.rec1_main[k].y = x[YDIM];
					ws.rec1_main[k].z = x[ZDIM];
					ws.rec1_main[k].h = data.h[pi];
					ws.rec1_main[k].rung = data.rungs[pi];
					ws.rec2_main[k].difco = data.difco[pi];
					ws.rec2_main[k].kappa = data.kappa[pi];
					ws.rec2_main[k].gamma = data.gamma[pi];
					ws.rec2_main[k].vec = data.dif_vec[pi];
					ws.rec2_main[k].oldrung = data.oldrung[pi];
					ws.rec2_main[k].mmw = data.mmw[pi];
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			int myrung = data.rungs[i];
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			bool active = myrung >= params.min_rung;
			bool semi_active = !active && data.sa_snk[snki];
			bool use = active || semi_active;
			const float m = data.m;
			const float minv = 1.f / m;																	// 4
			const float c0 = float(3.0f / 4.0f / M_PI * data.N);
			const float c0inv = float(1.0f / c0);

			if (use) {
				const auto x_i = data.x[i];
				const auto y_i = data.y[i];
				const auto z_i = data.z[i];
				const float h_i = data.h[i];
				const float hinv_i = 1.f / h_i;															// 4
				const float h3inv_i = (sqr(hinv_i) * hinv_i);										// 6
				const float rho_i = m * c0 * h3inv_i;													// 2
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;								// 5
				const float difco_i = data.difco[i];
				const float kappa_i = data.kappa[i];
				const auto vec0_i = data.vec0_snk[snki];
				const auto vec_i = data.dif_vec[i];
				const auto gamma_i = data.gamma[i];
				const auto mmw_i = data.mmw[i];
				const int jmax = round_up(ws.rec1_main.size(), block_size);
				ws.rec1.resize(0);
				ws.rec2.resize(0);
				for (int j = tid; j < jmax; j += block_size) {
					bool flag = false;
					int k;
					int total;
					if (j < ws.rec1_main.size()) {
						const auto rec = ws.rec1_main[j];
						const auto x_j = rec.x;
						const auto y_j = rec.y;
						const auto z_j = rec.z;
						const float h_j = rec.h;
						const float dx = distance(x_i, x_j);									// 2
						const float dy = distance(y_i, y_j);									// 2
						const float dz = distance(z_i, z_j);                         	// 2
						const float h2max = sqr(fmaxf(h_i, h_j));
						const float r2 = sqr(dx, dy, dz);
						if (r2 < h2max) {
							if (semi_active) {
								if (rec.rung >= params.min_rung) {
									flag = true;
								}
							} else {
								flag = true;
							}
						}
					}
					k = flag;
					compute_indices<HYDRO_BLOCK_SIZE>(k, total);
					const int offset = ws.rec1.size();
					const int next_size = offset + total;
					ws.rec1.resize(next_size);
					ws.rec2.resize(next_size);
					if (flag) {
						const int l = offset + k;
						ws.rec1[l] = ws.rec1_main[j];
						ws.rec2[l] = ws.rec2_main[j];
					}
				}
				dif_vector num;
				float den = 0.f;
				float den_A = 0.f;
				for (int fi = 0; fi < DIFCO_COUNT; fi++) {
					num[fi] = 0.f;
				}
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					auto rec1 = ws.rec1[j];
					auto rec2 = ws.rec2[j];
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);
					const float rinv = 1.0f / (1.0f + r);
					const float gamma_j = rec2.gamma;
					const float kappa_j = rec2.kappa;
					const float difco_j = rec2.difco;
					const float mmw_j = rec2.mmw;
					const float h_j = rec1.h;
					const float hinv_j = 1.f / h_j;															// 4
					const float h3inv_j = sqr(hinv_j) * hinv_j;
					const float rho_j = m * c0 * h3inv_j;													// 2
					const float dt_ij = 0.5f * fminf(rung_dt[myrung], rung_dt[rec1.rung]) * params.t0;
					const float rho_ij = 0.5f * (rho_i + rho_j);
					const float h_ij = 0.5f * (h_i + h_j);
					const float kappa_ij = 2.f * kappa_i * kappa_j / (kappa_i + kappa_j + 1e-30) ;
					const float difco_ij = 2.f * (difco_i * difco_j) / (difco_i + difco_j + 1e-30);
					const float dWdr_ij = 0.5f * (dkernelW_dq(fminf(r * hinv_i, 1.f)) / sqr(sqr(h_i)) + dkernelW_dq(fminf(r * hinv_j, 1.f)) / sqr(sqr(hinv_j)));
					const float diff_factor = -2.f * dt_ij * m / rho_ij * difco_ij * dWdr_ij * rinv;
					const float cond_factor = -dt_ij * m / (rho_i * rho_j) * kappa_ij * dWdr_ij * rinv;
					for (int fi = 0; fi < DIFCO_COUNT; fi++) {
						num[fi] += diff_factor * rec2.vec[fi];
					}
					den += diff_factor;
					float adjust = powf(rho_j, gamma_j - 1.f) * powf(rho_i, 1.f - gamma_i) * mmw_j / mmw_i;
					num[NCHEMFRACS] += cond_factor * rec2.vec[NCHEMFRACS] * adjust;
					den_A += diff_factor + cond_factor;
				}
				for (int fi = 0; fi < DIFCO_COUNT; fi++) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(num[fi]);
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(den);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(den_A);
				if (tid == 0) {
					den += 1.0f;
					den_A += 1.0f;
					num[NCHEMFRACS] += vec0_i[NCHEMFRACS];
					data.dvec_snk[snki][NCHEMFRACS] = num[NCHEMFRACS] / den_A - vec_i[NCHEMFRACS];
					for (int fi = 0; fi < NCHEMFRACS; fi++) {
						num[fi] += vec0_i[fi];
						data.dvec_snk[snki][fi] = num[fi] / den - vec_i[fi];
					}
				}
			}
		}
		shared_reduce_add<int, HYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
}

#define SIGMA 2.0f
#define ETA1 0.01f
#define ETA2 0.0001f

__global__ void sph_cuda_hydro(sph_run_params params, sph_run_cuda_data data, hydro_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	hydro_workspace& ws = workspaces[bid];
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	int flops = 0;
	while (index < data.nselfs) {
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
				const int next_size = offset + total;
				ws.rec1_main.resize(next_size);
				ws.rec2_main.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1_main[k].x = x[XDIM];
					ws.rec1_main[k].y = x[YDIM];
					ws.rec1_main[k].z = x[ZDIM];
					ws.rec1_main[k].h = data.h[pi];
					ws.rec1_main[k].rung = data.rungs[pi];
					if (data.gamma) {
						ws.rec2_main[k].gamma = data.gamma[pi];
					} else {
						ws.rec2_main[k].gamma = 5. / 3.;
					}
					ws.rec2_main[k].vx = data.vx[pi];
					ws.rec2_main[k].vy = data.vy[pi];
					ws.rec2_main[k].vz = data.vz[pi];
					ws.rec2_main[k].gx = data.gx[pi];
					ws.rec2_main[k].gy = data.gy[pi];
					ws.rec2_main[k].gz = data.gz[pi];
					ws.rec2_main[k].ent = data.ent[pi];
					ws.rec2_main[k].f0 = data.f0[pi];
					ws.rec2_main[k].fvel = data.fvel[pi];
					ws.rec2_main[k].alpha = data.alpha[pi];
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			int myrung = data.rungs[i];
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			bool active = myrung >= params.min_rung;
			bool semi_active = !active && data.sa_snk[snki];
			bool use = active || semi_active;
			bool first_step = params.phase == 1 && active;
			if (first_step && use && tid == 0) {
				data.dent_con[snki] = 0.0f;
				data.dvx_con[snki] = 0.f;
				data.dvy_con[snki] = 0.f;
				data.dvz_con[snki] = 0.f;
			}
			const float m = data.m;
			const float minv = 1.f / m;
			const float c0 = float(3.0f / 4.0f / M_PI * data.N);
			const float c0inv = 1.0f / c0;
			if (use) {
				const fixed32 x_i = data.x[i];
				const fixed32 y_i = data.y[i];
				const fixed32 z_i = data.z[i];
				const float vx_i = data.vx[i];
				const float vy_i = data.vy[i];
				const float vz_i = data.vz[i];
				const float gx_i = data.gx[i];
				const float gy_i = data.gy[i];
				const float gz_i = data.gz[i];
				const float h_i = data.h[i];
				const float hinv_i = 1.f / h_i;															// 4
				const float h3inv_i = sqr(hinv_i) * hinv_i;
				const float rho_i = m * c0 * h3inv_i;													// 2
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;								// 5
				const float ent_i = data.ent[i];
				const float gamma_i = data.gamma[i];
				const float p_i = ent_i * powf(rho_i, gamma_i);								// 5
				const float c_i = sqrtf(gamma_i * p_i * rhoinv_i);									// 6
				const float fvel_i = data.fvel[i];
				const float f0_i = data.f0[i];
				const float alpha_i = data.alpha[i];
				const float rho1mgamma_i = powf(rho_i, 1.0f - gamma_i);						// 5
				const int jmax = round_up(ws.rec1_main.size(), block_size);
				flops += 36;
				ws.rec1.resize(0);
				ws.rec2.resize(0);
				for (int j = tid; j < jmax; j += block_size) {
					bool flag = false;
					int k;
					int total;
					if (j < ws.rec1_main.size()) {
						const auto rec = ws.rec1_main[j];
						const auto x_j = rec.x;
						const auto y_j = rec.y;
						const auto z_j = rec.z;
						const float h_j = rec.h;
						const float dx = distance(x_j, x_i);
						const float dy = distance(y_j, y_i);
						const float dz = distance(z_j, z_i);
						const float h2max = sqr(fmaxf(h_j, h_i));
						const float r2 = sqr(dx, dy, dz);
						if (r2 < h2max) {
							if (semi_active) {
								if (rec.rung >= params.min_rung) {
									flag = true;
								}
							} else {
								flag = true;
							}
						}
					}
					k = flag;
					compute_indices<HYDRO_BLOCK_SIZE>(k, total);
					const int offset = ws.rec1.size();
					const int next_size = offset + total;
					ws.rec1.resize(next_size);
					ws.rec2.resize(next_size);
					if (flag) {
						const int l = offset + k;
						ws.rec1[l] = ws.rec1_main[j];
						ws.rec2[l] = ws.rec2_main[j];
					}
				}
				float dvxdx = 0.f;
				float dvxdy = 0.f;
				float dvxdz = 0.f;
				float dvydx = 0.f;
				float dvydy = 0.f;
				float dvydz = 0.f;
				float dvzdx = 0.f;
				float dvzdy = 0.f;
				float dvzdz = 0.f;
				float ddivv_dt = 0.f;
				float dent_pred = 0.f;
				float dvx_pred = 0.f;
				float dvy_pred = 0.f;
				float dvz_pred = 0.f;
				float dent_con = 0.f;
				float dvx_con = 0.f;
				float dvy_con = 0.f;
				float dvz_con = 0.f;
				float vsig = 0.f;
				const float ainv = 1.0f / params.a;
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					auto rec1 = ws.rec1[j];
					auto rec2 = ws.rec2[j];
					const float vx_j = rec2.vx;
					const float vy_j = rec2.vy;
					const float vz_j = rec2.vz;
					const float gx_j = rec2.gx;
					const float gy_j = rec2.gy;
					const float gz_j = rec2.gz;
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float h_j = rec1.h;
					const float hinv_j = 1.f / h_j;															// 4
					const float h3inv_j = sqr(hinv_j) * hinv_j;
					const float rho_j = m * c0 * h3inv_j;													// 2
					const float rhoinv_j = minv * c0inv * sqr(h_j) * h_j;								// 5
					const float ent_j = rec2.ent;
					const float gamma_j = rec2.gamma;
					const float p_j = ent_j * powf(rho_j, gamma_j);								// 5
					const float c_j = sqrtf(gamma_j * p_j * rhoinv_j);									// 6
					const float fvel_j = rec2.fvel;
					const float f0_j = rec2.f0;
					const float alpha_j = rec2.alpha;
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float vx_ij = vx_i - vx_j;
					const float vy_ij = vy_i - vy_j;
					const float vz_ij = vz_i - vz_j;
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);
					const float rinv = 1.0f / (1.0f + r);
					const float alpha_ij = 0.5f * (alpha_i * fvel_i + alpha_j * fvel_j);
					const float h_ij = 0.5f * (h_i + h_j);
					const float vdotr_ij = fminf(0.0f, x_ij * vx_ij + y_ij * vy_ij + z_ij * vz_ij);
					const float u_ij = vdotr_ij * h_ij / (r2 + ETA1 * sqr(h_ij));
					const float c_ij = 0.5f * (c_i + c_j);
					vsig = fmaxf(vsig, c_ij - vdotr_ij * rinv);
					const float rho_ij = 0.5f * (rho_i + rho_j);
					const float Pi = -alpha_ij * u_ij * (c_ij - SPH_BETA * u_ij) / rho_ij;
					const float q_i = fminf(r * hinv_i, 1.f);								// 1
					const float q_j = fminf(r * hinv_j, 1.f);									// 1
					const float dWdr_i = dkernelW_dq(q_i) * hinv_i * h3inv_i;
					const float dWdr_j = dkernelW_dq(q_j) * hinv_j * h3inv_j;
					const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);
					const float dWdr_x_ij = x_ij * rinv * dWdr_ij;
					const float dWdr_y_ij = y_ij * rinv * dWdr_ij;
					const float dWdr_z_ij = z_ij * rinv * dWdr_ij;
					const float dp_i = f0_i * p_i * powf(rho_i, SIGMA - 2.f) * powf(rho_j, -SIGMA);
					const float dp_j = f0_j * p_j * powf(rho_j, SIGMA - 2.f) * powf(rho_i, -SIGMA);
					const float dvx_dt = -m * (dp_i + dp_j + Pi) * dWdr_x_ij;
					const float dvy_dt = -m * (dp_i + dp_j + Pi) * dWdr_y_ij;
					const float dvz_dt = -m * (dp_i + dp_j + Pi) * dWdr_z_ij;
					const float dWdr_x_i = dWdr_i * rinv * x_ij;
					const float dWdr_y_i = dWdr_i * rinv * y_ij;
					const float dWdr_z_i = dWdr_i * rinv * z_ij;
					const float mrhoinv_i = m * rhoinv_i;
					float tmp = 0.f;
					tmp += (p_i * powf(rho_i, -SIGMA) + p_j * powf(rho_j, -SIGMA)) * (powf(rho_i, SIGMA - 1.0f) - powf(rho_j, SIGMA - 1.f));
					tmp += (powf(rho_i, SIGMA - 2.0f) + powf(rho_j, SIGMA - 2.f)) * (p_i * powf(rho_i, 1.f - SIGMA) - p_j * powf(rho_j, 1.f - SIGMA));
					ddivv_dt += m * tmp * dWdr_ij / rho_ij * rinv;
					ddivv_dt += mrhoinv_i * (gx_j - gx_i) * dWdr_x_i;
					ddivv_dt += mrhoinv_i * (gy_j - gy_i) * dWdr_y_i;
					ddivv_dt += mrhoinv_i * (gz_j - gz_i) * dWdr_z_i;
					dvxdx -= mrhoinv_i * vx_ij * dWdr_x_i;
					dvydx -= mrhoinv_i * vy_ij * dWdr_x_i;
					dvzdx -= mrhoinv_i * vz_ij * dWdr_x_i;
					dvxdy -= mrhoinv_i * vx_ij * dWdr_y_i;
					dvydy -= mrhoinv_i * vy_ij * dWdr_y_i;
					dvzdy -= mrhoinv_i * vz_ij * dWdr_y_i;
					dvxdz -= mrhoinv_i * vx_ij * dWdr_z_i;
					dvydz -= mrhoinv_i * vy_ij * dWdr_z_i;
					dvzdz -= mrhoinv_i * vz_ij * dWdr_z_i;
					float dt_pred, dt_con;
					dt_pred = 0.5f * rung_dt[myrung] * params.t0;		// 2
					dt_con = fminf(0.5f * rung_dt[rec1.rung] * (params.t0), dt_pred); // 3
					const float tmp2 = (vx_ij * dWdr_x_ij + vy_ij * dWdr_y_ij + vz_ij * dWdr_z_ij);
					const float dA_dt = 0.5f * m * (gamma_i - 1.f) * rho1mgamma_i * Pi * tmp2;
					if (first_step) {
						dent_pred += dA_dt * dt_pred;							// 2
						dvx_pred += dvx_dt * dt_pred;							// 2
						dvy_pred += dvy_dt * dt_pred;							// 2
						dvz_pred += dvz_dt * dt_pred;							// 2
						flops += 8;
					}
					dent_con += dA_dt * dt_con;									// 2
					dvx_con += dvx_dt * dt_con;								// 2
					dvy_con += dvy_dt * dt_con;								// 2
					dvz_con += dvz_dt * dt_con;								// 2
					flops += 181;
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dent_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvx_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvy_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvz_con);
				if (first_step) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dent_pred);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvx_pred);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvy_pred);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvz_pred);
				}
				float divv = f0_i * (dvxdx + dvydy + dvzdz);
				float curlv_x = f0_i * (dvzdy - dvydz);
				float curlv_y = f0_i * (-dvzdx + dvxdz);
				float curlv_z = f0_i * (dvydx - dvxdy);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddivv_dt);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(divv);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curlv_x);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curlv_y);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curlv_z);
				shared_reduce_max<float, HYDRO_BLOCK_SIZE>(vsig);
				if (tid == 0) {
					if (first_step) {
						data.dent_pred[snki] = dent_pred;
						data.dvx_pred[snki] = dvx_pred;
						data.dvy_pred[snki] = dvy_pred;
						data.dvz_pred[snki] = dvz_pred;
					}
					data.dent_con[snki] += dent_con;										// 1
					data.dvx_con[snki] += dvx_con;										// 1
					data.dvy_con[snki] += dvy_con;										// 1
					data.dvz_con[snki] += dvz_con;										// 1
					flops += 4;
					if (params.phase == 1 && !semi_active) {
						data.divv_snk[snki] = divv;
					}
					const float alpha_n = data.alpha_snk[snki];
					float& alpha_np1 = data.alpha_snk[snki];
					const float t0 = h_i / vsig / SPH_VISC_DECAY;
					const float balsara = fabsf(divv) / (sqrt(sqr(curlv_x, curlv_y, curlv_z)) + fabsf(divv) + ETA2 * c_i * hinv_i);
					float S = fmaxf(0.f, -ddivv_dt) * balsara;
					const float alpha_targ = SPH_ALPHA1 / (1.f + sqr(vsig) / (sqr(h_i) * S));
					float dt = 0.5f * rung_dt[myrung] * params.t0; // 3
					const float num = alpha_n + dt / t0 * alpha_targ;
					const float den = 1.f + dt / t0;
					alpha_np1 = fmaxf(alpha_targ, num / den);
				}
			}
		}
		shared_reduce_add<int, HYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
}

__global__ void sph_cuda_courant(sph_run_params params, sph_run_cuda_data data, courant_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	courant_workspace& ws = workspaces[bid];
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	float total_vsig_max = 0.;
	int max_rung_hydro = 0;
	int max_rung_grav = 0;
	int max_rung = 0;
	const bool stars = data.gx;
	const float Ginv = 1.f / data.G;
	int flops = 0;

	while (index < data.nselfs) {
		const sph_tree_node& self = data.trees[data.selfs[index]];
		if (self.nactive > 0) {
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
					}
					j = contains;
					compute_indices<HYDRO_BLOCK_SIZE>(j, total);
					const int offset = ws.rec1_main.size();
					const int next_size = offset + total;
					ws.rec1_main.resize(next_size);
					ws.rec2_main.resize(next_size);
					if (contains) {
						const int k = offset + j;
						ws.rec1_main[k].x = x[XDIM];
						ws.rec1_main[k].y = x[YDIM];
						ws.rec1_main[k].z = x[ZDIM];
						ws.rec2_main[k].vx = data.vx[pi];
						ws.rec2_main[k].vy = data.vy[pi];
						ws.rec2_main[k].vz = data.vz[pi];
						ws.rec2_main[k].ent = data.ent[pi];
						ws.rec1_main[k].h = data.h[pi];
						ws.rec2_main[k].T = data.T[pi];
						ws.rec2_main[k].lambda_e = data.lambda_e[pi];
						ws.rec2_main[k].mmw = data.mmw[pi];
						ws.rec2_main[k].alpha = data.alpha[pi];
						if (data.gamma) {
							ws.rec2_main[k].gamma = data.gamma[pi];
						}
						if (stars) {
							ws.rec2_main[k].gx = data.gx[pi];
							ws.rec2_main[k].gy = data.gy[pi];
							ws.rec2_main[k].gz = data.gz[pi];
						}
					}
				}
			}
			for (int i = self.part_range.first; i < self.part_range.second; i++) {
				int myrung = data.rungs[i];
				bool use = myrung >= params.min_rung;
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
					const float T_i = data.T[i];
					const float h_i = data.h[i];
					const float h2_i = sqr(h_i);
					const float hinv_i = 1.f / h_i;
					const float h3inv_i = sqr(hinv_i) * hinv_i;
					const float rho_i = m * c0 * h3inv_i;
					const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;
					const float alpha_i = data.alpha[i];
					const float gamma_i = data.gamma[i];
					const float ent_i = data.ent[i];
					const float p_i = ent_i * powf(rho_i, gamma_i);
					const float c_i = sqrtf(gamma_i * p_i * rhoinv_i);
					const float gx_i = data.gx[i];
					const float gy_i = data.gy[i];
					const float gz_i = data.gz[i];
					const int jmax = round_up(ws.rec1_main.size(), block_size);
					ws.rec1.resize(0);
					ws.rec2.resize(0);
					for (int j = tid; j < jmax; j += block_size) {
						bool flag = false;
						int k;
						int total;
						if (j < ws.rec1_main.size()) {
							const auto rec = ws.rec1_main[j];
							const auto x_j = rec.x;
							const auto y_j = rec.y;
							const auto z_j = rec.z;
							const float h_j = rec.h;
							const float dx = distance(x_i, x_j);
							const float dy = distance(y_i, y_j);
							const float dz = distance(z_i, z_j);
							const float r2 = sqr(dx, dy, dz);
							if (r2 < fmaxf(h_i, h_j)) {
								flag = true;
							}
						}
						k = flag;
						compute_indices<HYDRO_BLOCK_SIZE>(k, total);
						const int offset = ws.rec1.size();
						const int next_size = offset + total;
						ws.rec1.resize(next_size);
						ws.rec2.resize(next_size);
						if (flag) {
							const int l = offset + k;
							ws.rec1[l] = ws.rec1_main[j];
							ws.rec2[l] = ws.rec2_main[j];
						}
					}
					float vsig_max = 0.f;
					float dvx_dx = 0.0f;
					float dvx_dy = 0.0f;
					float dvx_dz = 0.0f;
					float dvy_dx = 0.0f;
					float dvy_dy = 0.0f;
					float dvy_dz = 0.0f;
					float dvz_dx = 0.0f;
					float dvz_dy = 0.0f;
					float dvz_dz = 0.0f;
					float drho_dh = 0.f;
					float dgx_dx = 0.f;
					float dgy_dy = 0.f;
					float dgz_dz = 0.f;
					float dT_dx = 0.f;
					float dT_dy = 0.f;
					float dT_dz = 0.f;
					float ax = 0.f;
					float ay = 0.f;
					float az = 0.f;
					for (int j = tid; j < ws.rec1.size(); j += block_size) {
						const auto rec1 = ws.rec1[j];
						const auto rec2 = ws.rec2[j];

						const auto x_j = rec1.x;
						const auto y_j = rec1.y;
						const auto z_j = rec1.z;
						const auto vx_j = rec2.vx;
						const auto vy_j = rec2.vy;
						const auto vz_j = rec2.vz;
//						const auto gx_j = rec2.gx;
//						const auto gy_j = rec2.gy;
//						const auto gz_j = rec2.gz;
						const float T_j = rec2.T;
						const float h_j = rec1.h;
						const float h2_j = sqr(h_j);
						const float hinv_j = 1.f / h_j;
						const float h3inv_j = sqr(hinv_j) * hinv_j;
						const float rho_j = m * c0 * h3inv_j;
						const float rhoinv_j = minv * c0inv * sqr(h_j) * h_j;
						const float alpha_j = rec2.alpha;
						const float gamma_j = rec2.gamma;
						const float ent_j = rec2.ent;
						const float p_j = ent_j * powf(rho_j, gamma_j);
						const float c_j = sqrtf(gamma_j * p_j * rhoinv_j);

						const float x_ij = distance(x_i, x_j);				// 2
						const float y_ij = distance(y_i, y_j);				// 2
						const float z_ij = distance(z_i, z_j);				// 2
						const float vx_ij = vx_i - vx_j;
						const float vy_ij = vy_i - vy_j;
						const float vz_ij = vz_i - vz_j;
						const float r2 = sqr(x_ij, y_ij, z_ij);
						const float r = sqrt(r2);
						const float rinv = 1.0f / (1.0f + r);
						const float alpha_ij = 0.5f * (alpha_i + alpha_j);
						const float h_ij = 0.5f * (h_i + h_j);
						const float vdotr_ij = fminf(0.0f, x_ij * vx_ij + y_ij * vy_ij + z_ij * vz_ij);
						const float u_ij = vdotr_ij * h_ij / (r2 + ETA1 * sqr(h_ij));
						const float c_ij = 0.5f * (c_i + c_j);
						const float rho_ij = 0.5f * (rho_i + rho_j);
						const float Pi = -alpha_ij * u_ij * (c_ij - SPH_BETA * u_ij) / rho_ij;
						const float q_i = fminf(r * hinv_i, 1.f);								// 1
						const float q_j = fminf(r * hinv_j, 1.f);									// 1
						const float dWdr_i = dkernelW_dq(q_i) * hinv_i * h3inv_i;
						const float dWdr_j = dkernelW_dq(q_j) * hinv_j * h3inv_j;
						const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);
						const float dWdr_x_ij = x_ij * rinv * dWdr_ij;
						const float dWdr_y_ij = y_ij * rinv * dWdr_ij;
						const float dWdr_z_ij = z_ij * rinv * dWdr_ij;
						const float dp_i = p_i * powf(rho_i, SIGMA - 2.f) * powf(rho_j, -SIGMA);
						const float dp_j = p_j * powf(rho_j, SIGMA - 2.f) * powf(rho_i, -SIGMA);
						const float dvx_dt = -m * (dp_i + dp_j + Pi) * dWdr_x_ij;
						const float dvy_dt = -m * (dp_i + dp_j + Pi) * dWdr_y_ij;
						const float dvz_dt = -m * (dp_i + dp_j + Pi) * dWdr_z_ij;
						const float dWdr_x_i = dWdr_i * rinv * x_ij;
						const float dWdr_y_i = dWdr_i * rinv * y_ij;
						const float dWdr_z_i = dWdr_i * rinv * z_ij;
						const float mrhoinv_i = m * rhoinv_i;
						dvx_dx -= mrhoinv_i * vx_ij * dWdr_x_i;
						dvy_dx -= mrhoinv_i * vy_ij * dWdr_x_i;
						dvz_dx -= mrhoinv_i * vz_ij * dWdr_x_i;
						dvx_dy -= mrhoinv_i * vx_ij * dWdr_y_i;
						dvy_dy -= mrhoinv_i * vy_ij * dWdr_y_i;
						dvz_dy -= mrhoinv_i * vz_ij * dWdr_y_i;
						dvx_dz -= mrhoinv_i * vx_ij * dWdr_z_i;
						dvy_dz -= mrhoinv_i * vy_ij * dWdr_z_i;
						dvz_dz -= mrhoinv_i * vz_ij * dWdr_z_i;
						const float hfac = h_i / h_ij;
						float this_vsig = c_ij * hfac;
						if (vdotr_ij < 0.f) {
							this_vsig += 0.6f * alpha_ij * c_ij * hfac;
							this_vsig -= 0.6f * alpha_ij * SPH_BETA * vdotr_ij * hfac;
						}
						vsig_max = fmaxf(vsig_max, this_vsig);									   // 2
						const float W = kernelW(fminf(r * hinv_i, 1.f)) * h3inv_i;      // 14
						const float mrhoinvW = m * rhoinv_i * W;
						ax += dvx_dt;
						ay += dvy_dt;
						az += dvz_dt;
						const float tmp3 = (T_j - T_i) * m * rhoinv_i;
						dT_dx += tmp3 * dWdr_x_i;
						dT_dy += tmp3 * dWdr_y_i;
						dT_dz += tmp3 * dWdr_z_i;
						drho_dh -= (3.f * kernelW(q_i) + dkernelW_dq(q_i));
						/*						if (stars) {
						 dgx_dx += (rec2.gx - mygx) * mydWdr_x * m * myrhoinv;
						 dgy_dy += (rec2.gy - mygy) * mydWdr_y * m * myrhoinv;
						 dgz_dz += (rec2.gz - mygz) * mydWdr_z * m * myrhoinv;
						 }*/

					}
					float div_v = dvx_dx + dvy_dy + dvz_dz;
					float curl_vx = dvz_dy - dvy_dz;
					float curl_vy = -dvz_dx + dvx_dz;
					float curl_vz = dvy_dx - dvx_dy;
					float shear_xx = dvx_dx - (1.f / 3.f) * div_v;
					float shear_yy = dvy_dy - (1.f / 3.f) * div_v;
					float shear_zz = dvz_dz - (1.f / 3.f) * div_v;
					float shear_xy = 0.5f * (dvx_dy + dvy_dx);
					float shear_xz = 0.5f * (dvx_dz + dvz_dx);
					float shear_yz = 0.5f * (dvy_dz + dvz_dy);
					float div_g;
					if (stars) {
						div_g = dgx_dx + dgy_dy + dgz_dz;
						shared_reduce_add<float, HYDRO_BLOCK_SIZE>(div_g);
					}
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(div_g);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ax);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ay);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(az);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dx);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dy);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dz);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(shear_xx);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(shear_xy);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(shear_xz);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(shear_yy);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(shear_yz);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(shear_zz);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(div_v);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vx);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vy);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vz);
					shared_reduce_max<float, HYDRO_BLOCK_SIZE>(vsig_max);

					if (tid == 0) {
						ax += gx_i;
						ay += gy_i;
						az += gz_i;
						const float sw = ETA2 * c_i * hinv_i;
						const float abs_div_v = fabsf(div_v);
						const float abs_curl_v = sqrtf(sqr(curl_vx, curl_vy, curl_vz));
						const float fvel = abs_div_v / (abs_div_v + abs_curl_v + sw);
						const float c0 = drho_dh * 4.0f * float(M_PI) / (9.0f * data.N);
						const float fpre = 1.0f / (1.0f + c0);
						//	PRINT("%e\n", fpre);
						div_v *= fpre;
						const float dt_cfl = params.a * h_i / vsig_max;
						const float Cdif = SPH_DIFFUSION_C * sqr(h_i) * sqrt(sqr(shear_xx, shear_yy, shear_zz) + 2.f * sqr(shear_xy, shear_xz, shear_yz));
						const float lt = T_i / (sqrt(sqr(dT_dx, dT_dy, dT_dz)) + 1.0e-10f * T_i);
						const float kappa_sp = data.kappa0 / data.colog[i]; // Jubelgas et al 2004, Smith et al 2021
						const float kappa = kappa_sp / (1.f + 4.2f * data.lambda_e[i] / lt);
						const float tmp = data.code_dif_to_cgs * constants::kb / sqr(sqr(params.a));
						float Dcond = 2.f * data.mmw[i] * (data.gamma[i] - 1.f) * kappa / tmp;
						data.kappa_snk[snki] = Dcond;
						data.fvel_snk[snki] = fvel;
						data.f0_snk[snki] = fpre;
						data.difco_snk[snki] = Cdif;
						total_vsig_max = fmaxf(total_vsig_max, vsig_max);
						float dthydro = params.cfl * dt_cfl;
						const float gx = data.gx_snk[snki];
						const float gy = data.gy_snk[snki];
						const float gz = data.gz_snk[snki];
						char& rung = data.rungs[i];
						const float g2 = sqr(gx, gy, gz);
						const float a2 = sqr(ax, ay, az);
						const float hsoft = fminf(fmaxf(h_i, data.hsoft_min), SPH_MAX_SOFT);
						const float factor = data.eta * sqrtf(params.a * hsoft);
						dthydro = fminf(fminf(factor / sqrtf(sqrtf(a2 + 1e-15f)), (float) params.t0), dthydro);
						const float dt_grav = fminf(factor / sqrtf(sqrtf(g2 + 1e-15f)), (float) params.t0);
						const float dt = fminf(dt_grav, dthydro);
						const int rung_hydro = ceilf(log2f(params.t0) - log2f(dthydro));
						const int rung_grav = ceilf(log2f(params.t0) - log2f(dt_grav));
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
						if (stars) {
							bool is_eligible = h_i < data.hstar0;
							if (is_eligible) {
								//	PRINT( "Removing sink particle\n");
							}
							/*							bool is_eligible = false;
							 const float N = ws.rec1.size();
							 float tdyn;
							 float mj;
							 float tcool;
							 if (div_v < 0.f) {
							 const float Gn32 = powf(data.G, -1.5);
							 float rho0 = data.rho0_b + data.rho0_c;
							 float delta = -Ginv * float(1.0 / 4.0 / M_PI) * div_g;
							 float delta_b = myrho - data.rho0_b;
							 float rho_tot = (rho0 + delta) * powf(params.a, -3.0);
							 tdyn = sqrtf(3.f * M_PI / (32.f * data.G * rho_tot)) / params.a;
							 if (delta_b / data.rho0_b > 10.0 && delta > 0.f) {
							 tcool = data.tcool_snk[snki];
							 if (tcool < tdyn) {
							 mj = Gn32 * rsqrt(myrho) * sqr(myc) * myc * powf(delta_b / delta, 1.5f) * powf(params.a, -1.5f);
							 const float msph = N * m;
							 if (mj < msph) {
							 is_eligible = true;
							 }
							 }
							 }
							 }*/
							if (is_eligible) {
								//float dt = rung_dt[rung] * params.t0;
								//data.tdyn_snk[snki] = tdyn;
								data.tdyn_snk[snki] = 1e-10f;
							} else {
								data.tdyn_snk[snki] = 1e+38;
							}
						}
					}
				}
			}
			shared_reduce_add<int, HYDRO_BLOCK_SIZE>(flops);
			if (tid == 0) {
				atomicAdd(&reduce->flops, (double) flops);
				index = atomicAdd(&reduce->counter, 1);
			}
			flops = 0;
			__syncthreads();
		}
	}
	if (tid == 0) {
		atomicMax(&reduce->vsig_max, total_vsig_max);
		atomicMax(&reduce->max_rung, max_rung);
		atomicMax(&reduce->max_rung_hydro, max_rung_hydro);
		atomicMax(&reduce->max_rung_grav, max_rung_grav);
	}
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
	static int dif_nblocks;
	static int courant_nblocks;
	static bool first = true;
	static char* workspace_ptr;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&smoothlen_nblocks, (const void*) sph_cuda_smoothlen, SMOOTHLEN_BLOCK_SIZE, 0));
		smoothlen_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&semiactive_nblocks, (const void*) sph_cuda_mark_semiactive, SMOOTHLEN_BLOCK_SIZE, 0));
		semiactive_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&dif_nblocks, (const void*) sph_cuda_diffusion, HYDRO_BLOCK_SIZE, 0));
		dif_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&hydro_nblocks, (const void*) sph_cuda_hydro, HYDRO_BLOCK_SIZE, 0));
		hydro_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&courant_nblocks, (const void*) sph_cuda_courant, HYDRO_BLOCK_SIZE, 0));
		courant_nblocks *= cuda_smp_count();
		size_t smoothlen_mem = sizeof(smoothlen_workspace) * smoothlen_nblocks;
		size_t semiactive_mem = sizeof(mark_semiactive_workspace) * semiactive_nblocks;
		size_t courant_mem = sizeof(courant_workspace) * courant_nblocks;
		size_t hydro_mem = sizeof(hydro_workspace) * hydro_nblocks;
		size_t dif_mem = sizeof(dif_workspace) * dif_nblocks;
		size_t max_mem = std::max(std::max(std::max(smoothlen_mem, semiactive_mem), std::max(hydro_mem, courant_mem)), dif_mem);
		CUDA_CHECK(cudaMalloc(&workspace_ptr, max_mem));
		PRINT("Allocating %i GB in workspace memory\n", max_mem / 1024 / 1024 / 1024);
//		sleep(10);
	}

	switch (params.run_type) {
	case SPH_RUN_SMOOTHLEN: {
		sph_cuda_smoothlen<<<smoothlen_nblocks, SMOOTHLEN_BLOCK_SIZE,0,stream>>>(params,data,(smoothlen_workspace*)workspace_ptr,reduce);
		cuda_stream_synchronize(stream);
		rc.rc = reduce->flag;
		rc.hmin = reduce->hmin;
		rc.hmax = reduce->hmax;
	}
	break;
	case SPH_RUN_MARK_SEMIACTIVE: {
		sph_cuda_mark_semiactive<<<semiactive_nblocks, SMOOTHLEN_BLOCK_SIZE,0,stream>>>(params,data,(mark_semiactive_workspace*)workspace_ptr,reduce);
		cuda_stream_synchronize(stream);
	}
	break;
	case SPH_RUN_HYDRO: {
		timer tm;
		tm.start();
		sph_cuda_hydro<<<hydro_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,(hydro_workspace*)workspace_ptr,reduce);
		cuda_stream_synchronize(stream);
		tm.stop();
		auto gflops = reduce->flops / tm.read() / (1024.0*1024*1024);
		PRINT( "HYDRO ran with %e GFLOPs\n", gflops);
	}
	break;
	case SPH_RUN_DIFFUSION: {
		timer tm;
		tm.start();
		sph_cuda_diffusion<<<dif_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,(dif_workspace*)workspace_ptr,reduce);
		cuda_stream_synchronize(stream);
		tm.stop();
	}
	break;
	case SPH_RUN_COURANT: {
		sph_cuda_courant<<<courant_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,(courant_workspace*)workspace_ptr,reduce);
		cuda_stream_synchronize(stream);
		rc.max_vsig = reduce->vsig_max;
		rc.max_rung_grav = reduce->max_rung_grav;
		rc.max_rung_hydro = reduce->max_rung_hydro;
		rc.max_rung = reduce->max_rung;
	}
}
	CUDA_CHECK(cudaFree(reduce));

	return rc;
}
