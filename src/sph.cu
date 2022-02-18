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

#define WORKSPACE_SIZE (128*1024)
#define HYDRO_SIZE (8*1024)

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
	float f0;
	float fvel;
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
	float h;
	float T;
	float lambda_e;
	float mmw;
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
					dh = 0.1f * h;
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
			const float minv = 1.f / m;
			const float c0 = float(3.0f / 4.0f / M_PI * data.N);
			const float c0inv = 1.0f / c0;
			if (use) {
				int myoldrung = data.oldrung[i];
				const auto myx = data.x[i];
				const auto myy = data.y[i];
				const auto myz = data.z[i];
				const float myh = data.h[i];
				const float myhinv = 1.f / myh;															// 4
				const float myh3inv = 1.f / (sqr(myh) * myh);										// 6
				const float myrho = m * c0 * myh3inv;													// 2
				const float myrhoinv = minv * c0inv * sqr(myh) * myh;								// 5
				const float mydifco = data.difco[i];
				const float mykappa = data.kappa[i];
				const auto myvec0 = data.vec0_snk[snki];
				const auto myvec = data.dif_vec[i];
				const auto mygamma = data.gamma[i];
				const auto mymmw = data.mmw[i];
				const int jmax = round_up(ws.rec1_main.size(), block_size);
				ws.rec1.resize(0);
				ws.rec2.resize(0);
				for (int j = tid; j < jmax; j += block_size) {
					bool flag = false;
					int k;
					int total;
					if (j < ws.rec1_main.size()) {
						const auto rec = ws.rec1_main[j];
						const auto x = rec.x;
						const auto y = rec.y;
						const auto z = rec.z;
						const float h = rec.h;
						const float dx = distance(x, myx);
						const float dy = distance(y, myy);
						const float dz = distance(z, myz);
						const float h2max = sqr(fmaxf(h, myh));
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
					const float dx = distance(myx, rec1.x);				// 2
					const float dy = distance(myy, rec1.y);				// 2
					const float dz = distance(myz, rec1.z);				// 2
					const float h = rec1.h;
					const float h3 = sqr(h) * h;								// 2
					const float r2 = sqr(dx, dy, dz);						// 5
					const float hinv = 1. / h;									// 4
					const float h3inv = hinv * sqr(hinv);					// 3
					const float rho = m * c0 * h3inv;						// 2
					const float dt_ij1 = 0.5f * fminf(rung_dt[myoldrung], rung_dt[rec2.oldrung]) * params.t0;
					const float dt_ij2 = 0.5f * fminf(rung_dt[myrung], rung_dt[rec1.rung]) * params.t0;
					const float dt_ij = dt_ij1 + dt_ij2;
					const float rho_ij = 0.5f * (rho + myrho);
					const float h_ij = 0.5f * (h + myh);
					const float d_ij = SPH_DIFFUSION_C * 0.5f * (rec2.difco + mydifco);
					float kappa_ij;
					if (mykappa + rec2.kappa > 0.f) {
						kappa_ij = 2.f * mykappa * rec2.kappa / (mykappa + rec2.kappa);
					} else {
						kappa_ij = 0.f;
					}
					const float r = sqrtf(r2);
					const float q_i = r * myhinv;
					const float q_j = r * hinv;
					const float dWdr_i = q_i < 1.f ? dkernelW_dq(q_i) * myhinv * myh3inv : 0.f;
					const float dWdr_j = q_j < 1.f ? dkernelW_dq(q_j) * hinv * h3inv : 0.f;
					const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);
					const float rinv = 1. / (r + 1.e-15f);
					const float factor = -dt_ij * m / rho_ij * d_ij * dWdr_ij * rinv;
					const float factor_cond = -dt_ij * m / (myrho * rho) * kappa_ij * dWdr_ij * rinv;
					for (int fi = 0; fi < DIFCO_COUNT; fi++) {
						num[fi] += factor * rec2.vec[fi];
					}
					float adjust = powf(rho, rec2.gamma - 1.f) * powf(myrho, 1.f - mygamma);
					adjust *= rec2.mmw / mymmw;
					num[NCHEMFRACS] += factor_cond * rec2.vec[NCHEMFRACS] * adjust;
					den += factor;
					den_A += factor + factor_cond;
				}
				for (int fi = 0; fi < DIFCO_COUNT; fi++) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(num[fi]);
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(den);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(den_A);
				if (tid == 0) {
					den += 1.0f;
					den_A += 1.0f;
					num[NCHEMFRACS] += myvec0[NCHEMFRACS];
					data.dvec_snk[snki][NCHEMFRACS] = num[NCHEMFRACS] / den_A - myvec[NCHEMFRACS];
					for (int fi = 0; fi < NCHEMFRACS; fi++) {
						num[fi] += myvec0[fi];
						data.dvec_snk[snki][fi] = num[fi] / den - myvec[fi];
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
					ws.rec2_main[k].ent = data.ent[pi];
					ws.rec2_main[k].f0 = data.f0[pi];
					ws.rec2_main[k].fvel = data.fvel[pi];
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
				const auto myx = data.x[i];
				const auto myy = data.y[i];
				const auto myz = data.z[i];
				const auto myvx = data.vx[i];
				const auto myvy = data.vy[i];
				const auto myvz = data.vz[i];
				const float myh = data.h[i];
				const float myhinv = 1.f / myh;															// 4
				const float myh3inv = 1.f / (sqr(myh) * myh);										// 6
				const float myrho = m * c0 * myh3inv;													// 2
				const float myrhoinv = minv * c0inv * sqr(myh) * myh;								// 5
				const float myent = data.ent[i];
				float mygamma;
				if (data.gamma) {
					mygamma = data.gamma[i];
				} else {
					mygamma = 5. / 3.;
				}
				const float myp = myent * powf(myrho, mygamma);								// 5
				if (data.ent[i] < 0.0) {
					PRINT("Negative entropy! %s %i\n", __FILE__, __LINE__);
					__trap();
				}
				const float myc = sqrtf(mygamma * myp * myrhoinv);									// 6
				const float myrho1mgammainv = powf(myrho, 1.0f - mygamma);						// 5
				const float myfvel = data.fvel[i];
				const float myf0 = data.f0[i];
				const float Prho2i = myp * myrhoinv * myrhoinv * myf0;							// 3
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
						const auto x = rec.x;
						const auto y = rec.y;
						const auto z = rec.z;
						const float h = rec.h;
						const float dx = distance(x, myx);
						const float dy = distance(y, myy);
						const float dz = distance(z, myz);
						const float h2max = sqr(fmaxf(h, myh));
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
				float divv = 0.f;
				float dent_pred = 0.f;
				float ddvx_pred = 0.f;
				float ddvy_pred = 0.f;
				float ddvz_pred = 0.f;
				float dent_con = 0.f;
				float ddvx_con = 0.f;
				float ddvy_con = 0.f;
				float ddvz_con = 0.f;
				const float ainv = 1.0f / params.a;
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					auto rec1 = ws.rec1[j];
					auto rec2 = ws.rec2[j];
					const float dx = distance(myx, rec1.x);				// 2
					const float dy = distance(myy, rec1.y);				// 2
					const float dz = distance(myz, rec1.z);				// 2
					const float h = rec1.h;
					const float h3 = sqr(h) * h;								// 2
					const float r2 = sqr(dx, dy, dz);						// 5
					const float hinv = 1. / h;									// 4
					const float h3inv = hinv * sqr(hinv);					// 3
					const float rho = m * c0 * h3inv;						// 2
					const float rhoinv = minv * c0inv * h3;				// 2
					float gamma;
					gamma = rec2.gamma;
					const float p = rec2.ent * powf(rho, gamma);			// 5
					if (p < 0.0) {
						PRINT("Negative entropy! %e %s %i\n", rec2.ent, __FILE__, __LINE__);
						__trap();
					}
					const float c = sqrtf(gamma * p * rhoinv);			// 6
					const float cij = 0.5f * (myc + c);						// 2
					const float hij = 0.5f * (h + myh);						// 2
					const float rho_ij = 0.5f * (rho + myrho);			// 2
					const float dvx = myvx - rec2.vx;						// 1
					const float dvy = myvy - rec2.vy;						// 1
					const float dvz = myvz - rec2.vz;						// 1
					const float r = sqrtf(r2);									// 4
					const float rinv = 1.0f / (r + float(1.0e-15));		// 5
					const float r2inv = 1.f / (sqr(r) + 0.01f * sqr(hij));	// 8
					const float uij = fminf(0.f, hij * (dvx * dx + dvy * dy + dvz * dz) * r2inv); //8
					const float Piij = (uij * (-float(SPH_ALPHA) * cij + float(SPH_BETA) * uij)) * 0.5f * (myfvel + rec2.fvel) / rho_ij; // 12
					const float qi = r * myhinv;								// 1
					const float qj = r * hinv;									// 1
					const float dWdri = (r < myh) * dkernelW_dq(qi) * myhinv * myh3inv * rinv; // 15
					const float dWdrj = (r < h) * dkernelW_dq(qj) * hinv * h3inv * rinv; // 15
					const float dWdri_x = dx * dWdri;						// 1
					const float dWdri_y = dy * dWdri;						// 1
					const float dWdri_z = dz * dWdri;						// 1
					const float dWdrj_x = dx * dWdrj;						// 1
					const float dWdrj_y = dy * dWdrj;						// 1
					const float dWdrj_z = dz * dWdrj;						// 1
					const float dWdrij_x = 0.5f * (dWdri_x + dWdrj_x);	// 2
					const float dWdrij_y = 0.5f * (dWdri_y + dWdrj_y);	// 2
					const float dWdrij_z = 0.5f * (dWdri_z + dWdrj_z);	// 2
					const float Prho2j = p * rhoinv * rhoinv * rec2.f0;	// 3
					float dviscx = Piij * dWdrij_x;							// 1
					float dviscy = Piij * dWdrij_y;							// 1
					float dviscz = Piij * dWdrij_z;							// 1
					float dpx = (Prho2j * dWdrj_x + Prho2i * dWdri_x) + dviscx; // 4
					float dpy = (Prho2j * dWdrj_y + Prho2i * dWdri_y) + dviscy; // 4
					float dpz = (Prho2j * dWdrj_z + Prho2i * dWdri_z) + dviscz; // 4
					const float dvxdt = -dpx * m;								// 2
					const float dvydt = -dpy * m;								// 2
					const float dvzdt = -dpz * m;								// 2
					divv -= myf0 * m * rhoinv * (dvx * dWdri_x + dvy * dWdri_y + dvz * dWdri_z); // 9
					float dt_pred, dt_con;
					dt_pred = 0.5f * rung_dt[myrung] * params.t0;		// 2
					dt_con = 0.5f * fminf(rung_dt[rec1.rung] * (params.t0), dt_pred); // 3
					float dAdt = (dviscx * dvx + dviscy * dvy + dviscz * dvz); // 5
					dAdt *= float(0.5) * m * (mygamma - 1.f) * myrho1mgammainv; // 5
					//				dAdt += dAdif;
					if (first_step) {
						dent_pred += dAdt * dt_pred;							// 2
						ddvx_pred += dvxdt * dt_pred;							// 2
						ddvy_pred += dvydt * dt_pred;							// 2
						ddvz_pred += dvzdt * dt_pred;							// 2
						flops += 8;
					}
					dent_con += dAdt * dt_con;									// 2
					ddvx_con += dvxdt * dt_con;								// 2
					ddvy_con += dvydt * dt_con;								// 2
					ddvz_con += dvzdt * dt_con;								// 2
					flops += 181;
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dent_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvx_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvy_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvz_con);
				if (first_step) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dent_pred);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvx_pred);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvy_pred);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvz_pred);
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(divv);
				if (tid == 0) {
					if (first_step) {
						data.dent_pred[snki] = dent_pred;
						data.dvx_pred[snki] = ddvx_pred;
						data.dvy_pred[snki] = ddvy_pred;
						data.dvz_pred[snki] = ddvz_pred;
					}
					data.dent_con[snki] += dent_con;										// 1
					data.dvx_con[snki] += ddvx_con;										// 1
					data.dvy_con[snki] += ddvy_con;										// 1
					data.dvz_con[snki] += ddvz_con;										// 1
					flops += 4;
					if (params.phase == 1 && !semi_active) {
						data.divv_snk[snki] = divv;
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
						ws.rec2_main[k].h = data.h[pi];
						ws.rec2_main[k].T = data.T[pi];
						ws.rec2_main[k].lambda_e = data.lambda_e[pi];
						ws.rec2_main[k].mmw = data.mmw[pi];
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
					float mygx, mygy, mygz;
					if (stars) {
						mygx = data.gx[i];
						mygy = data.gy[i];
						mygz = data.gz[i];
					}
					const auto myx = data.x[i];
					const auto myy = data.y[i];
					const auto myz = data.z[i];
					const auto myvx = data.vx[i];
					const auto myvy = data.vy[i];
					const auto myvz = data.vz[i];
					const float myT = data.T[i];
					const float myh = data.h[i];
					const float myh2 = sqr(myh);
					const float myhinv = 1.f / myh;
					const float myh3inv = 1.f / (sqr(myh) * myh);
					const float myrho = m * c0 * myh3inv;
					const float myrhoinv = minv * c0inv * sqr(myh) * myh;
					float gamma;
					if (data.gamma) {
						gamma = data.gamma[i];
					} else {
						gamma = 5.f / 3.f;
					}
					const float myp = data.ent[i] * powf(myrho, gamma);
					const float myc = sqrtf(gamma * myp * myrhoinv);
					const int jmax = round_up(ws.rec1_main.size(), block_size);
					ws.rec1.resize(0);
					ws.rec2.resize(0);
					for (int j = tid; j < jmax; j += block_size) {
						bool flag = false;
						int k;
						int total;
						if (j < ws.rec1_main.size()) {
							const auto rec = ws.rec1_main[j];
							const auto x = rec.x;
							const auto y = rec.y;
							const auto z = rec.z;
							const float dx = distance(x, myx);
							const float dy = distance(y, myy);
							const float dz = distance(z, myz);
							const float r2 = sqr(dx, dy, dz);
							if (r2 < myh2) {
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
					for (int j = tid; j < ws.rec1.size(); j += block_size) {
						const auto rec1 = ws.rec1[j];
						const auto rec2 = ws.rec2[j];
						const float dx = distance(myx, rec1.x);									// 2
						const float dy = distance(myy, rec1.y);									// 2
						const float dz = distance(myz, rec1.z);									// 2
						const float h = rec2.h;
						const float h3 = sqr(h) * h;													// 2
						const float r2 = sqr(dx, dy, dz);											// 5
						const float hinv = 1. / h;														// 4
						const float r = sqrt(r2);														// 4
						const float h3inv = hinv * sqr(hinv);										// 3
						const float rho = m * c0 * h3inv;											// 2
						const float rhoinv = minv * c0inv * h3;									// 2
						float gamma;
						if (data.gamma) {
							gamma = rec2.gamma;
						} else {
							gamma = 5.f / 3.f;
						}
						const float p = rec2.ent * powf(rho, gamma);								// 5
						const float c = sqrtf(gamma * p * rhoinv);								// 6
						const float dvx = myvx - rec2.vx;											// 1
						const float dvy = myvy - rec2.vy;											// 1
						const float dvz = myvz - rec2.vz;											// 1
						const float rinv = 1.f / (r + 1e-15f);										// 5
						const float ndv = fminf(0.f, (dx * dvx + dy * dvy + dz * dvz) * rinv); // 7
						const float hij = 0.5f * (h + myh);
						const float cij = 0.5f * (c + myc);
						const float hfac = myh / hij;
						float this_vsig = 1.25f * cij * hfac;
						if (ndv < 0.f) {
							this_vsig += 0.75f * SPH_ALPHA * cij * hfac;
							this_vsig -= 0.75f * SPH_BETA * ndv * hfac;
						}
						vsig_max = fmaxf(vsig_max, this_vsig);									   // 2
						const float q = r * myhinv;                                       // 1
						const float dWdr = dkernelW_dq(q) * rinv * myhinv * myh3inv;      // 14
						const float tmp = m * dWdr * rhoinv;
						const float dWdr_x = dx * tmp;
						const float dWdr_y = dy * tmp;
						const float dWdr_z = dz * tmp;
						dT_dx += (rec2.T - myT) * dWdr_x;
						dT_dy += (rec2.T - myT) * dWdr_y;
						dT_dz += (rec2.T - myT) * dWdr_z;
						dvx_dx -= dvx * dWdr_x;
						dvx_dy -= dvx * dWdr_y;
						dvx_dz -= dvx * dWdr_z;
						dvy_dx -= dvy * dWdr_x;
						dvy_dy -= dvy * dWdr_y;
						dvy_dz -= dvy * dWdr_z;
						dvz_dx -= dvz * dWdr_x;
						dvz_dy -= dvz * dWdr_y;
						dvz_dz -= dvz * dWdr_z;
						drho_dh -= q * dkernelW_dq(q);
						if (stars) {
							dgx_dx += (rec2.gx - mygx) * dWdr_x;
							dgy_dy += (rec2.gy - mygy) * dWdr_y;
							dgz_dz += (rec2.gz - mygz) * dWdr_z;
							const float W = kernelW(q) * myh3inv;
						}

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
						const float sw = 1e-4f * myc / myh;
						const float abs_div_v = fabsf(div_v);
						const float abs_curl_v = sqrtf(sqr(curl_vx, curl_vy, curl_vz));
						const float fvel = abs_div_v / (abs_div_v + abs_curl_v + sw);
						const float c0 = drho_dh * 4.0f * float(M_PI) / (9.0f * data.N);
						const float fpre = 1.0f / (1.0f + c0);
						div_v *= fpre;
						const float dt_cfl = params.a * myh / vsig_max;
						const float Cdif = params.a * sqr(myh) * sqrt(sqr(shear_xx, shear_yy, shear_zz) + 2.f * sqr(shear_xy, shear_xz, shear_yz));
						const float lt = myT / (sqrt(sqr(dT_dx, dT_dy, dT_dz)) + 1.0e-10f * myT);
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
						const float hsoft = fminf(fmaxf(myh, data.hsoft_min), SPH_MAX_SOFT);
						const float factor = data.eta * sqrtf(params.a * hsoft);
						const float dt_grav = fminf(factor / sqrtf(sqrtf(g2 + 1e-15f)), (float) params.t0);
						const float dt = fminf(dt_grav, dthydro);
						const int rung_hydro = ceilf(log2f(params.t0) - log2f(dthydro));
						const int rung_grav = ceilf(log2f(params.t0) - log2f(dt_grav));
						max_rung_hydro = max(max_rung_hydro, rung_hydro);
						max_rung_grav = max(max_rung_grav, rung_grav);
						rung = max(max((int) max(rung_hydro, rung_grav), max(params.min_rung, (int) rung - 1)), 1);
						max_rung = max(max_rung, rung);
						if (rung < 0 || rung >= MAX_RUNG) {
							PRINT("Rung out of range \n");
						}
						if (stars) {
							bool is_eligible = false;
							const float N = ws.rec1.size();
							float tdyn;
							float mj;
							float tcool;
							if (div_v < 0.f) {
								const float Gn32 = powf(data.G, -1.5);
								float rho0 = data.rho0_b + data.rho0_c;
								float delta = -Ginv * float(1.0 / 4.0 / M_PI) * div_g;
								float delta_b = myrho - data.rho0_b;
								float delta_c = delta - delta_b;
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
							}
							if (is_eligible) {
								float dt = rung_dt[rung] * params.t0;
								data.tdyn_snk[snki] = tdyn;
			//					rung = max(params.max_rung - 1, rung);
		//						max_rung = max(max_rung, rung);
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
