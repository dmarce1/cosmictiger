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
#include <cosmictiger/timer.hpp>

static __constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6),
		1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
				/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24),
		1.0 / (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

#define WORKSPACE_SIZE (512*SPH_NEIGHBOR_COUNT)
#define HYDRO_SIZE (20*SPH_NEIGHBOR_COUNT)

struct smoothlen_workspace {
	fixedcapvec<fixed32, WORKSPACE_SIZE> x;
	fixedcapvec<fixed32, WORKSPACE_SIZE> y;
	fixedcapvec<fixed32, WORKSPACE_SIZE> z;
};

struct mark_semiactive_workspace {
	fixedcapvec<fixed32, WORKSPACE_SIZE> x;
	fixedcapvec<fixed32, WORKSPACE_SIZE> y;
	fixedcapvec<fixed32, WORKSPACE_SIZE> z;
	fixedcapvec<float, WORKSPACE_SIZE> h;
	fixedcapvec<char, WORKSPACE_SIZE> rungs;
};

struct hydro_workspace {
	fixedcapvec<fixed32, WORKSPACE_SIZE> x0;
	fixedcapvec<fixed32, WORKSPACE_SIZE> y0;
	fixedcapvec<fixed32, WORKSPACE_SIZE> z0;
	fixedcapvec<float, WORKSPACE_SIZE> vx0;
	fixedcapvec<float, WORKSPACE_SIZE> vy0;
	fixedcapvec<float, WORKSPACE_SIZE> vz0;
	fixedcapvec<char, WORKSPACE_SIZE> rungs0;
	fixedcapvec<float, WORKSPACE_SIZE> ent0;
	fixedcapvec<float, WORKSPACE_SIZE> f00;
	fixedcapvec<float, WORKSPACE_SIZE> fvel0;
	fixedcapvec<float, WORKSPACE_SIZE> h0;
	fixedcapvec<fixed32, HYDRO_SIZE> x;
	fixedcapvec<fixed32, HYDRO_SIZE> y;
	fixedcapvec<fixed32, HYDRO_SIZE> z;
	fixedcapvec<float, HYDRO_SIZE> vx;
	fixedcapvec<char, HYDRO_SIZE> rungs;
	fixedcapvec<float, HYDRO_SIZE> vy;
	fixedcapvec<float, HYDRO_SIZE> vz;
	fixedcapvec<float, HYDRO_SIZE> ent;
	fixedcapvec<float, HYDRO_SIZE> f0;
	fixedcapvec<float, HYDRO_SIZE> fvel;
	fixedcapvec<float, HYDRO_SIZE> h;
};

struct courant_workspace {
	fixedcapvec<fixed32, WORKSPACE_SIZE> x0;
	fixedcapvec<fixed32, WORKSPACE_SIZE> y0;
	fixedcapvec<fixed32, WORKSPACE_SIZE> z0;
	fixedcapvec<float, WORKSPACE_SIZE> vx0;
	fixedcapvec<float, WORKSPACE_SIZE> vy0;
	fixedcapvec<float, WORKSPACE_SIZE> vz0;
	fixedcapvec<float, WORKSPACE_SIZE> ent0;
	fixedcapvec<float, WORKSPACE_SIZE> h0;
	fixedcapvec<fixed32, HYDRO_SIZE> x;
	fixedcapvec<fixed32, HYDRO_SIZE> y;
	fixedcapvec<fixed32, HYDRO_SIZE> z;
	fixedcapvec<float, HYDRO_SIZE> vx;
	fixedcapvec<float, HYDRO_SIZE> vy;
	fixedcapvec<float, HYDRO_SIZE> vz;
	fixedcapvec<float, HYDRO_SIZE> ent;
	fixedcapvec<float, HYDRO_SIZE> h;
};

struct fvels_workspace {
	fixedcapvec<fixed32, WORKSPACE_SIZE> x0;
	fixedcapvec<fixed32, WORKSPACE_SIZE> y0;
	fixedcapvec<fixed32, WORKSPACE_SIZE> z0;
	fixedcapvec<float, WORKSPACE_SIZE> vx0;
	fixedcapvec<float, WORKSPACE_SIZE> vy0;
	fixedcapvec<float, WORKSPACE_SIZE> vz0;
	fixedcapvec<float, WORKSPACE_SIZE> h0;
	fixedcapvec<fixed32, HYDRO_SIZE> x;
	fixedcapvec<fixed32, HYDRO_SIZE> y;
	fixedcapvec<fixed32, HYDRO_SIZE> z;
	fixedcapvec<float, HYDRO_SIZE> vx;
	fixedcapvec<float, HYDRO_SIZE> vy;
	fixedcapvec<float, HYDRO_SIZE> vz;
	fixedcapvec<float, HYDRO_SIZE> h;
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

		constexpr float A = float(float(21.0 * 2.0 / 3.0));
		constexpr float B = float(float(840.0 / 3.0));
		float hmin = 1e+20;
		float hmax = 0.0;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			if (data.rungs[i] >= params.min_rung) {
				if (ws.x.size() == 0) {
					PRINT("ZERO\n");
				}
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
						count += int(r2 < h2);                       // 1
						const float r = sqrt(r2);                    // 4
						const float q = r * hinv;                    // 1
						flops += 15;
						if (q < 1.f) {                               // 1
							const float q2 = sqr(q);                  // 1
							const float _1mq = 1.f - q;               // 1
							const float _1mq2 = sqr(_1mq);            // 1
							const float _1mq3 = _1mq * _1mq2;         // 1
							const float _1mq4 = _1mq * _1mq3;         // 1
							const float w = A * _1mq4 * fmaf(4.f, q, 1.f); // 4
							const float dwdh = B * _1mq3 * q2 * hinv; // 3
							f += w;                                   // 1
							dfdh += dwdh;                             // 1
							flops += 15;
						}
					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(f);
					shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(count);
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dfdh);
					if (tid == 0) {
						dh = 0.1f * h;
						if (count > 1) {
							f -= SPH_NEIGHBOR_COUNT;
							dh = -f / dfdh;
							if (dh > 0.5f * h) {
								dh = 0.5f * h;
							} else if (dh < -0.5f * h) {
								dh = -0.5f * h;
							}
							error = fabsf(logf(h + dh) - logf(h));
							if (tid == 0) {
								h += dh;
							}
						} else {
							if (count == 0) {
								PRINT("Can't find self\n");
							}
							if (tid == 0) {
								h *= 1.1;
							}
							error = 1.0;
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
					if (iter > 100) {
						if (tid == 0) {
							PRINT("%i %i %e %e\n", iter, count, h, dh);
							PRINT("density solver failed to converge %i\n", ws.x.size());
						}
						__trap();
					}
				} while (error > SPH_SMOOTHLEN_TOLER && !box_xceeded);
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
				data.sa_snk[snki] = false;
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
	while (index < data.nselfs) {
		int flops = 0;
		ws.x0.resize(0);
		ws.y0.resize(0);
		ws.z0.resize(0);
		ws.vx0.resize(0);
		ws.vy0.resize(0);
		ws.vz0.resize(0);
		ws.h0.resize(0);
		ws.ent0.resize(0);
		ws.rungs0.resize(0);
		ws.fvel0.resize(0);
		ws.f00.resize(0);
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
				const int offset = ws.x0.size();
				const int next_size = offset + total;
				ws.x0.resize(next_size);
				ws.y0.resize(next_size);
				ws.z0.resize(next_size);
				ws.vx0.resize(next_size);
				ws.vy0.resize(next_size);
				ws.vz0.resize(next_size);
				ws.ent0.resize(next_size);
				ws.rungs0.resize(next_size);
				ws.fvel0.resize(next_size);
				ws.f00.resize(next_size);
				ws.h0.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x0[k] = x[XDIM];
					ws.y0[k] = x[YDIM];
					ws.z0[k] = x[ZDIM];
					ws.vx0[k] = data.vx[pi];
					ws.vy0[k] = data.vy[pi];
					ws.vz0[k] = data.vz[pi];
					ws.rungs0[k] = data.rungs[pi];
					ws.ent0[k] = data.ent[pi];
					ws.h0[k] = data.h[pi];
					ws.fvel0[k] = data.fvel[pi];
					ws.f00[k] = data.f0[pi];
				}
			}
		}
		if (ws.x0.size() == 0) {
			if (tid == 0)
				PRINT("ZERO %i  %i  %i  %i %i\n", data.nselfs, self.neighbor_range.second - self.neighbor_range.first, data.selfs[index],
						data.neighbors[self.neighbor_range.first], index);
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			int myrung = data.rungs[i];
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			bool active = myrung >= params.min_rung;
			bool semi_active = !active && data.sa_snk[snki];
			bool use = active;
			if (!use && params.phase == 0) {
				use = semi_active;
			}
			if (params.phase == 1 && active && tid == 0) {
				data.dent_snk[snki] = 0.0f;
				data.dvx_snk[snki] = 0.f;
				data.dvy_snk[snki] = 0.f;
				data.dvz_snk[snki] = 0.f;
			}
			const float m = data.m;
			const float minv = 1.f / m;
			const float c0 = float(3.0f / 4.0f / M_PI * SPH_NEIGHBOR_COUNT);
			const float c0inv = 1.0f / c0;
			if (use) {
				const auto myx = data.x[i];
				const auto myy = data.y[i];
				const auto myz = data.z[i];
				const auto myvx = data.vx[i];
				const auto myvy = data.vy[i];
				const auto myvz = data.vz[i];
				const float myh = data.h[i];
				const float myhinv = 1.f / myh;
				const float myh3inv = 1.f / (sqr(myh) * myh);
				const float myrho = m * c0 * myh3inv;
				const float myrhoinv = minv * c0inv * sqr(myh) * myh;
				const float myp = data.ent[i] * powf(myrho, SPH_GAMMA);
				const float myc = sqrtf(SPH_GAMMA * myp * myrhoinv);
				const float myrho1mgammainv = powf(myrho, 1.0f - SPH_GAMMA);
				const float myfvel = data.fvel[i];
				const float myf0 = data.f0[i];
				const float Prho2i = myp * myrhoinv * myrhoinv * myf0;
				const int jmax = round_up(ws.x0.size(), block_size);
				ws.x.resize(0);
				ws.y.resize(0);
				ws.z.resize(0);
				ws.vx.resize(0);
				ws.vy.resize(0);
				ws.vz.resize(0);
				ws.h.resize(0);
				ws.ent.resize(0);
				ws.fvel.resize(0);
				ws.f0.resize(0);
				ws.rungs.resize(0);
				for (int j = tid; j < jmax; j += block_size) {
					bool flag = false;
					int k;
					int total;
					if (j < ws.x0.size()) {
						const auto x = ws.x0[j];
						const auto y = ws.y0[j];
						const auto z = ws.z0[j];
						const float h = ws.h0[j];
						const float dx = distance(x, myx);
						const float dy = distance(y, myy);
						const float dz = distance(z, myz);
						const float h2max = sqr(fmaxf(h, myh));
						const float r2 = sqr(dx, dy, dz);
						if (r2 < h2max) {
							if (params.phase == 0) {
								if (semi_active) {
									if (ws.rungs0[j] >= params.min_rung) {
										flag = true;
									}
								} else {
									flag = true;
								}
							} else {
								flag = true;
							}
						}
					}
					k = flag;
					compute_indices<HYDRO_BLOCK_SIZE>(k, total);
					const int offset = ws.x.size();
					const int next_size = offset + total;
					ws.x.resize(next_size);
					ws.y.resize(next_size);
					ws.z.resize(next_size);
					ws.vx.resize(next_size);
					ws.vy.resize(next_size);
					ws.vz.resize(next_size);
					ws.ent.resize(next_size);
					ws.fvel.resize(next_size);
					ws.f0.resize(next_size);
					ws.h.resize(next_size);
					ws.rungs.resize(next_size);
					if (flag) {
						const int l = offset + k;
						ws.x[l] = ws.x0[j];
						ws.y[l] = ws.y0[j];
						ws.z[l] = ws.z0[j];
						ws.vx[l] = ws.vx0[j];
						ws.vy[l] = ws.vy0[j];
						ws.vz[l] = ws.vz0[j];
						ws.ent[l] = ws.ent0[j];
						ws.rungs[l] = ws.rungs0[j];
						ws.h[l] = ws.h0[j];
						ws.fvel[l] = ws.fvel0[j];
						ws.f0[l] = ws.f00[j];
					}
				}
				float divv = 0.f;
				float dent = 0.f;
				float ddvx = 0.f;
				float ddvy = 0.f;
				float ddvz = 0.f;
				const float ainv = 1.0f / params.a;
				for (int j = tid; j < ws.x.size(); j += block_size) {
					constexpr float gamma = SPH_GAMMA;
					const float dx = distance(myx, ws.x[j]);
					const float dy = distance(myy, ws.y[j]);
					const float dz = distance(myz, ws.z[j]);
					const float h = ws.h[j];
					const float h3 = sqr(h) * h;
					const float r2 = sqr(dx, dy, dz);
					const float hinv = 1. / h;
					const float h3inv = hinv * sqr(hinv);
					const float rho = m * c0 * h3inv;
					const float rhoinv = minv * c0inv * h3;
					const float p = ws.ent[j] * powf(rho, gamma);
					const float c = sqrtf(gamma * p * rhoinv);
					const float cij = 0.5f * (myc + c);
					const float hij = 0.5f * (h + myh);
					const float rho_ij = 0.5f * (rho + myrho);
					const float dvx = myvx - ws.vx[j];
					const float dvy = myvy - ws.vy[j];
					const float dvz = myvz - ws.vz[j];
					const float r = sqrtf(r2);
					const float r2inv = 1.f / (sqr(r) + 0.01f * sqr(hij));
					const float uij = fminf(0.f, hij * (dvx * dx + dvy * dy + dvz * dz) * r2inv);
					const float Piij = (uij * (-float(SPH_ALPHA) * cij + float(SPH_BETA) * uij)) * 0.5f * (myfvel + ws.fvel[j]) / rho_ij;
					const float dWdri = (r < myh) * sph_dWdr_rinv(r, myhinv, myh3inv);
					const float dWdrj = (r < h) * sph_dWdr_rinv(r, hinv, h3inv);
					const float dWdri_x = dx * dWdri;
					const float dWdri_y = dy * dWdri;
					const float dWdri_z = dz * dWdri;
					const float dWdrj_x = dx * dWdrj;
					const float dWdrj_y = dy * dWdrj;
					const float dWdrj_z = dz * dWdrj;
					const float dWdrij_x = 0.5f * (dWdri_x + dWdrj_x);
					const float dWdrij_y = 0.5f * (dWdri_y + dWdrj_y);
					const float dWdrij_z = 0.5f * (dWdri_z + dWdrj_z);
					const float Prho2j = p * rhoinv * rhoinv * ws.f0[j];
					const float dviscx = Piij * dWdrij_x;
					const float dviscy = Piij * dWdrij_y;
					const float dviscz = Piij * dWdrij_z;
					const float dpx = (Prho2j * dWdrj_x + Prho2i * dWdri_x) + dviscx;
					const float dpy = (Prho2j * dWdrj_y + Prho2i * dWdri_y) + dviscy;
					const float dpz = (Prho2j * dWdrj_z + Prho2i * dWdri_z) + dviscz;
					const float dvxdt = -dpx * m;
					const float dvydt = -dpy * m;
					const float dvzdt = -dpz * m;
					divv -= ainv * m * rhoinv * (dvx * dWdri_x + dvy * dWdri_y + dvz * dWdri_z);
					float dt;
					if (params.phase == 0) {
						dt = 0.5f * min(rung_dt[ws.rungs[j]], rung_dt[myrung]) * (params.t0);
					} else if (params.phase == 1) {
						dt = rung_dt[myrung] * (params.t0);
					}
					float dAdt = (dviscx * dvx + dviscy * dvy + dviscz * dvz);
					dAdt *= float(0.5) * m * (SPH_GAMMA - 1.f) * myrho1mgammainv;
					dent += dAdt * dt;
					ddvx += dvxdt * dt;
					ddvy += dvydt * dt;
					ddvz += dvzdt * dt;
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dent);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvx);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvy);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddvz);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(divv);
				if (tid == 0) {
					data.dent_snk[snki] += dent;
					data.dvx_snk[snki] += ddvx;
					data.dvy_snk[snki] += ddvy;
					data.dvz_snk[snki] += ddvz;
					data.divv_snk[snki] += divv;
//					PRINT( "%e %e %e %e %e\n", dent, ddvx, ddvy, ddvy, ddvz, divv);
				}
			}
		}
		shared_reduce_add<int, HYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
//		if( tid == 0 )
//		PRINT( "%i of %i\n", index, data.nselfs);
		__syncthreads();
	}
//	print( "COMPLETE\n");
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
	while (index < data.nselfs) {
		int flops = 0;
		ws.x0.resize(0);
		ws.y0.resize(0);
		ws.z0.resize(0);
		ws.vx0.resize(0);
		ws.vy0.resize(0);
		ws.vz0.resize(0);
		ws.h0.resize(0);
		ws.ent0.resize(0);
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
				const int offset = ws.x0.size();
				const int next_size = offset + total;
				ws.x0.resize(next_size);
				ws.y0.resize(next_size);
				ws.z0.resize(next_size);
				ws.vx0.resize(next_size);
				ws.vy0.resize(next_size);
				ws.vz0.resize(next_size);
				ws.ent0.resize(next_size);
				ws.h0.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x0[k] = x[XDIM];
					ws.y0[k] = x[YDIM];
					ws.z0[k] = x[ZDIM];
					ws.vx0[k] = data.vx[pi];
					ws.vy0[k] = data.vy[pi];
					ws.vz0[k] = data.vz[pi];
					ws.ent0[k] = data.ent[pi];
					ws.h0[k] = data.h[pi];
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			int myrung = data.rungs[i];
			bool use = myrung >= params.min_rung;
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			if (!use && params.phase == 0) {
				use = data.sa_snk[snki];
			}
			const float m = data.m;
			const float minv = 1.f / m;
			const float c0 = float(3.0f / 4.0f / M_PI * SPH_NEIGHBOR_COUNT);
			const float c0inv = 1.0f / c0;
			if (use) {
				const auto myx = data.x[i];
				const auto myy = data.y[i];
				const auto myz = data.z[i];
				const auto myvx = data.vx[i];
				const auto myvy = data.vy[i];
				const auto myvz = data.vz[i];
				const float myh = data.h[i];
				const float myh2 = sqr(myh);
				const float myhinv = 1.f / myh;
				const float myh3inv = 1.f / (sqr(myh) * myh);
				const float myrho = m * c0 * myh3inv;
				const float myrhoinv = minv * c0inv * sqr(myh) * myh;
				const float myp = data.ent[i] * powf(myrho, SPH_GAMMA);
				const float myc = sqrtf(SPH_GAMMA * myp * myrhoinv);
				const int jmax = round_up(ws.x0.size(), block_size);
				ws.x.resize(0);
				ws.y.resize(0);
				ws.z.resize(0);
				ws.vx.resize(0);
				ws.vy.resize(0);
				ws.vz.resize(0);
				ws.h.resize(0);
				ws.ent.resize(0);
				for (int j = tid; j < jmax; j += block_size) {
					bool flag = false;
					int k;
					int total;
					if (j < ws.x0.size()) {
						const auto x = ws.x0[j];
						const auto y = ws.y0[j];
						const auto z = ws.z0[j];
						const float h = ws.h0[j];
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
					const int offset = ws.x.size();
					const int next_size = offset + total;
					ws.x.resize(next_size);
					ws.y.resize(next_size);
					ws.z.resize(next_size);
					ws.vx.resize(next_size);
					ws.vy.resize(next_size);
					ws.vz.resize(next_size);
					ws.ent.resize(next_size);
					ws.h.resize(next_size);
					if (flag) {
						const int l = offset + k;
						ws.x[l] = ws.x0[j];
						ws.y[l] = ws.y0[j];
						ws.z[l] = ws.z0[j];
						ws.vx[l] = ws.vx0[j];
						ws.vy[l] = ws.vy0[j];
						ws.vz[l] = ws.vz0[j];
						ws.ent[l] = ws.ent0[j];
						ws.h[l] = ws.h0[j];
					}
				}
				float vsig_max = 0.f;
				for (int j = tid; j < ws.x.size(); j += block_size) {
					constexpr float gamma = SPH_GAMMA;
					const float dx = distance(myx, ws.x[j]);
					const float dy = distance(myy, ws.y[j]);
					const float dz = distance(myz, ws.z[j]);
					const float h = ws.h[j];
					const float h3 = sqr(h) * h;
					const float r2 = sqr(dx, dy, dz);
					if (r2 != 0.f) {
						const float hinv = 1. / h;
						const float h3inv = hinv * sqr(hinv);
						const float rho = m * c0 * h3inv;
						const float rhoinv = minv * c0inv * h3;
						const float p = ws.ent[j] * powf(rho, gamma);
						const float c = sqrtf(gamma * p * rhoinv);
						const float dvx = myvx - ws.vx[j];
						const float dvy = myvy - ws.vy[j];
						const float dvz = myvz - ws.vz[j];
						const float rinv = rsqrtf(r2);
						const float dv = min(0.f, (dx * dvx + dy * dvy + dz * dvz) * rinv);
						const float this_vsig = myc + c - 3.f * dv;
						vsig_max = fmaxf(vsig_max, this_vsig);
					}
				}
				shared_reduce_max<float, HYDRO_BLOCK_SIZE>(vsig_max);
				if (tid == 0) {
					total_vsig_max = fmaxf(total_vsig_max, vsig_max);
					float dthydro = vsig_max / (params.a * myh);
					if (dthydro > 1.0e-99) {
						dthydro = SPH_CFL / dthydro;
					} else {
						dthydro = 1.0e99;
					}
					const float gx = data.gx_snk[snki];
					const float gy = data.gy_snk[snki];
					const float gz = data.gz_snk[snki];
					char& rung = data.rungs[i];
					const float g2 = sqr(gx, gy, gz);
					const float factor = data.eta * sqrtf(params.a * myh);
					const float dt_grav = fminf(factor / sqrtf(sqrtf(g2 + 1e-15f)), (float) params.t0);
					const float dt = fminf(dt_grav, dthydro);
					const int rung_hydro = ceilf(log2f(params.t0) - log2f(dthydro));
					const int rung_grav = ceilf(log2f(params.t0) - log2f(dt_grav));
					max_rung_hydro = max(max_rung_hydro, rung_hydro);
					max_rung_grav = max(max_rung_hydro, rung_grav);
					rung = max(max((int) max(rung_hydro, rung_grav), max(params.min_rung, (int) rung - 1)), 1);
					//			PRINT( "%i %e %e %e %e\n", rung, dt_grav, gx, gy, gz);
					if (rung < 0 || rung >= MAX_RUNG) {
						PRINT("Rung out of range \n");
					}

				}
			}
		}
		shared_reduce_add<int, HYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	if (tid == 0) {
		atomicMax(&reduce->vsig_max, total_vsig_max);
		atomicMax(&reduce->max_rung_hydro, max_rung_hydro);
		atomicMax(&reduce->max_rung_grav, max_rung_grav);
	}
}

__global__ void sph_cuda_fvels(sph_run_params params, sph_run_cuda_data data, fvels_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	fvels_workspace& ws = workspaces[bid];
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;

	while (index < data.nselfs) {
		int flops = 0;
		ws.x0.resize(0);
		ws.y0.resize(0);
		ws.z0.resize(0);
		ws.vx0.resize(0);
		ws.vy0.resize(0);
		ws.vz0.resize(0);
		ws.h0.resize(0);
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
				compute_indices<HYDRO_BLOCK_SIZE>(j, total);
				const int offset = ws.x0.size();
				const int next_size = offset + total;
				ws.x0.resize(next_size);
				ws.y0.resize(next_size);
				ws.z0.resize(next_size);
				ws.vx0.resize(next_size);
				ws.vy0.resize(next_size);
				ws.vz0.resize(next_size);
				ws.h0.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x0[k] = x[XDIM];
					ws.y0[k] = x[YDIM];
					ws.z0[k] = x[ZDIM];
					ws.vx0[k] = data.vx[pi];
					ws.vy0[k] = data.vy[pi];
					ws.vz0[k] = data.vz[pi];
					ws.h0[k] = data.h[pi];
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {

			const int snki = self.sink_part_range.first - self.part_range.first + i;
			int myrung = data.rungs[i];
			bool use = myrung >= params.min_rung;
			const float m = data.m;
			const float minv = 1.f / m;
			const float c0 = float(3.0f / 4.0f / M_PI * SPH_NEIGHBOR_COUNT);
			const float c0inv = 1.0f / c0;
			if (use) {
				const auto myx = data.x[i];
				const auto myy = data.y[i];
				const auto myz = data.z[i];
				const auto myvx = data.vx[i];
				const auto myvy = data.vy[i];
				const auto myvz = data.vz[i];
				const float myh = data.h[i];
				const float myh2 = sqr(myh);
				const float myhinv = 1.f / myh;
				const float myh3inv = 1.f / (sqr(myh) * myh);
				const float myrho = m * c0 * myh3inv;
				const float myrhoinv = minv * c0inv * sqr(myh) * myh;
				const float myp = data.ent_snk[snki] * powf(myrho, SPH_GAMMA);
				const float myc = sqrtf(SPH_GAMMA * myp * myrhoinv);
				const int jmax = round_up(ws.x0.size(), block_size);
				ws.x.resize(0);
				ws.y.resize(0);
				ws.z.resize(0);
				ws.vx.resize(0);
				ws.vy.resize(0);
				ws.vz.resize(0);
				ws.h.resize(0);
				for (int j = tid; j < jmax; j += block_size) {
					bool flag = false;
					int k;
					int total;
					if (j < ws.x0.size()) {
						const auto x = ws.x0[j];
						const auto y = ws.y0[j];
						const auto z = ws.z0[j];
						const float h = ws.h0[j];
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
					const int offset = ws.x.size();
					const int next_size = offset + total;
					ws.x.resize(next_size);
					ws.y.resize(next_size);
					ws.z.resize(next_size);
					ws.vx.resize(next_size);
					ws.vy.resize(next_size);
					ws.vz.resize(next_size);
					ws.h.resize(next_size);
					if (flag) {
						const int l = offset + k;
						ws.x[l] = ws.x0[j];
						ws.y[l] = ws.y0[j];
						ws.z[l] = ws.z0[j];
						ws.vx[l] = ws.vx0[j];
						ws.vy[l] = ws.vy0[j];
						ws.vz[l] = ws.vz0[j];
						ws.h[l] = ws.h0[j];
					}
				}
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
				for (int j = tid; j < ws.x.size(); j += block_size) {
					const float dx = distance(myx, ws.x[j]);
					const float dy = distance(myy, ws.y[j]);
					const float dz = distance(myz, ws.z[j]);
					const float h = ws.h[j];
					const float h3 = sqr(h) * h;
					const float r2 = sqr(dx, dy, dz);
					if (r2 != 0.f) {
						const float hinv = 1. / h;
						const float h3inv = hinv * sqr(hinv);
						const float rho = m * c0 * h3inv;
						const float rhoinv = minv * c0inv * h3;
						const float vx = ws.vx[j];
						const float vy = ws.vy[j];
						const float vz = ws.vz[j];
						const float r = sqrt(r2);
						const float dWdr = sph_dWdr_rinv(r, myhinv, myh3inv);
						const float tmp = m * dWdr * rhoinv;
						const float dWdr_x = dx * tmp;
						const float dWdr_y = dy * tmp;
						const float dWdr_z = dz * tmp;
						const float dvx = vx - myvx;
						const float dvy = vy - myvy;
						const float dvz = vz - myvz;
						dvx_dx += dvx * dWdr_x;
						dvx_dy += dvx * dWdr_y;
						dvx_dz += dvx * dWdr_z;
						dvy_dx += dvy * dWdr_x;
						dvy_dy += dvy * dWdr_y;
						dvy_dz += dvy * dWdr_z;
						dvz_dx += dvz * dWdr_x;
						dvz_dy += dvz * dWdr_y;
						dvz_dz += dvz * dWdr_z;
						drho_dh += sph_h4dWdh(r, myhinv);
					}
				}
				float div_v = dvx_dx + dvy_dy + dvz_dz;
				float curl_vx = dvz_dy - dvy_dz;
				float curl_vy = -dvz_dx + dvx_dz;
				float curl_vz = dvy_dx - dvx_dy;
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(div_v);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vx);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vy);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vz);
				const float sw = 1e-4f * myc / myh;
				const float abs_div_v = fabsf(div_v);
				const float abs_curl_v = sqrtf(sqr(curl_vx, curl_vy, curl_vz));
				const float fvel = abs_div_v / (abs_div_v + abs_curl_v + sw);
				const float c0 = drho_dh * 4.0f * float(M_PI) / (9.0f * SPH_NEIGHBOR_COUNT);
				const float fpre = 1.0f / (1.0f + c0);
				if (tid == 0) {
					data.fvel_snk[snki] = fvel;
					data.f0_snk[snki] = fpre;
					//				PRINT( "%e %e\n", fvel, fpre);
				}
			}
		}
		shared_reduce_add<int, HYDRO_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
}

sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream) {
	timer tm;
	sph_run_return rc;
	sph_reduction* reduce;
	int nblocks;
	CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(sph_reduction)));
	reduce->counter = reduce->flag = 0;
	reduce->hmin = std::numeric_limits<float>::max();
	reduce->hmax = 0.0f;
	reduce->flops = 0.0;
	reduce->vsig_max = 0.0;
	reduce->max_rung_grav = 0;
	reduce->max_rung_hydro = 0;
	switch (params.run_type) {
	case SPH_RUN_SMOOTHLEN: {
		smoothlen_workspace* workspaces;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) sph_cuda_smoothlen, SMOOTHLEN_BLOCK_SIZE, 0));
		nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaMalloc(&workspaces, sizeof(smoothlen_workspace) * nblocks));
		tm.start();
		sph_cuda_smoothlen<<<nblocks, SMOOTHLEN_BLOCK_SIZE,0,stream>>>(params,data,workspaces,reduce);
		cuda_stream_synchronize(stream);
		tm.stop();
		CUDA_CHECK(cudaFree(workspaces));
		rc.rc = reduce->flag;
		rc.hmin = reduce->hmin;
		rc.hmax = reduce->hmax;
		tm.stop();
		PRINT("Kernel ran at %e GFLOPS\n", reduce->flops / (1024 * 1024 * 1024) / tm.read());
	}
		break;
	case SPH_RUN_MARK_SEMIACTIVE: {
		mark_semiactive_workspace* workspaces;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) sph_cuda_mark_semiactive, SMOOTHLEN_BLOCK_SIZE, 0));
		nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaMalloc(&workspaces, sizeof(mark_semiactive_workspace) * nblocks));
		tm.start();
		sph_cuda_mark_semiactive<<<nblocks, SMOOTHLEN_BLOCK_SIZE,0,stream>>>(params,data,workspaces,reduce);
		cuda_stream_synchronize(stream);
		tm.stop();
		CUDA_CHECK(cudaFree(workspaces));
		tm.stop();
		PRINT("Kernel ran at %e GFLOPS\n", reduce->flops / (1024 * 1024 * 1024) / tm.read());
	}
		break;
	case SPH_RUN_HYDRO: {
		hydro_workspace* workspaces;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) sph_cuda_hydro, HYDRO_BLOCK_SIZE, 0));
		nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaMalloc(&workspaces, sizeof(hydro_workspace) * nblocks));
		tm.start();
		sph_cuda_hydro<<<nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,workspaces,reduce);

		cuda_stream_synchronize(stream);
		tm.stop();
		CUDA_CHECK(cudaFree(workspaces));
		tm.stop();
	}
		break;
	case SPH_RUN_COURANT: {
		courant_workspace* workspaces;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) sph_cuda_courant, HYDRO_BLOCK_SIZE, 0));
		nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaMalloc(&workspaces, sizeof(courant_workspace) * nblocks));
		sph_cuda_courant<<<nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,workspaces,reduce);
		cuda_stream_synchronize(stream);
		CUDA_CHECK(cudaFree(workspaces));
		rc.max_vsig = reduce->vsig_max;
		rc.max_rung_grav = reduce->max_rung_grav;
		rc.max_rung_hydro = reduce->max_rung_hydro;
	}
		break;
	case SPH_RUN_FVELS: {
		fvels_workspace* workspaces;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) sph_cuda_fvels, HYDRO_BLOCK_SIZE, 0));
		nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaMalloc(&workspaces, sizeof(fvels_workspace) * nblocks));
		sph_cuda_fvels<<<nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,workspaces,reduce);
		cuda_stream_synchronize(stream);
		CUDA_CHECK(cudaFree(workspaces));

	}
		break;
	}
	CUDA_CHECK(cudaFree(reduce));

	return rc;
}
