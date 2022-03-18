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
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/kernel.hpp>

static __constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6),
		1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
				/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24),
		1.0 / (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

#define WORKSPACE_SIZE (160*1024)
#define HYDRO_SIZE (8*1024)

__managed__ cuda_mem* memory;

template<class T>
class device_vector {
	int sz;
	int cap;
	T* ptr;
public:
	__device__ device_vector() {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			sz = 0;
			cap = 0;
			ptr = nullptr;
		}
		__syncthreads();
	}
	__device__ ~device_vector() {
		const int& tid = threadIdx.x;
		__syncthreads();
		if (tid == 0) {
			if (ptr) {
				memory->free(ptr);
			}
		}
		__syncthreads();
	}
	__device__ void shrink_to_fit() {
		const int& tid = threadIdx.x;
		const int& block_size = blockDim.x;
		__shared__ T* new_ptr;
		__syncthreads();
		int new_cap = max(1024 / sizeof(T), (size_t) 1);
		while (new_cap < sz) {
			new_cap *= 2;
		}
		if (tid == 0) {
			if (new_cap < cap) {
				new_ptr = (T*) memory->allocate(sizeof(T) * new_cap);
				if (new_ptr == nullptr) {
					PRINT("OOM in device_vector while requesting %i! \n", new_cap);
					__trap();

				}
			}
		}
		__syncthreads();
		if (ptr && new_cap < cap) {
			for (int i = tid; i < sz; i += block_size) {
				new_ptr[i] = ptr[i];
			}
		}
		__syncthreads();
		if (tid == 0 && new_cap < cap) {
			if (ptr) {
				memory->free(ptr);
			}
			ptr = new_ptr;
			cap = new_cap;
		}
		__syncthreads();
	}
	__device__
	void resize(int new_sz) {
		const int& tid = threadIdx.x;
		const int& block_size = blockDim.x;
		if (new_sz <= cap) {
			__syncthreads();
			if (tid == 0) {
				sz = new_sz;
			}
			__syncthreads();
		} else {
			__shared__ T* new_ptr;
			__syncthreads();
			int new_cap = max(1024 / sizeof(T), (size_t) 1);
			if (tid == 0) {
				while (new_cap < new_sz) {
					new_cap *= 2;
				}
				new_ptr = (T*) memory->allocate(sizeof(T) * new_cap);
				if (new_ptr == nullptr) {
					PRINT("OOM in device_vector while requesting %i! \n", new_cap);
					__trap();

				}
			}
			__syncthreads();
			if (ptr) {
				for (int i = tid; i < sz; i += block_size) {
					new_ptr[i] = ptr[i];
				}
			}
			__syncthreads();
			if (tid == 0) {
				if (ptr) {
					memory->free(ptr);
				}
				ptr = new_ptr;
				sz = new_sz;
				cap = new_cap;
			}
			__syncthreads();
		}
	}
	__device__
	int size() const {
		return sz;
	}
	__device__ T& operator[](int i) {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");
			__trap();
		}
#endif
		return ptr[i];
	}
	__device__
	                                                                                                                                                                                                                                                       const T& operator[](int i) const {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in device_vector\n");
			__trap();
		}
#endif
		return ptr[i];
	}
};

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

struct hydro_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};

struct parabolic_record1 {
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
	float eint;
	float alpha;
	float fpre;
	float crsv;
	float shearv;
	float divv;
};

struct parabolic_record2 {
	float gamma;
	float eint;
	float eint0;
	float shearv;
	float fpre;
	float gradT;
	float mmw;
	array<float, NCHEMFRACS> chem;
	array<float, NCHEMFRACS> chem0;
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
	float eint;
	float mmw;
	float gamma;
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

struct hydro_workspace {
	device_vector<hydro_record1> rec1_main;
	device_vector<hydro_record2> rec2_main;
	device_vector<hydro_record1> rec1;
	device_vector<hydro_record2> rec2;
};
struct parabolic_workspace {
	device_vector<parabolic_record1> rec1_main;
	device_vector<parabolic_record2> rec2_main;
	device_vector<parabolic_record1> rec1;
	device_vector<parabolic_record2> rec2;
};

struct xsph_workspace {
	device_vector<xsph_record1> rec1_main;
	device_vector<xsph_record2> rec2_main;
	device_vector<xsph_record1> rec1;
	device_vector<xsph_record2> rec2;
};

struct aux_workspace {
	device_vector<aux_record1> rec1_main;
	device_vector<aux_record2> rec2_main;
	device_vector<aux_record1> rec1;
	device_vector<aux_record2> rec2;
};

#define SMOOTHLEN_BLOCK_SIZE 512
#define HYDRO_BLOCK_SIZE 32

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
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs[i] >= params.min_rung;
			const bool semiactive = !active && data.sa_snk[snki];
			const bool use = params.phase == 0 ? active : semiactive;
			if (use) {
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
				float drho_dh;
				float dpot_dh;
				do {
					float max_dh = h / sqrtf(iter + 100);
					const float hinv = 1.f / h; // 4
					const float h2 = sqr(h);    // 1
					count = 0;
					f = 0.f;
					dfdh = 0.f;
					drho_dh = 0.f;
					dpot_dh = 0.f;
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
							const float dwdq = dkernelW_dq(q);
							const float dwdh = -q * dwdq * hinv; // 3
							const float pot = kernelPot(q);
							const float force = kernelFqinv(q) * q;
							drho_dh -= (3.f * kernelW(q) + q * dwdq);
							dpot_dh -= (pot - q * force);
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
						error = fmaxf(fabsf(f) / (data.N * float(3.0 / (4.0 * M_PI))), fabs(dh / h));
					} else {
						error = 1.f;
					}
					__syncthreads();
					if (tid == 0 && error > SPH_SMOOTHLEN_TOLER) {
						h += dh;
						if (iter > 30) {
							PRINT("over iteration on h solve - %i %e %e %e %e %i\n", iter, h, dh, max_dh, error, count);
						}
					}
					__syncthreads();
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
					if (tid == 0 && box_xceeded) {
						h -= dh;
					}
					__syncthreads();
					iter++;
					if (max_dh / h < SPH_SMOOTHLEN_TOLER) {
						if (tid == 0) {
							PRINT("density solver failed to converge %i\n", ws.x.size());
							__trap();
						}
					}
					shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(box_xceeded);
				} while (error > SPH_SMOOTHLEN_TOLER && !box_xceeded);
				if (!box_xceeded) {
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(drho_dh);
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dpot_dh);
					drho_dh *= 4.0f * float(M_PI) / (9.0f * data.N);
					dpot_dh *= sqr(h) * 4.0f * float(M_PI) / (9.f * data.N);
					const float fpre = 1.0f / (1.0f + drho_dh);
					data.fpre_snk[snki] = fpre;
					data.fpot_snk[snki] = dpot_dh * fpre;
					if (tid == 0 && h <= 0.f) {
						PRINT("Less than ZERO H! sph.cu %e\n", h);
						__trap();
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
					ws.rec2_main[k].vx = data.vx[pi];
					ws.rec2_main[k].vy = data.vy[pi];
					ws.rec2_main[k].vz = data.vz[pi];
					ws.rec2_main[k].eint = data.eint[pi];
					ws.rec2_main[k].alpha = data.alpha[pi];
					ws.rec2_main[k].fpre = data.fpre[pi];
					ws.rec2_main[k].crsv = data.crsv[pi];
					ws.rec2_main[k].shearv = data.shearv[pi];
					ws.rec2_main[k].divv = data.divv[pi];
					if (data.chemistry) {
						ws.rec2_main[k].gamma = data.gamma[pi];
					} else {
						ws.rec2_main[k].gamma = data.def_gamma;
					}
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
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
				float gamma_i;
				if (data.chemistry || params.conduction) {
					gamma_i = data.gamma[i];
				} else {
					gamma_i = data.def_gamma;
				}
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
				const float h2_i = sqr(h_i);
				const float hinv_i = 1.f / h_i;
				const float h3inv_i = (sqr(hinv_i) * hinv_i);
				const float rho_i = m * c0 * h3inv_i;
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;
				const float eint_i = data.eint[i];
				const float alpha_i = data.alpha[i];
				const float p_i = fmaxf(eint_i * rho_i * (gamma_i - 1.f), 0.f);
				const float c_i = sqrtf(gamma_i * p_i * rhoinv_i);
				const float crsv_i = data.crsv[i];
				const float shearv_i = data.shearv[i];
				const float divv_i = data.divv[i];
				const float fpre_i = data.fpre[i];
				const float balsara_i = fabs(divv_i) / (fabs(divv_i) + crsv_i + 1e-4f * c_i * hinv_i);
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
						const float x_ij = distance(x_j, x_i);
						const float y_ij = distance(y_j, y_i);
						const float z_ij = distance(z_j, z_i);
						const float h_j = rec.h;
						const float h2_j = sqr(h_j);
						const float r2 = sqr(x_ij, y_ij, z_ij);
						if (r2 < fmaxf(h2_i, h2_j)) {
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
				float ax = 0.f;
				float ay = 0.f;
				float az = 0.f;
				float de_dt = 0.f;
				float dtinv_cfl = 0.f;
				float one = 0.0f;
				float vsig = 0.0f;
				float Ri = 0.f;
				const float& adot = params.adot;
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					const auto rec1 = ws.rec1[j];
					const auto rec2 = ws.rec2[j];
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float vx_j = rec2.vx;
					const float vy_j = rec2.vy;
					const float vz_j = rec2.vz;
					const float divv_j = rec2.divv;
					const float crsv_j = rec2.crsv;
					const float fpre_j = rec2.fpre;
					const float h_j = rec1.h;
					const float h2_j = sqr(h_j);
					const float hinv_j = 1.f / h_j;															// 4
					const float h3inv_j = sqr(hinv_j) * hinv_j;
					const float rho_j = m * c0 * h3inv_j;													// 2
					const float rhoinv_j = minv * c0inv * sqr(h_j) * h_j;								// 5
					const float eint_j = rec2.eint;
					const float gamma_j = rec2.gamma;
					const float p_j = fmaxf(eint_j * rho_j * (gamma_j - 1.f), 0.f);								// 5
					const float c_j = sqrtf(gamma_j * p_j * rhoinv_j);									// 6
					const float alpha_j = rec2.alpha;
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
					const float balsara_j = fabs(divv_j) / (fabs(divv_j) + crsv_j + 1e-4f * c_j * hinv_j);
					const float vdotx_ij = fminf(0.0f, x_ij * vx_ij + y_ij * vy_ij + z_ij * vz_ij);
					const float w_ij = vdotx_ij * rinv;
					const float q_i = r * hinv_i;								// 1
					const float q_j = r * hinv_j;
					const float sqrtrho_i = sqrtf(rho_i);
					const float sqrtrho_j = sqrtf(rho_j);
					const float h_ij = sqrtf(h_i * h_j);
					const float rho_ij = sqrtrho_i * sqrtrho_j;
					const float rhoinv_ij = 1.f / rho_ij;
					const float c_ij = (sqrtrho_i * c_i + sqrtrho_j * c_j) / (sqrtrho_i + sqrtrho_j);
					const float vsig_ij = (c_ij - params.beta * w_ij);								// 1
					const float alpha_ij = 0.5f * (alpha_i * balsara_i + alpha_j * balsara_j);
					const float viscco_ij = alpha_ij * h_ij * vsig_ij;
					const float dvisc_ij = -viscco_ij * w_ij * rinv * rhoinv_ij;					//
					const float W_i = kernelW(q_i) * h3inv_i;
					const float dWdr_i = fpre_i * dkernelW_dq(q_i) * hinv_i * h3inv_i;
					const float dWdr_j = fpre_j * dkernelW_dq(q_j) * hinv_j * h3inv_j;
					const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);
					const float dWdr_x_i = x_ij * rinv * dWdr_i;
					const float dWdr_y_i = y_ij * rinv * dWdr_i;
					const float dWdr_z_i = z_ij * rinv * dWdr_i;
					const float dWdr_x_j = x_ij * rinv * dWdr_j;
					const float dWdr_y_j = y_ij * rinv * dWdr_j;
					const float dWdr_z_j = z_ij * rinv * dWdr_j;
					const float dWdr_x_ij = x_ij * rinv * dWdr_ij;
					const float dWdr_y_ij = y_ij * rinv * dWdr_ij;
					const float dWdr_z_ij = z_ij * rinv * dWdr_ij;
					const float dp_i = p_i * sqr(rhoinv_i);
					const float dp_j = p_j * sqr(rhoinv_j);
					const float dvx_dt = -m * ainv * (dp_i * dWdr_x_i + dp_j * dWdr_x_j + dvisc_ij * dWdr_x_ij);
					const float dvy_dt = -m * ainv * (dp_i * dWdr_y_i + dp_j * dWdr_y_j + dvisc_ij * dWdr_y_ij);
					const float dvz_dt = -m * ainv * (dp_i * dWdr_z_i + dp_j * dWdr_z_j + dvisc_ij * dWdr_z_ij);
					de_dt += ainv * dp_i * (vx0_ij * dWdr_x_i + vy0_ij * dWdr_y_i + vz0_ij * dWdr_z_i) * m;
					de_dt += 0.5f * ainv * dvisc_ij * (vx0_ij * dWdr_x_ij + vy0_ij * dWdr_y_ij + vz0_ij * dWdr_z_ij) * m;
					ax += dvx_dt;
					ay += dvy_dt;
					az += dvz_dt;
					vsig = fmaxf(vsig, vsig_ij);
					one += m * rhoinv_i * W_i;
					Ri += copysign(m * rhoinv_i * W_i, divv_j);
					const float hinv_ij = 1.f / h_ij;
					dtinv_cfl = fmaxf(dtinv_cfl, c_ij * hinv_ij + 0.6f * viscco_ij * hinv_j * hinv_i);
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(Ri);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(one);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(de_dt);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ax);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ay);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(az);
				shared_reduce_max<float, HYDRO_BLOCK_SIZE>(vsig);
				shared_reduce_max<float, HYDRO_BLOCK_SIZE>(dtinv_cfl);
				if (fabs(1. - one) > 1.0e-4 && tid == 0) {
					PRINT("one is off %e \n", one);
					__trap();
				}
				if (tid == 0) {
					ax += gx_i;
					ay += gy_i;
					az += gz_i;
					de_dt += (5.f - 3.f * gamma_i) * params.adot * ainv * eint_i;
					data.dvx_con[snki] = ax;
					data.dvy_con[snki] = ay;
					data.dvz_con[snki] = az;
					data.deint_con[snki] = de_dt;
					float dt_tot;
					if (params.phase == 1) {
						const float dtinv_eint1 = params.a * fabsf((gamma_i - 1.f) * (divv_i - 3.f * params.adot * ainv));
						const float dtinv_eint2 = fabs((5.f - 3.f * gamma_i) * params.adot);
						const float dtinv_eint = dtinv_eint1 + dtinv_eint2;
						float dtinv_hydro1 = 1.0e-30f;
						dtinv_hydro1 = fmaxf(dtinv_hydro1, dtinv_eint);
						dtinv_hydro1 = fmaxf(dtinv_hydro1, dtinv_cfl);
						const float a2 = sqr(ax, ay, az);
						const float dtinv_acc = sqrtf(sqrtf(a2) * hinv_i);
						const float dtinv_hydro2 = dtinv_acc;
						float dthydro = fminf(params.cfl * params.a / (dtinv_hydro1 + 1e-30f), data.eta * sqrtf(params.a) / (dtinv_hydro2 + 1e-30f));
						const float g2 = sqr(gx_i, gy_i, gz_i);
						const float dtinv_grav = sqrtf(sqrtf(g2) * hinv_i);
						float dtgrav = data.eta * sqrtf(params.a) / (dtinv_grav + 1e-30f);
						dthydro = fminf(dthydro, params.max_dt);
						dtgrav = fminf(dtgrav, params.max_dt);
						dt_tot = fminf(dtgrav, dthydro);
						total_vsig_max = fmaxf(total_vsig_max, dtinv_hydro1 * h_i);
						char& rung = data.rungs[i];
						data.oldrung_snk[snki] = rung;
						const int rung_hydro = ceilf(log2f(params.t0) - log2f(dthydro));
						const int rung_grav = ceilf(log2f(params.t0) - log2f(dtgrav));
						max_rung_hydro = max(max_rung_hydro, rung_hydro);
						max_rung_grav = max(max_rung_grav, rung_grav);
						if (h_i < data.hstar0) {
							rung = max(max((int) rung_grav, max(params.min_rung, (int) rung - 1)), 1);
							data.deint_con[snki] = 0.f;
							data.dvx_con[snki] = gx_i;
							data.dvy_con[snki] = gy_i;
							data.dvz_con[snki] = gz_i;
						} else {
							rung = max(max((int) max(rung_hydro, rung_grav), max(params.min_rung, (int) rung - 1)), 1);
						}
						max_rung = max(max_rung, rung);
						if (rung < 0 || rung >= MAX_RUNG) {
							if (tid == 0) {
								PRINT("Rung out of range \n");
								__trap();
							}
						}
					}
					float dalpha_dt;
					if (params.tau > 0.f) {
						const float ddivv_dt = (divv_i - data.divv0_snk[snki]) / (params.tau - data.taux0_snk[snki]);
						const float A2 = sqr(2.f * sqr(sqr(1.f - Ri)) * divv_i);
						const float limiter = A2 / (A2 + sqr(shearv_i) + 1e-30f);
						const float S = limiter * sqr(h_i) * fmaxf(0.f, -ddivv_dt) * params.a;
						const float alpha_targ = fmaxf(params.alpha1 * S / (S + sqr(vsig) + 1e-30f), params.alpha0);
						const float tauinv = (alpha_i < alpha_targ ? 1.f / dt_tot : params.alpha_decay * vsig * hinv_i * ainv);
						dalpha_dt = (alpha_targ - alpha_i) * tauinv;
					} else {
						dalpha_dt = 0.f;
					}
					data.dalpha_con[snki] = dalpha_dt;
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

__global__ void sph_cuda_parabolic(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ parabolic_workspace ws;
	new (&ws) parabolic_workspace();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	const float ainv = 1.0f / params.a;
	const double kb = (double) constants::kb * sqr((double) params.code_to_s) / ((double) params.code_to_g * sqr((double) params.code_to_cm));
	const double mh = (double) constants::mh / (double) params.code_to_g;
	const float code_to_density = powf((double) params.code_to_cm, -3.0f) * params.code_to_g;
	const float lambda_e0 = powf(3.0f, 1.5f) / (4.0f * sqrtf(float(M_PI)) * powf(constants::e, 4.));
	const float cv0 = kb / mh;
	const float kome = (double) constants::kb / (double) constants::me;
	const float kappa0 = 1.31f * constants::kb * sqr(params.a);
	const float code_dif_to_cgs = (double) params.code_to_g / (double) params.code_to_s / (double) params.code_to_s;
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
					ws.rec2_main[k].eint = data.eint[pi];
					ws.rec2_main[k].eint0 = data.eint0[pi];
					ws.rec2_main[k].fpre = data.fpre[pi];
					ws.rec2_main[k].shearv = data.shearv[pi];
					if (data.chemistry) {
						ws.rec2_main[k].gamma = data.gamma[pi];
						ws.rec2_main[k].chem = data.chem[pi];
						ws.rec2_main[k].chem0 = data.chem0[pi];
					} else {
						ws.rec2_main[k].gamma = data.def_gamma;
					}
					if (data.conduction) {
						ws.rec2_main[k].mmw = data.mmw[pi];
						ws.rec2_main[k].gradT = data.gradT[pi];
					}
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			int rung_i = data.rungs[i];
			const bool active = rung_i >= params.min_rung;
			const bool semiactive = !active && data.sa_snk[snki];
			const bool use = active || semiactive;
			const float m = data.m;
			const float minv = 1.f / m;
			const float c0 = float(3.0f / 4.0f / M_PI * data.N);
			const float c0inv = 1.0f / c0;
			if (use) {
				const auto x_i = data.x[i];
				const auto y_i = data.y[i];
				const auto z_i = data.z[i];
				const float h_i = data.h[i];
				float gamma_i;
				if (params.conduction) {
					gamma_i = data.gamma[i];
				} else {
					gamma_i = data.def_gamma;
				}
				const float h2_i = sqr(h_i);
				const float hinv_i = 1.f / h_i;
				const float h3inv_i = (sqr(hinv_i) * hinv_i);
				const float rho_i = m * c0 * h3inv_i;
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;
				const float eint_i = data.eint[i];
				const float shearv_i = data.shearv[i];
				const float fpre_i = data.fpre[i];
				array<float, NCHEMFRACS> chem_i;
				if (data.chemistry) {
					chem_i = data.chem[i];
				}
				float mu_i, T_i, ne_i, gradT_i;
				if (params.conduction) {
					mu_i = data.mmw[i];
					gradT_i = data.gradT[i];
					T_i = mu_i * eint_i / cv0 * sqr(ainv) / (gamma_i - 1.f);
					T_i = fmaxf(T_i, TMIN);
					ne_i = chem_i[CHEM_HP] - chem_i[CHEM_HN] + chem_i[CHEM_HEP] + 2.0f * chem_i[CHEM_HEP];
					ne_i *= rho_i * code_to_density / mh * sqr(ainv) * ainv;
				}
				const float dt_i = rung_dt[rung_i] * params.t0;
				const int jmax = round_up(ws.rec1_main.size(), block_size);
				ws.rec1.resize(0);
				ws.rec2.resize(0);
				for (int j = tid; j < jmax; j += block_size) {
					bool flag = false;
					int k;
					int total;
					if (j < ws.rec1_main.size()) {
						const auto rec = ws.rec1_main[j];
						const auto rung_j = rec.rung;
						if (active || rung_j >= params.min_rung) {
							const auto x_j = rec.x;
							const auto y_j = rec.y;
							const auto z_j = rec.z;
							const float x_ij = distance(x_j, x_i);
							const float y_ij = distance(y_j, y_i);
							const float z_ij = distance(z_j, z_i);
							const float h_j = rec.h;
							const float h2_j = sqr(h_j);
							const float r2 = sqr(x_ij, y_ij, z_ij);
							if (r2 < fmaxf(h2_i, h2_j)) {
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
				float den = 0.f;
				float den_eint = 0.f;
				float num_eint = 0.f;
				array<float, NCHEMFRACS> num_chem;
				if (data.chemistry) {
					for (int fi = 0; fi < NCHEMFRACS; fi++) {
						num_chem[fi] = 0.f;
					}
				}
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					const auto rec1 = ws.rec1[j];
					const auto rec2 = ws.rec2[j];
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float fpre_j = rec2.fpre;
					const float shearv_j = rec2.shearv;
					const float h_j = rec1.h;
					const auto rung_j = rec1.rung;
					const auto chem_j = rec2.chem;
					const float dt_j = rung_dt[rung_j] * params.t0;
					const float h2_j = sqr(h_j);
					const float hinv_j = 1.f / h_j;															// 4
					const float h3inv_j = sqr(hinv_j) * hinv_j;
					const float rho_j = m * c0 * h3inv_j;													// 2
					const float rhoinv_j = minv * c0inv * sqr(h_j) * h_j;								// 5
					const float eint_j = rec2.eint;
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float r2 = sqr(x_ij, y_ij, z_ij);
					if (r2 > 0.f) {
						const float r = sqrt(r2);
						const float rinv = 1.0f / r;
						const float q_i = r * hinv_i;								// 1
						const float q_j = r * hinv_j;
						const float sqrtrho_i = sqrtf(rho_i);
						const float sqrtrho_j = sqrtf(rho_j);
						const float h_ij = sqrtf(h_i * h_j);
						const float rho_ij = sqrtrho_i * sqrtrho_j;
						const float rhoinv_ij = 1.f / rho_ij;
						const float dWdr_i = fpre_i * dkernelW_dq(q_i) * hinv_i * h3inv_i;
						const float dWdr_j = fpre_j * dkernelW_dq(q_j) * hinv_j * h3inv_j;
						const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);
						const float dWdr_x_ij = x_ij * rinv * dWdr_ij;
						const float dWdr_y_ij = y_ij * rinv * dWdr_ij;
						const float dWdr_z_ij = z_ij * rinv * dWdr_ij;
						const float dWdr_rinv = (x_ij * dWdr_x_ij + y_ij * dWdr_y_ij + z_ij * dWdr_z_ij) * sqr(rinv);
						const float difco_ij = SPH_DIFFUSION_C * h_i * h_j * 0.5f * (shearv_i + shearv_j);
						const float dt_ij = fminf(dt_i, dt_j);
						const float phi_ij = -2.f * difco_ij * dWdr_rinv * m / rho_ij * dt_ij;
						den += phi_ij;
						den_eint += phi_ij;
						num_eint += phi_ij * eint_j;
						if (data.chemistry) {
							for (int fi = 0; fi < NCHEMFRACS; fi++) {
								num_chem[fi] += phi_ij * chem_j[fi];
							}
						}
						if (params.conduction) {
							const float gamma_j = rec2.gamma;
							const float gradT_j = rec2.gradT;
							const float mu_j = rec2.mmw;
							float T_j = mu_j * eint_j / cv0 * sqr(ainv) / (gamma_j - 1.f);
							T_j = fmaxf(T_j, TMIN);
							const float T_ij = 0.5f * (T_i / eint_i + T_j / eint_j)
									* sqr((sqrtrho_i * sqrtf(eint_i) + sqrtrho_j * sqrtf(eint_j)) / (sqrtrho_i + sqrtrho_j));
							float ne_j = chem_j[CHEM_HP] - chem_j[CHEM_HN] + chem_j[CHEM_HEP] + 2.0f * chem_j[CHEM_HEP];
							ne_j *= rho_j * code_to_density / mh * sqr(ainv) * ainv;
							const float ne_ij = sqrtf(ne_i * ne_j);
							if (ne_ij > 0.f) {
								const float colog_ij = 23.5f - log(sqrt(ne_ij) * pow(T_ij, -1.2f)) - sqrt((1e-5f + sqr(log(T_ij) - 2.f)) / 16.0f);
								const float lambda_e_ij = lambda_e0 / colog_ij / params.code_to_cm * ainv * sqr(float(constants::kb) * T_ij) / ne_ij;
								const float gradT_ij = 0.5f * (gradT_i + gradT_j);
								float correction = 1.f;
								if (gradT_ij > 0.f) {
									correction = 1.0f / (1.0f + 4.2f * lambda_e_ij / gradT_ij);
								}
								const float kappa_ij = correction * kappa0 * ne_ij * lambda_e_ij * sqrtf(kome * T_ij) / code_dif_to_cgs;
								const float phi_ij = -2.f * kappa_ij * dWdr_rinv * m / sqr(rho_ij) * dt_ij;
								num_eint += phi_ij * eint_j;
								den_eint += phi_ij;
							}
						}
					}
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(den);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(num_eint);
				if (data.chemistry) {
					for (int fi = 0; fi < NCHEMFRACS; fi++) {
						shared_reduce_add<float, HYDRO_BLOCK_SIZE>(num_chem[fi]);
					}
				}
				if (tid == 0.0) {
					den += 1.f;
					den_eint += 1.f;
					num_eint += data.eint0[i];
					data.deint_snk[snki] = num_eint / den_eint - eint_i;
					if (data.chemistry) {
						for (int fi = 0; fi < NCHEMFRACS; fi++) {
							num_chem[fi] += data.chem0[i][fi];
							data.dchem_snk[snki][fi] = num_chem[fi] / den - chem_i[fi];
						}
					}
				}
			}
		}
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}
	(&ws)->~parabolic_workspace();
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
					contains = self.outer_box.contains(x);
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
					if (params.conduction && params.phase == 1) {
						ws.rec2_main[k].gamma = data.gamma[pi];
						ws.rec2_main[k].mmw = data.mmw[pi];
						ws.rec2_main[k].eint = data.eint[pi];
					}
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			int rung_i = data.rungs[i];
			const bool active = rung_i >= params.min_rung;
			const bool semi_active = !active && data.sa_snk[snki];
			bool use = active || semi_active;
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
				const float fpre_i = data.fpre_snk[snki];
				const int jmax = round_up(ws.rec1_main.size(), block_size);
				float eint_i, mu_i, gamma_i, T_i;
				if (params.conduction && params.phase == 1) {
					eint_i = data.eint[i];
					mu_i = data.mmw[i];
					gamma_i = data.gamma[i];
					T_i = mu_i * eint_i / cv0 * sqr(ainv) / (gamma_i - 1.f);
				}
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
						const float x_ij = distance(x_j, x_i);
						const float y_ij = distance(y_j, y_i);
						const float z_ij = distance(z_j, z_i);
						const float r2 = sqr(x_ij, y_ij, z_ij);
						if (r2 < h2_i) {
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
				float dvx_dx = 0.0f;
				float dvx_dy = 0.0f;
				float dvx_dz = 0.0f;
				float dvy_dx = 0.0f;
				float dvy_dy = 0.0f;
				float dvy_dz = 0.0f;
				float dvz_dx = 0.0f;
				float dvz_dy = 0.0f;
				float dvz_dz = 0.0f;
				float dT_dx = 0.f;
				float dT_dy = 0.f;
				float dT_dz = 0.f;
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
					const float mrhoinv_i = m * rhoinv_i;
					dvx_dx -= mrhoinv_i * vx_ij * dWdr_x_i * ainv;
					dvy_dx -= mrhoinv_i * vy_ij * dWdr_x_i * ainv;
					dvz_dx -= mrhoinv_i * vz_ij * dWdr_x_i * ainv;
					dvx_dy -= mrhoinv_i * vx_ij * dWdr_y_i * ainv;
					dvy_dy -= mrhoinv_i * vy_ij * dWdr_y_i * ainv;
					dvz_dy -= mrhoinv_i * vz_ij * dWdr_y_i * ainv;
					dvx_dz -= mrhoinv_i * vx_ij * dWdr_z_i * ainv;
					dvy_dz -= mrhoinv_i * vy_ij * dWdr_z_i * ainv;
					dvz_dz -= mrhoinv_i * vz_ij * dWdr_z_i * ainv;
					if (params.conduction && params.phase == 1) {
						const float eint_j = rec2.eint;
						const float mu_j = rec2.mmw;
						const float gamma_j = rec2.gamma;
						const float T_j = mu_j * eint_j / cv0 * sqr(ainv) / (gamma_j - 1.f);
						const float tmp = logf(T_j / T_i);
						dT_dx += mrhoinv_i * tmp * dWdr_x_i * ainv;
						dT_dy += mrhoinv_i * tmp * dWdr_y_i * ainv;
						dT_dz += mrhoinv_i * tmp * dWdr_z_i * ainv;
					}
				}

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
				if (params.conduction && params.phase == 1) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dx);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dy);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dz);
				}
				if (tid == 0) {
					if (params.phase == 0) {
						data.divv0_snk[snki] = data.divv_snk[snki];
						data.taux0_snk[snki] = data.taux_snk[snki];
						data.taux_snk[snki] = params.tau;
					}
					data.divv_snk[snki] = div_v;
					data.crsv_snk[snki] = sqrtf(sqr(curl_vx) + sqr(curl_vy) + sqr(curl_vz));
					data.shearv_snk[snki] = sqrtf(sqr(shear_xx) + sqr(shear_yy) + sqr(shear_zz) + 2.0f * (sqr(shear_xy) + sqr(shear_yz) + sqr(shear_xz)));
					if (params.conduction && params.phase == 1) {
						data.gradT_snk[snki] = sqrtf(sqr(dT_dx) + sqr(dT_dy) + sqr(dT_dz));
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
	(&ws)->~aux_workspace();
}

__global__ void sph_cuda_xsph(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ xsph_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	int flops = 0;
	new (&ws) xsph_workspace();

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
					contains = (self.outer_box.contains(x));
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
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			int rung_i = data.rungs[i];
			const bool active = rung_i >= params.min_rung;
			bool use = active;
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
				const float mrhoinv_i = m * rhoinv_i;
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
						const float x_ij = distance(x_j, x_i);
						const float y_ij = distance(y_j, y_i);
						const float z_ij = distance(z_j, z_i);
						const float r2 = sqr(x_ij, y_ij, z_ij);
						flag = (r2 < h2_i);
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
				float xvx = 0.f;
				float xvy = 0.f;
				float xvz = 0.f;
				float one = 0.f;
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
					const float vx_ij = vx_i - vx_j;
					const float vy_ij = vy_i - vy_j;
					const float vz_ij = vz_i - vz_j;
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);
					const float q_i = r * hinv_i;
					const float W_i = kernelW(q_i) * h3inv_i;
					xvx -= mrhoinv_i * vx_ij * W_i;
					xvy -= mrhoinv_i * vy_ij * W_i;
					xvz -= mrhoinv_i * vz_ij * W_i;
					one += mrhoinv_i * W_i;
				}

				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(one);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(xvx);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(xvy);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(xvz);
				if (tid == 0) {
					if (fabs(one - 1.f) > 1.0e-4f) {
						PRINT("one = %e in xsph\n", one);
						__trap();
					}
					data.xvx_snk[snki] = xvx;
					data.xvy_snk[snki] = xvy;
					data.xvz_snk[snki] = xvz;
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
	(&ws)->~xsph_workspace();
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
				compute_indices<SMOOTHLEN_BLOCK_SIZE>(j, total);
				const int offset = ws.x.size();
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
				shared_reduce_max<int, SMOOTHLEN_BLOCK_SIZE>(max_rung_j);
				if (tid == 0) {
					if (rung_i < max_rung_j - 1) {
						changes++;
						rung_i = max_rung_j - 1;
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
	static int parabolic_nblocks;
	static int aux_nblocks;
	static int xsph_nblocks;
	static int rungs_nblocks;
	static bool first = true;
	if (first) {
		CUDA_CHECK(cudaMallocManaged(&memory, sizeof(cuda_mem)));
		new (memory) cuda_mem(8ULL * 1024ULL * 1024ULL * 1024ULL);
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&smoothlen_nblocks, (const void*) sph_cuda_smoothlen, SMOOTHLEN_BLOCK_SIZE, 0));
		smoothlen_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&semiactive_nblocks, (const void*) sph_cuda_mark_semiactive, SMOOTHLEN_BLOCK_SIZE, 0));
		semiactive_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&hydro_nblocks, (const void*) sph_cuda_hydro, HYDRO_BLOCK_SIZE, 0));
		hydro_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&parabolic_nblocks, (const void*) sph_cuda_parabolic, HYDRO_BLOCK_SIZE, 0));
		parabolic_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&aux_nblocks, (const void*) sph_cuda_aux, HYDRO_BLOCK_SIZE, 0));
		aux_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&xsph_nblocks, (const void*) sph_cuda_xsph, HYDRO_BLOCK_SIZE, 0));
		xsph_nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&rungs_nblocks, (const void*) sph_cuda_rungs, HYDRO_BLOCK_SIZE, 0));
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
		sph_cuda_mark_semiactive<<<semiactive_nblocks, SMOOTHLEN_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
	}
	break;
	case SPH_RUN_RUNGS: {
		timer tm;
		tm.start();
		sph_cuda_rungs<<<hydro_nblocks, SMOOTHLEN_BLOCK_SIZE,0,stream>>>(params,data,reduce);
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
	case SPH_RUN_PARABOLIC: {
		sph_cuda_parabolic<<<parabolic_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
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
		sph_cuda_aux<<<aux_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
	}
	break;
	case SPH_RUN_XSPH: {
		sph_cuda_xsph<<<xsph_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,reduce);
		cuda_stream_synchronize(stream);
	}
	break;
}
//	memory->reset();
	(cudaFree(reduce));
	memory->reset();
	return rc;
}
