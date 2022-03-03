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
		if (i > sz) {
			PRINT("Bound exceeded in device_vector\n");
			__trap();
		}
		return ptr[i];
	}
	__device__
	                             const T& operator[](int i) const {
		if (i > sz) {
			PRINT("Bound exceeded in device_vector\n");
			__trap();
		}
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
	float eint;
	float alpha;
	float f0;
	float fvel;
	float fpot;
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
	char oldrung;
};

struct hydro_workspace {
	device_vector<hydro_record1> rec1_main;
	device_vector<hydro_record2> rec2_main;
	device_vector<hydro_record1> rec1;
	device_vector<hydro_record2> rec2;
};

struct dif_workspace {
	device_vector<dif_record1> rec1_main;
	device_vector<dif_record2> rec2_main;
	device_vector<dif_record1> rec1;
	device_vector<dif_record2> rec2;
};

struct courant_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};

struct aux_record1 {
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
	float eint;
	float alpha;
};

struct aux_record2 {
//	float Y;
//	float Z;
	float gamma;
	float vx;
	float vy;
	float vz;
	float gx;
	float gy;
	float gz;
	float eint;
	float T;
	float lambda_e;
	float mmw;
	float alpha;
};

struct courant_workspace {
	device_vector<courant_record1> rec1_main;
	device_vector<courant_record2> rec2_main;
	device_vector<courant_record1> rec1;
	device_vector<courant_record2> rec2;
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

__global__ void sph_cuda_smoothlen(sph_run_params params, sph_run_cuda_data data, smoothlen_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__
	float error;
	__shared__ smoothlen_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws.x) device_vector<fixed32>();
	new (&ws.y) device_vector<fixed32>();
	new (&ws.z) device_vector<fixed32>();
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
			bool use;
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			if (params.phase == 0) {
				use = data.rungs[i] >= params.min_rung;
			} else {
				use = data.sa_snk[snki];
			}
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
				float h0 = h;
				do {
					float max_dh = h / sqrtf(iter + 100);
					const float hinv = 1.f / h; // 4
					const float h2 = sqr(h);    // 1
					count = 0;
					f = 0.f;
					dfdh = 0.f;
					for (int j = ws.x.size() - 1 - tid; j >= 0; j -= block_size) {
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
						if (self.outer_box.end[dim] < range_fixed(x[dim] + fixed32(h)) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
						if (range_fixed(x[dim]) < self.outer_box.begin[dim] + range_fixed(h) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
					}
					if (tid == 0) {
						if (box_xceeded) {
							//			PRINT( "Box exceeded with h = %e\n", h);
						}
						const float change = h / h0 - 1.0f;
						if (fabsf(change) > 1.0e-4) {
							//			PRINT( "change = %e\n", change);
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
	(&ws.x)->~device_vector<fixed32>();
	(&ws.y)->~device_vector<fixed32>();
	(&ws.z)->~device_vector<fixed32>();
}

__global__ void sph_cuda_mark_semiactive(sph_run_params params, sph_run_cuda_data data, mark_semiactive_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ mark_semiactive_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws.x) device_vector<fixed32>();
	new (&ws.y) device_vector<fixed32>();
	new (&ws.z) device_vector<fixed32>();
	new (&ws.h) device_vector<float>();
	new (&ws.rungs) device_vector<char>();

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
	(&ws.x)->~device_vector<fixed32>();
	(&ws.y)->~device_vector<fixed32>();
	(&ws.z)->~device_vector<fixed32>();
	(&ws.h)->~device_vector<float>();
	(&ws.rungs)->~device_vector<char>();

}

__global__ void sph_cuda_diffusion(sph_run_params params, sph_run_cuda_data data, dif_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ dif_workspace ws;
	new (&ws.rec1_main) dif_record1();
	new (&ws.rec2_main) dif_record2();
	new (&ws.rec1) dif_record1();
	new (&ws.rec2) dif_record2();
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
					ws.rec2_main[k].vec = data.dif_vec[pi];
					ws.rec2_main[k].oldrung = data.oldrung[pi];
					if (data.conduction) {
						ws.rec2_main[k].kappa = data.kappa[pi];
					} else {
						ws.rec2_main[k].kappa = 0.f;
					}
					if (data.chem) {
						ws.rec2_main[k].gamma = data.gamma[pi];
					} else {
						ws.rec2_main[k].gamma = data.def_gamma;
					}
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
				const auto vec0_i = data.vec0_snk[snki];
				const auto vec_i = data.dif_vec[i];
				float kappa_i;
				if (data.conduction) {
					kappa_i = data.kappa[i];
				} else {
					kappa_i = 0.f;
				}
				const int jmax = round_up(ws.rec1_main.size(), block_size);
				ws.rec1.resize(0);
				ws.rec2.resize(0);
				for (int j = tid; j < jmax; j += block_size) {
					bool flag = false;
					int k;
					int total;
					if (j < ws.rec1_main.size()) {
						const auto rec = ws.rec1_main[j];
						const fixed32 x_j = rec.x;
						const fixed32 y_j = rec.y;
						const fixed32 z_j = rec.z;
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
					const auto& rec1 = ws.rec1[j];
					const auto& rec2 = ws.rec2[j];
					const fixed32& x_j = rec1.x;
					const fixed32& y_j = rec1.y;
					const fixed32& z_j = rec1.z;
					const float& kappa_j = rec2.kappa;
					const float& difco_j = rec2.difco;
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);
					const float rinv = 1.0f / (1.0e-30f + r);
					const float h_j = rec1.h;
					const float hinv_j = 1.f / h_j;															// 4
					const float h3inv_j = sqr(hinv_j) * hinv_j;
					const float rho_j = m * c0 * h3inv_j;													// 2
					const float dt_ij = fminf(rung_dt[myrung], rung_dt[rec1.rung]) * params.t0;
					const float rho_ij = 0.5f * (rho_i + rho_j);
					const float h_ij = 0.5f * (h_i + h_j);
					const float kappa_ij = 2.f * kappa_i * kappa_j / (kappa_i + kappa_j + 1e-30);
					const float difco_ij = 2.f * (difco_i * difco_j) / (difco_i + difco_j + 1e-30);
					const float dWdr_ij = 0.5f * (dkernelW_dq(fminf(r * hinv_i, 1.f)) / sqr(sqr(h_i)) + dkernelW_dq(fminf(r * hinv_j, 1.f)) / sqr(sqr(hinv_j)));
					const float diff_factor = -2.f * dt_ij * m / rho_ij * difco_ij * dWdr_ij * rinv;
					const float cond_factor = -dt_ij * m / (rho_i * rho_j) * kappa_ij * dWdr_ij * rinv;
					for (int fi = 0; fi < DIFCO_COUNT; fi++) {
						num[fi] += diff_factor * rec2.vec[fi];
					}
					den += diff_factor;
					if (data.conduction) {
						//float adjust = mmw_j / mmw_i;
						num[NCHEMFRACS] += cond_factor * rec2.vec[NCHEMFRACS];													// * adjust;
						den_A += diff_factor + cond_factor;
					}
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
			atomicAdd(&reduce->flops, (float) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws.rec1_main)->~device_vector<dif_record1>();
	(&ws.rec2_main)->~device_vector<dif_record2>();
	(&ws.rec1)->~device_vector<dif_record1>();
	(&ws.rec2)->~device_vector<dif_record2>();
}

#define ETA1 0.01f
#define ETA2 0.0001f

__global__ void sph_cuda_hydro(sph_run_params params, sph_run_cuda_data data, hydro_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ hydro_workspace ws;
	new (&ws.rec1_main) hydro_record1();
	new (&ws.rec2_main) hydro_record2();
	new (&ws.rec1) hydro_record1();
	new (&ws.rec2) hydro_record2();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	int flops = 0;
	float total_vsig_max = 0.f;
	int max_rung = 0;
	int max_rung_hydro = 0;
	int max_rung_grav = 0;
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
					ws.rec2_main[k].vx = data.vx[pi];
					ws.rec2_main[k].vy = data.vy[pi];
					ws.rec2_main[k].vz = data.vz[pi];
					ws.rec2_main[k].eint = data.eint[pi];
					ws.rec2_main[k].f0 = data.f0[pi];
					ws.rec2_main[k].fvel = data.fvel[pi];
					ws.rec2_main[k].alpha = data.alpha[pi];
					if (data.chem) {
						ws.rec2_main[k].gamma = data.gamma[pi];
					} else {
						ws.rec2_main[k].gamma = data.def_gamma;
					}
					if (data.gravity) {
						ws.rec2_main[k].fpot = data.fpot[pi];
						ws.rec2_main[k].gx = data.gx[pi];
						ws.rec2_main[k].gy = data.gy[pi];
						ws.rec2_main[k].gz = data.gz[pi];
					} else {
						ws.rec2_main[k].gx = 0.f;
						ws.rec2_main[k].gy = 0.f;
						ws.rec2_main[k].gz = 0.f;
					}
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			int myrung = data.rungs[i];
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			bool active = myrung >= params.min_rung;
			bool use = active;
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
				float gx_i, gy_i, gz_i;
				if (data.gravity) {
					gx_i = data.gx[i];
					gy_i = data.gy[i];
					gz_i = data.gz[i];
				} else {
					gx_i = gy_i = gz_i;
				}
				const float h_i = data.h[i];
				const float hinv_i = 1.f / h_i;															// 4
				const float h3inv_i = sqr(hinv_i) * hinv_i;
				const float rho_i = m * c0 * h3inv_i;													// 2
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;								// 5
				const float eint_i = data.eint[i];
				float gamma_i;
				if (data.chem) {
					gamma_i = data.gamma[i];
				} else {
					gamma_i = data.def_gamma;
				}
				float fpot_i;
				if (data.gravity) {
					fpot_i = data.fpot[i];
				} else {
					fpot_i = 0.f;
				}
				const float p_i = fmaxf((gamma_i - 1.f) * rho_i * eint_i, 0.f);								// 5
				const float c_i = sqrtf(gamma_i * p_i * rhoinv_i);									// 6
				const float fvel_i = data.fvel[i];
				const float f0_i = data.f0[i];
				const float alpha_i = data.alpha[i];
				const float dpot_i = -0.5f * data.G * m * fpot_i * f0_i;
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
				float dvxdx = 0.f;
				float dvxdy = 0.f;
				float dvxdz = 0.f;
				float dvydx = 0.f;
				float dvydy = 0.f;
				float dvydz = 0.f;
				float dvzdx = 0.f;
				float dvzdy = 0.f;
				float dvzdz = 0.f;
				float deint_con = 0.f;
				float dvx_con = 0.f;
				float dvy_con = 0.f;
				float dvz_con = 0.f;
				float gx = 0.0;
				float gy = 0.0;
				float gz = 0.0;
				float vsig = 0.f;
				float cdt = 1e30f;
				float ddivv_dt = 0.f;
				const float ainv = 1.0f / params.a;
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					auto rec1 = ws.rec1[j];
					auto rec2 = ws.rec2[j];
					const float vx_j = rec2.vx;
					const float vy_j = rec2.vy;
					const float vz_j = rec2.vz;
					float gx_j, gy_j, gz_j;
					if (data.gravity) {
						gx_j = rec2.gx;
						gy_j = rec2.gy;
						gz_j = rec2.gz;
					} else {
						gx_j = gy_j = gz_j = 0.f;
					}
					const float fpot_j = rec2.fpot;
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float h_j = rec1.h;
					const float hinv_j = 1.f / h_j;															// 4
					const float h3inv_j = sqr(hinv_j) * hinv_j;
					const float rho_j = m * c0 * h3inv_j;													// 2
					const float rhoinv_j = minv * c0inv * sqr(h_j) * h_j;								// 5
					const float eint_j = rec2.eint;
					const float gamma_j = rec2.gamma;
					const float p_j = fmaxf(rho_j * (gamma_j - 1.f) * eint_j, 0.f);								// 5
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
					const float rinv = 1.0f / (1.0e-30f + r);
					const float alpha_ij = 0.5f * (alpha_i * fvel_i + alpha_j * fvel_j);
					const float h_ij = 0.5f * (h_i + h_j);
					const float vdotx_ij = fminf(0.0f, x_ij * vx_ij + y_ij * vy_ij + z_ij * vz_ij);
					const float u_ij = vdotx_ij * h_ij / (r2 + ETA1 * sqr(h_ij));
					const float c_ij = 0.5f * (c_i + c_j);
					const float this_vsig = c_ij - vdotx_ij * rinv;
					const float rho_ij = 0.5f * (rho_i + rho_j);
					const float Pi = -alpha_ij * u_ij * (c_ij - SPH_BETA * u_ij) / rho_ij;
					const float q_i = fminf(r * hinv_i, 1.f);								// 1
					const float q_j = fminf(r * hinv_j, 1.f);									// 1
					const float dWdr_i = f0_i * dkernelW_dq(q_i) * hinv_i * h3inv_i;
					const float dWdr_j = f0_j * dkernelW_dq(q_j) * hinv_j * h3inv_j;
					const float Wi = kernelW(q_i) * h3inv_i;
					const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);
					const float dWdr_x_ij = x_ij * rinv * dWdr_ij;
					const float dWdr_y_ij = y_ij * rinv * dWdr_ij;
					const float dWdr_z_ij = z_ij * rinv * dWdr_ij;
					const float dWdr_x_i = dWdr_i * rinv * x_ij;
					const float dWdr_y_i = dWdr_i * rinv * y_ij;
					const float dWdr_z_i = dWdr_i * rinv * z_ij;
					const float dWdr_x_j = dWdr_j * rinv * x_ij;
					const float dWdr_y_j = dWdr_j * rinv * y_ij;
					const float dWdr_z_j = dWdr_j * rinv * z_ij;
					const float dp_j = p_j * sqr(rhoinv_j);
					const float dp_i = p_i * sqr(rhoinv_i);
					const float dvx_dt = -m * ainv * (dp_i * dWdr_x_i + dp_j * dWdr_x_j + Pi * dWdr_x_ij);
					const float dvy_dt = -m * ainv * (dp_i * dWdr_y_i + dp_j * dWdr_y_j + Pi * dWdr_y_ij);
					const float dvz_dt = -m * ainv * (dp_i * dWdr_z_i + dp_j * dWdr_z_j + Pi * dWdr_z_ij);
					const float tmp1 = (vx_ij * dWdr_x_ij + vy_ij * dWdr_y_ij + vz_ij * dWdr_z_ij);
					const float tmp2 = (vx_ij * dWdr_x_i + vy_ij * dWdr_y_i + vz_ij * dWdr_z_i);
					const float de_dt = ainv * (0.5f * Pi * tmp1 + p_i * rhoinv_i * rhoinv_i * tmp2) * m;
					const float mrhoinv_i = m * rhoinv_i;
					ddivv_dt += m * (p_i * sqr(rhoinv_i) + p_j * sqr(rhoinv_j)) * (rho_i - rho_j) / rho_ij * dWdr_i * rinv;
					ddivv_dt += m * 2.f * (p_i * rhoinv_i - p_j * rhoinv_j) / rho_ij * dWdr_i * rinv;
					ddivv_dt += mrhoinv_i * (gx_j - gx_i) * dWdr_x_i;
					ddivv_dt += mrhoinv_i * (gy_j - gy_i) * dWdr_y_i;
					ddivv_dt += mrhoinv_i * (gz_j - gz_i) * dWdr_z_i;
					vsig += this_vsig * Wi * mrhoinv_i;
					dvxdx -= mrhoinv_i * vx_ij * dWdr_x_i;
					dvydx -= mrhoinv_i * vy_ij * dWdr_x_i;
					dvzdx -= mrhoinv_i * vz_ij * dWdr_x_i;
					dvxdy -= mrhoinv_i * vx_ij * dWdr_y_i;
					dvydy -= mrhoinv_i * vy_ij * dWdr_y_i;
					dvzdy -= mrhoinv_i * vz_ij * dWdr_y_i;
					dvxdz -= mrhoinv_i * vx_ij * dWdr_z_i;
					dvydz -= mrhoinv_i * vy_ij * dWdr_z_i;
					dvzdz -= mrhoinv_i * vz_ij * dWdr_z_i;
					flops += 8;
					deint_con += de_dt;									// 2
					dvx_con += dvx_dt;								// 2
					dvy_con += dvy_dt;								// 2
					dvz_con += dvz_dt;								// 2
					const float dpot_j = -0.5f * data.G * m * fpot_j * f0_j;
					gx += dpot_i * dWdr_x_i + dpot_j * dWdr_x_j;
					gy += dpot_i * dWdr_y_i + dpot_j * dWdr_y_j;
					gz += dpot_i * dWdr_z_i + dpot_j * dWdr_z_j;
					flops += 181;
					if (params.phase == 1) {
						float this_cdt;
						if (vdotx_ij < 0.0f) {
							this_cdt = h_ij * params.a / (c_ij + alpha_ij * 0.6f * (c_ij - SPH_BETA * vdotx_ij * rinv));
						} else {
							this_cdt = h_ij * params.a / c_ij;
						}
						total_vsig_max = fmaxf(total_vsig_max, h_i / cdt);
						cdt = fminf(cdt, this_cdt);
					}
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(deint_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvx_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvy_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dvz_con);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ddivv_dt);
				float divv = (dvxdx + dvydy + dvzdz);
				float curlv_x = (dvzdy - dvydz);
				float curlv_y = (-dvzdx + dvxdz);
				float curlv_z = (dvydx - dvxdy);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(divv);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curlv_x);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curlv_y);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curlv_z);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(vsig);
				if (params.phase == 1) {
					shared_reduce_min<float, HYDRO_BLOCK_SIZE>(cdt);
				}
				if (data.gravity) {
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(gx);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(gy);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(gz);
				}
				if (tid == 0) {
					if (data.gcentral != 0.f) {
						const float dx = x_i.to_float() - 0.5f;
						const float dy = y_i.to_float() - 0.5f;
						const float dz = z_i.to_float() - 0.5f;
						const float r = sqrt(sqr(dx, dy, dz));
						const float q = r / data.hcentral;
						float r3inv;
						if (q < 1.f) {
							r3inv = kernelFqinv(q) / (sqr(data.hcentral) * data.hcentral);
						} else {
							r3inv = 1.f / (sqr(r) * r);
						}
						dvx_con -= dx * data.gcentral * r3inv;
						dvy_con -= dy * data.gcentral * r3inv;
						dvz_con -= dz * data.gcentral * r3inv;
					}
					if (y_i.to_float() < 0.5) {
						dvy_con -= params.gy;
					} else {
						dvy_con += params.gy;
					}
					if (data.gravity) {
						if( params.phase== 0 ) {
						data.gx_snk[snki] += gx;
						data.gy_snk[snki] += gy;
						data.gz_snk[snki] += gz;
						}
						dvx_con += data.gx_snk[snki];
						dvy_con += data.gy_snk[snki];
						dvz_con += data.gz_snk[snki];
					}
					data.deint_con[snki] = deint_con;										// 1
					data.dvx_con[snki] = dvx_con;										// 1
					data.dvy_con[snki] = dvy_con;										// 1
					data.dvz_con[snki] = dvz_con;										// 1
					flops += 4;
					data.divv_snk[snki] = 0;
					const float alpha = data.alpha_snk[snki];
					const float t0inv = vsig * SPH_VISC_DECAY * hinv_i;
					const float balsara = fabsf(divv) / (sqrt(sqr(curlv_x, curlv_y, curlv_z)) + fabsf(divv) + ETA2 * c_i * hinv_i);
					float S = fmaxf(0.f, -ddivv_dt) * balsara;
					float dt = 0.5f * rung_dt[myrung] * params.t0; // 3
					const float alpha_targ = SPH_ALPHA1 * (sqr(h_i) * S / (sqr(h_i) * S + sqr(vsig)));
					float dalpha_dt;
					if (alpha < alpha_targ) {
						data.dalpha_con[snki] = 0.f;
						data.dalpha_pred[snki] = 0.f;
						data.alpha_snk[snki] = alpha_targ;
					} else {
						dalpha_dt = -(alpha - SPH_ALPHA0) * t0inv;
						data.dalpha_con[snki] = dalpha_dt;
					}
					if (params.phase) {
						float dthydro = params.cfl * cdt;
						const float gx = data.gx_snk[snki];
						const float gy = data.gy_snk[snki];
						const float gz = data.gz_snk[snki];
						char& rung = data.rungs[i];
						const float g2 = sqr(gx, gy, gz);
						const float a2 = sqr(dvx_con, dvy_con, dvz_con);
						const float hsoft = fminf(fmaxf(h_i, data.hsoft_min), SPH_MAX_SOFT);
						const float afactor = data.eta * sqrtf(params.a * h_i);
						const float gfactor = data.eta * sqrtf(params.a * hsoft);
						dthydro = fminf(fminf(afactor / sqrtf(sqrtf(a2 + 1e-15f)), (float) params.t0), dthydro);
						const float dt_grav = fminf(gfactor / sqrtf(sqrtf(g2 + 1e-15f)), (float) params.t0);
						const float dt = fminf(dt_grav, dthydro);
						int rung_hydro = ceilf(log2f(params.t0) - log2f(dthydro));
						const int rung_grav = ceilf(log2f(params.t0) - log2f(dt_grav));
						if( h_i >= data.hstar0) {
							max_rung_hydro = max(max_rung_hydro, rung_hydro);
							max_rung_grav = max(max_rung_grav, rung_grav);
							rung = max(max((int) max(rung_hydro, rung_grav), max(params.min_rung, (int) rung - 1)), 1);
							max_rung = max(max_rung, rung);
						} else {
							max_rung_grav = max(max_rung_grav, rung_grav);
							rung = max(max(rung_grav, max(params.min_rung, (int) rung - 1)), 1);
							max_rung = max(max_rung, rung);
						}
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
			atomicMax(&reduce->vsig_max, total_vsig_max);
			atomicMax(&reduce->max_rung, max_rung);
			atomicMax(&reduce->max_rung_hydro, max_rung_hydro);
			atomicMax(&reduce->max_rung_grav, max_rung_grav);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws.rec1_main)->~device_vector<hydro_record1>();
	(&ws.rec2_main)->~device_vector<hydro_record2>();
	(&ws.rec1)->~device_vector<hydro_record1>();
	(&ws.rec2)->~device_vector<hydro_record2>();
}

__global__ void sph_cuda_courant(sph_run_params params, sph_run_cuda_data data, courant_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ courant_workspace ws;
	new (&ws.rec1_main) courant_record1();
	new (&ws.rec2_main) courant_record2();
	new (&ws.rec1) courant_record1();
	new (&ws.rec2) courant_record2();
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
					const float h = data.h[pi];
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					if (self.outer_box.contains(x)) {
						contains = true;
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
					ws.rec2_main[k].eint = data.eint[pi];
					ws.rec1_main[k].h = data.h[pi];
					ws.rec2_main[k].alpha = data.alpha[pi];
					if (data.chem) {
						ws.rec2_main[k].gamma = data.gamma[pi];
					} else {
						ws.rec2_main[k].gamma = data.def_gamma;
					}
					/*						if (stars) {
					 ws.rec2_main[k].gx = data.gx[pi];
					 ws.rec2_main[k].gy = data.gy[pi];
					 ws.rec2_main[k].gz = data.gz[pi];
					 }*/
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
				const float h_i = data.h[i];
				float gamma_i;
				if (data.chem) {
					gamma_i = data.gamma[i];
				} else {
					gamma_i = data.def_gamma;
				}
				float gx_i;
				float gy_i;
				float gz_i;
				if (data.gravity) {
					gx_i = data.gx[i];
					gy_i = data.gy[i];
					gz_i = data.gz[i];
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
				const float alpha_i = data.alpha[i];
				const float p_i = fmaxf(data.eint[i] * rho_i * (gamma_i - 1.f), 0.f);
				const float c_i = sqrtf(gamma_i * p_i * rhoinv_i);
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
				float dgx_dx = 0.f;
				float dgy_dy = 0.f;
				float dgz_dz = 0.f;
				float ax = 0.f;
				float ay = 0.f;
				float az = 0.f;
				const float ainv = 1.0f / params.a;
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					const auto rec1 = ws.rec1[j];
					const auto rec2 = ws.rec2[j];
					const float vx_j = rec2.vx;
					const float vy_j = rec2.vy;
					const float vz_j = rec2.vz;
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float h_j = rec1.h;
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
					const float vx_ij = vx_i - vx_j;
					const float vy_ij = vy_i - vy_j;
					const float vz_ij = vz_i - vz_j;
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);
					const float rinv = 1.0f / (1.0e-30f + r);
					const float alpha_ij = 0.5f * (alpha_i + alpha_j);
					const float h_ij = 0.5f * (h_i + h_j);
					const float vdotx_ij = fminf(0.0f, x_ij * vx_ij + y_ij * vy_ij + z_ij * vz_ij);
					const float u_ij = vdotx_ij * h_ij / (r2 + ETA1 * sqr(h_ij));
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
					const float dWdr_x_i = dWdr_i * rinv * x_ij;
					const float dWdr_y_i = dWdr_i * rinv * y_ij;
					const float dWdr_z_i = dWdr_i * rinv * z_ij;
					const float dWdr_x_j = dWdr_j * rinv * x_ij;
					const float dWdr_y_j = dWdr_j * rinv * y_ij;
					const float dWdr_z_j = dWdr_j * rinv * z_ij;
					const float dp_i = p_i * sqr(rhoinv_i);
					const float dp_j = p_j * sqr(rhoinv_j);
					const float dvx_dt = -m * ainv * (dp_i * dWdr_x_i + dp_j * dWdr_x_j + Pi * dWdr_x_ij);
					const float dvy_dt = -m * ainv * (dp_i * dWdr_y_i + dp_j * dWdr_y_j + Pi * dWdr_y_ij);
					const float dvz_dt = -m * ainv * (dp_i * dWdr_z_i + dp_j * dWdr_z_j + Pi * dWdr_z_ij);
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
					if (vdotx_ij < 0.f) {
						this_vsig += 0.6f * alpha_ij * c_ij * hfac;
						this_vsig -= 0.6f * alpha_ij * SPH_BETA * vdotx_ij * rinv * hfac;
					}
					vsig_max = fmaxf(vsig_max, this_vsig);									   // 2
					ax += dvx_dt;
					ay += dvy_dt;
					az += dvz_dt;
				}
				float div_v = dvx_dx + dvy_dy + dvz_dz;
				float curl_vx = dvz_dy - dvy_dz;
				float curl_vy = -dvz_dx + dvx_dz;
				float curl_vz = dvy_dx - dvx_dy;
				float div_g;
				if (stars) {
					div_g = dgx_dx + dgy_dy + dgz_dz;
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(div_g);
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(div_g);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ax);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(ay);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(az);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(div_v);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vx);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vy);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vz);
				shared_reduce_max<float, HYDRO_BLOCK_SIZE>(vsig_max);

				if (tid == 0) {
					if (data.gcentral != 0.f) {
						const float dx = x_i.to_float() - 0.5f;
						const float dy = y_i.to_float() - 0.5f;
						const float dz = z_i.to_float() - 0.5f;
						const float r = sqrt(sqr(dx, dy, dz));
						const float q = r / data.hcentral;
						float r3inv;
						if (q < 1.f) {
							r3inv = kernelFqinv(q) / (sqr(data.hcentral) * data.hcentral);
						} else {
							r3inv = 1.f / (sqr(r) * r);
						}
						ax -= dx * data.gcentral * r3inv;
						ay -= dy * data.gcentral * r3inv;
						az -= dz * data.gcentral * r3inv;
					}
					ax += gx_i;
					ay += gy_i;
					az += gz_i;
					ay += params.gy;
					const float sw = 1e-4f * c_i / h_i;
					const float abs_div_v = fabsf(div_v);
					const float abs_curl_v = sqrtf(sqr(curl_vx, curl_vy, curl_vz));
					const float fvel = abs_div_v / (abs_div_v + abs_curl_v + sw);
					const float fpre = 1.0f / (1.0f + c0);
					const float dt_cfl = params.a * h_i / vsig_max;
					data.fvel_snk[snki] = fvel;
					total_vsig_max = fmaxf(total_vsig_max, vsig_max);
					float dthydro = params.cfl * dt_cfl;
					const float gx = data.gx_snk[snki];
					const float gy = data.gy_snk[snki];
					const float gz = data.gz_snk[snki];
					char& rung = data.rungs[i];
					const float g2 = sqr(gx, gy, gz);
					const float a2 = sqr(ax, ay, az);
					const float hsoft = fminf(fmaxf(h_i, data.hsoft_min), SPH_MAX_SOFT);
					const float afactor = data.eta * sqrtf(params.a * h_i);
					const float gfactor = data.eta * sqrtf(params.a * hsoft);
					dthydro = fminf(fminf(afactor / sqrtf(sqrtf(a2 + 1e-15f)), (float) params.t0), dthydro);
					const float dt_grav = fminf(gfactor / sqrtf(sqrtf(g2 + 1e-15f)), (float) params.t0);
					const float dt = fminf(dt_grav, dthydro);
					int rung_hydro = ceilf(log2f(params.t0) - log2f(dthydro));
					const int rung_grav = ceilf(log2f(params.t0) - log2f(dt_grav));

					if (rung < 0 || rung >= MAX_RUNG) {
						if (tid == 0) {
							PRINT("Rung out of range \n");
							__trap();
						}
					}
					if (stars) {
						bool is_eligible = h_i < data.hstar0;
						if (is_eligible) {
							data.tdyn_snk[snki] = 1e-10f;
							rung = max(max((int) rung_grav, max(params.min_rung, (int) rung - 1)), 1);
							max_rung = max(max_rung, rung);
						} else {
							data.tdyn_snk[snki] = 1e+38;
							max_rung_hydro = max(max_rung_hydro, rung_hydro);
							max_rung_grav = max(max_rung_grav, rung_grav);
							rung = max(max((int) max(rung_hydro, rung_grav), max(params.min_rung, (int) rung - 1)), 1);
							max_rung = max(max_rung, rung);
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
	if (tid == 0) {
		atomicMax(&reduce->vsig_max, total_vsig_max);
		atomicMax(&reduce->max_rung, max_rung);
		atomicMax(&reduce->max_rung_hydro, max_rung_hydro);
		atomicMax(&reduce->max_rung_grav, max_rung_grav);
	}
	(&ws.rec1_main)->~device_vector<courant_record1>();
	(&ws.rec2_main)->~device_vector<courant_record2>();
	(&ws.rec1)->~device_vector<courant_record1>();
	(&ws.rec2)->~device_vector<courant_record2>();
}

__global__ void sph_cuda_aux(sph_run_params params, sph_run_cuda_data data, aux_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	aux_workspace& ws = workspaces[bid];
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	int flops = 0;
	new (&ws.rec1_main) aux_record1();
	new (&ws.rec2_main) aux_record2();
	new (&ws.rec1) aux_record1();
	new (&ws.rec2) aux_record2();

	while (index < data.nselfs) {
		const sph_tree_node& self = data.trees[data.selfs[index]];
		ws.rec1_main.resize(0);
		ws.rec2_main.resize(0);
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			continue;
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					const float h = data.h[pi];
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					if (self.outer_box.contains(x)) {
						contains = true;
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
					ws.rec2_main[k].T = data.T[pi];
					ws.rec1_main[k].h = data.h[pi];
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			continue;
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
				const float h_i = data.h[i];
				const float h2_i = sqr(h_i);
				const float T_i = data.T[i];
				const float hinv_i = 1.f / h_i;
				const float h3inv_i = (sqr(hinv_i) * hinv_i);
				const float rho_i = m * c0 * h3inv_i;
				const float rhoinv_i = minv * c0inv * sqr(h_i) * h_i;
				const float eint_i = data.eint_snk[snki];
				float gamma_i;
				if (data.chem) {
					gamma_i = data.gamma[i];
				} else {
					gamma_i = data.def_gamma;
				}
				const float p_i = fmaxf((gamma_i - 1.f) * rho_i * eint_i, 0.f);								// 5
				const float c_i = sqrtf(gamma_i * p_i * rhoinv_i);									// 6
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
				float dpot_dh = 0.f;
				float dT_dx = 0.f;
				float dT_dy = 0.f;
				float dT_dz = 0.f;
				for (int j = tid; j < ws.rec1.size(); j += block_size) {
					const auto rec1 = ws.rec1[j];
					const auto rec2 = ws.rec2[j];
					const float T_j = rec2.T;
					const float vx_j = rec2.vx;
					const float vy_j = rec2.vy;
					const float vz_j = rec2.vz;
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float h_j = rec1.h;
					const float hinv_j = 1.f / h_j;															// 4
					const float h3inv_j = sqr(hinv_j) * hinv_j;
					const float rho_j = m * c0 * h3inv_j;													// 2
					const float rhoinv_j = minv * c0inv * sqr(h_j) * h_j;								// 5
					const float x_ij = distance(x_i, x_j);				// 2
					const float y_ij = distance(y_i, y_j);				// 2
					const float z_ij = distance(z_i, z_j);				// 2
					const float vx_ij = vx_i - vx_j;
					const float vy_ij = vy_i - vy_j;
					const float vz_ij = vz_i - vz_j;
					const float r2 = sqr(x_ij, y_ij, z_ij);
					const float r = sqrt(r2);
					const float rinv = 1.0f / (1.0e-30f + r);
					const float h_ij = 0.5f * (h_i + h_j);
					const float rho_ij = 0.5f * (rho_i + rho_j);
					const float q_i = fminf(r * hinv_i, 1.f);								// 1
					const float q_j = fminf(r * hinv_j, 1.f);								// 1
					const float dWdr_i = dkernelW_dq(q_i) * hinv_i * h3inv_i;
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
					if (params.phase == 0) {
						drho_dh -= (3.f * kernelW(q_i) + q_i * dkernelW_dq(q_i));
						const float q_ij = 0.5f * (q_i + q_j);
						const float pot = kernelPot(q_ij);
						const float force = kernelFqinv(q_ij) * q_ij;
						dpot_dh += 0.5 / sqr(h_ij) * (pot + q_ij * force);
					} else {
						dT_dx += (T_j - T_i) * dWdr_x_i * mrhoinv_i;
						dT_dy += (T_j - T_i) * dWdr_y_i * mrhoinv_i;
						dT_dz += (T_j - T_i) * dWdr_z_i * mrhoinv_i;
					}
				}
				float div_v = dvx_dx + dvy_dy + dvz_dz;
				float curl_vx = dvz_dy - dvy_dz;
				float curl_vy = -dvz_dx + dvx_dz;
				float curl_vz = dvy_dx - dvx_dy;
				float shear_xx, shear_yy, shear_zz, shear_xy, shear_xz, shear_yz;
				if (params.phase == 1) {
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
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dx);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dy);
					shared_reduce_add<float, HYDRO_BLOCK_SIZE>(dT_dz);
				} else {
					shared_reduce_max<float, HYDRO_BLOCK_SIZE>(drho_dh);
					shared_reduce_max<float, HYDRO_BLOCK_SIZE>(dpot_dh);
				}
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(div_v);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vx);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vy);
				shared_reduce_add<float, HYDRO_BLOCK_SIZE>(curl_vz);
				if (tid == 0) {
					const float sw = 1e-4f * c_i / h_i;
					const float abs_div_v = fabsf(div_v);
					const float abs_curl_v = sqrtf(sqr(curl_vx, curl_vy, curl_vz));
					const float fvel = abs_div_v / (abs_div_v + abs_curl_v + sw);
					data.divv_snk[snki] = div_v;
					data.fvel_snk[snki] = fvel;
					if (params.phase == 0) {
						const float c0 = drho_dh * 4.0f * float(M_PI) / (9.0f * data.N);
						const float fpre = 1.0f / (1.0f + c0);
						const float fg = -dpot_dh * h_i / (3.f * rho_i);
						data.f0_snk[snki] = fpre;
						data.fpot_snk[snki] = fg;
					} else {
						const float Cdif = SPH_DIFFUSION_C * sqr(h_i) * sqrt(sqr(shear_xx, shear_yy, shear_zz) + 2.f * sqr(shear_xy, shear_xz, shear_yz)) / params.a;
						if (data.conduction) {
							const float lt = T_i / (sqrt(sqr(dT_dx, dT_dy, dT_dz)) + 1.0e-10f * T_i);
							const float kappa_sp = data.kappa0 / data.colog[i]; // Jubelgas et al 2004, Smith et al 2021
							const float kappa = kappa_sp / (1.f + 4.2f * data.lambda_e[i] / lt);
							const float tmp = data.code_dif_to_cgs * constants::kb / (sqr(params.a) * params.a);
							data.kappa_snk[snki] = 2.f * data.mmw[i] * (data.gamma[i] - 1.f) * kappa / tmp;
						}
						data.difco_snk[snki] = Cdif;
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
	(&ws.rec1_main)->~device_vector<aux_record1>();
	(&ws.rec2_main)->~device_vector<aux_record2>();
	(&ws.rec1)->~device_vector<aux_record1>();
	(&ws.rec2)->~device_vector<aux_record2>();
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
	static int aux_nblocks;
	static bool first = true;
	static char* workspace_ptr;
	if (first) {
		CUDA_CHECK(cudaMallocManaged(&memory, sizeof(cuda_mem)));
		new (memory) cuda_mem(4ULL * 1024ULL * 1024ULL * 1024ULL);
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&smoothlen_nblocks, (const void*) sph_cuda_smoothlen, SMOOTHLEN_BLOCK_SIZE, 0));
		smoothlen_nblocks *= cuda_smp_count() * 2 / 3;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&semiactive_nblocks, (const void*) sph_cuda_mark_semiactive, SMOOTHLEN_BLOCK_SIZE, 0));
		semiactive_nblocks *= cuda_smp_count() * 2 / 3;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&dif_nblocks, (const void*) sph_cuda_diffusion, HYDRO_BLOCK_SIZE, 0));
		dif_nblocks *= cuda_smp_count() * 2 / 3;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&hydro_nblocks, (const void*) sph_cuda_hydro, HYDRO_BLOCK_SIZE, 0));
		hydro_nblocks *= cuda_smp_count() * 2 / 3;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&courant_nblocks, (const void*) sph_cuda_courant, HYDRO_BLOCK_SIZE, 0));
		courant_nblocks *= cuda_smp_count() * 2 / 3;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&aux_nblocks, (const void*) sph_cuda_aux, HYDRO_BLOCK_SIZE, 0));
		aux_nblocks *= cuda_smp_count() * 2 / 3;
		size_t smoothlen_mem = sizeof(smoothlen_workspace) * smoothlen_nblocks;
		size_t semiactive_mem = sizeof(mark_semiactive_workspace) * semiactive_nblocks;
		size_t courant_mem = sizeof(courant_workspace) * courant_nblocks;
		size_t aux_mem = sizeof(aux_workspace) * aux_nblocks;
		size_t hydro_mem = sizeof(hydro_workspace) * hydro_nblocks;
		size_t dif_mem = sizeof(dif_workspace) * dif_nblocks;
		size_t max_mem = std::max(aux_mem, std::max(std::max(std::max(smoothlen_mem, semiactive_mem), std::max(hydro_mem, courant_mem)), dif_mem));
		CUDA_CHECK(cudaMallocManaged(&workspace_ptr, max_mem));
		PRINT("Allocating %i GB in workspace memory\n", max_mem / 1024 / 1024 / 1024);
//		sleep(10);
	}
	switch (params.run_type) {
	case SPH_RUN_SMOOTHLEN: {
		sph_cuda_smoothlen<<<smoothlen_nblocks, SMOOTHLEN_BLOCK_SIZE,0,stream>>>(params,data,(smoothlen_workspace*)workspace_ptr,reduce);
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
		rc.max_vsig = reduce->vsig_max;
		rc.max_rung_grav = reduce->max_rung_grav;
		rc.max_rung_hydro = reduce->max_rung_hydro;
		rc.max_rung = reduce->max_rung;
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
	case SPH_RUN_AUX: {
		sph_cuda_aux<<<aux_nblocks, HYDRO_BLOCK_SIZE,0,stream>>>(params,data,(aux_workspace*)workspace_ptr,reduce);
		cuda_stream_synchronize(stream);
	}
	break;
}
//	memory->reset();
	(cudaFree(reduce));
	memory->reset();
	return rc;
}
