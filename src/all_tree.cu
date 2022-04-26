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
#define SOFTLENS_BLOCK_SIZE 128
#define DERIVATIVES_BLOCK_SIZE 64

#include <cosmictiger/all_tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/cuda_reduce.hpp>

struct softlens_record {
	fixed32 x;
	fixed32 y;
	fixed32 z;
};

struct softlens_workspace {
	device_vector<softlens_record> rec;
};

struct all_tree_reduction {
	int counter;
	int flag;
	float hmin;
	float hmax;
	float flops;
};

__global__ void cuda_softlens(all_tree_data params, all_tree_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ softlens_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) softlens_workspace();
	array<fixed32, NDIM> x;
	float error;
	while (index < params.nselfs) {

		int flops = 0;
		__syncthreads();
		ws.rec.resize(0);
		const tree_node& self = params.trees[params.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const tree_node& other = params.trees[params.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = params.x[pi];
					x[YDIM] = params.y[pi];
					x[ZDIM] = params.z[pi];
					contains = (self.obox.contains(x));
				}
				j = contains;
				compute_indices < SOFTLENS_BLOCK_SIZE > (j, total);
				const int offset = ws.rec.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec[k].x = x[XDIM];
					ws.rec[k].y = x[YDIM];
					ws.rec[k].z = x[ZDIM];
				}
			}
		}
		float hmin = 1e+20f;
		float hmax = 0.0;
		const float w0 = kernelW(0.f);
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const auto rung = params.rung_snk[snki];
			const bool active = rung >= params.minrung;
			const auto& converged = params.converged_snk[snki];
			const bool use = active && !converged;
			if (use) {
				x[XDIM] = params.x[i];
				x[YDIM] = params.y[i];
				x[ZDIM] = params.z[i];
				int box_xceeded = false;
				int iter = 0;
				float& h = params.softlen_snk[snki];
				float drho_dh;
				float rhoh3;
				float last_dlogh = 0.0f;
				float w1 = 1.f;
				do {
					const float hinv = 1.f / h; 										// 4
					const float h2 = sqr(h);    										// 1
					drho_dh = 0.f;
					rhoh3 = 0.f;
					float rhoh30 = (3.0f * params.N) / (4.0f * float(M_PI));   // 4
					flops += 9;
					for (int j = tid; j < ws.rec.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.rec[j].x);        // 1
						const float dy = distance(x[YDIM], ws.rec[j].y);        // 1
						const float dz = distance(x[ZDIM], ws.rec[j].z);        // 1
						const float r2 = sqr(dx, dy, dz);                     // 5
						const float r = sqrtf(r2);                            // 4
						const float q = r * hinv;                             // 1
						flops += 13;
						if (q < 1.f) {                                        // 1
							const float w = kernelW(q);
							const float dwdq = dkernelW_dq(q);
							const float dwdh = -q * dwdq * hinv; 					// 3
							drho_dh -= dwdq;												// 1
							rhoh3 += w;														// 1
							flops += 5;
						}

					}
					shared_reduce_add<float, SOFTLENS_BLOCK_SIZE>(drho_dh);	 // 127
					shared_reduce_add<float, SOFTLENS_BLOCK_SIZE>(rhoh3);	 // 127
					if (tid) {
						flops += 2 * (SOFTLENS_BLOCK_SIZE - 1);
					}
					float dlogh;
					__syncthreads();

					if (rhoh3 <= 1.01f * w0) {											// 2
						if (tid == 0) {
							h *= 1.1f;														// 1
							flops++;
						}
						flops++;
						iter--;
						error = 1.0;
					} else {
						drho_dh *= 0.33333333333f / rhoh30;							// 5
						const float fpre = fminf(fmaxf(1.0f / (drho_dh), 0.5f), 2.0f);							// 6
						dlogh = fminf(fmaxf(powf(rhoh30 / rhoh3, fpre * 0.3333333333333333f) - 1.f, -.1f), .1f); // 9
						error = fabs(1.0f - rhoh3 / rhoh30); // 3
						if (last_dlogh * dlogh < 0.f) { // 2
							w1 *= 0.9f;                       // 1
							flops++;
						} else {
							w1 = fminf(1.f, w1 / 0.9f); // 5
							flops += 5;
						}
						if (tid == 0) {
							h *= (1.f + w1 * dlogh);    // 3
							flops += 3;
						}
						flops += 23;
						last_dlogh = dlogh;
					}
					flops += 2;

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
						if (self.obox.end[dim] < range_fixed(x[dim] + fixed32(h)) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
						if (range_fixed(x[dim]) < self.obox.begin[dim] + range_fixed(h) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
					}
					__syncthreads();
					iter++;
					shared_reduce_add<int, SOFTLENS_BLOCK_SIZE>(box_xceeded);
				} while (error > 5e-5f && !box_xceeded);
				if (box_xceeded) {
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
					}
				}
				hmin = fminf(hmin, h);
				hmax = fmaxf(hmax, h);
			}
		}
		shared_reduce_add<int, SOFTLENS_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicMax(&reduce->hmax, hmax);
			atomicMin(&reduce->hmin, hmin);
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~softlens_workspace();
}


struct derivatives_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};

struct derivatives_workspace {
	device_vector<derivatives_record1> rec1;
};

__global__ void cuda_derivatives(all_tree_data params, all_tree_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ derivatives_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) derivatives_workspace();
	array<fixed32, NDIM> x;
	while (index < params.nselfs) {
		int flops = 0;
		__syncthreads();
		ws.rec1.resize(0);
		const tree_node& self = params.trees[params.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const tree_node& other = params.trees[params.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = params.x[pi];
					x[YDIM] = params.y[pi];
					x[ZDIM] = params.z[pi];
					contains = (self.obox.contains(x));
					if (!contains) {
						contains = true;
						const float& h = params.h[pi];
						for (int dim = 0; dim < NDIM; dim++) {
							if (distance(x[dim], self.ibox.begin[dim]) + h < 0.f) {
								flops += 3;
								contains = false;
								break;
							}
							if (distance(self.ibox.end[dim], x[dim]) + h < 0.f) {
								flops += 3;
								contains = false;
								break;
							}
						}
					}
				}
				j = contains;
				compute_indices < DERIVATIVES_BLOCK_SIZE > (j, total);
				const int offset = ws.rec1.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec1.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1[k].x = x[XDIM];
					ws.rec1[k].y = x[YDIM];
					ws.rec1[k].z = x[ZDIM];
					ws.rec1[k].h = params.h[pi];
				}
			}
		}
		__syncthreads();
		float error;
		float hmin = 1e+20;
		float hmax = 0.0;
		const float w0 = kernelW(0.f);
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const auto rung = params.rung_snk[snki];
			const bool active = rung >= params.minrung;
			int semiactive = 0;
			x[XDIM] = params.x[i];
			x[YDIM] = params.y[i];
			x[ZDIM] = params.z[i];
			const fixed32& x_i = x[XDIM];
			const fixed32& y_i = x[YDIM];
			const fixed32& z_i = x[ZDIM];
			const float& h_i = params.softlen_snk[snki];
			const float hinv_i = 1.f / h_i; 										// 4
			const float h2_i = sqr(h_i);    										// 1
			if (!active) {
				const int jmax = round_up(ws.rec1.size(), block_size);
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.rec1.size()) {
						const auto& rec1 = ws.rec1[j];
						const auto& x_j = rec1.x;
						const auto& y_j = rec1.y;
						const auto& z_j = rec1.z;
						const auto& h_j = rec1.h;
						const auto h2_j = sqr(h_j);									// 1
						const float x_ij = distance(x_i, x_j);						// 1
						const float y_ij = distance(y_i, y_j);						// 1
						const float z_ij = distance(z_i, z_j);						// 1
						const float r2 = sqr(x_ij, y_ij, z_ij);					// 5
						if (r2 < fmaxf(h2_i, h2_j)) {									// 2
							semiactive++;
						}
						flops += 11;
					}
					shared_reduce_add<int, DERIVATIVES_BLOCK_SIZE>(semiactive);
					if (semiactive) {
						break;
					}
				}
			}
			int box_xceeded = false;
			if (semiactive) {
				int iter = 0;
				float& h = params.softlen_snk[snki];
				float drho_dh;
				float rhoh3;
				float last_dlogh = 0.0f;
				float w1 = 1.f;
				do {
					const float hinv = 1.f / h; 										// 4
					const float h2 = sqr(h);    										// 1
					drho_dh = 0.f;
					rhoh3 = 0.f;
					float rhoh30 = (3.0f * params.N) / (4.0f * float(M_PI));   // 4
					flops += 9;
					for (int j = tid; j < ws.rec1.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.rec1[j].x);        // 1
						const float dy = distance(x[YDIM], ws.rec1[j].y);        // 1
						const float dz = distance(x[ZDIM], ws.rec1[j].z);        // 1
						const float r2 = sqr(dx, dy, dz);                     // 5
						const float r = sqrtf(r2);                            // 4
						const float q = r * hinv;                             // 1
						flops += 13;
						if (q < 1.f) {                                        // 1
							const float w = kernelW(q);
							const float dwdq = dkernelW_dq(q);
							const float dwdh = -q * dwdq * hinv; 					// 3
							drho_dh -= dwdq;												// 1
							rhoh3 += w;														// 1
							flops += 5;
						}

					}
					shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(drho_dh);	 // 127
					shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(rhoh3);	 // 127
					if (tid) {
						flops += 2 * (DERIVATIVES_BLOCK_SIZE - 1);
					}
					float dlogh;
					__syncthreads();

					if (rhoh3 <= 1.01f * w0) {											// 2
						if (tid == 0) {
							h *= 1.1f;														// 1
							flops++;
						}
						flops++;
						iter--;
						error = 1.0;
					} else {
						drho_dh *= 0.33333333333f / rhoh30;							// 5
						const float fpre = fminf(fmaxf(1.0f / (drho_dh), 0.5f), 2.0f);							// 6
						dlogh = fminf(fmaxf(powf(rhoh30 / rhoh3, fpre * 0.3333333333333333f) - 1.f, -.1f), .1f); // 9
						error = fabs(1.0f - rhoh3 / rhoh30); // 3
						if (last_dlogh * dlogh < 0.f) { // 2
							w1 *= 0.9f;                       // 1
							flops++;
						} else {
							w1 = fminf(1.f, w1 / 0.9f); // 5
							flops += 5;
						}
						if (tid == 0) {
							h *= (1.f + w1 * dlogh);    // 3
							flops += 3;
						}
						flops += 23;
						last_dlogh = dlogh;
					}
					flops += 2;

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
						if (self.obox.end[dim] < range_fixed(x[dim] + fixed32(h)) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
						if (range_fixed(x[dim]) < self.obox.begin[dim] + range_fixed(h) + range_fixed::min()) {
							box_xceeded = true;
							break;
						}
					}
					__syncthreads();
					iter++;
					shared_reduce_add<int, DERIVATIVES_BLOCK_SIZE>(box_xceeded);
				} while (error > 1e-5f && !box_xceeded);
				if (box_xceeded) {
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
					}
				}
				hmin = fminf(hmin, h);
				hmax = fmaxf(hmax, h);
			}
			if (active || (semiactive && !box_xceeded)) {
				__syncthreads();
				const int snki = self.sink_part_range.first - self.part_range.first + i;
				const float& h_i = params.softlen_snk[snki];
				const fixed32& x_i = x[XDIM];
				const fixed32& y_i = x[YDIM];
				const fixed32& z_i = x[ZDIM];
				const float hinv_i = 1.f / h_i; 										// 4
				const float h3inv_i = sqr(hinv_i) * hinv_i;
				const float h2_i = sqr(h_i);    										// 1
				float drho_dh;
				const float c0 = float(3.0f / 4.0f / M_PI * params.N);     // 1
				drho_dh = 0.f;
				float rhoh30 = (3.0f * params.N) / (4.0f * float(M_PI));   // 5
				__syncthreads();
				flops += 10;
				const int jmax = round_up(ws.rec1.size(), DERIVATIVES_BLOCK_SIZE);
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.rec1.size()) {
						const auto& rec1 = ws.rec1[j];
						const fixed32& x_j = rec1.x;
						const fixed32& y_j = rec1.y;
						const fixed32& z_j = rec1.z;
						const float x_ij = distance(x_i, x_j); // 1
						const float y_ij = distance(y_i, y_j); // 1
						const float z_ij = distance(z_i, z_j); // 1
						const float r2 = sqr(x_ij, y_ij, z_ij);
						const float r = sqrtf(r2);                    // 4
						const float q = r * hinv_i;                    // 1
						if (q < 1.f) {                               // 1
							const float w = kernelW(q);
							const float dwdq = dkernelW_dq(q);
							drho_dh -= q * dwdq;                      // 2
							flops += 2;
						}
						flops += 9;
					}
				}
				shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(drho_dh);
				flops += (DERIVATIVES_BLOCK_SIZE - 1);
				const float omega_i = 0.33333333333f * drho_dh / rhoh30;								// 4
				__syncthreads();
				if (tid == 0) {
					if(params.type_snk[snki] == SPH_TYPE) {
						params.sph_omega_snk[params.cat_snk[snki]] = omega_i;
					}
				}
			}
		}
		shared_reduce_add<int, DERIVATIVES_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicMax(&reduce->hmax, hmax);
			atomicMin(&reduce->hmin, hmin);
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~derivatives_workspace();
}




softlens_return all_tree_softlens_cuda(all_tree_data params, cudaStream_t stream) {
	softlens_return rc;
	all_tree_reduction* reduce;
	CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(all_tree_reduction)));
	reduce->counter = reduce->flag = 0;
	reduce->hmin = std::numeric_limits<float>::max();
	reduce->hmax = 0.0f;
	reduce->flops = 0.0;
	static int softlens_nblocks;
	static bool first = true;
	timer tm;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&softlens_nblocks, (const void*) cuda_softlens, SOFTLENS_BLOCK_SIZE, 0));
		softlens_nblocks *= cuda_smp_count();
	}
	tm.start();
	cuda_softlens<<<softlens_nblocks, SOFTLENS_BLOCK_SIZE,0,stream>>>(params,reduce);
	cuda_stream_synchronize(stream);
	rc.fail = reduce->flag;
	rc.hmin = reduce->hmin;
	rc.hmax = reduce->hmax;
	tm.stop();
	rc.flops = reduce->flops;
	CUDA_CHECK(cudaFree(reduce));
	return rc;

}


softlens_return all_tree_derivatives_cuda(all_tree_data params, cudaStream_t stream) {
	softlens_return rc;
	all_tree_reduction* reduce;
	CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(all_tree_reduction)));
	reduce->counter = reduce->flag = 0;
	reduce->hmin = std::numeric_limits<float>::max();
	reduce->hmax = 0.0f;
	reduce->flops = 0.0;
	static int derivatives_nblocks;
	static bool first = true;
	timer tm;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&derivatives_nblocks, (const void*) cuda_derivatives, SOFTLENS_BLOCK_SIZE, 0));
		derivatives_nblocks *= cuda_smp_count();
	}
	tm.start();
	cuda_derivatives<<<derivatives_nblocks, SOFTLENS_BLOCK_SIZE,0,stream>>>(params,reduce);
	cuda_stream_synchronize(stream);
	rc.fail = reduce->flag;
	rc.hmin = reduce->hmin;
	rc.hmax = reduce->hmax;
	tm.stop();
	rc.flops = reduce->flops;
	CUDA_CHECK(cudaFree(reduce));
	return rc;

}
