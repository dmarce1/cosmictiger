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

#include <cosmictiger/sph_cuda.hpp>

struct smoothlen_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
};
struct smoothlen_workspace {
	device_vector<smoothlen_record1> x;
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
		__syncthreads();
		ws.x.resize(0);
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
				compute_indices < SMOOTHLEN_BLOCK_SIZE > (j, total);
				const int offset = ws.x.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x[k].x = x[XDIM];
					ws.x[k].y = x[YDIM];
					ws.x[k].z = x[ZDIM];
				}
			}
		}
		ALWAYS_ASSERT(found_self);
		ALWAYS_ASSERT(ws.x.size());
		float hmin = 1e+20;
		float hmax = 0.0;
		const float w0 = kernelW(0.f);
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const bool active = data.rungs_snk[data.dm_index_snk[snki]] >= params.min_rung;
			const bool converged = data.converged_snk[snki];
			const bool use = active && !converged;
			if (use) {
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				int box_xceeded = false;
				int iter = 0;
				float& h = data.rec2_snk[snki].h;
				float drho_dh;
				float rhoh3;
				float last_dlogh = 0.0f;
				float w1 = 1.f;
				do {
					const float hinv = 1.f / h; 										// 4
					const float h2 = sqr(h);    										// 1
					drho_dh = 0.f;
					rhoh3 = 0.f;
					float rhoh30 = (3.0f * data.N) / (4.0f * float(M_PI));   // 4
					flops += 9;
					for (int j = tid; j < ws.x.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.x[j].x);        // 1
						const float dy = distance(x[YDIM], ws.x[j].y);        // 1
						const float dz = distance(x[ZDIM], ws.x[j].z);        // 1
						const float r2 = sqr(dx, dy, dz);                     // 5
						const float r = sqrtf(r2);                            // 4
						const float q = r * hinv;                             // 1
						flops += 13;
						if (q < 1.f) {                                        // 1
							float w;
							const float dwdq = dkernelW_dq(q, &w, &flops);
							const float dwdh = -q * dwdq * hinv; 					// 3
							drho_dh -= dwdq;												// 1
							rhoh3 += w;														// 1
							flops += 5;
						}

					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(drho_dh);	 // 127
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(rhoh3);	 // 127
					if (tid) {
						flops += 2 * (SMOOTHLEN_BLOCK_SIZE - 1);
					}
					float dlogh;
					__syncthreads();

					if (rhoh3 <= 1.01f * w0) {											// 2
						PRINT("ZERO neighbors %i\n", ws.x.size());
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
				if (box_xceeded) {
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
		flops = 0;
		__syncthreads();
	}
	(&ws)->~smoothlen_workspace();
}
