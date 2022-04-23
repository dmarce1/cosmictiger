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

struct conduction_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
	char rung;
};
struct conduction_record2 {
	float entr;
	float kappa;
	float omega;
};

struct conduction_workspace {
	device_vector<int> neighbors;
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
	const float cons = -2.f * sqr(params.a) * m;
	int flops = 4;
	while (index < data.nselfs) {
		__syncthreads();
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
				compute_indices < CONDUCTION_BLOCK_SIZE > (j, total);
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
					ws.rec2[k].omega = data.omega[pi];
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

				ws.neighbors.resize(0);
				const float jmax = round_up(ws.rec1.size(), block_size);
				const auto x_i = data.x[i];
				const auto y_i = data.y[i];
				const auto z_i = data.z[i];
				const float h_i = data.h[i];
				const float A_i = data.entr[i];
				for (int j = tid; j < jmax; j += block_size) {
					int k;
					int total;
					bool use = false;
					if (j < ws.rec1.size()) {
						const auto rec1 = ws.rec1[j];
						const fixed32 x_j = rec1.x;
						const fixed32 y_j = rec1.y;
						const fixed32 z_j = rec1.z;
						const auto rung_j = rec1.rung;
						const float h_j = rec1.h;
						const float x_ij = distance(x_i, x_j);				// 1
						const float y_ij = distance(y_i, y_j);				// 1
						const float z_ij = distance(z_i, z_j);				// 1
						const float r2 = sqr(x_ij, y_ij, z_ij);			// 5
						if (r2 < fmaxf(sqr(h_i), sqr(h_j)) && r2 > 0.f) {				// 4
							use = active || rung_j >= params.min_rung;
						}
						flops += 12;
					}
					k = use;
					compute_indices < CONDUCTION_BLOCK_SIZE > (k, total);
					const int offset = ws.neighbors.size();
					__syncthreads();
					const int next_size = offset + total;
					ws.neighbors.resize(next_size);
					if (use) {
						const int l = offset + k;
						ws.neighbors[l] = j;
					}
				}
				__syncthreads();

				const float omega_i = data.omega[i];
				const auto rung_i = data.rungs[i];
				const float kappa_i = data.kappa[i];
				const float h2_i = sqr(h_i);													// 1
				const float c0 = float(3.0f / (4.0f * M_PI)) * data.N;				// 1
				const float hinv_i = 1.f / h_i;												// 4
				const float h3inv_i = (sqr(hinv_i) * hinv_i);							// 2
				const float rho_i = m * c0 * h3inv_i;										// 2
				const float ene_i = A_i * powf(rho_i, gamma0 - 1.0f);	// 11
				const float dt_i = rung_dt[rung_i] * params.t0;							// 1
				const float rhoinv_i = 1.f / (rho_i);							// 5
				float den = 0.f;
				float num = 0.f;
				flops += 28;
				for (int j = tid; j < ws.neighbors.size(); j += block_size) {
					const int kk = ws.neighbors[j];
					const auto& rec1 = ws.rec1[kk];
					const auto& rec2 = ws.rec2[kk];
					const fixed32 x_j = rec1.x;
					const fixed32 y_j = rec1.y;
					const fixed32 z_j = rec1.z;
					const float h_j = rec1.h;
					const auto rung_j = rec1.rung;
					const float x_ij = distance(x_i, x_j);				// 1
					const float y_ij = distance(y_i, y_j);				// 1
					const float z_ij = distance(z_i, z_j);				// 1
					const float r2 = sqr(x_ij, y_ij, z_ij);			// 5
					const float h2_j = sqr(h_j);							// 1
					flops += 11;
					if (r2 < fmaxf(h2_j, h2_i)) {							// 2
						const float A_j = rec2.entr;
						const float r = sqrtf(r2);							// 4
						const float rinv = 1.f / r;						// 4
						const float q_i = r * hinv_i;						// 1
						flops += 10;
						const float omega_j = rec2.omega;
						const float kappa_j = rec2.kappa;
						const float hinv_j = 1.f / h_j;															// 4
						const float h3inv_j = sqr(hinv_j) * hinv_j;											// 2
						const float rho_j = m * c0 * h3inv_j;													// 2
						const float q_j = r * hinv_j;																// 1
						float w;
						const float dWdr_i = dkernelW_dq(q_i, &w, &flops) * h3inv_i * hinv_i / omega_i;																// 3
						const float dWdr_j = dkernelW_dq(q_j, &w, &flops) * h3inv_j * hinv_j / omega_j;																// 3
						const float dWdr_ij = 0.5f * (dWdr_i + dWdr_j);										// 22
						const float kappa_ij = 2.f * kappa_i * kappa_j / (kappa_i + kappa_j + 1.0e-35f); // 8
						const float dt_j = rung_dt[rung_j] * params.t0;										// 2
						const float dt_ij = fminf(dt_i, dt_j);													// 2
						const float D_ij = cons * kappa_ij * dWdr_ij / (rho_i * rho_j) * dt_ij * rinv; // 10
						num = fmaf(D_ij, A_j * powf(rho_j * rhoinv_i, gamma0 - 1.f), num); // 14
						den += D_ij;																					// 1
						flops += 69;
					}
				}
				shared_reduce_add<float, CONDUCTION_BLOCK_SIZE>(num); //31
				shared_reduce_add<float, CONDUCTION_BLOCK_SIZE>(den); //31
				if (tid == 0) {
					flops += (CONDUCTION_BLOCK_SIZE - 1) * 2;
					flops += 7;
					const float A0 = data.entr0_snk[snki];
					const float A1 = (A0 + num) / (1.f + den);
					const float dA = A1 - A_i;
					data.dentr1_snk[snki] = dA;
//					ALWAYS_ASSERT(A1-A0==0.0);
					ALWAYS_ASSERT(isfinite(dA));
				}
			}
		}

		shared_reduce_add<int, CONDUCTION_BLOCK_SIZE>(flops);
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
			atomicAdd(&reduce->flops, (float) flops);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~conduction_workspace();

}

