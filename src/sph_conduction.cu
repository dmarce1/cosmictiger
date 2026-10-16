/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <cosmictiger/sph_cuda.hpp>

struct conduction_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
	char rung;
};
struct conduction_record2 {
#ifdef ENTROPY
	float entr;
#else
	float eint;
#endif
	float kappa;
	float omega;
	float cfrac;
	float rho;
	char star;
};

struct conduction_workspace {
	device_vector<int> neighbors;
	device_vector<conduction_record1> rec1;
	device_vector<conduction_record2> rec2;
};

__global__ void sph_cuda_conduction(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	int const tid = threadIdx.x;
	int const block_size = blockDim.x;
	__shared__ int index;
	__shared__ conduction_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) conduction_workspace();
	float const gamma0 = data.def_gamma;
	float const m = data.m;
	array<fixed32, NDIM> x;
	float const cons = -2.f * sqr(params.a) * m;
	int flops = 4;
	while (index < data.nselfs) {
		__syncthreads();
		ws.rec1.resize(0);
		ws.rec2.resize(0);
		sph_tree_node const& self = data.trees[data.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			sph_tree_node const& other = data.trees[data.neighbors[ni]];
			int const maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
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
				int const offset = ws.rec1.size();
				__syncthreads();
				int const next_size = offset + total;
				ws.rec1.resize(next_size);
				ws.rec2.resize(next_size);
				if (contains) {
					int const k = offset + j;
					ws.rec1[k].x = x[XDIM];
					ws.rec1[k].y = x[YDIM];
					ws.rec1[k].z = x[ZDIM];
					ALWAYS_ASSERT(h > 0.f);
					ws.rec1[k].h = h;
					ws.rec1[k].rung = data.rungs[pi];
#ifdef ENTROPY
					ws.rec2[k].entr = data.entr[pi];
#else
					ws.rec2[k].eint = data.eint[pi];
#endif
					ws.rec2[k].kappa = data.kappa[pi];
					ws.rec2[k].omega = data.omega[pi];
					ws.rec2[k].rho = data.rho[pi];
					if (params.stars) {
						ws.rec2[k].cfrac = data.cold_frac[pi];
						ws.rec2[k].star = data.stars[pi];
					} else {
						ws.rec2[k].cfrac = 0.f;
						ws.rec2[k].star = false;
					}
				}
			}
		}
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			int const snki = self.sink_part_range.first - self.part_range.first + i;
			bool const active = data.rungs[i] >= params.min_rung;
			bool const semiactive = !active && data.sa_snk[snki];
			bool const converged = data.converged_snk[snki];
			if ((active || semiactive) && !converged) {

				ws.neighbors.resize(0);
				float const jmax = round_up(ws.rec1.size(), block_size);
				bool star_i = false;
				if (params.stars) {
					star_i = data.stars[i];
				}
				auto const& x_i = data.x[i];
				auto const& y_i = data.y[i];
				auto const& z_i = data.z[i];
				float const& h_i = data.h[i];
#ifdef ENTROPY
				float const& A_i = data.entr[i];
#else
				float const& eint_i = data.eint[i];
#endif
				float cfrac_i = 0.f;
				if (params.stars) {
					cfrac_i = data.cold_frac[i];
				}
				float const hfrac_i = 1.f - cfrac_i;
				for (int j = tid; j < jmax; j += block_size) {
					int k;
					int total;
					bool use = false;
					if (j < ws.rec1.size()) {
						auto const rec1 = ws.rec1[j];
						fixed32 const x_j = rec1.x;
						fixed32 const y_j = rec1.y;
						fixed32 const z_j = rec1.z;
						auto const rung_j = rec1.rung;
						float const h_j = rec1.h;
						float const x_ij = distance(x_i, x_j);			  // 1
						float const y_ij = distance(y_i, y_j);			  // 1
						float const z_ij = distance(z_i, z_j);			  // 1
						float const r2 = sqr(x_ij, y_ij, z_ij);			  // 5
						if (r2 < fmaxf(sqr(h_i), sqr(h_j)) && r2 > 0.f) { // 4
							use = active || rung_j >= params.min_rung;
						}
						flops += 12;
					}
					k = use;
					compute_indices<CONDUCTION_BLOCK_SIZE>(k, total);
					int const offset = ws.neighbors.size();
					__syncthreads();
					int const next_size = offset + total;
					ws.neighbors.resize(next_size);
					if (use) {
						int const l = offset + k;
						ws.neighbors[l] = j;
					}
				}
				__syncthreads();

				float const& omega_i = data.omega[i];
				auto const& rung_i = data.rungs[i];
				float const& kappa_i = data.kappa[i];
				float const& rho_i = data.rho[i];
				float const h2_i = sqr(h_i);													  // 1
				float const hinv_i = 1.f / h_i;													  // 4
				float const h3inv_i = (sqr(hinv_i) * hinv_i);									  // 2
				float const dt_i = rung_i >= params.min_rung ? rung_dt[rung_i] * params.t0 : 0.f; // 1
				float const rhoinv_i = 1.f / (rho_i);											  // 5
				float den = 0.f;
				float num = 0.f;
				flops += 28;
				if (!star_i) {

					for (int j = tid; j < ws.neighbors.size(); j += block_size) {
						int const kk = ws.neighbors[j];
						auto const& rec1 = ws.rec1[kk];
						auto const& rec2 = ws.rec2[kk];
						auto const& star_j = rec2.star;
						if (!star_j) {
							fixed32 const x_j = rec1.x;
							fixed32 const y_j = rec1.y;
							fixed32 const z_j = rec1.z;
							float const h_j = rec1.h;
							auto const rung_j = rec1.rung;
							float const x_ij = distance(x_i, x_j);	// 1
							float const y_ij = distance(y_i, y_j);	// 1
							float const z_ij = distance(z_i, z_j);	// 1
							float const r2 = sqr(x_ij, y_ij, z_ij); // 5
							float const h2_j = sqr(h_j);			// 1
							flops += 11;
							if (r2 < fmaxf(h2_j, h2_i)) { // 2
								float const& cfrac_j = rec2.cfrac;
#ifdef ENTROPY
								float const& A_j = rec2.entr;
#else
								float const& eint_j = rec2.eint;
#endif
								float const& rho_j = rec2.rho; // 2
								float const hfrac_j = 1.f - cfrac_j;
								float const r = sqrtf(r2);	  // 4
								float const rinv = 1.f / r;	  // 4
								float const q_i = r * hinv_i; // 1
								flops += 10;
								float const omega_j = rec2.omega;
								float const kappa_j = rec2.kappa;
								float const hinv_j = 1.f / h_j;													  // 4
								float const h3inv_j = sqr(hinv_j) * hinv_j;										  // 2
								float const q_j = r * hinv_j;													  // 1
								float const dWdr_i = dkernelW_dq(q_i) * h3inv_i * hinv_i / omega_i;				  // 3
								float const dWdr_j = dkernelW_dq(q_j) * h3inv_j * hinv_j / omega_j;				  // 3
								float const dWdr_ij = 0.5f * (dWdr_i + dWdr_j);									  // 22
								float const kappa_ij = 2.f * kappa_i * kappa_j / (kappa_i + kappa_j + 1.0e-35f);  // 8
								float const dt_j = rung_j >= params.min_rung ? rung_dt[rung_j] * params.t0 : 0.f; // 1
								float const dt_ij = 0.5f * (dt_i + dt_j);										  // 2
								float const D_ij = cons * kappa_ij * dWdr_ij / (rho_i * rho_j) * dt_ij * rinv;	  // 10
								ALWAYS_ASSERT(D_ij >= 0.f);
#ifdef ENTROPY
								num = fmaf(D_ij, A_j * powf(rho_j * rhoinv_i * hfrac_j / hfrac_i, gamma0 - 1.f), num); // 14
#else
								num = fmaf(D_ij, eint_j, num); // 14
#endif
								den += D_ij; // 1
								flops += 69;
							}
						}
					}
				}
				shared_reduce_add<float, CONDUCTION_BLOCK_SIZE>(num); // 31
				shared_reduce_add<float, CONDUCTION_BLOCK_SIZE>(den); // 31
				if (tid == 0) {
					flops += (CONDUCTION_BLOCK_SIZE - 1) * 2;
					flops += 7;
#ifdef ENTROPY
					float const A0 = data.entr0_snk[snki];
					float const A1 = (A0 + num) / (1.f + den);
					float const dA = A1 - A_i;
					data.dentr1_snk[snki] = dA;
					ALWAYS_ASSERT(A_i + dA >= 0.0);
#else
					float const eint0 = data.eint0_snk[snki];
					float const eint1 = (eint0 + num) / (1.f + den);
					float const deint = eint1 - eint_i;
					data.deint1_snk[snki] = deint;
					ALWAYS_ASSERT(eint_i + deint >= 0.0);
#endif
				}
			}
		}

		shared_reduce_add<int, CONDUCTION_BLOCK_SIZE>(flops);
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
			atomicAdd(&reduce->flops, (float)flops);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~conduction_workspace();
}
