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

struct cond_init_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
};
struct cond_init_record2 {
#ifdef ENTROPY
	float entr;
#else
	float eint;
#endif
	float cfrac;
	float rho;
	char star;
};

struct cond_init_workspace {
	device_vector<cond_init_record1> rec1;
	device_vector<cond_init_record2> rec2;
	device_vector<int> neighbors;
};

__global__ void sph_cuda_cond_init(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	int const tid = threadIdx.x;
	int const block_size = blockDim.x;
	__shared__ int index;
	__shared__ cond_init_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) cond_init_workspace();
	int flops = 0;
	array<fixed32, NDIM> x;
	float const gamma0 = data.def_gamma;
	float const code_to_energy = sqr(params.code_to_cm) / sqr(params.code_to_s);					   // 5
	float const code_to_density = params.code_to_g / pow(params.code_to_cm, 3.);					   // 12
	float const colog0 = log(1.5 * pow(constants::kb, 1.5) * pow(constants::e, -3) * pow(M_PI, -0.5)); // 35
	float const kappa0 = 20.0 * pow(2.0 / M_PI, 1.5) * pow(constants::kb, 2.5) * pow(constants::me, -0.5) * pow(constants::e, -4.0) * params.code_to_s *
						 params.code_to_cm / (params.code_to_g * constants::avo);									// 49
	float const propc0 = 0.4 * (gamma0 - 1.0) * sqrtf(2.0 * constants::kb / (M_PI * constants::me)) / constants::c; // 17
	float const cv0 = constants::kb / (gamma0 - 1.0f);																// 5
	float const invgm1 = 1.f / (gamma0 - 1.0);
	flops += 128;
	while (index < data.nselfs) {
		__syncthreads();
		int flops = 0;
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
					rung = data.rungs[pi];
					if (rung >= params.min_rung) {
						x[XDIM] = data.x[pi];
						x[YDIM] = data.y[pi];
						x[ZDIM] = data.z[pi];
						contains = (self.outer_box.contains(x));
						if (!contains) {
							contains = true;
							float const& h = data.h[pi];
							for (int dim = 0; dim < NDIM; dim++) {
								if (distance(x[dim], self.inner_box.begin[dim]) + h < 0.f) {
									flops += 3;
									contains = false;
									break;
								}
								if (distance(self.inner_box.end[dim], x[dim]) + h < 0.f) {
									flops += 3;
									contains = false;
									break;
								}
							}
						}
					}
				}
				j = contains;
				compute_indices<COND_INIT_BLOCK_SIZE>(j, total);
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
					ws.rec1[k].h = h;
#ifdef ENTROPY
					ws.rec2[k].entr = data.entr[pi];
#else
					ws.rec2[k].eint = data.eint[pi];
#endif
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
		float const& m = data.m;
		flops++;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			int const snki = self.sink_part_range.first - self.part_range.first + i;
			bool const active = data.rungs[i] >= params.min_rung;
			bool const semiactive = !active && data.sa_snk[snki];
			flops += 2;
			if (semiactive || active) {
				float cfrac_i = 0.f;
				bool star_i = false;
				if (params.stars) {
					cfrac_i = data.cold_frac[i];
					star_i = data.stars[i];
				}
				float const& rho_i = data.rho[i]; // 2
				float const& h_i = data.h[i];
#ifdef ENTROPY
				float const& A_i = data.entr[i];
#else
				float const& eint_i = data.eint[i];
#endif
				auto const& x_i = data.x[i];
				auto const& y_i = data.y[i];
				auto const& z_i = data.z[i];
				float const hfrac_i = 1.f - cfrac_i;
				float const hinv_i = 1.f / h_i;
				float const h2_i = sqr(h_i);
				float const h3inv_i = (sqr(hinv_i) * hinv_i); // 2
#ifdef ENTROPY
				float const eint_i = A_i * powf(rho_i * hfrac_i, gamma0 - 1.0f); // 11
#endif
				float gradx = 0.0f;
				float grady = 0.0f;
				float gradz = 0.0f;
				float const omega_i = data.omega_snk[snki];
				flops += 11;
				ws.neighbors.resize(0);
				float const jmax = round_up(ws.rec1.size(), block_size);
				for (int j = tid; j < jmax; j += block_size) {
					int k;
					int total;
					bool use = false;
					if (j < ws.rec1.size()) {
						auto const rec1 = ws.rec1[j];
						fixed32 const x_j = rec1.x;
						fixed32 const y_j = rec1.y;
						fixed32 const z_j = rec1.z;
						float const h_j = rec1.h;
						float const x_ij = distance(x_i, x_j);	// 1
						float const y_ij = distance(y_i, y_j);	// 1
						float const z_ij = distance(z_i, z_j);	// 1
						float const r2 = sqr(x_ij, y_ij, z_ij); // 5
						if (r2 < fmaxf(sqr(h_i), sqr(h_j))) {	// 4
							use = true;
						}
						flops += 12;
					}
					k = use;
					compute_indices<COND_INIT_BLOCK_SIZE>(k, total);
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
				if (!star_i) {
					for (int j = tid; j < ws.neighbors.size(); j += COND_INIT_BLOCK_SIZE) {
						int const kk = ws.neighbors[j];
						auto const& rec1 = ws.rec1[kk];
						auto const& rec2 = ws.rec2[kk];
						auto const& star_j = rec2.star;
						if (!star_j) {
#ifdef ENTROPY
							float const& rho_j = rec2.rho; // 2
							float const& A_j = rec2.entr;
#else
							float const& eint_j = rec2.eint;
#endif
							float const& cfrac_j = rec2.cfrac;
							float const hfrac_j = 1.f - cfrac_j;
							fixed32 const x_j = rec1.x;
							fixed32 const y_j = rec1.y;
							fixed32 const z_j = rec1.z;
							float const x_ij = distance(x_i, x_j);	// 1
							float const y_ij = distance(y_i, y_j);	// 1
							float const z_ij = distance(z_i, z_j);	// 1
							float const r2 = sqr(x_ij, y_ij, z_ij); // 5
							flops += 10;
							if (r2 < h2_i && r2 != 0.0f) { // 2
								float const h_j = rec1.h;
								float const r = sqrtf(r2);					// 4
								float const rinv = 1.f / r;					// 4
								float const q = r * hinv_i;					// 1
								float const hinv_j = 1.f / h_j;				// 4
								float const h3inv_j = sqr(hinv_j) * hinv_j; // 2
#ifdef ENTROPY
								float const eint_j = A_j * powf(rho_j * hfrac_j, gamma0 - 1.0f); // 11
#endif
								float const dwdq = dkernelW_dq(q);
								float const dWdr_i = dwdq * h3inv_i * hinv_i / omega_i; // 3
								float const tmp = dWdr_i * rinv * (eint_j - eint_i);	// 14
								gradx = fmaf(tmp, x_ij, gradx);							// 2
								grady = fmaf(tmp, y_ij, grady);							// 2
								gradz = fmaf(tmp, z_ij, gradz);							// 2
								flops += 49;
							}
						}
					}
					float const tmp = m / (params.a * rho_i * eint_i); // 5
					gradx *= tmp;									   // 1
					grady *= tmp;									   // 1
					gradz *= tmp;									   // 1
					flops += 8;
					shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(gradx); // 31
					shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(grady); // 31
					shared_reduce_add<float, COND_INIT_BLOCK_SIZE>(gradz); // 31
					if (tid == 0) {
						flops += 93;
						auto const& frac_i = data.rec1_snk[snki].frac;
#ifdef ENTROPY
						float const& A_i = data.rec2_snk[snki].A;
#else
						float const& eint_i = data.rec2_snk[snki].eint;
#endif
						float const& H = frac_i[CHEM_H];
						float const& Hp = frac_i[CHEM_HP];
						float const& Hn = frac_i[CHEM_HN];
						float const& H2 = frac_i[CHEM_H2];
						float const& He = frac_i[CHEM_HE];
						float const& Hep = frac_i[CHEM_HEP];
						float const& Hepp = frac_i[CHEM_HEPP];
						float const grad2 = sqr(gradx, grady, gradz);																			 // 5
						float const gradToT = sqrtf(grad2);																						 // 4
						float const rho0 = rho_i * hfrac_i / (sqr(params.a) * params.a);														 // 7
						float n0 = (H + fmaf(2.f, Hp, fmaf(.5f, H2, fmaf(.25f, He, fmaf(.5f, Hep, .75f * Hepp)))));								 // 10
						float const mmw_i = 1.0f / n0;																							 // 4
						float const ne_i = fmaxf((Hp - Hn + fmaf(0.25f, Hep, 0.5f * Hepp)) * rho0 * (constants::avo * code_to_density), 1e-30f); // 8
#ifdef ENTROPY
						float const eint = code_to_energy * A_i * powf(rho0 * hfrac_i, gamma0 - 1.0) * invgm1; // 13
#else
						float const eint = eint_i;
#endif
						float const T_i = mmw_i * eint / (cv0 * constants::avo);					 // 6
						float const colog_i = colog0 + 1.5f * logf(T_i) - 0.5f * logf(ne_i);		 // 20
						float kappa_i = mmw_i * (gamma0 - 1.f) * kappa0 * powf(T_i, 2.5f) / colog_i; // 15
						float const sigmax_i = propc0 * sqrtf(T_i);									 // 5
						float const R = 2.f * mmw_i * kappa_i * gradToT / (rho_i * sigmax_i);		 // 7
						float const phi = (2.f + 3.f * R) / (2.f + 3.f * R + 3.f * sqr(R));			 // 11
						kappa_i *= phi;																 // 1
						ALWAYS_ASSERT(kappa_i >= 0.0f);
						data.kap_snk[snki] = kappa_i;
#ifdef ENTROPY
						data.entr0_snk[snki] = A_i;
#else
						data.eint0_snk[snki] = eint_i;
#endif
					}
				}
			}
		}
		shared_reduce_add<int, COND_INIT_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (float)flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~cond_init_workspace();
}
